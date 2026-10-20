import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import (control_replicates, holm, interval, kendall_tau, load_outcomes, matched_replicates,
                                  one_sided, project_builds, significant_flips)
from faithfulcut.control import transfer_widths
from faithfulcut.dante import PROJECTS as JAVA

HISTORY, CONTENT, HYBRID, SPLIT, DANTE = "LatestFail+CC", "TF-IDF+CC", "Hybrid-RRF", "Hybrid", "DANTE"
WORK, ALPHA, PI = 0.2, 0.1, 0.1
TRACKING = ["hadoop", "hive", "jackrabbit-oak", "james", "kafka", "tvm"]
FIRST_FAILURES = ["activemq", "hadoop", "hbase", "hive", "jackrabbit-oak", "james", "kafka", "karaf", "tvm"]
SAVINGS = ["hive", "james", "kafka"]
WORK_MARGIN, MAX_MISSES, MIN_SAVED = 0.02, 3, 0.10
SECONDARY_HISTORY = ["LatestFail", "FailCount+CC", "TF-FailFreq+CC", "DANTE-H"]
SELECTOR_SECONDS = {"TF-IDF": "tfidf_seconds", "TF-IDF+CC": "tfidf_seconds", "Hybrid": "tfidf_seconds",
                    "Hybrid-RRF": "tfidf_seconds", "DANTE": "dante_seconds"}
REGIMES = ("full", "timed", "faithful")


def present(rows: pd.DataFrame, **keys) -> bool:
    return bool(match(rows, **keys).any())


def match(rows: pd.DataFrame, **keys) -> np.ndarray:
    return np.logical_and.reduce([rows[k].to_numpy() == v for k, v in keys.items()])


def fmt(replicates: np.ndarray) -> str:
    i = interval(replicates)
    return f"{i['estimate']:.3f} [{i['low']:.3f}, {i['high']:.3f}]"


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return x


def test(name: str, claim: str, replicates: np.ndarray, p: float, **extra) -> dict:
    return {"hypothesis": name, "claim": claim, **interval(replicates), "p": p, **extra}


def external(name: str, claim: str, path: str | None) -> tuple[dict, list]:
    """A hypothesis analysed by its own script: its result file, or p = 1 if it has not been run."""
    if path is None:
        return {"hypothesis": name, "claim": claim, "estimate": np.nan, "low": np.nan, "high": np.nan, "p": 1.0,
                "note": "not run"}, []
    result = json.loads(Path(path).read_text())
    return {"hypothesis": name, "claim": claim, **{k: result[k] for k in ("estimate", "low", "high", "p")}}, \
        result.get("secondary", [])


def certificates(rows: pd.DataFrame) -> dict:
    """Seeds whose full-stream base escape rate falls outside the adaptive and the normal certificates, and the
    certificates' mean half-widths."""
    rows = rows[rows["operational_faulty"] > 0]
    out = {"seeds": len(rows), "adaptive_misses": 0, "normal_misses": 0, "adaptive": [], "normal": []}
    for r in rows.itertuples():
        caught = r.operational_faulty - r.operational_escape
        w = transfer_widths(caught, r.operational_audited_escape, r.pi, r.builds - caught)
        error = abs(r.operational_escape / r.operational_faulty - w["estimate"])
        out["adaptive_misses"] += int(error > w["adaptive"])
        out["normal_misses"] += int(error > w["normal"])
        out["adaptive"].append(w["adaptive"])
        out["normal"].append(w["normal"])
    return {**out, "adaptive": float(np.mean(out["adaptive"])), "normal": float(np.mean(out["normal"]))}


def main():
    parser = argparse.ArgumentParser(description="The registered tests and descriptive tables for one period: "
                                                 "pull-request bootstrap, Holm within the primary and the "
                                                 "secondary family")
    parser.add_argument("--price", nargs="+", required=True, help="folders with price.csv and price_builds.npz")
    parser.add_argument("--control", required=True, help="folder with control.csv and control_builds.npz")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--selector-seconds", default="data/results/overhead/selector_seconds.csv")
    parser.add_argument("--industrial", help="result file of H7, once run")
    parser.add_argument("--llm", help="result file of H8, once run")
    parser.add_argument("--out", required=True, help="markdown report; the JSON goes beside it")
    args = parser.parse_args()
    evaluate = tuple(args.evaluate)
    later = evaluate[1] > 1
    labels = ["operational"] if later else ["operational", "deconf", "firstfail"]
    price_rows, price_out = load_outcomes(args.price, "price")
    control_rows, control_out = load_outcomes([args.control], "control")
    projects = sorted(set(price_rows["project"]) & set(control_rows["project"]))
    orders = sorted(price_rows["scorer"].unique())
    price, control = {}, {}
    for project in projects:
        builds, weights = project_builds(project, evaluate, args.replicates)
        for order in orders:
            for regime in REGIMES:
                key = dict(project=project, scorer=order, regime=regime, pi=0.0)
                if present(price_rows, **key):
                    price[(order, project, regime)] = matched_replicates(price_rows, price_out, builds, weights,
                                                                         "operational", WORK, **key)
        for order in control_rows["scorer"].unique():
            for kind in ("online", "tuned"):
                key = dict(project=project, scorer=order, controller=kind, alpha=ALPHA, pi=PI)
                if present(control_rows, **key):
                    run = {}
                    for label in labels:
                        run[label], run["work"] = control_replicates(control_rows, control_out, builds, weights,
                                                                     label, **key)
                    control[(order, kind, project)] = run
        print(project, flush=True)

    def covered(order: str) -> list[str]:
        return [p for p in projects if all((order, p, r) in price for r in REGIMES)]

    def macro(order: str, regime: str, among: list[str]) -> np.ndarray:
        return np.nanmean([price[(order, p, regime)] for p in among], axis=0)

    def ratio(order: str, among: list[str] | None = None, top: str = "faithful", bottom: str = "full") -> np.ndarray:
        among = covered(order) if among is None else among
        return macro(order, top, among) / macro(order, bottom, among)

    def difference(order: str, base: str, metric: str, among: list[str]) -> tuple[np.ndarray, list[str]]:
        among = [p for p in among if (order, "online", p) in control and (base, "online", p) in control]
        d = np.nanmean([control[(order, "online", p)][metric] - control[(base, "online", p)][metric]
                        for p in among], axis=0)
        return d, among

    primary, secondary, engineering = [], [], []
    shared = [p for p in covered(HISTORY) if p in covered(CONTENT)]
    history = ratio(HISTORY, shared)
    primary.append(test("H1", f"{HISTORY}: FAITHFUL/FULL operational escape at 20% work > 1 ({len(shared)} "
                              "projects)", history, one_sided(history, above=1.0)))
    immunity = history - ratio(CONTENT, shared)
    primary.append(test("H2", f"FAITHFUL/FULL ratio of {HISTORY} minus that of {CONTENT} > 0", immunity,
                        one_sided(immunity, above=0.0)))
    if not later:
        first, among = difference(HYBRID, HISTORY, "firstfail", FIRST_FAILURES)
        work, _ = difference(HYBRID, HISTORY, "work", FIRST_FAILURES)
        p = max(one_sided(first, below=0.0), one_sided(work, below=WORK_MARGIN))
        primary.append(test("H5", f"{HYBRID} minus {HISTORY}: FirstFail escape < 0 and work < +{WORK_MARGIN} at "
                                  f"equal risk ({len(among)} projects)", first, p, work=fmt(work)))
    dante = ratio(DANTE)
    primary.append(test("H6", f"{DANTE}: FAITHFUL/FULL operational escape at 20% work > 1 ({len(covered(DANTE))} "
                              "projects)", dante, one_sided(dante, above=1.0)))
    industrial_secondary = []
    if not later:
        row, industrial_secondary = external("H7", "LatestFail: FAITHFUL/FULL missed failing verdicts > 1 on the "
                                                   "industrial logs", args.industrial)
        primary.append(row)
        primary.append(external("H8", "Flash-Lite (risk fusion) minus TF-IDF+CC: FirstFail recall at 5% of "
                                      "predicted time > 0", args.llm)[0])

    for order in SECONDARY_HISTORY:
        if covered(order):
            r = ratio(order)
            secondary.append(test(f"H1/{order}", f"{order}: FAITHFUL/FULL > 1 ({len(covered(order))} projects)",
                                  r, one_sided(r, above=1.0)))
    if not later:
        for order, label in ((CONTENT, "firstfail"), (SPLIT, "firstfail"), (HYBRID, "deconf"), (CONTENT, "deconf")):
            d, among = difference(order, HISTORY, label, FIRST_FAILURES)
            work, _ = difference(order, HISTORY, "work", FIRST_FAILURES)
            secondary.append(test(f"H5/{order}/{label}", f"{order} minus {HISTORY}: {label} escape < 0 at equal "
                                                         f"risk ({len(among)} projects)", d,
                                  one_sided(d, below=0.0), work=fmt(work)))
        d, among = difference(DANTE, HISTORY, "firstfail", [p for p in FIRST_FAILURES if p in JAVA])
        secondary.append(test(f"H6/{DANTE}/firstfail", f"{DANTE} minus {HISTORY}: FirstFail escape > 0 at equal "
                                                       f"risk ({len(among)} projects)", d, one_sided(d, above=0.0)))
        secondary += [{**s, "hypothesis": f"H7/{s['order']}", "claim": f"{s['order']}: FAITHFUL/FULL > 1 on the "
                                                                         "industrial logs"}
                      for s in industrial_secondary]

    selector = pd.read_csv(args.selector_seconds).groupby("project").mean(numeric_only=True) \
        if Path(args.selector_seconds).exists() else pd.DataFrame()
    for order in (HISTORY, HYBRID):
        among = [p for p in TRACKING if (order, "online", p) in control and (order, "tuned", p) in control]
        if not among:
            continue
        deviation = {kind: np.mean([np.abs(control[(order, kind, p)]["operational"] - ALPHA) for p in among], axis=0)
                     for kind in ("online", "tuned")}
        gap = deviation["online"] - deviation["tuned"]
        misses = {p: certificates(control_rows[match(control_rows, project=p, scorer=order, controller="online",
                                                     alpha=ALPHA, pi=PI)]) for p in among}
        engineering.append({
            "criterion": f"H3/{order}", "claim": f"online minus dev-tuned mean |escape - alpha| < 0 on {len(among)} "
                                                 f"projects; adaptive certificate misses <= {MAX_MISSES} per project",
            **interval(gap), "online": float(deviation["online"][0]), "tuned": float(deviation["tuned"][0]),
            "misses": "; ".join(f"{p} {m['adaptive_misses']}/{m['seeds']}" for p, m in misses.items()),
            "met": bool(gap[0] < 0 and all(m["adaptive_misses"] <= MAX_MISSES for m in misses.values()))})
    for order in (HISTORY, HYBRID):
        for project in SAVINGS:
            if (order, "online", project) not in control:
                continue
            rows = control_rows[match(control_rows, project=project, scorer=order, controller="online", alpha=ALPHA,
                                      pi=PI)]
            seconds = rows["decision_seconds"].mean()
            if order in SELECTOR_SECONDS and project in selector.index:
                seconds += selector.loc[project, SELECTOR_SECONDS[order]]
            overhead = seconds / (rows["full_hours"].mean() * 3600)
            saved = 1 - control[(order, "online", project)]["work"] - overhead
            engineering.append({"criterion": f"H4/{order}/{project}",
                                "claim": f"net work saved, audits and selector time included: lower bound >= "
                                         f"{MIN_SAVED}", **interval(saved), "selector_seconds": float(seconds),
                                "overhead": float(overhead),
                                "met": bool(interval(saved)["low"] >= MIN_SAVED)})

    families = []
    for title, family in (("Primary hypotheses", primary), ("Secondary hypotheses", secondary)):
        df = pd.DataFrame(family)
        if len(df):
            df["holm"] = holm(df["p"].to_numpy())
            df["supported"] = df["holm"] < 0.05
        families.append((title, df))

    tables = {}
    rows = []
    for order in orders:
        among = covered(order)
        if not among:
            continue
        rows.append({"order": order, "projects": len(among),
                     **{regime: float(macro(order, regime, among)[0]) for regime in REGIMES},
                     "timing (TIMED/FULL)": fmt(ratio(order, among, "timed", "full")),
                     "censoring (FAITHFUL/TIMED)": fmt(ratio(order, among, "faithful", "timed")),
                     "total (FAITHFUL/FULL)": fmt(ratio(order, among))})
    tables["Price of information: macro operational escape at 20% executed work"] = pd.DataFrame(rows)
    tables["FAITHFUL/FULL ratio per project"] = pd.DataFrame(
        [{"order": o, "project": p, "ratio": fmt(price[(o, p, "faithful")] / price[(o, p, "full")])}
         for o in orders for p in covered(o)])

    ranked = [o for o in orders if covered(o)]
    common = [p for p in projects if all(p in covered(o) for o in ranked)]
    groups = [("macro", *({o: macro(o, regime, common) for o in ranked} for regime in ("full", "faithful")))]
    groups += [(p, {o: price[(o, p, "full")] for o in ranked if p in covered(o)},
                {o: price[(o, p, "faithful")] for o in ranked if p in covered(o)}) for p in projects]
    summary, pairs = [], []
    for name, full, faithful in groups:
        flips = significant_flips(full, faithful)
        names = sorted(full)
        tau = kendall_tau(np.array([full[o] for o in names]), np.array([faithful[o] for o in names]))
        counts = flips["status"].value_counts()
        summary.append({"projects": name if name != "macro" else f"macro ({len(common)})", "orders": len(names),
                        "pairs": len(names) * (len(names) - 1) // 2, "significant under FULL": len(flips),
                        "reversed": int(counts.get("reversed", 0)), "lost": int(counts.get("lost", 0)),
                        "tau": fmt(tau)})
        pairs += [{"projects": name, **r} for r in flips[flips["status"] != "kept"].to_dict("records")]
    tables["Ranking under FULL and FAITHFUL: pairs significant under FULL, and Kendall's tau"] = pd.DataFrame(summary)
    tables["Pairs that reverse or lose significance under FAITHFUL"] = pd.DataFrame(pairs)

    escape_rows = []
    for key, g in control_rows[control_rows["controller"].isin(["online", "tuned"])].groupby(
            ["project", "controller", "alpha", "pi", "scorer"]):
        escape_rows.append({**dict(zip(["project", "controller", "alpha", "pi", "order"], key)),
                            **{label: g[f"{label}_escape"].sum() / max(g[f"{label}_faulty"].sum(), 1)
                               for label in labels},
                            "work": g["work"].mean()})
    tables["Escape of the base selection by label, against alpha"] = pd.DataFrame(escape_rows)

    feedback = []
    for key, g in control_rows[match(control_rows, controller="online", alpha=ALPHA, pi=PI)].groupby(
            ["project", "scorer"]):
        feedback.append({"project": key[0], "order": key[1], "ttff_hours": g["operational_ttff_hours"].mean(),
                         "ttff_share": g["operational_ttff_share"].mean(),
                         "masked": g["operational_masked"].sum() / max(g["operational_faulty"].sum(), 1),
                         "verdict_hours": g["verdict_hours"].mean(), "full_hours": g["full_hours"].mean(),
                         "decision_seconds": g["decision_seconds"].mean(),
                         **({"deconf_masked": g["deconf_masked"].sum() / max(g["deconf_faulty"].sum(), 1)}
                            if not later else {}),
                         **{c: g[c].mean() for c in ("alarms", "alarm_builds") if c in g}})
    tables["Feedback time, masking and alarms (online, alpha 0.1, pi 0.1)"] = pd.DataFrame(feedback)

    widths = []
    for key, g in control_rows[control_rows["controller"] == "online"].groupby(["project", "scorer", "alpha", "pi"]):
        c = certificates(g)
        widths.append({**dict(zip(["project", "order", "alpha", "pi"], key)), "seeds": c["seeds"],
                       "adaptive_misses": c["adaptive_misses"], "adaptive_width": c["adaptive"],
                       "normal_misses": c["normal_misses"], "normal_width": c["normal"]})
    tables["Certificates: seeds missed and mean half-width"] = pd.DataFrame(widths)

    parts = [f"# Registered analysis, builds at positions [{evaluate[0]}, {evaluate[1]})\n",
             f"{args.replicates} pull-request bootstrap resamples per project.\n"]
    parts += [table(f"{title} (one-sided bootstrap p, Holm within the family)", df) for title, df in families]
    parts.append(table("Engineering criteria", pd.DataFrame(engineering)))
    parts += [table(title, df) for title, df in tables.items()]
    out = Path(args.out)
    out.write_text("\n".join(parts))
    record = {"evaluate": list(evaluate), "replicates": args.replicates,
              **{title.split()[0].lower(): df.to_dict("records") for title, df in families},
              "engineering": engineering, "tables": {title: df.to_dict("records") for title, df in tables.items()}}
    out.with_suffix(".json").write_text(json.dumps(clean(record), indent=1))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
