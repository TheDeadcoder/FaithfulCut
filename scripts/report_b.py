import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import (at_matched_work, compare, condition_sums, control_replicates, holm, interval,
                                  load_outcomes, matched_replicates, one_sided, project_builds, replicate_rates)

HISTORY, CONTENT, HYBRID, LEARNED, WEIGHTED = "LatestFail+CC", "TF-IDF+CC", "Hybrid-RRF", "Learned+CC", "Learned-W+CC"
WORK, ALPHA, PI, MARGIN = 0.2, 0.1, 0.1, 0.02
PROJECTS = ["hadoop", "jackrabbit-oak", "james", "kafka", "tvm"]
MODULE_PROJECTS = ["hadoop", "jackrabbit-oak", "james", "kafka"]
SCHEDULE, ROTATE = 8, 0.1
# columns of runs that predate an option, filled with the option's default
DEFAULTS = {"target": "operational", "cut": "cost", "path_filter": False, "schedule": 0, "rotate": 0.0}


def with_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    for column, value in DEFAULTS.items():
        rows[column] = np.where(rows[column].isna(), value, rows[column]) if column in rows else value
    return rows


def fmt(replicates: np.ndarray) -> str:
    i = interval(replicates)
    return f"{i['estimate']:.3f} [{i['low']:.3f}, {i['high']:.3f}]"


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return x


def main():
    parser = argparse.ArgumentParser(description="The second registration's tests on the later builds: "
                                                 "pull-request bootstrap, Holm within the primary and the secondary "
                                                 "family")
    parser.add_argument("--price", nargs="+", required=True, help="folders with the registered price grids")
    parser.add_argument("--repair", required=True, help="folder with the repair price grid")
    parser.add_argument("--learned", nargs="+", required=True, help="folders with the learned orders' price grids")
    parser.add_argument("--control", nargs="+", required=True, help="folders with controller runs")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True, help="markdown report; the JSON goes beside it")
    args = parser.parse_args()
    evaluate = tuple(args.evaluate)
    price_rows, price_out = load_outcomes(args.price, "price")
    repair_rows, repair_out = load_outcomes([args.repair], "price")
    learned_rows, learned_out = load_outcomes(args.learned, "price")
    control_rows, control_out = load_outcomes(args.control, "control")
    repair_rows, control_rows = with_defaults(repair_rows), with_defaults(control_rows)
    work, escape, matched = {}, {}, {}
    for project in PROJECTS:
        builds, weights = project_builds(project, evaluate, args.replicates)

        def run(order: str, label: str = "operational", **keys):
            key = dict(project=project, scorer=order, controller="online", alpha=ALPHA, pi=PI, **keys)
            if not np.logical_and.reduce([control_rows[k].to_numpy() == v for k, v in key.items()]).any():
                return
            rate, share = control_replicates(control_rows, control_out, builds, weights, label, **key)
            arm = (order, label, tuple(sorted(keys.items())))
            work[arm + (project,)], escape[arm + (project,)] = share, rate

        for order in (HISTORY, HYBRID, CONTENT):
            run(order, "newfail", target="newfail", cut="cost", path_filter=False)
        for order in (HISTORY, HYBRID, CONTENT, "Module", "Path", "Module+TF-IDF"):
            run(order, target="operational", cut="cost", path_filter=False)
        for order in (HYBRID, CONTENT):
            run(order, target="operational", cut="relevance", path_filter=False)
        run(HYBRID, target="operational", cut="cost", path_filter=True)
        for schedule, rotate in ((0, 0.0), (SCHEDULE, 0.0), (0, ROTATE)):
            sums = condition_sums(repair_rows, repair_out, "operational", project=project, scorer=HISTORY,
                                  regime="faithful", pi=0.0, schedule=schedule, rotate=rotate)
            sums["executed"] = sums["base"]
            _, rate, share = replicate_rates(sums, builds, weights)
            matched[("repair", schedule, rotate, project)] = at_matched_work(rate, share, WORK)
        for order, rows, out in ((LEARNED, learned_rows, learned_out), (WEIGHTED, learned_rows, learned_out),
                                 (HISTORY, price_rows, price_out)):
            for regime, pi in (("full", 0.0), ("faithful", 0.0), ("faithful", 0.1)):
                key = dict(project=project, scorer=order, regime=regime, pi=pi)
                if np.logical_and.reduce([rows[k].to_numpy() == v for k, v in key.items()]).any():
                    matched[(order, regime, pi, project)] = matched_replicates(rows, out, builds, weights,
                                                                               "operational", WORK, **key)
        print(project, flush=True)

    def macro(table: dict, key: tuple, among: list[str]) -> np.ndarray:
        return np.nanmean([table[key + (p,)] for p in among], axis=0)

    newfail = (("cut", "cost"), ("path_filter", False), ("target", "newfail"))
    operational = (("cut", "cost"), ("path_filter", False), ("target", "operational"))
    threshold = (("cut", "relevance"), ("path_filter", False), ("target", "operational"))
    filtered = (("cut", "cost"), ("path_filter", True), ("target", "operational"))

    def difference(table: dict, a: tuple, b: tuple, among: list[str]) -> np.ndarray:
        return macro(table, a, among) - macro(table, b, among)

    def test(name: str, claim: str, replicates: np.ndarray, p: float, **extra) -> dict:
        return {"hypothesis": name, "claim": claim, **interval(replicates), "p": p, **extra}

    def equal_risk(order: str, base: str, arm: tuple, label: str, among: list[str], above: bool, title: str,
                   name: str, margin: bool = False) -> dict:
        d = difference(work, (order, label, arm), (base, label, arm), among)
        p = one_sided(d, above=0.0) if above else one_sided(d, below=0.0)
        extra = {}
        if margin:
            e = difference(escape, (order, label, arm), (base, label, arm), among)
            p = max(p, one_sided(e, below=MARGIN))
            extra["escape"] = fmt(e)
        return test(name, f"{title} ({len(among)} projects)", d, p, **extra)

    def variant(order: str, arm: tuple, base_arm: tuple, among: list[str], above: bool, title: str, name: str,
                margin: bool = False) -> dict:
        d = difference(work, (order, "operational", arm), (order, "operational", base_arm), among)
        p = one_sided(d, above=0.0) if above else one_sided(d, below=0.0)
        extra = {}
        if margin:
            e = difference(escape, (order, "operational", arm), (order, "operational", base_arm), among)
            p = max(p, one_sided(e, below=MARGIN))
            extra["escape"] = fmt(e)
        return test(name, f"{title} ({len(among)} projects)", d, p, **extra)

    def ratio(order: str) -> np.ndarray:
        return macro(matched, (order, "faithful", 0.0), PROJECTS) / macro(matched, (order, "full", 0.0), PROJECTS)

    repair = {arm: macro(matched, ("repair",) + arm, PROJECTS) for arm in ((0, 0.0), (SCHEDULE, 0.0), (0, ROTATE))}
    learned = ratio(LEARNED)
    primary = [
        equal_risk(HYBRID, HISTORY, newfail, "newfail", PROJECTS, False,
                   f"{HYBRID} minus {HISTORY}, controller on new failures: work < 0 and new-failure escape < "
                   f"+{MARGIN}", "B1", margin=True),
        equal_risk("Module", CONTENT, operational, "operational", MODULE_PROJECTS, True,
                   f"Module minus {CONTENT}: work > 0 at equal risk", "B2"),
        variant(HYBRID, threshold, operational, PROJECTS, False,
                f"{HYBRID}, relevance cut minus cost cut: work < 0 and escape < +{MARGIN}", "B3", margin=True),
        test("B4", f"{HISTORY}, FAITHFUL at {WORK:.0%} selection work: escape with a full run every {SCHEDULE} "
                   f"builds minus without < 0 ({len(PROJECTS)} projects)", repair[(SCHEDULE, 0.0)] - repair[(0, 0.0)],
             one_sided(repair[(SCHEDULE, 0.0)] - repair[(0, 0.0)], below=0.0)),
        test("B5", f"{LEARNED}: FAITHFUL/FULL operational escape at {WORK:.0%} work > 1 ({len(PROJECTS)} projects)",
             learned, one_sided(learned, above=1.0)),
    ]
    weighted = macro(matched, (WEIGHTED, "faithful", 0.1), PROJECTS) - macro(matched, (LEARNED, "faithful", 0.1),
                                                                             PROJECTS)
    gap = learned - ratio(HISTORY)
    secondary = [
        equal_risk(CONTENT, HISTORY, newfail, "newfail", PROJECTS, False,
                   f"{CONTENT} minus {HISTORY}, controller on new failures: work < 0", "B1/content"),
        equal_risk("Path", CONTENT, operational, "operational", PROJECTS, True,
                   f"Path minus {CONTENT}: work > 0 at equal risk", "B2/path"),
        equal_risk("Module+TF-IDF", CONTENT, operational, "operational", MODULE_PROJECTS, True,
                   f"Module+TF-IDF minus {CONTENT}: work > 0 at equal risk", "B2/fusion"),
        variant(CONTENT, threshold, operational, PROJECTS, False, f"{CONTENT}, relevance cut minus cost cut: work < 0",
                "B3/content"),
        variant(HYBRID, filtered, operational, PROJECTS, True, f"{HYBRID}, path filter minus none: work > 0",
                "B3/filter"),
        test("B4/rotation", f"{HISTORY}: escape with stalest-first rotation within {ROTATE:.0%} of predicted cost "
                            f"minus without < 0 ({len(PROJECTS)} projects)", repair[(0, ROTATE)] - repair[(0, 0.0)],
             one_sided(repair[(0, ROTATE)] - repair[(0, 0.0)], below=0.0)),
        test("B5/history", f"FAITHFUL/FULL ratio of {LEARNED} minus that of {HISTORY} < 0", gap,
             one_sided(gap, below=0.0)),
        test("B5/weighted", f"{WEIGHTED} minus {LEARNED}, FAITHFUL with {PI:.0%} audits: escape differs from 0 "
                            "(two-sided)", weighted, compare(weighted, np.zeros_like(weighted))["p"]),
    ]
    families = []
    for title, family in (("Primary hypotheses", primary), ("Secondary hypotheses", secondary)):
        df = pd.DataFrame(family)
        df["holm"] = holm(df["p"].to_numpy())
        df["supported"] = df["holm"] < 0.05
        families.append((title, df))
    descriptive = []
    for order, label, arm in ((HISTORY, "newfail", newfail), (HYBRID, "newfail", newfail),
                              (CONTENT, "newfail", newfail), (CONTENT, "operational", operational),
                              ("Module", "operational", operational),
                              ("Path", "operational", operational), (HYBRID, "operational", operational),
                              (HYBRID, "operational", threshold), (HYBRID, "operational", filtered)):
        for project in PROJECTS:
            key = (order, label, arm, project)
            if key in work:
                descriptive.append({"order": order, "label": label, "arm": dict(arm)["cut"] + (
                    " filter" if dict(arm)["path_filter"] else ""), "project": project,
                                    "work": work[key][0], "escape": escape[key][0]})
    parts = [table(f"{title} (one-sided bootstrap p, Holm within the family; {args.replicates} pull-request "
                   "resamples)", df) for title, df in families]
    parts.append(table(f"Controller runs, alpha {ALPHA}, pi {PI}: executed work and escape on the controller's label",
                       pd.DataFrame(descriptive)))
    out = Path(args.out)
    out.write_text("\n".join(parts))
    out.with_suffix(".json").write_text(json.dumps(
        {title: [clean(r) for r in df.to_dict("records")] for title, df in families}, indent=1))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
