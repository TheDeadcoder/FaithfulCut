import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import interval, load_outcomes, matched_replicates, project_builds

ORDERS = ["LatestFail+CC", "LatestFail", "DANTE", "Hybrid-RRF", "TF-IDF+CC"]
CAUSES = ["new", "seen", "hidden"]


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def fmt(replicates: np.ndarray, sign: str = "") -> str:
    i = interval(replicates)
    return f"{i['estimate']:{sign}.3f} [{i['low']:{sign}.3f}, {i['high']:{sign}.3f}]"


def by_budget(folders: list[str], evaluate: tuple[float, float], targets: list[float], replicates: int):
    rows, outcomes = load_outcomes(folders, "price")
    projects = sorted(rows["project"].unique())
    values = {}
    for project in projects:
        builds, weights = project_builds(project, evaluate, replicates)
        for order in ORDERS:
            for regime in ("full", "faithful"):
                key = dict(project=project, scorer=order, regime=regime, pi=0.0)
                if np.logical_and.reduce([rows[k].to_numpy() == v for k, v in key.items()]).any():
                    for w in targets:
                        values[(order, regime, w, project)] = matched_replicates(rows, outcomes, builds, weights,
                                                                                 "operational", w, **key)
        print(project, flush=True)
    out = []
    for order in ORDERS:
        for w in targets:
            among = [p for p in projects if all((order, r, w, p) in values for r in ("full", "faithful"))]
            if not among:
                continue
            full, faithful = (np.nanmean([values[(order, r, w, p)] for p in among], axis=0)
                              for r in ("full", "faithful"))
            out.append({"order": order, "work": w, "projects": len(among), "full": full[0], "faithful": faithful[0],
                        "ratio": fmt(faithful / full), "gap": fmt(faithful - full, "+")})
    return pd.DataFrame(out)


def anatomy(folder: str, work: float) -> pd.DataFrame:
    """Escape rates by cause, latency and wasted work interpolated at the target executed work, as macros."""
    df = pd.read_csv(Path(folder) / "price.csv")
    faulty = df["operational_faulty"].replace(0, np.nan)
    metrics = {f"escape {c}": df[f"operational_escape_{c}"] / faulty for c in CAUSES}
    metrics.update({"escape": df["operational_escape"] / faulty, "hidden builds": df["operational_hidden_builds"],
                    "hidden hours": df["operational_hidden_hours"], "waste": df["waste_share"]})
    df = df.assign(**metrics)
    rows = []
    for key, g in df.groupby(["project", "scorer", "regime", "seed"]):
        g = g.sort_values("work")
        if g["work"].min() <= work <= g["work"].max():
            rows.append({**dict(zip(["project", "order", "regime", "seed"], key)),
                         **{m: np.interp(work, g["work"], g[m]) for m in metrics}})
    at = pd.DataFrame(rows).drop(columns="seed").groupby(["order", "regime", "project"]).mean()
    macro = at.groupby(["order", "regime"]).mean().reset_index()
    full = macro[macro["regime"] == "full"].set_index("order")
    faithful = macro[macro["regime"] == "faithful"].set_index("order")
    out = faithful.drop(columns="regime").add_prefix("faithful ")
    out["extra escape"] = faithful["escape"] - full["escape"]
    for c in CAUSES:
        out[f"extra {c}"] = faithful[f"escape {c}"] - full[f"escape {c}"]
    out["full waste"] = full["waste"]
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Why history orders lose under faithful feedback: the price across "
                                                 "budgets, escapes by cause, hidden-failure latency, wasted work and "
                                                 "masking")
    parser.add_argument("--price", nargs="+", required=True, help="price grids with per-build outcomes")
    parser.add_argument("--anatomy", required=True, help="price grid run with latency and anatomy")
    parser.add_argument("--control", required=True, help="controller runs for masking")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--targets", nargs="+", type=float, default=[0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4])
    parser.add_argument("--work", type=float, default=0.2)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    parts = [table(f"Operational escape under FULL and FAITHFUL (no audits) across executed work, macro "
                   f"({args.replicates} pull-request resamples)",
                   by_budget(args.price, tuple(args.evaluate), args.targets, args.replicates)),
             table(f"FAITHFUL escapes by cause at {args.work:.0%} executed work, and the extra over FULL: new "
                   "(class never failed before), seen (its earlier failure was observed), hidden (it was not); "
                   "hidden builds and hours until a skipped failure's class runs again; share of selected test "
                   "time spent on failures outside DeConf", anatomy(args.anatomy, args.work))]
    control = pd.read_csv(Path(args.control) / "control.csv")
    control = control[(control["controller"] == "online") & (control["alpha"] == 0.1) & (control["pi"] == 0.1)]
    masking = control.groupby(["project", "scorer"])[["operational_masked", "operational_faulty", "deconf_masked",
                                                      "deconf_faulty"]].sum()
    masking = pd.DataFrame({"operational masked": masking["operational_masked"] / masking["operational_faulty"],
                            "deconf masked": masking["deconf_masked"] / masking["deconf_faulty"].replace(0, np.nan)})
    parts.append(table("Failing builds whose real failure the selection missed while catching another failure "
                       "(online, alpha 0.1, pi 0.1)", masking.unstack("scorer").round(3).reset_index()))
    Path(args.out).write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
