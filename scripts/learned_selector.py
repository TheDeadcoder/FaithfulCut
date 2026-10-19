import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import (compare, control_replicates, interval, load_outcomes, matched_replicates,
                                  project_builds)

LEARNED, WEIGHTED, HISTORY = "Learned+CC", "Learned-W+CC", "LatestFail+CC"
ORDERS = [LEARNED, HISTORY, "TF-IDF+CC", "Hybrid-RRF"]
CONDITIONS = [("full", 0.0), ("timed", 0.0), ("faithful", 0.0), ("faithful", 0.1)]
WORK = 0.2


def match(rows: pd.DataFrame, **keys) -> np.ndarray:
    return np.logical_and.reduce([rows[k].to_numpy() == v for k, v in keys.items()])


def fmt(replicates: np.ndarray) -> str:
    i = interval(replicates)
    return f"{i['estimate']:.3f} [{i['low']:.3f}, {i['high']:.3f}]"


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def main():
    parser = argparse.ArgumentParser(description="The learned order under censoring: its price of information "
                                                 "against the heuristic orders, the ranking under FULL and FAITHFUL, "
                                                 "propensity-weighted training, and work at equal risk")
    parser.add_argument("--price", nargs="+", required=True, help="folders with price.csv and price_builds.npz")
    parser.add_argument("--control", nargs="+", required=True, help="folders with control.csv and control_builds.npz")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pi", type=float, default=0.1)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    price_rows, price_out = load_outcomes(args.price, "price")
    control_rows, control_out = load_outcomes(args.control, "control")
    projects = sorted(set(price_rows["project"]))
    escape, control = {}, {}
    for project in projects:
        builds, weights = project_builds(project, tuple(args.evaluate), args.replicates)
        for order in ORDERS + [WEIGHTED]:
            for regime, pi in CONDITIONS:
                key = dict(project=project, scorer=order, regime=regime, pi=pi)
                if match(price_rows, **key).any():
                    escape[(order, regime, pi, project)] = matched_replicates(price_rows, price_out, builds, weights,
                                                                              "operational", WORK, **key)
        for order in ORDERS + [WEIGHTED]:
            key = dict(project=project, scorer=order, controller="online", alpha=args.alpha, pi=args.pi)
            if match(control_rows, **key).any():
                control[(order, project)] = control_replicates(control_rows, control_out, builds, weights,
                                                               "operational", **key)
        print(project, flush=True)

    def macro(order: str, regime: str, pi: float, among: list[str]) -> np.ndarray:
        return np.nanmean([escape[(order, regime, pi, p)] for p in among], axis=0)

    def covered(conditions: list[tuple[str, str, float]]) -> list[str]:
        return [p for p in projects if all((o, r, pi, p) in escape for o, r, pi in conditions)]

    regimes = [(o, r, 0.0) for o in ORDERS for r in ("full", "timed", "faithful")]
    among = covered(regimes)
    price = []
    for order in ORDERS:
        full, timed, faithful = (macro(order, r, 0.0, among) for r in ("full", "timed", "faithful"))
        price.append({"order": order, "full": full[0], "timed": timed[0], "faithful": faithful[0],
                      "price": fmt(faithful / full), "timing": fmt(timed / full), "censoring": fmt(faithful / timed)})
    ranking = []
    for order in ORDERS[1:]:
        for regime in ("full", "faithful"):
            ranking.append({"pair": f"{LEARNED} minus {order}", "regime": regime,
                            **compare(macro(LEARNED, regime, 0.0, among), macro(order, regime, 0.0, among))})
    per_project = [{"project": p, **{f"{o} {r}": escape[(o, r, 0.0, p)][0] for o in (LEARNED, HISTORY)
                                     for r in ("full", "faithful")}} for p in among]
    weighted = covered([(LEARNED, "faithful", 0.1), (WEIGHTED, "faithful", 0.1)])
    ipw = [{"project": p, "plain": escape[(LEARNED, "faithful", 0.1, p)][0],
            "weighted": escape[(WEIGHTED, "faithful", 0.1, p)][0],
            "difference": fmt(escape[(WEIGHTED, "faithful", 0.1, p)] - escape[(LEARNED, "faithful", 0.1, p)])}
           for p in weighted]
    plain, weighted_macro = (macro(o, "faithful", 0.1, weighted) for o in (LEARNED, WEIGHTED))
    ipw.append({"project": "macro", "plain": plain[0], "weighted": weighted_macro[0],
                "difference": fmt(weighted_macro - plain)})
    controlled = [p for p in projects if all((o, p) in control for o in ORDERS + [WEIGHTED])]
    equal_risk = []
    for order in ORDERS + [WEIGHTED]:
        rate, work = (np.nanmean([control[(order, p)][i] for p in controlled], axis=0) for i in (0, 1))
        base = np.nanmean([control[(HISTORY, p)][1] for p in controlled], axis=0)
        equal_risk.append({"order": order, "escape": fmt(rate), "work": fmt(work),
                           f"work minus {HISTORY}": fmt(work - base)})
    parts = [table(f"Price of information at {WORK:.0%} executed work, no audits: macro over {len(among)} projects "
                   f"({args.replicates} pull-request resamples)", pd.DataFrame(price)),
             table(f"Escape of {LEARNED} minus each order, under FULL and under FAITHFUL (two-sided p)",
                   pd.DataFrame(ranking)),
             table("Escape per project", pd.DataFrame(per_project)),
             table(f"Propensity-weighted training: escape at {WORK:.0%} executed work under FAITHFUL with 10% audits",
                   pd.DataFrame(ipw)),
             table(f"Online controller, alpha {args.alpha}, pi {args.pi}: operational escape and executed work, macro "
                   f"over {len(controlled)} projects", pd.DataFrame(equal_risk))]
    Path(args.out).write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
