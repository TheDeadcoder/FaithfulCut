import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import (compare, control_replicates, holm, interval, load_build_outcomes,
                                  matched_replicates, project_builds)

HISTORY = "LatestFail+CC"
PRICE_ORDERS = ["LatestFail+CC", "DANTE", "Hybrid", "Hybrid-RRF", "TF-IDF+CC", "QTF"]
CHALLENGERS = ["TF-IDF+CC", "Hybrid", "Hybrid-RRF", "DANTE"]


def present(rows: pd.DataFrame, **keys) -> bool:
    return bool(np.logical_and.reduce([rows[k].to_numpy() == v for k, v in keys.items()]).any())


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    return f"## {title}\n\n```\n{body}\n```\n"


def main():
    parser = argparse.ArgumentParser(description="Hypothesis tests with pull-request bootstrap intervals")
    parser.add_argument("--price", required=True, help="folder with price.csv and price_builds.npz")
    parser.add_argument("--control", required=True, help="folder with control.csv and control_builds.npz")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--min-first-failures", type=int, default=25)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluate = tuple(args.evaluate)
    price_rows, price_out = load_build_outcomes(Path(args.price), "price")
    control_rows, control_out = load_build_outcomes(Path(args.control), "control")
    projects = sorted(set(price_rows["project"]) & set(control_rows["project"]))
    parts, tests = [], []
    ratios, equal_risk, tracking = {o: {} for o in PRICE_ORDERS}, {}, []
    for project in projects:
        builds, weights = project_builds(project, evaluate, args.replicates)
        for order in PRICE_ORDERS:
            if not present(price_rows, project=project, scorer=order, regime="faithful", pi=0.0):
                continue
            rate = {regime: matched_replicates(price_rows, price_out, builds, weights, "operational",
                                               project=project, scorer=order, regime=regime, pi=0.0)
                    for regime in ("full", "faithful")}
            ratios[order][project] = rate
        for alpha, pi in ((0.1, 0.1), (0.05, 0.25)):
            key = dict(project=project, controller="online", alpha=alpha, pi=pi)
            for order in [HISTORY] + CHALLENGERS:
                if not present(control_rows, scorer=order, **key):
                    continue
                ff, work = control_replicates(control_rows, control_out, builds, weights, "firstfail", scorer=order,
                                              **key)
                op, _ = control_replicates(control_rows, control_out, builds, weights, "operational", scorer=order,
                                           **key)
                match = np.logical_and.reduce([control_rows[k].to_numpy() == v
                                               for k, v in {**key, "scorer": order}.items()])
                faulty = control_rows.loc[match, "firstfail_faulty"].mean()
                equal_risk[(alpha, pi, order, project)] = (ff, work, faulty)
                tracking.append({"alpha": alpha, "pi": pi, "order": order, "project": project,
                                 **{f"escape_{k}": v for k, v in interval(op).items()},
                                 **{f"saved_{k}": v for k, v in interval(1 - work).items()}})
    rows = []
    for order, by_project in ratios.items():
        if not by_project:
            continue
        macro = {r: np.nanmean([v[r] for v in by_project.values()], axis=0) for r in ("full", "faithful")}
        for project, rate in list(by_project.items()) + [("macro", macro)]:
            ratio = rate["faithful"] / rate["full"]
            boot = ratio[1:][~np.isnan(ratio[1:])]
            p = float(min(1.0, ((boot <= 1).sum() + 1) / (len(boot) + 1)))
            rows.append({"order": order, "project": project, **interval(ratio), "p_one_sided": p})
            if project == "macro" and order != "QTF" and order != "TF-IDF+CC":
                tests.append((f"H1 {order}: FAITHFUL/FULL escape ratio > 1 (macro, 20% work)", p))
    parts.append(table("Price of information: FAITHFUL/FULL operational escape at 20% executed work",
                       pd.DataFrame(rows)))
    rows = []
    for alpha, pi in ((0.1, 0.1), (0.05, 0.25)):
        eligible = [p for p in projects if (alpha, pi, HISTORY, p) in equal_risk
                    and equal_risk[(alpha, pi, HISTORY, p)][2] >= args.min_first_failures]
        for order in CHALLENGERS:
            pairs = [p for p in eligible if (alpha, pi, order, p) in equal_risk]
            if not pairs:
                continue
            for name, i in (("first-failure escape", 0), ("executed work", 1)):
                a = np.nanmean([equal_risk[(alpha, pi, order, p)][i] for p in pairs], axis=0)
                b = np.nanmean([equal_risk[(alpha, pi, HISTORY, p)][i] for p in pairs], axis=0)
                c = compare(a, b)
                rows.append({"alpha": alpha, "pi": pi, "order": order, "measure": name, "projects": len(pairs), **c})
                if (alpha, pi) == (0.1, 0.1) and name == "first-failure escape":
                    tests.append((f"H4 {order} minus {HISTORY}: first-failure escape at equal risk", c["p"]))
    parts.append(table(f"Equal operational risk (online controller): order minus {HISTORY}, macro over projects "
                       f"with at least {args.min_first_failures} first-failure builds", pd.DataFrame(rows)))
    parts.append(table("Online controller: operational escape and saved test work", pd.DataFrame(tracking)))
    family = pd.DataFrame(tests, columns=["hypothesis", "p"])
    family["holm"] = holm(family["p"].to_numpy())
    parts.insert(0, table("Tests (bootstrap p-values, Holm-adjusted)", family))
    Path(args.out).write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
