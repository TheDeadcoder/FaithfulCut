import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import (at_matched_work, condition_sums, control_replicates, energy, interval, load_outcomes,
                                  project_builds, replicate_rates, work_at_rate)

ORDERS = ["LatestFail+CC", "Hybrid-RRF", "TF-IDF+CC"]
SELECTOR_SECONDS = {"TF-IDF": "tfidf_seconds", "TF-IDF+CC": "tfidf_seconds", "Hybrid": "tfidf_seconds",
                    "Hybrid-RRF": "tfidf_seconds", "DANTE": "dante_seconds"}
BUILDS = 1000


def match(rows: pd.DataFrame, **keys) -> np.ndarray:
    return np.logical_and.reduce([rows[k].to_numpy() == v for k, v in keys.items()])


def fmt(replicates: np.ndarray, digits: int = 3) -> str:
    i = interval(replicates)
    return f"{i['estimate']:.{digits}f} [{i['low']:.{digits}f}, {i['high']:.{digits}f}]"


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def converted(hours: float) -> dict[str, str]:
    e = energy(hours)
    return {"kWh": f"{e['kwh_low']:.1f} to {e['kwh_high']:.1f}",
            "kg CO2e": f"{e['kg_co2e_low']:.1f} to {e['kg_co2e_high']:.1f}",
            "dollars": f"{e['dollars_low']:.0f} to {e['dollars_high']:.0f}"}


def ledger(results: dict, orders: list[str], projects: list[str], column: str) -> pd.DataFrame:
    """Per project and order, and as the mean over projects, a share of test time and the test-hours it amounts to
    per BUILDS builds, with intervals, converted to energy, emissions and runner cost."""
    rows = []
    for order in orders:
        named = []
        for project in projects:
            if (order, project) in results:
                share, hours = results[(order, project)]
                named.append((project, share, share * hours * BUILDS))
        if named:
            named.append(("mean", np.nanmean([n[1] for n in named], axis=0),
                          np.nanmean([n[2] for n in named], axis=0)))
        for project, share, total in named:
            rows.append({"project": project, "order": order, column: fmt(share),
                         f"test-hours per {BUILDS} builds": fmt(total, 0), **converted(float(total[0]))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="The online controller's net saving in test time, energy, "
                                                 "emissions and runner cost; feedback time; and the price of "
                                                 "censoring in test time, with pull-request bootstrap intervals")
    parser.add_argument("--price", nargs="+", required=True, help="folders with price.csv and price_builds.npz")
    parser.add_argument("--control", required=True, help="folder with control.csv and control_builds.npz")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--orders", nargs="+", default=ORDERS)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pi", type=float, default=0.1)
    parser.add_argument("--work", type=float, default=0.2)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--selector-seconds", default="data/results/overhead/selector_seconds.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    price_rows, price_out = load_outcomes(args.price, "price")
    control_rows, control_out = load_outcomes([args.control], "control")
    selector = pd.read_csv(args.selector_seconds).groupby("project").mean(numeric_only=True) \
        if Path(args.selector_seconds).exists() else pd.DataFrame()
    projects = sorted(set(price_rows["project"]) & set(control_rows["project"]))
    saved, extra, feedback, escapes = {}, {}, [], []
    for project in projects:
        builds, weights = project_builds(project, tuple(args.evaluate), args.replicates)
        for order in args.orders:
            key = dict(project=project, scorer=order, controller="online", alpha=args.alpha, pi=args.pi)
            rows = control_rows[match(control_rows, **key)]
            if len(rows):
                seconds = rows["decision_seconds"].mean()
                if order in SELECTOR_SECONDS and project in selector.index:
                    seconds += selector.loc[project, SELECTOR_SECONDS[order]]
                full = rows["full_hours"].mean()
                _, work = control_replicates(control_rows, control_out, builds, weights, "operational", **key)
                saved[(order, project)] = (1 - work - seconds / (full * 3600), full)
                feedback.append({"project": project, "order": order,
                                 "first failure hours": rows["operational_ttff_hours"].mean(),
                                 "first failure share": rows["operational_ttff_share"].mean(),
                                 "verdict hours": rows["verdict_hours"].mean(), "full suite hours": full})
            curves = {}
            for regime in ("full", "faithful"):
                key = dict(project=project, scorer=order, regime=regime, pi=0.0)
                if match(price_rows, **key).any():
                    _, rate, work = replicate_rates(condition_sums(price_rows, price_out, "operational", **key),
                                                    builds, weights)
                    curves[regime] = rate, work
            if len(curves) == 2:
                target = at_matched_work(*curves["full"], args.work)
                full = price_rows.loc[match(price_rows, project=project, scorer=order), "full_hours"].mean()
                extra[(order, project)] = (work_at_rate(*curves["faithful"], target) - args.work, full)
                escapes.append({"project": project, "order": order, "full": target[0],
                                "faithful": at_matched_work(*curves["faithful"], args.work)[0]})
        print(project, flush=True)
    parts = [table(f"Net saving of the online controller (alpha {args.alpha}, pi {args.pi}) after audits and "
                   f"selector time, as a share of test time and per {BUILDS} builds; energy, emissions and runner "
                   f"cost from a 2-core executor at half load to an 8-core one at full load",
                   ledger(saved, args.orders, projects, "saved share")),
             table("Feedback in hours of test time: until the first failure is found, and until the verdict, against "
                   "the full suite", pd.DataFrame(feedback)),
             table(f"Operational escape at {args.work:.0%} executed work, without audits", pd.DataFrame(escapes)),
             table(f"The price of censoring in test time: the extra share FAITHFUL needs to reach FULL's escape at "
                   f"{args.work:.0%} work, and what it costs per {BUILDS} builds",
                   ledger(extra, args.orders, projects, "extra share"))]
    Path(args.out).write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
