import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import control_replicates, interval, load_build_outcomes, matched_replicates, project_builds

LABELS = ["operational", "firstfail", "newfail"]


def fmt(replicates: np.ndarray) -> str:
    i = interval(replicates)
    return f"{i['estimate']:+.3f} [{i['low']:+.3f}, {i['high']:+.3f}]"


def present(rows: pd.DataFrame, **keys) -> bool:
    return bool(np.logical_and.reduce([rows[k].to_numpy() == v for k, v in keys.items()]).any())


def main():
    parser = argparse.ArgumentParser(description="A variant's results against a baseline's for the same orders, with "
                                                 "pull-request bootstrap intervals for the macro differences; each "
                                                 "folder holds one variant of each order")
    parser.add_argument("--kind", choices=["price", "control"], required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--orders", nargs="+", required=True)
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--work", nargs="+", type=float, default=[0.1, 0.2, 0.3], help="matched work for price grids")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pi", type=float, default=0.1)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = {name: load_build_outcomes(Path(folder), args.kind) for name, folder in
            (("variant", args.variant), ("baseline", args.baseline))}
    labels = [l for l in LABELS if all(l in list(out["labels"]) for _, out in runs.values())]
    projects = sorted(set.intersection(*(set(rows["project"]) for rows, _ in runs.values())))
    if args.kind == "price":
        conditions = sorted(set.intersection(*(set(map(tuple, rows[["regime", "pi"]].to_numpy().tolist()))
                                                for rows, _ in runs.values())))
        cases = [(order, regime, pi, label, w) for order in args.orders for regime, pi in conditions
                 for label in labels if label != "newfail" for w in args.work]
    else:
        cases = [(order, None, None, label, None) for order in args.orders for label in labels + ["work"]]
    values = {}
    for project in projects:
        builds, weights = project_builds(project, tuple(args.evaluate), args.replicates)
        for name, (rows, outcomes) in runs.items():
            for order, regime, pi, label, w in cases:
                if args.kind == "price":
                    key = dict(project=project, scorer=order, regime=regime, pi=pi)
                    if present(rows, **key):
                        values[(name, order, regime, pi, label, w, project)] = matched_replicates(
                            rows, outcomes, builds, weights, label, w, **key)
                else:
                    key = dict(project=project, scorer=order, controller="online", alpha=args.alpha, pi=args.pi)
                    if present(rows, **key) and label != "work":
                        rate, work = control_replicates(rows, outcomes, builds, weights, label, **key)
                        values[(name, order, regime, pi, label, w, project)] = rate
                        values[(name, order, regime, pi, "work", w, project)] = work
        print(project, flush=True)
    out = []
    for order, regime, pi, label, w in cases:
        among = [p for p in projects if all((n, order, regime, pi, label, w, p) in values for n in runs)]
        if not among:
            continue
        macro = {n: np.nanmean([values[(n, order, regime, pi, label, w, p)] for p in among], axis=0) for n in runs}
        out.append({"order": order, **({"regime": regime, "pi": pi, "work": w} if args.kind == "price" else {}),
                    "metric": label, "projects": len(among), "variant": interval(macro["variant"])["estimate"],
                    "baseline": interval(macro["baseline"])["estimate"],
                    "difference": fmt(macro["variant"] - macro["baseline"])})
    title = "Escape at matched executed work" if args.kind == "price" else \
        f"Online controller at alpha {args.alpha}, pi {args.pi}: escape and executed work"
    body = pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    text = f"## {title}: variant minus baseline (macro, {args.replicates} pull-request resamples)\n\n" \
           f"Variant: {args.variant}. Baseline: {args.baseline}.\n\n```\n{body}\n```\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
