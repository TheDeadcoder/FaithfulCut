import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import control_replicates, interval, load_build_outcomes, project_builds
from faithfulcut.dante import PROJECTS as JAVA

HISTORY = "LatestFail+CC"
CHALLENGERS = ["TF-IDF+CC", "Hybrid-RRF", "Hybrid", "DANTE", "QTF"]
FIRST_FAILURES = ["activemq", "hadoop", "hbase", "hive", "jackrabbit-oak", "james", "kafka", "karaf", "tvm"]
LABELS = ["newfail", "firstfail", "operational"]


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def fmt(replicates: np.ndarray) -> str:
    i = interval(replicates)
    return f"{i['estimate']:+.3f} [{i['low']:+.3f}, {i['high']:+.3f}]"


def agreement(rows: pd.DataFrame, outcomes: dict, alpha: float) -> pd.DataFrame:
    """Per project and audit rate, evaluated builds with a new failure and with a FirstFail failure, and the share of
    each that has the other, as means over seeds."""
    labels = list(outcomes["labels"])
    bit = {label: 3 * labels.index(label) for label in ("newfail", "firstfail")}
    keep = rows[(rows["scorer"] == HISTORY) & (rows["controller"] == "online") & (rows["alpha"] == alpha)]
    out = []
    for (project, pi), g in keep.groupby(["project", "pi"]):
        counts = []
        for row in g.index:
            span = slice(*np.searchsorted(outcomes["row"], [row, row + 1]))
            flags = outcomes["flags"][span].astype(np.int64)
            new, first = ((flags >> bit[label]) & 1 for label in ("newfail", "firstfail"))
            counts.append((new.sum(), first.sum(), (new & first).sum()))
        new, first, both = np.mean(counts, axis=0)
        out.append({"project": project, "pi": pi, "new": new, "firstfail": first,
                    "firstfail with a new failure": both / max(first, 1), "new with a firstfail": both / max(new, 1)})
    return pd.DataFrame(out)


def escapes(rows: pd.DataFrame) -> pd.DataFrame:
    """Escape of the base selection per label, pooled over seeds, with executed work and controller updates."""
    out = []
    for key, g in rows[rows["controller"] == "online"].groupby(["project", "alpha", "pi", "scorer"]):
        out.append({**dict(zip(["project", "alpha", "pi", "order"], key)),
                    **{label: g[f"{label}_escape"].sum() / max(g[f"{label}_faulty"].sum(), 1) for label in LABELS},
                    "work": g["work"].mean(), "updates": g["updates"].mean()})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="New failures as the controller's target: agreement with FirstFail, "
                                                 "escape by label, and pull-request bootstrap comparisons with "
                                                 "LatestFail+CC at equal risk")
    parser.add_argument("--target", required=True, help="control results whose target is the new-failure label")
    parser.add_argument("--operational", required=True, help="control results whose target is the operational label")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pi", type=float, default=0.1)
    parser.add_argument("--projects", nargs="+", default=FIRST_FAILURES)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluate = tuple(args.evaluate)
    runs = {"newfail": load_build_outcomes(Path(args.target), "control"),
            "operational": load_build_outcomes(Path(args.operational), "control")}
    parts = [table("Builds with a new failure and with a FirstFail failure (means over seeds)",
                   agreement(*runs["newfail"], args.alpha))]
    for target, (rows, _) in runs.items():
        parts.append(table(f"Escape by label with the {target} target (online controller)", escapes(rows)))
    reps = {}
    for project in args.projects:
        builds, weights = project_builds(project, evaluate, args.replicates)
        for target, (rows, outcomes) in runs.items():
            for order in [HISTORY, *CHALLENGERS]:
                key = dict(project=project, scorer=order, controller="online", alpha=args.alpha, pi=args.pi)
                if not np.logical_and.reduce([rows[k].to_numpy() == v for k, v in key.items()]).any():
                    continue
                run = {}
                for label in LABELS:
                    run[label], run["work"] = control_replicates(rows, outcomes, builds, weights, label, **key)
                reps[(target, order, project)] = run
        print(project, flush=True)
    comparisons = []
    for target in runs:
        for order in CHALLENGERS:
            among = [p for p in args.projects if (target, order, p) in reps and (target, HISTORY, p) in reps
                     and (order != "DANTE" or p in JAVA)]
            if not among:
                continue
            diff = {m: np.nanmean([reps[(target, order, p)][m] - reps[(target, HISTORY, p)][m] for p in among], axis=0)
                    for m in ["work", *LABELS]}
            comparisons.append({"target": target, "order": order, "projects": len(among),
                                **{f"{m} minus {HISTORY}": fmt(d) for m, d in diff.items()}})
    parts.append(table(f"Differences to {HISTORY} at equal risk (alpha {args.alpha}, pi {args.pi}; macro, "
                       f"{args.replicates} pull-request resamples)", pd.DataFrame(comparisons)))
    Path(args.out).write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
