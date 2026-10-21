import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import auc, budget_flags, interval, load_outcomes, one_sided, project_builds
from faithfulcut.experiments import FULL_RUN_BIT

SHARES = [0.05, 0.1, 0.2]
LABELS = ["operational", "deconf", "firstfail", "newfail"]
STEP = 0.1


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def fmt(replicates: np.ndarray) -> str:
    i = interval(replicates)
    return f"{i['estimate']:.3f} [{i['low']:.3f}, {i['high']:.3f}]"


def fmt_row(r: pd.Series) -> str:
    return f"{r['estimate']:.3f} [{r['low']:.3f}, {r['high']:.3f}]"


def mean(values: list[float]) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def build_sums(outcomes: dict, runs, builds: np.ndarray, shares: list[float], labels: list[str],
               step: float) -> tuple[dict, dict]:
    """Per evaluated build, sums over runs of executed and full test time, of the review flag for each budget, of the
    test time a full run adds to a flagged build, and per label of escapes (faulty builds that neither the selection
    nor a full run caught), their flagged part and the part random review at the run's flagged share would catch;
    per label and run, the AUC of skipped relevance between escaped and caught faulty builds."""
    names = list(outcomes["labels"])
    sums = defaultdict(lambda: np.zeros(len(builds)))
    aucs = defaultdict(list)
    for row in runs:
        span = slice(*np.searchsorted(outcomes["row"], [row, row + 1]))
        b = np.searchsorted(builds, outcomes["build"][span])
        assert np.array_equal(builds[b], outcomes["build"][span])
        flags = outcomes["flags"][span].astype(np.int64)
        signal = outcomes["signal"][span].astype(float)
        full, executed = (outcomes[k][span].astype(float) for k in ("full", "executed"))
        full_run = ((flags >> FULL_RUN_BIT) & 1).astype(bool)
        flagged = {x: budget_flags(signal, x, step) for x in shares}
        sums["full"][b] += full
        sums["executed"][b] += executed
        for x in shares:
            sums[f"reviewed {x}"][b] += flagged[x]
            sums[f"added {x}"][b] += flagged[x] * (full - executed)
        for label in labels:
            i = 3 * names.index(label)
            faulty, caught = (((flags >> i + k) & 1).astype(bool) for k in (0, 1))
            escaped = faulty & ~caught & ~full_run
            sums[f"{label} escaped"][b] += escaped
            for x in shares:
                sums[f"{label} flagged {x}"][b] += flagged[x] * escaped
                sums[f"{label} random {x}"][b] += flagged[x].mean() * escaped
            aucs[label].append(auc(np.nan_to_num(signal[faulty], nan=-np.inf), ~caught[faulty]))
    return sums, aucs


def main():
    parser = argparse.ArgumentParser(description="Review triage by skipped change relevance: a build is flagged when "
                                                 "the share of its change relevance outside the cut exceeds an online "
                                                 "threshold that tracks a review budget; reports the online "
                                                 "controller's escapes in flagged builds against random review at the "
                                                 "same rate, the test time a full run of them adds, and how well the "
                                                 "signal separates escaped from caught faulty builds")
    parser.add_argument("--control", nargs="+", required=True, help="control results with per-build outcomes")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pi", type=float, default=0.1)
    parser.add_argument("--shares", nargs="+", type=float, default=SHARES)
    parser.add_argument("--labels", nargs="+", default=LABELS)
    parser.add_argument("--step", type=float, default=STEP, help="the review threshold's step size")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True, help="markdown report; its tables go beside it as csv")
    args = parser.parse_args()
    evaluate = tuple(args.evaluate)
    rows, outcomes = load_outcomes(args.control, "control")
    keep = rows[(rows["controller"] == "online") & (rows["alpha"] == args.alpha) & (rows["pi"] == args.pi)]
    seeds = keep["seed"].nunique()
    labels = [label for label in args.labels if label in list(outcomes["labels"])]
    pooled, per_project = defaultdict(float), []
    for project, g in keep.groupby("project"):
        builds, weights = project_builds(project, evaluate, args.replicates)
        for order, runs in g.groupby("scorer"):
            sums, aucs = build_sums(outcomes, runs.index, builds, args.shares, labels, args.step)
            metrics = list(sums)
            totals = weights @ np.column_stack([sums[m] for m in metrics])
            for j, m in enumerate(metrics):
                pooled[(order, m)] = pooled[(order, m)] + totals[:, j]
            per_project.append({"project": project, "order": order, "builds": len(builds), "runs": len(runs),
                                **{m: totals[0, j] for j, m in enumerate(metrics)},
                                **{f"{label} auc": mean(v) for label, v in aucs.items()}})
        print(project, flush=True)
    projects = pd.DataFrame(per_project)
    orders = sorted(keep["scorer"].unique())

    def ratio(order: str, num: str, den: str) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return pooled[(order, num)] / pooled[(order, den)]

    results, work = [], []
    for order in orders:
        mine = projects[projects["order"] == order]
        build_runs = (mine["builds"] * mine["runs"]).sum()
        for label in labels:
            if not pooled[(order, f"{label} escaped")][0]:
                continue
            has = mine[mine[f"{label} escaped"] > 0]
            for x in args.shares:
                share = ratio(order, f"{label} flagged {x}", f"{label} escaped")
                chance = ratio(order, f"{label} random {x}", f"{label} escaped")
                results.append({"order": order, "label": label, "review": x,
                                "reviewed": pooled[(order, f"reviewed {x}")][0] / build_runs,
                                "escapes per seed": pooled[(order, f"{label} escaped")][0] / seeds,
                                **interval(share), "random": chance[0], "lift": share[0] / chance[0],
                                "p": one_sided(share - chance, above=0.0),
                                "macro": (has[f"{label} flagged {x}"] / has[f"{label} escaped"]).mean(),
                                "projects": len(has), "auc": mean(mine[f"{label} auc"])})
        work.append({"order": order, "work": ratio(order, "executed", "full")[0],
                     **{f"added at {x:.0%}": fmt(ratio(order, f"added {x}", "full")) for x in args.shares}})
    results, work = pd.DataFrame(results), pd.DataFrame(work)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out.with_suffix(".csv"), index=False)
    projects.to_csv(out.with_name(f"{out.stem}-projects.csv"), index=False)

    cells = results.assign(cell=[f"{fmt_row(r)} x{r['lift']:.1f}" for _, r in results.iterrows()])
    wide = cells.pivot_table(index=["order", "label"], columns="review", values="cell", aggfunc="first")
    wide.columns = [f"review {x:.0%}" for x in wide.columns]
    discrimination = results[results["review"] == args.shares[0]].pivot(index="order", columns="label", values="auc")
    main_orders = [o for o in ("LatestFail+CC", "TF-IDF+CC", "Hybrid-RRF") if o in orders]
    shown = [label for label in ("operational", "firstfail", "newfail") if label in set(results["label"])]
    middle = args.shares[len(args.shares) // 2]
    detail = projects[projects["order"].isin(main_orders)].assign(**{
        f"{label} at {middle:.0%}": projects[f"{label} flagged {middle}"]
        / projects[f"{label} escaped"].replace(0, np.nan) for label in shown})
    detail = detail[["project", "order", "builds", *[f"{label} at {middle:.0%}" for label in shown],
                     *[f"{label} auc" for label in shown]]]
    parts = [
        table(f"Share of escapes in reviewed builds (online, alpha {args.alpha}, pi {args.pi}, {seeds} seeds; pooled "
              f"over projects with {args.replicates} pull-request resamples; x = lift over random review at the same "
              "rate)", wide.reset_index()),
        table("Realized review rate, escapes per seed, the share random review would catch, one-sided p for "
              "catching more than it, and the macro share over projects with escapes",
              results[["order", "label", "review", "reviewed", "escapes per seed", "random", "p", "macro",
                       "projects"]]),
        table("Executed share of test time, and what a full run of the reviewed builds adds", work),
        table("AUC of skipped relevance, escaped against caught faulty builds (mean over seeds, then projects)",
              discrimination.reset_index()),
        table(f"Per project, main orders: share of escapes in builds flagged at a {middle:.0%} budget, and AUC",
              detail),
    ]
    out.write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
