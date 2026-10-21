import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import (at_matched_work, condition_sums, control_replicates, energy, interval, load_outcomes,
                                  one_sided, project_builds, replicate_rates)

LLM = "LLM"
CONTENT = "TF-IDF+CC"
LABELS = ["operational", "firstfail", "newfail"]
PRICE_LABELS = ["operational", "deconf", "firstfail"]
WORKS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def fmt(replicates: np.ndarray, digits: int = 3) -> str:
    i = interval(replicates)
    return f"{i['estimate']:+.{digits}f} [{i['low']:+.{digits}f}, {i['high']:+.{digits}f}]"


def main():
    parser = argparse.ArgumentParser(description="The LLM order and the oracle orders under the online controller on "
                                                 "one project, each against TF-IDF+CC on the same builds and seeds: "
                                                 "escape by label and executed work per controller target, with "
                                                 "pull-request bootstrap differences, and the LLM's net saving once "
                                                 "the model's tokens are paid")
    parser.add_argument("--control", nargs="+", required=True, help="control results with the LLM or oracle orders")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--project", default="james")
    parser.add_argument("--answers", help="the token ledger's per-answer table (ledger-answers.csv), for the LLM's "
                                          "net saving")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--price", nargs="*", default=[], help="price results with the LLM order and TF-IDF+CC, for "
                                                               "escape at matched executed work")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True, help="markdown report; its tables go beside it as csv")
    args = parser.parse_args()
    evaluate = tuple(args.evaluate)
    rows, outcomes = load_outcomes(args.control, "control")
    rows = rows[(rows["project"] == args.project) & (rows["controller"] == "online")]
    builds, weights = project_builds(args.project, evaluate, args.replicates)
    reps = {}
    for (target, order, alpha, pi), _ in rows.groupby(["target", "scorer", "alpha", "pi"]):
        run = {}
        for label in LABELS:
            if label in list(outcomes["labels"]):
                run[label], run["work"] = control_replicates(rows, outcomes, builds, weights, label,
                                                             project=args.project, scorer=order, controller="online",
                                                             target=target, alpha=alpha, pi=pi)
        reps[(target, order, alpha, pi)] = run
    labels = [label for label in LABELS if any(np.isfinite(r.get(label, [np.nan])[0]) for r in reps.values())]

    first = rows.index[0]
    span = slice(*np.searchsorted(outcomes["row"], [first, first + 1]))
    full_hours = outcomes["full"][span].astype(float).mean() / 3600
    paid, tokens = pd.DataFrame(), None
    if args.answers:
        paid = pd.read_csv(args.answers)
        paid = paid[(paid["model"] == args.model) & (paid["project"] == args.project)
                    & (paid["position"] >= evaluate[0]) & (paid["position"] < evaluate[1])]
        tokens = paid["dollars"].mean()
    per_hour = energy(1.0)

    levels, gaps = [], []
    for (target, order, alpha, pi), run in sorted(reps.items()):
        levels.append({"target": target, "alpha": alpha, "pi": pi, "order": order,
                       **{label: run[label][0] for label in labels}, "work": run["work"][0]})
        base = reps.get((target, CONTENT, alpha, pi))
        if order == CONTENT or base is None:
            continue
        work = run["work"] - base["work"]
        row = {"target": target, "alpha": alpha, "pi": pi, "order": order, "work": fmt(work),
               "p work < 0": one_sided(work, below=0.0), **{label: fmt(run[label] - base[label]) for label in labels}}
        if order == LLM and tokens is not None:
            hours = -work * full_hours
            row.update({"test hours saved": fmt(hours),
                        **{f"net dollars saved, {cores}": fmt(hours * per_hour[f"dollars_{end}"] - tokens, 4)
                           for end, cores in (("low", "2-core"), ("high", "8-core"))}})
        gaps.append(row)
    levels, gaps = pd.DataFrame(levels), pd.DataFrame(gaps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    levels.to_csv(out.with_suffix(".csv"), index=False)
    gaps.to_csv(out.with_name(f"{out.stem}-differences.csv"), index=False)
    seeds = rows["seed"].nunique()
    cost = (f"; for the LLM, the test time saved and the net saving after the model's tokens: {len(paid)} answers "
            f"at ${tokens:.4f} per build, {full_hours:.2f} full-suite hours per build") if tokens is not None else ""
    parts = [table(f"Escape by label and executed work ({args.project}, online controller, {seeds} seeds)", levels),
             table(f"Each order minus {CONTENT} on the same builds, seeds and target ({args.replicates} pull-request "
                   f"resamples; one-sided p for less work){cost}", gaps)]
    if args.price:
        matched = budgets(args.price, args.project, builds, weights)
        matched.to_csv(out.with_name(f"{out.stem}-budgets.csv"), index=False)
        parts.append(table(f"Escape at matched executed work without audits, LLM against {CONTENT} on the same "
                           "builds and seeds (one-sided p for less escape)", matched))
    out.write_text("\n".join(parts))
    print("\n".join(parts))


def budgets(folders: list[str], project: str, builds: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    """Per label and executed share of test time, each order's escape and the LLM's difference."""
    rows, outcomes = load_outcomes(folders, "price")
    out = []
    for label in PRICE_LABELS:
        curves = {}
        for order in (LLM, CONTENT):
            sums = condition_sums(rows, outcomes, label, project=project, scorer=order, regime="faithful", pi=0.0)
            _, rate, executed = replicate_rates(sums, builds, weights, numerator="executed_escape")
            curves[order] = {w: at_matched_work(rate, executed, w) for w in WORKS}
        for w in WORKS:
            gap = curves[LLM][w] - curves[CONTENT][w]
            if np.isfinite(gap[0]):
                out.append({"label": label, "work": w, LLM: curves[LLM][w][0], CONTENT: curves[CONTENT][w][0],
                            "difference": fmt(gap), "p": one_sided(gap, below=0.0)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
