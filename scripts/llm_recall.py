import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut import lrts
from faithfulcut.analysis import interval, one_sided, pooled_share, project_builds
from faithfulcut.ir import REPOS
from faithfulcut.labels import transfer
from faithfulcut.llm import FUSIONS, evaluate
from faithfulcut.suite import load_suite

EVALUATE = (0.7, 1.0)
BUDGET = 0.05
BASELINE = "TF-IDF+CC"


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def first_failure_builds(project: str, seed: int = 0) -> tuple[list[int], dict[int, int]]:
    """The project's confirmatory builds with a FirstFail failure, and for each another confirmatory build with a
    diff, drawn at random, whose diff the shuffled control shows in place of its own."""
    suite = load_suite(project)
    position = suite.builds["position"].to_numpy()
    confirmatory = (position >= EVALUATE[0]) & (position < EVALUATE[1])
    first = np.bincount(suite.build_of_rows(), weights=transfer(suite)["firstfail"],
                        minlength=len(suite.builds)) > 0
    builds = np.flatnonzero(first & confirmatory).tolist()
    pool = [int(u) for u in np.flatnonzero(confirmatory)
            if lrts.diff_path(project, suite.builds["pr_name"].iat[u], suite.builds["build_id"].iat[u]).exists()]
    rng = np.random.default_rng(seed)
    return builds, {t: int(rng.choice([u for u in pool if u != t])) for t in builds}


def main():
    parser = argparse.ArgumentParser(description="H8: first-failure recall at 5% of predicted test time, the LLM's "
                                                 "risk order against TF-IDF+CC on the confirmatory split, pooled over "
                                                 "builds, with a pull-request bootstrap within each project")
    parser.add_argument("--projects", nargs="*", default=list(REPOS))
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--fusion", choices=FUSIONS, default="risk")
    parser.add_argument("--budget", type=float, default=BUDGET)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true", help="count builds and prompt sizes; never call the model")
    parser.add_argument("--shuffled", action="store_true",
                        help="also ask with each build's diff replaced by another confirmatory build's")
    parser.add_argument("--out", required=True, help="markdown report; the result for report.py and the per-build "
                                                     "table go beside it")
    args = parser.parse_args()
    frames = []
    for project in args.projects:
        builds, diffs = first_failure_builds(project)
        rows = pd.DataFrame(evaluate(project, builds, args.model, args.budget, args.fusion, args.workers,
                                     args.dry_run))
        if args.shuffled and not args.dry_run and len(rows):
            shuffled = pd.DataFrame(evaluate(project, builds, args.model, args.budget, args.fusion, args.workers,
                                             diffs=diffs, variant="-shuffled"))
            rows = rows.merge(shuffled[["build", "LLM", "well_formed"]].rename(
                columns={"LLM": "LLM-shuffled", "well_formed": "shuffled_well_formed"}), on="build")
        frames.append(rows)
        print(project, len(rows), flush=True)
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    if args.dry_run:
        print(df.groupby("project").size().to_string())
        print(f"{len(df)} builds; prompt characters: total {df['prompt_chars'].sum()}, median "
              f"{int(df['prompt_chars'].median())}")
        return

    orders = [o for o in ("QTF", "LatestFail+CC", "TF-IDF+CC", "LLM", "LLM-shuffled") if o in df]
    weights = {}
    for project, g in df.groupby("project"):
        builds, w = project_builds(project, EVALUATE, args.replicates)
        assert np.isin(g["build"], builds).all()
        weights[project] = w[:, np.searchsorted(builds, g["build"].to_numpy())]

    def pooled(order: str) -> np.ndarray:
        return pooled_share([(weights[p], g[order]) for p, g in df.groupby("project")])

    def macro(order: str) -> np.ndarray:
        return np.nanmean([pooled_share([(weights[p], g[order])]) for p, g in df.groupby("project")], axis=0)

    def compare(a: str, b: str) -> dict:
        out = {}
        for name, statistic in (("pooled", pooled), ("macro", macro)):
            d = statistic(a) - statistic(b)
            out[name] = {**interval(d), "p": one_sided(d, above=0.0)}
        return out

    main_result = compare("LLM", BASELINE)
    result = {**main_result["pooled"], "macro": main_result["macro"], "builds": len(df),
              "recall": {o: float(df[o].mean()) for o in orders}, "well_formed": float(df["well_formed"].mean())}
    comparisons = [("LLM", BASELINE)]
    if "LLM-shuffled" in df:
        comparisons += [("LLM-shuffled", BASELINE), ("LLM", "LLM-shuffled")]
        result["shuffled"] = {f"{a} minus {b}": compare(a, b) for a, b in comparisons[1:]}
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))

    recall = [{"project": p, "builds": len(g), **{o: g[o].mean() for o in orders}} for p, g in df.groupby("project")]
    recall.append({"project": "pooled", "builds": len(df), **{o: df[o].mean() for o in orders}})
    tests = [{"comparison": f"{a} minus {b}", "statistic": name, **values}
             for a, b in comparisons for name, values in compare(a, b).items()]
    usage = {"well formed": df["well_formed"].mean(), "latency p50": df["latency"].median(),
             "latency p95": df["latency"].quantile(0.95), "input tokens": int(df["input_tokens"].sum()),
             "output tokens": int(df["output_tokens"].sum())}
    parts = [table(f"FirstFail recall at {args.budget:.0%} of predicted test time ({args.model}, {args.fusion} "
                   "fusion)", pd.DataFrame(recall)),
             table(f"Differences in recall ({args.replicates} pull-request resamples within each project; one-sided "
                   "p for > 0); H8 is the pooled LLM minus TF-IDF+CC", pd.DataFrame(tests)),
             table("Answers", pd.DataFrame([usage]))]
    out.write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
