import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.ir import REPOS
from faithfulcut.labels import transfer
from faithfulcut.llm import FUSIONS, evaluate
from faithfulcut.suite import load_suite

BUDGET = 0.2


def pilot_builds(projects: list[str], size: int, seed: int) -> list[tuple[str, int]]:
    candidates = []
    for project in projects:
        suite = load_suite(project)
        position = suite.builds["position"].to_numpy()
        first = np.bincount(suite.build_of_rows(), weights=transfer(suite)["firstfail"],
                            minlength=len(suite.builds)) > 0
        candidates += [(project, int(t)) for t in np.flatnonzero(first & (position >= 0.2) & (position < 0.7))]
    pick = np.random.default_rng(seed).choice(len(candidates), min(size, len(candidates)), replace=False)
    return [candidates[i] for i in sorted(pick)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="*", default=list(REPOS))
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, help="query only the first N builds of the sample, for a trial run")
    parser.add_argument("--fusion", choices=FUSIONS, default="risk")
    parser.add_argument("--offline", action="store_true", help="use saved answers only; never call the model")
    parser.add_argument("--budget", type=float, default=BUDGET, help="share of predicted time the order may use")
    args = parser.parse_args()
    picked = pilot_builds(args.projects, args.size, args.seed)[:args.limit]
    rows = []
    for project in sorted({p for p, _ in picked}):
        rows += evaluate(project, [t for p, t in picked if p == project], args.model, args.budget, args.fusion,
                         args.workers, args.dry_run, args.offline)
    df = pd.DataFrame(rows)
    out = Path("data") / "results" / "llm-pilot"
    out.mkdir(parents=True, exist_ok=True)
    suffix = ("" if args.fusion == "risk" else f"-{args.fusion}") + ("" if args.budget == BUDGET else f"-{args.budget}")
    df.to_csv(out / ("dry-run.csv" if args.dry_run else f"builds-{args.model}{suffix}.csv"), index=False)
    orders = [o for o in ("QTF", "LatestFail+CC", "TF-IDF+CC", "LLM") if o in df]
    print(f"{len(df)} builds; first-failure recall at {args.budget:.0%} of predicted time:")
    print(df[orders].mean().round(3).to_string())
    if args.dry_run:
        print("prompt characters p50/p95:", int(df["prompt_chars"].median()), int(df["prompt_chars"].quantile(0.95)))
        return
    best = max(("QTF", "LatestFail+CC", "TF-IDF+CC"), key=lambda o: df[o].mean())
    gain = df["LLM"] - df[best]
    boot = np.random.default_rng(0).choice(gain.to_numpy(), (10000, len(gain))).mean(axis=1)
    low, high = np.quantile(boot, [0.05, 0.95])
    print(f"LLM minus {best}: {gain.mean():+.3f} (90% interval {low:+.3f} to {high:+.3f});",
          "G4 passes" if gain.mean() >= 0.03 and low > 0 else "G4 does not pass")
    print("well formed:", round(df["well_formed"].mean(), 3), "| latency p50/p95 (s):",
          round(df["latency"].median(), 1), round(df["latency"].quantile(0.95), 1))
    print("input / output / thinking tokens per build, median:",
          int(df["input_tokens"].median()), int(df["output_tokens"].median()), int(df["thinking_tokens"].median()))


if __name__ == "__main__":
    main()
