import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.ir import REPOS, load_ir
from faithfulcut.labels import transfer
from faithfulcut.llm import cached_answer, prompt, risk_levels, timed_state
from faithfulcut.suite import load_suite

EPS = 1e-6
BUDGET = 0.2
FUSIONS = ("risk", "cost", "rrf")


def caught(keys: list[np.ndarray], duration: np.ndarray, failing: np.ndarray, rng: np.random.Generator,
           budget: float = BUDGET) -> float:
    """Whether a failing class enters the prefix within `budget` of predicted time under the lexicographic keys."""
    order = np.lexsort([rng.random(len(duration))] + keys[::-1])
    cumulative = np.cumsum(duration[order])
    depth = np.empty(len(order))
    depth[order] = cumulative / cumulative[-1] if cumulative[-1] > 0 else 0.0
    return float((depth[failing] <= budget).any())


def llm_keys(ranks: np.ndarray, tfidf_cc: np.ndarray, duration: np.ndarray, fusion: str) -> list[np.ndarray]:
    """Order keys for the LLM's risk levels: risk first then TF-IDF+CC ("risk"); risk value per predicted second
    ("cost"); or reciprocal rank fusion of the "risk" order with TF-IDF+CC ("rrf")."""
    if fusion == "risk":
        return [ranks, tfidf_cc]
    if fusion == "cost":
        return [-(3 - ranks) / (duration + EPS), tfidf_cc]
    fused = np.zeros(len(ranks))
    for order in (np.lexsort((tfidf_cc, ranks)), np.argsort(tfidf_cc, kind="stable")):
        position = np.empty(len(ranks))
        position[order] = np.arange(1, len(ranks) + 1)
        fused += 1 / (60 + position)
    return [-fused]


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


def evaluate(project: str, builds: list[int], model: str, dry_run: bool, workers: int, fusion: str = "risk",
             offline: bool = False, budget: float = BUDGET) -> list[dict]:
    suite = load_suite(project)
    first = transfer(suite)["firstfail"]
    ir = load_ir(suite, "seen")
    state = timed_state(suite, builds)
    rows = []

    def one(t):
        catalog, duration, since_fail = state[t]
        if dry_run:
            return t, None, len(prompt(suite, t, catalog))
        return t, cached_answer(suite, t, catalog, model, offline), None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(one, builds))
    for t, answer, prompt_chars in answers:
        catalog, duration, since_fail = state[t]
        rows_t = suite.rows(t)
        position = {c: i for i, c in enumerate(catalog)}
        failing_cls = suite.cls[rows_t][first[rows_t]]
        if not all(c in position for c in failing_cls) or (offline and answer is None):
            continue
        failing = np.array([position[c] for c in failing_cls])
        scores = np.zeros(len(suite.names))
        part = slice(ir["offsets"][t], ir["offsets"][t + 1])
        scores[ir["cls"][part]] = ir["score"][part]
        tfidf_cc = 1 / (scores[catalog] + EPS) * (duration + EPS)
        rng = lambda: np.random.default_rng([t, len(catalog)])
        row = {"project": project, "build": t, "classes": len(catalog),
               "QTF": caught([duration], duration, failing, rng(), budget),
               "LatestFail+CC": caught([(since_fail + EPS) * (duration + EPS)], duration, failing, rng(), budget),
               "TF-IDF+CC": caught([tfidf_cc], duration, failing, rng(), budget)}
        if answer is None:
            row["prompt_chars"] = prompt_chars
        else:
            ranks, ok = risk_levels(answer, suite.names[catalog])
            usage = answer.get("usageMetadata", {})
            row.update({"LLM": caught(llm_keys(ranks, tfidf_cc, duration, fusion), duration, failing, rng(), budget),
                        "well_formed": ok,
                        "latency": answer["latency"], "input_tokens": usage.get("promptTokenCount"),
                        "output_tokens": usage.get("candidatesTokenCount"),
                        "thinking_tokens": usage.get("thoughtsTokenCount", 0)})
        rows.append(row)
    return rows


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
        rows += evaluate(project, [t for p, t in picked if p == project], args.model, args.dry_run, args.workers,
                         args.fusion, args.offline, args.budget)
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
