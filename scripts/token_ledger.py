import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut import lrts
from faithfulcut.analysis import energy, interval, one_sided, project_builds
from faithfulcut.llm import FUSIONS, budget_for_share, evaluate
from faithfulcut.suite import load_suite

EVALUATE = (0.7, 1.0)
BUDGET = 0.05
GRID = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
ORDERS = ["QTF", "LatestFail+CC", "TF-IDF+CC", "LLM"]
BASELINE = "TF-IDF+CC"
# USD per million input and output tokens (output includes reasoning): standard on-demand rate, global endpoint
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50), "gemini-3.8-flash": (1.50, 7.50)}
# Elsworth et al.'s median Gemini Apps text prompt: an order-of-magnitude anchor for one call, not a measurement
WH_PER_PROMPT = 0.24


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.4g}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def fmt(replicates: np.ndarray, digits: int = 3) -> str:
    i = interval(replicates)
    return f"{i['estimate']:.{digits}f} [{i['low']:.{digits}f}, {i['high']:.{digits}f}]"


def split(position: float) -> str:
    return "dev" if position < EVALUATE[0] else "held-out" if position < EVALUATE[1] else "later"


def cost(model: str, input_tokens, output_tokens):
    """List-price dollars of a call; output tokens include reasoning."""
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def answers(cache: Path) -> pd.DataFrame:
    """Every saved answer of a priced model: its build, split, tokens, latency and list-price cost."""
    positions, rows = {}, []
    for path in sorted(cache.glob("*/*/*.json")):
        model, project = path.parent.parent.name, path.parent.name
        if model not in PRICES:
            continue
        if project not in positions:
            b = load_suite(project, later=True).builds
            positions[project] = dict(zip(map(lrts.build_name, b["pr_name"], b["build_id"]), b["position"]))
        answer = json.loads(path.read_text())
        usage = answer.get("usageMetadata", {})
        tokens = {"input": usage.get("promptTokenCount", 0), "output": usage.get("candidatesTokenCount", 0),
                  "thinking": usage.get("thoughtsTokenCount", 0)}
        position = positions[project][path.stem]
        rows.append({"model": model, "project": project, "build": path.stem, "position": position,
                     "split": split(position), **tokens, "latency": answer["latency"],
                     "dollars": cost(model, tokens["input"], tokens["output"] + tokens["thinking"])})
    return pd.DataFrame(rows)


def ledger(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, split), g in df.groupby(["model", "split"]):
        out.append({"model": model, "split": split, "answers": len(g), "input median": g["input"].median(),
                    "input mean": g["input"].mean(), "output median": g["output"].median(),
                    "thinking median": g["thinking"].median(), "latency p50": g["latency"].median(),
                    "latency p95": g["latency"].quantile(0.95), "latency p99": g["latency"].quantile(0.99),
                    "dollars per build": g["dollars"].mean(), "dollars per 1000 builds": 1000 * g["dollars"].mean()})
    return pd.DataFrame(out)


def full_hours(projects: list[str]) -> pd.DataFrame:
    """Mean full-suite test hours per evaluated build, per project."""
    out = []
    for project in projects:
        suite = load_suite(project)
        position = suite.builds["position"].to_numpy()
        held = (position >= EVALUATE[0]) & (position < EVALUATE[1])
        hours = np.bincount(suite.build_of_rows(), weights=suite.duration, minlength=len(suite.builds)) / 3600
        out.append({"project": project, "builds": int(held.sum()), "full hours": hours[held].mean()})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="The LLM order's token ledger: tokens, latency and list-price cost "
                                                 "of every saved answer; first-failure recall over budgets on the "
                                                 "held-out builds; the budget TF-IDF+CC needs to match the LLM's "
                                                 "recall, as test time, money and energy, and the suite size at which "
                                                 "the LLM pays for itself")
    parser.add_argument("--builds", required=True, help="the per-build table of the held-out recall run (h8.csv)")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--fusion", choices=FUSIONS, default="risk")
    parser.add_argument("--budget", type=float, default=BUDGET)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--cache", default=None, help="the LLM answer cache (default: FAITHFULCUT_CACHE/llm-v1)")
    parser.add_argument("--out", required=True, help="markdown report; tables and key numbers go beside it")
    args = parser.parse_args()
    cache = Path(args.cache or Path(os.environ["FAITHFULCUT_CACHE"]) / "llm-v1")
    paid = answers(cache)
    costs = ledger(paid)

    h8 = pd.read_csv(args.builds)
    frames, weights = [], []
    for project, g in h8.groupby("project"):
        rows = pd.DataFrame(evaluate(project, g["build"].tolist(), args.model, args.budget, args.fusion,
                                     offline=True)).sort_values("build")
        assert len(rows) == len(g), project
        builds, w = project_builds(project, EVALUATE, args.replicates)
        weights.append(w[:, np.searchsorted(builds, rows["build"].to_numpy())])
        frames.append(rows)
    df, w = pd.concat(frames, ignore_index=True), np.hstack(weights)
    total = w.sum(axis=1)

    def recall(order: str, budget: float) -> np.ndarray:
        return w @ (df[f"{order} depth"].to_numpy() <= budget) / total

    curve = []
    for budget in GRID:
        gap = recall("LLM", budget) - recall(BASELINE, budget)
        curve.append({"budget": budget, **{o: recall(o, budget)[0] for o in ORDERS},
                      f"LLM minus {BASELINE}": fmt(gap), "p": one_sided(gap, above=0.0)})
    curve = pd.DataFrame(curve)

    target = recall("LLM", args.budget)
    matched = budget_for_share(df[f"{BASELINE} depth"].to_numpy(), w, target)
    extra = matched - args.budget
    hours = full_hours(sorted(h8["project"].unique()))
    mean_hours = (hours["builds"] * hours["full hours"]).sum() / hours["builds"].sum()
    held_cost = cost(args.model, df["input_tokens"], df["output_tokens"] + df["thinking_tokens"]).mean()
    per_hour = energy(1.0)
    saved_hours = extra * mean_hours
    value = {
        "builds": len(df), "llm recall": interval(target), "matched budget": interval(matched),
        "extra budget": interval(extra), "mean full hours": mean_hours, "saved test hours": interval(saved_hours),
        "llm dollars per build": held_cost, "llm wh per call": WH_PER_PROMPT,
    }
    for end, cores in (("low", "2-core"), ("high", "8-core")):
        dollars = per_hour[f"dollars_{end}"]
        value[f"saved dollars {cores}"] = interval(saved_hours * dollars)
        value[f"break-even full hours {cores}"] = held_cost / (extra[0] * dollars) if extra[0] > 0 else None
        value[f"saved wh {cores}"] = 1000 * per_hour[f"kwh_{end}"] * saved_hours[0]
    lines = [(f"LLM first-failure recall at {args.budget:.0%}", fmt(target)),
             (f"budget {BASELINE} needs for the same recall", fmt(matched)),
             ("test time the LLM saves at equal recall (share of the suite)", fmt(extra)),
             ("mean full-suite hours per held-out build", f"{mean_hours:.2f}"),
             ("test hours saved per build", fmt(saved_hours)),
             ("LLM dollars per build (list price)", f"{held_cost:.4f}")]
    for end, cores in (("low", "2-core"), ("high", "8-core")):
        even = value[f"break-even full hours {cores}"]
        lines += [(f"runner dollars saved per build, {cores}", fmt(saved_hours * per_hour[f"dollars_{end}"], 4)),
                  (f"full-suite hours at which the LLM pays for itself, {cores}",
                   f"{even:.2f}" if even else "never"),
                  (f"test energy saved per build, {cores} (Wh)", f"{value[f'saved wh {cores}']:.2f}")]
    lines.append(("LLM energy per call, order-of-magnitude anchor (Wh)", f"{WH_PER_PROMPT}"))
    summary = pd.DataFrame(lines, columns=["quantity", "value"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    costs.to_csv(out.with_suffix(".csv"), index=False)
    paid.to_csv(out.with_name(f"{out.stem}-answers.csv"), index=False)
    curve.to_csv(out.with_name(f"{out.stem}-recall.csv"), index=False)
    hours.to_csv(out.with_name(f"{out.stem}-hours.csv"), index=False)
    out.with_suffix(".json").write_text(json.dumps(value, indent=1))
    parts = [table("Every saved answer, by model and split (list price; output includes reasoning)", costs),
             table(f"First-failure recall on the held-out builds ({len(df)} builds, {args.model}, {args.fusion} "
                   f"fusion; {args.replicates} pull-request resamples within projects; one-sided p for > 0)", curve),
             table("The LLM's recall gain as test time, money and energy", summary),
             table("Full-suite hours per held-out build", hours)]
    out.write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
