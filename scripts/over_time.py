import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import (at_matched_work, condition_sums, interval, load_outcomes, project_builds,
                                  replicate_rates)
from faithfulcut.suite import load_suite

WORK = 0.2
REGIMES = ["full", "timed", "faithful"]
TRACKING = ["hadoop", "hive", "jackrabbit-oak", "james", "kafka", "tvm"]


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def fmt(replicates: np.ndarray) -> str:
    i = interval(replicates)
    return f"{i['estimate']:.3f} [{i['low']:.3f}, {i['high']:.3f}]"


def windows(suite, builds: np.ndarray, evaluate: tuple[float, float], k: int) -> np.ndarray:
    """Each evaluated build's window: k equal spans of position."""
    position = suite.builds["position"].to_numpy()[builds]
    return np.minimum(((position - evaluate[0]) / (evaluate[1] - evaluate[0]) * k).astype(int), k - 1)


def per_window(sums: pd.DataFrame, builds: np.ndarray, weights: np.ndarray, window: np.ndarray, k: int,
               numerator: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per window, the replicates of the rate and executed share of test time per lambda."""
    out = []
    for w in range(k):
        inside = window == w
        part = sums[sums["build"].isin(builds[inside])]
        _, rate, executed = replicate_rates(part, builds[inside], weights[:, inside], numerator=numerator)
        out.append((rate, executed))
    return out


def price(rows: pd.DataFrame, outcomes: dict, evaluate: tuple[float, float], k: int, replicates: int,
          label: str) -> dict:
    """Replicates of each order's escape at matched work per window, regime and project, without audits."""
    out = {}
    keep = rows[rows["pi"] == 0.0]
    for project, g in keep.groupby("project"):
        suite = load_suite(project, later=evaluate[1] > 1)
        builds, weights = project_builds(project, evaluate, replicates)
        window = windows(suite, builds, evaluate, k)
        for order, regime in g[["scorer", "regime"]].drop_duplicates().itertuples(index=False):
            sums = condition_sums(rows, outcomes, label, project=project, scorer=order, regime=regime, pi=0.0)
            for w, (rate, executed) in enumerate(per_window(sums, builds, weights, window, k, "executed_escape")):
                out[(w, order, regime, project)] = at_matched_work(rate, executed, WORK)
        print("price", project, flush=True)
    return out


def control(rows: pd.DataFrame, outcomes: dict, evaluate: tuple[float, float], k: int, replicates: int,
            alpha: float, pi: float, projects: list[str]) -> dict:
    """Replicates of each controller's base escape and executed share of test time per window and project."""
    out = {}
    keep = rows[(rows["alpha"] == alpha) & (rows["pi"] == pi) & rows["project"].isin(projects)]
    for project, g in keep.groupby("project"):
        suite = load_suite(project, later=evaluate[1] > 1)
        builds, weights = project_builds(project, evaluate, replicates)
        window = windows(suite, builds, evaluate, k)
        for order, kind in g[["scorer", "controller"]].drop_duplicates().itertuples(index=False):
            sums = condition_sums(rows, outcomes, "operational", project=project, scorer=order, controller=kind,
                                  alpha=alpha, pi=pi)
            for w, (rate, executed) in enumerate(per_window(sums, builds, weights, window, k, "escaped")):
                out[(w, order, kind, project)] = (rate[0], executed[0])
        print("control", project, flush=True)
    return out


def churn(project: str, evaluate: tuple[float, float], k: int) -> pd.DataFrame:
    """Per window, label-free change in the test catalog and the code: builds running a class never run before,
    the share of test time on such classes, classes run for the first time, and builds changing a file in a directory
    no earlier change touched; with the catalog's size at the window's end."""
    suite = load_suite(project, later=evaluate[1] > 1)
    n = len(suite.builds)
    of_row = suite.build_of_rows()
    first_row = np.zeros(len(suite.cls), dtype=bool)
    _, first = np.unique(suite.cls, return_index=True)
    first_row[first] = True
    new_classes = np.bincount(of_row, weights=first_row, minlength=n)
    new_time = np.bincount(of_row, weights=suite.duration * first_row, minlength=n)
    time = np.bincount(of_row, weights=suite.duration, minlength=n)
    seen_dirs, new_dir = set(), np.zeros(n, dtype=bool)
    for t in range(n):
        dirs = {path.rsplit("/", 1)[0] if "/" in path else "" for path in suite.files[suite.changed(t)]}
        new_dir[t] = bool(dirs - seen_dirs) and t > 0
        seen_dirs |= dirs
    position = suite.builds["position"].to_numpy()
    builds = np.flatnonzero((position >= evaluate[0]) & (position < evaluate[1]))
    window = windows(suite, builds, evaluate, k)
    out = []
    for w in range(k):
        b = builds[window == w]
        if not len(b):
            continue
        out.append({"project": project, "window": w, "builds": len(b),
                    "builds with a new class": float((new_classes[b] > 0).mean()),
                    "time on new classes": new_time[b].sum() / max(time[b].sum(), 1e-9),
                    "new classes": int(new_classes[b].sum()), "builds with a new directory": float(new_dir[b].mean()),
                    "catalog": int(new_classes[:b.max() + 1].sum())})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="Evolution over equal windows of one evaluated span: each order's "
                                                 "price of information, how the online controller and the dev-tuned "
                                                 "budget track alpha without retuning, and label-free churn of the "
                                                 "test catalog and the code")
    parser.add_argument("--price", nargs="+", required=True, help="price results with per-build outcomes")
    parser.add_argument("--control", nargs="+", required=True, help="control results with per-build outcomes")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--windows", type=int, required=True)
    parser.add_argument("--label", default="operational")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--pi", type=float, default=0.1)
    parser.add_argument("--projects", nargs="+", default=TRACKING, help="projects for the tracking comparison")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True, help="markdown report; its tables go beside it as csv")
    args = parser.parse_args()
    evaluate, k = tuple(args.evaluate), args.windows

    rows, outcomes = load_outcomes(args.price, "price")
    prices = price(rows, outcomes, evaluate, k, args.replicates, args.label)
    del rows, outcomes
    ratios = []
    for w in range(k):
        for order in sorted({key[1] for key in prices}):
            projects = sorted({key[3] for key in prices if key[:2] == (w, order)})
            among = [p for p in projects if all(np.isfinite(prices[(w, order, r, p)][0]) for r in REGIMES)]
            if not among:
                continue
            macro = {r: np.nanmean([prices[(w, order, r, p)] for p in among], axis=0) for r in REGIMES}
            ratios.append({"window": w, "order": order, "projects": len(among),
                           **{f"escape {r}": macro[r][0] for r in REGIMES},
                           "ratio": fmt(macro["faithful"] / macro["full"]),
                           "timing": fmt(macro["timed"] / macro["full"])})
    ratios = pd.DataFrame(ratios)

    rows, outcomes = load_outcomes(args.control, "control")
    controls = control(rows, outcomes, evaluate, k, args.replicates, args.alpha, args.pi, args.projects)
    del rows, outcomes
    tracking = []
    for w in range(k):
        for order in sorted({key[1] for key in controls}):
            among = [p for p in args.projects if all((w, order, kind, p) in controls and
                                                     np.isfinite(controls[(w, order, kind, p)][0][0])
                                                     for kind in ("online", "tuned"))]
            if not among:
                continue
            gap = {kind: np.mean([np.abs(controls[(w, order, kind, p)][0] - args.alpha) for p in among], axis=0)
                   for kind in ("online", "tuned")}
            tracking.append({"window": w, "order": order, "projects": len(among),
                             **{f"escape {kind}": np.mean([controls[(w, order, kind, p)][0][0] for p in among])
                                for kind in ("online", "tuned")},
                             **{f"work {kind}": np.mean([controls[(w, order, kind, p)][1][0] for p in among])
                                for kind in ("online", "tuned")},
                             "gap online": gap["online"][0], "gap tuned": gap["tuned"][0],
                             "online minus tuned": fmt(gap["online"] - gap["tuned"])})
    tracking = pd.DataFrame(tracking)

    changes = pd.concat([churn(p, evaluate, k) for p in sorted({key[3] for key in prices})], ignore_index=True)
    pooled = changes.assign(**{c: changes[c] * changes["builds"] for c in ("builds with a new class",
                                                                          "builds with a new directory")})
    pooled = pooled.groupby("window")[["builds", "builds with a new class", "builds with a new directory",
                                       "new classes"]].sum()
    for c in ("builds with a new class", "builds with a new directory"):
        pooled[c] = pooled[c] / pooled["builds"]
    pooled = pooled.reset_index()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ratios.to_csv(out.with_name(f"{out.stem}-price.csv"), index=False)
    tracking.to_csv(out.with_name(f"{out.stem}-tracking.csv"), index=False)
    changes.to_csv(out.with_name(f"{out.stem}-churn.csv"), index=False)
    span = (evaluate[1] - evaluate[0]) / k
    bounds = ", ".join(f"{w}: [{evaluate[0] + w * span:.3g}, {evaluate[0] + (w + 1) * span:.3g})" for w in range(k))
    parts = [table(f"Price of information per window ({args.label}, {WORK:.0%} executed work, macro over projects with "
                   f"all three regimes; FAITHFUL/FULL and TIMED/FULL with {args.replicates} pull-request resamples). "
                   f"Windows by position: {bounds}", ratios),
             table(f"Tracking per window (alpha {args.alpha}, pi {args.pi}; base escape and executed work, means over "
                   f"projects; mean |escape - alpha|, online minus the dev-tuned budget)", tracking),
             table("Label-free churn per window, pooled over projects", pooled),
             table("Label-free churn per window and project", changes)]
    out.write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
