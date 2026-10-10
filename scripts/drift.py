import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut import lrts
from faithfulcut.control import Monitor, OnlineCut
from faithfulcut.experiments import setup
from faithfulcut.replay import replay


def stage_events(project: str) -> np.ndarray:
    """Builds, in suite order, that run a stage name no earlier build ran."""
    builds = lrts.load_builds(project)
    stages = lrts.load_runs(project).groupby(["pr_name", "build_id"])["stage_id"].apply(set)
    seen, events = set(), np.zeros(len(builds), dtype=bool)
    for t, b in enumerate(builds.itertuples(index=False)):
        current = stages[(b.pr_name, b.build_id)]
        events[t] = t > 0 and bool(current - seen)
        seen |= current
    return events


def drift_rows(project: str, seed: int, scorer: str, alpha: float, pi: float, window: int) -> list[dict]:
    suite, labels, scorers, evaluate = setup(project, seed, (0.2, 0.7))
    events = stage_events(project)[:len(suite.builds)]
    after = np.zeros(len(suite.builds), dtype=bool)
    for t in np.flatnonzero(events):
        after[t:t + window] = True
    rep = replay(suite, "faithful", scorers[scorer], lam=OnlineCut(alpha, 0.2), pi=pi, seed=seed,
                 label=labels["operational"], cap=0.8, monitor=Monitor(alpha))
    of_row = suite.build_of_rows()
    n = len(suite.builds)
    faulty = np.bincount(of_row, weights=labels["operational"], minlength=n) > 0
    caught = np.bincount(of_row, weights=labels["operational"] & rep.selected, minlength=n) > 0
    unseen = np.bincount(of_row, weights=suite.duration * np.isneginf(rep.depth), minlength=n) / rep.full_cost
    rows = []
    for name, mask in (("after events", after & evaluate), ("elsewhere", ~after & evaluate)):
        rows.append({"project": project, "seed": seed, "span": name, "builds": int(mask.sum()),
                     "escape": (faulty & ~caught & mask).sum() / max((faulty & mask).sum(), 1),
                     "alarm_forced": (rep.reason[mask] == 3).mean(), "unseen_share": unseen[mask].mean()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", nargs="*", default=["kafka", "tvm", "hbase"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--scorer", default="LatestFail+CC")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    rows = [r for p in args.projects for s in range(args.seeds)
            for r in drift_rows(p, s, args.scorer, 0.1, 0.1, args.window)]
    df = pd.DataFrame(rows)
    out = Path("data") / "results" / "drift"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "drift.csv", index=False)
    print(df.groupby(["project", "span"])[["builds", "escape", "alarm_forced", "unseen_share"]].mean().round(3))


if __name__ == "__main__":
    main()
