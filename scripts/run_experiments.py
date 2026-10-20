import argparse
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.experiments import available, control_task, price_task


def price_jobs(cfg: dict) -> list:
    p, evaluate = cfg["price"], tuple(cfg["evaluate"])
    arms = [(regime, pi, 0, 0.0) for regime, pi in p["conditions"]]
    arms += [("faithful", 0.0, k, 0.0) for k in p.get("schedules", [])]
    arms += [("faithful", 0.0, 0, r) for r in p.get("rotations", [])]
    return [(price_task, (project, scorer, regime, pi, seed, p["lambdas"], evaluate, cfg.get("retire_after"),
                          p.get("per_build", False), k, r, p.get("latency", False)))
            for project in cfg["projects"] for scorer in p["scorers"] if available(project, scorer)
            for regime, pi, k, r in arms for seed in range(p["seeds"])]


def control_jobs(cfg: dict) -> list:
    c, evaluate = cfg["control"], tuple(cfg["evaluate"])
    runs = [(kind, alpha, pi) for kind in c["controllers"] for alpha in c["alphas"] for pi in c["pis"]]
    runs += [(f"fixed-{lam}", 0.0, pi) for lam in c["fixed"] for pi in c["pis"]]
    runs += [("meta", 0.0, c["meta_pi"])] if "meta_pi" in c else []
    if c.get("tuned"):
        tuned = pd.read_csv(c["tuned"])
        runs += [("tuned", alpha, pi) for alpha in c["alphas"] for pi in c["pis"]]
        c = {**c, "tuned": {(r.project, r.scorer, r.alpha): r.lam for r in tuned.itertuples()}}
    c = {**c, "retire_after": cfg.get("retire_after")}
    return [(control_task, (project, scorer, kind, alpha, pi, seed, c, evaluate))
            for project in cfg["projects"] for scorer in c["scorers"] if available(project, scorer)
            for kind, alpha, pi in runs for seed in range(c["seeds"])]


def call(job):
    fn, args = job
    return fn(*args)


def run(jobs: list, workers: int, label: str, out: Path):
    """Run the jobs; summaries go to <label>.csv and any per-build outcomes to <label>_builds.npz, keyed by row."""
    rows, parts, labels = [], [], None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(call, jobs, chunksize=4), 1):
            for row in result:
                builds = row.pop("_builds", None)
                if builds is not None:
                    labels = builds.pop("labels")
                    parts.append({**builds, "row": np.full(len(builds["build"]), len(rows), np.int32)})
                rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                print(f"{label}: {i}/{len(jobs)} jobs", flush=True)
    pd.DataFrame(rows).to_csv(out / f"{label}.csv", index=False)
    if parts:
        np.savez_compressed(out / f"{label}_builds.npz", labels=np.array(labels),
                            **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--only", choices=["price", "control"])
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    args = parser.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    out = Path("data") / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    for name, jobs in (("price", price_jobs), ("control", control_jobs)):
        if args.only in (None, name) and name in cfg:
            run(jobs(cfg), args.workers, name, out)


if __name__ == "__main__":
    main()
