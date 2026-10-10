import argparse
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from faithfulcut.experiments import control_task, price_task


def price_jobs(cfg: dict) -> list:
    p, evaluate = cfg["price"], tuple(cfg["evaluate"])
    return [(price_task, (project, scorer, regime, pi, seed, p["lambdas"], evaluate, cfg.get("retire_after")))
            for project in cfg["projects"] for scorer in p["scorers"] for regime, pi in p["conditions"]
            for seed in range(p["seeds"])]


def control_jobs(cfg: dict) -> list:
    c, evaluate = cfg["control"], tuple(cfg["evaluate"])
    runs = [(kind, alpha, pi) for kind in c["controllers"] for alpha in c["alphas"] for pi in c["pis"]]
    runs += [(f"fixed-{lam}", 0.0, pi) for lam in c["fixed"] for pi in c["pis"]]
    runs += [("meta", 0.0, c["meta_pi"])]
    c = {**c, "retire_after": cfg.get("retire_after")}
    return [(control_task, (project, scorer, kind, alpha, pi, seed, c, evaluate))
            for project in cfg["projects"] for scorer in c["scorers"] for kind, alpha, pi in runs
            for seed in range(c["seeds"])]


def call(job):
    fn, args = job
    return fn(*args)


def run(jobs: list, workers: int, label: str) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(call, jobs, chunksize=4), 1):
            rows.extend(result)
            if i % 100 == 0 or i == len(jobs):
                print(f"{label}: {i}/{len(jobs)} jobs", flush=True)
    return pd.DataFrame(rows)


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
        if args.only in (None, name):
            run(jobs(cfg), args.workers, name).to_csv(out / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
