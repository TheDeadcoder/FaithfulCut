import argparse
from concurrent.futures import ProcessPoolExecutor

from faithfulcut.ir import REPOS
from faithfulcut.labels import load_operational, transfer
from faithfulcut.suite import load_suite


def build(project: str, seeds: int) -> str:
    suite = load_suite(project)
    transfer(suite)
    for seed in range(seeds):
        load_operational(suite, seed)
    return project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", nargs="*", default=list(REPOS))
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=len(args.projects)) as pool:
        for project in pool.map(build, args.projects, [args.seeds] * len(args.projects)):
            print("labels ready:", project, flush=True)


if __name__ == "__main__":
    main()
