import argparse
from concurrent.futures import ProcessPoolExecutor

from faithfulcut import dante
from faithfulcut.ir import REPOS, load_ir
from faithfulcut.labels import load_operational, transfer
from faithfulcut.suite import load_suite


def build(project: str, seeds: int) -> str:
    """Suite, labels and change scores for the timeline that continues past the studied window; reports only sizes."""
    suite = load_suite(project, later=True)
    transfer(suite)
    load_ir(suite, "seen")
    if project in dante.PROJECTS:
        dante.load_relevance(suite)
    for seed in range(seeds):
        load_operational(suite, seed)
    return f"{project}: {int((suite.builds['position'] >= 1).sum())} later builds, {len(suite.names)} classes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", nargs="*", default=[p for p in REPOS if p != "log4j"])
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=len(args.projects)) as pool:
        for line in pool.map(build, args.projects, [args.seeds] * len(args.projects)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
