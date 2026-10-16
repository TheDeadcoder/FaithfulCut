import sys
from concurrent.futures import ProcessPoolExecutor

from faithfulcut.dante import PROJECTS, load_relevance
from faithfulcut.suite import load_suite


def build(project: str) -> str:
    rel = load_relevance(load_suite(project))
    return f"{project}: {len(rel['offsets']) - 1} builds, {int(rel['missing'].sum())} without their base commit"


def main():
    projects = sys.argv[1:] or list(PROJECTS)
    with ProcessPoolExecutor(max_workers=len(projects)) as pool:
        for line in pool.map(build, projects):
            print(line, flush=True)


if __name__ == "__main__":
    main()
