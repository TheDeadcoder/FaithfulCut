import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from faithfulcut.ir import REPOS
from faithfulcut.structure import UNKNOWN, load_structure
from faithfulcut.suite import load_suite


def build(project: str, later: bool = False) -> str:
    structure = load_structure(load_suite(project, later))
    missing = int((structure["path"] == UNKNOWN).all(axis=1).sum())
    return f"{project}: {len(structure['path'])} builds, {missing} without structure"


def main():
    later = "--later" in sys.argv
    projects = [a for a in sys.argv[1:] if a != "--later"] or list(REPOS)
    with ProcessPoolExecutor(max_workers=len(projects)) as pool:
        for line in pool.map(build, projects, [later] * len(projects)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
