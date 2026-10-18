import subprocess
import sys

import numpy as np

from faithfulcut.ir import REPOS, is_test_source, lrts_pattern, repo_dir
from faithfulcut.suite import load_suite


def main():
    """Test sources renamed between the base commits of each project's first and last studied builds, and how many of
    the renamed files belong to classes the suite ran under both names."""
    for project in sys.argv[1:] or list(REPOS):
        suite = load_suite(project)
        order = np.argsort(suite.builds["position"].to_numpy())
        first, last = suite.builds["trunk_sha"].iloc[order[0]], suite.builds["trunk_sha"].iloc[order[-1]]
        git = ["git", "--git-dir", str(repo_dir(project))]
        commits = int(subprocess.run(git + ["rev-list", "--count", f"{first}..{last}"], capture_output=True,
                                     text=True).stdout.strip() or 0)
        log = subprocess.run(git + ["log", "--diff-filter=R", "-M", "--name-status", "--format=", f"{first}..{last}"],
                             capture_output=True, text=True).stdout
        renames = [line.split("\t")[1:3] for line in log.splitlines() if line.startswith("R")]
        tests = [(old, new) for old, new in renames if is_test_source(old) or is_test_source(new)]
        stems = {lrts_pattern(project, n).rsplit("/", 1)[-1].split(".")[0] for n in suite.names}
        both = sum(1 for old, new in tests if all(p.rsplit("/", 1)[-1].split(".")[0] in stems for p in (old, new))
                   and old.rsplit("/", 1)[-1] != new.rsplit("/", 1)[-1])
        print(f"{project}: {commits} commits, {len(renames)} renames, {len(tests)} of test sources, {both} renamed "
              f"classes the suite ran under both names", flush=True)


if __name__ == "__main__":
    main()
