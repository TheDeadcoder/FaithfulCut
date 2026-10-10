import shutil
import subprocess
import sys

from faithfulcut.ir import REPOS, repo_dir

ATTEMPTS = 5


def complete(project: str) -> bool:
    target = repo_dir(project)
    check = ["git", "--git-dir", str(target), "rev-parse", "--verify", "-q", "HEAD"]
    return target.exists() and subprocess.run(check, capture_output=True).returncode == 0


def clone(project: str) -> bool:
    target = repo_dir(project)
    for _ in range(ATTEMPTS):
        if complete(project):
            return True
        shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", "http.version=HTTP/1.1", "clone", "-q", "--bare",
                        f"https://github.com/{REPOS[project]}.git", str(target)])
    return complete(project)


def main():
    failed = [p for p in (sys.argv[1:] or REPOS) if not clone(p)]
    print("failed:", " ".join(failed) if failed else "none")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
