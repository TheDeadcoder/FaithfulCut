import sys

from faithfulcut.ir import REPOS, load_ir
from faithfulcut.suite import load_suite


def main():
    for project in sys.argv[1:] or REPOS:
        suite = load_suite(project)
        for corpus in ("seen", "report"):
            ir = load_ir(suite, corpus)
            print(project, corpus, "builds without their base commit:", int(ir["missing"].sum()), flush=True)


if __name__ == "__main__":
    main()
