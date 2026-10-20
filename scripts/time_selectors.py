import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from faithfulcut import lrts
from faithfulcut.dante import PROJECTS as DANTE_PROJECTS
from faithfulcut.dante import _class_stem, _Relevance
from faithfulcut.ir import REPOS, _corpora, _documents, _Repo, parse_diff, tfidf
from faithfulcut.suite import load_suite

WARM, MEASURED = 20, 100


def main():
    """Seconds per build to compute each change-based score with warm caches, as a CI service would, on consecutive
    builds from the start of the dev split."""
    rows = []
    for project in sys.argv[1:] or list(REPOS):
        suite = load_suite(project)
        first = int(np.searchsorted(suite.builds["position"].to_numpy(), 0.2))
        repo = _Repo(project)
        rel = _Relevance(project, {_class_stem(n) for n in suite.names}) if project in DANTE_PROJECTS else None
        try:
            for t, classes in _corpora(suite, "seen"):
                if t < first:
                    continue
                if t >= first + WARM + MEASURED:
                    break
                b = suite.builds.iloc[t]
                diff = lrts.diff_path(project, b["pr_name"], b["build_id"]).read_text(encoding="utf-8",
                                                                                     errors="ignore")
                names = suite.names[classes].tolist()
                clock = perf_counter()
                files = repo.tree(b["trunk_sha"])
                if files is not None:
                    tfidf(_documents(repo, names, files, parse_diff(diff)), repo.counts(diff), len(repo.terms))
                row = {"project": project, "build": t, "classes": len(names),
                       "tfidf_seconds": perf_counter() - clock}
                if rel is not None:
                    clock = perf_counter()
                    files = rel.repo.tree(b["trunk_sha"])
                    if files is not None:
                        rel.build(b["trunk_sha"], files, diff, names)
                    row["dante_seconds"] = perf_counter() - clock
                if t >= first + WARM:
                    rows.append(row)
        finally:
            repo.close()
            if rel is not None:
                rel.close()
        print(project, flush=True)
    df = pd.DataFrame(rows)
    out = Path("data") / "results" / "overhead"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "selector_seconds.csv", index=False)
    cols = [c for c in ("tfidf_seconds", "dante_seconds") if c in df]
    print(df.groupby("project")[cols].describe(percentiles=[0.5, 0.95]).loc[:, (slice(None), ["50%", "95%", "max"])]
          .round(3).to_string())


if __name__ == "__main__":
    main()
