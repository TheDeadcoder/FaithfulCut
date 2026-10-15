import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut import lrts

VERSION = 2
BUILD_COLUMNS = ["pr_name", "build_id", "trunk_sha", "build_head_sha", "start", "finish", "position"]
ROW_ARRAYS = ["offsets", "cls", "failed", "failed_everywhere", "duration", "files", "file_offsets", "file_ids"]


@dataclass
class Suite:
    project: str
    names: np.ndarray
    builds: pd.DataFrame
    offsets: np.ndarray
    cls: np.ndarray
    failed: np.ndarray
    failed_everywhere: np.ndarray
    duration: np.ndarray
    files: np.ndarray
    file_offsets: np.ndarray
    file_ids: np.ndarray

    def rows(self, t: int) -> slice:
        return slice(self.offsets[t], self.offsets[t + 1])

    def changed(self, t: int) -> np.ndarray:
        return self.file_ids[self.file_offsets[t]:self.file_offsets[t + 1]]

    def build_of_rows(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.builds)), np.diff(self.offsets))

    def before(self, position: float) -> "Suite":
        """The builds whose position is below `position`; later builds are never replayed."""
        n = int((self.builds["position"].to_numpy() < position).sum())
        r, f = self.offsets[n], self.file_offsets[n]
        return Suite(self.project, self.names, self.builds.iloc[:n].reset_index(drop=True), self.offsets[:n + 1],
                     self.cls[:r], self.failed[:r], self.failed_everywhere[:r], self.duration[:r], self.files,
                     self.file_offsets[:n + 1], self.file_ids[:f])


def build_suite(project: str) -> Suite:
    builds = lrts.load_builds(project)
    stages = lrts.load_runs(project).groupby(["pr_name", "build_id"])["stage_id"].apply(list)
    index, offsets, parts = {}, [0], []
    file_index, file_offsets, file_parts = {}, [0], []
    for b in builds.itertuples(index=False):
        results = pd.concat([lrts.read_results(project, b.pr_name, b.build_id, s) for s in stages[(b.pr_name, b.build_id)]])
        per_class = results.groupby("testclass").agg(
            failed=("outcome", "max"), everywhere=("outcome", "min"), duration=("duration", "sum"),
        )
        ids = np.array([index.setdefault(n, len(index)) for n in per_class.index], dtype=np.int32)
        parts.append((ids, per_class["failed"].to_numpy() == 1, per_class["everywhere"].to_numpy() == 1,
                      per_class["duration"].to_numpy()))
        offsets.append(offsets[-1] + len(ids))
        changed = lrts.changed_files(project, b.pr_name, b.build_id, b.trunk_sha, b.build_head_sha)
        file_parts.append(np.array([file_index.setdefault(f, len(file_index)) for f in changed], dtype=np.int32))
        file_offsets.append(file_offsets[-1] + len(changed))
    cls, failed, everywhere, duration = (np.concatenate(p) for p in zip(*parts))
    return Suite(project, np.array(list(index)), builds[BUILD_COLUMNS], np.array(offsets), cls, failed, everywhere,
                 duration, np.array(list(file_index), dtype=str), np.array(file_offsets),
                 np.concatenate(file_parts + [np.empty(0, np.int32)]))


def cache_path(project: str) -> Path | None:
    root = os.environ.get("FAITHFULCUT_CACHE")
    return Path(root) / f"suite-v{VERSION}" / f"{project}.npz" if root else None


def load_suite(project: str) -> Suite:
    path = cache_path(project)
    if path is not None and path.exists():
        z = np.load(path)
        builds = pd.DataFrame({c: z[c] for c in BUILD_COLUMNS})
        return Suite(project, z["names"], builds, *(z[a] for a in ROW_ARRAYS))
    suite = build_suite(project)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        builds = {c: suite.builds[c].to_numpy(dtype=str if suite.builds[c].dtype == object else None)
                  for c in BUILD_COLUMNS}
        np.savez_compressed(path, names=suite.names, **builds, **{a: getattr(suite, a) for a in ROW_ARRAYS})
    return suite
