import dataclasses

import numpy as np
import pandas as pd

from faithfulcut.suite import Suite


def make_suite(rows, start, finish, everywhere=None, changed=None):
    n = len(rows)
    builds = pd.DataFrame({
        "pr_name": [f"PR-{t}" for t in range(n)], "build_id": 1, "trunk_sha": "", "build_head_sha": "",
        "start": start, "finish": finish, "position": np.arange(n) / n,
    })
    cls = np.concatenate([r[0] for r in rows]).astype(np.int32)
    failed = np.concatenate([r[1] for r in rows]).astype(bool)
    changed = changed if changed is not None else [[] for _ in range(n)]
    n_files = max([max(c, default=-1) for c in changed]) + 1
    return Suite(
        "synthetic", np.array([f"C{i}" for i in range(cls.max() + 1)]), builds,
        np.cumsum([0] + [len(r[0]) for r in rows]), cls, failed,
        failed.copy() if everywhere is None else np.asarray(everywhere, dtype=bool),
        np.concatenate([r[2] for r in rows]).astype(float),
        np.array([f"f{i}" for i in range(n_files)], dtype=str), np.cumsum([0] + [len(c) for c in changed]),
        np.array([f for c in changed for f in c], dtype=np.int32),
    )


def random_suite(seed=0, n_builds=80, n_classes=30, spread=40.0, n_files=8):
    rng = np.random.default_rng(seed)
    start = np.cumsum(rng.uniform(1, 10, n_builds))
    finish = start + rng.uniform(0.1, spread, n_builds)
    born = np.where(np.arange(n_classes) < n_classes // 2, 0, rng.integers(0, n_builds, n_classes))
    dies = rng.integers(n_builds // 3, 2 * n_builds, n_classes)
    base = rng.lognormal(0, 1, n_classes)
    rate = rng.uniform(0, 0.4, n_classes)
    rows, changed = [], []
    for t in range(n_builds):
        alive = np.flatnonzero((born <= t) & (t < dies))
        rows.append((alive, rng.random(len(alive)) < rate[alive], base[alive] * rng.uniform(0.5, 1.5, len(alive))))
        changed.append(sorted(rng.choice(n_files, rng.integers(1, 4), replace=False).tolist()))
    return make_suite(rows, start, finish, changed=changed)


def perturb(suite, rows, seed=1, everywhere=True):
    rng = np.random.default_rng(seed)
    failed = suite.failed.copy()
    failed[rows] = ~failed[rows]
    duration = suite.duration.copy()
    duration[rows] *= rng.uniform(0.1, 10, int(rows.sum()))
    return dataclasses.replace(
        suite, failed=failed, failed_everywhere=failed.copy() if everywhere else suite.failed_everywhere,
        duration=duration,
    )
