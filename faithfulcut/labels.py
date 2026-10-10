import os
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut import lrts_ref
from faithfulcut.replay import audit_coins
from faithfulcut.suite import Suite

STREAM_RATE = 0.05
VERSION = 1
TRANSFER_STAGE_RULE = {"tvm": "ran"}


def flaky_issues_path() -> Path:
    return Path(os.environ["FAITHFULCUT_CACHE"]) / "flaky_issues.csv"


def flaky_windows(project: str) -> pd.DataFrame:
    df = lrts_ref.flaky_tests()
    df = df[df["project"] == project]
    created = pd.read_csv(flaky_issues_path()).set_index("url")["created"]
    out = pd.DataFrame({
        "testclass": [lrts_ref.jira_class(t) for t in df["flaky_test"]],
        "created": df["url"].map(created).fillna(-np.inf).to_numpy(),
        "fixed": df["fix_timestamp"].to_numpy(),
    })
    return out.dropna(subset=["testclass"]).reset_index(drop=True)


def _count(suite: Suite, t: int, runs: np.ndarray, fails: np.ndarray, sign: int):
    rows = suite.rows(t)
    runs[suite.cls[rows]] += sign
    fails[suite.cls[rows]] += sign * suite.failed[rows]


def _warmup_threshold(suite: Suite, warm: np.ndarray, min_runs: int) -> float:
    runs, fails = np.zeros(len(suite.names)), np.zeros(len(suite.names))
    for t in np.flatnonzero(warm):
        _count(suite, t, runs, fails, 1)
    failing = (runs >= min_runs) & (fails > 0)
    rate = fails[failing] / runs[failing]
    return rate.mean() + 3 * rate.std(ddof=1) if len(rate) > 1 else np.inf


def operational(suite: Suite, seed: int, issues: pd.DataFrame | None = None, window: int = 100,
                min_runs: int = 5, warmup: float = 0.2) -> np.ndarray:
    """Per-row operational failures for one audit seed; every policy run with that seed shares them."""
    issues = flaky_windows(suite.project) if issues is None else issues
    n_builds, n_classes = len(suite.builds), len(suite.names)
    start = suite.builds["start"].to_numpy()
    finish = suite.builds["finish"].to_numpy()
    warm = suite.builds["position"].to_numpy() < warmup
    stream = np.flatnonzero(warm | (audit_coins(seed, n_builds) < STREAM_RATE))
    stream = stream[np.argsort(finish[stream], kind="stable")]
    created, fixed = issues["created"].to_numpy(), issues["fixed"].to_numpy()
    matches = [np.flatnonzero([n.endswith(c) for n in suite.names]) for c in issues["testclass"]]
    warm_end = start[np.argmin(warm)] if not warm.all() else np.inf
    tau = _warmup_threshold(suite, warm & (finish <= warm_end), min_runs)
    runs, fails = np.zeros(n_classes), np.zeros(n_classes)
    label = suite.failed_everywhere.copy()
    added = 0
    for t in range(n_builds):
        while added < len(stream) and finish[stream[added]] <= start[t]:
            _count(suite, stream[added], runs, fails, 1)
            if added >= window:
                _count(suite, stream[added - window], runs, fails, -1)
            added += 1
        flaky = np.zeros(n_classes, dtype=bool)
        for k in np.flatnonzero((created <= start[t]) & (start[t] < fixed)):
            flaky[matches[k]] = True
        frequent = (runs >= min_runs) & (fails > tau * runs)
        rows = suite.rows(t)
        label[rows] &= ~flaky[suite.cls[rows]] & ~frequent[suite.cls[rows]]
    return label


def load_operational(suite: Suite, seed: int) -> np.ndarray:
    """`operational` for the default parameters, cached per project and seed."""
    root = os.environ.get("FAITHFULCUT_CACHE")
    path = Path(root) / f"operational-v{VERSION}" / f"{suite.project}-{seed}.npy" if root else None
    if path is not None and path.exists():
        return np.load(path)
    label = operational(suite, seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.npy")
        np.save(partial, label)
        os.replace(partial, path)
    return label


def _transfer_path(project: str) -> Path | None:
    root = os.environ.get("FAITHFULCUT_CACHE")
    return Path(root) / f"transfer-v{VERSION}" / f"{project}.npz" if root else None


def _transfer_flags(suite: Suite) -> dict[str, np.ndarray]:
    path = _transfer_path(suite.project)
    if path is not None and path.exists():
        return dict(np.load(path))
    failures = lrts_ref.failure_masks(suite.project)
    for label, masks in lrts_ref.FILTERS.items():
        failures[label] = ~failures[list(masks)].any(axis=1)
        failures[f"{label}_nostage"] = ~failures[[m for m in masks if m != "unique"]].any(axis=1)
    flags = [c for c in failures.columns if c.startswith(lrts_ref.LABELS)]
    per_class = failures.groupby(["pr_name", "build_id", "testclass"])[flags].any().reset_index()
    build_index = {(b.pr_name, b.build_id): t for t, b in enumerate(suite.builds.itertuples(index=False))}
    class_index = {n: c for c, n in enumerate(suite.names)}
    n_classes = len(suite.names)
    row_keys = suite.build_of_rows().astype(np.int64) * n_classes + suite.cls
    order = np.argsort(row_keys, kind="stable")
    keys = np.array([build_index[(p, b)] * n_classes + class_index[c]
                     for p, b, c in per_class[["pr_name", "build_id", "testclass"]].itertuples(index=False)],
                    dtype=np.int64)
    rows = order[np.searchsorted(row_keys[order], keys)]
    out = {}
    for flag in flags:
        out[flag] = np.zeros(len(suite.cls), dtype=bool)
        out[flag][rows[per_class[flag].to_numpy()]] = True
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
    return out


def transfer(suite: Suite, stage_rule: str | None = None) -> dict[str, np.ndarray]:
    """Per-row DeConf and FirstFail failures; a class carries a label if any of its stage runs does.

    stage_rule "build" keeps LRTS's rule (failed in every stage of the build); "ran" requires failing in every
    stage where the class ran. The default is the project's entry in TRANSFER_STAGE_RULE, else "build".
    """
    stage_rule = stage_rule or TRANSFER_STAGE_RULE.get(suite.project, "build")
    flags = _transfer_flags(suite)
    if stage_rule == "build":
        return {label: flags[label] for label in lrts_ref.LABELS}
    if stage_rule == "ran":
        return {label: flags[f"{label}_nostage"] & suite.failed_everywhere for label in lrts_ref.LABELS}
    raise ValueError(f"unknown stage rule {stage_rule}")
