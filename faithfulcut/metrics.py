import heapq

import numpy as np

from faithfulcut.replay import Replay
from faithfulcut.suite import Suite


def escapes(suite: Suite, rep: Replay, label: np.ndarray, builds: np.ndarray) -> tuple[int, int, int]:
    """Faulty builds, base escapes and executed escapes among `builds` for a per-row failure `label`."""
    of_row = suite.build_of_rows()
    n = len(suite.builds)
    faulty = np.bincount(of_row, weights=label, minlength=n) > 0
    caught = np.bincount(of_row, weights=label & rep.selected, minlength=n) > 0
    found = np.bincount(of_row, weights=label & rep.executed(suite), minlength=n) > 0
    base, executed = faulty & ~caught, faulty & ~found
    return int(faulty[builds].sum()), int(base[builds].sum()), int(executed[builds].sum())


def saved_work(rep: Replay, builds: np.ndarray) -> float:
    return 1.0 - rep.executed_cost[builds].sum() / rep.full_cost[builds].sum()


def apfdc(suite: Suite, rep: Replay, label: np.ndarray, builds: np.ndarray) -> np.ndarray:
    """APFDc (one fault per labeled failure) of each build's full order: unseen classes first, then by depth."""
    values = np.full(len(suite.builds), np.nan)
    for t in np.flatnonzero(builds):
        rows = suite.rows(t)
        failed = label[rows]
        if not failed.any():
            continue
        order = np.argsort(rep.depth[rows], kind="stable")
        d, f = suite.duration[rows][order] + 0.001, failed[order]
        tail = np.cumsum(d[::-1])[::-1]
        values[t] = (tail[f] - d[f] / 2).sum() / (f.sum() * d.sum())
    return values


def first_failure_time(suite: Suite, rep: Replay, label: np.ndarray, builds: np.ndarray) -> np.ndarray:
    """Test time until each build's first labeled failure among the classes it ran, unseen classes first and then
    by depth; NaN where no labeled failure ran."""
    executed = rep.executed(suite)
    out = np.full(len(suite.builds), np.nan)
    for t in np.flatnonzero(builds):
        rows = suite.rows(t)
        order = np.argsort(rep.depth[rows], kind="stable")
        ran = executed[rows][order]
        hit = label[rows][order] & ran
        if hit.any():
            out[t] = np.cumsum(np.where(ran, suite.duration[rows][order], 0.0))[np.argmax(hit)]
    return out


def hidden_latency(suite: Suite, rep: Replay, label: np.ndarray, builds: np.ndarray) -> tuple[int, float, float]:
    """Labeled failing rows of `builds` that did not run, and the mean number of builds and of hours until their class
    next runs; rows whose class never runs again are left out of the means."""
    n = len(suite.builds)
    of_row = suite.build_of_rows()
    executed = rep.executed(suite)
    hidden = np.flatnonzero(label & ~executed & builds[of_row])
    runs = np.sort(suite.cls[executed].astype(np.int64) * n + of_row[executed])
    i = np.searchsorted(runs, suite.cls[hidden].astype(np.int64) * n + of_row[hidden], side="right")
    found = i < len(runs)
    found[found] = runs[i[found]] // n == suite.cls[hidden][found]
    later = runs[i[found]] % n
    start = suite.builds["start"].to_numpy()
    hours = (start[later] - start[of_row[hidden][found]]) / 3600
    return len(hidden), float((later - of_row[hidden][found]).mean()) if len(later) else np.nan, \
        float(hours.mean()) if len(later) else np.nan


def escape_causes(suite: Suite, rep: Replay, label: np.ndarray, builds: np.ndarray,
                  observed: np.ndarray) -> dict[str, int]:
    """Escaped faulty builds among `builds`, by why the selection missed them: "new" when none of the failing classes
    had failed in a build finished before the start, "seen" when one had and the replay observed that failure (rows
    in `observed`), "hidden" when such failures exist but went unobserved."""
    start, finish = suite.builds["start"].to_numpy(), suite.builds["finish"].to_numpy()
    failed_before = np.zeros(len(suite.names), dtype=bool)
    seen_before = np.zeros(len(suite.names), dtype=bool)
    out = {"new": 0, "seen": 0, "hidden": 0}
    pending = []
    for t in range(len(start)):
        while pending and pending[0][0] <= start[t]:
            rows = suite.rows(heapq.heappop(pending)[1])
            fail = suite.failed[rows]
            failed_before[suite.cls[rows][fail]] = True
            seen_before[suite.cls[rows][fail & observed[rows]]] = True
        rows = suite.rows(t)
        failing = suite.cls[rows][label[rows]]
        if builds[t] and len(failing) and not (label[rows] & rep.selected[rows]).any():
            out["seen" if seen_before[failing].any() else "hidden" if failed_before[failing].any() else "new"] += 1
        heapq.heappush(pending, (finish[t], t))
    return out
