import numpy as np

from faithfulcut.replay import Replay
from faithfulcut.suite import Suite


def escapes(suite: Suite, rep: Replay, label: np.ndarray, builds: np.ndarray) -> tuple[int, int, int]:
    """Faulty builds, base escapes and executed escapes among `builds` for a per-row failure `label`."""
    of_row = suite.build_of_rows()
    n = len(suite.builds)
    faulty = np.bincount(of_row, weights=label, minlength=n) > 0
    caught = np.bincount(of_row, weights=label & rep.selected, minlength=n) > 0
    base = faulty & ~caught
    executed = base & ~(rep.audited | rep.forced)
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
