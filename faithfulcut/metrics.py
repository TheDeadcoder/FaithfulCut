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
