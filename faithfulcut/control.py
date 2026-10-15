import math
from collections import deque

import numpy as np
from scipy import stats

# Kinds of feedback a controller receives for a faulty build seen in full, when that build finishes.
WARMUP, AUDIT, ORACLE = "warmup", "audit", "oracle"


def crc_quantile(first_depths, alpha: float) -> float:
    """Smallest cut whose conformal risk bound (misses + 1) / (n + 1) is at most alpha."""
    d = np.sort(np.asarray(first_depths, dtype=float))
    n = len(d)
    allowed = math.floor((n + 1) * alpha - 1 + 1e-9)
    if allowed < 0:
        return 1.0
    return -np.inf if allowed >= n else float(d[n - allowed - 1])


def pac_quantile(first_depths, alpha: float, delta: float) -> float:
    """Smallest cut whose misses pass the binomial tail test P(Bin(n, alpha) <= misses) <= delta."""
    d = np.sort(np.asarray(first_depths, dtype=float))
    n = len(d)
    passing = np.flatnonzero(stats.binom.cdf(np.arange(n + 1), n, alpha) <= delta)
    if not len(passing):
        return 1.0
    allowed = int(passing.max())
    return -np.inf if allowed >= n else float(d[n - allowed - 1])


class FixedCut:
    def __init__(self, lam: float):
        self.value = lam

    def decide(self, t: int, warm: bool) -> float:
        return self.value

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str):
        pass


class OnlineCut:
    """lambda += eta * (loss - alpha) on each audited faulty build, no projection and no reset.

    It starts, once the warm-up ends, from the conformal quantile of the warm-up's first-failure depths.
    """

    def __init__(self, alpha: float, eta: float, start: float | None = None):
        self.alpha, self.eta, self.value = alpha, eta, start
        self.warm, self.losses, self.path = [], [], []

    def decide(self, t: int, warm: bool) -> float:
        if self.value is None and not warm:
            self.value = crc_quantile(self.warm, self.alpha)
            self.path.append(self.value)
        return 1.0 if self.value is None else self.value

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str):
        if kind == WARMUP:
            if self.value is None:
                self.warm.append(depths.min())
            return
        if self.value is None:
            return
        loss = 0.0 if caught else 1.0
        self.losses.append(loss)
        self.value += self.eta * (loss - self.alpha)
        self.path.append(self.value)


class SplitCut:
    """Conformal (or, with pac_delta, PAC) cut over the last `window` faulty builds seen in full."""

    def __init__(self, alpha: float, window: int, pac_delta: float | None = None):
        self.alpha, self.delta = alpha, pac_delta
        self.first = deque(maxlen=window)
        self.value = 1.0

    def decide(self, t: int, warm: bool) -> float:
        return self.value

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str):
        self.first.append(depths.min())
        if self.delta is None:
            self.value = crc_quantile(self.first, self.alpha)
        else:
            self.value = pac_quantile(self.first, self.alpha, self.delta)


class MetaCut:
    """Every `period` faulty builds seen in full, recalibrate to the test and change recall targets."""

    def __init__(self, test_recall: float = 0.95, change_recall: float = 0.999, period: int = 20,
                 window: int | None = None):
        self.test_recall, self.change_recall, self.period = test_recall, change_recall, period
        self.seen = deque(maxlen=window)
        self.since = 0
        self.calibrated = False
        self.value = 1.0

    def recalibrate(self):
        rows = np.sort(np.concatenate(list(self.seen)))
        first = np.sort([d.min() for d in self.seen])
        by_test = rows[max(math.ceil(self.test_recall * len(rows)) - 1, 0)]
        by_change = first[max(math.ceil(self.change_recall * len(first)) - 1, 0)]
        self.value = float(max(by_test, by_change))
        self.calibrated, self.since = True, 0

    def decide(self, t: int, warm: bool) -> float:
        if not warm and not self.calibrated and self.seen:
            self.recalibrate()
        return self.value

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str):
        self.seen.append(depths)
        if kind != WARMUP:
            self.since += 1
            if self.since >= self.period:
                self.recalibrate()


class Monitor:
    """Alarm when a time-uniform lower confidence bound on the audited escape rate exceeds alpha.

    One-sided normal-mixture boundary for Bernoulli losses (variance proxy 1/4); it restarts after each alarm.
    """

    def __init__(self, alpha: float, delta: float = 0.05, rho: float = 5.0):
        self.alpha, self.delta, self.rho = alpha, delta, rho
        self.n, self.misses = 0, 0.0

    def lower(self) -> float:
        v = self.n / 4 + self.rho
        return (self.misses - math.sqrt(v * math.log(v / (self.rho * self.delta ** 2)))) / self.n

    def update(self, loss: float) -> bool:
        self.n += 1
        self.misses += loss
        if self.lower() > self.alpha:
            self.n, self.misses = 0, 0.0
            return True
        return False


def online_slack(eta: float, updates: int, pending: int) -> float:
    """Long-run bound on |mean audited loss - alpha| for the online cut started in [0, 1]."""
    return (1 + eta) / (eta * updates) + 2 * pending / updates


def max_pending(starts: np.ndarray, finishes: np.ndarray) -> int:
    """Most builds decided but not yet finished at any one time."""
    events = sorted([(s, 1) for s in starts] + [(f, -1) for f in finishes], key=lambda e: (e[0], e[1]))
    level = peak = 0
    for _, step in events:
        level += step
        peak = max(peak, level)
    return peak


def transfer_widths(n_faulty: int, audited_misses: int, pi: float, delta: float = 0.05) -> dict[str, float]:
    """Half-widths for the full-stream escape rate around the inverse-propensity estimate from audits.

    The audited-miss martingale sum_k (A_k - pi) L_k has conditional variance pi (1 - pi) S, where S is
    the unknown number of escapes; "worst" bounds S by n_faulty, "adaptive" takes the largest S still
    consistent with the audited misses (Freedman), and "normal" plugs in the estimate.
    """
    log_term = math.log(2 / delta)
    b = max(pi, 1 - pi)

    def freedman(s):
        return math.sqrt(2 * pi * (1 - pi) * s * log_term) + 2 * b * log_term / 3

    estimate = audited_misses / (pi * n_faulty)
    consistent = [s for s in range(n_faulty + 1) if abs(audited_misses - pi * s) <= freedman(s)]
    adaptive = max(abs(s / n_faulty - estimate) for s in consistent) if consistent else 1.0
    z = stats.norm.ppf(1 - delta / 2)
    return {
        "normal": z * math.sqrt(pi * (1 - pi) * audited_misses / pi) / (pi * n_faulty),
        "adaptive": adaptive,
        "worst": freedman(n_faulty) / (pi * n_faulty),
    }
