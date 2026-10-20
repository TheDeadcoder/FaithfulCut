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

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str, weight: float = 1.0):
        pass


class OnlineCut:
    """lambda += eta * weight * (loss - alpha) on each audited faulty build, no projection and no reset.

    It starts, once the warm-up ends, from the conformal quantile of the warm-up's first-failure depths. The weight
    is 1 under uniform audits and pi / pi_t under adaptive ones.
    """

    def __init__(self, alpha: float, eta: float, start: float | None = None):
        self.alpha, self.eta, self.value = alpha, eta, start
        self.warm, self.losses, self.path = [], [], []

    def decide(self, t: int, warm: bool) -> float:
        if self.value is None and not warm:
            self.value = crc_quantile(self.warm, self.alpha)
            self.path.append(self.value)
        return 1.0 if self.value is None else self.value

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str, weight: float = 1.0):
        if kind == WARMUP:
            if self.value is None:
                self.warm.append(depths.min())
            return
        if self.value is None:
            return
        loss = 0.0 if caught else 1.0
        self.losses.append(loss)
        self.value += self.eta * weight * (loss - self.alpha)
        self.path.append(self.value)


class SplitCut:
    """Conformal (or, with pac_delta, PAC) cut over the last `window` faulty builds seen in full."""

    def __init__(self, alpha: float, window: int, pac_delta: float | None = None):
        self.alpha, self.delta = alpha, pac_delta
        self.first = deque(maxlen=window)
        self.value = 1.0

    def decide(self, t: int, warm: bool) -> float:
        return self.value

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str, weight: float = 1.0):
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

    def observe(self, t: int, depths: np.ndarray, caught: bool, kind: str, weight: float = 1.0):
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


def _rate(caught: float, escapes):
    return escapes / np.maximum(caught + escapes, 1e-12)


def _half_width(caught: float, escapes: float, spread: float) -> float:
    centre = _rate(caught, escapes)
    return float(max(_rate(caught, escapes + spread) - centre, centre - _rate(caught, max(escapes - spread, 0.0))))


def transfer_widths(caught: int, audited_misses: int, pi: float, limit: int, delta: float = 0.05) -> dict[str, float]:
    """The full-stream escape rate over faulty builds, estimated from what a deployment sees, with half-widths.

    Faulty builds are the `caught` ones, which fail visibly, plus S escapes, which only audits reveal, so an interval
    for S maps onto the rate S / (caught + S). The audited-miss martingale sum_k (A_k - pi) L_k has conditional
    variance pi (1 - pi) S: "adaptive" takes every S up to `limit` (the builds whose selection caught nothing) still
    consistent with the audited misses (Freedman), "normal" plugs in the inverse-propensity estimate of S, and "worst"
    is Freedman's width with every faulty build an escape, at the estimated number of faulty builds.
    """
    log_term = math.log(2 / delta)
    b = max(pi, 1 - pi)

    def freedman(s):
        return np.sqrt(2 * pi * (1 - pi) * s * log_term) + 2 * b * log_term / 3

    escapes = audited_misses / pi
    estimate = float(_rate(caught, escapes))
    s = np.arange(limit + 1)
    consistent = s[np.abs(audited_misses - pi * s) <= freedman(s)]
    adaptive = float(np.abs(_rate(caught, consistent) - estimate).max()) if len(consistent) else 1.0
    spread = stats.norm.ppf(1 - delta / 2) * math.sqrt((1 - pi) * audited_misses) / pi
    faulty = caught + escapes
    return {"estimate": estimate, "normal": _half_width(caught, escapes, spread), "adaptive": adaptive,
            "worst": float(freedman(faulty) / (pi * faulty)) if faulty else 1.0}


def ht_widths(faulty: np.ndarray, caught: np.ndarray, audited: np.ndarray, pi: np.ndarray,
              delta: float = 0.05) -> dict[str, float]:
    """As transfer_widths when each build is audited with its own known probability pi, over the evaluated builds:
    the Horvitz-Thompson count of escapes, "normal" from its estimated variance, and "worst" from Freedman's
    inequality with every faulty build an escape, its variance estimated from the audited faulty builds. `faulty` is
    read only where a build is caught or audited. With one pi for all builds the widths equal transfer_widths'."""
    log_term = math.log(2 / delta)
    n_caught = int((faulty & caught).sum())
    seen = audited & faulty & ~caught
    escapes = float((1 / pi[seen]).sum())
    spread = stats.norm.ppf(1 - delta / 2) * math.sqrt(((1 - pi[seen]) / pi[seen] ** 2).sum())
    sampled = audited & faulty
    odds = np.average((1 - pi[sampled]) / pi[sampled], weights=1 / pi[sampled]) if sampled.any() else 0.0
    n_faulty = n_caught + escapes
    bound = max(1.0, 1 / pi.min() - 1)
    worst = (math.sqrt(2 * n_faulty * odds * log_term) + 2 * bound * log_term / 3) / n_faulty if n_faulty else 1.0
    return {"estimate": float(_rate(n_caught, escapes)), "normal": _half_width(n_caught, escapes, spread),
            "worst": float(worst)}
