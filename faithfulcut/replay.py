import heapq
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from faithfulcut.control import AUDIT, ORACLE, WARMUP, FixedCut, Monitor
from faithfulcut.suite import Suite

REGIMES = ("full", "timed", "faithful")


def audit_coins(seed: int, n_builds: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n_builds)


def tie_breaks(seed: int, t: int, n_classes: int) -> np.ndarray:
    return np.random.default_rng([seed, t]).random(n_classes)


class History:
    def __init__(self, n_classes: int):
        self.known = np.zeros(n_classes, dtype=bool)
        self.retired = np.zeros(n_classes, dtype=bool)
        self.absent = np.zeros(n_classes, dtype=np.int64)
        self.runs = np.zeros(n_classes)
        self.fails = np.zeros(n_classes)
        self.weighted_runs = np.zeros(n_classes)
        self.weighted_fails = np.zeros(n_classes)
        self.since_fail = np.zeros(n_classes)
        self.last_duration = np.zeros(n_classes)
        self.cofails = {}
        self.weighted_cofails = {}

    def catalog(self) -> np.ndarray:
        return np.flatnonzero(self.known & ~self.retired)

    def observe(self, cls: np.ndarray, failed: np.ndarray, duration: np.ndarray, weight: np.ndarray,
                files: np.ndarray):
        self.known[cls] = True
        self.retired[cls] = False
        self.absent[cls] = 0
        self.runs[cls] += 1
        self.fails[cls] += failed
        self.weighted_runs[cls] += weight
        self.weighted_fails[cls] += weight * failed
        self.since_fail[cls] = np.where(failed, 1, self.since_fail[cls] + 1)
        self.last_duration[cls] = duration
        failing = list(zip(cls[failed].tolist(), weight[failed].tolist()))
        for f in files.tolist():
            counts = self.cofails.setdefault(f, {})
            weighted = self.weighted_cofails.setdefault(f, {})
            for c, w in failing:
                counts[c] = counts.get(c, 0.0) + 1.0
                weighted[c] = weighted.get(c, 0.0) + w

    def max_cofail(self, files: np.ndarray, weighted: bool = False) -> np.ndarray:
        table = self.weighted_cofails if weighted else self.cofails
        best = np.zeros(len(self.known))
        for f in files.tolist():
            counts = table.get(f)
            if counts:
                np.maximum.at(best, np.fromiter(counts, np.int64, len(counts)),
                              np.fromiter(counts.values(), float, len(counts)))
        return best

    def miss(self, cls: np.ndarray, retire_after: int):
        self.absent[cls] += 1
        self.retired[cls] |= self.absent[cls] >= retire_after


Scorer = Callable[[History, int], np.ndarray]


@dataclass
class Replay:
    lam: np.ndarray
    audited: np.ndarray
    forced: np.ndarray
    known: np.ndarray
    full_cost: np.ndarray
    base_cost: np.ndarray
    executed_cost: np.ndarray
    seconds: np.ndarray
    # predicted-cost fraction of the ordered catalog up to and including each present class; -inf runs by default
    depth: np.ndarray
    selected: np.ndarray
    history: History
    # why a build ran in full regardless of its cut: 0 it did not, 1 warm-up, 2 size cap, 3 alarm
    reason: np.ndarray
    cut: object

    def executed(self, suite: Suite) -> np.ndarray:
        return self.selected | np.repeat(self.audited | self.forced, np.diff(suite.offsets))


def _feedback(history: History, suite: Suite, regime: str, t: int, selected: np.ndarray, absent: np.ndarray,
              full_run: bool, by_audit: bool, pi: float, retire_after: int):
    rows = suite.rows(t)
    seen = np.ones(len(selected), dtype=bool) if full_run or regime != "faithful" else selected
    if by_audit and regime == "faithful":
        weight = np.where(selected, 1.0, 1.0 / pi)[seen]
    else:
        weight = np.ones(int(seen.sum()))
    history.observe(suite.cls[rows][seen], suite.failed[rows][seen], suite.duration[rows][seen], weight,
                    suite.changed(t))
    history.miss(absent, retire_after)


def replay(suite: Suite, regime: str, scorer: Scorer, lam, pi: float = 0.0, seed: int = 0,
           warmup: float = 0.2, retire_after: int = 3, label: np.ndarray | None = None, feedback: str = "audits",
           cap: float = np.inf, monitor: Monitor | None = None, hold: int = 20) -> Replay:
    """Replay the suite under one information regime.

    `lam` is a fixed cut or a controller; with a per-row `label`, every faulty build seen in full (warm-up or
    coin audit, or every faulty build when feedback is "all") reaches the controller when it finishes.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime}")
    cut = lam if hasattr(lam, "decide") else FixedCut(lam)
    n_builds, n_classes, n_rows = len(suite.builds), len(suite.names), len(suite.cls)
    start = suite.builds["start"].to_numpy()
    finish = suite.builds["finish"].to_numpy()
    warm = suite.builds["position"].to_numpy() < warmup
    audited = audit_coins(seed, n_builds) < pi
    history = History(n_classes)
    rep = Replay(
        lam=np.zeros(n_builds), audited=audited, forced=np.zeros(n_builds, dtype=bool),
        known=np.zeros(n_builds, dtype=np.int64), full_cost=np.zeros(n_builds), base_cost=np.zeros(n_builds),
        executed_cost=np.zeros(n_builds), seconds=np.zeros(n_builds),
        depth=np.zeros(n_rows, dtype=np.float32), selected=np.zeros(n_rows, dtype=bool), history=history,
        reason=np.zeros(n_builds, dtype=np.int8), cut=cut,
    )
    forced = rep.forced
    pending, notes, alarm_until = [], {}, -1

    def deliver(u: int, now: int):
        nonlocal alarm_until
        _feedback(history, suite, regime, u, *notes.pop(u), audited[u] or forced[u], audited[u] and not forced[u],
                  pi, retire_after)
        if label is None:
            return
        rows = suite.rows(u)
        labeled = label[rows]
        kind = WARMUP if warm[u] else AUDIT if audited[u] else ORACLE if feedback == "all" else None
        if kind is None or not labeled.any():
            return
        caught = bool((rep.selected[rows] & labeled).any())
        cut.observe(u, rep.depth[rows][labeled].astype(float), caught, kind)
        if monitor is not None and kind == AUDIT and monitor.update(0.0 if caught else 1.0):
            alarm_until = max(alarm_until, now + hold)

    for t in range(n_builds):
        while pending and pending[0][0] <= start[t]:
            deliver(heapq.heappop(pending)[1], t)
        clock = perf_counter()
        rows = suite.rows(t)
        cls = suite.cls[rows]
        lam_t = cut.decide(t, bool(warm[t]))
        catalog = history.catalog()
        order = catalog[np.lexsort((tie_breaks(seed, t, n_classes)[catalog], scorer(history, t)[catalog]))]
        cumulative = np.cumsum(history.last_duration[order])
        depth_by_class = np.full(n_classes, -np.inf)
        depth_by_class[order] = cumulative / cumulative[-1] if len(order) and cumulative[-1] > 0 else 0.0
        depth = depth_by_class[cls]
        selected = depth <= lam_t
        rep.seconds[t] = perf_counter() - clock

        rep.reason[t] = 1 if warm[t] else 2 if lam_t >= cap else 3 if t < alarm_until else 0
        forced[t] = rep.reason[t] > 0
        duration = suite.duration[rows]
        rep.lam[t] = lam_t
        rep.depth[rows], rep.selected[rows], rep.known[t] = depth, selected, len(catalog)
        rep.full_cost[t], rep.base_cost[t] = duration.sum(), duration[selected].sum()
        full_run = audited[t] or forced[t]
        rep.executed_cost[t] = rep.full_cost[t] if full_run else rep.base_cost[t]

        present = np.zeros(n_classes, dtype=bool)
        present[cls] = True
        checked = catalog if full_run or regime != "faithful" else catalog[depth_by_class[catalog] <= lam_t]
        notes[t] = (selected, checked[~present[checked]])
        if regime == "full":
            deliver(t, t + 1)
        else:
            heapq.heappush(pending, (finish[t], t))
    while pending:
        deliver(heapq.heappop(pending)[1], n_builds)
    return rep
