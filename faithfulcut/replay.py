import heapq
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

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

    def catalog(self) -> np.ndarray:
        return np.flatnonzero(self.known & ~self.retired)

    def observe(self, cls: np.ndarray, failed: np.ndarray, duration: np.ndarray, weight: np.ndarray):
        self.known[cls] = True
        self.retired[cls] = False
        self.absent[cls] = 0
        self.runs[cls] += 1
        self.fails[cls] += failed
        self.weighted_runs[cls] += weight
        self.weighted_fails[cls] += weight * failed
        self.since_fail[cls] = np.where(failed, 1, self.since_fail[cls] + 1)
        self.last_duration[cls] = duration

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
    history.observe(suite.cls[rows][seen], suite.failed[rows][seen], suite.duration[rows][seen], weight)
    history.miss(absent, retire_after)


def replay(suite: Suite, regime: str, scorer: Scorer, lam: float, pi: float = 0.0, seed: int = 0,
           warmup: float = 0.2, retire_after: int = 3) -> Replay:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime}")
    n_builds, n_classes, n_rows = len(suite.builds), len(suite.names), len(suite.cls)
    start = suite.builds["start"].to_numpy()
    finish = suite.builds["finish"].to_numpy()
    forced = suite.builds["position"].to_numpy() < warmup
    audited = audit_coins(seed, n_builds) < pi
    history = History(n_classes)
    rep = Replay(
        lam=np.full(n_builds, lam, dtype=float), audited=audited, forced=forced,
        known=np.zeros(n_builds, dtype=np.int64), full_cost=np.zeros(n_builds), base_cost=np.zeros(n_builds),
        executed_cost=np.zeros(n_builds), seconds=np.zeros(n_builds),
        depth=np.zeros(n_rows, dtype=np.float32), selected=np.zeros(n_rows, dtype=bool), history=history,
    )
    pending, notes = [], {}

    def deliver(u: int):
        _feedback(history, suite, regime, u, *notes.pop(u), audited[u] or forced[u], audited[u] and not forced[u],
                  pi, retire_after)

    for t in range(n_builds):
        while pending and pending[0][0] <= start[t]:
            deliver(heapq.heappop(pending)[1])
        clock = perf_counter()
        rows = suite.rows(t)
        cls = suite.cls[rows]
        catalog = history.catalog()
        order = catalog[np.lexsort((tie_breaks(seed, t, n_classes)[catalog], scorer(history, t)[catalog]))]
        cumulative = np.cumsum(history.last_duration[order])
        depth_by_class = np.full(n_classes, -np.inf)
        depth_by_class[order] = cumulative / cumulative[-1] if len(order) and cumulative[-1] > 0 else 0.0
        depth = depth_by_class[cls]
        selected = depth <= lam
        rep.seconds[t] = perf_counter() - clock

        duration = suite.duration[rows]
        rep.depth[rows], rep.selected[rows], rep.known[t] = depth, selected, len(catalog)
        rep.full_cost[t], rep.base_cost[t] = duration.sum(), duration[selected].sum()
        full_run = audited[t] or forced[t]
        rep.executed_cost[t] = rep.full_cost[t] if full_run else rep.base_cost[t]

        present = np.zeros(n_classes, dtype=bool)
        present[cls] = True
        checked = catalog if full_run or regime != "faithful" else catalog[depth_by_class[catalog] <= lam]
        notes[t] = (selected, checked[~present[checked]])
        if regime == "full":
            deliver(t)
        else:
            heapq.heappush(pending, (finish[t], t))
    while pending:
        deliver(heapq.heappop(pending)[1])
    return rep
