from collections import deque
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from faithfulcut.replay import History, Scorer
from faithfulcut.suite import Suite

EPS = 1e-6
WINDOW = 500
REFIT = 100
NEGATIVES = 0.1
MIN_FAILURES = 10
FEATURES = ("relevance", "relevance share", "co-failures", "failures", "runs", "failure rate", "runs since failure",
            "observations since seen", "log duration", "changed files")


def features(h: History, relevance: np.ndarray, changed: np.ndarray) -> np.ndarray:
    """Each class's features at a build's start, in the order of FEATURES: the change's TF-IDF relevance, raw and as
    a share of the build's highest; co-failures with the changed files; the observed failures, runs, smoothed
    failure rate and runs since the last failure; observations since the class was last seen; its last duration;
    and the number of changed files."""
    top = relevance.max(initial=0.0)
    share = relevance / top if top > 0 else relevance
    return np.column_stack([relevance, share, h.max_cofail(changed), h.fails, h.runs, (h.fails + 1) / (h.runs + 2),
                            h.since_fail, h.observed - h.last_seen, np.log1p(h.last_duration),
                            np.full(len(relevance), len(changed))]).astype(np.float32)


class Learned:
    """A failure model trained on the latest WINDOW observed builds and refit after every REFIT more, ordering the
    catalog by predicted failure probability per predicted second; until it has MIN_FAILURES failures to learn from,
    by change relevance per second. A passing row enters training with probability NEGATIVES and stands for
    1 / NEGATIVES rows. With `weighted`, an observation also counts by its audit propensity weight, as in the -W
    orders. Each replay starts it afresh."""

    def __init__(self, suite: Suite, relevance: Callable[[int], np.ndarray], weighted: bool = False, seed: int = 0):
        self.suite, self.relevance, self.weighted, self.seed = suite, relevance, weighted, seed
        self.history = None

    def _start(self, h: History):
        threadpool_limits(1, user_api="openmp")
        self.history, self.model, self.fresh = h, None, 0
        self.pending, self.window = {}, deque(maxlen=WINDOW)
        h.listeners.append(self._observe)

    def _observe(self, build: int, cls: np.ndarray, failed: np.ndarray, weight: np.ndarray):
        X = self.pending.pop(build, None)
        if X is None:
            return
        keep = failed | (np.random.default_rng([self.seed, build]).random(len(cls)) < NEGATIVES)
        w = np.where(failed, 1.0, 1 / NEGATIVES) * (weight if self.weighted else 1.0)
        self.window.append((X[cls[keep]], failed[keep], w[keep]))
        self.fresh += 1

    def _fit(self):
        if sum(int(y.sum()) for _, y, _ in self.window) < MIN_FAILURES:
            return
        X, y, w = (np.concatenate(part) for part in zip(*self.window))
        if y.all():
            return
        self.model = HistGradientBoostingClassifier(early_stopping=False, random_state=self.seed)
        self.model.fit(X, y, sample_weight=w)
        self.fresh = 0

    def __call__(self, h: History, t: int) -> np.ndarray:
        if h is not self.history:
            self._start(h)
        relevance = self.relevance(t)
        X = features(h, relevance, self.suite.changed(t))
        self.pending[t] = X
        if self.window and (self.model is None or self.fresh >= REFIT):
            self._fit()
        if self.model is None:
            return 1 / (relevance + EPS) * (h.last_duration + EPS)
        p = np.zeros(len(relevance))
        catalog = h.catalog()
        if len(catalog):
            p[catalog] = self.model.predict_proba(X[catalog])[:, 1]
        return 1 / (p + EPS) * (h.last_duration + EPS)


def learned_scorers(suite: Suite, relevance: Callable[[int], np.ndarray], seed: int) -> dict[str, Scorer]:
    """The learned order trained on the observed outcomes as they are, and weighted by audit propensity."""
    return {"Learned+CC": Learned(suite, relevance, seed=seed),
            "Learned-W+CC": Learned(suite, relevance, weighted=True, seed=seed)}
