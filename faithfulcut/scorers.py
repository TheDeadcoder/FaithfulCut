from typing import Callable

import numpy as np

from faithfulcut.replay import History, Scorer
from faithfulcut.suite import Suite

EPS = 1e-6
SCALE = 1e9


def qtf(history: History, t: int) -> np.ndarray:
    return history.last_duration


def history_scorers(suite: Suite) -> dict[str, Scorer]:
    """LRTS's history orders, each also cost-cognizant (+CC); -W variants weight observations by audit propensity."""

    def latestfail(h: History, t: int) -> np.ndarray:
        return h.since_fail + EPS

    def failcount(h: History, t: int) -> np.ndarray:
        return 1 / (h.fails + EPS)

    def failcount_w(h: History, t: int) -> np.ndarray:
        return 1 / (h.weighted_fails + EPS)

    def tf_failfreq(h: History, t: int) -> np.ndarray:
        return 1 / (h.max_cofail(suite.changed(t)) + EPS)

    def tf_failfreq_w(h: History, t: int) -> np.ndarray:
        return 1 / (h.max_cofail(suite.changed(t), weighted=True) + EPS)

    def cost_cognizant(base: Scorer) -> Scorer:
        return lambda h, t: base(h, t) * (h.last_duration + EPS)

    base = {"LatestFail": latestfail, "FailCount": failcount, "FailCount-W": failcount_w,
            "TF-FailFreq": tf_failfreq, "TF-FailFreq-W": tf_failfreq_w}
    return {"QTF": qtf, **base, **{f"{name}+CC": cost_cognizant(f) for name, f in base.items()}}


def ir_scorers(suite: Suite, ir: dict[str, np.ndarray]) -> dict[str, Scorer]:
    """LRTS's TF-IDF relevance of each class's test sources to the build's diff, plain and cost-cognizant."""

    def tfidf(h: History, t: int) -> np.ndarray:
        scores = np.zeros(len(suite.names))
        part = slice(ir["offsets"][t], ir["offsets"][t + 1])
        scores[ir["cls"][part]] = ir["score"][part]
        return 1 / (scores + EPS)

    return {"TF-IDF": tfidf, "TF-IDF+CC": lambda h, t: tfidf(h, t) * (h.last_duration + EPS)}


def depth_split(first: Scorer, second: Scorer) -> Scorer:
    """Each class's smaller cumulative predicted cost in the two orders, so a cut at depth d runs the union of both
    orders' prefixes up to d: the budget is shared equally between them."""

    def score(h: History, t: int) -> np.ndarray:
        catalog = h.catalog()
        cost = h.last_duration[catalog]
        best = np.full(len(catalog), np.inf)
        for order in (np.argsort(s(h, t)[catalog], kind="stable") for s in (first, second)):
            depth = np.empty(len(catalog))
            depth[order] = np.cumsum(cost[order])
            best = np.minimum(best, depth)
        out = np.full(len(h.known), np.inf)
        out[catalog] = best
        return out

    return score


def rank_fusion(first: Scorer, second: Scorer, k: int = 60) -> Scorer:
    """Reciprocal rank fusion of two orders."""

    def score(h: History, t: int) -> np.ndarray:
        catalog = h.catalog()
        fused = np.zeros(len(catalog))
        for order in (np.argsort(s(h, t)[catalog], kind="stable") for s in (first, second)):
            ranks = np.empty(len(catalog))
            ranks[order] = np.arange(1, len(catalog) + 1)
            fused += 1 / (k + ranks)
        out = np.zeros(len(h.known))
        out[catalog] = -fused
        return out

    return score


def hybrid_scorers(history: Scorer, content: Scorer) -> dict[str, Scorer]:
    """History and change relevance combined: an equal budget split, and reciprocal rank fusion."""
    return {"Hybrid": depth_split(history, content), "Hybrid-RRF": rank_fusion(history, content)}


def structure_scorers(structure: Callable[[], dict[str, np.ndarray]], content: Scorer) -> dict[str, Scorer]:
    """Outcome-free orders from the change's structural footprint, module distance and path distance, each breaking
    ties by duration as QTF does; and the module order fused with content by reciprocal rank."""

    def by(key: str) -> Scorer:
        return lambda h, t: structure()[key][t] * SCALE + h.last_duration

    module = by("module")
    return {"Module": module, "Path": by("path"), "Module+TF-IDF": rank_fusion(module, content)}
