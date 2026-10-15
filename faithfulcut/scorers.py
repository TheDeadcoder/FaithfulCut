import numpy as np

from faithfulcut.replay import History, Scorer
from faithfulcut.suite import Suite

EPS = 1e-6


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
