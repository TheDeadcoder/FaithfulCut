import numpy as np
import pytest
from synthetic import make_suite, perturb, random_suite

from faithfulcut import learned
from faithfulcut.learned import EPS, Learned
from faithfulcut.replay import History, replay


def relevance_of(suite, seed=0):
    table = np.random.default_rng(seed).random((len(suite.builds), len(suite.names)))
    return lambda t: table[t]


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(learned, "REFIT", 10)
    monkeypatch.setattr(learned, "MIN_FAILURES", 3)


@pytest.mark.parametrize("weighted", [False, True])
def test_the_learned_order_ignores_rows_it_never_ran(small, weighted):
    s = random_suite()
    relevance = relevance_of(s)
    scorer = Learned(s, relevance, weighted)
    a = replay(s, "faithful", scorer, lam=0.3, pi=0.2, seed=3)
    hidden = ~a.executed(s)
    assert hidden.any() and scorer.model is not None
    t = perturb(s, hidden)
    b = replay(t, "faithful", Learned(t, relevance, weighted), lam=0.3, pi=0.2, seed=3)
    assert np.array_equal(a.selected, b.selected) and np.array_equal(a.depth, b.depth)


def test_each_replay_starts_the_model_afresh(small):
    s = random_suite()
    scorer = Learned(s, relevance_of(s))
    first = replay(s, "full", scorer, lam=0.3)
    second = replay(s, "full", scorer, lam=0.3)
    assert np.array_equal(first.selected, second.selected) and np.array_equal(first.depth, second.depth)


def test_the_model_ranks_a_class_that_keeps_failing_first(small):
    n = 60
    start = np.arange(n) * 10.0
    s = make_suite([(np.arange(5), np.arange(5) == 2, np.ones(5))] * n, start, start + 1)
    scorer = Learned(s, relevance_of(s))
    rep = replay(s, "full", scorer, lam=0.3)
    assert scorer.model is not None and np.argmin(scorer(rep.history, n - 1)) == 2


def test_before_its_first_fit_it_orders_by_relevance_per_second_and_weights_count_when_asked(monkeypatch):
    monkeypatch.setattr(learned, "NEGATIVES", 1.0)
    s = random_suite()
    relevance = relevance_of(s)
    for weighted, weights in ((False, [1.0, 1.0]), (True, [1.0, 4.0])):
        scorer, h = Learned(s, relevance, weighted), History(len(s.names))
        assert np.allclose(scorer(h, 0), 1 / (relevance(0) + EPS) * EPS)
        h.observe(np.array([0, 1]), np.array([True, False]), np.ones(2), np.array([1.0, 4.0]), np.array([], int), 0)
        _, y, w = scorer.window[0]
        assert y.tolist() == [True, False] and w.tolist() == weights
