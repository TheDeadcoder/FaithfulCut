import numpy as np
import pytest
from synthetic import make_suite, perturb, random_suite

from faithfulcut.replay import replay
from faithfulcut.scorers import history_scorers


def sequential(failing, n_classes=3, changed=None, durations=None):
    rows = [(np.arange(n_classes), np.isin(np.arange(n_classes), f), np.ones(n_classes) if durations is None
             else np.asarray(durations, dtype=float)) for f in failing]
    start = np.arange(len(failing)) * 10.0
    return make_suite(rows, start, start + 1, changed=changed)


def first_in_order(suite, scorer, t):
    rep = replay(suite, "faithful", scorer, lam=1.0, warmup=0.0)
    rows = suite.rows(t)
    return suite.cls[rows][np.argsort(rep.depth[rows], kind="stable")].tolist()


def test_latestfail_puts_the_most_recent_failure_first():
    s = sequential([[], [0], [1], []])
    assert first_in_order(s, history_scorers(s)["LatestFail"], 3)[0] == 1


def test_failcount_orders_by_observed_failures():
    s = sequential([[2], [2], [1], []])
    assert first_in_order(s, history_scorers(s)["FailCount"], 3)[:2] == [2, 1]


def test_tf_failfreq_follows_classes_that_failed_with_the_changed_files():
    s = sequential([[2], [1], [], []], changed=[[0], [1], [0], [1]])
    scorer = history_scorers(s)["TF-FailFreq"]
    assert first_in_order(s, scorer, 2)[0] == 2
    assert first_in_order(s, scorer, 3)[0] == 1


def test_audit_only_cofailures_carry_inverse_propensity_weight():
    n = 40
    s = sequential([[1]] * n, n_classes=2, changed=[[0]] * n, durations=[1.0, 100.0])
    rep = replay(s, "faithful", history_scorers(s)["QTF"], lam=0.5, pi=0.25, seed=11, warmup=0.1)
    forced, by_audit = rep.forced.sum(), (rep.audited & ~rep.forced).sum()
    assert by_audit > 0
    assert rep.history.cofails[0][1] == forced + by_audit
    assert np.isclose(rep.history.weighted_cofails[0][1], forced + by_audit / 0.25)


@pytest.mark.parametrize("name", ["TF-FailFreq-W+CC", "FailCount-W+CC", "LatestFail+CC"])
def test_faithful_history_scorers_ignore_rows_they_never_ran(name):
    s = random_suite()
    a = replay(s, "faithful", history_scorers(s)[name], lam=0.3, pi=0.2, seed=3)
    hidden = ~a.executed(s)
    b_suite = perturb(s, hidden)
    b = replay(b_suite, "faithful", history_scorers(b_suite)[name], lam=0.3, pi=0.2, seed=3)
    assert np.array_equal(a.selected, b.selected) and np.array_equal(a.depth, b.depth)
