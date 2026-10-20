import numpy as np
import pytest
from synthetic import make_suite, perturb, random_suite

from faithfulcut.replay import History, replay
from faithfulcut.scorers import (depth_split, history_scorers, hybrid_scorers, ir_scorers, rank_fusion,
                                 structure_scorers)
from faithfulcut.structure import UNKNOWN


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


def random_history(seed, n=60):
    rng = np.random.default_rng(seed)
    h = History(n)
    h.known[:] = rng.random(n) < 0.8
    h.last_duration[:] = rng.lognormal(0, 1, n)
    return h, rng.random(n), rng.random(n)


@pytest.mark.parametrize("seed", range(5))
def test_depth_split_runs_the_union_of_both_prefixes(seed):
    h, a, b = random_history(seed)
    split = depth_split(lambda h, t: a, lambda h, t: b)(h, 0)
    catalog = h.catalog()
    cost = h.last_duration[catalog]
    depth = {}
    for name, values in (("a", a), ("b", b)):
        order = np.argsort(values[catalog], kind="stable")
        depth[name] = np.empty(len(catalog))
        depth[name][order] = np.cumsum(cost[order])
    for d in np.quantile(depth["a"], [0.1, 0.3, 0.6]):
        union = set(catalog[(depth["a"] <= d) | (depth["b"] <= d)])
        assert set(catalog[split[catalog] <= d]) == union
    assert np.isinf(split[~h.known]).all()


def test_rank_fusion_favours_classes_high_in_both_orders():
    h, a, b = random_history(0)
    catalog = h.catalog()
    a[catalog[0]], b[catalog[0]] = -1.0, -1.0
    fused = rank_fusion(lambda h, t: a, lambda h, t: b)(h, 0)
    assert catalog[np.argmin(fused[catalog])] == catalog[0]
    assert np.allclose(fused, rank_fusion(lambda h, t: b, lambda h, t: a)(h, 0))


def synthetic_ir(suite, seed=0):
    rng = np.random.default_rng(seed)
    cls = [np.flatnonzero(rng.random(len(suite.names)) < 0.4) for _ in range(len(suite.builds))]
    return {"offsets": np.cumsum([0] + [len(c) for c in cls]), "cls": np.concatenate(cls),
            "score": rng.random(sum(len(c) for c in cls))}


@pytest.mark.parametrize("name", ["Hybrid", "Hybrid-RRF"])
def test_faithful_hybrids_ignore_rows_they_never_ran(name):
    s = random_suite()
    ir = synthetic_ir(s)

    def hybrid(suite):
        return hybrid_scorers(history_scorers(suite)["LatestFail+CC"], ir_scorers(suite, ir)["TF-IDF+CC"])[name]

    a = replay(s, "faithful", hybrid(s), lam=0.3, pi=0.2, seed=3)
    hidden = ~a.executed(s)
    assert hidden.any()
    b_suite = perturb(s, hidden)
    b = replay(b_suite, "faithful", hybrid(b_suite), lam=0.3, pi=0.2, seed=3)
    assert np.array_equal(a.selected, b.selected) and np.array_equal(a.depth, b.depth)


def test_structural_orders_run_the_nearest_classes_first_and_break_ties_by_duration():
    s = sequential([[], [], []], n_classes=4, durations=[4.0, 3.0, 2.0, 1.0])
    distance = np.array([[UNKNOWN] * 4, [UNKNOWN] * 4, [1, 0, 1, UNKNOWN]], dtype=np.uint8)
    structure = {"module": distance, "path": distance[:, ::-1].copy()}
    scorers = structure_scorers(lambda: structure, ir_scorers(s, {"offsets": np.zeros(4, int), "cls": np.empty(0, int),
                                                                  "score": np.empty(0)})["TF-IDF+CC"])
    assert first_in_order(s, scorers["Module"], 2) == [1, 2, 0, 3]
    assert first_in_order(s, scorers["Path"], 2) == [2, 3, 1, 0]
