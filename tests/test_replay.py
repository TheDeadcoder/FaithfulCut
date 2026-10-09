import dataclasses

import numpy as np
import pandas as pd
import pytest

from faithfulcut.replay import REGIMES, audit_coins, replay
from faithfulcut.scorers import qtf
from faithfulcut.suite import Suite


def make_suite(rows, start, finish):
    n = len(rows)
    builds = pd.DataFrame({
        "pr_name": [f"PR-{t}" for t in range(n)], "build_id": 1, "trunk_sha": "", "build_head_sha": "",
        "start": start, "finish": finish, "position": np.arange(n) / n,
    })
    cls = np.concatenate([r[0] for r in rows]).astype(np.int32)
    failed = np.concatenate([r[1] for r in rows]).astype(bool)
    return Suite(
        "synthetic", np.array([f"C{i}" for i in range(cls.max() + 1)]), builds,
        np.cumsum([0] + [len(r[0]) for r in rows]), cls, failed, failed.copy(),
        np.concatenate([r[2] for r in rows]).astype(float),
    )


def random_suite(seed=0, n_builds=80, n_classes=30, spread=40.0):
    rng = np.random.default_rng(seed)
    start = np.cumsum(rng.uniform(1, 10, n_builds))
    finish = start + rng.uniform(0.1, spread, n_builds)
    born = np.where(np.arange(n_classes) < n_classes // 2, 0, rng.integers(0, n_builds, n_classes))
    dies = rng.integers(n_builds // 3, 2 * n_builds, n_classes)
    base = rng.lognormal(0, 1, n_classes)
    rate = rng.uniform(0, 0.4, n_classes)
    rows = []
    for t in range(n_builds):
        alive = np.flatnonzero((born <= t) & (t < dies))
        rows.append((alive, rng.random(len(alive)) < rate[alive], base[alive] * rng.uniform(0.5, 1.5, len(alive))))
    return make_suite(rows, start, finish)


def perturb(suite, rows, seed=1):
    rng = np.random.default_rng(seed)
    failed = suite.failed.copy()
    failed[rows] = ~failed[rows]
    duration = suite.duration.copy()
    duration[rows] *= rng.uniform(0.1, 10, int(rows.sum()))
    return dataclasses.replace(suite, failed=failed, failed_everywhere=failed.copy(), duration=duration)


def mixed(history, t):
    return (history.last_duration * history.since_fail
            - history.weighted_fails / (history.weighted_runs + 1) + 0.01 * history.runs)


def assert_same_decisions(a, b, suite, builds):
    rows = np.repeat(builds, np.diff(suite.offsets))
    assert np.array_equal(a.selected[rows], b.selected[rows])
    assert np.array_equal(a.depth[rows], b.depth[rows])
    assert np.array_equal(a.known[builds], b.known[builds])


def test_audit_coins_are_nested_and_ignore_data_and_policy():
    s = random_suite()
    low = replay(s, "faithful", qtf, lam=0.3, pi=0.1, seed=7).audited
    high = replay(s, "timed", mixed, lam=0.6, pi=0.3, seed=7).audited
    other = replay(perturb(s, np.ones(len(s.cls), dtype=bool)), "full", qtf, lam=0.3, pi=0.1, seed=7).audited
    assert low.any() and high[low].all()
    assert np.array_equal(low, other)
    assert np.array_equal(low, audit_coins(7, len(s.builds)) < 0.1)


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7])
def test_faithful_ignores_rows_it_never_ran(lam):
    s = random_suite()
    a = replay(s, "faithful", mixed, lam=lam, pi=0.2, seed=3)
    hidden = ~a.executed(s)
    assert hidden.any()
    b = replay(perturb(s, hidden), "faithful", mixed, lam=lam, pi=0.2, seed=3)
    assert_same_decisions(a, b, s, np.ones(len(s.builds), dtype=bool))


@pytest.mark.parametrize("regime", ["timed", "faithful"])
def test_finish_gated_regimes_ignore_builds_unfinished_at_decision(regime):
    s = random_suite()
    start, finish = s.builds["start"].to_numpy(), s.builds["finish"].to_numpy()
    a = replay(s, regime, mixed, lam=0.4, pi=0.2, seed=5)
    for u in (20, 45, 70):
        late = np.repeat(finish > start[u], np.diff(s.offsets))
        b = replay(perturb(s, late), regime, mixed, lam=0.4, pi=0.2, seed=5)
        assert_same_decisions(a, b, s, start <= start[u])


def test_full_regime_ignores_builds_not_yet_started():
    s = random_suite()
    a = replay(s, "full", mixed, lam=0.4, seed=5)
    for u in (20, 45, 70):
        later = np.arange(len(s.builds)) >= u
        b = replay(perturb(s, np.repeat(later, np.diff(s.offsets))), "full", mixed, lam=0.4, seed=5)
        assert_same_decisions(a, b, s, np.arange(len(s.builds)) <= u)


def test_full_regime_reads_unfinished_builds_that_timed_cannot():
    s = random_suite(spread=200.0)
    a = replay(s, "full", mixed, lam=0.4, seed=5)
    b = replay(s, "timed", mixed, lam=0.4, seed=5)
    assert not np.array_equal(a.depth, b.depth)


def test_regimes_agree_without_overlap_when_everything_runs():
    s = random_suite(spread=0.5)
    reps = [replay(s, r, mixed, lam=1.0, seed=2) for r in REGIMES]
    for other in reps[1:]:
        assert_same_decisions(reps[0], other, s, np.ones(len(s.builds), dtype=bool))


def test_faithful_with_every_build_audited_equals_timed():
    s = random_suite()
    a = replay(s, "faithful", mixed, lam=0.3, pi=1.0, seed=4)
    b = replay(s, "timed", mixed, lam=0.3, pi=1.0, seed=4)
    assert_same_decisions(a, b, s, np.ones(len(s.builds), dtype=bool))


def test_cut_runs_the_longest_prefix_within_budget():
    rows = [(np.arange(4), np.zeros(4, dtype=bool), np.array([1.0, 2.0, 3.0, 4.0]))] * 2
    s = make_suite(rows, start=np.array([0.0, 10.0]), finish=np.array([1.0, 11.0]))
    rep = replay(s, "faithful", qtf, lam=0.35, warmup=0.0)
    assert rep.selected[:4].all()
    assert np.array_equal(rep.selected[4:], [True, True, False, False])
    assert np.allclose(rep.depth[4:], [0.1, 0.3, 0.6, 1.0])


def test_unseen_classes_run_by_default_and_absent_classes_retire():
    present = [[0, 1], [0, 1], [0], [0], [0], [0, 2]]
    rows = [(np.array(p), np.zeros(len(p), dtype=bool), np.ones(len(p))) for p in present]
    start = np.arange(len(present)) * 10.0
    s = make_suite(rows, start, start + 1)
    rep = replay(s, "faithful", qtf, lam=1.0, warmup=0.0, retire_after=2)
    assert list(rep.known) == [0, 2, 2, 2, 1, 1]
    assert np.array_equal(rep.depth[s.rows(5)], [1.0, -np.inf])


def test_audit_only_observations_carry_inverse_propensity_weight():
    n = 40
    rows = [(np.array([0, 1]), np.zeros(2, dtype=bool), np.array([1.0, 100.0]))] * n
    start = np.arange(n) * 10.0
    s = make_suite(rows, start, start + 1)
    rep = replay(s, "faithful", qtf, lam=0.5, pi=0.25, seed=11, warmup=0.1)
    forced, by_audit = rep.forced.sum(), (rep.audited & ~rep.forced).sum()
    assert by_audit > 0
    assert rep.history.runs[0] == n
    assert rep.history.runs[1] == forced + by_audit
    assert np.isclose(rep.history.weighted_runs[1], forced + by_audit / 0.25)
