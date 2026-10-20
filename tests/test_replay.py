import numpy as np
import pytest
from synthetic import make_suite, perturb, random_suite

from faithfulcut.metrics import escape_causes, hidden_latency
from faithfulcut.replay import REGIMES, audit_coins, replay
from faithfulcut.scorers import qtf


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


def three_classes(failing, n_builds):
    rows = [(np.arange(3), np.isin(np.arange(3), failing.get(t, [])), np.ones(3)) for t in range(n_builds)]
    start = np.arange(n_builds) * 10.0
    return make_suite(rows, start, start + 1)


def test_scheduled_full_runs_follow_the_seeded_phase_after_the_warmup():
    s = three_classes({}, 10)
    rep = replay(s, "faithful", qtf, lam=0.0, seed=1, warmup=0.2, schedule=4)
    assert np.flatnonzero(rep.reason == 4).tolist() == [3, 7] and rep.forced[[3, 7]].all()


def test_rotation_runs_the_stalest_classes_within_its_budget():
    s = three_classes({}, 4)
    rep = replay(s, "faithful", qtf, lam=0.0, warmup=0.0, rotate=0.34)
    assert [s.cls[s.rows(t)][rep.rotated[s.rows(t)]].tolist() for t in range(4)] == [[], [0], [1], [2]]
    assert not rep.selected[s.rows(1)].any() and rep.executed_cost[1] == 1.0 and rep.base_cost[1] == 0.0


def test_hidden_failures_wait_until_their_class_runs_again():
    s = three_classes({1: [0], 3: [0]}, 4)
    rep = replay(s, "faithful", qtf, lam=0.0, warmup=0.0, schedule=2)
    n, builds, hours = hidden_latency(s, rep, s.failed, np.ones(4, dtype=bool))
    assert (n, builds) == (2, 1.0) and hours == pytest.approx(10 / 3600)


def test_escapes_are_new_seen_or_hidden_by_what_the_replay_observed():
    s = three_classes({0: [0], 1: [0], 2: [1], 3: [1]}, 4)
    rep = replay(s, "faithful", qtf, lam=0.0, warmup=0.0)
    every = np.ones(4, dtype=bool)
    assert escape_causes(s, rep, s.failed, every, rep.executed(s)) == {"new": 1, "seen": 1, "hidden": 1}
    assert escape_causes(s, rep, s.failed, every, np.ones(len(s.cls), dtype=bool)) == {"new": 1, "seen": 2, "hidden": 0}


def by_index(history, t):
    return np.arange(len(history.known), dtype=float)


def test_a_relevance_threshold_covers_its_share_of_the_mass_and_runs_everything_without_one():
    s = three_classes({}, 3)
    weights = {1: np.array([0.6, 0.3, 0.1]), 2: np.zeros(3)}
    mass = lambda t: weights.get(t, np.ones(3))
    run = lambda tau: replay(s, "faithful", by_index, lam=-np.log(1 - tau), warmup=0.0, mass=mass)
    assert s.cls[s.rows(1)][run(0.5).selected[s.rows(1)]].tolist() == [0]
    assert s.cls[s.rows(1)][run(0.7).selected[s.rows(1)]].tolist() == [0, 1]
    assert s.cls[s.rows(2)][run(0.2).selected[s.rows(2)]].tolist() == [0, 1, 2]


def test_a_filter_runs_only_its_classes_and_the_unseen_ones():
    s = make_suite([(np.arange(2), np.zeros(2, bool), np.ones(2)), (np.arange(3), np.zeros(3, bool), np.ones(3))],
                   np.array([0.0, 10.0]), np.array([1.0, 11.0]))
    only = lambda t: np.array([False, True, False]) if t == 1 else None
    rep = replay(s, "faithful", by_index, lam=1.0, warmup=0.0, only=only)
    assert s.cls[s.rows(1)][rep.selected[s.rows(1)]].tolist() == [1, 2]
