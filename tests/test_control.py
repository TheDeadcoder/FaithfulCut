import numpy as np
import pytest
from scipy import stats
from synthetic import make_suite, perturb, random_suite

from faithfulcut.control import (
    AUDIT, WARMUP, Cusum, FixedCut, MetaCut, Monitor, OnlineCut, SplitCut, crc_quantile, ht_widths, max_pending,
    online_slack, pac_quantile, transfer_widths,
)
from faithfulcut.replay import AUDIT_FLOOR, audit_coins, replay
from faithfulcut.scorers import qtf


def test_crc_allows_floor_of_n_plus_one_alpha_minus_one_misses():
    d = np.linspace(0.01, 0.99, 59)
    assert crc_quantile(d[:19], 0.05) == d[18]
    assert crc_quantile(d, 0.05) == d[56]
    assert crc_quantile(d[:18], 0.05) == 1.0


def test_pac_needs_59_clean_builds_at_five_percent():
    d = np.linspace(0.01, 0.99, 200)
    assert pac_quantile(d[:58], 0.05, 0.05) == 1.0
    assert pac_quantile(d[:59], 0.05, 0.05) == d[58]
    allowed = max(m for m in range(201) if stats.binom.cdf(m, 200, 0.05) <= 0.05)
    assert pac_quantile(d, 0.05, 0.05) == d[199 - allowed]


def feed(cut, depths, kinds=None):
    for t, d in enumerate(depths):
        lam = cut.decide(t, False)
        cut.observe(t, np.array([d]), bool(d <= lam), AUDIT if kinds is None else kinds[t])


def test_online_identity_range_and_bound_hold_on_any_sequence():
    alpha, eta = 0.1, 0.2
    cut = OnlineCut(alpha, eta, start=0.5)
    depths = np.random.default_rng(1).beta(0.5, 0.5, 3000)
    feed(cut, depths)
    losses = np.array(cut.losses)
    assert np.isclose((losses - alpha).sum(), (cut.path[-1] - 0.5) / eta)
    assert min(cut.path) >= -eta * alpha - 1e-12 and max(cut.path) <= 1 + eta * (1 - alpha) + 1e-12
    assert abs(losses.mean() - alpha) <= online_slack(eta, len(losses), 0)


def test_online_cut_holds_alpha_on_a_drifting_stream_seen_only_through_audits():
    alpha, eta, pi, n = 0.1, 0.2, 0.2, 6000
    rng = np.random.default_rng(7)
    depths = np.concatenate([rng.beta(2, 8, n // 2), rng.beta(5, 3, n // 2)])
    audited = rng.random(n) < pi
    cut = OnlineCut(alpha, eta, start=0.5)
    missed = np.zeros(n)
    for t in range(n):
        lam = cut.decide(t, False)
        missed[t] = depths[t] > lam
        if audited[t]:
            cut.observe(t, depths[t:t + 1], not missed[t], AUDIT)
    assert abs(np.mean(cut.losses) - alpha) <= online_slack(eta, len(cut.losses), 0)
    assert abs(missed.mean() - alpha) < 0.02


def test_engine_feeds_coin_audits_only_and_scores_the_prefix_chosen_before_the_build():
    s = random_suite(n_builds=120)
    label = s.failed
    cut = OnlineCut(0.2, 0.1)
    rep = replay(s, "faithful", qtf, lam=cut, pi=0.3, seed=2, label=label)
    finish = s.builds["finish"].to_numpy()
    expected = []
    for t in sorted(range(len(s.builds)), key=lambda u: (finish[u], u)):
        rows = s.rows(t)
        if rep.audited[t] and not rep.reason[t] == 1 and label[rows].any():
            expected.append(0.0 if (rep.selected[rows] & label[rows]).any() else 1.0)
    assert cut.losses == expected and len(expected) > 10
    assert len(np.unique(rep.lam[rep.reason == 0])) > 1


def test_size_cap_and_alarms_force_full_runs_while_the_coin_keeps_flipping():
    s = random_suite(n_builds=120)
    capped = replay(s, "faithful", qtf, lam=0.6, pi=0.1, seed=4, cap=0.5)
    assert (capped.reason[capped.reason != 1] == 2).all() and capped.forced.all()
    assert np.array_equal(capped.audited, audit_coins(4, len(s.builds)) < 0.1)
    rows = [(np.arange(3), np.ones(3, dtype=bool), np.ones(3)) for _ in range(80)]
    start = np.arange(80) * 10.0
    failing = make_suite(rows, start, start + 1)
    alarmed = replay(failing, "faithful", qtf, lam=-1.0, pi=0.5, seed=1, label=failing.failed,
                     monitor=Monitor(alpha=0.05, delta=0.5, rho=1.0), hold=5)
    alarms = np.flatnonzero(alarmed.reason == 3)
    assert len(alarms) >= 5 and alarmed.audited[alarms].any() and not alarmed.audited[alarms].all()
    assert alarmed.alarms and all(alarmed.reason[t] == 3 for t in alarmed.alarms if t < len(rows))


def test_monitor_rarely_raises_false_alarms_and_catches_a_jump():
    rng = np.random.default_rng(0)

    def alarm_share(rate):
        hits = 0
        for _ in range(200):
            m = Monitor(alpha=0.05)
            hits += any(m.update(float(x)) for x in rng.random(500) < rate)
        return hits / 200

    assert alarm_share(0.05) <= 0.06
    assert alarm_share(0.25) >= 0.95


def test_transfer_widths_reproduce_the_worked_kafka_example():
    w = transfer_widths(caught=655, audited_misses=9, pi=0.25, limit=1345)
    assert w["estimate"] == pytest.approx(36 / 691)
    assert w["worst"] == pytest.approx(0.19, abs=0.005)
    assert w["normal"] < w["adaptive"] < w["worst"]


def test_adaptive_certificate_covers_the_escape_rate_without_the_count_of_faulty_builds():
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(400):
        misses = rng.binomial(60, 0.1)
        w = transfer_widths(caught=540, audited_misses=misses, pi=0.1, limit=2460)
        covered.append(abs(w["estimate"] - 0.1) <= w["adaptive"])
    assert np.mean(covered) >= 0.95


def test_max_pending_counts_overlaps_but_not_touching_intervals():
    assert max_pending(np.array([0, 1, 2, 6, 10]), np.array([10, 3, 5, 8, 12])) == 3
    assert max_pending(np.array([0, 5]), np.array([5, 9])) == 1


def test_split_cut_counts_warmup_builds_in_its_window():
    cut = SplitCut(alpha=0.05, window=19)
    depths = np.linspace(0.05, 0.95, 19)
    feed(cut, depths, kinds=[WARMUP] * 19)
    assert cut.value == depths.max()


def test_meta_cut_recalibrates_to_its_recall_targets():
    cut = MetaCut(test_recall=0.5, change_recall=0.5, period=2)
    for t, d in enumerate([np.array([0.1, 0.9]), np.array([0.2, 0.3]), np.array([0.4]), np.array([0.6, 0.8])]):
        cut.observe(t, d, False, AUDIT)
    rows = np.sort([0.1, 0.9, 0.2, 0.3, 0.4, 0.6, 0.8])
    assert cut.value == max(rows[3], np.sort([0.1, 0.2, 0.4, 0.6])[1])


def test_fixed_cut_ignores_feedback():
    cut = FixedCut(0.3)
    feed(cut, [0.9, 0.9, 0.9])
    assert cut.decide(0, False) == 0.3


def test_unequal_probability_widths_match_the_uniform_ones_when_pi_is_constant():
    rng = np.random.default_rng(0)
    faulty, audited = rng.random(1000) < 0.4, rng.random(1000) < 0.25
    caught = faulty & (rng.random(1000) > 0.1)
    misses = int((audited & faulty & ~caught).sum())
    ht = ht_widths(faulty, caught, audited, np.full(1000, 0.25))
    uniform = transfer_widths(int((faulty & caught).sum()), misses, 0.25, int((~(faulty & caught)).sum()))
    assert ht["normal"] == pytest.approx(uniform["normal"]) and ht["worst"] == pytest.approx(uniform["worst"])
    assert ht["estimate"] == pytest.approx(uniform["estimate"])


def test_online_step_scales_with_its_weight():
    a, b = OnlineCut(0.1, 0.2, start=0.5), OnlineCut(0.1, 0.2, start=0.5)
    a.observe(0, np.array([0.9]), False, AUDIT)
    b.observe(0, np.array([0.9]), False, AUDIT, weight=2.0)
    assert b.value - 0.5 == pytest.approx(2 * (a.value - 0.5))


def change_share(seed=0, n_builds=200, n_classes=30):
    relevance = np.random.default_rng(seed).random((n_builds, n_classes)) ** 3

    def signal(t, depth_by_class, lam):
        r = relevance[t]
        return float(r[depth_by_class > lam].sum() / r.sum())

    return signal


def test_adaptive_audit_probability_follows_the_signal_over_its_running_mean():
    s = random_suite(n_builds=200)
    rep = replay(s, "faithful", qtf, lam=0.3, pi=0.2, seed=3, signal=change_share(), adaptive=True)
    warm = s.builds["position"].to_numpy() < 0.2
    total = count = 0.0
    for t in np.flatnonzero(~warm):
        expected = 0.2 if not count else min(max(0.2 * rep.signal[t] * count / total, AUDIT_FLOOR), 1.0)
        assert rep.pi[t] == pytest.approx(expected)
        total, count = total + rep.signal[t], count + 1
    coins = audit_coins(3, len(s.builds))
    assert np.array_equal(rep.audited[~warm], coins[~warm] < rep.pi[~warm])
    assert rep.audited[~warm & (coins < AUDIT_FLOOR)].all()
    assert 0.12 < rep.pi[~warm].mean() < 0.3


def test_adaptive_audits_ignore_rows_never_run():
    s = random_suite(n_builds=200)
    a = replay(s, "faithful", qtf, lam=0.3, pi=0.2, seed=3, signal=change_share(), adaptive=True)
    b = replay(perturb(s, ~a.executed(s)), "faithful", qtf, lam=0.3, pi=0.2, seed=3, signal=change_share(),
               adaptive=True)
    assert np.array_equal(a.audited, b.audited) and np.array_equal(a.selected, b.selected)


def test_cusum_alarms_on_a_run_of_misses_and_rarely_at_the_target_rate():
    c = Cusum(alpha=0.1, ratio=3.0, h=5.0)
    assert [c.update(1.0) for _ in range(5)] == [False, False, False, False, True]
    rng = np.random.default_rng(0)
    c = Cusum(alpha=0.1, ratio=3.0, h=5.0)
    assert sum(c.update(float(x)) for x in rng.random(2000) < 0.1) <= 3
