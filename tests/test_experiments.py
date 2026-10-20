import json

import numpy as np
import pandas as pd
import pytest
from synthetic import make_suite, random_suite

from faithfulcut.analysis import at_work, break_even_hours, energy
from faithfulcut.experiments import summarize
from faithfulcut.labels import cold_rows
from faithfulcut.llm import risk_levels, timed_state
from faithfulcut.metrics import first_failure_time
from faithfulcut.replay import replay
from faithfulcut.scorers import qtf


def test_before_keeps_only_builds_below_the_position():
    s = random_suite(n_builds=50)
    head = s.before(0.7)
    n = int((s.builds["position"] < 0.7).sum())
    assert len(head.builds) == n == 35
    assert np.array_equal(head.cls, s.cls[:s.offsets[n]])
    assert np.array_equal(head.changed(n - 1), s.changed(n - 1))


def test_matched_work_interpolates_between_lambda_points():
    rows = []
    for lam, work, escapes in ((0.1, 0.1, 6), (0.3, 0.3, 2)):
        row = {"project": "p", "scorer": "s", "regime": "full", "pi": 0.0, "seed": 0, "lam": lam, "work": work}
        for label in ("operational", "deconf", "firstfail"):
            row.update({f"{label}_faulty": 10, f"{label}_escape": escapes, f"{label}_executed_escape": escapes,
                        f"{label}_audited": 0, f"{label}_audited_escape": 0, f"{label}_rows": 20,
                        f"{label}_rows_selected": 20 - 2 * escapes})
        rows.append(row)
    matched = at_work(pd.DataFrame(rows), targets=(0.2,))
    assert matched["operational_escape_rate"].item() == pytest.approx(0.4)
    assert matched["operational_recall"].item() == pytest.approx(0.6)


def test_timed_state_reads_only_builds_finished_before_the_start():
    rows = [(np.array([0]), np.array([True]), np.array([5.0])), (np.array([1]), np.array([False]), np.array([7.0])),
            (np.array([0, 1]), np.array([False, False]), np.array([1.0, 1.0]))]
    s = make_suite(rows, start=np.array([0.0, 1.0, 2.0]), finish=np.array([1.5, 10.0, 3.0]))
    catalog, duration, since_fail = timed_state(s, [2])[2]
    assert catalog.tolist() == [0] and duration.tolist() == [5.0] and since_fail.tolist() == [1.0]


def test_llm_answers_rank_named_classes_and_flag_malformed_output():
    names = np.array(["a.ATest", "a.BTest", "a.CTest"])
    text = json.dumps({"tests": [{"name": "a.BTest", "risk": "high"}, {"name": "x.Unknown", "risk": "high"},
                                 {"name": "a.ATest", "risk": "LOW"}]})
    ranks, ok = risk_levels({"candidates": [{"content": {"parts": [{"text": text}]}}]}, names)
    assert ok and ranks.tolist() == [2, 0, 3]
    ranks, ok = risk_levels({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}, names)
    assert not ok and ranks.tolist() == [3, 3, 3]


def two_builds(failing, durations):
    cls, duration = np.arange(len(durations)), np.asarray(durations, dtype=float)
    rows = [(cls, np.zeros(len(cls), dtype=bool), duration), (cls, np.isin(cls, failing), duration)]
    return make_suite(rows, start=np.array([0.0, 10.0]), finish=np.array([1.0, 11.0]))


def test_first_failure_time_adds_test_time_in_run_order():
    s = two_builds([0], [3.0, 1.0, 2.0])
    rep = replay(s, "full", qtf, lam=1.0, warmup=0.0)
    assert first_failure_time(s, rep, s.failed, np.array([False, True]))[1] == 6.0


def test_an_escape_is_masked_when_another_selected_class_fails():
    s = two_builds([0, 1], [5.0, 1.0])
    label = np.array([False, False, True, False])
    rep = replay(s, "full", qtf, lam=0.2, warmup=0.0)
    out = summarize(s, rep, {"x": label}, np.array([False, True]), timing=True)
    assert (out["x_faulty"], out["x_escape"], out["x_masked"]) == (1, 1, 1)
    assert np.isnan(out["x_ttff_hours"]) and out["verdict_hours"] == 1.0 / 3600


def test_cold_rows_count_runs_in_builds_finished_before_the_start():
    rows = [(np.array([0]), np.array([False]), np.array([1.0]))] * 6
    rows.append((np.array([0, 1]), np.zeros(2, bool), np.ones(2)))
    start = np.arange(7) * 10.0
    s = make_suite(rows, start=start, finish=start + 1)
    assert cold_rows(s).tolist() == [False, True, True, True, True, False, False, False]
    late = make_suite(rows, start=start, finish=np.where(np.arange(7) == 4, 100.0, start + 1))
    assert cold_rows(late)[5]


def test_energy_and_break_even_scale_linearly():
    e = energy(10.0)
    assert (e["kwh_low"], e["kwh_high"]) == pytest.approx((0.1, 3.0))
    assert (e["kg_co2e_low"], e["kg_co2e_high"]) == pytest.approx((0.002, 2.1))
    assert break_even_hours(36.0, 0.5) == pytest.approx(0.02) and break_even_hours(1.0, 1.0) == np.inf
