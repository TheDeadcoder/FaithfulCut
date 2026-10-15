import json

import numpy as np
import pandas as pd
import pytest
from synthetic import make_suite, random_suite

from faithfulcut.analysis import at_work
from faithfulcut.llm import risk_levels, timed_state


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
