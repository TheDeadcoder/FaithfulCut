import io
import json
import subprocess
import urllib.error

import numpy as np
import pytest

from faithfulcut import llm
from faithfulcut.llm import ask, budget_for_share, caught, first_depth, llm_keys, risk_levels

NAMES = np.array(["A", "B", "C"])


def answer(tests) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"tests": tests})}]}}]}


def test_risk_levels_rank_named_classes_and_ignore_unknown_names():
    ranks, ok = risk_levels(answer([{"name": "C", "risk": "high"}, {"name": "Z", "risk": "high"},
                                    {"name": "A", "risk": "LOW"}]), NAMES)
    assert ok and ranks.tolist() == [2, 3, 0]


def test_a_malformed_answer_names_nothing():
    for bad in ({}, {"candidates": []}, answer([{"name": "C", "risk": "high"}, {"risk": "low"}])):
        ranks, ok = risk_levels(bad, NAMES)
        assert not ok and ranks.tolist() == [3, 3, 3]


def test_the_risk_order_without_names_is_tf_idf_cc():
    rng = np.random.default_rng(1)
    for _ in range(50):
        duration, tfidf_cc = rng.uniform(1, 5, 20), rng.integers(0, 4, 20).astype(float)
        failing = rng.choice(20, 2, replace=False)
        keys = llm_keys(np.full(20, 3), tfidf_cc, duration, "risk")
        assert caught(keys, duration, failing, np.random.default_rng(7), 0.1) == \
            caught([tfidf_cc], duration, failing, np.random.default_rng(7), 0.1)


def test_a_failing_class_is_caught_when_it_starts_within_the_budget():
    duration, key = np.ones(4), np.arange(4.0)
    assert caught([key], duration, np.array([1]), np.random.default_rng(0), 0.5) == 1.0
    assert caught([key], duration, np.array([1]), np.random.default_rng(0), 0.25) == 0.0


def network(monkeypatch, outcomes) -> list[float]:
    """Each request gets the next outcome, raised if it is an error; the pauses between attempts are recorded."""
    calls, pauses = iter(outcomes), []

    def urlopen(request, timeout):
        outcome = next(calls)
        if isinstance(outcome, BaseException):
            raise outcome
        return io.BytesIO(json.dumps(outcome).encode())

    monkeypatch.setattr(llm, "_request", lambda model, body: None)
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm.time, "sleep", pauses.append)
    return pauses


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "error", {}, io.BytesIO(b"detail"))


def test_first_depth_is_the_predicted_time_share_by_which_a_failing_class_has_run():
    duration = np.array([1.0, 2.0, 1.0])
    assert first_depth([np.array([0, 1, 2])], duration, np.array([1, 2]), np.random.default_rng(0)) == 0.75


def test_budget_for_share_is_the_smallest_depth_reaching_the_share():
    depth = np.array([0.3, 0.1, 0.2, 0.5])
    weights = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 1.0, 1.0]])
    assert budget_for_share(depth, weights, np.array([0.5, 0.5])).tolist() == [0.2, 0.1]


def test_ask_waits_out_dropped_connections_rate_limits_and_rejected_tokens(monkeypatch):
    monkeypatch.setitem(llm._token, "value", "old")
    pauses = network(monkeypatch, [urllib.error.URLError("offline"), ConnectionResetError(), http_error(429),
                                   http_error(401), {"candidates": []}])
    assert ask("q", "m")["candidates"] == [] and llm._token["value"] is None
    assert len(pauses) == 4 and 8 <= pauses[0] <= 12 and pauses[0] < pauses[1] < pauses[2] < pauses[3]


def test_ask_raises_at_once_on_a_bad_request_and_after_its_patience(monkeypatch):
    pauses = network(monkeypatch, [http_error(400)])
    with pytest.raises(RuntimeError, match="HTTP 400"):
        ask("q", "m")
    assert pauses == []
    network(monkeypatch, [subprocess.CalledProcessError(1, "gcloud")] * 3)
    with pytest.raises(RuntimeError, match="gave up"):
        ask("q", "m", patience=20)
