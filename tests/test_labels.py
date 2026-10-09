import dataclasses
import os

import numpy as np
import pandas as pd
import pytest
from synthetic import make_suite, perturb, random_suite

from faithfulcut import lrts_ref
from faithfulcut.labels import STREAM_RATE, operational, transfer
from faithfulcut.replay import audit_coins, replay
from faithfulcut.scorers import qtf
from faithfulcut.suite import load_suite

NO_ISSUES = pd.DataFrame({"testclass": pd.Series(dtype=str), "created": pd.Series(dtype=float),
                          "fixed": pd.Series(dtype=float)})
needs_data = pytest.mark.skipif(
    not (os.environ.get("LRTS_REPO") and os.environ.get("LRTS_DIR")),
    reason="LRTS_REPO and LRTS_DIR are not set",
)


def chronic_suite(n=100):
    """C0 fails in every build; C1 to C20 each fail in every tenth build, staggered."""
    rows = [(np.arange(21), np.array([True] + [t % 10 == c % 10 for c in range(1, 21)]), np.ones(21))
            for t in range(n)]
    start = np.arange(n) * 10.0
    return make_suite(rows, start, start + 1)


def test_chronic_failures_are_frequent_and_occasional_ones_count():
    s = chronic_suite()
    label = operational(s, seed=0, issues=NO_ISSUES)
    assert not label[(s.cls == 0) & (s.build_of_rows() >= 5)].any()
    assert label[(s.cls != 0) & s.failed].all()


def test_failures_must_hold_in_every_stage_where_the_class_ran():
    s = chronic_suite()
    s = dataclasses.replace(s, failed_everywhere=s.failed & (s.cls != 3))
    label = operational(s, seed=0, issues=NO_ISSUES)
    assert not label[s.cls == 3].any()
    assert label[(s.cls == 4) & s.failed].all()


def test_flaky_issues_mask_failures_only_while_open():
    s = chronic_suite()
    issues = pd.DataFrame({"testclass": ["C7"], "created": [300.0], "fixed": [600.0]})
    label = operational(s, seed=0, issues=issues)
    start = s.builds["start"].to_numpy()[s.build_of_rows()]
    c7 = (s.cls == 7) & s.failed
    open_now = (start >= 300) & (start < 600)
    assert (c7 & open_now).any() and not label[c7 & open_now].any()
    assert label[c7 & ~open_now].all()


def test_frequency_rule_ignores_builds_unfinished_at_build_start():
    s = random_suite()
    start, finish = s.builds["start"].to_numpy(), s.builds["finish"].to_numpy()
    a = operational(s, seed=3, issues=NO_ISSUES, min_runs=2)
    for u in (20, 45, 70):
        late = np.repeat(finish > start[u], np.diff(s.offsets))
        b = operational(perturb(s, late, everywhere=False), seed=3, issues=NO_ISSUES, min_runs=2)
        early = np.repeat(start <= start[u], np.diff(s.offsets))
        assert np.array_equal(a[early], b[early])


def test_label_stream_is_seen_in_full_by_every_audited_policy():
    s = random_suite()
    stream = (s.builds["position"].to_numpy() < 0.2) | (audit_coins(5, len(s.builds)) < STREAM_RATE)
    rep = replay(s, "faithful", qtf, lam=0.1, pi=0.1, seed=5)
    assert (rep.audited | rep.forced)[stream].all()


@needs_data
def test_transfer_labels_mark_the_reference_runs_and_agree_on_single_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("FAITHFULCUT_CACHE", str(tmp_path))
    suite = load_suite("activemq")
    labels = transfer(suite)
    faulty = np.bincount(suite.build_of_rows(), weights=labels["deconf"], minlength=len(suite.builds)) > 0
    ref = lrts_ref.reference_apfdc("activemq")
    ref = ref.loc[(ref["label"] == "deconf") & ref["QTF"].notna(), ["pr_name", "build_id"]].drop_duplicates()
    expected = suite.builds.merge(ref, how="left", indicator=True)["_merge"].eq("both").to_numpy()
    assert np.array_equal(faulty, expected)
    ran = transfer(suite, stage_rule="ran")
    assert all(np.array_equal(labels[k], ran[k]) for k in labels)
