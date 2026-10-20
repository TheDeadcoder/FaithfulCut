import os

import numpy as np
import pytest

from faithfulcut import lrts
from faithfulcut.suite import load_suite

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LRTS_REPO") and os.environ.get("LRTS_DIR") and os.environ.get("FAITHFULCUT_CACHE")),
    reason="LRTS_REPO, LRTS_DIR and FAITHFULCUT_CACHE are not all set",
)


def test_later_builds_follow_the_studied_window_in_start_order():
    studied, extended = lrts.load_builds("hbase"), lrts.load_builds("hbase", later=True)
    n = len(studied)
    head = extended.iloc[:n]
    assert head["pr_name"].tolist() == studied["pr_name"].tolist()
    assert np.array_equal(head["position"], studied["position"])
    later = extended.iloc[n:]
    assert len(later) == 206 and later["start"].min() > studied["start"].max()
    assert (later["position"] >= 1).all() and later["position"].is_monotonic_increasing


def test_extended_suite_keeps_the_studied_rows():
    studied, extended = load_suite("hbase"), load_suite("hbase", later=True)
    head = extended.before(1.0)
    assert extended.key == "hbase-later" and studied.key == "hbase"
    assert np.array_equal(head.names[:len(studied.names)], studied.names)
    for a in ("offsets", "cls", "failed", "failed_everywhere", "duration", "file_offsets", "file_ids"):
        assert np.array_equal(getattr(head, a), getattr(studied, a)), a
