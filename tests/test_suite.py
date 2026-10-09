import os

import numpy as np
import pandas as pd
import pytest

from faithfulcut import lrts
from faithfulcut.replay import replay
from faithfulcut.scorers import qtf
from faithfulcut.suite import ROW_ARRAYS, load_suite

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LRTS_REPO") and os.environ.get("LRTS_DIR")),
    reason="LRTS_REPO and LRTS_DIR are not set",
)


@pytest.mark.parametrize("project", ["activemq", "log4j"])
def test_suite_merges_stage_runs_and_survives_the_cache(project, tmp_path, monkeypatch):
    monkeypatch.setenv("FAITHFULCUT_CACHE", str(tmp_path))
    suite, cached = load_suite(project), load_suite(project)
    for a in ROW_ARRAYS + ["names"]:
        assert np.array_equal(getattr(suite, a), getattr(cached, a))
    assert (suite.builds.astype(str).to_numpy() == cached.builds.astype(str).to_numpy()).all()

    stages = lrts.load_runs(project).groupby(["pr_name", "build_id"])["stage_id"].apply(list)
    mismatches = []
    for t, b in enumerate(suite.builds.itertuples(index=False)):
        runs = [lrts.read_results(project, b.pr_name, b.build_id, s) for s in stages[(b.pr_name, b.build_id)]]
        rows = suite.rows(t)
        names = suite.names[suite.cls[rows]]
        failing = set().union(*(set(r.loc[r["outcome"] == 1, "testclass"]) for r in runs))
        everywhere = {n for n in failing if all(r.loc[r["testclass"] == n, "outcome"].min() == 1
                                                for r in runs if (r["testclass"] == n).any())}
        ok = (
            set(names) == set().union(*(set(r["testclass"]) for r in runs))
            and set(names[suite.failed[rows]]) == failing
            and set(names[suite.failed_everywhere[rows]]) == everywhere
            and np.isclose(suite.duration[rows].sum(), sum(r["duration"].sum() for r in runs))
        )
        if not ok:
            mismatches.append((b.pr_name, b.build_id))
    assert not mismatches, mismatches[:20]


def test_replay_extremes_on_real_data():
    suite = load_suite("activemq")
    everything = replay(suite, "faithful", qtf, lam=1.0)
    assert everything.selected.all()
    nothing = replay(suite, "faithful", qtf, lam=-1.0)
    after_warmup = ~np.repeat(nothing.forced, np.diff(suite.offsets))
    assert np.all(nothing.depth[nothing.selected & after_warmup] == -np.inf)
    assert nothing.executed_cost[~nothing.forced].sum() < 0.05 * nothing.full_cost[~nothing.forced].sum()
