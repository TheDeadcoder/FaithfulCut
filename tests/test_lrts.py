import hashlib
import json
import os
import subprocess

import pandas as pd
import pytest

from faithfulcut import lrts

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LRTS_REPO") and os.environ.get("LRTS_DIR")),
    reason="LRTS_REPO and LRTS_DIR are not set",
)

LRTS_COMMIT = "2c1b68e9b1d4ec6c4f304869e36af2e3168da889"
DATASET_SHA256 = "49570f8dca753c385f25346f8aaaded3a841b0e67dabf36fbb67260a79b02987"
CHECKED_PROJECTS = ["activemq", "hadoop", "log4j"]


def first_20(mismatches):
    return f"{len(mismatches)} mismatches, first 20:\n" + "\n".join(map(str, mismatches[:20]))


def test_inputs_match_pins():
    head = subprocess.run(
        ["git", "-C", str(lrts.repo_dir()), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    digest = hashlib.sha256(lrts.metadata_path().read_bytes()).hexdigest()
    assert (head, digest) == (LRTS_COMMIT, DATASET_SHA256)


def test_every_run_has_results_and_every_build_has_change_data():
    meta = lrts.metadata()
    missing = [
        (r.project, r.pr_name, r.build_id, r.stage_id)
        for r in meta.itertuples(index=False)
        if not lrts.results_path(r.project, r.pr_name, r.build_id, r.stage_id).is_file()
    ]
    for b in meta.drop_duplicates(["project", "pr_name", "build_id"]).itertuples(index=False):
        if not lrts.diff_path(b.project, b.pr_name, b.build_id).is_file():
            missing.append((b.project, b.pr_name, b.build_id, "diff"))
        if not lrts.compare_pages(b.project, b.pr_name, b.build_id, b.trunk_sha, b.build_head_sha):
            missing.append((b.project, b.pr_name, b.build_id, "comparison"))
    assert not missing, first_20(missing)


@pytest.mark.parametrize("project", CHECKED_PROJECTS)
def test_results_match_metadata(project):
    meta = lrts.metadata()
    meta = meta[meta["project"] == project].set_index(lrts.RUN_KEY)
    runs = lrts.load_runs(project)
    assert len(runs) == len(meta)
    mismatches = []
    for r in runs.itertuples(index=False):
        res = lrts.read_results(project, r.pr_name, r.build_id, r.stage_id)
        exp = meta.loc[(project, r.pr_name, r.build_id, r.stage_id)]
        got = (int((res["outcome"] == 0).sum()), int((res["outcome"] == 1).sum()), round(res["duration"].sum(), 3))
        want = (int(exp["num_pass_class"]), int(exp["num_fail_class"]), round(exp["test_suite_duration_s"], 3))
        if not set(res["outcome"]) <= {0, 1} or got[:2] != want[:2] or got[2] > want[2] + 1:
            mismatches.append((r.pr_name, r.build_id, r.stage_id, got, want))
    assert not mismatches, first_20(mismatches)


@pytest.mark.parametrize("project", CHECKED_PROJECTS)
def test_comparison_pages_match_change_stats(project):
    stats = pd.read_csv(lrts.repo_dir() / "artifact" / "evaluation" / "change_info" / "change_stats.csv")
    stats = stats[stats["project"] == project].set_index(["pr_name", "build_id"])["num_changed_files"]
    builds = lrts.load_runs(project).drop_duplicates(["pr_name", "build_id"])
    assert len(builds) == len(stats)
    mismatches = []
    for b in builds.itertuples(index=False):
        pages = lrts.compare_pages(project, b.pr_name, b.build_id, b.trunk_sha, b.build_head_sha)
        listed = sum(len(json.loads(p.read_text()).get("files", [])) for p in pages)
        if listed != stats.loc[(b.pr_name, b.build_id)]:
            mismatches.append((b.pr_name, b.build_id, listed, int(stats.loc[(b.pr_name, b.build_id)])))
    assert not mismatches, first_20(mismatches)
