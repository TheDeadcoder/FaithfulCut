import json
import os
import posixpath

import numpy as np
import pandas as pd
import pytest
from synthetic import perturb, random_suite

from faithfulcut import dante_ref, lrts
from faithfulcut.dante import (LOOKBACK, MIN_EVENTS, _features, apply_hunks, dante_scorers, diff_by_file, find_source,
                               minmax_tfidf, source_tokens, stickiness, term_counts)
from faithfulcut.replay import History, replay

needs_dante = pytest.mark.skipif(not os.environ.get("DANTE_DIR"), reason="DANTE_DIR is not set")
needs_data = pytest.mark.skipif(
    not (os.environ.get("DANTE_DIR") and os.environ.get("LRTS_DIR") and os.environ.get("LRTS_REPO")),
    reason="DANTE_DIR, LRTS_DIR and LRTS_REPO are not all set",
)


def dante_stickiness(cls, failed):
    rows = pd.DataFrame({"testclass": cls, "outcome": failed.astype(int)})
    out = []
    for n in range(1, LOOKBACK + 1):
        sticky = rows.groupby("testclass")["outcome"].shift(1).rolling(window=n, min_periods=1).max() == 1.0
        events = rows["outcome"][sticky]
        out.append(0.0 if len(events) < MIN_EVENTS else events.sum() / len(events))
    return np.array(out)


@pytest.mark.parametrize("seed, n_rows", [(0, 400), (1, 400), (2, 60), (3, 25), (4, 1000)])
def test_stickiness_follows_dantes_rolling_computation(seed, n_rows):
    rng = np.random.default_rng(seed)
    cls, failed = rng.integers(0, 15, n_rows), rng.random(n_rows) < 0.2
    assert np.allclose(stickiness(cls, failed), dante_stickiness(cls, failed))


def test_weighted_stickiness_counts_each_row_by_its_weight():
    rng = np.random.default_rng(5)
    cls, failed = rng.integers(0, 10, 300), rng.random(300) < 0.3
    assert np.allclose(stickiness(cls, failed, np.full(300, 4.0)), stickiness(cls, failed))
    weight = np.where(rng.random(300) < 0.5, 1.0, 10.0)
    assert not np.allclose(stickiness(cls, failed, weight), stickiness(cls, failed))


def test_java_tokens_drop_comments_annotations_and_keywords_and_split_identifiers():
    code = ('/* Licensed */ package org.apache.kafka; // header comment\n'
            '@Test(timeout = 10) public void testHTTPServerSplitBrain_v2() { String s = "Fetch request timed-out"; }')
    assert source_tokens(code, ".java") == ["brain", "fetch", "http", "kafka", "out", "request", "server", "split",
                                            "test", "timed"]


def test_other_languages_use_the_generic_tokenizer():
    assert source_tokens("def test_op(): # comment\n    return x_value", ".py") == ["test_op", "x_value"]
    assert source_tokens("", ".java") == [] and source_tokens(None, ".java") == []


def test_diff_lines_per_code_file_follow_dantes_parser():
    diff = "\n".join([
        "diff --git a/src/Foo.java b/src/Foo.java", "index 1..2 100644", "--- a/src/Foo.java", "+++ b/src/Foo.java",
        "@@ -1,3 +1,3 @@", " context", "-old line", "+new line", "--- removed line that looks like a header",
        "diff --git a/pom.xml b/pom.xml", "--- a/pom.xml", "+++ b/pom.xml", "@@ -1 +1 @@", "-<a/>", "+<b/>",
    ])
    assert diff_by_file(diff) == {"src/Foo.java": "old line\nnew line"}


def test_source_match_prefers_package_folders_then_test_folders_then_path_order():
    fqn = "org.apache.foo.TestBar"
    assert find_source(["a/org/apache/foo/TestBar.java", "b/test/foo/TestBar.java", "c/TestBar.scala"], fqn) \
        == "a/org/apache/foo/TestBar.java"
    assert find_source(["main/foo/TestBar.java", "src/test/foo/TestBar.java"], fqn) == "src/test/foo/TestBar.java"
    assert find_source(["x/foo/TestBar.java", "y/foo/TestBar.java"], fqn) == "x/foo/TestBar.java"
    assert find_source(["anything/TestBar.txt"], fqn) == "anything/TestBar.txt"
    assert find_source([], fqn) is None


def test_hunks_apply_at_the_nearest_match_ignoring_whitespace_and_skip_otherwise():
    hunk = (2, ["b", "c"], ["b", "X", "c"])
    assert apply_hunks(["a", "b", "c", "d"], [hunk]) == ["a", "b", "X", "c", "d"]
    assert apply_hunks(["z", "a", "b ", "c", "d"], [hunk]) == ["z", "a", "b", "X", "c", "d"]
    assert apply_hunks(["a", "q", "d"], [hunk]) == ["a", "q", "d"]
    assert apply_hunks([""], [(0, [], ["new", "file"])]) == ["new", "file", ""]


def test_minmax_tfidf_matches_scikit_learn():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [["alpha", "beta", "gamma"], ["beta", "delta"], ["gamma", "delta", "epsilon", "zeta"], ["eta", "foo$bar"]]
    queries = [["beta", "gamma", "unseen"], ["eta", "theta"], ["nothing", "matches"]]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([" ".join(d) for d in docs])
    expected = np.zeros(len(docs))
    for q in queries:
        s = cosine_similarity(vectorizer.transform([" ".join(q)]), matrix).ravel()
        s = (s - s.min()) / (s.max() - s.min()) if s.max() - s.min() > 1e-9 else np.full(len(docs), 0.5)
        expected = np.maximum(expected, s)
    terms = {}
    got = minmax_tfidf([term_counts(d, terms) for d in docs], [term_counts(q, terms) for q in queries], len(terms))
    assert np.allclose(got, expected)


def test_window_holds_the_latest_builds_within_the_run_budget():
    h = History(4)
    stages = np.array([60, 30, 20])
    for b, (cls, failed) in enumerate([([0, 1], [1, 0]), ([1, 2], [1, 0]), ([2, 3], [1, 1])]):
        h.observe(np.array(cls), np.array(failed, dtype=bool), np.ones(2), np.ones(2), np.empty(0, int), b)
    f = _features(h, stages, 4)
    assert f["runs"].tolist() == [0, 1, 2, 1] and f["fails"].tolist() == [0, 1, 1, 1]
    assert f["rows"][0].tolist() == [False, False, False, False]
    assert [lag for lag, _ in f["lags"]] == [1, 2, 3] and f["lags"][0][1].tolist() == [2, 3]


def synthetic_relevance(suite, seed=0):
    rng = np.random.default_rng(seed)
    offsets, cls, score = [0], [], []
    for t in range(len(suite.builds)):
        c = np.flatnonzero(rng.random(len(suite.names)) < 0.5)
        cls.append(c)
        score.append(rng.random(len(c)))
        offsets.append(offsets[-1] + len(c))
    return {"offsets": np.array(offsets), "cls": np.concatenate(cls), "score": np.concatenate(score)}


@pytest.mark.parametrize("name", ["DANTE", "DANTE-W", "DANTE-H"])
def test_faithful_dante_ignores_rows_it_never_ran(name):
    s = random_suite()
    stages = np.random.default_rng(2).integers(1, 30, len(s.builds))
    relevance = synthetic_relevance(s)
    scorer = dante_scorers(s, lambda: relevance, stages)[name]
    a = replay(s, "faithful", scorer, lam=0.4, pi=0.2, seed=3)
    hidden = ~a.executed(s)
    assert hidden.any()
    b = replay(perturb(s, hidden), "faithful", scorer, lam=0.4, pi=0.2, seed=3)
    assert np.array_equal(a.selected, b.selected) and np.array_equal(a.depth, b.depth)


def test_timed_dante_ignores_builds_unfinished_at_decision():
    s = random_suite()
    stages = np.random.default_rng(2).integers(1, 30, len(s.builds))
    scorer = dante_scorers(s, lambda: synthetic_relevance(s), stages)["DANTE"]
    start, finish = s.builds["start"].to_numpy(), s.builds["finish"].to_numpy()
    a = replay(s, "timed", scorer, lam=0.4, pi=0.2, seed=5)
    for u in (20, 45, 70):
        late = np.repeat(finish > start[u], np.diff(s.offsets))
        b = replay(perturb(s, late), "timed", scorer, lam=0.4, pi=0.2, seed=5)
        rows = np.repeat(start <= start[u], np.diff(s.offsets))
        assert np.array_equal(a.selected[rows], b.selected[rows])


@needs_data
def test_dante_reproduces_its_published_apfdc_exactly():
    r = dante_ref.recompute()
    assert len(r) >= 40 and r["project"].nunique() == 9
    assert np.abs(r["dante_released"] - r["dante_published"]).max() < 1e-9
    assert np.abs(r["latestfail_recency"] - r["latestfail_published"]).max() < 1e-9


@needs_data
def test_diff_tokens_match_dantes_processed_builds():
    meta = dante_ref.release_metadata()
    for path in dante_ref.processed_builds():
        _, run = dante_ref.target_run(meta, path.stem)
        project = path.stem.split("_build_")[0]
        diff = lrts.diff_path(project, run["pr_name"], run["build_id"]).read_text(encoding="utf-8", errors="ignore")
        ours = {p: source_tokens(t, posixpath.splitext(p)[1]) for p, t in diff_by_file(diff).items()}
        theirs = json.loads(path.read_text())["git_diff_data"]["tokenized_diff_by_file"]
        assert {p: set(t) for p, t in ours.items()} == {p: set(t) for p, t in theirs.items()}, path.stem


@needs_dante
def test_tfidf_from_dantes_tokens_matches_its_stored_relevance():
    checked = 0
    for path in dante_ref.processed_builds():
        d = json.loads(path.read_text())
        tests = [tc for tc in d["test_cases"].values() if tc.get("source_file_found")]
        if not tests or any(tc.get("tokens") is None for tc in tests):
            continue
        terms = {}
        corpus = [i for i, tc in enumerate(tests) if tc["tokens"]]
        queries = [term_counts(t, terms) for t in d["git_diff_data"]["tokenized_diff_by_file"].values() if t]
        got = np.zeros(len(tests))
        got[corpus] = minmax_tfidf([term_counts(tests[i]["tokens"], terms) for i in corpus], queries, len(terms))
        stored = [max([s["tfidf"] for s in tc["relevance_to_changed_files"].values()], default=0.0) for tc in tests]
        assert np.abs(got - np.array(stored)).max() < 1e-9, path.stem
        checked += 1
    assert checked > 0
