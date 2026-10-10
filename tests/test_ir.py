import re
from collections import Counter

import numpy as np
import pytest

from faithfulcut.ir import _documents, lrts_pattern, parse_diff, tfidf, tokens


def lrts_low_tokenization(text):
    out = []
    for t in re.findall(r"[A-Za-z]+", text):
        new = ""
        for i in range(len(t) - 1):
            new += ("*" + t[i]) if (t[i].isupper() and t[i + 1].islower()) else t[i]
        new += t[-1]
        out += [x for x in new.lower().split("*") if len(x) > 0]
    return out


@pytest.mark.parametrize("text", [
    "public void testSplitBrain() { HTTPServer s = new HTTPServer(8080); }",
    "assertEquals(xMLParser, a1b2C3dEf); __init__ ZkTest ABCdef aB Ab",
    "def test_op_level1(): return np.array([1, 2]).sum()  # Ünïcode X",
])
def test_tokens_follow_lrts_tokenization_and_drop_single_letters(text):
    assert tokens(text) == [t for t in lrts_low_tokenization(text) if len(t) > 1]


@pytest.mark.parametrize("project, testclass, pattern", [
    ("kafka", "org.apache.kafka.clients.FooTest$Inner", "clients/FooTest."),
    ("hive", "org.apache.hadoop.hive.cli.split2.TestCliDriver", "/TestCliDriver."),
    ("tvm", "tests.python.relay.test_op_level1", "relay/test_op_level1."),
    ("tvm", "tests.python.relay.test_op_level1.TestThing", "test_op_level1."),
    ("hadoop", "Sample", "/TestSample.cc"),
    ("hbase", "Sample", "Sample"),
])
def test_lrts_path_patterns(project, testclass, pattern):
    assert lrts_pattern(project, testclass) == pattern


DIFF = """diff --git a/src/test/AlphaTest.java b/src/test/AlphaTest.java
index 1..2 100644
--- a/src/test/AlphaTest.java
+++ b/src/test/AlphaTest.java
@@ -1,3 +1,3 @@
 class AlphaTest {
--- removed dashes
+++ added pluses
diff --git a/src/test/BetaTest.java b/src/test/BetaTest.java
new file mode 100644
--- /dev/null
+++ b/src/test/BetaTest.java
@@ -0,0 +1 @@
+class BetaTest {}
diff --git a/src/test/GammaTest.java b/src/test/GammaTest.java
deleted file mode 100644
--- a/src/test/GammaTest.java
+++ /dev/null
@@ -1 +0,0 @@
-class GammaTest {}
diff --git a/src/test/OldTest.java b/src/test/NewTest.java
similarity index 90%
rename from src/test/OldTest.java
rename to src/test/NewTest.java
--- a/src/test/OldTest.java
+++ b/src/test/NewTest.java
@@ -1 +1 @@
-class OldTest {}
+class NewTest {}
"""


def test_parse_diff_handles_edits_additions_deletions_renames_and_dashed_lines():
    files = parse_diff(DIFF)
    assert files["src/test/AlphaTest.java"] == ("src/test/AlphaTest.java", "++ added pluses", "-- removed dashes")
    assert files["src/test/BetaTest.java"] == (None, "class BetaTest {}", "")
    assert files["src/test/GammaTest.java"][2] is None
    assert files["src/test/NewTest.java"] == ("src/test/OldTest.java", "class NewTest {}", "class OldTest {}")


class FakeRepo:
    def __init__(self, project, blobs):
        self.project, self.contents, self.terms = project, blobs, {}

    def counts(self, text):
        c = Counter(self.terms.setdefault(t, len(self.terms)) for t in tokens(text))
        return np.fromiter(c.keys(), np.int64, len(c)), np.fromiter(c.values(), float, len(c))

    def blob(self, oid):
        return self.counts(self.contents[oid])


def as_counter(repo, doc):
    names = {i: t for t, i in repo.terms.items()}
    c = Counter()
    for i, n in zip(*doc):
        c[names[i]] += n
    return +c


def test_documents_apply_the_diff_as_term_count_deltas():
    base = {"a": "class AlphaTest {\n-- removed dashes\n}", "g": "class GammaTest {}", "o": "class OldTest {}"}
    files = {"src/test/AlphaTest.java": "a", "src/test/GammaTest.java": "g", "src/test/OldTest.java": "o"}
    repo = FakeRepo("kafka", base)
    classes = ["x.test.AlphaTest", "x.test.BetaTest", "x.test.GammaTest", "x.test.NewTest", "x.test.OldTest"]
    docs = _documents(repo, classes, files, parse_diff(DIFF))
    final = {
        "x.test.AlphaTest": "class AlphaTest {\n++ added pluses\n}", "x.test.BetaTest": "class BetaTest {}",
        "x.test.GammaTest": "", "x.test.NewTest": "class NewTest {}", "x.test.OldTest": "",
    }
    for c, doc in zip(classes, docs):
        assert as_counter(repo, doc) == Counter(tokens(final[c])), c


def test_tfidf_matches_sklearn_cosine_similarity():
    sklearn_text = pytest.importorskip("sklearn.feature_extraction.text")
    pairwise = pytest.importorskip("sklearn.metrics.pairwise")
    rng = np.random.default_rng(0)
    vocab = [f"term{chr(97 + i)}" for i in range(12)]
    texts = [" ".join(rng.choice(vocab, rng.integers(0, 15))) for _ in range(9)]
    query_text = " ".join(rng.choice(vocab, 10)) + " unseenword"
    repo = FakeRepo("kafka", {})
    docs = [repo.counts(t) for t in texts]
    query = repo.counts(query_text)
    ours = tfidf(docs, query, len(repo.terms))
    model = sklearn_text.TfidfVectorizer()
    reference = pairwise.cosine_similarity(model.fit_transform(texts), model.transform([query_text])).ravel()
    assert np.allclose(ours, reference)
