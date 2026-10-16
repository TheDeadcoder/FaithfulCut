import heapq
import os
import re
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from scipy import sparse

from faithfulcut import lrts
from faithfulcut.suite import Suite

VERSION = 1
REPOS = {
    "activemq": "apache/activemq", "hadoop": "apache/hadoop", "hbase": "apache/hbase", "hive": "apache/hive",
    "jackrabbit-oak": "apache/jackrabbit-oak", "james": "apache/james-project", "kafka": "apache/kafka",
    "karaf": "apache/karaf", "log4j": "apache/logging-log4j2", "tvm": "apache/tvm",
}
SOURCE_SUFFIXES = (".java", ".scala", ".py", ".cc", ".c", ".groovy")
WORD = re.compile(r"[A-Za-z]+")
CAMEL = re.compile(r"([A-Z])(?=[a-z])")
SPLIT = re.compile(r"split[0-9]+")


def repo_dir(project: str) -> Path:
    return Path(os.environ["FAITHFULCUT_CACHE"]) / "repos" / f"{project}.git"


def tokens(text: str) -> list[str]:
    out = []
    for word in WORD.findall(text):
        out.extend(t for t in CAMEL.sub(r"*\1", word).lower().split("*") if len(t) > 1)
    return out


def lrts_pattern(project: str, testclass: str) -> str:
    name = testclass.split("$")[0]
    if project == "tvm":
        parts = name.split(".")
        return "/".join(parts[-2:] if parts[-1].islower() else parts[-2:-1]) + "."
    if "." in testclass:
        pattern = "/".join(name.split(".")[-2:]) + "."
        if project == "hive" and SPLIT.search(pattern):
            pattern = "/" + pattern.split("/")[-1]
        return pattern
    return f"/Test{name}.cc" if project == "hadoop" else name


def is_test_source(path: str) -> bool:
    return path.endswith(SOURCE_SUFFIXES) and "test" in path and not any(p.startswith(".") for p in path.split("/"))


def parse_diff(text: str) -> dict[str, tuple[str | None, str, str]]:
    """Map each new path (or deleted old path) to (old path or None if added, added text, removed text)."""
    files, current, in_hunk = {}, None, False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current, in_hunk = {"old": None, "new": None, "added": [], "removed": []}, False
            files[len(files)] = current
        elif current is None:
            continue
        elif line.startswith("@@"):
            in_hunk = True
        elif not in_hunk:
            if line.startswith("--- "):
                current["old"] = None if line[4:] == "/dev/null" else line[4:].removeprefix("a/")
            elif line.startswith("+++ "):
                current["new"] = None if line[4:] == "/dev/null" else line[4:].removeprefix("b/")
            elif line.startswith("rename from "):
                current["old"] = line[len("rename from "):]
            elif line.startswith("rename to "):
                current["new"] = line[len("rename to "):]
        elif line.startswith("+"):
            current["added"].append(line[1:])
        elif line.startswith("-"):
            current["removed"].append(line[1:])
    out = {}
    for f in files.values():
        path = f["new"] or f["old"]
        if path is not None:
            out[path] = (f["old"], "\n".join(f["added"]), "\n".join(f["removed"]) if f["new"] else None)
    return out


class _Repo:
    def __init__(self, project: str, trees: int = 64, keep=is_test_source):
        self.project = project
        self.keep = keep
        self.git = repo_dir(project)
        self.batch = subprocess.Popen(["git", "--git-dir", str(self.git), "cat-file", "--batch"],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.terms: dict[str, int] = {}
        self.blobs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.trees: OrderedDict = OrderedDict()
        self.max_trees = trees

    def close(self):
        self.batch.stdin.close()
        self.batch.wait()

    def counts(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        c = Counter(self.terms.setdefault(t, len(self.terms)) for t in tokens(text))
        return np.fromiter(c.keys(), np.int64, len(c)), np.fromiter(c.values(), np.float64, len(c))

    def text(self, oid: str) -> str | None:
        self.batch.stdin.write(f"{oid}\n".encode())
        self.batch.stdin.flush()
        header = self.batch.stdout.readline().split()
        if len(header) < 3:
            return None
        return self.batch.stdout.read(int(header[2]) + 1)[:-1].decode("utf-8", errors="ignore")

    def blob(self, oid: str) -> tuple[np.ndarray, np.ndarray]:
        if oid not in self.blobs:
            content = self.text(oid)
            self.blobs[oid] = (np.empty(0, np.int64), np.empty(0)) if content is None else self.counts(content)
        return self.blobs[oid]

    def tree(self, sha: str) -> dict[str, str] | None:
        if sha not in self.trees:
            listing = subprocess.run(["git", "--git-dir", str(self.git), "ls-tree", "-r", "-z", "--full-tree", sha],
                                     capture_output=True)
            files = None
            if listing.returncode == 0:
                files = {}
                for entry in listing.stdout.decode("utf-8", errors="ignore").split("\0"):
                    if not entry:
                        continue
                    meta, path = entry.split("\t", 1)
                    mode, kind, oid = meta.split()
                    if kind == "blob" and self.keep(path):
                        files[path] = oid
            self.trees[sha] = files
            if len(self.trees) > self.max_trees:
                self.trees.popitem(last=False)
        self.trees.move_to_end(sha)
        return self.trees[sha]


def _documents(repo: _Repo, classes: list[str], files: dict[str, str], changes: dict) -> list:
    """Term counts per class: matched test files at the base commit with the build's diff applied as count deltas."""
    content = {path: ("blob", oid) for path, oid in files.items()}
    for path, (old, added, removed) in changes.items():
        if removed is None:
            content.pop(path, None)
            continue
        if not is_test_source(path):
            continue
        base = ("blob", files[old]) if old in files else None
        if old is not None and old != path:
            content.pop(old, None)
        content[path] = ("delta", base, added, removed)
    by_stem: dict[str, list[str]] = {}
    for path in content:
        by_stem.setdefault(path.rsplit("/", 1)[-1].split(".")[0], []).append(path)
    docs = []
    for c in classes:
        pattern = lrts_pattern(repo.project, c)
        candidates = by_stem.get(pattern.rsplit("/", 1)[-1].split(".")[0], []) if "/" in pattern else content
        ids, counts = [], []
        for path in candidates:
            if pattern not in path:
                continue
            entry = content[path]
            if entry[0] == "blob":
                i, n = repo.blob(entry[1])
                ids.append(i), counts.append(n)
            else:
                _, base, added, removed = entry
                if base is not None:
                    i, n = repo.blob(base[1])
                    ids.append(i), counts.append(n)
                i, n = repo.counts(added)
                ids.append(i), counts.append(n)
                i, n = repo.counts(removed)
                ids.append(i), counts.append(-n)
        docs.append((np.concatenate(ids) if ids else np.empty(0, np.int64),
                     np.concatenate(counts) if counts else np.empty(0)))
    return docs


def tfidf(docs: list, query: tuple[np.ndarray, np.ndarray], n_terms: int) -> np.ndarray:
    """Cosine similarity of the query to each document under sklearn's default TfidfVectorizer weighting."""
    rows = np.repeat(np.arange(len(docs)), [len(d[0]) for d in docs])
    ids = np.concatenate([d[0] for d in docs] + [np.empty(0, np.int64)])
    vals = np.concatenate([d[1] for d in docs] + [np.empty(0)])
    matrix = sparse.csr_matrix((vals, (rows, ids)), shape=(len(docs), n_terms))
    matrix.data = np.maximum(matrix.data, 0)
    matrix.eliminate_zeros()
    df = np.bincount(matrix.indices, minlength=n_terms)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    matrix.data *= idf[matrix.indices]
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    q = np.zeros(n_terms)
    np.add.at(q, query[0], query[1])
    q = np.where(df > 0, q * idf, 0.0)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return np.zeros(len(docs))
    return np.divide(matrix @ q, norms * q_norm, out=np.zeros(len(docs)), where=norms > 0)


def _corpora(suite: Suite, corpus: str):
    start, finish = suite.builds["start"].to_numpy(), suite.builds["finish"].to_numpy()
    seen = np.zeros(len(suite.names), dtype=bool)
    pending = []
    for t in range(len(suite.builds)):
        if corpus == "report":
            yield t, np.unique(suite.cls[suite.rows(t)])
            continue
        while pending and pending[0][0] <= start[t]:
            seen[suite.cls[suite.rows(heapq.heappop(pending)[1])]] = True
        yield t, np.flatnonzero(seen)
        heapq.heappush(pending, (finish[t], t))


def _ir_path(project: str, corpus: str) -> Path:
    return Path(os.environ["FAITHFULCUT_CACHE"]) / f"ir-v{VERSION}" / f"{project}-{corpus}.npz"


def load_ir(suite: Suite, corpus: str = "seen") -> dict[str, np.ndarray]:
    """Per-build TF-IDF scores of the diff against each corpus class's test sources.

    corpus "seen": classes observed in builds finished before the build starts; "report": the build's own classes.
    """
    path = _ir_path(suite.project, corpus)
    if path.exists():
        return dict(np.load(path))
    repo = _Repo(suite.project)
    offsets, cls, scores, missing = [0], [], [], np.zeros(len(suite.builds), dtype=bool)
    try:
        for t, classes in _corpora(suite, corpus):
            b = suite.builds.iloc[t]
            files = repo.tree(b["trunk_sha"])
            missing[t] = files is None
            if files is None or len(classes) == 0:
                offsets.append(offsets[-1])
                continue
            diff = lrts.diff_path(suite.project, b["pr_name"], b["build_id"]).read_text(errors="ignore")
            docs = _documents(repo, suite.names[classes].tolist(), files, parse_diff(diff))
            query = repo.counts(diff)
            s = tfidf(docs, query, len(repo.terms))
            keep = s > 0
            cls.append(classes[keep].astype(np.int32))
            scores.append(s[keep].astype(np.float32))
            offsets.append(offsets[-1] + int(keep.sum()))
    finally:
        repo.close()
    out = {"offsets": np.array(offsets), "cls": np.concatenate(cls + [np.empty(0, np.int32)]),
           "score": np.concatenate(scores + [np.empty(0, np.float32)]), "missing": missing}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out
