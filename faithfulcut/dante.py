import os
import posixpath
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from faithfulcut import lrts
from faithfulcut.ir import _corpora, _Repo
from faithfulcut.replay import History, Scorer
from faithfulcut.suite import Suite

VERSION = 1
PROJECTS = ("activemq", "hadoop", "hbase", "hive", "jackrabbit-oak", "james", "kafka", "karaf", "log4j")
WINDOW_RUNS = 100
LOOKBACK = 10
MIN_EVENTS = 20
WEIGHT_HISTORY, WEIGHT_RELEVANCE, RELEVANCE_POWER = 1.0, 0.5, 1.2
IGNORED_EXTENSIONS = {".xml", ".html", ".json", ".yaml", ".yml", ".properties", ".md", ".txt", ".gitignore",
                      ".gitattributes", ".dockerignore", ""}
KEYWORDS = {
    "if", "else", "for", "while", "return", "new", "try", "catch", "finally", "this", "self", "true", "false", "null",
    "none", "def", "class", "public", "private", "protected", "static", "void", "int", "string", "function", "var",
    "let", "const", "import", "from", "package", "extends", "implements", "interface", "enum", "func", "go", "defer",
    "chan", "select", "range", "type", "struct", "map", "error", "nil", "util", "io", "lang", "net", "org", "apache",
    "common", "utils", "base", "abstract", "helper",
}
TEST_DIRS = ("/test/", "/tests/", "/itest/", "/it/")
ANALYZER = re.compile(r"(?u)\b\w\w+\b")
HUNK = re.compile(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def _clean_java(code: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    cleaned = re.sub(r"//.*", "", cleaned)
    cleaned = re.sub(r"@\w+(\s*\(.*?\))?", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_scala(code: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    cleaned = re.sub(r"//.*", "", cleaned)
    cleaned = re.sub(r"@\w+(\s*\(.*?\))?", "", cleaned)
    cleaned = re.sub(r'"""[\s\S]*?"""', "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _clean_generic(code: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    cleaned = re.sub(r"//.*", "", cleaned)
    cleaned = re.sub(r"#.*", "", cleaned)
    cleaned = re.sub(r'"""[\s\S]*?"""', "", cleaned)
    cleaned = re.sub(r"'''[\s\S]*?'''", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _split(identifiers: list[str]) -> list[str]:
    out = []
    for ident in identifiers:
        for part in re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", ident)).split():
            out.extend(p.lower() for p in part.split("_") if len(p) > 2)
    return out


def _tokenize_java(cleaned: str) -> list[str]:
    out = _split(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_$]*\b", cleaned))
    for literal in re.findall(r'"([^"\\]*(\\.[^"\\]*)*)"', cleaned):
        if len(literal[0]) > 3:
            out.extend(w for w in re.findall(r"\b\w+\b", literal[0].lower()) if len(w) > 2)
    return out


def _tokenize_scala(cleaned: str) -> list[str]:
    return _split(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", cleaned))


def _tokenize_generic(cleaned: str) -> list[str]:
    return [t.lower() for t in re.findall(r"\b[a-zA-Z_]\w*\b", cleaned) if len(t) > 2]


LANGUAGES = {".java": (_clean_java, _tokenize_java), ".scala": (_clean_scala, _tokenize_scala)}


def source_tokens(code: str | None, extension: str) -> list[str]:
    """DANTE's cleaning and split-and-expand tokenization: distinct tokens, generic keywords removed."""
    if not code:
        return []
    clean, tokenize = LANGUAGES.get(extension, (_clean_generic, _tokenize_generic))
    cleaned = clean(code)
    return sorted(set(tokenize(cleaned)) - KEYWORDS) if cleaned else []


def diff_by_file(text: str) -> dict[str, str]:
    """DANTE's split of a diff into each code file's added and removed lines."""
    changed, current = {}, None
    for line in (text or "").splitlines():
        if line.startswith("diff --git a/"):
            parts = line.split(" b/", 1)
            code = len(parts) == 2 and posixpath.splitext(parts[1])[1].lower() not in IGNORED_EXTENSIONS
            current = parts[1] if code else None
        if current and ((line.startswith("+") and not line.startswith("+++"))
                        or (line.startswith("-") and not line.startswith("---"))):
            changed.setdefault(current, []).append(line[1:])
    return {path: "\n".join(lines) for path, lines in changed.items() if lines}


def find_source(candidates: list[str], fqn: str) -> str | None:
    """DANTE's match of a test class to a file named like it: most trailing package folders in common, test folders
    preferred, the first path in sorted order on ties."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    package = [p.lower() for p in fqn.split(".")[:-1]]
    best, best_score = None, -1.0
    for path in candidates:
        folders = posixpath.dirname(path).split("/") if "/" in path else ["."]
        depth = 0
        for i in range(1, min(len(package), len(folders)) + 1):
            if package[-i] != folders[-i].lower():
                break
            depth += 1
        score = depth + (0.5 if any(d in "/" + path.lower() for d in TEST_DIRS) else 0.0)
        if score > best_score:
            best, best_score = path, score
    return best


def _stem(path: str) -> str:
    return posixpath.splitext(posixpath.basename(path))[0].lower()


def _class_stem(fqn: str) -> str:
    return fqn.split(".")[-1].split("$")[0].lower()


def _index(files: dict[str, str]) -> dict[str, list[str]]:
    index = {}
    for path in sorted(files):
        if not any(part.startswith(".") for part in path.split("/")[:-1]):
            index.setdefault(_stem(path), []).append(path)
    return index


def _patches(text: str) -> list[dict]:
    """Old and new path (None when added or deleted) and hunks (old start, old lines, new lines) of each file."""
    files, current, in_hunk = [], None, False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current, in_hunk = {"old": None, "new": None, "hunks": []}, False
            files.append(current)
        elif current is None:
            continue
        elif in_hunk and line[:1] in (" ", "+", "-", ""):
            old, new = current["hunks"][-1][1:]
            if line[:1] != "+":
                old.append(line[1:])
            if line[:1] != "-":
                new.append(line[1:])
        elif m := HUNK.match(line):
            in_hunk = True
            current["hunks"].append((int(m.group(1)), [], []))
        elif not in_hunk:
            if line.startswith("--- "):
                current["old"] = None if line[4:] == "/dev/null" else line[4:].removeprefix("a/")
            elif line.startswith("+++ "):
                current["new"] = None if line[4:] == "/dev/null" else line[4:].removeprefix("b/")
            elif line.startswith("rename from "):
                current["old"] = line[len("rename from "):]
            elif line.startswith("rename to "):
                current["new"] = line[len("rename to "):]
    return files


def apply_hunks(lines: list[str], hunks: list) -> list[str]:
    """Apply hunks at the nearest position where their old lines match, ignoring whitespace; skip any that do not."""
    out, pos = [], 0
    for start, old, new in hunks:
        expected = start - 1 if old else start
        last = len(lines) - len(old)
        at = None
        for i in sorted(range(pos, last + 1), key=lambda i: abs(i - expected)):
            if all(lines[i + j].split() == o.split() for j, o in enumerate(old)):
                at = i
                break
        if at is None:
            continue
        out.extend(lines[pos:at])
        out.extend(new)
        pos = at + len(old)
    return out + lines[pos:]


def term_counts(tokens: list[str], terms: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Term ids and counts of a token list as scikit-learn's default analyzer splits the space-joined tokens."""
    c = Counter(terms.setdefault(w, len(terms)) for w in ANALYZER.findall(" ".join(tokens).lower()))
    return np.fromiter(c.keys(), np.int64, len(c)), np.fromiter(c.values(), np.float64, len(c))


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _Relevance:
    """Per-build DANTE relevance: max over changed code files of the min-max-normalized TF-IDF cosine between the
    file's diff tokens and each class's test source, at the base commit with the build's diff applied."""

    def __init__(self, project: str, stems: set[str], indexes: int = 64):
        self.repo = _Repo(project, keep=lambda path: _stem(path) in stems)
        self.terms: dict[str, int] = {}
        self.vectors: dict[str, tuple] = {}
        self.indexes: OrderedDict = OrderedDict()
        self.max_indexes = indexes

    def close(self):
        self.repo.close()

    def vector(self, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return term_counts(tokens, self.terms)

    def index(self, sha: str, files: dict[str, str]) -> dict[str, list[str]]:
        if sha not in self.indexes:
            self.indexes[sha] = _index(files)
            if len(self.indexes) > self.max_indexes:
                self.indexes.popitem(last=False)
        self.indexes.move_to_end(sha)
        return self.indexes[sha]

    def doc(self, oid: str, path: str) -> tuple | None:
        if oid not in self.vectors:
            tokens = source_tokens(_normalize_newlines(self.repo.text(oid) or ""), posixpath.splitext(path)[1])
            self.vectors[oid] = self.vector(tokens) if tokens else None
        return self.vectors[oid]

    def build(self, sha: str, files: dict[str, str], diff: str, classes: list[str]) -> np.ndarray:
        index = self.index(sha, files)
        patched, removed, added = {}, set(), {}
        for f in _patches(diff):
            if f["old"] is not None and f["old"] != f["new"]:
                removed.add(f["old"])
            if f["new"] is not None:
                patched[f["new"]] = f
                if f["new"] not in files or f["old"] != f["new"]:
                    added.setdefault(_stem(f["new"]), []).append(f["new"])
        local, rows, docs = {}, [], []
        for i, fqn in enumerate(classes):
            stem = _class_stem(fqn)
            candidates = index.get(stem, [])
            if removed or stem in added:
                candidates = sorted({p for p in candidates if p not in removed} | set(added.get(stem, ())))
            path = find_source(candidates, fqn)
            if path is None:
                continue
            if path in patched:
                if path not in local:
                    f = patched[path]
                    old = self.repo.text(files[f["old"]]) if f["old"] in files else ""
                    lines = apply_hunks(_normalize_newlines(old or "").split("\n"), f["hunks"])
                    tokens = source_tokens("\n".join(lines), posixpath.splitext(path)[1])
                    local[path] = self.vector(tokens) if tokens else None
                vec = local[path]
            else:
                vec = self.doc(files[path], path)
            if vec is not None:
                rows.append(i)
                docs.append(vec)
        queries = [self.vector(t) for path, text in diff_by_file(diff).items()
                   if (t := source_tokens(text, posixpath.splitext(path)[1]))]
        out = np.zeros(len(classes))
        if docs and queries:
            out[rows] = minmax_tfidf(docs, queries, len(self.terms))
        return out


def minmax_tfidf(docs: list[tuple], queries: list[tuple], n_terms: int) -> np.ndarray:
    """For each document, the max over queries of its TF-IDF cosine min-max normalized across documents (0.5 when
    all are equal); TF-IDF as scikit-learn's defaults fitted on the documents."""
    counts = np.concatenate([c for _, c in docs])
    ids = np.concatenate([i for i, _ in docs])
    indptr = np.concatenate([[0], np.cumsum([len(i) for i, _ in docs])])
    matrix = sparse.csr_matrix((counts, ids, indptr), shape=(len(docs), n_terms))
    df = np.bincount(ids, minlength=n_terms)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    matrix.data *= idf[matrix.indices]
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    best = np.zeros(len(docs))
    for q_ids, q_counts in queries:
        keep = df[q_ids] > 0
        q = np.zeros(n_terms)
        q[q_ids[keep]] = q_counts[keep] * idf[q_ids[keep]]
        q_norm = np.linalg.norm(q)
        s = np.divide(matrix @ q, norms * q_norm, out=np.zeros(len(docs)), where=(norms > 0) & (q_norm > 0))
        spread = s.max() - s.min()
        best = np.maximum(best, (s - s.min()) / spread if spread > 1e-9 else np.full(len(docs), 0.5))
    return best


def _relevance_path(project: str) -> Path:
    return Path(os.environ["FAITHFULCUT_CACHE"]) / f"dante-v{VERSION}" / f"{project}.npz"


def load_relevance(suite: Suite) -> dict[str, np.ndarray]:
    """DANTE relevance of every class seen before each build starts, cached per project."""
    path = _relevance_path(suite.project)
    if path.exists():
        return dict(np.load(path))
    rel = _Relevance(suite.project, {_class_stem(n) for n in suite.names})
    offsets, cls, scores, missing = [0], [], [], np.zeros(len(suite.builds), dtype=bool)
    try:
        for t, classes in _corpora(suite, "seen"):
            b = suite.builds.iloc[t]
            files = rel.repo.tree(b["trunk_sha"])
            missing[t] = files is None
            if files is None or len(classes) == 0:
                offsets.append(offsets[-1])
                continue
            diff = lrts.diff_path(suite.project, b["pr_name"], b["build_id"]).read_text(encoding="utf-8",
                                                                                       errors="ignore")
            s = rel.build(b["trunk_sha"], files, diff, suite.names[classes].tolist())
            keep = s > 0
            cls.append(classes[keep].astype(np.int32))
            scores.append(s[keep].astype(np.float32))
            offsets.append(offsets[-1] + int(keep.sum()))
    finally:
        rel.close()
    out = {"offsets": np.array(offsets), "cls": np.concatenate(cls + [np.empty(0, np.int32)]),
           "score": np.concatenate(scores + [np.empty(0, np.float32)]), "missing": missing}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def sticky_probabilities(prev: np.ndarray, failed: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """DANTE's P(fail | sticky) for look-back windows 1..LOOKBACK, as its code computes it: a row is sticky for n
    when any of the last n rows (across classes, in window order) has a failure as its class's previous outcome."""
    p = np.zeros(LOOKBACK)
    if len(prev) == 0:
        return p
    idx = np.arange(len(prev))
    last = np.maximum.accumulate(np.where(prev, idx, -1))
    distance = np.minimum(np.where(last >= 0, idx - last, LOOKBACK), LOOKBACK)
    w = np.ones(len(prev)) if weight is None else weight
    counts = np.cumsum(np.bincount(distance, minlength=LOOKBACK + 1)[:LOOKBACK])
    total = np.cumsum(np.bincount(distance, weights=w, minlength=LOOKBACK + 1)[:LOOKBACK])
    fails = np.cumsum(np.bincount(distance, weights=w * failed, minlength=LOOKBACK + 1)[:LOOKBACK])
    enough = counts >= MIN_EVENTS
    p[enough] = fails[enough] / total[enough]
    return p


def stickiness(cls: np.ndarray, failed: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """sticky_probabilities for rows in window order, each row's previous outcome taken from its class's last row."""
    order = np.argsort(cls, kind="stable")
    prev_sorted = np.zeros(len(cls), dtype=bool)
    prev_sorted[1:] = (cls[order][1:] == cls[order][:-1]) & failed[order][:-1]
    prev = np.empty(len(cls), dtype=bool)
    prev[order] = prev_sorted
    return sticky_probabilities(prev, failed, weight)


def stage_counts(suite: Suite) -> np.ndarray:
    runs = lrts.load_runs(suite.project).groupby(["pr_name", "build_id"]).size()
    return runs.reindex(pd.MultiIndex.from_arrays([suite.builds["pr_name"], suite.builds["build_id"]])).to_numpy()


def _previous(h: History, n: int) -> dict:
    """Each observed build's rows: whether the class failed at its previous observation, and that observation's
    sequence number (-1 if none); kept incrementally as feedback arrives."""
    state = h.derived.setdefault("dante-previous", {"seq": -1, "last_seq": np.full(n, -1), "failed": np.zeros(n, bool),
                                                    "rows": {}})
    for _, c, f, _, seq in h.recent:
        if seq > state["seq"]:
            state["rows"][seq] = (state["failed"][c], state["last_seq"][c])
            state["last_seq"][c], state["failed"][c], state["seq"] = seq, f, seq
    oldest = h.recent[0][4] if h.recent else 0
    for seq in [s for s in state["rows"] if s < oldest]:
        del state["rows"][seq]
    return state["rows"]


def _features(h: History, stages: np.ndarray, n: int) -> dict:
    """Window counts, previous outcomes and recent failures from the observed builds holding the latest WINDOW_RUNS
    stage runs; recomputed only after new feedback, stickiness only when a scorer asks for it."""
    cached = h.derived.get("dante")
    if cached is not None and cached[0] == h.observed:
        return cached[1]
    previous = _previous(h, n)
    entries = list(h.recent)
    k = runs = 0
    for build, *_ in reversed(entries):
        runs += stages[build]
        if runs > WINDOW_RUNS and k:
            break
        k += 1
    window = entries[len(entries) - k:]
    start = window[0][4] if window else 0
    cls = np.concatenate([e[1] for e in window] + [np.empty(0, np.int64)]).astype(np.int64)
    failed = np.concatenate([e[2] for e in window] + [np.empty(0, bool)]).astype(bool)
    weight = np.concatenate([e[3] for e in window] + [np.empty(0)]).astype(float)
    prev_failed = np.concatenate([previous[e[4]][0] for e in window] + [np.empty(0, bool)])
    prev_seq = np.concatenate([previous[e[4]][1] for e in window] + [np.empty(0, np.int64)])
    features = {
        "runs": np.bincount(cls, minlength=n).astype(float), "fails": np.bincount(cls, weights=failed, minlength=n),
        "wruns": np.bincount(cls, weights=weight, minlength=n),
        "wfails": np.bincount(cls, weights=weight * failed, minlength=n),
        "rows": (prev_failed & (prev_seq >= start), failed, weight),
        "lags": [(lag, e[1][e[2]]) for lag, e in enumerate(reversed(entries[-LOOKBACK:]), 1)],
    }
    h.derived["dante"] = (h.observed, features)
    return features


def _stickiness(f: dict, weighted: bool) -> np.ndarray:
    key = "pw" if weighted else "p"
    if key not in f:
        prev, failed, weight = f["rows"]
        f[key] = sticky_probabilities(prev, failed, weight if weighted else None)
    return f[key]


def dante_scorers(suite: Suite, relevance: Callable[[], dict[str, np.ndarray]] | None,
                  stages: np.ndarray) -> dict[str, Scorer]:
    """DANTE-CCH: (failure rate + stickiness + relevance) times failures over predicted duration, from the
    observed window. DANTE-H omits the relevance term; -W weights observations by audit propensity. `relevance` is
    called when first needed."""
    n = len(suite.names)

    def make(relevant: bool, weighted: bool) -> Scorer:
        def score(h: History, t: int) -> np.ndarray:
            f = _features(h, stages, n)
            runs, fails = (f["wruns"], f["wfails"]) if weighted else (f["runs"], f["fails"])
            p = _stickiness(f, weighted)
            value = np.divide(fails, runs, out=np.zeros(n), where=runs > 0)
            for lag, failing in f["lags"]:
                value[failing] += p[lag - 1] / lag
            value *= WEIGHT_HISTORY
            if relevant:
                r, rel = relevance(), np.zeros(n)
                part = slice(r["offsets"][t], r["offsets"][t + 1])
                rel[r["cls"][part]] = r["score"][part]
                value += WEIGHT_RELEVANCE * rel ** RELEVANCE_POWER
            return -value * np.maximum(fails, 0.1) / np.maximum(h.last_duration, 0.001)

        return score

    out = {"DANTE-H": make(False, False), "DANTE-H-W": make(False, True)}
    if relevance is not None:
        out.update({"DANTE": make(True, False), "DANTE-W": make(True, True)})
    return out
