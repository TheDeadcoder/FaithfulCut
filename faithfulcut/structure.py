import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np

from faithfulcut.ir import _Repo, is_test_source, lrts_pattern
from faithfulcut.suite import Suite

VERSION = 1
UNKNOWN = 255
NO_MODULES = ("tvm",)
SNAPSHOTS = 16
BUILD_FILES = ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
               "gradle.properties")
SOURCE_PARTS = {"main", "test", "tests", "it", "java", "scala", "groovy", "kotlin", "resources", "testFixtures"}
TVM_ROOTS = ("tests/python/", "python/tvm/", "include/tvm/", "src/")
GRADLE_PROJECT = re.compile(r"^project\(':([^']+)'\)", re.MULTILINE)
GRADLE_DEPENDENCY = re.compile(r"project\(':([^']+)'\)")


def is_build_file(path: str) -> bool:
    return path.rsplit("/", 1)[-1] in BUILD_FILES or path.startswith("gradle/")


def _keep(path: str) -> bool:
    return is_test_source(path) or (is_build_file(path) and "/src/" not in f"/{path}")


def dirs(path: str, project: str) -> tuple[str, ...]:
    """The path's directories with source roots removed, so a class and its test share their package."""
    if project == "tvm":
        root = next((r for r in TVM_ROOTS if path.startswith(r)), "")
        return tuple(path[len(root):].split("/")[:-1])
    parts = path.split("/")
    if "src" in parts:
        i = parts.index("src")
        j = i + 1
        while j < min(i + 3, len(parts) - 1) and parts[j] in SOURCE_PARTS:
            j += 1
        parts = parts[:i] + parts[j:]
    return tuple(parts[:-1])


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element, name: str):
    return next((c for c in element if _local(c.tag) == name), None)


def maven_module(text: str) -> tuple[str | None, list[str]]:
    """A POM's artifactId, and the artifactIds it depends on: its parent and its dependencies, profiles included."""
    try:
        project = ET.fromstring(text)
    except ET.ParseError:
        return None, []
    own = _child(project, "artifactId")
    needs = []
    parent = _child(project, "parent")
    if parent is not None and _child(parent, "artifactId") is not None:
        needs.append(_child(parent, "artifactId").text)
    holders = [project] + [p for profiles in project if _local(profiles.tag) == "profiles" for p in profiles]
    for holder in holders:
        block = _child(holder, "dependencies")
        for dep in block if block is not None else []:
            if _child(dep, "artifactId") is not None:
                needs.append(_child(dep, "artifactId").text)
    return (own.text if own is not None else None), [n for n in needs if n]


def gradle_modules(text: str) -> dict[str, set[str]]:
    """Kafka-style root build.gradle: each `project(':a:b') {` block's directory and the project directories it uses."""
    marks = [(m.start(), m.group(1)) for m in GRADLE_PROJECT.finditer(text)]
    out = {}
    for (begin, name), end in zip(marks, [m[0] for m in marks[1:]] + [len(text)]):
        uses = {d.replace(":", "/") for d in GRADLE_DEPENDENCY.findall(text[begin:end])} - {name.replace(":", "/")}
        out.setdefault(name.replace(":", "/"), set()).update(uses)
    return out


class _Snapshot:
    """Module graph, test paths and directory prefixes at one commit."""

    def __init__(self, repo: _Repo, files: dict[str, str], names: np.ndarray, project: str, parsed: dict):
        self.project = project
        tests = [p for p in files if is_test_source(p)]
        builds = [p for p in files if is_build_file(p)]
        uses: dict[str, set[str]] = {}
        if project != "tvm":
            poms = [p for p in builds if p.endswith("pom.xml")]
            artifacts = {}
            for path in poms:
                if files[path] not in parsed:
                    parsed[files[path]] = maven_module(repo.text(files[path]) or "")
                own, needs = parsed[files[path]]
                root = path.rsplit("/", 1)[0] if "/" in path else ""
                uses[root] = set(needs)
                if own:
                    artifacts[own] = root
            uses = {r: {artifacts[a] for a in needs if a in artifacts} - {r} for r, needs in uses.items()}
            if "build.gradle" in files:
                if files["build.gradle"] not in parsed:
                    parsed[files["build.gradle"]] = gradle_modules(repo.text(files["build.gradle"]) or "")
                for root, needs in parsed[files["build.gradle"]].items():
                    uses.setdefault(root, set()).update(needs | {""})
                for p in tests:
                    if "/src/" in p:
                        uses.setdefault(p.split("/src/")[0], {""})
                uses.setdefault("", set())
        self.roots = set(uses)
        self.dependents: dict[str, set[str]] = {r: set() for r in uses}
        for root, needs in uses.items():
            for need in needs:
                self.dependents.setdefault(need, set()).add(root)
        by_stem: dict[str, list[str]] = {}
        for path in sorted(tests):
            by_stem.setdefault(path.rsplit("/", 1)[-1].split(".")[0], []).append(path)
        self.prefixes: dict[tuple, int] = {}
        depth = max([len(dirs(p, project)) for p in tests] + [0]) + 1
        self.ancestors = np.full((len(names), depth), -1, dtype=np.int64)
        self.module = np.full(len(names), "", dtype=object)
        self.known = np.zeros(len(names), dtype=bool)
        for c, name in enumerate(names):
            pattern = lrts_pattern(project, name)
            stem = pattern.rsplit("/", 1)[-1].split(".")[0]
            path = next((p for p in by_stem.get(stem, []) if pattern in p), None)
            if path is None:
                continue
            self.known[c] = True
            self.module[c] = self.module_of(path)
            d = dirs(path, project)
            for k in range(len(d) + 1):
                self.ancestors[c, k] = self.prefixes.setdefault(d[:len(d) - k], len(self.prefixes))

    def module_of(self, path: str) -> str | None:
        parts = path.split("/")[:-1]
        for k in range(len(parts), -1, -1):
            root = "/".join(parts[:k])
            if root in self.roots:
                return root
        return None

    def distances(self, changed: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Per class: steps from the changed modules along reverse dependencies, and directory levels from the test
        up to the nearest ancestor shared with a changed file; UNKNOWN where undefined."""
        module = np.full(len(self.known), UNKNOWN, dtype=np.uint8)
        path = np.full(len(self.known), UNKNOWN, dtype=np.uint8)
        if not changed:
            return module, path
        start = set()
        for p in changed:
            root = self.module_of(p)
            if root is not None and (root != "" or is_build_file(p)):
                start.add(root)
        reach, queue = {r: 0 for r in start}, deque(start)
        while queue:
            r = queue.popleft()
            for d in self.dependents.get(r, ()):
                if d not in reach:
                    reach[d] = reach[r] + 1
                    queue.append(d)
        for c in np.flatnonzero(self.known):
            if self.module[c] in reach:
                module[c] = min(reach[self.module[c]], UNKNOWN - 1)
        marked = np.zeros(len(self.prefixes) + 1, dtype=bool)
        for p in changed:
            d = dirs(p, self.project)
            for k in range(len(d) + 1):
                i = self.prefixes.get(d[:k])
                if i is not None:
                    marked[i] = True
        hit = marked[self.ancestors]
        found = hit.any(axis=1) & self.known
        path[found] = np.minimum(hit[found].argmax(axis=1), UNKNOWN - 1)
        return module, path


def _structure_path(key: str) -> Path:
    return Path(os.environ["FAITHFULCUT_CACHE"]) / f"structure-v{VERSION}" / f"{key}.npz"


def load_structure(suite: Suite) -> dict[str, np.ndarray]:
    """Per build and class, module and path distance from the build's changed files at its base commit (UNKNOWN for
    classes without a matched test source, and for builds whose base commit is missing)."""
    path = _structure_path(suite.key)
    if path.exists():
        return dict(np.load(path))
    n, m = len(suite.builds), len(suite.names)
    module = np.full((n, m), UNKNOWN, dtype=np.uint8)
    distance = np.full((n, m), UNKNOWN, dtype=np.uint8)
    repo = _Repo(suite.project, trees=4, keep=_keep)
    snapshots, parsed = OrderedDict(), {}
    try:
        for t, sha in enumerate(suite.builds["trunk_sha"]):
            if sha not in snapshots:
                files = repo.tree(sha)
                snapshots[sha] = None if files is None else _Snapshot(repo, files, suite.names, suite.project, parsed)
                if len(snapshots) > SNAPSHOTS:
                    snapshots.popitem(last=False)
            snapshots.move_to_end(sha)
            snapshot = snapshots[sha]
            if snapshot is not None:
                module[t], distance[t] = snapshot.distances(suite.files[suite.changed(t)].tolist())
    finally:
        repo.close()
    out = {"module": module, "path": distance}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out
