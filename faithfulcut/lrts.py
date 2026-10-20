import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RUN_KEY = ["project", "pr_name", "build_id", "stage_id"]


def repo_dir() -> Path:
    return Path(os.environ["LRTS_REPO"])


def data_dir() -> Path:
    return Path(os.environ["LRTS_DIR"])


def metadata_path() -> Path:
    return repo_dir() / "artifact" / "metadata" / "dataset.csv"


def metadata() -> pd.DataFrame:
    return pd.read_csv(metadata_path(), dtype={"stage_id": str})


def release_metadata() -> pd.DataFrame:
    """The processed release's own run list, which continues past the studied window."""
    return pd.read_csv(data_dir() / "dataset.csv", dtype={"stage_id": str})


def load_runs(project: str, later: bool = False) -> pd.DataFrame:
    """The studied runs; with `later`, also the release's runs of builds that start after the studied window."""
    meta = metadata()
    meta = meta[meta["project"] == project]
    if later:
        release = release_metadata()
        release = release[release["project"] == project]
        studied = set(zip(meta["pr_name"], meta["build_id"]))
        after = (release["build_timestamp"] > meta["build_timestamp"].max()) & np.array(
            [k not in studied for k in zip(release["pr_name"], release["build_id"])], dtype=bool)
        meta = pd.concat([meta, release[after]])
    runs = meta[RUN_KEY + ["trunk_sha", "build_head_sha"]].copy()
    runs["start"] = meta["build_timestamp"]
    runs["finish"] = meta["build_timestamp"] + meta["build_duration"]
    return runs.reset_index(drop=True)


def load_builds(project: str, later: bool = False) -> pd.DataFrame:
    """Builds in start order. Studied builds take positions in [0, 1) by rank; later builds follow from 1."""
    builds = load_runs(project, later).drop_duplicates(["pr_name", "build_id"]).drop(columns="stage_id")
    builds = builds.sort_values(["start", "pr_name", "build_id"], kind="stable").reset_index(drop=True)
    n = len(load_runs(project).drop_duplicates(["pr_name", "build_id"])) if later else len(builds)
    rank = builds.index.to_numpy()
    builds["position"] = np.where(rank < n, rank / n, 1 + (rank - n) / max(len(builds) - n, 1))
    return builds


def build_name(pr_name: str, build_id: int) -> str:
    return f"{pr_name}_build{build_id}"


def results_path(project: str, pr_name: str, build_id: int, stage_id: str) -> Path:
    return (
        data_dir() / "processed_test_result" / project / build_name(pr_name, build_id)
        / f"stage_{stage_id}" / "test_class.csv.zip"
    )


def read_results(project: str, pr_name: str, build_id: int, stage_id: str) -> pd.DataFrame:
    path = results_path(project, pr_name, build_id, stage_id)
    return pd.read_csv(path, usecols=["testclass", "outcome", "duration"])


def diff_path(project: str, pr_name: str, build_id: int) -> Path:
    return data_dir() / "shadata" / project / "diff" / f"{build_name(pr_name, build_id)}.diff"


def compare_pages(project: str, pr_name: str, build_id: int, base: str, head: str) -> list[Path]:
    folder = data_dir() / "shadata" / project / "compare_commits" / build_name(pr_name, build_id)
    pages = []
    while (page := folder / f"{base}_{head}_page{len(pages) + 1}.json").exists():
        pages.append(page)
    return pages


def changed_files(project: str, pr_name: str, build_id: int, base: str, head: str) -> list[str]:
    pages = compare_pages(project, pr_name, build_id, base, head)
    if not pages:
        raise FileNotFoundError(f"no comparison pages for {project} {build_name(pr_name, build_id)}")
    return sorted({f["filename"] for page in pages for f in json.loads(page.read_text()).get("files", [])})
