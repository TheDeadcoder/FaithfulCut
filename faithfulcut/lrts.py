import json
import os
from pathlib import Path

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


def load_runs(project: str) -> pd.DataFrame:
    meta = metadata()
    meta = meta[meta["project"] == project]
    runs = meta[RUN_KEY + ["trunk_sha", "build_head_sha"]].copy()
    runs["start"] = meta["build_timestamp"]
    runs["finish"] = meta["build_timestamp"] + meta["build_duration"]
    return runs.reset_index(drop=True)


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
