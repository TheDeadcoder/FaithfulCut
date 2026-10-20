import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut import lrts
from faithfulcut.dante import LOOKBACK, stickiness

WINDOW = 100


def artifact_dir() -> Path:
    return Path(os.environ["DANTE_DIR"])


def published() -> pd.DataFrame:
    """DANTE's published per-build prioritization APFDc, one row per build and method."""
    return pd.concat([pd.read_csv(p) for p in sorted((artifact_dir() / "experiments" / "RQ1").glob("*.csv"))])


def _apfdc(order: list, scores: pd.DataFrame) -> float:
    duration = scores["duration"].loc[order].to_numpy()
    failed = scores["outcome"].loc[order].to_numpy() == 1
    total = duration.sum()
    if not failed.any():
        return 1.0
    if total == 0:
        return 0.0
    detection = (np.cumsum(duration) - duration / 2.0)[failed]
    return 1.0 - detection.sum() / (failed.sum() * total) + 1.0 / (2.0 * len(order))


def processed_builds() -> list[Path]:
    return sorted((artifact_dir() / "src" / "processed_builds").glob("*.json"))


def release_metadata() -> pd.DataFrame:
    return pd.read_csv(lrts.data_dir() / "dataset.csv", low_memory=False)


def target_run(meta: pd.DataFrame, stem: str) -> tuple[pd.DataFrame, pd.Series]:
    """The project's runs in DANTE's start order and the run a processed build file belongs to."""
    project = stem.split("_build_")[0]
    runs = meta[meta["project"].str.lower() == project].copy()
    runs["ts"] = pd.to_numeric(runs["build_timestamp"], errors="coerce").fillna(0)
    runs = runs.sort_values(by="ts", ascending=True).reset_index(drop=True)
    names = (runs["pr_name"].astype(str).str.replace("/", "_") + "_" + runs["build_id"].astype(str) + "_"
             + runs["stage_id"].astype(str))
    return runs, runs[[f"{project}_build_{n}" in stem for n in names]].iloc[0]


def _history(meta: pd.DataFrame, project: str, stem: str):
    runs, target = target_run(meta, stem)
    frames = []
    for r in runs[runs["ts"] < target["ts"]].tail(WINDOW).itertuples():
        build = f"{str(r.pr_name).replace('/', '_')}_build{r.build_id}"
        path = lrts.data_dir() / "processed_test_result" / project / build / f"stage_{r.stage_id}" / "test_class.csv.zip"
        if path.exists():
            frames.append(pd.read_csv(path, usecols=["testclass", "outcome", "duration"]))
    return frames


def recompute() -> pd.DataFrame:
    """Every published build that ships DANTE's processed inputs, rescored with DANTE's code beside its published
    APFDc: DANTE-CCH without and with the stored relevance term, and LatestFail by cost and history and by recency."""
    meta = release_metadata()
    pub = published()
    out = []
    for path in processed_builds():
        ref = pub[pub["build"] == path.stem].set_index("method")["apfdc"]
        if ref.empty:
            continue
        project = path.stem.split("_build_")[0]
        frames = _history(meta, project, path.stem)
        rows = pd.concat(frames, ignore_index=True)
        thfr = rows.groupby("testclass")["outcome"].mean().to_dict()
        p = stickiness(pd.factorize(rows["testclass"])[0], rows["outcome"].to_numpy() == 1)
        fails, last_fail = defaultdict(int), {}
        for i, frame in enumerate(frames):
            for name, outcome in zip(frame["testclass"], frame["outcome"]):
                fails[name] += outcome
                if outcome == 1:
                    last_fail[name] = len(frames) - i
        scores = {}
        for fqn, tc in json.loads(path.read_text())["test_cases"].items():
            if not tc.get("source_file_found") or tc.get("duration", 0) <= 0:
                continue
            history = thfr.get(fqn, 0) + sum(p[n - 1] / n for n in range(1, LOOKBACK + 1)
                                             if tc.get(f"last_outcome_{n}", 0) == 1)
            stored = [s.get("tfidf", 0) for s in tc.get("relevance_to_changed_files", {}).values()]
            relevance = (max(stored) if stored else 0.0) ** 1.2
            factor = max(1.0 / (1.0 / (fails.get(fqn, 0) + 1.0)) - 1.0, 0.1)
            d = max(tc["duration"], 0.001)
            lf = last_fail.get(fqn, 9999)
            scores[fqn] = {"released": history * factor / d, "stored": (history + relevance) * factor / d,
                           "lf_coded": lf * d / factor, "lf_recency": lf, "duration": tc["duration"],
                           "outcome": tc["outcome"]}
        scores = pd.DataFrame.from_dict(scores, orient="index")
        order = {c: scores.sort_values(by=c, ascending=c.startswith("lf")).index.tolist()
                 for c in ("released", "stored", "lf_coded", "lf_recency")}
        out.append({"build": path.stem, "project": project, "history_runs": len(frames),
                    "dante_published": ref["DANTE-CCH"], "dante_released": _apfdc(order["released"], scores),
                    "dante_stored": _apfdc(order["stored"], scores),
                    "latestfail_published": ref["LatestFail-CCH"],
                    "latestfail_coded": _apfdc(order["lf_coded"], scores),
                    "latestfail_recency": _apfdc(order["lf_recency"], scores)})
    return pd.DataFrame(out)
