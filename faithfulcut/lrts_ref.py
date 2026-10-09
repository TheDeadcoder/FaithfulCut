import numpy as np
import pandas as pd

from faithfulcut import lrts

EPS = 1e-6
SEEDS = range(10)
TCPS = ("QTF", "LastFail")
LABELS = ("deconf", "firstfail")


def _jira_class(test: str):
    tokens = test.split("#")
    if len(tokens) == 2:
        return tokens[0]
    return None if "[" in tokens[0] else tokens[0]


def flaky_classes(project: str) -> pd.DataFrame:
    df = pd.read_csv(lrts.repo_dir() / "artifact" / "flaky_test" / "flaky_tests.csv")
    df["testclass"] = [_jira_class(t) for t in df["flaky_test"]]
    df = df.sort_values(["project", "fix_timestamp"]).drop(columns="flaky_test")
    df = df.drop_duplicates(subset=["project", "testclass"]).dropna()
    return df[df["project"] == project].reset_index(drop=True)


def _ends_with_any(names: np.ndarray, suffixes) -> np.ndarray:
    suffixes = tuple(suffixes)
    return np.array([n.endswith(suffixes) for n in names], dtype=bool)


def _apfdc(order: np.ndarray, duration: np.ndarray, failed: np.ndarray) -> float:
    d, f = duration[order], failed[order]
    if not f.any():
        return np.nan
    tail = np.cumsum(d[::-1])[::-1]
    return float((tail[f] - d[f] / 2).sum() / (f.sum() * d.sum()))


def _mean_apfdc(key: np.ndarray, keep: np.ndarray, duration: np.ndarray, failed: np.ndarray, run_no: int) -> float:
    scores = []
    for seed in SEEDS:
        tie = np.random.default_rng([seed, run_no]).uniform(size=len(key))
        order = np.lexsort((tie, key))
        scores.append(_apfdc(order[keep[order]], duration + 0.001, failed))
    return float(np.mean(scores))


def _read(project: str):
    runs = lrts.load_runs(project)
    index = {}
    data = []
    for r in runs.itertuples(index=False):
        df = pd.read_csv(lrts.results_path(project, r.pr_name, r.build_id, r.stage_id))
        ids = np.fromiter((index.setdefault(n, len(index)) for n in df["testclass"]), np.int64, len(df))
        data.append((ids, df["duration"].to_numpy(), df["outcome"].to_numpy() == 1, df["last_outcome"].to_numpy() == 1))
    return runs, np.array(list(index), dtype=object), data


def _frequent(names: np.ndarray, runs: pd.DataFrame, data: list) -> dict:
    out = {}
    for stage, idx in runs.groupby("stage_id").groups.items():
        counts = np.bincount(np.concatenate([data[i][0][data[i][2]] for i in idx] + [np.empty(0, np.int64)]),
                             minlength=len(names))
        freq = pd.Series(counts[counts > 0])
        threshold = round(freq.mean(), 2) + 3 * round(freq.std(), 2)
        out[stage] = _ends_with_any(names, names[counts > threshold])
    return out


def _stage_common(names: np.ndarray, runs: pd.DataFrame, data: list) -> dict:
    out = {}
    for _, idx in runs.groupby(["pr_name", "build_id"]).groups.items():
        failed = [set(data[i][0][data[i][2]]) for i in idx]
        common = names[sorted(set.intersection(*failed))]
        for i in idx:
            out[i] = common
    return out


def reference_apfdc(project: str) -> pd.DataFrame:
    runs, names, data = _read(project)
    flaky = flaky_classes(project)
    jira = np.array([[n.endswith(c) for c in flaky["testclass"]] for n in names], dtype=bool).reshape(len(names), -1)
    frequent = _frequent(names, runs, data)
    common = _stage_common(names, runs, data)
    rows = []
    for stage, idx in runs.groupby("stage_id").groups.items():
        last_failure = np.zeros(len(names))
        last_duration = np.zeros(len(names))
        failed_before = np.zeros(len(names), dtype=bool)
        for i in sorted(idx, key=lambda i: runs.at[i, "start"]):
            ids, duration, failed, last_failed = data[i]
            run = runs.loc[i]
            flaky_now = jira[ids][:, (flaky["fix_timestamp"] > run["start"]).to_numpy()].any(axis=1)
            moved = failed | (failed != last_failed)
            unique = failed.copy()
            unique[failed] = ~_ends_with_any(names[ids[failed]], common[i])
            filtered = {
                "deconf": (moved & flaky_now) | (moved & frequent[stage][ids]) | unique,
                "firstfail": (moved & flaky_now) | unique | (failed & failed_before[ids]),
            }
            keys = {"QTF": last_duration[ids] + EPS, "LastFail": last_failure[ids] + EPS}
            for label in LABELS:
                keep = ~filtered[label]
                values = [np.nan] * len(TCPS)
                if (failed & keep).any():
                    values = [_mean_apfdc(keys[tcp], keep, duration, failed, i) for tcp in TCPS]
                rows.append((run["pr_name"], run["build_id"], stage, label, *values))
            last_failure[ids] = np.where(failed, 1, last_failure[ids] + 1)
            last_duration[ids] = duration
            failed_before[ids] |= failed
    return pd.DataFrame(rows, columns=["pr_name", "build_id", "stage_id", "label", *TCPS])
