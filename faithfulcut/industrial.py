import numpy as np
import pandas as pd

ORDERS = ("LatestFail", "FailCount", "QTF")
WARMUP = 0.2
BUDGET = 0.2


def load_log(path) -> pd.DataFrame:
    """A CI log in RETECS's format, one row per logged test execution: its cycle, its test (from `Name`), its
    duration and whether it failed, in cycle order."""
    df = pd.read_csv(path, sep=";", usecols=["Name", "Duration", "Verdict", "Cycle"])
    log = pd.DataFrame({"cycle": df["Cycle"].to_numpy(), "test": pd.factorize(df["Name"])[0],
                        "duration": df["Duration"].astype(float).to_numpy(),
                        "failed": df["Verdict"].astype(str).str.lower().isin(["1", "true"]).to_numpy()})
    return log.sort_values("cycle", kind="stable").reset_index(drop=True)


def replay_log(log: pd.DataFrame, order: str, regime: str, seed: int, warmup: float = WARMUP,
               budget: float = BUDGET) -> pd.DataFrame:
    """Per cycle after the warm-up, failing verdicts and those the selection missed. During the warm-up every logged
    test runs. Afterwards a test runs at its first appearance; the others are taken in `order` while their logged
    durations fit within `budget` of the cycle's total, ties broken at random. FULL learns from every logged verdict,
    FAITHFUL only from the tests that ran."""
    bounds = np.flatnonzero(np.diff(log["cycle"].to_numpy(), prepend=-1, append=-1))
    test, duration, failed = (log[c].to_numpy() for c in ("test", "duration", "failed"))
    n = int(test.max()) + 1
    seen, fails = np.zeros(n, dtype=bool), np.zeros(n)
    last_fail, last_duration = np.full(n, -1.0), np.zeros(n)
    rng = np.random.default_rng(seed)
    warm = int(np.ceil(warmup * (len(bounds) - 1)))
    rows = []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        t, d, f = test[a:b], duration[a:b], failed[a:b]
        run = np.ones(b - a, dtype=bool)
        if k >= warm:
            key = {"LatestFail": -last_fail, "FailCount": -fails, "QTF": last_duration}[order][t]
            ranked = np.lexsort((rng.random(b - a), key))
            ranked = ranked[seen[t[ranked]]]
            run = ~seen[t]
            run[ranked[np.cumsum(d[ranked]) <= budget * d.sum()]] = True
            rows.append({"cycle": int(log["cycle"].iat[a]), "failing": int(f.sum()), "missed": int((f & ~run).sum())})
        watch = np.ones(b - a, dtype=bool) if regime == "full" else run
        seen[t[watch]] = True
        np.add.at(fails, t[watch], f[watch])
        last_fail[t[watch & f]] = k
        last_duration[t[watch]] = d[watch]
    return pd.DataFrame(rows)
