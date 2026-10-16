from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("operational", "deconf", "firstfail", "firstfail_cold")
HISTORY = ["LatestFail", "LatestFail+CC", "FailCount+CC", "FailCount-W+CC", "TF-FailFreq+CC", "TF-FailFreq-W+CC"]
PRICE_KEYS = ["project", "scorer", "regime", "pi", "seed"]
CONTROL_KEYS = ["project", "scorer", "controller", "alpha", "pi"]
WATTS = (10.0, 300.0)
GRAMS_CO2E_PER_KWH = (20.0, 700.0)


def _labels(df: pd.DataFrame) -> list[str]:
    return [label for label in LABELS if f"{label}_faulty" in df]


def rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for label in _labels(out):
        faulty = out[f"{label}_faulty"].replace(0, np.nan)
        out[f"{label}_escape_rate"] = out[f"{label}_executed_escape"] / faulty
        out[f"{label}_base_escape_rate"] = out[f"{label}_escape"] / faulty
        if f"{label}_masked" in out:
            out[f"{label}_masked_rate"] = out[f"{label}_masked"] / faulty
        out[f"{label}_audited_rate"] = out[f"{label}_audited_escape"] / out[f"{label}_audited"].replace(0, np.nan)
        out[f"{label}_recall"] = out[f"{label}_rows_selected"] / out[f"{label}_rows"].replace(0, np.nan)
    return out


def at_work(price: pd.DataFrame, targets=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Executed escape and test recall interpolated at matched executed work, per replay group and seed."""
    df = rates(price)
    metrics = [f"{label}_{m}" for label in _labels(df) for m in ("escape_rate", "recall")]
    rows = []
    for key, g in df.groupby(PRICE_KEYS):
        g = g.sort_values("work")
        for w in targets:
            if g["work"].min() <= w <= g["work"].max():
                rows.append({**dict(zip(PRICE_KEYS, key)), "work": w,
                             **{m: np.interp(w, g["work"], g[m]) for m in metrics}})
    return pd.DataFrame(rows)


def condition_means(matched: pd.DataFrame) -> pd.DataFrame:
    """Means over seeds per project and condition."""
    keys = ["scorer", "regime", "pi", "work", "project"]
    return matched.groupby(keys).mean(numeric_only=True).drop(columns="seed").reset_index()


def price_of_information(means: pd.DataFrame, label: str = "operational", work: float = 0.2) -> pd.DataFrame:
    """FULL, TIMED and FAITHFUL (no audits) escape and recall per order and project, with the FAITHFUL/FULL
    escape ratio; the macro row averages, per order, the projects where all three regimes are defined."""
    m = means[(means["work"] == work) & (means["pi"] == 0.0)]
    table = m.pivot_table(index=["scorer", "project"], columns="regime",
                          values=[f"{label}_escape_rate", f"{label}_recall"], dropna=False)
    escape, recall = table[f"{label}_escape_rate"], table[f"{label}_recall"]
    out = pd.DataFrame({
        "escape_full": escape["full"], "escape_timed": escape["timed"], "escape_faithful": escape["faithful"],
        "recall_full": recall["full"], "recall_faithful": recall["faithful"],
    }).reset_index()
    macro = out.dropna().groupby("scorer").mean(numeric_only=True).reset_index()
    macro["projects"] = out.dropna().groupby("scorer").size().to_numpy()
    out = pd.concat([out, macro.assign(project="macro")], ignore_index=True)
    out["ratio"] = out["escape_faithful"] / out["escape_full"]
    out["recall_gap"] = out["recall_full"] - out["recall_faithful"]
    return out


def g3(means: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """G3 as fixed in the plan: operational label, 20% work, macro over projects, FAITHFUL without audits."""
    table = price_of_information(means)
    macro = table[(table["project"] == "macro") & table["scorer"].isin(HISTORY)]
    return bool(((macro["ratio"] >= 1.5) | (macro["recall_gap"] >= 0.03)).any()), macro


def control_summary(control: pd.DataFrame) -> pd.DataFrame:
    df = rates(control)
    g = df.groupby(CONTROL_KEYS)
    out = g.agg(
        seeds=("seed", "size"), faulty=("operational_faulty", "mean"),
        escape=("operational_base_escape_rate", "mean"),
        escape_p05=("operational_base_escape_rate", lambda x: x.quantile(0.05)),
        escape_p95=("operational_base_escape_rate", lambda x: x.quantile(0.95)),
        executed=("operational_escape_rate", "mean"), audited=("operational_audited_rate", "mean"),
        deconf=("deconf_base_escape_rate", "mean"), firstfail=("firstfail_base_escape_rate", "mean"),
        work=("work", "mean"), forced=("forced", "mean"),
    )
    for col in ("width_normal", "width_adaptive", "width_worst", "updates", "pending", "lam_min", "lam_max",
                "work_with_warmup", "verdict_hours", "full_hours", "decision_seconds", "operational_ttff_hours",
                "operational_ttff_share", "deconf_masked_rate", "firstfail_cold_base_escape_rate"):
        if col in df:
            out[col] = g[col].mean()
    return out.reset_index()


def energy(test_hours: float) -> dict[str, float]:
    """Energy and emissions of test time over the stated ranges of executor power and grid intensity."""
    kwh = [test_hours * w / 1000 for w in WATTS]
    return {"kwh_low": kwh[0], "kwh_high": kwh[1], "kg_co2e_low": kwh[0] * GRAMS_CO2E_PER_KWH[0] / 1000,
            "kg_co2e_high": kwh[1] * GRAMS_CO2E_PER_KWH[1] / 1000}


def break_even_hours(overhead_seconds: float, work: float) -> float:
    """Full-suite test time per build at which the test time a selection saves pays for the selector's own time."""
    return overhead_seconds / 3600 / (1 - work) if work < 1 else np.inf


def tuned_budgets(price: pd.DataFrame, alphas, pi: float = 0.1, regime: str = "faithful") -> pd.DataFrame:
    """Per project and order, the smallest budget whose operational escape of the base selection, averaged over
    seeds and made non-increasing in lambda, reaches alpha, interpolated over the lambda grid (1.0 if none does)."""
    df = rates(price[(price["regime"] == regime) & (price["pi"] == pi)])
    curve = df.groupby(["project", "scorer", "lam"])["operational_base_escape_rate"].mean().reset_index()
    out = []
    for (project, scorer), g in curve.groupby(["project", "scorer"]):
        lam = g["lam"].to_numpy()
        escape = np.minimum.accumulate(g["operational_base_escape_rate"].to_numpy())
        for alpha in alphas:
            ok = np.flatnonzero(escape <= alpha)
            if len(ok) == 0:
                value = 1.0
            elif ok[0] == 0:
                value = lam[0]
            else:
                i = ok[0]
                value = np.interp(alpha, [escape[i], escape[i - 1]], [lam[i], lam[i - 1]])
            out.append({"project": project, "scorer": scorer, "alpha": alpha, "lam": float(value)})
    return pd.DataFrame(out)


def load_build_outcomes(folder: Path, kind: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = pd.read_csv(Path(folder) / f"{kind}.csv")
    z = np.load(Path(folder) / f"{kind}_builds.npz")
    return rows, {k: z[k] for k in z.files}


def condition_sums(rows: pd.DataFrame, outcomes: dict[str, np.ndarray], label: str, **keys) -> pd.DataFrame:
    """For the rows matching `keys`, per lambda and build, sums over seeds of faulty, escaped (base selection),
    escaped without a full run, masked, executed and full test time."""
    match = np.logical_and.reduce([rows[k].to_numpy() == v for k, v in keys.items()])
    keep = match[outcomes["row"]]
    i = list(outcomes["labels"]).index(label)
    flags = outcomes["flags"][keep].astype(np.int64)
    faulty, caught, masked = ((flags >> 3 * i + b) & 1 for b in range(3))
    full_run = (flags >> 15) & 1
    lam = rows["lam"].to_numpy()[outcomes["row"][keep]] if "lam" in rows else np.zeros(int(keep.sum()))
    df = pd.DataFrame({"lam": lam, "build": outcomes["build"][keep], "faulty": faulty,
                       "escaped": faulty & (1 - caught), "executed_escape": faulty & (1 - caught) & (1 - full_run),
                       "masked": masked, "executed": outcomes["executed"][keep].astype(float),
                       "full": outcomes["full"][keep].astype(float)})
    return df.groupby(["lam", "build"]).sum().reset_index()


def cluster_weights(clusters: np.ndarray, replicates: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap weights per replicate and build: clusters drawn with replacement, each build weighted by how often
    its cluster is drawn. The first replicate is the original sample."""
    ids, inverse = np.unique(clusters, return_inverse=True)
    draws = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=replicates)
    draws[0] = 1
    return draws[:, inverse].astype(float)


def replicate_rates(sums: pd.DataFrame, builds: np.ndarray, weights: np.ndarray, numerator: str = "escaped",
                    denominator: str = "faulty") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per lambda (sorted) and replicate, the weighted ratio of the two columns and the executed share of test time."""
    lams = np.sort(sums["lam"].unique())
    rate, work = np.empty((len(lams), len(weights))), np.empty((len(lams), len(weights)))
    for j, lam in enumerate(lams):
        g = sums[sums["lam"] == lam].set_index("build").reindex(builds, fill_value=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate[j] = weights @ g[numerator].to_numpy() / (weights @ g[denominator].to_numpy())
            work[j] = weights @ g["executed"].to_numpy() / (weights @ g["full"].to_numpy())
    return lams, rate, work


def at_matched_work(rate: np.ndarray, work: np.ndarray, target: float) -> np.ndarray:
    """Each replicate's rate interpolated at the target executed share of test time (NaN outside its range)."""
    out = np.full(rate.shape[1], np.nan)
    for b in range(rate.shape[1]):
        order = np.argsort(work[:, b])
        w, r = work[order, b], rate[order, b]
        if w[0] <= target <= w[-1]:
            out[b] = np.interp(target, w, r)
    return out


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Paired bootstrap comparison of a and b (first replicate is the point estimate): difference, 95% percentile
    interval and two-sided p-value."""
    d = a - b
    boot = d[1:][~np.isnan(d[1:])]
    tail = min((boot <= 0).sum(), (boot >= 0).sum())
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"difference": float(d[0]), "low": float(low), "high": float(high),
            "p": float(min(1.0, 2 * (tail + 1) / (len(boot) + 1)))}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1.0)
    out = np.empty(len(p))
    out[order] = adjusted
    return out


def build_clusters(builds: pd.DataFrame, by: str) -> np.ndarray:
    """Cluster of each build for the bootstrap: its pull request, or the week it started in."""
    return builds["pr_name"].to_numpy() if by == "pr" else (builds["start"].to_numpy() // (7 * 86400)).astype(int)
