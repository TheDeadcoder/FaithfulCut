import numpy as np
import pandas as pd

LABELS = ("operational", "deconf", "firstfail")
HISTORY = ["LatestFail", "LatestFail+CC", "FailCount+CC", "FailCount-W+CC", "TF-FailFreq+CC", "TF-FailFreq-W+CC"]
PRICE_KEYS = ["project", "scorer", "regime", "pi", "seed"]
CONTROL_KEYS = ["project", "scorer", "controller", "alpha", "pi"]


def rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for label in LABELS:
        faulty = out[f"{label}_faulty"].replace(0, np.nan)
        out[f"{label}_escape_rate"] = out[f"{label}_executed_escape"] / faulty
        out[f"{label}_base_escape_rate"] = out[f"{label}_escape"] / faulty
        out[f"{label}_audited_rate"] = out[f"{label}_audited_escape"] / out[f"{label}_audited"].replace(0, np.nan)
        out[f"{label}_recall"] = out[f"{label}_rows_selected"] / out[f"{label}_rows"].replace(0, np.nan)
    return out


def at_work(price: pd.DataFrame, targets=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Executed escape and test recall interpolated at matched executed work, per replay group and seed."""
    df = rates(price)
    metrics = [f"{label}_{m}" for label in LABELS for m in ("escape_rate", "recall")]
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
    for col in ("width_normal", "width_adaptive", "width_worst", "updates", "pending", "lam_min", "lam_max"):
        if col in df:
            out[col] = g[col].mean()
    return out.reset_index()
