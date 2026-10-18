import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.1


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def mechanism(row, base_pi: float = 0.0) -> str:
    if getattr(row, "regime", "faithful") == "full":
        return "FULL"
    if row.schedule:
        return f"full run every {int(row.schedule)} builds"
    if row.rotate:
        return f"stalest-first rotation {row.rotate:g}"
    return f"random audits {row.pi - base_pi:g}" if row.pi > base_pi else "none"


def read(folders: list[str], kind: str, base_pi: float = 0.0) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(Path(f) / f"{kind}.csv") for f in folders], ignore_index=True)
    for column in ("schedule", "rotate"):
        df[column] = df[column].fillna(0) if column in df else 0
    df["mechanism"] = [mechanism(r, base_pi) for r in df.itertuples()]
    return df


def matched(price: pd.DataFrame, target: float) -> pd.DataFrame:
    """Per project, order, mechanism and seed, metrics interpolated at the target share of test time the selection
    itself runs."""
    df = price.assign(escape=price["operational_escape"] / price["operational_faulty"].replace(0, np.nan),
                      firstfail=price["firstfail_escape"] / price["firstfail_faulty"].replace(0, np.nan),
                      extra=price["work"] - price["base_work"])
    metrics = ["escape", "firstfail", "extra"] + (["operational_hidden_builds"] if "operational_hidden_builds" in df
                                                  else [])
    rows = []
    for key, g in df.groupby(["project", "scorer", "mechanism", "seed"]):
        g = g.sort_values("base_work")
        if g["base_work"].min() <= target <= g["base_work"].max():
            rows.append({**dict(zip(["project", "order", "mechanism", "seed"], key)),
                         **{m: np.interp(target, g["base_work"], g[m]) for m in metrics if g[m].notna().all()}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Buying back what the selector does not see: random audits, "
                                                 "scheduled full runs and stalest-first rotation")
    parser.add_argument("--repair", nargs="+", required=True, help="FAITHFUL price grids with the mechanisms")
    parser.add_argument("--full", nargs="+", required=True, help="price grids whose FULL rows are the reference")
    parser.add_argument("--control", nargs="+", required=True, help="controller runs with the mechanisms")
    parser.add_argument("--base-pi", type=float, default=0.05, help="the controller runs' own audit rate")
    parser.add_argument("--work", type=float, default=0.2, help="share of test time the selection runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    repair = read(args.repair, "price")
    full = read(args.full, "price")
    full = full[(full["regime"] == "full") & full["scorer"].isin(repair["scorer"].unique())]
    at = matched(pd.concat([repair, full], ignore_index=True), args.work)
    means = at.groupby(["order", "mechanism", "project"]).mean(numeric_only=True).drop(columns="seed")
    macro = means.groupby(["order", "mechanism"]).mean().reset_index()
    out = []
    for order, g in macro.groupby("order"):
        g = g.set_index("mechanism")
        full, none = g.loc["FULL", "escape"], g.loc["none", "escape"]
        for name, r in g.iterrows():
            out.append({"order": order, "mechanism": name, "escape": r["escape"], "firstfail": r["firstfail"],
                        "extra work": r["extra"], "loss recovered": (none - r["escape"]) / (none - full),
                        "hidden builds": r.get("operational_hidden_builds", np.nan)})
    parts = [table(f"Selection at {args.work:.0%} of test time, macro over projects: escape, extra work spent, share "
                   "of the censoring loss recovered, and builds until a hidden failure's class runs again",
                   pd.DataFrame(out))]
    control = read(args.control, "control", args.base_pi)
    control = control[control["controller"] == "online"]
    rows = []
    for (order, name, pi), g in control.groupby(["scorer", "mechanism", "pi"]):
        per = g.groupby("project").agg(escape=("operational_escape", "sum"), faulty=("operational_faulty", "sum"),
                                       work=("work", "mean"), base=("base_work", "mean"),
                                       hidden=("operational_hidden_builds", "mean"),
                                       width=("width_adaptive", "mean"))
        rate = per["escape"] / per["faulty"]
        rows.append({"order": order, "mechanism": name, "audits": pi, "escape": rate.mean(),
                     "|escape - alpha|": (rate - ALPHA).abs().mean(), "work": per["work"].mean(),
                     "extra work": (per["work"] - per["base"]).mean(), "hidden builds": per["hidden"].mean(),
                     "adaptive width": per["width"].mean()})
    parts.append(table(f"Online controller at alpha {ALPHA}, macro over projects: the controller reads only the random "
                       "audits", pd.DataFrame(rows)))
    Path(args.out).write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
