import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.drift import QUARTERS

KEYS = ["project", "scorer", "pi", "seed"]


def main():
    parser = argparse.ArgumentParser(description="Injected drift against clean replays of the same seed: alarms, "
                                                 "and extra escapes and work per quarter of the horizon")
    parser.add_argument("folder", help="results with drift.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = pd.read_csv(Path(args.folder) / "drift.csv")
    clean = df[df["kind"] == "clean"].set_index(KEYS)
    rows = []
    for (kind, scorer, pi), g in df[df["kind"] != "clean"].groupby(["kind", "scorer", "pi"]):
        g = g.set_index(KEYS)
        base = clean.loc[g.index]
        rise = lambda d: (d["lam_max"] - d["lam_start"]).to_numpy()
        pairs = [(rise(g.loc[p])[:, None], rise(base.loc[p])[None, :]) for p in g.index.unique("project")]
        auc = [(x > c).mean() + (x == c).mean() / 2 for x, c in pairs]
        row = {"drift": kind, "order": scorer, "pi": pi, "runs": len(g),
               "alarmed": (g["alarms_after"] > 0).mean(), "alarmed clean": (base["alarms_after"] > 0).mean(),
               "delay": g["first_alarm"].median(),
               "false per 1000": 1000 * base["alarms_before"].sum() / base["builds_before"].sum(),
               "lambda rise AUC": float(np.mean(auc))}
        for q in range(QUARTERS):
            rate = lambda d: d[f"escape_{q}"] / d[f"faulty_{q}"].replace(0, np.nan)
            excess = (rate(g) - rate(base)).groupby(level="project").mean()
            row[f"escape Q{q}"] = excess.mean()
            row[f"work Q{q}"] = (g[f"work_{q}"] - base[f"work_{q}"]).groupby(level="project").mean().mean()
        rows.append(row)
    body = pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}")
    text = "## Injected drift minus clean replay (macro over projects)\n\nAlarmed: share of runs with an alarm in " \
           "the horizon; delay: median builds to the first alarm; lambda rise AUC: chance that the controller's " \
           "largest rise over the horizon exceeds a clean run's of the same project, ties counting half; escape and " \
           f"work: extra escape rate and test work per quarter of the horizon.\n\n```\n{body}\n```\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
