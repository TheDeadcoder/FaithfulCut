import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.analysis import cluster_weights, interval, one_sided
from faithfulcut.industrial import ORDERS, load_log, replay_log

BLOCK = 10
PRIMARY = "LatestFail"


def table(title: str, df: pd.DataFrame) -> str:
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}") if len(df) else "(none)"
    return f"## {title}\n\n```\n{body}\n```\n"


def history_coverage(path: str) -> dict:
    """How each row's precomputed LastResults compares with the test's earlier executions logged in the file: the
    history a time-budgeted agent reads if it is fed that column."""
    df = pd.read_csv(path, sep=";", usecols=["Name", "LastResults", "Cycle"]).sort_values("Cycle", kind="stable")
    earlier = df.groupby("Name").cumcount().to_numpy()
    listed = df["LastResults"].astype(str).str.strip("[] ").map(lambda s: len(s.split(",")) if s else 0).to_numpy()
    return {"log": Path(path).stem, "rows": len(df), "covers every earlier execution": (listed >= earlier).mean(),
            "equal": (listed == earlier).mean(), "longer": (listed > earlier).mean()}


def main():
    parser = argparse.ArgumentParser(description="H7: the price of censoring on industrial CI logs, FULL against "
                                                 "FAITHFUL, with a bootstrap over blocks of consecutive cycles")
    parser.add_argument("--logs", nargs="+", required=True, help="CSV logs in RETECS's format")
    parser.add_argument("--coverage", nargs="*", default=[], help="logs whose LastResults column to compare")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--out", required=True, help="markdown report; the result for report.py goes beside it")
    args = parser.parse_args()
    rates, rows = {}, []
    for path in args.logs:
        log, name = load_log(path), Path(path).stem
        for order in ORDERS:
            for regime in ("full", "faithful"):
                runs = [replay_log(log, order, regime, seed) for seed in range(args.seeds)]
                missed = np.sum([r["missed"].to_numpy() for r in runs], axis=0)
                failing = np.sum([r["failing"].to_numpy() for r in runs], axis=0)
                weights = cluster_weights(np.arange(len(missed)) // BLOCK, args.replicates, seed=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rates[(name, order, regime)] = weights @ missed / (weights @ failing)
                rows.append({"log": name, "order": order, "regime": regime, "cycles": len(missed),
                             "missed": missed.sum() / failing.sum()})
        print(name, flush=True)
    logs = [Path(p).stem for p in args.logs]
    results, macro = {}, []
    for order in ORDERS:
        full, faithful = (np.nanmean([rates[(log, order, r)] for log in logs], axis=0) for r in ("full", "faithful"))
        ratio = faithful / full
        results[order] = {**interval(ratio), "p": one_sided(ratio, above=1.0)}
        macro.append({"order": order, "full": full[0], "faithful": faithful[0], **interval(ratio),
                      "p": results[order]["p"]})
    result = {**results[PRIMARY], "secondary": [{"order": o, **results[o]} for o in ORDERS if o != PRIMARY]}
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    parts = [table(f"Missed failing verdicts over failing verdicts, pooled over {args.seeds} seeds",
                   pd.DataFrame(rows)),
             table(f"FAITHFUL/FULL ratio of the mean rate over the logs ({args.replicates} resamples of blocks of "
                   f"{BLOCK} cycles; one-sided p for > 1)", pd.DataFrame(macro))]
    if args.coverage:
        parts.append(table("Rows whose precomputed LastResults lists at least as many verdicts as the test's earlier "
                           "logged executions", pd.DataFrame([history_coverage(p) for p in args.coverage])))
    out.write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
