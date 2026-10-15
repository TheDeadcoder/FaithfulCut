import argparse
from pathlib import Path

import pandas as pd

from faithfulcut.analysis import at_work, condition_means, control_summary, g3, price_of_information

FOCUS = ["macro", "kafka", "james", "hive"]
RQ4_SCORERS = ["QTF", "TF-IDF", "TF-IDF+CC", "LatestFail+CC", "TF-FailFreq+CC"]


def section(title: str, df: pd.DataFrame) -> str:
    return f"## {title}\n\n```\n{df.to_string(index=False, float_format=lambda x: f'{x:.3f}')}\n```\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="folder holding price.csv and control.csv")
    args = parser.parse_args()
    folder = Path(args.results)
    parts = []
    if (folder / "price.csv").exists():
        means = condition_means(at_work(pd.read_csv(folder / "price.csv")))
        means.to_csv(folder / "price_matched.csv", index=False)
        holds, table = g3(means)
        parts.append(section(f"G3 (operational, 20% work, macro): {'holds' if holds else 'does not hold'}", table))
        for label in ("operational", "deconf"):
            for work in (0.1, 0.2, 0.3):
                t = price_of_information(means, label, work)
                parts.append(section(f"Price of information, {label}, {work:.0%} work", t[t["project"].isin(FOCUS)]))
        rq4 = price_of_information(means, "firstfail", 0.2)
        parts.append(section("Reading the change, FirstFail, 20% work",
                             rq4[rq4["project"].isin(FOCUS) & rq4["scorer"].isin(RQ4_SCORERS)]))
        audits = means[(means["regime"] == "faithful") & (means["work"] == 0.2) & means["project"].isin(FOCUS)]
        parts.append(section("FAITHFUL without and with 10% audits, 20% work",
                             audits.pivot_table(index=["project", "scorer"], columns="pi",
                                                values="operational_escape_rate").reset_index()))
    if (folder / "control.csv").exists():
        summary = control_summary(pd.read_csv(folder / "control.csv"))
        summary.to_csv(folder / "control_summary.csv", index=False)
        for project in ("kafka", "james", "hive"):
            parts.append(section(f"Risk control, {project}", summary[summary["project"] == project]))
    (folder / "report.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
