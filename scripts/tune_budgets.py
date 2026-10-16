import argparse
from pathlib import Path

import pandas as pd

from faithfulcut.analysis import tuned_budgets


def main():
    parser = argparse.ArgumentParser(description="Fixed budgets tuned on dev price grids, one per project, order "
                                                 "and alpha, for the 'tuned' controller")
    parser.add_argument("results", nargs="+", help="folders holding price.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1])
    parser.add_argument("--pi", type=float, default=0.1)
    args = parser.parse_args()
    price = pd.concat([pd.read_csv(Path(folder) / "price.csv") for folder in args.results])
    tuned = tuned_budgets(price, args.alphas, args.pi)
    tuned.to_csv(args.out, index=False)
    print(tuned.pivot_table(index=["project", "alpha"], columns="scorer", values="lam").round(3).to_string())


if __name__ == "__main__":
    main()
