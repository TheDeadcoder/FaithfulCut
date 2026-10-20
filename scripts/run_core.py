import subprocess
import sys
from pathlib import Path

CONFIGS = Path("configs") / "confirmatory"
RESULTS = Path("data") / "results" / "confirmatory"
EVALUATE = ("0.7", "1.0")


def main():
    """Run every registered configuration on the confirmatory split, then the registered tests and tables."""
    for config in sorted(CONFIGS.glob("*.toml")):
        subprocess.run([sys.executable, "scripts/run_experiments.py", str(config), *sys.argv[1:]], check=True)
    subprocess.run([sys.executable, "scripts/report.py", "--price", str(RESULTS / "price-main"),
                    str(RESULTS / "price-rest"), "--control", str(RESULTS / "control-main"), "--evaluate", *EVALUATE,
                    "--out", str(RESULTS / "core.md")], check=True)


if __name__ == "__main__":
    main()
