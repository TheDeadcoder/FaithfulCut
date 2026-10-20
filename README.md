# FaithfulCut

FaithfulCut replays commit-aware test selection on real long-running CI test suites. Each component sees only the feedback it would have in deployment: outcomes of tests it ran or audited, and only after their builds finished. The study measures what selection saves and risks under that feedback, and calibrates the selection cut from random full-suite audits.

## Setup

Python 3.11 or later.

```bash
pip install -e ".[dev]"
pytest
```

## Inputs

The study uses the LRTS dataset of long-running test suites.

| Variable | Points to |
| --- | --- |
| `LRTS_REPO` | A clone of [LRTS](https://github.com/lrtsuser/LRTS) checked out at commit `2c1b68e9b1d4ec6c4f304869e36af2e3168da889` |
| `LRTS_DIR` | The `long_running_test_suites` folder unpacked from the processed LRTS release (`long_running_test_suites.zip`, linked from the LRTS README, SHA-256 `49ab497ce43b1706a7f8d340859c602be6adbcc1996cd81d0438523209cb9d6c`) |
| `FAITHFULCUT_CACHE` | Optional. A writable folder for derived data, such as per-project suites, which are built from the inputs on first use |
| `DANTE_DIR` | Optional. The parts of DANTE's replication package that `scripts/fetch_dante.py` fetches |

The runs studied are those listed in `$LRTS_REPO/artifact/metadata/dataset.csv`. Its SHA-256 is `49570f8dca753c385f25346f8aaaded3a841b0e67dabf36fbb67260a79b02987`.

When both variables are set, `pytest` also checks the pinned inputs and the consistency of the data.

Operational failure labels need the creation times of the issues in LRTS's flaky-test list. Fetch them once into `FAITHFULCUT_CACHE` from the public JIRA and GitHub APIs:

```bash
python scripts/fetch_flaky_issues.py
```

TF-IDF relevance reads test sources from bare clones of the ten upstream repositories, also kept in `FAITHFULCUT_CACHE`:

```bash
python scripts/clone_repos.py
```

```bash
python scripts/build_ir.py
```

DANTE's scorer is ported from its MIT-licensed replication package ([Zenodo record 18900675](https://zenodo.org/records/18900675)). This fetches the package's code, paper, published prioritization results and sample processed builds (about 13 MB of the 3.6 GB archive) into a new folder. With `DANTE_DIR` set to that folder, `pytest` also checks the port against DANTE's own outputs:

```bash
python scripts/fetch_dante.py <folder>
```

DANTE's relevance scores for its nine Java projects are computed once into `FAITHFULCUT_CACHE`:

```bash
python scripts/build_dante.py
```

The structural orders rank tests by module distance and by path distance from the build's changed files. Module distance follows the module graph at the build's base commit, read from its `pom.xml` files or Kafka's root `build.gradle`. Both are computed once into `FAITHFULCUT_CACHE`:

```bash
python scripts/build_structure.py
```

## Experiments

Labels for every audit seed are computed once, then each configuration runs in parallel. Results go to `data/results/<name>`.

```bash
python scripts/build_labels.py
```

```bash
python scripts/run_experiments.py configs/dev.toml
```

```bash
python scripts/analyze.py data/results/dev
```

With `per_build = true` in a configuration's `[price]` or `[control]` section, each evaluated build's outcome is also saved to `<section>_builds.npz`; the bootstrap helpers in `faithfulcut.analysis` resample those outcomes by pull request or by week. Fixed budgets tuned on a dev price grid are written once, and a `[control]` section's `tuned` entry points to the file:

```bash
python scripts/tune_budgets.py data/results/dev --out tuned_budgets.csv
```

The registered tests and tables read the per-build outcomes of the price grids and the main controller run, and report pull-request bootstrap intervals with Holm-adjusted p-values, as markdown and JSON:

```bash
python scripts/report.py --price data/results/dev-full/price-main data/results/dev-full/price-rest --control data/results/dev-full/control-main --evaluate 0.2 0.7 --out data/results/dev-full/core.md
```

`PREREGISTRATION.md` fixes the hypotheses, settings and analysis for the held-out builds. One command runs every registered configuration on the confirmatory split, then the report:

```bash
python scripts/run_core.py
```

```bash
python scripts/drift.py
```

A controller can instead target new failures, failures of classes never seen failing in a finished warm-up or audited build, with `target = "newfail"` in its `[control]` section. Every control run also reports escapes on that label. This compares the two targets:

```bash
python scripts/new_failures.py --target data/results/dev-full/control-newfail --operational data/results/dev-full/control-main --evaluate 0.2 0.7 --out data/results/dev-full/new-failures.md
```

The module set, meaning the changed modules and every module depending on them, also runs as a fixed selector at its natural operating point:

```bash
python scripts/module_set.py --evaluate 0.2 0.7 --out data/results/dev-full/module-set.md
```

Two more ways to see what a selector skips can be added alongside random audits:
- scheduled full runs every k builds: `schedules` in a price grid, or `schedule` in a `[control]` section;
- stalest-first rotation within a share of the predicted suite cost: `rotations`, or `rotate`.

With `latency = true`, a run also records how many builds a skipped failure stays unseen. This compares the three at equal extra cost:

```bash
python scripts/information_repair.py --repair data/results/dev-full/price-repair --full data/results/dev-full/price-main --control data/results/dev-full/control-repair* --out data/results/dev-full/information-repair.md
```

Two options change how much a build tests. With `threshold = true`, the cut covers a share of the change's TF-IDF relevance mass instead of a share of predicted test time; λ is then -log of the uncovered share. With `path_filter = true`, a change that touches only documentation and test sources runs just its changed and new tests. This compares a variant's results with a baseline's for the same orders:

```bash
python scripts/compare_variants.py --kind control --variant data/results/dev-full/control-threshold --baseline data/results/dev-full/control-main --orders TF-IDF+CC Hybrid-RRF --evaluate 0.2 0.7 --out data/results/dev-full/threshold-control.md
```

The processed release also lists builds that start after the studied window. They form a second held-out period: replayed after the studied builds as one timeline, at positions from 1 on, so a configuration with `evaluate = [1.0, 2.0]` scores them. Their suites, operational labels and change scores are built once:

```bash
python scripts/build_later.py
```

The change-based scores are precomputed for replay speed; their per-build computing time is measured separately on consecutive dev builds:

```bash
python scripts/time_selectors.py
```

The LLM relevance pilot keeps every response in `FAITHFULCUT_CACHE`. It authenticates with `GEMINI_API_KEY` when set, and otherwise with Application Default Credentials on the Google Cloud project in `GOOGLE_CLOUD_PROJECT` (location `GOOGLE_CLOUD_LOCATION`, default `global`):

```bash
python scripts/llm_pilot.py
```
