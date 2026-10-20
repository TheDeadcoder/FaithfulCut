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

```bash
python scripts/drift.py
```

The change-based scores are precomputed for replay speed; their per-build computing time is measured separately on consecutive dev builds:

```bash
python scripts/time_selectors.py
```

The LLM relevance pilot keeps every response in `FAITHFULCUT_CACHE`. It authenticates with `GEMINI_API_KEY` when set, and otherwise with Application Default Credentials on the Google Cloud project in `GOOGLE_CLOUD_PROJECT` (location `GOOGLE_CLOUD_LOCATION`, default `global`):

```bash
python scripts/llm_pilot.py
```
