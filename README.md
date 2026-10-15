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

```bash
python scripts/drift.py
```

The LLM relevance pilot reads an API key from `GEMINI_API_KEY` and keeps every response in `FAITHFULCUT_CACHE`:

```bash
python scripts/llm_pilot.py
```
