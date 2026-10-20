# FaithfulCut

Continuous integration runs a project's whole test suite on every change, and for large projects that takes hours. Running only the tests a change is likely to break saves most of that time. But a selector can skip a failing test, it has to keep working as the code and its tests change, and it has to save more than it costs to run. FaithfulCut studies how to select tests per commit under those constraints, and how to measure selectors the way they would actually run.

## The idea

Most evaluations of test selection replay a project's CI history and let the selector read the outcome of every earlier test run. A deployed selector never has that: it learns a test's outcome only if it ran the test, and only once the build has finished. FaithfulCut replays history under three information regimes:

- **FULL**: the outcomes of every build that started earlier, as most replays assume.
- **TIMED**: the outcomes of builds that have finished.
- **FAITHFUL**: the outcomes of the tests the selector ran, in builds that have finished.

The gap between FULL and FAITHFUL is the *price of information*: how much a conventional replay flatters a selector.

To bound the risk of skipping failures, FaithfulCut runs the whole suite on a small random share of builds. These audits let an online controller adjust how much of each suite to run, so that the *escape rate*, the share of failing builds whose failures the selection misses, stays at a chosen target. The same audits give a confidence bound on that rate, computed only from what the deployment sees.

## What was evaluated

**Data.** Ten open-source projects with long-running test suites, from the LRTS dataset: ActiveMQ, Hadoop, HBase, Hive, Jackrabbit Oak, James, Kafka, Karaf, Log4j and TVM. They contribute 21,255 builds (57,437 CI runs), followed by 13,389 later builds. Three public industrial CI logs, Paint Control and IOF/ROL from ABB Robotics and GSDTSR from Google, replicate the central measurement.

**Selectors.**
- History: most recently failed first (LatestFail), most often failed first (FailCount) and co-failure with the changed files (TF-FailFreq), each also in a cost-aware form (+CC).
- Duration: quickest test first (QTF).
- Change content: the TF-IDF relevance of the change's diff to each test's sources (TF-IDF, TF-IDF+CC).
- Structure: distance in the module dependency graph, and directory distance.
- Combinations: fusions of history with content (Hybrid, Hybrid-RRF), and DANTE's published scorer.
- Learned: a gradient-boosted failure model, and an LLM that reads the diff.

**Experiments.**
- The price of information for every selector at equal test time, split into feedback delay and censoring, and its effect on how selectors rank.
- Risk control: the online controller against fixed budgets, conformal and PAC calibration, and periodic recalibration. It was run at several audit rates and with audits steered by the change, and the coverage of its confidence bounds was measured.
- Failure labels: operational failures with known flaky tests excluded, first failures, and failures of tests never seen failing.
- Where the extra misses come from, and how long failures stay hidden.
- Buying information back with scheduled full runs and rotation of the stalest tests.
- Budgets set by the change's relevance, and a filter for changes that touch only documentation or tests.
- Drift: injected test renames, failure bursts and slowdowns, and detectors for them.
- Cost: net savings after audits and the selector's own time, in test-hours, energy and money.

**Design.** Every design choice was made on a development split. Hypotheses and analyses were written down, in `PREREGISTRATION.md` and `PREREGISTRATION-B.md`, before the held-out builds and the later builds were evaluated. Intervals come from resampling whole pull requests, with Holm correction.

## Findings

Unless stated otherwise, escape rates are compared at 20% of test time.

- **Full-information replays flatter history-based selection.**
  - LatestFail+CC's escape rate is 1.32 times higher under FAITHFUL than under FULL on the held-out builds, and 2.98 times higher on the later builds. DANTE's is 1.45 and 2.34 times higher.
  - On the industrial logs, the share of failing verdicts LatestFail misses is 1.04 times higher.
  - Feedback delay contributes almost nothing; the price comes from the outcomes of tests the selector never ran.
- **Reading the change is immune.** TF-IDF ordering pays no price: 0.97 and 1.01 times.
- **Faithful evaluation can reverse rankings.** On the five projects with the most later builds, a learned failure model escapes more often than LatestFail+CC under FULL (0.186 against 0.089). Under FAITHFUL it escapes least often of four selectors (0.172 against 0.283).
- **Audits hold the risk where it is set.** With the whole suite run on 10% of builds, the online controller kept the escape rate closer to its 10% target than a budget tuned on development data: on the held-out builds, its mean gap to the target was 0.008 against 0.040. Its confidence bound covered the true rate in every seed.
- **Selection pays for itself.**
  - After audits and its own computing time, selection saves 28–66% of test time on Hive, James and Kafka.
  - On the later builds, LatestFail+CC and Hybrid-RRF each save about 8,400 test-hours per 1,000 builds for an average project.
  - Matching the protection a full-information replay reports takes LatestFail+CC about 3,900 more test-hours per 1,000 builds, and TF-IDF+CC about 300.
- **What is certified decides which selector is cheapest.** On the later builds, certified on failures of tests never seen failing, Hybrid-RRF needs 9.5 percentage points less test time than LatestFail+CC at the same risk. Reading the change needs 7.7 points less than following the module graph.
- **Scheduled full runs repair history.** On the later builds, a full run every 8 builds lowers LatestFail+CC's escape rate by 14.8 points at the same selection budget.
- **An LLM adds little to TF-IDF.** Gemini 3.5 Flash-Lite put a first-failing test within the first 5% of predicted test time in 84.1% of builds, against 81.9% for TF-IDF+CC.
- **A reference loader for the industrial logs gives full information.** In RETECS, each test's precomputed result history reaches the selector, including results of runs the selector skipped.

## Repository

| Path | Contents |
| --- | --- |
| `faithfulcut/` | Replay engine, selectors, controllers, failure labels and statistics |
| `scripts/` | Data preparation, the experiment runner, analyses and reports |
| `configs/` | Experiment configurations for the development, held-out and later builds |
| `tests/` | Unit tests; leak tests, which check that a FAITHFUL selector's decisions never depend on outcomes it did not observe; and checks that pinned inputs and registered configurations are unchanged |

## Setup

Python 3.11 or later:

```bash
pip install -e ".[dev]"
pytest
```

The experiments read the LRTS dataset. Set three environment variables:

| Variable | Points to |
| --- | --- |
| `LRTS_REPO` | A clone of [LRTS](https://github.com/lrtsuser/LRTS) at the commit pinned in `tests/test_lrts.py` |
| `LRTS_DIR` | The `long_running_test_suites` folder from LRTS's processed release |
| `FAITHFULCUT_CACHE` | A writable folder for derived data |

With them set, `pytest` also checks the inputs. Build the derived data once:

```bash
python scripts/fetch_flaky_issues.py
python scripts/clone_repos.py
python scripts/build_ir.py
python scripts/build_dante.py
python scripts/build_labels.py
python scripts/build_structure.py
python scripts/build_later.py
python scripts/time_selectors.py
```

`python scripts/run_core.py` runs the held-out evaluation and its report. `PREREGISTRATION.md` and `PREREGISTRATION-B.md` list the commands for the later builds, and any configuration runs with `python scripts/run_experiments.py <config>`. The industrial replication reads the logs published in [atcs-data](https://bitbucket.org/HelgeS/atcs-data). The LLM ranking needs access to Gemini on Google Cloud.
