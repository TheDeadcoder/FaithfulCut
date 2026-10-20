# Pre-registration

The commit that adds this file fixes:
- this file;
- the configurations in `configs/confirmatory` and `configs/later`, whose hashes `tests/test_registration.py` pins;
- the budgets in `configs/tuned_budgets.csv`;
- the analysis in `scripts/report.py` and `scripts/run_core.py`.

Neither the confirmatory split nor the later builds had been evaluated when it was made. The confirmatory split is evaluated once, after this commit. The later builds are evaluated once, after a second registration adds further arms, which leaves everything here unchanged. Any change after registration is disclosed with the results. Every result is reported, whatever its outcome.

## Data

- **Studied builds:** the runs in LRTS's `dataset.csv` (pinned in the README), in start order per project, at positions [0, 1).
  - Warm-up [0, 0.2): every test runs.
  - Development [0.2, 0.7): every design choice was made on it.
  - Confirmatory [0.7, 1.0): earlier builds serve only as history.
- **Later builds:** the processed release's builds that start after each project's last studied build. They are replayed after the studied builds as one timeline, at positions from 1 on. Log4j has none. Only the operational label exists there.
- **Projects:** ActiveMQ, Hadoop, HBase, Hive, Jackrabbit Oak, James, Kafka, Karaf, Log4j and TVM. Projects enter each analysis by development counts only:
  - **Tracking set:** at least 250 operationally failing development builds: Hadoop, Hive, Jackrabbit Oak, James, Kafka, TVM.
  - **First-failure set:** at least 10 development builds with a FirstFail failure: every project except Log4j.
  - **Java set:** DANTE's nine projects, every project except TVM.

## Measures

- **Labels:**
  - **Operational** (primary, and the controller's target). A class fails operationally when it fails in every stage it ran in, unless either of these held when the build started:
    - an LRTS flaky-test issue was open for it;
    - over the last 100 finished builds of a 5% uniform audit stream, it failed more often than the warm-up's mean failure rate plus three standard deviations.
    It is computable in deployment, and there is one label per audit seed, shared by every policy.
  - **DeConf and FirstFail:** LRTS's labels for the studied builds. For TVM, a class must fail in every stage where it ran.
- **Build escape:** a build with a labelled failure escapes when its base selection (the cut, before any audit or forced full run) selects none of its labelled failing classes. The escape rate is escapes over faulty builds, pooled over seeds.
- **Executed work:** executed test time over full-suite test time on the evaluated builds. Executed time counts the selection, audits and forced full runs. Price grids interpolate escape at 20% executed work over the budgets 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7 and 0.85.
- **Net work saved:** one minus executed work, minus selector time over full-suite test time. Selector time is the replay's ordering time plus, for orders that read the change, the per-build relevance time that `scripts/time_selectors.py` measures.
- **Certificate:** an interval for the full-stream base escape rate, built only from what a deployment sees: caught builds and audited misses. Its adaptive half-width (Freedman, δ = 0.05) is defined in `transfer_widths`. A seed misses when its true rate falls outside the interval.

## Settings

- **Regimes:**
  - FULL: outcomes of every earlier-started build.
  - TIMED: every finished build.
  - FAITHFUL: only tests the policy ran, from finished builds.
  - Audits run the full suite with probability π, from coins shared by every policy with the same seed.
- **Orders:**
  - History: LatestFail, LatestFail+CC, FailCount+CC and TF-FailFreq+CC, the last two also audit-weighted (-W).
  - Outcome-free: QTF, which reads durations, and TF-IDF and TF-IDF+CC, which read the change.
  - DANTE: history weight 1.0, change score active with weight 0.5 and power 1.2. DANTE-H is its released configuration, without the change score. Both also come audit-weighted (-W).
  - Hybrid-RRF: reciprocal rank fusion of LatestFail+CC and TF-IDF+CC, k = 60.
  - Hybrid: the two sharing the budget equally.
- **Controllers:**
  - Online: η = 0.2, size cap 0.8, warm-up 0.2. Its drift monitor uses a normal mixture, δ = 0.05, ρ = 5, and runs the full suite for 20 builds after an alarm.
  - Dev-tuned fixed budget: the smallest λ whose development escape reaches α under FAITHFUL with 10% audits (`configs/tuned_budgets.csv`).
  - Also: oracle, split CRC (19 builds), PAC (59 builds, δ = 0.05), fixed budgets, Meta-style (every 20 faulty builds), and audits allocated by the change signal.
- **Seeds:** 30 for `price-main` and `control-main`, 10 elsewhere.
- **LLM:** `gemini-3.5-flash-lite` with the pilot's prompt, ordered by its risk levels and then TF-IDF+CC. This is the "risk" fusion; all three fusions tied at 5% on development.

## Statistics

- **Bootstrap:**
  - 10,000 resamples of whole pull requests over each project's evaluated builds, seed 0.
  - A macro averages projects within each replicate.
  - Intervals are 95% percentile intervals.
  - A one-sided p is (replicates on the wrong side + 1) / (valid replicates + 1).
- **Primary family:** H1, H2, H5, H6, H7 and H8, Holm-adjusted at 0.05. A registered hypothesis that is not run enters with p = 1.
- **Secondary family:** Holm-adjusted within itself.
- **Engineering criteria:** H3 and H4 carry no multiplicity correction.

## Primary hypotheses

| | Claim | Data | Test |
| --- | --- | --- | --- |
| H1 | Censoring raises the escapes of history orders. | Confirmatory split, all ten projects. LatestFail+CC, operational escape at 20% executed work, FAITHFUL without audits against FULL. | Macro FAITHFUL/FULL ratio > 1. |
| H2 | Outcome-free orders are largely immune. | As H1. | Ratio of LatestFail+CC minus ratio of TF-IDF+CC > 0. |
| H5 | Fusing content with history catches more first failures at equal risk, for no extra work. | Confirmatory split, first-failure set. Online controller, α = 0.10, π = 0.10. FirstFail escape and executed work, macro differences. | Hybrid-RRF minus LatestFail+CC: FirstFail escape < 0, and work < +0.02. Intersection-union: p is the larger of the two one-sided p-values. |
| H6 | The strongest full-information order also pays for censoring. | Confirmatory split, Java set. DANTE, as H1. | Macro FAITHFUL/FULL ratio > 1. |
| H7 | Censoring raises misses on industrial CI logs. Registered now, run later. | The Paint Control and IOF/ROL logs (ABB Robotics) and GSDTSR (Google): cycles in log order, the first 20% as warm-up in which every logged test runs. A test runs at its first appearance. Otherwise tests are taken in order while their logged durations fit within 20% of the cycle's total logged duration. FULL reads every earlier logged verdict; FAITHFUL only the verdicts of tests the order ran. Ties break at random over 10 seeds. Metric: missed failing verdicts over failing verdicts. | LatestFail: mean FAITHFUL rate over the three logs divided by the mean FULL rate > 1. Bootstrap over blocks of 10 consecutive cycles. |
| H8 | An LLM reading the change finds first failures that TF-IDF misses. Registered now, run later. | Every confirmatory FirstFail build of the ten projects whose failing classes are in the catalog of builds finished before its start. FirstFail recall: a first-failing class enters the first 5% of predicted test time. | Flash-Lite minus TF-IDF+CC > 0. A malformed answer falls back to TF-IDF+CC. |

A hypothesis is supported when its Holm-adjusted p is below 0.05.

## Engineering criteria

| | Criterion | Met when |
| --- | --- | --- |
| H3 | The online controller tracks α better than a budget tuned on development, and its certificate holds. Confirmatory split, tracking set, α = 0.10, π = 0.10, for LatestFail+CC and Hybrid-RRF. | Mean over the projects of \|escape − α\| is smaller online than tuned, and the adaptive certificate misses in at most 3 of 30 seeds on every project. |
| H4 | Selection pays after audits and its own time. Online controller, α = 0.10, π = 0.10, LatestFail+CC and Hybrid-RRF, on Hive, James and Kafka. | Lower 95% bound of net work saved ≥ 0.10 on each. |

## Secondary hypotheses

- H1 for LatestFail, FailCount+CC, TF-FailFreq+CC and DANTE-H: ratio > 1.
- In H5's setting, minus LatestFail+CC:
  - TF-IDF+CC and Hybrid: FirstFail escape < 0;
  - Hybrid-RRF and TF-IDF+CC: DeConf escape < 0.
- In H5's setting, on the Java projects of the first-failure set: DANTE minus LatestFail+CC, FirstFail escape > 0.
- H7 for FailCount (most observed failures first) and QTF: ratio > 1.

## Later builds

These are evaluated after the second registration, with the same metrics and settings:
- H1, H2 and H6, Holm-adjusted over the three;
- H3 and H4 as engineering criteria;
- the secondary H1 orders.

Projects are those with later builds.

## Reported without tests

- **Price of information:** the price of timing (TIMED/FULL) and of censoring (FAITHFUL/TIMED) per order, and ratios per project.
- **Ranking:** order pairs whose escape difference is significant under FULL and that reverse or lose significance under FAITHFUL, and Kendall's tau between the FULL and FAITHFUL rankings, with its interval.
- **Escape by label against α,** per project, order, α and π.
- **Cost and feedback:** feedback time, masking and monitor alarms; selector time, energy and cost conversions.
- **LLM control:** if H8 is supported, the same builds are re-queried with each build's diff replaced by the diff of another randomly chosen confirmatory build of the same project. This separates reading the change from remembering the project.
- **Certificates:** widths and coverage, normal and adaptive.
- **Controllers:** the audit-rate frontier (π 0.05 and 0.5), the other controllers and fixed budgets, change-allocated audits and review flags.
- **Sensitivity:** to η, the size cap and the warm-up.
- **Weighted orders:** the -W orders.

## Running

Confirmatory split:

```bash
python scripts/run_core.py
```

Later builds, after the second registration:

```bash
for c in configs/later/*.toml; do python scripts/run_experiments.py "$c"; done
```

```bash
python scripts/report.py --price data/results/later/price-main data/results/later/price-rest --control data/results/later/control-main --evaluate 1.0 2.0 --out data/results/later/core.md
```
