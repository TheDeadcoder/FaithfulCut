# Pre-registration, second stage

The commit that adds this file fixes:
- this file;
- the configurations in `configs/later-b`, whose hashes `tests/test_registration.py` pins;
- the analysis in `scripts/report_b.py`.

It adds arms for the later builds to the first registration (`PREREGISTRATION.md`) and leaves that registration's hypotheses, configurations and analysis unchanged. Each hypothesis here comes from an effect seen on the development split. They were chosen after the first registration's confirmatory results were known, and before any later build was evaluated. The later builds are evaluated once, after this registration, for both registrations. Any change after registration is disclosed with the results. Every result is reported, whatever its outcome.

## Data

- **Later builds:** as in the first registration. They are replayed after the studied builds as one timeline and evaluated at positions [1, 2).
- **Projects:** those with at least 500 later builds whose change relevance covers at least 95% of them, decided from build counts and inputs only: Hadoop (681 later builds), Jackrabbit Oak (794), James (1,949), Kafka (4,245) and TVM (5,094). Change relevance and structure cover 98.6–100% of their later builds.
- **Module graph:** TVM has none, so the hypotheses on module orders use the other four.
- **James's inputs:** 483 of its later builds lacked their base commit in our clone. Before this registration, the head refs of their 147 pull requests were fetched, which recovered the inputs of all but 7 of them.

## Measures

As in the first registration, plus:
- **New-failure label:** a failure of a class never seen failing in a warm-up or coin-audited build that finished before the build started. It is computable in deployment and stands in for FirstFail, which the later builds lack.
- **Selection work:** the test time the base selection runs over full-suite test time, without audits, forced or scheduled full runs and rotated classes.

## Settings

As in the first registration, plus these arms. Controllers run online with α = 0.10, π = 0.10 and 30 seeds; price grids use the first registration's budgets and 10 seeds.
- **New-failure target:** the controller tracks new-failure escape instead of operational escape.
- **Structural orders,** ties broken by duration:
  - Module: reverse-dependency distance from the changed modules in the module graph at the build's base commit (Maven POMs; Kafka's Gradle projects).
  - Path: directory levels from the test up to the nearest ancestor it shares with a changed file.
  - Module+TF-IDF: reciprocal rank fusion of Module with TF-IDF+CC.
- **Relevance threshold:** the cut covers a share of the change's TF-IDF relevance mass instead of a share of predicted test time. λ is −log of the uncovered share, floored at 0.001. A build without relevance runs everything.
- **Path filter:** a change touching only documentation and test sources runs just its changed and new tests.
- **Repair,** neither of which reaches the controller:
  - after the warm-up, a full run every 8 builds, at a phase set by the seed;
  - stalest-first rotation, which also runs the unselected classes observed longest ago, within 10% of the predicted suite cost.
- **Learned orders:**
  - Learned+CC is a gradient-boosted failure model: scikit-learn's HistGradientBoostingClassifier with default settings, except that early stopping is off (100 iterations), seeded.
  - Its ten features are change relevance (raw and as a share of the build's highest), co-failures with the changed files, observed failures, runs, smoothed failure rate, runs since the last failure, observations since last seen, log duration, and the number of changed files.
  - It trains on the last 500 observed builds and refits after every 100. It keeps passing rows with probability 0.1, each weighted 10.
  - It orders classes by predicted failure probability per predicted second.
  - Learned-W+CC also weights each observation by the inverse of its probability of being observed: 1/π for a class seen only through an audit.

## Primary hypotheses

| | Claim | Data | Test |
| --- | --- | --- | --- |
| B1 | Certified on new failures, a hybrid order needs less work than history. | Online controller tracking new failures. | Hybrid-RRF minus LatestFail+CC: work < 0, and new-failure escape < +0.02. Intersection-union: p is the larger of the two one-sided p-values. |
| B2 | Reading the change beats the module graph. | Online controller, operational target; the four projects with a module graph. | Module minus TF-IDF+CC: work > 0. |
| B3 | Letting the change's relevance set the budget saves work. | Online controller, operational target. | Hybrid-RRF with the relevance threshold minus with the cost cut: work < 0, and operational escape < +0.02. Intersection-union. |
| B4 | Restoring observations repairs history. | Price grid, FAITHFUL without audits. | LatestFail+CC: operational escape at 20% selection work with a full run every 8 builds minus without < 0. |
| B5 | A learned order also pays for censoring. | Price grid without audits. | Learned+CC: macro FAITHFUL/FULL operational escape at 20% executed work > 1. |

Differences are macro means over the projects. Work is executed work at equal risk; escapes are on the controller's label.

## Secondary hypotheses

- In B1's setting: TF-IDF+CC minus LatestFail+CC, work < 0.
- In B2's setting: Path minus TF-IDF+CC, work > 0, on all five projects; Module+TF-IDF minus TF-IDF+CC, work > 0.
- In B3's setting: TF-IDF+CC with the relevance threshold minus with the cost cut, work < 0; Hybrid-RRF with the path filter minus without, work > 0.
- In B4's setting: rotation minus none, escape < 0.
- In B5's setting: the ratio of Learned+CC minus that of LatestFail+CC < 0; with 10% audits (FAITHFUL), Learned-W+CC minus Learned+CC, escape at 20% executed work differs from 0 (two-sided).

## Statistics

As in the first registration:
- 10,000 resamples of whole pull requests over each project's evaluated builds, seed 0.
- A macro averages projects within each replicate.
- Intervals are 95% percentile intervals.
- A one-sided p is (replicates on the wrong side + 1) / (valid replicates + 1); a two-sided p doubles the smaller side.

Holm at 0.05 runs within the primary family (B1–B5) and within the secondary family, apart from the first registration's families. A hypothesis is supported when its Holm-adjusted p is below 0.05.

## Reported without tests

- Each arm's work and escape per project.
- Hidden-failure latency with and without repair.
- The learned orders' price of timing and of censoring, and their work and escape under the online controller (10 seeds).

## Running

After the first registration's later configurations:

```bash
for c in configs/later-b/*.toml; do python scripts/run_experiments.py "$c"; done
```

```bash
python scripts/report_b.py --price data/results/later/price-main --repair data/results/later-b/price-repair --learned data/results/later-b/price-learned data/results/later-b/price-learned-w --control data/results/later/control-main data/results/later-b/control-newfail data/results/later-b/control-structure data/results/later-b/control-threshold data/results/later-b/control-filter --evaluate 1.0 2.0 --out data/results/later-b/report.md
```

```bash
python scripts/learned_selector.py --price data/results/later/price-main data/results/later-b/price-learned data/results/later-b/price-learned-w --control data/results/later/control-main data/results/later-b/control-learned --evaluate 1.0 2.0 --replicates 10000 --out data/results/later-b/learned.md
```
