import json

import numpy as np
import pandas as pd
import pytest
from synthetic import make_suite, random_suite

from faithfulcut.analysis import (at_matched_work, at_work, break_even_hours, cluster_weights, compare,
                                  condition_sums, energy, holm, kendall_tau, load_outcomes, one_sided, pooled_share,
                                  ranking_flips, replicate_rates, significant_flips, tuned_budgets, work_at_rate)
from faithfulcut.control import FixedCut
from faithfulcut.experiments import is_documentation, make_controller, summarize
from faithfulcut.labels import cold_rows, new_failures
from faithfulcut.llm import risk_levels, timed_state
from faithfulcut.metrics import first_failure_time
from faithfulcut.replay import replay
from faithfulcut.scorers import qtf


def test_before_keeps_only_builds_below_the_position():
    s = random_suite(n_builds=50)
    head = s.before(0.7)
    n = int((s.builds["position"] < 0.7).sum())
    assert len(head.builds) == n == 35
    assert np.array_equal(head.cls, s.cls[:s.offsets[n]])
    assert np.array_equal(head.changed(n - 1), s.changed(n - 1))


def test_matched_work_interpolates_between_lambda_points():
    rows = []
    for lam, work, escapes in ((0.1, 0.1, 6), (0.3, 0.3, 2)):
        row = {"project": "p", "scorer": "s", "regime": "full", "pi": 0.0, "seed": 0, "lam": lam, "work": work}
        for label in ("operational", "deconf", "firstfail"):
            row.update({f"{label}_faulty": 10, f"{label}_escape": escapes, f"{label}_executed_escape": escapes,
                        f"{label}_audited": 0, f"{label}_audited_escape": 0, f"{label}_rows": 20,
                        f"{label}_rows_selected": 20 - 2 * escapes})
        rows.append(row)
    matched = at_work(pd.DataFrame(rows), targets=(0.2,))
    assert matched["operational_escape_rate"].item() == pytest.approx(0.4)
    assert matched["operational_recall"].item() == pytest.approx(0.6)


def test_timed_state_reads_only_builds_finished_before_the_start():
    rows = [(np.array([0]), np.array([True]), np.array([5.0])), (np.array([1]), np.array([False]), np.array([7.0])),
            (np.array([0, 1]), np.array([False, False]), np.array([1.0, 1.0]))]
    s = make_suite(rows, start=np.array([0.0, 1.0, 2.0]), finish=np.array([1.5, 10.0, 3.0]))
    catalog, duration, since_fail = timed_state(s, [2])[2]
    assert catalog.tolist() == [0] and duration.tolist() == [5.0] and since_fail.tolist() == [1.0]


def test_llm_answers_rank_named_classes_and_flag_malformed_output():
    names = np.array(["a.ATest", "a.BTest", "a.CTest"])
    text = json.dumps({"tests": [{"name": "a.BTest", "risk": "high"}, {"name": "x.Unknown", "risk": "high"},
                                 {"name": "a.ATest", "risk": "LOW"}]})
    ranks, ok = risk_levels({"candidates": [{"content": {"parts": [{"text": text}]}}]}, names)
    assert ok and ranks.tolist() == [2, 0, 3]
    ranks, ok = risk_levels({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}, names)
    assert not ok and ranks.tolist() == [3, 3, 3]


def two_builds(failing, durations):
    cls, duration = np.arange(len(durations)), np.asarray(durations, dtype=float)
    rows = [(cls, np.zeros(len(cls), dtype=bool), duration), (cls, np.isin(cls, failing), duration)]
    return make_suite(rows, start=np.array([0.0, 10.0]), finish=np.array([1.0, 11.0]))


def test_first_failure_time_adds_test_time_in_run_order():
    s = two_builds([0], [3.0, 1.0, 2.0])
    rep = replay(s, "full", qtf, lam=1.0, warmup=0.0)
    assert first_failure_time(s, rep, s.failed, np.array([False, True]))[1] == 6.0


def test_an_escape_is_masked_when_another_selected_class_fails():
    s = two_builds([0, 1], [5.0, 1.0])
    label = np.array([False, False, True, False])
    rep = replay(s, "full", qtf, lam=0.2, warmup=0.0)
    out = summarize(s, rep, {"x": label}, np.array([False, True]), timing=True)
    assert (out["x_faulty"], out["x_escape"], out["x_masked"]) == (1, 1, 1)
    assert np.isnan(out["x_ttff_hours"]) and out["verdict_hours"] == 1.0 / 3600


def test_cold_rows_count_runs_in_builds_finished_before_the_start():
    rows = [(np.array([0]), np.array([False]), np.array([1.0]))] * 6
    rows.append((np.array([0, 1]), np.zeros(2, bool), np.ones(2)))
    start = np.arange(7) * 10.0
    s = make_suite(rows, start=start, finish=start + 1)
    assert cold_rows(s).tolist() == [False, True, True, True, True, False, False, False]
    late = make_suite(rows, start=start, finish=np.where(np.arange(7) == 4, 100.0, start + 1))
    assert cold_rows(late)[5]


def test_energy_and_break_even_scale_linearly():
    e = energy(10.0)
    low = 10 * 2 * ((0.74 + 0.5 * 2.76) / 1000 + 4 * 0.000392) * 1.135
    assert e["kwh_low"] == pytest.approx(low) and e["kg_co2e_low"] == pytest.approx(low * 0.458)
    assert e["dollars_low"] == pytest.approx(3.6) and e["dollars_high"] == pytest.approx(14.4)
    assert energy(20.0)["kwh_high"] == pytest.approx(2 * e["kwh_high"])
    assert break_even_hours(36.0, 0.5) == pytest.approx(0.02) and break_even_hours(1.0, 1.0) == np.inf


def test_per_build_flags_record_faulty_caught_masked_and_full_runs():
    s = two_builds([0, 1], [5.0, 1.0])
    label = np.array([False, False, True, False])
    rep = replay(s, "full", qtf, lam=0.2, warmup=0.0)
    builds = summarize(s, rep, {"x": label, "all": s.failed}, np.array([False, True]), per_build=True)["_builds"]
    assert builds["labels"] == ["x", "all"] and builds["build"].tolist() == [1]
    flags = int(builds["flags"][0])
    assert [(flags >> b) & 1 for b in range(6)] == [1, 0, 1, 1, 1, 0] and not flags >> 15
    assert builds["executed"].tolist() == [1.0] and builds["full"].tolist() == [6.0]
    assert builds["base"].tolist() == [1.0]


def outcome_rows(escapes, faulty=4, lams=(0.1, 0.3), works=(0.1, 0.3)):
    rows, parts = [], []
    for seed in range(2):
        for lam, work, n_escaped in zip(lams, works, escapes):
            flags = np.zeros(faulty + 1, np.uint16)
            flags[:faulty] = 1 | (np.arange(faulty) >= n_escaped).astype(np.uint16) << 1
            parts.append({"row": np.full(faulty + 1, len(rows)), "build": np.arange(faulty + 1),
                          "executed": np.full(faulty + 1, work, np.float32), "full": np.ones(faulty + 1, np.float32),
                          "flags": flags})
            rows.append({"scorer": "s", "seed": seed, "lam": lam})
    outcomes = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return pd.DataFrame(rows), {**outcomes, "labels": np.array(["x"])}


def test_bootstrap_rates_interpolate_at_matched_work_for_every_replicate():
    rows, outcomes = outcome_rows(escapes=(3, 1))
    sums = condition_sums(rows, outcomes, "x", scorer="s")
    assert sums["faulty"].sum() == 16 and sums.loc[sums["lam"] == 0.1, "escaped"].sum() == 6
    weights = cluster_weights(np.arange(5), replicates=200, seed=1)
    assert np.all(weights[0] == 1) and np.all(weights.sum(axis=1) == 5)
    lams, rate, work = replicate_rates(sums, np.arange(5), weights)
    assert rate[:, 0] == pytest.approx([0.75, 0.25]) and work[:, 0] == pytest.approx([0.1, 0.3])
    assert at_matched_work(rate, work, 0.2)[0] == pytest.approx(0.5)


def test_work_at_a_rate_inverts_the_interpolated_curve():
    rows, outcomes = outcome_rows(escapes=(3, 1))
    _, rate, work = replicate_rates(condition_sums(rows, outcomes, "x", scorer="s"), np.arange(5),
                                    cluster_weights(np.arange(5), replicates=50, seed=1))
    assert work_at_rate(rate, work, at_matched_work(rate, work, 0.2))[0] == pytest.approx(0.2)
    point = (rate[:, :1], work[:, :1])
    assert work_at_rate(*point, np.array([0.25]))[0] == pytest.approx(0.3)
    assert np.isnan(work_at_rate(*point, np.array([0.8]))[0]) and np.isnan(work_at_rate(*point, np.array([0.1]))[0])


def test_pooled_share_weighs_units_over_strata():
    a, b = (np.array([[1.0, 1.0], [2.0, 0.0]]), [1, 0]), (np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]), [1, 1, 0])
    assert pooled_share([a, b]).tolist() == pytest.approx([3 / 5, 1.0])
    assert np.isnan(pooled_share([b])[1])


def test_compare_and_holm():
    a = np.random.default_rng(0).normal(0.3, 0.01, 1001)
    same = compare(a, a)
    assert same["difference"] == 0 and same["p"] == 1.0
    apart = compare(a, a - 0.1)
    assert apart["low"] > 0 and apart["p"] < 0.01
    assert holm(np.array([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_tuned_budget_is_the_smallest_lambda_meeting_alpha():
    rows = []
    for seed in range(2):
        for lam, escaped in ((0.1, 5), (0.2, 3), (0.3, 1), (0.4, 2)):
            row = {"project": "p", "scorer": "s", "regime": "faithful", "pi": 0.1, "seed": seed, "lam": lam}
            for label in ("operational", "deconf", "firstfail"):
                row.update({f"{label}_faulty": 10, f"{label}_escape": escaped, f"{label}_executed_escape": escaped,
                            f"{label}_audited": 0, f"{label}_audited_escape": 0, f"{label}_rows": 10,
                            f"{label}_rows_selected": 10 - escaped})
            rows.append(row)
    tuned = tuned_budgets(pd.DataFrame(rows), [0.4, 0.2, 0.05]).set_index("alpha")["lam"]
    assert tuned[0.4] == pytest.approx(0.15) and tuned[0.2] == pytest.approx(0.25) and tuned[0.05] == 1.0
    cut = make_controller("tuned", 0.2, {"tuned": {("p", "s", 0.2): 0.25}, "cap": 0.8}, "p", "s")[0]
    assert isinstance(cut, FixedCut) and cut.decide(0, False) == 0.25


def test_builds_skipping_most_change_relevance_are_flagged_for_review():
    s = two_builds([0, 1], [5.0, 1.0])
    label = np.array([False, False, True, False])
    rep = replay(s, "full", qtf, lam=0.2, warmup=0.0, signal=lambda t, depth, lam: 0.9 if t == 1 else 0.1)
    out = summarize(s, rep, {"x": label}, np.array([False, True]))
    assert out["flagged"] == 1 and out["x_flagged_escape"] == 1 and out["audit_rate"] == 0.0


def test_ranking_flips_count_pairs_that_change_order():
    rows = []
    for project, values in (("p", {"a": (0.1, 0.3), "b": (0.2, 0.2), "c": (0.3, 0.4)}),
                            ("q", {"a": (0.1, 0.1), "b": (0.2, 0.2), "c": (0.3, 0.3)})):
        for scorer, (full, faithful) in values.items():
            for regime, value in (("full", full), ("faithful", faithful)):
                rows.append({"project": project, "scorer": scorer, "regime": regime, "pi": 0.0, "work": 0.2,
                             "operational_escape_rate": value})
    flips = ranking_flips(pd.DataFrame(rows)).set_index("project")
    assert flips.loc["p", "flips"] == 1 and flips.loc["p", "flipped"] == "a / b"
    assert flips.loc["q", "flips"] == 0 and flips.loc["q", "tau"] == pytest.approx(1.0)
    assert flips.loc["macro", "orders"] == 3


def test_one_sided_p_counts_replicates_on_the_wrong_side():
    reps = np.concatenate([[1.4], np.linspace(0.9, 1.9, 999)])
    assert one_sided(reps, above=1.0) == pytest.approx((np.sum(np.linspace(0.9, 1.9, 999) <= 1.0) + 1) / 1000)
    assert one_sided(reps, below=0.5) == 1.0


def test_kendall_tau_per_replicate():
    a = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    b = np.array([[0.1, 0.3], [0.2, 0.2], [0.3, 0.1]])
    assert kendall_tau(a, b).tolist() == [1.0, -1.0]


def test_only_pairs_significant_under_full_count_as_reversed_or_lost():
    rng = np.random.default_rng(0)
    noise = lambda: rng.normal(0, 0.01, 1001)
    full = {"a": 0.1 + noise(), "b": 0.3 + noise(), "c": 0.31 + noise()}
    faithful = {"a": 0.4 + noise(), "b": 0.3 + noise(), "c": 0.3 + noise()}
    flips = significant_flips(full, faithful).set_index("pair")["status"]
    assert flips.to_dict() == {"a / b": "reversed", "a / c": "reversed"}
    faithful["a"] = 0.3 + noise()
    assert significant_flips(full, faithful).set_index("pair")["status"].to_dict() == {"a / b": "lost",
                                                                                      "a / c": "lost"}


def test_outcomes_from_several_folders_keep_their_rows(tmp_path):
    for i, name in enumerate(("one", "two")):
        folder = tmp_path / name
        folder.mkdir()
        rows, outcomes = outcome_rows(escapes=(3 - i, 1))
        rows.assign(scorer=name).to_csv(folder / "price.csv", index=False)
        np.savez(folder / "price_builds.npz", **outcomes, **({"base": outcomes["executed"]} if i else {}))
    rows, outcomes = load_outcomes([tmp_path / "one", tmp_path / "two"], "price")
    assert len(rows) == 8 and outcomes["row"].max() == 7 and "base" not in outcomes
    assert "base" in load_outcomes([tmp_path / "two"], "price")[1]
    np.savez(tmp_path / "two" / "price_builds.npz", **{**outcomes, "labels": np.array(["y"])})
    with pytest.raises(ValueError, match="different labels"):
        load_outcomes([tmp_path / "one", tmp_path / "two"], "price")
    two = condition_sums(rows, outcomes, "x", scorer="two")
    assert two.loc[two["lam"] == 0.1, "escaped"].sum() == 4


def test_new_failures_are_classes_never_seen_failing_in_a_finished_full_run():
    two = np.array([0, 1])
    rows = [(two, np.array([True, False]), np.ones(2)), (two, np.array([True, True]), np.ones(2)),
            (two, np.array([False, True]), np.ones(2)), (np.arange(3), np.array([True, True, True]), np.ones(3))]
    start = np.arange(4) * 10.0
    s = make_suite(rows, start, start + 1)
    assert new_failures(s, s.failed, seed=0, pi=1.0).tolist() == [True, False, False, True, False, False, False,
                                                                   False, True]
    assert new_failures(s, s.failed, seed=0, pi=0.0).tolist() == [True, False, False, True, False, True, False,
                                                                   True, True]
    rows = [(two, np.array([False, False]), np.ones(2)), (two, np.array([True, False]), np.ones(2)),
            (two, np.array([True, False]), np.ones(2))]
    start = np.arange(3) * 10.0
    late = make_suite(rows, start, np.array([1.0, 25.0, 21.0]))
    assert new_failures(late, late.failed, seed=0, pi=1.0)[4:].tolist() == [True, False]


def test_documentation_is_text_images_and_doc_folders_but_not_code_or_config():
    assert all(map(is_documentation, ["README.md", "docs/a/b.java", "x/src/site/page.xml", "LICENSE", "NOTICE.txt"]))
    assert not any(map(is_documentation, ["pom.xml", "core/src/main/java/A.java", "conf/log4j.properties"]))
