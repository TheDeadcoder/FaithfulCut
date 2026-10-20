from functools import cache, lru_cache

import numpy as np

from faithfulcut import dante
from faithfulcut.control import (FixedCut, MetaCut, Monitor, OnlineCut, SplitCut, ht_widths, max_pending,
                                 transfer_widths)
from faithfulcut.ir import load_ir
from faithfulcut.metrics import first_failure_time
from faithfulcut.labels import cold_rows, load_operational, transfer
from faithfulcut.replay import replay
from faithfulcut.scorers import history_scorers, hybrid_scorers, ir_scorers
from faithfulcut.suite import Suite, load_suite


@lru_cache(maxsize=2)
def setup(project: str, seed: int, evaluate: tuple[float, float]):
    """The suite up to the end of the evaluated span, its labels, the scorers and the evaluated builds."""
    full = load_suite(project, later=evaluate[1] > 1)
    suite = full.before(evaluate[1])
    rows = suite.offsets[-1]
    labels = {"operational": load_operational(full, seed)[:rows]}
    labels.update({k: v[:rows] for k, v in transfer(full).items()})
    labels["firstfail_cold"] = labels["firstfail"] & cold_rows(suite)
    scorers = {**history_scorers(suite), **ir_scorers(suite, load_ir(full, "seen"))}
    scorers.update(hybrid_scorers(scorers["LatestFail+CC"], scorers["TF-IDF+CC"]))
    if project in dante.PROJECTS:
        relevance = cache(lambda: dante.load_relevance(full))
        scorers.update(dante.dante_scorers(suite, relevance, dante.stage_counts(full)))
    position = suite.builds["position"].to_numpy()
    return suite, labels, scorers, (position >= evaluate[0]) & (position < evaluate[1])


def available(project: str, scorer: str) -> bool:
    return not scorer.startswith("DANTE") or project in dante.PROJECTS


FULL_RUN_BIT = 15
FLAG_SHARE = 0.5


@lru_cache(maxsize=2)
def change_signal(project: str, later: bool = False):
    """Share of the build's TF-IDF relevance mass that falls on classes outside its cut."""
    ir = load_ir(load_suite(project, later), "seen")

    def signal(t: int, depth_by_class: np.ndarray, lam: float) -> float:
        part = slice(ir["offsets"][t], ir["offsets"][t + 1])
        score = ir["score"][part]
        total = score.sum()
        return float(score[depth_by_class[ir["cls"][part]] > lam].sum() / total) if total > 0 else 0.0

    return signal


def summarize(suite: Suite, rep, labels: dict[str, np.ndarray], evaluate: np.ndarray, timing: bool = False,
              per_build: bool = False) -> dict:
    """Test work, feedback time and selector time and, per label, faulty builds, escapes, escapes masked by another
    selected failure and selected failing rows over the evaluated builds; `timing` adds the time to first failure.
    `per_build` adds, under "_builds", each evaluated build's test time, change signal and flags: for the i-th label,
    bits 3i, 3i+1 and 3i+2 mark faulty, caught by the selection and masked; bit FULL_RUN_BIT marks a full run."""
    of_row = suite.build_of_rows()
    n = len(suite.builds)
    full_run = rep.audited | rep.forced
    coin = rep.audited & (rep.reason != 1)
    with_warmup = evaluate | (rep.reason == 1)
    out = {
        "builds": int(evaluate.sum()),
        "work": rep.executed_cost[evaluate].sum() / rep.full_cost[evaluate].sum(),
        "base_work": rep.base_cost[evaluate].sum() / rep.full_cost[evaluate].sum(),
        "work_with_warmup": rep.executed_cost[with_warmup].sum() / rep.full_cost[with_warmup].sum(),
        "forced": int((rep.forced & evaluate).sum()),
        "lam_median": float(np.median(rep.lam[evaluate])),
        "verdict_hours": rep.executed_cost[evaluate].mean() / 3600,
        "full_hours": rep.full_cost[evaluate].mean() / 3600,
        "decision_seconds": rep.seconds[evaluate].mean(),
        "audit_rate": coin[evaluate].mean(),
        "audit_work": (rep.full_cost - rep.base_cost)[evaluate & coin & ~rep.forced].sum()
        / rep.full_cost[evaluate].sum(),
    }
    flagged = rep.signal >= FLAG_SHARE
    if not np.isnan(rep.signal[evaluate]).all():
        out["flagged"] = int((flagged & evaluate).sum())
    flags = (full_run.astype(np.uint16) << FULL_RUN_BIT)
    for i, (name, label) in enumerate(labels.items()):
        faulty = (np.bincount(of_row, weights=label, minlength=n) > 0) & evaluate
        caught = np.bincount(of_row, weights=label & rep.selected, minlength=n) > 0
        other = np.bincount(of_row, weights=suite.failed & ~label & rep.selected, minlength=n) > 0
        rows = label & evaluate[of_row]
        masked = faulty & ~caught & other
        flags |= (faulty.astype(np.uint16) << 3 * i) | (caught.astype(np.uint16) << 3 * i + 1) \
            | (masked.astype(np.uint16) << 3 * i + 2)
        out.update({
            f"{name}_faulty": int(faulty.sum()),
            f"{name}_escape": int((faulty & ~caught).sum()),
            f"{name}_executed_escape": int((faulty & ~caught & ~full_run).sum()),
            f"{name}_masked": int(masked.sum()),
            f"{name}_flagged_escape": int((faulty & ~caught & flagged).sum()),
            f"{name}_audited": int((faulty & coin).sum()),
            f"{name}_audited_escape": int((faulty & coin & ~caught).sum()),
            f"{name}_rows": int(rows.sum()),
            f"{name}_rows_selected": int((rows & rep.selected).sum()),
        })
        if timing:
            ttff = first_failure_time(suite, rep, label, faulty)
            found = ~np.isnan(ttff)
            out[f"{name}_ttff_hours"] = ttff[found].mean() / 3600 if found.any() else np.nan
            out[f"{name}_ttff_share"] = (ttff[found] / rep.full_cost[found]).mean() if found.any() else np.nan
    if per_build:
        idx = np.flatnonzero(evaluate)
        out["_builds"] = {"labels": list(labels), "build": idx.astype(np.int32),
                          "executed": rep.executed_cost[idx].astype(np.float32),
                          "full": rep.full_cost[idx].astype(np.float32), "flags": flags[idx],
                          "signal": rep.signal[idx].astype(np.float32)}
    return out


def price_task(project: str, scorer: str, regime: str, pi: float, seed: int, lambdas: list[float],
               evaluate: tuple[float, float], retire_after: int | None, per_build: bool = False) -> list[dict]:
    suite, labels, scorers, mask = setup(project, seed, evaluate)
    return [
        {"project": project, "scorer": scorer, "regime": regime, "pi": pi, "seed": seed, "lam": lam,
         **summarize(suite, replay(suite, regime, scorers[scorer], lam=lam, pi=pi, seed=seed,
                                   retire_after=retire_after), labels, mask, per_build=per_build)}
        for lam in lambdas
    ]


def make_controller(kind: str, alpha: float, cfg: dict, project: str = "", scorer: str = ""):
    """The cut, its feedback mode, its monitor and its size cap for one controller kind; "tuned" is the fixed budget
    tuned on the dev split for the project, order and alpha."""
    if kind == "online":
        return OnlineCut(alpha, cfg["eta"]), "audits", Monitor(alpha), cfg["cap"]
    if kind == "oracle":
        return OnlineCut(alpha, cfg["eta"]), "all", None, np.inf
    if kind == "split":
        return SplitCut(alpha, cfg["split_window"]), "audits", None, cfg["cap"]
    if kind == "pac":
        return SplitCut(alpha, cfg["pac_window"], cfg["pac_delta"]), "audits", None, cfg["cap"]
    if kind == "meta":
        return MetaCut(period=cfg["meta_period"]), "audits", None, np.inf
    if kind == "tuned":
        return FixedCut(cfg["tuned"][(project, scorer, alpha)]), "audits", None, cfg["cap"]
    if kind.startswith("fixed-"):
        return FixedCut(float(kind.removeprefix("fixed-"))), "audits", None, cfg["cap"]
    raise ValueError(f"unknown controller {kind}")


def control_task(project: str, scorer: str, kind: str, alpha: float, pi: float, seed: int, cfg: dict,
                 evaluate: tuple[float, float]) -> list[dict]:
    """One controller replay; a kind ending in "+adaptive" audits by the change signal instead of uniformly."""
    suite, labels, scorers, mask = setup(project, seed, evaluate)
    base, _, audits = kind.partition("+")
    cut, feedback, monitor, cap = make_controller(base, alpha, cfg, project, scorer)
    rep = replay(suite, "faithful", scorers[scorer], lam=cut, pi=pi, seed=seed, label=labels["operational"],
                 feedback=feedback, cap=cap, monitor=monitor, retire_after=cfg.get("retire_after"),
                 signal=change_signal(project, evaluate[1] > 1), adaptive=audits == "adaptive")
    row = {"project": project, "scorer": scorer, "controller": kind, "alpha": alpha, "pi": pi, "seed": seed,
           **summarize(suite, rep, labels, mask, timing=True, per_build=cfg.get("per_build", False))}
    if isinstance(cut, OnlineCut):
        of_row = suite.build_of_rows()
        faulty = np.bincount(of_row, weights=labels["operational"], minlength=len(suite.builds)) > 0
        updates = rep.audited & faulty & (rep.reason != 1)
        start, finish = suite.builds["start"].to_numpy(), suite.builds["finish"].to_numpy()
        row.update({"updates": len(cut.losses), "pending": max_pending(start[updates], finish[updates]),
                    "lam_min": min(cut.path), "lam_max": max(cut.path)})
    if row["operational_faulty"]:
        widths = transfer_widths(row["operational_faulty"], row["operational_audited_escape"], pi)
        row.update({f"width_{k}": v for k, v in widths.items()})
        of_row, n = suite.build_of_rows(), len(suite.builds)
        faulty = np.flatnonzero((np.bincount(of_row, weights=labels["operational"], minlength=n) > 0) & mask)
        caught = np.bincount(of_row, weights=labels["operational"] & rep.selected, minlength=n) > 0
        coin = rep.audited & (rep.reason != 1)
        ht = ht_widths(~caught[faulty], coin[faulty], rep.pi[faulty])
        row.update({"escape_ht": ht["estimate"], "width_ht_normal": ht["normal"], "width_ht_worst": ht["worst"]})
    return [row]
