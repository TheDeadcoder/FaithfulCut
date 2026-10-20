from functools import cache, lru_cache

import numpy as np

from faithfulcut import dante
from faithfulcut.control import (Cusum, FixedCut, MetaCut, Monitor, OnlineCut, SplitCut, ht_widths, max_pending,
                                 transfer_widths)
from faithfulcut.ir import is_test_source, load_ir, lrts_pattern
from faithfulcut.metrics import escape_causes, first_failure_time, hidden_latency
from faithfulcut.labels import cold_rows, load_operational, new_failures, transfer
from faithfulcut.learned import learned_scorers
from faithfulcut.replay import replay
from faithfulcut.scorers import history_scorers, hybrid_scorers, ir_scorers, relevance, structure_scorers
from faithfulcut.structure import NO_MODULES, load_structure
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
    ir = load_ir(full, "seen")
    scorers = {**history_scorers(suite), **ir_scorers(suite, ir)}
    scorers.update(hybrid_scorers(scorers["LatestFail+CC"], scorers["TF-IDF+CC"]))
    scorers.update(structure_scorers(cache(lambda: load_structure(full)), scorers["TF-IDF+CC"]))
    scorers.update(learned_scorers(suite, relevance(suite, ir), seed))
    if project in dante.PROJECTS:
        dante_relevance = cache(lambda: dante.load_relevance(full))
        scorers.update(dante.dante_scorers(suite, dante_relevance, dante.stage_counts(full)))
    position = suite.builds["position"].to_numpy()
    return suite, labels, scorers, (position >= evaluate[0]) & (position < evaluate[1])


def available(project: str, scorer: str) -> bool:
    if scorer.startswith("DANTE"):
        return project in dante.PROJECTS
    return not scorer.startswith("Module") or project not in NO_MODULES


FULL_RUN_BIT = 15
FLAG_SHARE = 0.5
DOCUMENTATION = (".md", ".txt", ".rst", ".adoc", ".html", ".htm", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf")


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


@lru_cache(maxsize=2)
def change_mass(project: str, later: bool = False):
    """Each class's TF-IDF relevance to the build's change: the mass a relevance threshold covers."""
    full = load_suite(project, later)
    return relevance(full, load_ir(full, "seen"))


def is_documentation(path: str) -> bool:
    parts = path.split("/")
    return (path.lower().endswith(DOCUMENTATION) or bool({"docs", "doc", "site"} & set(parts[:-1]))
            or parts[-1].split(".")[0].upper() in {"LICENSE", "NOTICE", "README", "CHANGES", "CHANGELOG"})


@lru_cache(maxsize=2)
def path_filter(project: str, later: bool = False):
    """For a build whose change touches only documentation and test sources, a mask of its changed tests' classes;
    None for any other build."""
    full = load_suite(project, later)
    stems: dict[str, list[int]] = {}
    for c, name in enumerate(full.names):
        stems.setdefault(lrts_pattern(project, name).rsplit("/", 1)[-1].split(".")[0], []).append(c)

    def allowed(t: int) -> np.ndarray | None:
        files = full.files[full.changed(t)].tolist()
        if not files or not all(is_documentation(f) or is_test_source(f) for f in files):
            return None
        out = np.zeros(len(full.names), dtype=bool)
        for f in files:
            if is_test_source(f):
                out[stems.get(f.rsplit("/", 1)[-1].split(".")[0], [])] = True
        return out

    return allowed


def summarize(suite: Suite, rep, labels: dict[str, np.ndarray], evaluate: np.ndarray, timing: bool = False,
              per_build: bool = False, latency: bool = False, anatomy: str | None = None) -> dict:
    """Test work, feedback time and selector time and, per label, faulty builds, escapes, escapes masked by another
    selected failure and selected failing rows over the evaluated builds; `timing` adds the time to first failure.
    `per_build` adds, under "_builds", each evaluated build's test time, change signal and flags: for the i-th label,
    bits 3i, 3i+1 and 3i+2 mark faulty, caught by the selection and masked; bit FULL_RUN_BIT marks a full run.
    `latency` adds failing rows that did not run and the mean builds and hours until their class runs again.
    `anatomy`, the replay's regime, adds escapes by cause (escape_causes) and the share of selected test time spent on
    failures outside DeConf."""
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
    executed = rep.executed(suite)
    for i, (name, label) in enumerate(labels.items()):
        faulty = (np.bincount(of_row, weights=label, minlength=n) > 0) & evaluate
        caught = np.bincount(of_row, weights=label & rep.selected, minlength=n) > 0
        found = np.bincount(of_row, weights=label & executed, minlength=n) > 0
        other = np.bincount(of_row, weights=suite.failed & ~label & rep.selected, minlength=n) > 0
        rows = label & evaluate[of_row]
        masked = faulty & ~caught & other
        flags |= (faulty.astype(np.uint16) << 3 * i) | (caught.astype(np.uint16) << 3 * i + 1) \
            | (masked.astype(np.uint16) << 3 * i + 2)
        out.update({
            f"{name}_faulty": int(faulty.sum()),
            f"{name}_escape": int((faulty & ~caught).sum()),
            f"{name}_executed_escape": int((faulty & ~found).sum()),
            f"{name}_masked": int(masked.sum()),
            f"{name}_flagged_escape": int((faulty & ~caught & flagged).sum()),
            f"{name}_audited": int((faulty & coin).sum()),
            f"{name}_audited_escape": int((faulty & coin & ~caught).sum()),
            f"{name}_rows": int(rows.sum()),
            f"{name}_rows_selected": int((rows & rep.selected).sum()),
        })
        if latency:
            out[f"{name}_hidden"], out[f"{name}_hidden_builds"], out[f"{name}_hidden_hours"] = \
                hidden_latency(suite, rep, label, evaluate)
        if anatomy:
            seen = executed if anatomy == "faithful" else np.ones(len(suite.cls), dtype=bool)
            out.update({f"{name}_escape_{k}": v for k, v in escape_causes(suite, rep, label, evaluate, seen).items()})
        if timing:
            ttff = first_failure_time(suite, rep, label, faulty)
            found = ~np.isnan(ttff)
            out[f"{name}_ttff_hours"] = ttff[found].mean() / 3600 if found.any() else np.nan
            out[f"{name}_ttff_share"] = (ttff[found] / rep.full_cost[found]).mean() if found.any() else np.nan
    if anatomy and "deconf" in labels:
        chosen = rep.selected & evaluate[of_row]
        out["waste_share"] = suite.duration[chosen & suite.failed & ~labels["deconf"]].sum() / \
            max(suite.duration[chosen].sum(), 1e-9)
    if per_build:
        idx = np.flatnonzero(evaluate)
        out["_builds"] = {"labels": list(labels), "build": idx.astype(np.int32),
                          "executed": rep.executed_cost[idx].astype(np.float32),
                          "full": rep.full_cost[idx].astype(np.float32), "flags": flags[idx],
                          "signal": rep.signal[idx].astype(np.float32)}
    return out


def price_task(project: str, scorer: str, regime: str, pi: float, seed: int, lambdas: list[float],
               evaluate: tuple[float, float], retire_after: int | None, per_build: bool = False, schedule: int = 0,
               rotate: float = 0.0, latency: bool = False, threshold: bool = False,
               only_tests: bool = False, anatomy: bool = False) -> list[dict]:
    """Replays over the lambda grid; `threshold` cuts at a share of the change's relevance mass instead of predicted
    cost, and `only_tests` runs just the changed and unseen tests when a change touches no production code."""
    suite, labels, scorers, mask = setup(project, seed, evaluate)
    later = evaluate[1] > 1
    extra = {"mass": change_mass(project, later) if threshold else None,
             "only": path_filter(project, later) if only_tests else None}
    return [
        {"project": project, "scorer": scorer, "regime": regime, "pi": pi, "schedule": schedule, "rotate": rotate,
         "cut": "relevance" if threshold else "cost", "path_filter": only_tests, "seed": seed, "lam": lam,
         **summarize(suite, replay(suite, regime, scorers[scorer], lam=lam, pi=pi, seed=seed, retire_after=retire_after,
                                   schedule=schedule, rotate=rotate, **extra), labels, mask, per_build=per_build,
                     latency=latency, anatomy=regime if anatomy else None)}
        for lam in lambdas
    ]


def make_controller(kind: str, alpha: float, cfg: dict, project: str = "", scorer: str = ""):
    """The cut, its feedback mode, its monitor and its size cap for one controller kind; "tuned" is the fixed budget
    tuned on the dev split for the project, order and alpha. The online cut's monitor is a CUSUM when the config's
    `monitor` says "cusum"."""
    if kind == "online":
        monitor = Cusum(alpha, cfg.get("cusum_ratio", 3.0), cfg.get("cusum_h", 5.0)) \
            if cfg.get("monitor") == "cusum" else Monitor(alpha)
        return OnlineCut(alpha, cfg["eta"]), "audits", monitor, cfg["cap"]
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
    """One controller replay; a kind ending in "+adaptive" audits by the change signal instead of uniformly. The
    controller's target is the operational label unless the config's `target` names another."""
    suite, labels, scorers, mask = setup(project, seed, evaluate)
    labels = {**labels, "newfail": new_failures(suite, labels["operational"], seed, pi, cfg.get("warmup", 0.2))}
    target = cfg.get("target", "operational")
    base, _, audits = kind.partition("+")
    cut, feedback, monitor, cap = make_controller(base, alpha, cfg, project, scorer)
    rep = replay(suite, "faithful", scorers[scorer], lam=cut, pi=pi, seed=seed, label=labels[target],
                 feedback=feedback, cap=cap, monitor=monitor, retire_after=cfg.get("retire_after"),
                 warmup=cfg.get("warmup", 0.2),
                 signal=change_signal(project, evaluate[1] > 1), adaptive=audits == "adaptive",
                 schedule=cfg.get("schedule", 0), rotate=cfg.get("rotate", 0.0),
                 mass=change_mass(project, evaluate[1] > 1) if cfg.get("threshold") else None,
                 only=path_filter(project, evaluate[1] > 1) if cfg.get("path_filter") else None)
    alarms = np.array([t for t in rep.alarms if t < len(mask)], dtype=int)
    row = {"project": project, "scorer": scorer, "controller": kind, "alpha": alpha, "pi": pi, "seed": seed,
           "target": target, "schedule": cfg.get("schedule", 0), "rotate": cfg.get("rotate", 0.0),
           "cut": "relevance" if cfg.get("threshold") else "cost", "path_filter": bool(cfg.get("path_filter")),
           **summarize(suite, rep, labels, mask, timing=True, per_build=cfg.get("per_build", False),
                       latency=cfg.get("latency", False)),
           "alarms": int(mask[alarms].sum()), "alarm_builds": int((mask & (rep.reason == 3)).sum())}
    of_row, n = suite.build_of_rows(), len(suite.builds)
    faulty = np.bincount(of_row, weights=labels[target], minlength=n) > 0
    if isinstance(cut, OnlineCut):
        updates = rep.audited & faulty & (rep.reason != 1)
        start, finish = suite.builds["start"].to_numpy(), suite.builds["finish"].to_numpy()
        row.update({"updates": len(cut.losses), "pending": max_pending(start[updates], finish[updates]),
                    "lam_min": min(cut.path), "lam_max": max(cut.path)})
    if row[f"{target}_faulty"]:
        caught = row[f"{target}_faulty"] - row[f"{target}_escape"]
        widths = transfer_widths(caught, row[f"{target}_audited_escape"], pi, row["builds"] - caught)
        row.update({"escape_ipw": widths.pop("estimate"), **{f"width_{k}": v for k, v in widths.items()}})
        idx = np.flatnonzero(mask)
        found = np.bincount(of_row, weights=labels[target] & rep.selected, minlength=n) > 0
        coin = rep.audited & (rep.reason != 1)
        ht = ht_widths(faulty[idx], found[idx], coin[idx], rep.pi[idx])
        row.update({"escape_ht": ht["estimate"], "width_ht_normal": ht["normal"], "width_ht_worst": ht["worst"]})
    return [row]
