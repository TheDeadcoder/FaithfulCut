from functools import cache, lru_cache

import numpy as np

from faithfulcut import dante
from faithfulcut.control import FixedCut, MetaCut, Monitor, OnlineCut, SplitCut, max_pending, transfer_widths
from faithfulcut.ir import load_ir
from faithfulcut.labels import load_operational, transfer
from faithfulcut.replay import replay
from faithfulcut.scorers import history_scorers, hybrid_scorers, ir_scorers
from faithfulcut.suite import Suite, load_suite


@lru_cache(maxsize=2)
def setup(project: str, seed: int, evaluate: tuple[float, float]):
    """The suite up to the end of the evaluated span, its labels, the scorers and the evaluated builds."""
    full = load_suite(project)
    suite = full.before(evaluate[1])
    rows = suite.offsets[-1]
    labels = {"operational": load_operational(full, seed)[:rows]}
    labels.update({k: v[:rows] for k, v in transfer(full).items()})
    scorers = {**history_scorers(suite), **ir_scorers(suite, load_ir(full, "seen"))}
    scorers.update(hybrid_scorers(scorers["LatestFail+CC"], scorers["TF-IDF+CC"]))
    if project in dante.PROJECTS:
        relevance = cache(lambda: dante.load_relevance(full))
        scorers.update(dante.dante_scorers(suite, relevance, dante.stage_counts(full)))
    position = suite.builds["position"].to_numpy()
    return suite, labels, scorers, (position >= evaluate[0]) & (position < evaluate[1])


def available(project: str, scorer: str) -> bool:
    return not scorer.startswith("DANTE") or project in dante.PROJECTS


def summarize(suite: Suite, rep, labels: dict[str, np.ndarray], evaluate: np.ndarray) -> dict:
    """Test work and, per label, faulty builds, escapes and selected failing rows over the evaluated builds."""
    of_row = suite.build_of_rows()
    n = len(suite.builds)
    full_run = rep.audited | rep.forced
    coin = rep.audited & (rep.reason != 1)
    out = {
        "builds": int(evaluate.sum()),
        "work": rep.executed_cost[evaluate].sum() / rep.full_cost[evaluate].sum(),
        "base_work": rep.base_cost[evaluate].sum() / rep.full_cost[evaluate].sum(),
        "forced": int((rep.forced & evaluate).sum()),
        "lam_median": float(np.median(rep.lam[evaluate])),
    }
    for name, label in labels.items():
        faulty = (np.bincount(of_row, weights=label, minlength=n) > 0) & evaluate
        caught = np.bincount(of_row, weights=label & rep.selected, minlength=n) > 0
        rows = label & evaluate[of_row]
        out.update({
            f"{name}_faulty": int(faulty.sum()),
            f"{name}_escape": int((faulty & ~caught).sum()),
            f"{name}_executed_escape": int((faulty & ~caught & ~full_run).sum()),
            f"{name}_audited": int((faulty & coin).sum()),
            f"{name}_audited_escape": int((faulty & coin & ~caught).sum()),
            f"{name}_rows": int(rows.sum()),
            f"{name}_rows_selected": int((rows & rep.selected).sum()),
        })
    return out


def price_task(project: str, scorer: str, regime: str, pi: float, seed: int, lambdas: list[float],
               evaluate: tuple[float, float], retire_after: int | None) -> list[dict]:
    suite, labels, scorers, mask = setup(project, seed, evaluate)
    return [
        {"project": project, "scorer": scorer, "regime": regime, "pi": pi, "seed": seed, "lam": lam,
         **summarize(suite, replay(suite, regime, scorers[scorer], lam=lam, pi=pi, seed=seed,
                                   retire_after=retire_after), labels, mask)}
        for lam in lambdas
    ]


def make_controller(kind: str, alpha: float, cfg: dict):
    """The cut, its feedback mode, its monitor and its size cap for one controller kind."""
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
    if kind.startswith("fixed-"):
        return FixedCut(float(kind.removeprefix("fixed-"))), "audits", None, cfg["cap"]
    raise ValueError(f"unknown controller {kind}")


def control_task(project: str, scorer: str, kind: str, alpha: float, pi: float, seed: int, cfg: dict,
                 evaluate: tuple[float, float]) -> list[dict]:
    suite, labels, scorers, mask = setup(project, seed, evaluate)
    cut, feedback, monitor, cap = make_controller(kind, alpha, cfg)
    rep = replay(suite, "faithful", scorers[scorer], lam=cut, pi=pi, seed=seed, label=labels["operational"],
                 feedback=feedback, cap=cap, monitor=monitor, retire_after=cfg.get("retire_after"))
    row = {"project": project, "scorer": scorer, "controller": kind, "alpha": alpha, "pi": pi, "seed": seed,
           **summarize(suite, rep, labels, mask)}
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
    return [row]
