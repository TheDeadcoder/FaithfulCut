import dataclasses

import numpy as np

from faithfulcut.experiments import make_controller, setup
from faithfulcut.ir import load_ir
from faithfulcut.replay import replay
from faithfulcut.scorers import history_scorers, hybrid_scorers, ir_scorers
from faithfulcut.suite import Suite

QUARTERS = 4


def renamed(suite: Suite, ir: dict, t0: int, share: float, rng: np.random.Generator) -> tuple[Suite, dict]:
    """A share of the classes running from build t0 on take new identities there: their history resets, while their
    change relevance, read from the same sources, stays."""
    after = slice(suite.offsets[t0], None)
    present = np.unique(suite.cls[after])
    chosen = rng.choice(present, int(round(share * len(present))), replace=False)
    new = np.full(len(suite.names), -1)
    new[chosen] = len(suite.names) + np.arange(len(chosen))
    cls = suite.cls.copy()
    moved = new[cls[after]] >= 0
    cls[after] = np.where(moved, new[cls[after]], cls[after])
    ir_cls = ir["cls"].copy()
    tail = slice(ir["offsets"][t0], None)
    ir_cls[tail] = np.where(new[ir_cls[tail]] >= 0, new[ir_cls[tail]], ir_cls[tail])
    names = np.concatenate([suite.names, np.char.add(suite.names[chosen].astype(str), "~renamed")])
    return dataclasses.replace(suite, names=names, cls=cls), {**ir, "cls": ir_cls}


def burst(suite: Suite, label: np.ndarray, t0: int, n_classes: int, builds: int, rng: np.random.Generator,
          skipped: np.ndarray | None = None) -> tuple[Suite, np.ndarray]:
    """Classes that never failed before build t0, run in at least half of the next `builds` builds and, if given, are
    `skipped`, fail in every one of those builds they run in."""
    t1 = min(t0 + builds, len(suite.builds))
    before, window = slice(0, suite.offsets[t0]), slice(suite.offsets[t0], suite.offsets[t1])
    failed_before = np.zeros(len(suite.names), dtype=bool)
    failed_before[suite.cls[before][suite.failed[before]]] = True
    runs = np.bincount(suite.cls[window], minlength=len(suite.names))
    candidates = (runs >= 0.5 * (t1 - t0)) & ~failed_before
    candidates = np.flatnonzero(candidates if skipped is None else candidates & skipped)
    chosen = rng.choice(candidates, min(n_classes, len(candidates)), replace=False)
    rows = np.zeros(len(suite.cls), dtype=bool)
    rows[window] = np.isin(suite.cls[window], chosen)
    return dataclasses.replace(suite, failed=suite.failed | rows, failed_everywhere=suite.failed_everywhere | rows), \
        label | rows


def slowed(suite: Suite, t0: int, share: float, factor: float, rng: np.random.Generator) -> Suite:
    """A share of the classes running from build t0 on take `factor` times as long from there."""
    after = slice(suite.offsets[t0], None)
    present = np.unique(suite.cls[after])
    chosen = rng.choice(present, int(round(share * len(present))), replace=False)
    duration = suite.duration.copy()
    duration[after] = np.where(np.isin(suite.cls[after], chosen), duration[after] * factor, duration[after])
    return dataclasses.replace(suite, duration=duration)


def drift_task(project: str, scorer: str, kind: str, pi: float, seed: int, cfg: dict,
               evaluate: tuple[float, float]) -> list[dict]:
    """The online controller and its monitor on a replay with drift of `kind` injected at position cfg["start"]
    ("clean" for none; "rename:0.5" overrides the kind's default severity), with the seed fixing which classes drift.
    A burst hits classes the clean replay of the same order skipped at the injection. Reports escapes and work per
    quarter of the horizon after the injection, and alarms before and after it."""
    suite, labels, _, mask = setup(project, seed, evaluate)
    label, ir = labels["operational"], load_ir(suite, "seen")
    position = suite.builds["position"].to_numpy()
    t0 = int(np.searchsorted(position, cfg["start"]))
    end = min(t0 + cfg["horizon"], int(np.flatnonzero(mask).max()) + 1)
    def run(suite: Suite, label: np.ndarray, ir: dict):
        scorers = {**history_scorers(suite), **ir_scorers(suite, ir)}
        scorers.update(hybrid_scorers(scorers["LatestFail+CC"], scorers["TF-IDF+CC"]))
        cut, feedback, monitor, cap = make_controller("online", cfg["alpha"], cfg, project, scorer)
        return replay(suite, "faithful", scorers[scorer], lam=cut, pi=pi, seed=seed, label=label, feedback=feedback,
                      cap=cap, monitor=monitor, warmup=cfg.get("warmup", 0.2))

    rng = np.random.default_rng(seed)
    name, _, severity = kind.partition(":")
    if name == "rename":
        suite, ir = renamed(suite, ir, t0, float(severity or cfg["rename_share"]), rng)
    elif name == "burst":
        rows = suite.rows(t0)
        skipped = np.zeros(len(suite.names), dtype=bool)
        skipped[suite.cls[rows][~run(suite, label, ir).selected[rows]]] = True
        suite, label = burst(suite, label, t0, int(severity or cfg["burst_classes"]), cfg["burst_builds"], rng,
                             skipped)
    elif name == "slow":
        suite = slowed(suite, t0, cfg["slow_share"], float(severity or cfg["slow_factor"]), rng)
    rep = run(suite, label, ir)
    of_row, n = suite.build_of_rows(), len(suite.builds)
    faulty = np.bincount(of_row, weights=label, minlength=n) > 0
    escaped = faulty & ~(np.bincount(of_row, weights=label & rep.selected, minlength=n) > 0)
    first = int(np.flatnonzero(rep.reason != 1).min())
    alarms = np.array(rep.alarms, dtype=int)
    after = alarms[(alarms >= t0) & (alarms < end)]
    row = {"project": project, "scorer": scorer, "kind": kind, "pi": pi, "seed": seed, "start": t0,
           "horizon": end - t0, "builds_before": t0 - first,
           "alarms_before": int(((alarms >= first) & (alarms < t0)).sum()), "alarms_after": len(after),
           "first_alarm": int(after.min() - t0) if len(after) else np.nan, "lam_start": float(rep.lam[t0]),
           "lam_max": float(rep.lam[t0:end].max())}
    for q, part in enumerate(np.array_split(np.arange(t0, end), QUARTERS)):
        row.update({f"faulty_{q}": int(faulty[part].sum()), f"escape_{q}": int(escaped[part].sum()),
                    f"work_{q}": rep.executed_cost[part].sum() / rep.full_cost[part].sum()})
    return [row]
