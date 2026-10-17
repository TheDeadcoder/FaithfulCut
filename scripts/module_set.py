import argparse
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from faithfulcut.labels import load_operational, new_failures, transfer
from faithfulcut.structure import NO_MODULES, UNKNOWN, load_structure
from faithfulcut.suite import load_suite

PROJECTS = ["activemq", "hadoop", "hbase", "hive", "jackrabbit-oak", "james", "kafka", "karaf", "log4j"]


def selection(suite, structure: dict[str, np.ndarray]) -> np.ndarray:
    """Rows the module set runs: classes in the changed modules and their dependents, and classes not seen in any
    build finished before the start; every row of a build without module information."""
    start, finish = suite.builds["start"].to_numpy(), suite.builds["finish"].to_numpy()
    seen = np.zeros(len(suite.names), dtype=bool)
    out = np.zeros(len(suite.cls), dtype=bool)
    pending = []
    for t in range(len(suite.builds)):
        while pending and pending[0][0] <= start[t]:
            seen[suite.cls[suite.rows(heapq.heappop(pending)[1])]] = True
        rows = suite.rows(t)
        cls = suite.cls[rows]
        if (structure["path"][t, cls] < UNKNOWN).any():
            out[rows] = (structure["module"][t, cls] < UNKNOWN) | ~seen[cls]
        else:
            out[rows] = True
        heapq.heappush(pending, (finish[t], t))
    return out


def main():
    parser = argparse.ArgumentParser(description="The module set as a fixed selector: test work and escapes per label")
    parser.add_argument("--evaluate", nargs=2, type=float, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--pi", type=float, default=0.1, help="audit rate defining new failures")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for project in PROJECTS:
        if project in NO_MODULES:
            continue
        full = load_suite(project, later=args.evaluate[1] > 1)
        suite = full.before(args.evaluate[1])
        m = suite.offsets[-1]
        position = suite.builds["position"].to_numpy()
        evaluated = (position >= args.evaluate[0]) & (position < args.evaluate[1])
        of_row = suite.build_of_rows()
        chosen = selection(suite, load_structure(full))[:m]
        weight = suite.duration * evaluated[of_row]
        row = {"project": project, "work": weight[chosen].sum() / weight.sum(),
               "classes": (chosen & evaluated[of_row]).sum() / evaluated[of_row].sum()}
        labels = {k: [v[:m]] for k, v in transfer(full).items()}
        labels["operational"], labels["newfail"] = [], []
        for seed in range(args.seeds):
            operational = load_operational(full, seed)[:m]
            labels["operational"].append(operational)
            labels["newfail"].append(new_failures(suite, operational, seed, args.pi))
        for name, versions in labels.items():
            rates = []
            for label in versions:
                faulty = (np.bincount(of_row, weights=label, minlength=len(position)) > 0) & evaluated
                caught = np.bincount(of_row, weights=label & chosen, minlength=len(position)) > 0
                rates.append((faulty & ~caught).sum() / max(faulty.sum(), 1))
            row[name] = float(np.mean(rates))
        rows.append(row)
        print(project, flush=True)
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.mean(numeric_only=True).to_frame().T.assign(project="macro")], ignore_index=True)
    body = df.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    text = f"## Module set as a fixed selector, builds at positions [{args.evaluate[0]}, {args.evaluate[1]})\n\n" \
           f"Work is the share of test time run; escapes are per label, means over seeds.\n\n```\n{body}\n```\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
