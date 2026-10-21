import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from faithfulcut.llm import answer_path, cached_answer, prompt, timed_state
from faithfulcut.suite import load_suite


def main():
    parser = argparse.ArgumentParser(description="Ask the model about every build of a project within a span of "
                                                 "positions, saving each answer as it arrives; builds already "
                                                 "answered are skipped, and a build whose request fails is reported")
    parser.add_argument("--project", required=True)
    parser.add_argument("--span", nargs=2, type=float, required=True)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="count builds and prompt sizes; never call the model")
    args = parser.parse_args()
    suite = load_suite(args.project, later=args.span[1] > 1)
    position = suite.builds["position"].to_numpy()
    builds = np.flatnonzero((position >= args.span[0]) & (position < args.span[1])).tolist()
    todo = [t for t in builds if not answer_path(suite, t, args.model).exists()]
    state = timed_state(suite, todo)
    print(f"{len(builds)} builds, {len(builds) - len(todo)} answered, {len(todo)} to ask", flush=True)
    if args.dry_run:
        chars = [len(prompt(suite, t, state[t][0])) for t in todo]
        print(f"prompt characters: total {sum(chars)}, median {int(np.median(chars)) if chars else 0}")
        return
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(cached_answer, suite, t, state[t][0], args.model): t for t in todo}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except (RuntimeError, OSError) as error:
                failed.append(futures[future])
                print(f"build {futures[future]}: {error}", flush=True)
            if done % 50 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} asked, {len(failed)} failed", flush=True)
    if failed:
        print("failed builds:", " ".join(map(str, sorted(failed))))


if __name__ == "__main__":
    main()
