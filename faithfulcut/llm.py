import heapq
import http.client
import json
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from faithfulcut import lrts
from faithfulcut.ir import load_ir
from faithfulcut.labels import transfer
from faithfulcut.suite import Suite, load_suite

API_KEY_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ADC_ENDPOINT = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
                "/publishers/google/models/{model}:generateContent")
_token = {"value": None, "expires": 0.0}
_token_lock = threading.Lock()
RISK = {"high": 0, "medium": 1, "low": 2}
TRANSIENT = (401, 429, 500, 502, 503, 504)
EPS = 1e-6
FUSIONS = ("risk", "cost", "rrf")
PROMPT = """A pull request changed the code shown in the diff below. The project's test classes are listed after it.
Name the test classes most likely to FAIL because of this change, highest risk first.
Answer only with JSON of the form {{"tests": [{{"name": "<exact name from the list>", "risk": "high" | "medium" | "low"}}]}}.
Use only names from the list, and name at most {limit} classes.

DIFF (possibly truncated):
{diff}

TEST CLASSES:
{catalog}
"""


def timed_state(suite: Suite, builds: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Known classes, last observed durations and runs since the last failure at each requested build's start,
    from every build finished by then."""
    start, finish = suite.builds["start"].to_numpy(), suite.builds["finish"].to_numpy()
    wanted = set(builds)
    known = np.zeros(len(suite.names), dtype=bool)
    duration = np.zeros(len(suite.names))
    since_fail = np.zeros(len(suite.names))
    pending, out = [], {}
    for t in range(len(suite.builds)):
        while pending and pending[0][0] <= start[t]:
            rows = suite.rows(heapq.heappop(pending)[1])
            cls = suite.cls[rows]
            known[cls] = True
            duration[cls] = suite.duration[rows]
            since_fail[cls] = np.where(suite.failed[rows], 1, since_fail[cls] + 1)
        if t in wanted:
            catalog = np.flatnonzero(known)
            out[t] = (catalog, duration[catalog].copy(), since_fail[catalog].copy())
        heapq.heappush(pending, (finish[t], t))
    return out


def prompt(suite: Suite, t: int, catalog: np.ndarray, max_diff_chars: int = 24000, limit: int = 50,
           diff_of: int | None = None) -> str:
    """The prompt for build t: its diff, or build `diff_of`'s, and t's catalog."""
    b = suite.builds.iloc[t if diff_of is None else diff_of]
    diff = lrts.diff_path(suite.project, b["pr_name"], b["build_id"]).read_text(errors="ignore")
    return PROMPT.format(limit=limit, diff=diff[:max_diff_chars], catalog="\n".join(suite.names[catalog]))


def _adc_token() -> str:
    """A short-lived access token from Application Default Credentials, refreshed well before it expires."""
    with _token_lock:
        if _token["value"] is None or time.time() > _token["expires"]:
            _token["value"] = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                                             capture_output=True, text=True, check=True, timeout=120).stdout.strip()
            _token["expires"] = time.time() + 45 * 60
        return _token["value"]


def _request(model: str, body: bytes) -> urllib.request.Request:
    """API key when GEMINI_API_KEY is set; otherwise Application Default Credentials on GOOGLE_CLOUD_PROJECT."""
    headers = {"Content-Type": "application/json"}
    if os.environ.get("GEMINI_API_KEY"):
        headers["x-goog-api-key"] = os.environ["GEMINI_API_KEY"]
        return urllib.request.Request(API_KEY_ENDPOINT.format(model=model), data=body, headers=headers)
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    headers.update({"Authorization": f"Bearer {_adc_token()}", "X-Goog-User-Project": project})
    url = ADC_ENDPOINT.format(project=project, location=location, model=model)
    return urllib.request.Request(url, data=body, headers=headers)


def ask(text: str, model: str, patience: float = 1800.0) -> dict:
    """One synchronous generateContent call; returns the response JSON with the measured latency. A dropped
    connection, a timeout, a failed token refresh, a rejected token, a rate limit or a server error is retried after
    about 10 s, then twice as long each time up to 5 min, until `patience` seconds have been spent waiting. Any other
    HTTP error raises at once."""
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }).encode()
    waited, delay = 0.0, 10.0
    while True:
        clock = time.perf_counter()
        try:
            with urllib.request.urlopen(_request(model, body), timeout=120) as response:
                return {**json.load(response), "latency": time.perf_counter() - clock}
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT:
                detail = error.read().decode(errors="ignore")[:500]
                raise RuntimeError(f"{model}: HTTP {error.code}: {detail}") from error
            if error.code == 401:
                _token["value"] = None
            reason = f"HTTP {error.code}"
        except (OSError, http.client.HTTPException, subprocess.SubprocessError, ValueError) as error:
            reason = type(error).__name__
        if waited >= patience:
            raise RuntimeError(f"{model}: gave up after waiting {waited:.0f} s ({reason})")
        pause = min(delay, 300.0) * random.uniform(0.8, 1.2)
        print(f"{model}: {reason}; retrying in {pause:.0f} s", flush=True)
        time.sleep(pause)
        waited, delay = waited + pause, delay * 2


def answer_path(suite: Suite, t: int, model: str) -> Path:
    b = suite.builds.iloc[t]
    return (Path(os.environ["FAITHFULCUT_CACHE"]) / "llm-v1" / model / suite.project
            / f"{lrts.build_name(b['pr_name'], b['build_id'])}.json")


def cached_answer(suite: Suite, t: int, catalog: np.ndarray, model: str, offline: bool = False,
                  diff_of: int | None = None, variant: str = "") -> dict | None:
    """The model's saved answer for the build, else a new one, saved as soon as it arrives; offline, None when nothing
    is saved. Answers to prompts showing another build's diff are kept apart under `variant`."""
    path = answer_path(suite, t, model + variant)
    if path.exists():
        return json.loads(path.read_text())
    if offline:
        return None
    answer = ask(prompt(suite, t, catalog, diff_of=diff_of), model)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(answer))
    partial.replace(path)
    return answer


def risk_levels(answer: dict, catalog_names: np.ndarray) -> tuple[np.ndarray, bool]:
    """Risk rank per catalog class (3 when unnamed) and whether the answer was well formed. A malformed answer
    names nothing, so the risk order falls back to its tie-breaker."""
    ranks = np.full(len(catalog_names), 3)
    position = {n: i for i, n in enumerate(catalog_names)}
    try:
        text = answer["candidates"][0]["content"]["parts"][0]["text"]
        for item in json.loads(text)["tests"]:
            i = position.get(item["name"])
            if i is not None:
                ranks[i] = min(ranks[i], RISK.get(str(item.get("risk", "low")).lower(), 2))
        return ranks, True
    except (KeyError, IndexError, TypeError, ValueError):
        return np.full(len(catalog_names), 3), False


def caught(keys: list[np.ndarray], duration: np.ndarray, failing: np.ndarray, rng: np.random.Generator,
           budget: float) -> float:
    """Whether a failing class enters the prefix within `budget` of predicted time under the lexicographic keys."""
    order = np.lexsort([rng.random(len(duration))] + keys[::-1])
    cumulative = np.cumsum(duration[order])
    depth = np.empty(len(order))
    depth[order] = cumulative / cumulative[-1] if cumulative[-1] > 0 else 0.0
    return float((depth[failing] <= budget).any())


def llm_keys(ranks: np.ndarray, tfidf_cc: np.ndarray, duration: np.ndarray, fusion: str) -> list[np.ndarray]:
    """Order keys for the LLM's risk levels: risk first then TF-IDF+CC ("risk"); risk value per predicted second
    ("cost"); or reciprocal rank fusion of the "risk" order with TF-IDF+CC ("rrf")."""
    if fusion == "risk":
        return [ranks, tfidf_cc]
    if fusion == "cost":
        return [-(3 - ranks) / (duration + EPS), tfidf_cc]
    fused = np.zeros(len(ranks))
    for order in (np.lexsort((tfidf_cc, ranks)), np.argsort(tfidf_cc, kind="stable")):
        position = np.empty(len(ranks))
        position[order] = np.arange(1, len(ranks) + 1)
        fused += 1 / (60 + position)
    return [-fused]


def evaluate(project: str, builds: list[int], model: str, budget: float, fusion: str = "risk", workers: int = 4,
             dry_run: bool = False, offline: bool = False, diffs: dict[int, int] | None = None,
             variant: str = "") -> list[dict]:
    """Per build whose first-failing classes are all in its catalog, whether QTF, LatestFail+CC, TF-IDF+CC and the
    LLM's order with `fusion` put one of them within `budget` of predicted time. `diffs` shows a build another
    build's diff, with the answers kept under `variant`."""
    suite = load_suite(project)
    first = transfer(suite)["firstfail"]
    ir = load_ir(suite, "seen")
    state = timed_state(suite, builds)
    diffs = diffs or {}
    rows = []

    def one(t):
        catalog = state[t][0]
        if dry_run:
            return t, None, len(prompt(suite, t, catalog, diff_of=diffs.get(t)))
        return t, cached_answer(suite, t, catalog, model, offline, diffs.get(t), variant), None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(one, builds))
    for t, answer, prompt_chars in answers:
        catalog, duration, since_fail = state[t]
        rows_t = suite.rows(t)
        position = {c: i for i, c in enumerate(catalog)}
        failing_cls = suite.cls[rows_t][first[rows_t]]
        if not all(c in position for c in failing_cls) or (offline and answer is None):
            continue
        failing = np.array([position[c] for c in failing_cls])
        scores = np.zeros(len(suite.names))
        part = slice(ir["offsets"][t], ir["offsets"][t + 1])
        scores[ir["cls"][part]] = ir["score"][part]
        tfidf_cc = 1 / (scores[catalog] + EPS) * (duration + EPS)
        rng = lambda: np.random.default_rng([t, len(catalog)])
        row = {"project": project, "build": t, "classes": len(catalog),
               "QTF": caught([duration], duration, failing, rng(), budget),
               "LatestFail+CC": caught([(since_fail + EPS) * (duration + EPS)], duration, failing, rng(), budget),
               "TF-IDF+CC": caught([tfidf_cc], duration, failing, rng(), budget)}
        if answer is None:
            row["prompt_chars"] = prompt_chars
        else:
            ranks, ok = risk_levels(answer, suite.names[catalog])
            usage = answer.get("usageMetadata", {})
            row.update({"LLM": caught(llm_keys(ranks, tfidf_cc, duration, fusion), duration, failing, rng(), budget),
                        "well_formed": ok,
                        "latency": answer["latency"], "input_tokens": usage.get("promptTokenCount"),
                        "output_tokens": usage.get("candidatesTokenCount"),
                        "thinking_tokens": usage.get("thoughtsTokenCount", 0)})
        rows.append(row)
    return rows
