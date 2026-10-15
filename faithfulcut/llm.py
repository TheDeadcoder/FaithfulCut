import heapq
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from faithfulcut import lrts
from faithfulcut.suite import Suite

API_KEY_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ADC_ENDPOINT = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
                "/publishers/google/models/{model}:generateContent")
_token = {"value": None, "expires": 0.0}
_token_lock = threading.Lock()
RISK = {"high": 0, "medium": 1, "low": 2}
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


def prompt(suite: Suite, t: int, catalog: np.ndarray, max_diff_chars: int = 24000, limit: int = 50) -> str:
    b = suite.builds.iloc[t]
    diff = lrts.diff_path(suite.project, b["pr_name"], b["build_id"]).read_text(errors="ignore")
    return PROMPT.format(limit=limit, diff=diff[:max_diff_chars], catalog="\n".join(suite.names[catalog]))


def _adc_token() -> str:
    """A short-lived access token from Application Default Credentials, refreshed well before it expires."""
    with _token_lock:
        if _token["value"] is None or time.time() > _token["expires"]:
            _token["value"] = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                                             capture_output=True, text=True, check=True).stdout.strip()
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


def ask(text: str, model: str, retries: int = 5) -> dict:
    """One synchronous generateContent call; returns the response JSON with the measured latency."""
    body = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }).encode()
    for attempt in range(retries):
        clock = time.perf_counter()
        try:
            with urllib.request.urlopen(_request(model, body), timeout=300) as response:
                return {**json.load(response), "latency": time.perf_counter() - clock}
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                detail = error.read().decode(errors="ignore")[:500]
                raise RuntimeError(f"{model}: HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt * 5)


def cached_answer(suite: Suite, t: int, catalog: np.ndarray, model: str) -> dict:
    b = suite.builds.iloc[t]
    path = (Path(os.environ["FAITHFULCUT_CACHE"]) / "llm-v1" / model / suite.project
            / f"{lrts.build_name(b['pr_name'], b['build_id'])}.json")
    if path.exists():
        return json.loads(path.read_text())
    answer = ask(prompt(suite, t, catalog), model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(answer))
    return answer


def risk_levels(answer: dict, catalog_names: np.ndarray) -> tuple[np.ndarray, bool]:
    """Risk rank per catalog class (3 when unnamed) and whether the answer was well formed."""
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
        return ranks, False
