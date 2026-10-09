import json
import sys
import time
import urllib.request

import pandas as pd

from faithfulcut import lrts_ref
from faithfulcut.labels import flaky_issues_path

HEADERS = {"Accept": "application/json", "User-Agent": "faithfulcut"}


def get(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
        return json.load(response)


def created(url: str) -> float:
    url = url.split("?")[0].rstrip("/")
    if "issues.apache.org/jira/browse/" in url:
        key = url.rsplit("/", 1)[-1]
        stamp = get(f"https://issues.apache.org/jira/rest/api/2/issue/{key}?fields=created")["fields"]["created"]
    else:
        owner, repo, _, number = url.split("/")[-4:]
        stamp = get(f"https://api.github.com/repos/{owner}/{repo}/issues/{number}")["created_at"]
    return pd.Timestamp(stamp).timestamp()


def main():
    path = flaky_issues_path()
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["url", "created"])
    todo = [u for u in lrts_ref.flaky_tests()["url"].drop_duplicates() if u not in set(done["url"])]
    fetched, failed = [], []
    for url in todo:
        try:
            fetched.append((url, created(url)))
        except Exception as error:
            failed.append((url, error))
        time.sleep(0.5)
    new = pd.DataFrame(fetched, columns=["url", "created"])
    out = pd.concat([done, new], ignore_index=True) if len(done) else new
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{len(out)} issues with creation times, {len(failed)} failed")
    for url, error in failed:
        print(url, error)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
