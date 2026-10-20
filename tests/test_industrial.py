import pandas as pd

from faithfulcut.industrial import load_log, replay_log


def write_log(path, rows):
    pd.DataFrame([{"Id": i, "Name": name, "Duration": 1, "LastRun": "2016-01-01 00:00", "LastResults": "[]",
                   "Verdict": verdict, "Cycle": cycle} for i, (cycle, name, verdict) in enumerate(rows, 1)]
                 ).to_csv(path, sep=";", index=False)
    return load_log(path)


def test_faithful_history_misses_what_the_order_did_not_run(tmp_path):
    log = write_log(tmp_path / "log.csv", [(1, "a", 1), (1, "b", 0), (2, "a", 0), (2, "b", 1), (3, "a", 0),
                                          (3, "b", 1)])
    full = replay_log(log, "LatestFail", "full", seed=0, budget=0.5)
    faithful = replay_log(log, "LatestFail", "faithful", seed=0, budget=0.5)
    assert full["missed"].tolist() == [1, 0] and faithful["missed"].tolist() == [1, 1]
    assert full["failing"].tolist() == faithful["failing"].tolist() == [1, 1]


def test_a_test_runs_at_its_first_appearance_beyond_the_budget(tmp_path):
    log = write_log(tmp_path / "log.csv", [(1, "a", 0), (1, "b", 0), (2, "a", 0), (2, "b", 0), (2, "c", 1),
                                          (3, "a", 0)])
    assert replay_log(log, "QTF", "faithful", seed=0, budget=0.0)["missed"].tolist() == [0, 0]


def test_verdicts_written_as_words_or_digits_load_alike(tmp_path):
    words = write_log(tmp_path / "words.csv", [(1, "a", "True"), (1, "b", "False")])
    digits = write_log(tmp_path / "digits.csv", [(1, "a", 1), (1, "b", 0)])
    assert words["failed"].tolist() == digits["failed"].tolist() == [True, False]
