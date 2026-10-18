import numpy as np
from synthetic import make_suite

from faithfulcut.drift import burst, renamed, slowed


def four_builds(failing=None):
    failing = failing or {}
    rows = [(np.arange(3), np.isin(np.arange(3), failing.get(t, [])), np.ones(3)) for t in range(4)]
    start = np.arange(4) * 10.0
    return make_suite(rows, start, start + 1)


def test_renamed_classes_take_new_identities_from_the_injection_on_and_keep_their_relevance():
    s = four_builds()
    ir = {"offsets": np.array([0, 3, 6, 9, 12]), "cls": np.tile(np.arange(3), 4), "score": np.ones(12)}
    out, out_ir = renamed(s, ir, 2, 1 / 3, np.random.default_rng(0))
    moved = int(np.setdiff1d(np.arange(3), out.cls[out.offsets[2]:out.offsets[3]])[0])
    assert len(out.names) == 4 and out.names[3] == f"{s.names[moved]}~renamed"
    assert np.array_equal(out.cls[:6], s.cls[:6]) and (out.cls[6:] == 3).sum() == 2
    assert np.array_equal(out_ir["cls"][:6], ir["cls"][:6]) and (out_ir["cls"][6:] == 3).sum() == 2


def test_a_burst_makes_stable_classes_fail_in_its_window():
    s = four_builds({0: [0]})
    out, label = burst(s, s.failed.copy(), 1, 1, 2, np.random.default_rng(0))
    hit = out.failed & ~s.failed
    assert hit[s.offsets[1]:s.offsets[3]].sum() == 2 and not hit[s.offsets[3]:].any()
    assert not np.isin(s.cls[hit], [0]).any() and np.array_equal(label, out.failed)


def test_a_duration_shift_slows_a_share_of_the_classes_from_the_injection_on():
    out = slowed(four_builds(), 2, 2 / 3, 4.0, np.random.default_rng(0))
    assert (out.duration[:6] == 1).all() and (out.duration[6:] == 4).sum() == 4
