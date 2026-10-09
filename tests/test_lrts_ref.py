import os

import pytest

from faithfulcut import lrts, lrts_ref

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LRTS_REPO") and os.environ.get("LRTS_DIR")),
    reason="LRTS_REPO and LRTS_DIR are not set",
)


@pytest.mark.parametrize("project, qtf, lastfail", [("activemq", 0.768, 0.775), ("hadoop", 0.694, 0.701)])
def test_deconf_apfdc_reproduces_lrts_before_70_percent(project, qtf, lastfail):
    res = lrts_ref.reference_apfdc(project)
    res = res[res["label"] == "deconf"].merge(lrts.load_builds(project), on=["pr_name", "build_id"])
    early = res[res["position"] < 0.7]
    got = (round(early["QTF"].mean(), 4), round(early["LastFail"].mean(), 4))
    assert abs(got[0] - qtf) <= 0.01 and abs(got[1] - lastfail) <= 0.01, got
