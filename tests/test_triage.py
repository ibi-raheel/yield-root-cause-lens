"""The triage CLI is a thin wrapper: on the real SECOM data its bins must equal the pipeline's results.json."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import RESULTS, SECOM_DIR

import triage

pytestmark = pytest.mark.skipif(not (SECOM_DIR / "secom.data").exists(), reason="SECOM data not downloaded")


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    path = tmp_path_factory.mktemp("triage") / "triage.json"
    triage.main(["--secom", str(SECOM_DIR), "--out", str(path)])
    return json.loads(path.read_text())


def test_triage_matches_pipeline_results(out):
    res = json.loads(RESULTS.read_text())
    rs, pp = res["regime_screen"], res["power_plan"]
    by = {l["signal"]: l for l in out["leads"]}
    # every standard-screen lead lands in the pipeline's bucket with the pipeline's confirm-by date
    for l in rs["leads"]:
        assert by[l["feature"]]["bucket"] == l["bucket"], l["feature"]
    for l in pp["leads"]:
        assert by[l["feature"]]["days_to_confirm"] == l["days_to_confirm"], l["feature"]
    # the only WALK lead is the one within-period lead that held on later units
    walk = [l["signal"] for l in out["leads"] if l["bin"] == "WALK TO THE LINE"]
    assert walk == rs["strat_fdr_and_holdout_list"] == ["S336"]
    assert out["counts"] == {"WALK TO THE LINE": 1, "CHECK THE CHANGE RECORDS": 12, "CONFIRM OR DROP BY DAY N": 6,
                             "RE-CHART WEEKLY": 1, "DROP": 3}
    assert sorted(l["signal"] for l in out["leads"] if l["note"].startswith("re-test")) == ["S103", "S510"]
    ch, cp = out["change"], res["yield_time"]["change_point"]
    assert ch["date"] == cp["day"] and ch["p"] == cp["permutation_p"]
    assert ch["fails_per_1000_before"] == pp["economics"]["fails_per_1000"]["before"]
    assert ch["fails_per_1000_after"] == pp["economics"]["fails_per_1000"]["after"]
    assert out["mdd"] == pp["min_detectable_delta_80"]


def test_csv_input_matches_secom_on_a_real_slice(tmp_path):
    """The CSV loader gives the same answer as the SECOM loader on the same real units (first 600, 40 signals)."""
    from yrcl.load import load_secom
    X, fail, ts = load_secom(SECOM_DIR)
    cols = list(X.columns[:40])
    df = pd.concat([ts.rename("ts"), fail.rename("fail"), X[cols]], axis=1).iloc[:600]
    csv = tmp_path / "log.csv"
    df.sample(frac=1, random_state=0).to_csv(csv, index=False)  # shuffled rows: the loader must sort by time
    a = triage.triage(*triage.load_csv(csv))
    b = triage.triage(X[cols].iloc[:600].reset_index(drop=True), fail.iloc[:600].reset_index(drop=True),
                      ts.iloc[:600].reset_index(drop=True))
    assert a["change"] == b["change"]
    assert [(l["signal"], l["bin"]) for l in a["leads"]] == [(l["signal"], l["bin"]) for l in b["leads"]]
    assert np.isfinite(a["mdd"])
