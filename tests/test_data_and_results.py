"""Checks against the real SECOM files and the exported results."""
import json

import numpy as np
import pytest
from scipy import stats as sps

from conftest import RESULTS, SECOM_DIR
from yrcl import stats as st
from yrcl.load import load_secom

need_data = pytest.mark.skipif(not (SECOM_DIR / "secom.data").exists(), reason="SECOM not downloaded")
need_results = pytest.mark.skipif(not RESULTS.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def secom():
    return load_secom(SECOM_DIR)


@pytest.fixture(scope="module")
def res():
    return json.loads(RESULTS.read_text())


@need_data
def test_shape_and_labels(secom):
    X, fail, ts = secom
    assert X.shape == (1567, 590)
    assert fail.sum() == 104  # documented in secom.names
    assert ts.is_monotonic_increasing


@need_data
@need_results
def test_yield_hand_calc(secom, res):
    # hand calc: 1463 passes / 1567 units = 93.364 %
    assert res["health"]["yield_pct"] == pytest.approx(100 * 1463 / 1567, abs=1e-3)
    assert res["health"]["passes"] == 1463


@need_data
@need_results
def test_top_signal_U_by_hand_from_rank_sums(secom, res):
    X, fail, _ = secom
    n_tr = res["health"]["train_units"]
    top = res["signals"]["top20"][0]
    v = X[top["feature"]].to_numpy(float)[:n_tr]
    y = fail.to_numpy()[:n_tr]
    ok = ~np.isnan(v)
    v, y = v[ok], y[ok]
    ranks = sps.rankdata(v)  # average ranks for ties
    na, nb = int(y.sum()), int((1 - y).sum())
    U = ranks[y == 1].sum() - na * (na + 1) / 2
    assert 2 * U / (na * nb) - 1 == pytest.approx(top["delta"], abs=1e-3)
    ref = sps.mannwhitneyu(v[y == 1], v[y == 0], method="asymptotic")
    assert ref.pvalue == pytest.approx(top["p"], abs=1e-8)


@need_results
def test_q_values_consistent_and_monotone(res):
    rows = res["signals"]["top20"]
    qs = [r["q"] for r in rows]
    assert all(q2 >= q1 - 1e-12 for q1, q2 in zip(qs, qs[1:]))
    assert res["signals"]["fdr_q_lt_05"] <= res["signals"]["raw_p_lt_05"]


@need_results
def test_split_is_time_ordered(res):
    h = res["health"]
    assert h["train_units"] + h["test_units"] == h["units"]
    assert h["train_fails"] + h["test_fails"] == h["fails"]
    assert h["split_time"] > h["first_unit"]


@need_results
def test_baselines_reported(res):
    m = res["models"]
    assert m["results"]["Majority class"]["roc_auc"] == 0.5
    assert m["results"]["Majority class"]["pr_auc"] == pytest.approx(m["test_fail_rate"], abs=1e-3)


@need_results
def test_change_point_numbers_consistent(res):
    cp = res["yield_time"]["change_point"]
    n = res["health"]["units"]
    yb, ya = cp["yield_before_pct"] / 100, cp["yield_after_pct"] / 100
    overall = (yb * cp["units_before"] + ya * cp["units_after"]) / n
    assert overall * 100 == pytest.approx(res["health"]["yield_pct"], abs=0.02)
    assert cp["ci_before_pct"][0] < cp["yield_before_pct"] < cp["ci_before_pct"][1]
