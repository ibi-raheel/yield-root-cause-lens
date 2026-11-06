"""Stage 3 (trained models vs baselines): hand checks, leakage guards, planted-signal recovery,
and consistency of the exported AI-vs-baseline numbers."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from conftest import RESULTS
from yrcl import ai

need_results = pytest.mark.skipif(not RESULTS.exists(), reason="run the pipeline first")


def test_bernoulli_cusum_by_hand():
    # p0 = 0.2, p1 = 0.1: a pass adds ln(0.9/0.8) = 0.117783, a fail adds ln(0.5) = -0.693147
    y = np.array([0, 0, 1, 0, 0, 0])
    inc = ai.bernoulli_cusum_llr(y, 0.2, 0.1)
    assert inc[0] == pytest.approx(0.117783, abs=1e-6)
    assert inc[2] == pytest.approx(-0.693147, abs=1e-6)
    S, alarm = ai.cusum_path(inc, h=0.3)
    # S: 0.1178, 0.2356, max(0, 0.2356-0.6931)=0, 0.1178, 0.2356, 0.3533 -> alarm at index 5
    assert S == pytest.approx([0.117783, 0.235566, 0.0, 0.117783, 0.235566, 0.353349], abs=1e-5)
    assert alarm == 5
    assert ai.run_length(inc, 0.3) == 6
    assert ai.run_length(inc, 10.0) == len(inc)


def test_welch_t_matches_scipy_and_hand():
    # hand: fails [4, 6] mean 5 var 2; passes [1, 2, 3] mean 2 var 1; se = sqrt(2/2 + 1/3) = 1.154701
    v = np.array([4.0, 6.0, 1.0, 2.0, 3.0, np.nan])
    y = np.array([1, 1, 0, 0, 0, 0])
    assert ai.welch_t(v, y) == pytest.approx(3 / np.sqrt(1 + 1 / 3), rel=1e-12)
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 2, 30), rng.normal(0, 1, 200)
    v = np.concatenate([a, b]); y = np.r_[np.ones(30), np.zeros(200)].astype(int)
    assert ai.welch_t(v, y) == pytest.approx(sps.ttest_ind(a, b, equal_var=False).statistic, rel=1e-10)


def test_block_resampling_keeps_days_whole():
    ts = pd.Series(pd.to_datetime(["2008-08-01 01:00", "2008-08-01 05:00", "2008-08-02 03:00",
                                   "2008-08-03 01:00", "2008-08-03 02:00", "2008-08-03 09:00"]))
    blocks = ai.day_blocks(ts)
    assert blocks.tolist() == [0, 0, 1, 2, 2, 2]
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = ai.block_subsample(blocks, 0.5, rng)
        for b in np.unique(blocks[idx]):          # a sampled day comes with all of its units
            assert set(np.where(blocks == b)[0]) <= set(idx)
        bb = ai.block_bootstrap(blocks, rng)
        assert len(set(np.unique(blocks[bb]))) >= 1
        for b in np.unique(blocks[bb]):
            assert (np.sum(blocks[bb] == b) % np.sum(blocks == b)) == 0


def test_cut_at_day_never_splits_a_day():
    from run_ai import cut_at_day
    day = np.repeat(np.arange(10), 7)
    for f in [0.33, 0.5, 0.71, 0.9]:
        c = cut_at_day(day, f)
        assert day[c] != day[c - 1]


def test_standardize_uses_training_rows_only():
    Xtr = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    med, mu, sd = ai.standardize_fit(Xtr)
    assert med.tolist() == [3.0, 3.0]
    Z = ai.standardize_apply(np.array([[100.0, np.nan]]), med, mu, sd)
    # the test row does not move the training statistics; NaN takes the training median
    assert Z[0, 1] == pytest.approx((3.0 - 3.0) / np.std([3.0, 2.0, 4.0]))
    assert Z[0, 0] == pytest.approx((100 - 3.0) / np.std([1.0, 3.0, 5.0]))


def test_stability_selection_recovers_planted_signals():
    rng = np.random.default_rng(3)
    n, p = 600, 60
    X = rng.normal(size=(n, p))
    logit = -3.0 + 1.5 * X[:, 4] - 1.5 * X[:, 17]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    blocks = np.repeat(np.arange(60), 10)
    prob, coef = ai.stability_selection(X, y, blocks, (0.02, 0.05, 0.1), n_resamples=40, seed=0)
    order, sign = ai.stability_ranking(prob, coef)
    assert set(order[:2]) == {4, 17}
    assert sign[4] > 0 and sign[17] < 0
    assert np.median(prob) < 0.5


def test_heldout_delta_sign_convention():
    v = np.array([5.0, 6.0, 7.0, 1.0, 2.0, 3.0])
    y = np.array([1, 1, 1, 0, 0, 0])
    assert ai.heldout_delta(v, y, +1.0) == pytest.approx(1.0)
    assert ai.heldout_delta(v, y, -1.0) == pytest.approx(-1.0)
    assert np.isnan(ai.heldout_delta(v, y, 0.0))


def test_calibrate_h_reaches_target():
    # iid Bernoulli(0.2) stream monitored for a drop to 0.1
    sample = lambda g, L: ai.bernoulli_cusum_llr((g.random(L) < 0.2).astype(int), 0.2, 0.1)
    h, arl, table = ai.calibrate_h(sample, 200, np.arange(0.5, 8.01, 0.25), n_sim=200, length=3000, seed=0)
    assert arl >= 200
    assert all(a < 200 for _, a in table[:-1])


@need_results
def test_exported_ai_section_is_consistent():
    res = json.loads(RESULTS.read_text())["ai"]
    for rw in res["ranker"]["per_cut"]:
        assert rw["diff_ai_vs_t"] == pytest.approx(rw["ai_mean_delta"] - rw["t_mean_delta"], abs=2e-4)
        lo, hi = rw["diff_ai_vs_t_ci"]
        assert lo <= hi
    cuts = [rw["cut_unit"] for rw in res["ranker"]["per_cut"]]
    assert cuts == sorted(cuts)
    for rw in res["regime_model"]["per_cut"]:
        assert rw["train_regime_units"] < rw["train_pooled_units"]
        assert 0 <= rw["auc_regime"] <= 1
    an = res["anomaly"]
    assert an["isolation_forest"]["arl0_sim"] >= an["target_arl0_units"]
    assert an["bernoulli_cusum"]["arl0_sim"] >= an["target_arl0_units"]
    assert an["reference_units"] < an["ci_lower_unit"] < an["change_unit"]
