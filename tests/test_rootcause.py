"""Tests for the change log, the regime-stratified screen and the power / economics plan."""
import json

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression

from conftest import RESULTS
from yrcl import rootcause as rc
from yrcl import stats as st

need_results = pytest.mark.skipif(not RESULTS.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def res():
    return json.loads(RESULTS.read_text())


# ---------------- change log ----------------
def test_planted_step_recovered_exactly():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 1000)
    v[300:] += 3.0
    s = rc.rank_step(v, 50)
    assert s["unit"] == 300
    assert s["delta"] > 0.9


def test_planted_step_with_gaps_maps_back_to_unit_index():
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1, 1000)
    v[400:] -= 4.0
    gaps = rng.random(1000) < 0.3
    v[gaps] = np.nan
    s = rc.rank_step(v, 50)
    first_obs_after = int(np.where(~np.isnan(v[400:]))[0][0]) + 400
    assert s["unit"] == first_obs_after
    assert s["delta"] < -0.9


def test_presence_step_on_logging_switch():
    m = np.zeros(600, bool)
    m[250:] = np.arange(350) % 2 == 0  # half the units stop logging from unit 250
    s = rc.presence_step(m, 50)
    assert s["unit"] in (250, 251) and s["pre"] == 1.0 and s["post"] == pytest.approx(0.5, abs=0.01)


def test_alignment_permutation_detects_planted_cluster_and_not_noise():
    rng = np.random.default_rng(3)
    n = 800
    planted = []
    for _ in range(15):
        v = rng.normal(0, 1, n)
        v[400:] += 2.5
        planted.append(v)
    noise = [rng.normal(0, 1, n) for _ in range(15)]
    obs, _, p = rc.alignment_permutation_p(planted + noise, 380, 420, 0.474, n_perm=99, seed=0)
    assert obs == 15 and p <= 0.02
    _, _, p0 = rc.alignment_permutation_p(noise, 380, 420, 0.474, n_perm=99, seed=0)
    assert p0 > 0.05


# ---------------- stratified screen ----------------
def test_stratified_single_stratum_equals_scipy_mwu():
    rng = np.random.default_rng(4)
    v = rng.normal(0, 1, 300)
    y = (rng.random(300) < 0.2).astype(int)
    v[y == 1] += 0.4
    Z, p = rc.stratified_mwu(v, y, np.zeros(300))
    ref = sps.mannwhitneyu(v[y == 1], v[y == 0], use_continuity=False, method="asymptotic")
    assert p == pytest.approx(ref.pvalue, rel=1e-9)


def test_stratified_test_holds_alpha_where_pooled_test_inflates():
    """Pure time trend + step in fail rate: no within-regime link, so stratified must stay near 5%."""
    n_sim = 300
    fp_pool = fp_strat = 0
    for i in range(n_sim):
        v, y, s = rc.simulate_confounded(seed=5000 + i)
        fp_pool += st.mann_whitney(v[y == 1], v[y == 0])[1] < 0.05
        fp_strat += rc.stratified_mwu(v, y, s)[1] < 0.05
    assert 0.02 <= fp_strat / n_sim <= 0.09
    assert fp_pool / n_sim > 0.5


def test_logistic_loglik_matches_sklearn_unpenalised():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 500)
    s = (np.arange(500) >= 200).astype(int)
    y = (rng.random(500) < 1 / (1 + np.exp(-(-2 + 0.8 * x - s)))).astype(int)
    F = np.column_stack([(s == 0), (s == 1), x]).astype(float)
    ll = rc._logit_loglik(F, y)
    m = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=5000, tol=1e-10).fit(F, y)
    p = m.predict_proba(F)[:, 1]
    ll_ref = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert ll == pytest.approx(ll_ref, abs=1e-4)


def test_youden_threshold_on_separable_data():
    v = np.array([1, 2, 3, 4, 10, 11, 12], float)
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    assert rc.youden_threshold(v, y) == 10.0


# ---------------- power ----------------
def test_power_formulas_by_hand():
    # SE0 = sqrt(471 / (3*26*444)) = sqrt(471/34632) = 0.11662
    assert rc.se_delta_null(26, 444) == pytest.approx(0.11662, abs=1e-4)
    # delta_min = (1.6449 + 0.8416) * 0.11662 = 0.2900
    assert rc.min_detectable_delta(26, 444) == pytest.approx(0.2900, abs=5e-4)
    # Noether: N = 2.4865^2 / (3 * 0.0436 * 0.9564 * 0.09) = 6.1825 / 0.011259 = 549.1
    assert rc.noether_total_n(0.3, 0.0436) == pytest.approx(549.1, abs=0.5)


def test_mc_power_matches_formula():
    rng = np.random.default_rng(7)
    pv = rng.gamma(2.0, 1.0, 2000)  # skewed, like many line signals
    se0 = rc.se_delta_null(26, 444)
    for d in (0.0, 0.3):
        emp = rc.mc_power(pv, d, 26, 444, n_sim=400, seed=1)
        form = 0.05 if d == 0 else sps.norm.cdf(d / se0 - sps.norm.ppf(0.95))
        # the null-variance formula is conservative under a shift: power is never lower than
        # predicted (within MC error) and is at most ~10 points higher for skewed data
        assert form - 0.03 <= emp <= form + 0.10


def test_individuals_limits_by_hand():
    c, lcl, ucl = rc.individuals_limits(np.array([10, 12, 11, 13, 12.0]))
    # mean 11.6, MRbar = (2+1+2+1)/4 = 1.5, 3*1.5/1.128 = 3.989
    assert c == pytest.approx(11.6) and ucl - c == pytest.approx(3.9894, abs=1e-3)


# ---------------- real-data results ----------------
@need_results
def test_results_sections_consistent(res):
    cl, rs, pp = res["change_log"], res["regime_screen"], res["power_plan"]
    assert cl["window_units"][0] <= cl["yield_step_unit"] <= cl["window_units"][1]
    assert 0 < cl["alignment_p"] <= 1 and len(cl["level_steps"]) == cl["aligned_large"]
    assert sum(rs["bucket_counts"].values()) == rs["pooled_fdr_leads"] == len(rs["leads"])
    assert rs["simulation"]["fp_stratified"] < 0.1 < rs["simulation"]["fp_pooled"]
    cp = res["yield_time"]["change_point"]
    e = pp["economics"]["fails_per_1000"]
    assert e["before"] == pytest.approx(1000 - 10 * cp["yield_before_pct"], abs=0.2)
    assert e["after"] == pytest.approx(1000 - 10 * cp["yield_after_pct"], abs=0.2)
    assert pp["min_detectable_delta_80"] == pytest.approx(rc.min_detectable_delta(pp["test_fails"], pp["test_passes"]), abs=1e-3)
