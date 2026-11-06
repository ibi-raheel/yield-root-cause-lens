"""Hand calculations and library cross-checks for every statistic in the pipeline."""
import numpy as np
import pytest
from scipy import stats as sps

from yrcl import stats as st


def test_mann_whitney_hand_rank_sum():
    # a = fails, b = passes. Pooled sorted: 1(b) 2(b) 3(a) 4(b) 5(a) 6(a)
    a, b = np.array([3.0, 5.0, 6.0]), np.array([1.0, 2.0, 4.0])
    # rank sum of a = 3 + 5 + 6 = 14 ; U_a = 14 - 3*4/2 = 8
    U, p = st.mann_whitney(a, b)
    assert U == 8.0
    # pairs where a > b: (3>1,3>2, 5>1,5>2,5>4, 6>1,6>2,6>4) = 8 of 9
    assert st.cliffs_delta(U, 3, 3) == pytest.approx(2 * 8 / 9 - 1)


def test_mann_whitney_ties_hand():
    a, b = np.array([2.0, 2.0, 3.0]), np.array([1.0, 2.0])
    # pooled: 1 -> rank 1 ; three 2s -> ranks 2,3,4 avg 3 ; 3 -> rank 5. R_a = 3+3+5 = 11, U_a = 11 - 6 = 5
    U, _ = st.mann_whitney(a, b)
    assert U == 5.0


def test_mann_whitney_matches_scipy_random():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = np.round(rng.normal(0.3, 1, 40), 1)
        b = np.round(rng.normal(0, 1, 300), 1)
        U, p = st.mann_whitney(a, b)
        ref = sps.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True)
        assert U == pytest.approx(ref.statistic)
        assert p == pytest.approx(ref.pvalue, rel=1e-9)


def test_cliffs_delta_bruteforce():
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 10, 25).astype(float), rng.integers(0, 10, 30).astype(float)
    U, _ = st.mann_whitney(a, b)
    brute = (np.sign(a[:, None] - b[None, :])).mean()
    assert st.cliffs_delta(U, len(a), len(b)) == pytest.approx(brute)


def test_bh_hand():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # sorted 0.01,0.03,0.04,0.20 -> m p/j = 0.04, 0.06, 0.0533, 0.20 -> step-up min: 0.04, 0.0533, 0.0533, 0.20
    q = st.benjamini_hochberg(p)
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.20])


def test_bh_matches_scipy():
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=400), rng.uniform(0, 1e-4, 40)])
    np.testing.assert_allclose(st.benjamini_hochberg(p), sps.false_discovery_control(p, method="bh"))


def test_bh_ignores_nan():
    q = st.benjamini_hochberg(np.array([0.01, np.nan, 0.02]))
    assert np.isnan(q[1]) and q[0] == pytest.approx(0.02) and q[2] == pytest.approx(0.02)


def test_cusum_hand():
    # mu0 = 0, sigma = 1, k = 0.5, h = 2: z = 0, 1, 1, 1, ...
    S, alarm = st.cusum_upper(np.array([0, 1, 1, 1, 1, 1.0]), 0.0, 1.0, k=0.5, h=2.0)
    np.testing.assert_allclose(S, [0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert alarm == 5  # first S_t strictly above h


def test_cusum_in_control_arl_is_long():
    # ARL0 for k=0.5, h=5 is about 930 (Montgomery, Table 9.3, h=5: 938.2 with Siegmund approx 930)
    rng = np.random.default_rng(4)
    rls = []
    for _ in range(200):
        x = rng.normal(size=5000)
        _, a = st.cusum_upper(x, 0, 1, 0.5, 5.0)
        rls.append(5000 if a is None else a + 1)
    assert 700 < np.mean(rls) < 1200


def test_cusum_arl1_one_sigma_shift():
    # ARL1 for a 1-sigma shift with k=0.5, h=5 is about 10.4 (Montgomery Table 9.3)
    rng = np.random.default_rng(5)
    rls = []
    for _ in range(2000):
        _, a = st.cusum_upper(rng.normal(1.0, 1.0, 200), 0, 1, 0.5, 5.0)
        rls.append(a + 1)
    assert np.mean(rls) == pytest.approx(10.4, abs=0.6)


def test_single_change_point_step():
    y = np.r_[np.zeros(100), np.ones(60)]
    tau, gain = st.single_change_point(y, min_seg=10)
    assert tau == 100
    # SSE of one mean on a 0/1 step = n p (1-p) with p = 60/160 ; after split SSE = 0
    assert gain == pytest.approx(160 * (60 / 160) * (100 / 160))


def test_change_point_permutation_null_vs_step():
    rng = np.random.default_rng(6)
    y0 = (rng.uniform(size=600) < 0.07).astype(float)
    y1 = np.r_[(rng.uniform(size=200) < 0.15), (rng.uniform(size=400) < 0.04)].astype(float)
    assert st.change_point_permutation_p(y1, n_perm=199, min_seg=30) < 0.01
    assert st.change_point_permutation_p(y0, n_perm=199, min_seg=30) > 0.01
