"""Statistics written out explicitly so each one can be checked by hand.

Equations are in METHOD.md.
"""
import numpy as np
from scipy import stats as sps


def rankdata_avg(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given the average rank."""
    return sps.rankdata(x, method="average")


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U for samples a (fails) and b (passes).

    U_a = R_a - n_a (n_a + 1) / 2, where R_a is the rank sum of a in the pooled sample.
    p from the normal approximation with tie correction and continuity correction
    (the same as scipy's method="asymptotic").
    Returns (U_a, p).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan, np.nan
    pooled = np.concatenate([a, b])
    r = rankdata_avg(pooled)
    Ra = r[:na].sum()
    U = Ra - na * (na + 1) / 2.0
    n = na + nb
    mu = na * nb / 2.0
    _, counts = np.unique(pooled, return_counts=True)
    tie = (counts**3 - counts).sum()
    var = na * nb / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return U, 1.0
    z = (abs(U - mu) - 0.5) / np.sqrt(var)
    p = 2 * sps.norm.sf(max(z, 0.0))
    return U, min(p, 1.0)


def cliffs_delta(U: float, na: int, nb: int) -> float:
    """Cliff's delta = P(a > b) - P(a < b) = 2 U / (na nb) - 1."""
    return 2.0 * U / (na * nb) - 1.0


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH adjusted p-values (q-values).

    q_(i) = min_{j >= i} ( m p_(j) / j ), capped at 1, for p sorted ascending.
    NaN p-values are ignored and returned as NaN.
    """
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def cusum_upper(x: np.ndarray, mu0: float, sigma: float, k: float = 0.5, h: float = 5.0):
    """One-sided upper tabular CUSUM in standardised units.

    z_t = (x_t - mu0) / sigma ;  S_t = max(0, S_{t-1} + z_t - k) ;  alarm when S_t > h.
    Returns (S, index of first alarm or None).
    """
    z = (np.asarray(x, float) - mu0) / sigma
    S = np.zeros(len(z))
    s = 0.0
    alarm = None
    for t, zt in enumerate(z):
        s = max(0.0, s + zt - k)
        S[t] = s
        if alarm is None and s > h:
            alarm = t
    return S, alarm


def cusum_lower(x, mu0, sigma, k=0.5, h=5.0):
    """Lower CUSUM: S_t = max(0, S_{t-1} - z_t - k)."""
    S, alarm = cusum_upper(-np.asarray(x, float), -mu0, sigma, k, h)
    return S, alarm


def single_change_point(y: np.ndarray, min_seg: int = 30) -> tuple[int, float]:
    """Least-squares single change in mean.

    Choose tau minimising SSE = sum_{t<tau}(y_t - ybar_1)^2 + sum_{t>=tau}(y_t - ybar_2)^2.
    Equivalent to maximising tau (n - tau) / n * (ybar_1 - ybar_2)^2.
    Returns (tau, gain) where gain is the reduction in SSE versus one mean.
    """
    y = np.asarray(y, float)
    n = len(y)
    c = np.cumsum(y)
    total = c[-1]
    taus = np.arange(min_seg, n - min_seg + 1)
    s1 = c[taus - 1]
    m1 = s1 / taus
    m2 = (total - s1) / (n - taus)
    gain = taus * (n - taus) / n * (m1 - m2) ** 2
    i = int(np.argmax(gain))
    return int(taus[i]), float(gain[i])


def change_point_permutation_p(y, n_perm=999, min_seg=30, seed=0) -> float:
    """p-value for the max-gain statistic under random reordering of y (no change)."""
    rng = np.random.default_rng(seed)
    _, g0 = single_change_point(y, min_seg)
    hits = 0
    y = np.asarray(y, float)
    for _ in range(n_perm):
        _, g = single_change_point(rng.permutation(y), min_seg)
        hits += g >= g0
    return (hits + 1) / (n_perm + 1)


def bootstrap_mean_ci(y, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    means = y[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
