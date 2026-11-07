"""Root-cause analyses on top of the screen: what stepped with yield, regime-stratified
screening, and power / sample-size planning. Equations are in METHOD.md.
"""
import numpy as np
from scipy import stats as sps

from .stats import mann_whitney, single_change_point


# ---------------------------------------------------------------- 1. step alignment
def rank_step(v: np.ndarray, min_seg: int = 50):
    """Single change point on the rank-transformed observed values of a series.

    v: full series in unit order (NaN = not measured). The step is found on the observed
    subsequence; the location is mapped back to a unit index.
    Returns dict(unit, obs_index, delta) where delta is Cliff's delta of post vs pre
    computed from the rank sum (delta > 0: values went up), or None if too few values.
    """
    v = np.asarray(v, float)
    pos = np.where(~np.isnan(v))[0]
    if len(pos) < 2 * min_seg + 1:
        return None
    x = v[pos]
    if np.all(x == x[0]):
        return None
    r = sps.rankdata(x)
    return _step_from_ranks(r, pos, min_seg)


def _step_from_ranks(r, pos, min_seg):
    k, _ = single_change_point(r, min_seg)
    n = len(r)
    n1, n2 = k, n - k
    R2 = r[k:].sum()
    U2 = R2 - n2 * (n2 + 1) / 2.0
    delta = 2.0 * U2 / (n1 * n2) - 1.0
    return {"unit": int(pos[k]), "obs_index": int(k), "delta": float(delta)}


def presence_step(ind_missing: np.ndarray, min_seg: int = 50):
    """Change point on a 0/1 presence series (1 = measured). Returns unit, pre/post presence."""
    pres = 1.0 - np.asarray(ind_missing, float)
    if pres.mean() in (0.0, 1.0):
        return None
    k, _ = single_change_point(pres, min_seg)
    return {"unit": int(k), "pre": float(pres[:k].mean()), "post": float(pres[k:].mean())}


def aligned_count(series_list, lo, hi, min_abs_delta, min_seg=50, shifts=None):
    """Number of series whose rank step lies in units [lo, hi] with |delta| >= min_abs_delta.

    shifts: optional array of circular shifts (one per series) applied before the search.
    """
    c = 0
    for j, v in enumerate(series_list):
        if shifts is not None:
            v = np.roll(v, int(shifts[j]))
        s = rank_step(v, min_seg)
        if s is not None and lo <= s["unit"] <= hi and abs(s["delta"]) >= min_abs_delta:
            c += 1
    return c


def alignment_permutation_p(series_list, lo, hi, min_abs_delta, n_perm=999, min_seg=50, seed=0):
    """Is the number of signals stepping inside [lo, hi] more than timing chance?

    Null: each series is circularly shifted by an independent uniform offset, which keeps its
    own autocorrelation and distribution but breaks its alignment with the yield step.
    Returns (observed, null_counts, p).
    """
    rng = np.random.default_rng(seed)
    n = len(series_list[0])
    obs = aligned_count(series_list, lo, hi, min_abs_delta, min_seg)
    # precompute ranks once per series (rotation does not change the ranks, only the order)
    pre = []
    for v in series_list:
        v = np.asarray(v, float)
        ok = ~np.isnan(v)
        r = np.full(n, np.nan)
        if ok.sum() >= 2 * min_seg + 1 and np.nanstd(v) > 0:
            r[ok] = sps.rankdata(v[ok])
        pre.append(r)
    null = np.zeros(n_perm, int)
    for b in range(n_perm):
        sh = rng.integers(0, n, len(pre))
        c = 0
        for r, k in zip(pre, sh):
            if np.isnan(r).all():
                continue
            rr = np.roll(r, int(k))
            pos = np.where(~np.isnan(rr))[0]
            s = _step_from_ranks(rr[pos], pos, min_seg)
            if lo <= s["unit"] <= hi and abs(s["delta"]) >= min_abs_delta:
                c += 1
        null[b] = c
    p = (1 + int((null >= obs).sum())) / (n_perm + 1)
    return obs, null, p


def bootstrap_step_location(v, unit_step, n_boot=200, min_seg=50, seed=0):
    """Resample observed values within each segment (pre / post the fitted step), re-fit, and
    return the 2.5 and 97.5% quantiles of the unit index of the step."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    pos = np.where(~np.isnan(v))[0]
    x = v[pos]
    k = int(np.searchsorted(pos, unit_step))
    a, b = x[:k], x[k:]
    locs = []
    for _ in range(n_boot):
        xb = np.concatenate([rng.choice(a, len(a)), rng.choice(b, len(b))])
        r = sps.rankdata(xb)
        kk, _ = single_change_point(r, min_seg)
        locs.append(pos[kk])
    return int(np.quantile(locs, 0.025)), int(np.quantile(locs, 0.975))


# ---------------------------------------------------------------- 2. stratified screen
def mwu_z(a, b):
    """Signed MWU z (positive: a tends to be larger), tie-corrected, no continuity correction."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan, na, nb
    pooled = np.concatenate([a, b])
    r = sps.rankdata(pooled)
    U = r[:na].sum() - na * (na + 1) / 2.0
    n = na + nb
    _, cnt = np.unique(pooled, return_counts=True)
    var = na * nb / 12.0 * ((n + 1) - (cnt**3 - cnt).sum() / (n * (n - 1)))
    if var <= 0:
        return np.nan, na, nb
    return (U - na * nb / 2.0) / np.sqrt(var), na, nb


def stratified_mwu(v, y, strata):
    """Weighted combination of within-stratum MWU z-scores (van Elteren-type weights).

    Z = sum_k w_k z_k / sqrt(sum_k w_k^2),  w_k = sqrt(n_fail,k n_pass,k / (n_k + 1)).
    Only fail-vs-pass contrasts inside a stratum enter, so a level difference between strata
    cannot create an association. Returns (Z, two-sided p).
    """
    v = np.asarray(v, float); y = np.asarray(y); strata = np.asarray(strata)
    num, den = 0.0, 0.0
    for s in np.unique(strata):
        m = strata == s
        z, na, nb = mwu_z(v[m & (y == 1)], v[m & (y == 0)])
        if np.isnan(z):
            continue
        w = np.sqrt(na * nb / (na + nb + 1.0))
        num += w * z
        den += w * w
    if den == 0:
        return np.nan, np.nan
    Z = num / np.sqrt(den)
    return float(Z), float(2 * sps.norm.sf(abs(Z)))


def _logit_loglik(X, y):
    """Maximum log-likelihood of a logistic regression (Newton-Raphson, small dense X)."""
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = np.clip(X @ beta, -30, 30)
    return float(np.sum(y * eta - np.log1p(np.exp(eta))))


def logistic_lr_p(v, y, strata):
    """Likelihood-ratio p for the signal in  logit P(fail) = regime fixed effects + b * rank(v).

    The signal enters as its standardised rank (robust to outliers); missing units dropped.
    """
    v = np.asarray(v, float); y = np.asarray(y, float); strata = np.asarray(strata)
    ok = ~np.isnan(v)
    v, y, strata = v[ok], y[ok], strata[ok]
    levels = np.unique(strata)
    F = np.column_stack([(strata == s).astype(float) for s in levels])
    r = sps.rankdata(v)
    r = (r - r.mean()) / r.std()
    l0 = _logit_loglik(F, y)
    l1 = _logit_loglik(np.column_stack([F, r]), y)
    return float(sps.chi2.sf(max(2 * (l1 - l0), 0.0), 1))


def youden_threshold(v, y):
    """Value that maximises TPR - FPR for 'fail if value beyond threshold' in the fail direction."""
    v = np.asarray(v, float); y = np.asarray(y)
    ok = ~np.isnan(v)
    v, y = v[ok], y[ok]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    sgn = 1.0 if np.median(v[y == 1]) >= np.median(v[y == 0]) else -1.0
    best, thr = -1, np.nan
    for t in np.unique(v):
        pred = sgn * v >= sgn * t
        j = pred[y == 1].mean() - pred[y == 0].mean()
        if j > best:
            best, thr = j, t
    return float(thr)


def simulate_confounded(n=1100, tau=350, p_pre=0.145, p_post=0.044, seed=0):
    """A pure time trend plus a step in fail rate: the signal has no link to failure within regime."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = (rng.random(n) < np.where(t < tau, p_pre, p_post)).astype(int)
    v = -t / n + rng.normal(0, 0.3, n)
    strata = (t >= tau).astype(int)
    return v, y, strata


# ---------------------------------------------------------------- 3. power
Z = sps.norm.ppf


def se_delta_null(na, nb):
    """SE of Cliff's delta under H0:  sqrt((na + nb + 1) / (3 na nb))."""
    return np.sqrt((na + nb + 1) / (3.0 * na * nb))


def min_detectable_delta(na, nb, alpha=0.05, power=0.8, one_sided=True):
    """delta_min = (z_{1-alpha} + z_{power}) * SE0(delta)."""
    za = Z(1 - alpha) if one_sided else Z(1 - alpha / 2)
    return float((za + Z(power)) * se_delta_null(na, nb))


def noether_total_n(delta, fail_frac, alpha=0.05, power=0.8, one_sided=True):
    """Noether (1987):  N = (z_a + z_b)^2 / (12 c (1-c) (p' - 1/2)^2),  p' = (delta + 1)/2,
    i.e.  N = (z_a + z_b)^2 / (3 c (1-c) delta^2).  Returns total units N."""
    za = Z(1 - alpha) if one_sided else Z(1 - alpha / 2)
    return float((za + Z(power)) ** 2 / (3.0 * fail_frac * (1 - fail_frac) * delta**2))


def population_delta(a, b):
    """Cliff's delta between two samples via the U statistic."""
    U, _ = mann_whitney(a, b)
    return 2 * U / (len(a) * len(b)) - 1


def shift_for_delta(pass_vals, target, rng, n_big=4000):
    """Location shift s such that delta(pass + s, pass) ~= target (bisection on a large resample)."""
    a = rng.choice(pass_vals, n_big)
    b = rng.choice(pass_vals, n_big)
    sd = np.std(pass_vals)
    lo, hi = 0.0, 10 * sd
    for _ in range(40):
        mid = (lo + hi) / 2
        d = population_delta(a + mid, b)
        lo, hi = (mid, hi) if d < target else (lo, mid)
    return (lo + hi) / 2


def mc_power(pass_vals, target, na, nb, n_sim=1000, alpha=0.05, seed=0):
    """Empirical power of the one-sided MWU when fails = real pass values shifted to a set delta."""
    rng = np.random.default_rng(seed)
    pass_vals = np.asarray(pass_vals, float)
    pass_vals = pass_vals[~np.isnan(pass_vals)]
    # jitter breaks ties so the shift moves the whole distribution (documented in METHOD.md)
    pass_vals = pass_vals + rng.normal(0, 1e-6 * (np.std(pass_vals) + 1e-12), len(pass_vals))
    s = shift_for_delta(pass_vals, target, rng) if target > 0 else 0.0
    hits = 0
    for _ in range(n_sim):
        a = rng.choice(pass_vals, na) + s
        b = rng.choice(pass_vals, nb)
        hits += sps.mannwhitneyu(a, b, alternative="greater").pvalue < alpha
    return hits / n_sim


def individuals_limits(x):
    """Individuals chart: centre = mean, limits = mean +/- 3 MRbar / 1.128."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    mr = np.abs(np.diff(x)).mean()
    c = x.mean()
    return float(c), float(c - 3 * mr / 1.128), float(c + 3 * mr / 1.128)


# ---------------------------------------------------------------- 4. lead triage
DRIFT_DELTA = 0.33   # Romano "medium": a level move between regimes worth calling drift
RHO_TIME = 0.3       # |Spearman rho| with unit order above which a lead is called a time trend


def classify_lead(within_sig, within_sign_matches, holds_later, drift, rho, d_pass_regime,
                  drift_delta=DRIFT_DELTA, rho_time=RHO_TIME) -> str:
    """Sort one lead into a bucket.

    within_sig: within-regime (stratified) BH q < 0.05. within_sign_matches: same direction as the pooled effect.
    holds_later: one-sided p < 0.05 on the later (test) units. drift: passes move by |delta| >= drift_delta
    between regimes or from train to test. rho: Spearman correlation with unit order on training units.
    d_pass_regime: Cliff's delta of passes, later regimes vs the first.
    """
    within = bool(within_sig and within_sign_matches)
    if within and holds_later:
        return "a_replicating"
    if within and drift:
        return "c_drifting"
    if within:
        return "e_unconfirmed"
    if abs(rho) > rho_time or abs(d_pass_regime) >= drift_delta:
        return "b_regime_proxy"
    return "d_noise"
