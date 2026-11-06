"""Stage 3: trained models that do real work, each scored against a simple baseline on
held-out time. Equations are in METHOD.md, section 8.

1. Root-cause ranker: stability selection (L1 logistic over day-blocked half-subsamples).
   Scored by how well its top-k signals replicate on later units, against a Welch t ranking.
2. Regime-aware failure model: trained only inside the post-change regime.
3. Anomaly detector: Isolation Forest on the 444 signals with a CUSUM on its scores, against a
   Bernoulli CUSUM on the pass/fail stream, both calibrated to the same in-control run length.
"""
import numpy as np
from scipy import stats as sps

from .stats import cliffs_delta, mann_whitney


# ---------------------------------------------------------------- blocks and resampling
def day_blocks(ts) -> np.ndarray:
    """Integer block id per unit: one block per calendar day, in time order."""
    d = np.asarray(ts.dt.normalize().astype("int64"))
    _, inv = np.unique(d, return_inverse=True)
    return inv.astype(int)


def block_subsample(blocks: np.ndarray, frac: float, rng) -> np.ndarray:
    """Unit indices of a random subset of whole blocks (without replacement)."""
    ub = np.unique(blocks)
    pick = rng.choice(ub, max(1, int(round(frac * len(ub)))), replace=False)
    return np.where(np.isin(blocks, pick))[0]


def block_bootstrap(blocks: np.ndarray, rng) -> np.ndarray:
    """Unit indices of a block bootstrap: whole blocks drawn with replacement."""
    ub = np.unique(blocks)
    pick = rng.choice(ub, len(ub), replace=True)
    members = {b: np.where(blocks == b)[0] for b in ub}
    return np.concatenate([members[b] for b in pick])


# ---------------------------------------------------------------- 1. ranking
def welch_t(v: np.ndarray, y: np.ndarray) -> float:
    """Welch t statistic (fails minus passes), NaNs dropped. NaN if undefined."""
    a, b = v[(y == 1) & ~np.isnan(v)], v[(y == 0) & ~np.isnan(v)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return np.nan
    return float((a.mean() - b.mean()) / se)


def t_ranking(X: np.ndarray, y: np.ndarray):
    """Rank columns by |Welch t|. Returns (order, sign)."""
    t = np.array([welch_t(X[:, j], y) for j in range(X.shape[1])])
    score = np.nan_to_num(np.abs(t), nan=-1.0)
    return np.argsort(-score, kind="stable"), np.sign(np.nan_to_num(t))


def standardize_fit(X: np.ndarray):
    """Median-impute and z-score with statistics from X only (no peeking at other folds)."""
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi.mean(axis=0), Xi.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return med, mu, sd


def standardize_apply(X, med, mu, sd):
    Xi = np.where(np.isnan(X), med, X)
    return (Xi - mu) / sd


def stability_selection(X: np.ndarray, y: np.ndarray, blocks: np.ndarray, Cs=(0.02, 0.05, 0.1),
                        n_resamples: int = 100, frac: float = 0.5, extra=None, seed: int = 0):
    """Selection probability of each column under L1 logistic over day-blocked subsamples.

    Each resample takes half of the training days (whole days, so consecutive units stay
    together), imputes and scales on that half only, and fits an L1 logistic at each C.
    Following Meinshausen & Buhlmann (2010), a column's score is its maximum selection
    frequency over the C grid. extra: optional (n, q) design columns appended unscaled
    (e.g. a regime indicator) that are fitted but not ranked.
    Returns (prob, mean_coef) where mean_coef is the average signed coefficient at the middle C.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    p = X.shape[1]
    sel = np.zeros((len(Cs), p))
    coef_sum = np.zeros(p)
    done = 0
    for _ in range(n_resamples):
        idx = block_subsample(blocks, frac, rng)
        if y[idx].sum() < 3:
            continue
        med, mu, sd = standardize_fit(X[idx])
        A = standardize_apply(X[idx], med, mu, sd)
        if extra is not None:
            A = np.hstack([A, extra[idx]])
        for ci, C in enumerate(Cs):
            m = LogisticRegression(penalty="l1", C=C, solver="liblinear", class_weight="balanced",
                                   max_iter=2000, random_state=0).fit(A, y[idx])
            w = m.coef_.ravel()[:p]
            sel[ci] += w != 0
            if ci == len(Cs) // 2:
                coef_sum += w
        done += 1
    prob = (sel / max(done, 1)).max(axis=0)
    return prob, coef_sum / max(done, 1)


def stability_ranking(prob: np.ndarray, mean_coef: np.ndarray):
    """Order by selection probability, ties broken by |mean coefficient|. Returns (order, sign)."""
    order = np.lexsort((-np.abs(mean_coef), -prob))
    return order, np.sign(mean_coef)


def heldout_delta(v: np.ndarray, y: np.ndarray, sign: float) -> float:
    """Cliff's delta of fails vs passes on held-out units, signed so >0 means the training
    direction replicated. NaN if fewer than 2 observed values in either group."""
    a, b = v[(y == 1) & ~np.isnan(v)], v[(y == 0) & ~np.isnan(v)]
    if len(a) < 2 or len(b) < 2 or sign == 0:
        return np.nan
    U, _ = mann_whitney(a, b)
    return float(sign * cliffs_delta(U, len(a), len(b)))


def heldout_p(v, y, sign) -> float:
    a, b = v[(y == 1) & ~np.isnan(v)], v[(y == 0) & ~np.isnan(v)]
    if len(a) < 2 or len(b) < 2 or sign == 0:
        return np.nan
    alt = "greater" if sign > 0 else "less"
    return float(sps.mannwhitneyu(a, b, alternative=alt).pvalue)


def topk_scores(Xte: np.ndarray, yte: np.ndarray, order, sign, k: int):
    """Held-out signed deltas and one-sided p-values of the top-k columns of a ranking."""
    cols = list(order[:k])
    d = np.array([heldout_delta(Xte[:, j], yte, sign[j]) for j in cols])
    p = np.array([heldout_p(Xte[:, j], yte, sign[j]) for j in cols])
    return cols, d, p


# ---------------------------------------------------------------- 3. change detection
def bernoulli_cusum_llr(y: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Per-unit log-likelihood-ratio increments for a shift of the fail rate from p0 to p1:
    fail -> log(p1/p0), pass -> log((1-p1)/(1-p0)) (Reynolds & Stoumbos 1999)."""
    y = np.asarray(y)
    return np.where(y == 1, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))


def cusum_path(inc: np.ndarray, h: float):
    """S_t = max(0, S_{t-1} + inc_t). Returns (path, index of first S_t >= h or None)."""
    S = np.zeros(len(inc))
    s = 0.0
    alarm = None
    for t, x in enumerate(inc):
        s = max(0.0, s + x)
        S[t] = s
        if alarm is None and s >= h:
            alarm = t
    return S, alarm


def run_length(inc: np.ndarray, h: float) -> int:
    """Units until the first alarm (len(inc) if none)."""
    _, a = cusum_path(inc, h)
    return len(inc) if a is None else a + 1


def calibrate_h(sample_inc, target_arl: float, h_grid, n_sim: int = 300, length: int = 4000, seed: int = 0):
    """Smallest h whose simulated in-control average run length reaches target_arl.

    sample_inc(rng, length) -> one in-control increment series. Run lengths are censored at
    `length`, so the estimate is a lower bound when many runs reach it.
    Returns (h, arl_at_h, table).
    """
    rng = np.random.default_rng(seed)
    sims = [sample_inc(rng, length) for _ in range(n_sim)]
    table = []
    for h in h_grid:
        arl = float(np.mean([run_length(s, h) for s in sims]))
        table.append((float(h), arl))
        if arl >= target_arl:
            return float(h), arl, table
    return float(h_grid[-1]), table[-1][1], table


def block_resample_series(x: np.ndarray, blocks: np.ndarray, rng, length: int) -> np.ndarray:
    """Concatenate whole blocks of x drawn with replacement until `length` values
    (a moving-block style bootstrap that keeps within-day autocorrelation)."""
    ub = np.unique(blocks)
    members = [x[blocks == b] for b in ub]
    out = []
    n = 0
    while n < length:
        seg = members[rng.integers(len(members))]
        out.append(seg)
        n += len(seg)
    return np.concatenate(out)[:length]
