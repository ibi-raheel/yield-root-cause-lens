"""Stage 3: trained models scored against simple baselines on held-out time.

Usage:  python -I pipeline/run_ai.py <secom_dir> <results_json>

Adds res["ai"] with three sections (ranker, regime_model, anomaly). Every comparison uses
time-ordered, day-blocked validation and a day-block bootstrap CI on the AI-minus-baseline
difference. Nothing here sees units after a training cut when it is fitted.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from run_pipeline import MAX_MISSING, r
from yrcl import ai
from yrcl import stats as st
from yrcl.load import load_secom

SEED = 0
CUT_FRACS = [0.5, 0.6, 0.7, 0.8]   # training cuts (snapped to the next day boundary)
PRIMARY_CUT = 0.7
KS = [5, 10, 20]
PRIMARY_K = 10
N_STAB = 100
CS = (0.02, 0.05, 0.1)
N_BOOT = 1000
REF_DAYS = 12                       # anomaly reference window: first 12 days of production
TARGET_ARL0 = 500                   # in-control average run length, units, for both detectors
CUSUM_K = 0.5
NAMES = ("ai", "t", "mwu", "ai_regime", "t_regime")
# pre-declared comparisons: (AI ranking, baseline ranking)
PAIRS = {"ai_vs_t": ("ai", "t"), "ai_vs_mwu": ("ai", "mwu"), "ai_regime_vs_t_regime": ("ai_regime", "t_regime")}


def cut_at_day(day_id: np.ndarray, frac: float) -> int:
    """First unit index of the first day that starts at or after frac*n (no day is split)."""
    n = len(day_id)
    i = int(round(frac * n))
    while 0 < i < n and day_id[i] == day_id[i - 1]:
        i += 1
    return i


def usable_cols(X: pd.DataFrame, idx) -> list:
    sub = X.iloc[idx]
    return list(X.columns[(sub.isna().mean() <= MAX_MISSING) & (sub.nunique(dropna=True) > 1)])


def boot_ci(vals, lo=0.025, hi=0.975):
    v = np.asarray([x for x in vals if not np.isnan(x)])
    return [r(np.quantile(v, lo), 4), r(np.quantile(v, hi), 4)], r((v <= 0).mean(), 4)


# ============================================================ 1. root-cause ranker
def ranker_section(X, y, ts, day_id):
    n = len(y)
    cuts = {f: cut_at_day(day_id, f) for f in CUT_FRACS}
    rows, store = [], {}
    for f, c in cuts.items():
        tr = np.arange(c)
        cols = usable_cols(X, tr)
        A = X[cols].to_numpy(float)
        tau_tr, _ = st.single_change_point(y[tr].astype(float), 50)   # regime from train only
        reg = (np.arange(n) >= tau_tr).astype(float)[:, None] * 3.0     # scaled: lightly penalised
        prob, mc = ai.stability_selection(A[tr], y[tr], day_id[tr], CS, N_STAB, 0.5, extra=reg[tr], seed=SEED + c)
        o_ai, s_ai = ai.stability_ranking(prob, mc)
        o_t, s_t = ai.t_ranking(A[tr], y[tr])
        # secondary baseline: Mann-Whitney ranking (rank-based, robust to outliers)
        from run_pipeline import screen
        scr = screen(X[cols].iloc[tr], y[tr])
        pm = scr["p"].reindex(cols).to_numpy()
        o_u = np.argsort(np.nan_to_num(pm, nan=2.0), kind="stable")
        s_u = np.sign(np.nan_to_num(scr["delta"].reindex(cols).to_numpy()))
        # regime-aware pair: both rankers see only the post-change units of the training window
        rt = np.arange(tau_tr, c)
        prob_r, mc_r = ai.stability_selection(A[rt], y[rt], day_id[rt], CS, N_STAB, 0.5, seed=SEED + c + 1)
        o_air, s_air = ai.stability_ranking(prob_r, mc_r)
        o_tr, s_tr = ai.t_ranking(A[rt], y[rt])
        store[f] = dict(cut=c, cols=cols, A=A, rankings={"ai": (o_ai, s_ai), "t": (o_t, s_t), "mwu": (o_u, s_u),
                                                         "ai_regime": (o_air, s_air), "t_regime": (o_tr, s_tr)},
                        prob=prob, tau=tau_tr)
        te = np.arange(c, n)
        for k in KS:
            res_k = {}
            for name, (o, s) in store[f]["rankings"].items():  # noqa
                cl, d, p = ai.topk_scores(A[te], y[te], o, s, k)
                res_k[name] = dict(mean_delta=float(np.nanmean(d)), replicated=int(np.nansum(p < 0.05)),
                                   cols=[cols[j] for j in cl])
            rows.append(dict(cut_frac=f, cut_unit=c, cut_day=str(ts.iloc[c].date()), k=k,
                             test_units=int(n - c), test_fails=int(y[te].sum()), tau_train=int(tau_tr), **{
                                 f"{nm}_mean_delta": r(v["mean_delta"], 4) for nm, v in res_k.items()}, **{
                                 f"{nm}_replicated": v["replicated"] for nm, v in res_k.items()},
                             ai_top=res_k["ai"]["cols"] if k == PRIMARY_K else None,
                             t_top=res_k["t"]["cols"] if k == PRIMARY_K else None,
                             mwu_top=res_k["mwu"]["cols"] if k == PRIMARY_K else None))
    # common day-block bootstrap over the held-out period of the earliest cut
    first = min(cuts.values())
    rng = np.random.default_rng(SEED + 99)
    days = np.unique(day_id[first:])
    members = {d: np.where(day_id == d)[0] for d in days}
    boots = {(f, k, b): [] for f in CUT_FRACS for k in KS for b in PAIRS}
    pooled = {(k, b): [] for k in KS for b in PAIRS}
    for _ in range(N_BOOT):
        pick = rng.choice(days, len(days), replace=True)
        ub = np.concatenate([members[d] for d in pick])
        per = {(k, b): [] for k in KS for b in PAIRS}
        for f in CUT_FRACS:
            S = store[f]
            u = ub[ub >= S["cut"]]
            Ab, yb = S["A"][u], y[u]
            if yb.sum() < 3:
                continue
            cache = {}   # unsigned held-out delta per column, computed once per resample

            def dlt_of(j, sgn):
                if j not in cache:
                    cache[j] = ai.heldout_delta(Ab[:, j], yb, 1.0)
                return sgn * cache[j] if sgn != 0 else np.nan

            for k in KS:
                md = {}
                for nm in NAMES:
                    o, s = S["rankings"][nm]
                    md[nm] = np.nanmean([dlt_of(j, s[j]) for j in o[:k]])
                for b, (x1, x2) in PAIRS.items():
                    dlt = md[x1] - md[x2]
                    boots[(f, k, b)].append(dlt)
                    per[(k, b)].append(dlt)
        for key, v in per.items():
            if len(v) == len(CUT_FRACS):
                pooled[key].append(float(np.mean(v)))
    for rw in rows:
        f, k = rw["cut_frac"], rw["k"]
        for b, (x1, x2) in PAIRS.items():
            rw[f"diff_{b}"] = r(rw[f"{x1}_mean_delta"] - rw[f"{x2}_mean_delta"], 4)
            rw[f"diff_{b}_ci"], rw[f"diff_{b}_p_le0"] = boot_ci(boots[(f, k, b)])
    pooled_rows = []
    for k in KS:
        sub = [rw for rw in rows if rw["k"] == k]
        pr = {"k": k}
        for nm in NAMES:
            pr[f"{nm}_mean_delta"] = r(np.mean([rw[f"{nm}_mean_delta"] for rw in sub]), 4)
            pr[f"{nm}_replicated"] = int(sum(rw[f"{nm}_replicated"] for rw in sub))
        for b, (x1, x2) in PAIRS.items():
            pr[f"diff_{b}"] = r(pr[f"{x1}_mean_delta"] - pr[f"{x2}_mean_delta"], 4)
            pr[f"diff_{b}_ci"], pr[f"diff_{b}_p_le0"] = boot_ci(pooled[(k, b)])
        pr["slots"] = k * len(CUT_FRACS)
        pooled_rows.append(pr)
    S = store[PRIMARY_CUT]
    top_stab = [{"feature": S["cols"][j], "selection_prob": r(S["prob"][j], 2)} for j in S["rankings"]["ai"][0][:15]]
    return {"cut_fracs": CUT_FRACS, "primary_cut": PRIMARY_CUT, "ks": KS, "primary_k": PRIMARY_K,
            "n_resamples": N_STAB, "C_grid": list(CS), "subsample": "half of training days, whole days",
            "metric": "mean held-out Cliff's delta of the top-k, signed by the training direction",
            "per_cut": rows, "pooled": pooled_rows, "top_stability_primary": top_stab}, store


# ============================================================ 2. regime-aware model
def l1_fit_predict(Atr, ytr, Ate, blocks_tr, Cs=(0.01, 0.03, 0.1, 0.3), seed=0):
    """L1 logistic with C chosen by forward-chaining CV over whole days inside the training set."""
    med, mu, sd = ai.standardize_fit(Atr)
    Ztr, Zte = ai.standardize_apply(Atr, med, mu, sd), ai.standardize_apply(Ate, med, mu, sd)
    ub = np.unique(blocks_tr)
    folds = np.array_split(ub, 4)
    best, bestC = -1, Cs[0]
    for C in Cs:
        sc = []
        for i in range(1, 4):
            a = np.isin(blocks_tr, np.concatenate(folds[:i]))
            b = np.isin(blocks_tr, folds[i])
            if ytr[a].sum() < 3 or ytr[b].sum() < 1:
                continue
            md, m_, s_ = ai.standardize_fit(Atr[a])
            m = LogisticRegression(penalty="l1", C=C, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)
            m.fit(ai.standardize_apply(Atr[a], md, m_, s_), ytr[a])
            sc.append(roc_auc_score(ytr[b], m.decision_function(ai.standardize_apply(Atr[b], md, m_, s_))))
        if sc and np.mean(sc) > best:
            best, bestC = float(np.mean(sc)), C
    m = LogisticRegression(penalty="l1", C=bestC, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed).fit(Ztr, ytr)
    return m.decision_function(Zte), bestC, int((m.coef_ != 0).sum())


def regime_section(X, y, ts, day_id, store):
    n = len(y)
    rows, preds = [], {}
    for f in CUT_FRACS[1:]:
        S = store[f]
        c, cols, A, tau_tr = S["cut"], S["cols"], S["A"], S["tau"]
        te = np.arange(c, n)
        pooled_tr = np.arange(c)
        regime_tr = np.arange(tau_tr, c)
        s_pool, C_pool, nz_pool = l1_fit_predict(A[pooled_tr], y[pooled_tr], A[te], day_id[pooled_tr])
        s_reg, C_reg, nz_reg = l1_fit_predict(A[regime_tr], y[regime_tr], A[te], day_id[regime_tr])
        # simple baseline: the single best Welch-t signal inside the post-change training regime
        o_t, s_t = ai.t_ranking(A[regime_tr], y[regime_tr])
        j = o_t[0]
        med = np.nanmedian(A[regime_tr][:, j])
        s_one = s_t[j] * np.where(np.isnan(A[te][:, j]), med, A[te][:, j])
        preds[f] = dict(te=te, pool=s_pool, reg=s_reg, one=s_one)
        yt = y[te]
        rows.append(dict(cut_frac=f, cut_day=str(ts.iloc[c].date()), train_regime_units=int(len(regime_tr)),
                         train_regime_fails=int(y[regime_tr].sum()), train_pooled_units=int(c),
                         test_units=int(len(te)), test_fails=int(yt.sum()),
                         auc_regime=r(roc_auc_score(yt, s_reg), 4), auc_pooled=r(roc_auc_score(yt, s_pool), 4),
                         auc_single=r(roc_auc_score(yt, s_one), 4), single_feature=cols[j],
                         ap_regime=r(average_precision_score(yt, s_reg), 4),
                         ap_pooled=r(average_precision_score(yt, s_pool), 4),
                         ap_single=r(average_precision_score(yt, s_one), 4), fail_rate=r(yt.mean(), 4),
                         C_regime=C_reg, nonzero_regime=nz_reg, C_pooled=C_pool, nonzero_pooled=nz_pool))
    rng = np.random.default_rng(SEED + 7)
    first = min(store[f]["cut"] for f in CUT_FRACS[1:])
    days = np.unique(day_id[first:])
    members = {d: np.where(day_id == d)[0] for d in days}
    bt = {(f, b): [] for f in CUT_FRACS[1:] for b in ("pool", "one")}
    bt_mean = {"pool": [], "one": []}
    for _ in range(N_BOOT):
        pick = rng.choice(days, len(days), replace=True)
        ub = np.concatenate([members[d] for d in pick])
        n_before = {b: len(bt[(CUT_FRACS[-1], b)]) for b in ("pool", "one")}
        cur = {"pool": [], "one": []}
        for f, P in preds.items():
            c = P["te"][0]
            u = ub[ub >= c] - c
            yb = y[P["te"]][u]
            if yb.sum() < 2 or yb.sum() == len(yb):
                continue
            a_reg = roc_auc_score(yb, P["reg"][u])
            bt[(f, "pool")].append(a_reg - roc_auc_score(yb, P["pool"][u]))
            bt[(f, "one")].append(a_reg - roc_auc_score(yb, P["one"][u]))
            cur["pool"].append(bt[(f, "pool")][-1]); cur["one"].append(bt[(f, "one")][-1])
        for b in ("pool", "one"):
            if len(cur[b]) == len(preds):
                bt_mean[b].append(float(np.mean(cur[b])))
    for rw in rows:
        f = rw["cut_frac"]
        rw["diff_vs_pooled"] = r(rw["auc_regime"] - rw["auc_pooled"], 4)
        rw["diff_vs_pooled_ci"], rw["diff_vs_pooled_p_le0"] = boot_ci(bt[(f, "pool")])
        rw["diff_vs_single"] = r(rw["auc_regime"] - rw["auc_single"], 4)
        rw["diff_vs_single_ci"], rw["diff_vs_single_p_le0"] = boot_ci(bt[(f, "one")])
    mean_row = {"auc_regime": r(np.mean([rw["auc_regime"] for rw in rows]), 4),
                "auc_pooled": r(np.mean([rw["auc_pooled"] for rw in rows]), 4),
                "auc_single": r(np.mean([rw["auc_single"] for rw in rows]), 4)}
    mean_row["diff_vs_pooled"] = r(mean_row["auc_regime"] - mean_row["auc_pooled"], 4)
    mean_row["diff_vs_pooled_ci"], mean_row["diff_vs_pooled_p_le0"] = boot_ci(bt_mean["pool"])
    mean_row["diff_vs_single"] = r(mean_row["auc_regime"] - mean_row["auc_single"], 4)
    mean_row["diff_vs_single_ci"], mean_row["diff_vs_single_p_le0"] = boot_ci(bt_mean["one"])
    # sensitivity to the solver's coordinate-shuffle seed (liblinear), which can flip the CV choice of C
    sens = []
    for sd_ in range(6):
        a_r, a_p = [], []
        for f in CUT_FRACS[1:]:
            S = store[f]
            c, A, tau_tr = S["cut"], S["A"], S["tau"]
            te = np.arange(c, n)
            a_r.append(roc_auc_score(y[te], l1_fit_predict(A[tau_tr:c], y[tau_tr:c], A[te], day_id[tau_tr:c], seed=sd_)[0]))
            a_p.append(roc_auc_score(y[te], l1_fit_predict(A[:c], y[:c], A[te], day_id[:c], seed=sd_)[0]))
        sens.append({"seed": sd_, "auc_regime_mean": r(np.mean(a_r), 4), "auc_pooled_mean": r(np.mean(a_p), 4),
                     "diff_vs_pooled": r(np.mean(a_r) - np.mean(a_p), 4),
                     "diff_vs_single": r(np.mean(a_r) - mean_row["auc_single"], 4)})
    return {"model": "L1 logistic, C by forward-chaining CV over whole days", "per_cut": rows, "mean_over_cuts": mean_row,
            "solver_seed_sensitivity": sens}


# ============================================================ 3. anomaly detection
def anomaly_section(X, y, ts, day_id, tau, loc_lo_day):
    n = len(y)
    R = int(np.where(day_id == day_id[0] + REF_DAYS)[0][0]) if (day_id == day_id[0] + REF_DAYS).any() else 150
    ref = np.arange(R)
    cols = usable_cols(X, ref)
    A = X[cols].to_numpy(float)
    med = np.nanmedian(A[ref], axis=0)
    q1, q3 = np.nanpercentile(A[ref], [25, 75], axis=0)
    iqr = np.where((q3 - q1) > 0, q3 - q1, np.nanstd(A[ref], axis=0) + 1e-12)
    Z = np.clip((np.where(np.isnan(A), med, A) - med) / iqr, -20, 20)

    def iforest(idx, seed):
        return IsolationForest(n_estimators=300, max_samples=min(128, len(idx)), random_state=seed).fit(Z[idx])

    # out-of-sample reference scores: leave one day out
    oos = np.zeros(R)
    for d in np.unique(day_id[ref]):
        hold = ref[day_id[ref] == d]
        m = iforest(ref[day_id[ref] != d], SEED + int(d))
        oos[hold] = -m.score_samples(Z[hold])
    m = iforest(ref, SEED)
    score = -m.score_samples(Z)
    score[ref] = oos
    mu, sd = oos.mean(), oos.std(ddof=1)
    z = (score - mu) / sd
    inc_if = z - CUSUM_K
    p0 = float(y[ref].mean())
    p1 = p0 / 2.0
    inc_b = ai.bernoulli_cusum_llr(y, p0, p1)
    rb = day_id[ref]
    h_grid_if = np.round(np.arange(1.0, 60.01, 0.25), 2)
    h_grid_b = np.round(np.arange(0.5, 15.01, 0.05), 2)
    h_if, arl_if, _ = ai.calibrate_h(lambda g, L: ai.block_resample_series(inc_if[ref], rb, g, L),
                                     TARGET_ARL0, h_grid_if, 300, 4000, SEED + 1)
    h_b, arl_b, _ = ai.calibrate_h(lambda g, L: ai.block_resample_series(inc_b[ref], rb, g, L),
                                   TARGET_ARL0, h_grid_b, 300, 4000, SEED + 2)
    # the same calibration with iid Bernoulli(p0), the textbook in-control model
    h_b_iid, arl_b_iid, _ = ai.calibrate_h(lambda g, L: ai.bernoulli_cusum_llr((g.random(L) < p0).astype(int), p0, p1),
                                           TARGET_ARL0, h_grid_b, 300, 4000, SEED + 3)
    L = int(np.where(ts.dt.normalize() >= pd.Timestamp(loc_lo_day))[0][0])

    def detect(inc, h, start, L_, tau_):
        """Run from `start`; alarms before L_ are false alarms (path resets). Returns
        (false_alarm_units, first alarm unit >= L_ or None)."""
        s, fa = 0.0, []
        for t in range(start, len(inc)):
            s = max(0.0, s + inc[t])
            if s >= h:
                if t < L_:
                    fa.append(t); s = 0.0
                else:
                    return fa, t
        return fa, None

    out = {}
    for name, inc, h in [("isolation_forest", inc_if, h_if), ("bernoulli_cusum", inc_b, h_b),
                         ("bernoulli_cusum_iid_h", inc_b, h_b_iid)]:
        fa, a = detect(inc, h, R, L, tau)
        out[name] = {"h": h, "false_alarms_before_ci": len(fa), "false_alarm_days": [str(ts.iloc[t].date()) for t in fa],
                     "alarm_unit": a, "alarm_day": None if a is None else str(ts.iloc[a].date()),
                     "delay_units": None if a is None else int(a - tau),
                     "delay_days": None if a is None else r((ts.iloc[a] - ts.iloc[tau]).total_seconds() / 86400, 1)}
    out["isolation_forest"]["arl0_sim"] = r(arl_if, 1)
    out["bernoulli_cusum"]["arl0_sim"] = r(arl_b, 1)
    out["bernoulli_cusum_iid_h"]["arl0_sim"] = r(arl_b_iid, 1)
    # standard 50-unit batch CUSUM on the fail rate, mu0 from the reference window
    B = 50
    nb = (n - R) // B
    bx = y[R:R + nb * B].reshape(nb, B).mean(axis=1)
    sig = np.sqrt(p0 * (1 - p0) / B)
    _, al = st.cusum_lower(bx, p0, sig, 0.5, 5.0)
    a_batch = None if al is None else R + (al + 1) * B - 1   # alarm known at the end of the batch
    out["batch_cusum_50"] = {"h": 5.0, "k": 0.5, "alarm_unit": a_batch,
                             "alarm_day": None if a_batch is None else str(ts.iloc[a_batch].date()),
                             "delay_units": None if a_batch is None else int(a_batch - tau)}
    # day-block bootstrap of the delay difference: resample whole days within the monitored
    # pre-change stretch and within the post-change stretch (scores held fixed)
    blk = day_id.copy() * 2 + (np.arange(n) >= tau)
    pre_idx, post_idx = np.arange(R, tau), np.arange(tau, n)
    rng = np.random.default_rng(SEED + 5)
    diffs, fa_if, fa_b, miss_if, miss_b = [], 0, 0, 0, 0
    for _ in range(N_BOOT):
        pick_pre = ai.block_resample_series(pre_idx, blk[pre_idx], rng, len(pre_idx))
        pick_post = ai.block_resample_series(post_idx, blk[post_idx], rng, len(post_idx))
        order = np.concatenate([np.arange(R), pick_pre, pick_post])
        res = []
        for inc, h in [(inc_if, h_if), (inc_b, h_b)]:
            fa, a = detect(inc[order], h, R, L, tau)
            res.append((fa, a))
        fa_if += len(res[0][0]) > 0; fa_b += len(res[1][0]) > 0
        if res[0][1] is None: miss_if += 1
        if res[1][1] is None: miss_b += 1
        if res[0][1] is not None and res[1][1] is not None:
            diffs.append(float(res[0][1] - res[1][1]))
    d = np.array(diffs)
    out["bootstrap"] = {"n": N_BOOT, "delay_diff_if_minus_bernoulli_median": r(np.median(d), 1),
                        "delay_diff_ci": [r(np.quantile(d, 0.025), 1), r(np.quantile(d, 0.975), 1)],
                        "share_if_earlier": r((d < 0).mean(), 3),
                        "share_runs_if_false_alarm": r(fa_if / N_BOOT, 3), "share_runs_bern_false_alarm": r(fa_b / N_BOOT, 3),
                        "share_if_no_detection": r(miss_if / N_BOOT, 3), "share_bern_no_detection": r(miss_b / N_BOOT, 3)}
    # score trace for inspection: daily mean z
    dd = pd.DataFrame({"day": ts.dt.date.astype(str), "z": z, "fail": y}).groupby("day").agg(z=("z", "mean"), fails=("fail", "sum"), units=("fail", "size")).reset_index()
    out["mean_z_reference_oos"] = r(z[:R].mean(), 3)
    out["mean_z_pre_change_monitored"] = r(z[R:tau].mean(), 3)
    out["mean_z_post_change"] = r(z[tau:].mean(), 3)
    out.update({"reference_units": R, "reference_end_day": str(ts.iloc[R].date()), "reference_fail_rate": r(p0, 4),
                "p1": r(p1, 4), "signals_used": len(cols), "target_arl0_units": TARGET_ARL0, "cusum_k_if": CUSUM_K,
                "change_unit": int(tau), "change_day": str(ts.iloc[tau].date()), "ci_lower_unit": L,
                "ci_lower_day": str(ts.iloc[L].date()),
                "daily_score": [{"day": a, "z": r(b, 3), "fails": int(c), "units": int(e)} for a, b, c, e in dd.itertuples(index=False, name=None)]})
    return out


def main(secom_dir, res_path):
    t0 = time.time()
    res = json.loads(Path(res_path).read_text())
    X, fail, ts = load_secom(Path(secom_dir))
    y = fail.to_numpy()
    day_id = ai.day_blocks(ts)
    cp = res["yield_time"]["change_point"]
    ranker, store = ranker_section(X, y, ts, day_id)
    print("ranker done", round(time.time() - t0, 1))
    regime = regime_section(X, y, ts, day_id, store)
    print("regime done", round(time.time() - t0, 1))
    anom = anomaly_section(X, y, ts, day_id, cp["unit_index"], cp["location_ci_days"][0])
    res["ai"] = {"ranker": ranker, "regime_model": regime, "anomaly": anom, "runtime_s": r(time.time() - t0, 1)}
    Path(res_path).write_text(json.dumps(res, separators=(",", ":")))
    for rw in ranker["per_cut"]:
        print({k: v for k, v in rw.items() if not k.endswith("_top")})
    for rw in ranker["pooled"]:
        print(rw)
    print(ranker["top_stability_primary"][:10])
    for rw in regime["per_cut"]:
        print(rw)
    print(regime["mean_over_cuts"])
    print(json.dumps({k: v for k, v in anom.items() if k != "daily_score"}, indent=1))
    print("runtime", res["ai"]["runtime_s"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
