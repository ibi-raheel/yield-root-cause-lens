"""End-to-end pipeline: SECOM line data -> ranked signals, change point, held-out validation.

Usage:  python -I pipeline/run_pipeline.py <secom_dir> <out_json>
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yrcl import stats as st
from yrcl.load import load_secom

SEED = 0
TRAIN_FRAC = 0.70
MAX_MISSING = 0.50
N_BOOT_STAB = 50
TOPK = 10
BATCH = 50  # units per batch for the CUSUM chart


def r(x, n=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), n)


def screen(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Mann-Whitney U, BH q-value and Cliff's delta for each column (fails vs passes)."""
    rows = []
    f = y == 1
    for c in X.columns:
        v = X[c].to_numpy(float)
        a, b = v[f], v[~f]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 3 or len(b) < 3 or np.nanstd(v) == 0:
            rows.append((c, np.nan, np.nan, np.nan, len(a), len(b)))
            continue
        U, p = st.mann_whitney(a, b)
        rows.append((c, U, p, st.cliffs_delta(U, len(a), len(b)), len(a), len(b)))
    df = pd.DataFrame(rows, columns=["feature", "U", "p", "delta", "n_fail", "n_pass"]).set_index("feature")
    df["q"] = st.benjamini_hochberg(df["p"].to_numpy())
    return df


def one_sided_p(a, b, direction):
    """One-sided MWU p in the given direction (+1: fails higher)."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    from scipy.stats import mannwhitneyu
    alt = "greater" if direction > 0 else "less"
    return float(mannwhitneyu(a, b, alternative=alt).pvalue)


def main(secom_dir, out_json):
    t0 = time.time()
    X, fail, ts = load_secom(Path(secom_dir))
    y = fail.to_numpy()
    n, m = X.shape
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds")}

    # ---------- 1. data health ----------
    miss = X.isna().mean()
    const = X.nunique(dropna=True) <= 1
    n_split = int(round(TRAIN_FRAC * n))
    tr, te = np.arange(n_split), np.arange(n_split, n)
    miss_tr = X.iloc[tr].isna().mean()
    const_tr = X.iloc[tr].nunique(dropna=True) <= 1
    keep = X.columns[(miss_tr <= MAX_MISSING) & (~const_tr)]
    out["health"] = {
        "units": n, "features_raw": m, "fails": int(y.sum()), "passes": int(n - y.sum()),
        "yield_pct": r(100 * (1 - y.mean()), 3),
        "cells_missing_pct": r(100 * X.isna().to_numpy().mean(), 2),
        "features_constant": int(const.sum()),
        "features_over_50pct_missing": int((miss > MAX_MISSING).sum()),
        "features_kept": int(len(keep)),
        "first_unit": str(ts.iloc[0]), "last_unit": str(ts.iloc[-1]),
        "train_units": int(len(tr)), "test_units": int(len(te)),
        "train_fails": int(y[tr].sum()), "test_fails": int(y[te].sum()),
        "split_time": str(ts.iloc[n_split]),
    }

    # ---------- 2. yield over time, change point, CUSUM ----------
    day = ts.dt.floor("D")
    daily = pd.DataFrame({"day": day, "fail": y}).groupby("day")["fail"].agg(["sum", "size"]).reset_index()
    tau, gain = st.single_change_point(y.astype(float), min_seg=50)
    p_cp = st.change_point_permutation_p(y.astype(float), n_perm=999, min_seg=50, seed=SEED)
    before, after = y[:tau], y[tau:]
    ci_b = st.bootstrap_mean_ci(1 - before, seed=1)
    ci_a = st.bootstrap_mean_ci(1 - after, seed=2)
    # bootstrap the shift itself
    rng = np.random.default_rng(3)
    diffs = [(1 - rng.choice(after, len(after))).mean() - (1 - rng.choice(before, len(before))).mean() for _ in range(2000)]
    # where could the change be? bootstrap the location by resampling within segments
    taus = []
    for _ in range(200):
        yb = np.concatenate([rng.choice(before, len(before)), rng.choice(after, len(after))])
        taus.append(st.single_change_point(yb.astype(float), 50)[0])
    nb = n // BATCH
    bx = y[: nb * BATCH].reshape(nb, BATCH).mean(axis=1)
    p0 = y.mean()
    sig = np.sqrt(p0 * (1 - p0) / BATCH)
    Su, au = st.cusum_upper(bx, p0, sig, k=0.5, h=5.0)
    Sl, al = st.cusum_lower(bx, p0, sig, k=0.5, h=5.0)
    out["yield_time"] = {
        "daily": [{"day": str(d.date()), "units": int(s), "fails": int(f_), "yield_pct": r(100 * (1 - f_ / s), 2)}
                  for d, f_, s in zip(daily["day"], daily["sum"], daily["size"])],
        "change_point": {
            "unit_index": tau, "time": str(ts.iloc[tau]), "day": str(ts.iloc[tau].date()),
            "yield_before_pct": r(100 * (1 - before.mean()), 2), "yield_after_pct": r(100 * (1 - after.mean()), 2),
            "ci_before_pct": [r(100 * ci_b[0], 2), r(100 * ci_b[1], 2)],
            "ci_after_pct": [r(100 * ci_a[0], 2), r(100 * ci_a[1], 2)],
            "shift_pts": r(100 * (after.mean() * -1 + before.mean()), 2),
            "shift_ci_pts": [r(100 * np.quantile(diffs, 0.025), 2), r(100 * np.quantile(diffs, 0.975), 2)],
            "location_ci_days": [str(ts.iloc[int(np.quantile(taus, 0.025))].date()), str(ts.iloc[int(np.quantile(taus, 0.975))].date())],
            "permutation_p": r(p_cp, 4), "units_before": int(tau), "units_after": int(n - tau),
        },
        "cusum": {
            "batch_units": BATCH, "mu0": r(p0, 5), "sigma": r(sig, 5), "k": 0.5, "h": 5.0,
            "batches": [{"i": i, "start": str(ts.iloc[i * BATCH].date()), "fail_rate": r(bx[i], 4),
                         "S_hi": r(Su[i], 3), "S_lo": r(Sl[i], 3)} for i in range(nb)],
            "alarm_hi_batch": au, "alarm_lo_batch": al,
            "alarm_hi_day": None if au is None else str(ts.iloc[au * BATCH].date()),
            "alarm_lo_day": None if al is None else str(ts.iloc[al * BATCH].date()),
        },
    }

    # ---------- 3. univariate screen on train, replication on test ----------
    Xk = X[keep]
    scr = screen(Xk.iloc[tr], y[tr])
    # missing-rate difference (fail minus pass) on train
    isna_tr = Xk.iloc[tr].isna()
    scr["miss_fail"] = isna_tr[y[tr] == 1].mean()
    scr["miss_pass"] = isna_tr[y[tr] == 0].mean()
    # point-biserial correlation on all units (the naive "top correlated" list)
    Xm = Xk.fillna(Xk.median())
    corr = Xm.apply(lambda c: np.corrcoef(c, y)[0, 1] if c.std() > 0 else np.nan)
    scr["corr_all"] = corr
    # held-out replication
    rep_p, rep_d = [], []
    for c in scr.index:
        v = Xk[c].to_numpy(float)[te]
        a, b = v[y[te] == 1], v[y[te] == 0]
        d = scr.loc[c, "delta"]
        if np.isnan(d):
            rep_p.append(np.nan); rep_d.append(np.nan); continue
        rep_p.append(one_sided_p(a, b, np.sign(d)))
        aa, bb = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(aa) and len(bb):
            U, _ = st.mann_whitney(aa, bb)
            rep_d.append(st.cliffs_delta(U, len(aa), len(bb)))
        else:
            rep_d.append(np.nan)
    # time confounding: Spearman of signal vs unit order (train), and the screen re-run
    # only on train units after the change point (is the effect just "early vs late"?)
    from scipy.stats import spearmanr
    rho = []
    for c in scr.index:
        v = Xk[c].to_numpy(float)[tr]
        ok = ~np.isnan(v)
        rho.append(spearmanr(np.arange(len(v))[ok], v[ok]).statistic if ok.sum() > 10 and np.nanstd(v) > 0 else np.nan)
    scr["rho_time"] = rho
    post = tr[tr >= tau]
    pc = []
    for c in scr.index:
        v = Xk[c].to_numpy(float)[post]
        a, b = v[y[post] == 1], v[y[post] == 0]
        d = scr.loc[c, "delta"]
        pc.append(np.nan if np.isnan(d) else one_sided_p(a, b, np.sign(d)))
    scr["p_post_cp_1s"] = pc
    scr["p_test_1s"] = rep_p
    scr["delta_test"] = rep_d
    scr["fdr_pass"] = scr["q"] < 0.05
    scr["holdout_pass"] = scr["p_test_1s"] < 0.05
    scr = scr.sort_values("p")
    n_fdr = int(scr["fdr_pass"].sum())
    n_raw = int((scr["p"] < 0.05).sum())

    # ---------- 4. missingness as its own signal ----------
    miss_rows = []
    for c in Xk.columns:
        ind_tr = Xk[c].iloc[tr].isna().to_numpy()
        if ind_tr.mean() == 0 or ind_tr.mean() == 1:
            continue
        from scipy.stats import fisher_exact
        tab = [[int((ind_tr & (y[tr] == 1)).sum()), int((ind_tr & (y[tr] == 0)).sum())],
               [int((~ind_tr & (y[tr] == 1)).sum()), int((~ind_tr & (y[tr] == 0)).sum())]]
        miss_rows.append((c, fisher_exact(tab)[1], ind_tr[y[tr] == 1].mean(), ind_tr[y[tr] == 0].mean()))
    md = pd.DataFrame(miss_rows, columns=["feature", "p", "miss_fail", "miss_pass"]).set_index("feature")
    md["q"] = st.benjamini_hochberg(md["p"].to_numpy()) if len(md) else []
    md = md.sort_values("p")

    # ---------- 5. models on time-ordered split ----------
    ind_cols = [c for c in keep if 0.05 <= miss_tr[c] <= MAX_MISSING]
    def design(idx):
        D = Xk.iloc[idx].copy()
        for c in ind_cols:
            D[f"{c}_missing"] = X[c].iloc[idx].isna().astype(float).to_numpy()
        return D
    Dtr, Dte = design(tr), design(te)
    ytr, yte = y[tr], y[te]
    imp = SimpleImputer(strategy="median").fit(Dtr)
    Atr, Ate = imp.transform(Dtr), imp.transform(Dte)
    cols = list(Dtr.columns)

    # L1 logistic, C chosen by forward-chaining CV on train only (PR-AUC)
    tss = TimeSeriesSplit(n_splits=3)
    cv = {}
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        sc = []
        for a_, b_ in tss.split(Atr):
            if ytr[b_].sum() == 0 or ytr[a_].sum() == 0:
                continue
            mdl = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                                                    class_weight="balanced", max_iter=2000))
            mdl.fit(Atr[a_], ytr[a_])
            sc.append(average_precision_score(ytr[b_], mdl.predict_proba(Atr[b_])[:, 1]))
        cv[C] = float(np.mean(sc))
    C_best = max(cv, key=cv.get)
    lr = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", C=C_best, solver="liblinear",
                                                           class_weight="balanced", max_iter=2000)).fit(Atr, ytr)
    s_lr = lr.predict_proba(Ate)[:, 1]
    coef = lr[-1].coef_.ravel()
    n_nonzero = int((coef != 0).sum())

    hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3, min_samples_leaf=20,
                                         l2_regularization=1.0, class_weight="balanced", random_state=SEED).fit(Atr, ytr)
    s_hgb = hgb.predict_proba(Ate)[:, 1]

    # single best feature baseline: best train MWU p, sign from train delta
    best = scr.index[0]
    sgn = np.sign(scr.loc[best, "delta"])
    s_single = sgn * imp.transform(Dte)[:, cols.index(best)]
    s_major = np.zeros(len(yte))

    def ci_metric(fn, s, nb=1000, seed=7):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(nb):
            i = rng.integers(0, len(yte), len(yte))
            if yte[i].sum() == 0 or yte[i].sum() == len(i):
                continue
            vals.append(fn(yte[i], s[i]))
        return [r(np.quantile(vals, 0.025), 3), r(np.quantile(vals, 0.975), 3)]

    def prec_at(s, k):
        o = np.argsort(-s, kind="stable")[:k]
        return float(yte[o].mean())

    models = {}
    for name, s in [("L1 logistic", s_lr), ("Gradient boosting", s_hgb), (f"Single best feature ({best})", s_single),
                    ("Majority class", s_major)]:
        fpr, tpr, _ = roc_curve(yte, s)
        pr, rc, _ = precision_recall_curve(yte, s)
        sel = np.unique(np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int))
        selp = np.unique(np.linspace(0, len(pr) - 1, min(60, len(pr))).astype(int))
        k_ = int(yte.sum())
        models[name] = {
            "roc_auc": r(roc_auc_score(yte, s), 3), "roc_auc_ci": ci_metric(roc_auc_score, s),
            "pr_auc": r(average_precision_score(yte, s), 3), "pr_auc_ci": ci_metric(average_precision_score, s),
            "precision_at_k": {str(k): r(prec_at(s, k), 3) for k in [10, 25, 50]},
            "recall_at_top10pct": r(yte[np.argsort(-s, kind="stable")[: int(0.1 * len(yte))]].sum() / k_, 3),
            "roc": [[r(fpr[i], 3), r(tpr[i], 3)] for i in sel],
            "pr": [[r(rc[i], 3), r(pr[i], 3)] for i in selp],
        }
    out["models"] = {
        "test_fail_rate": r(yte.mean(), 4), "pr_auc_baseline": r(yte.mean(), 4),
        "l1_C": C_best, "l1_cv_pr_auc": {str(k): r(v, 4) for k, v in cv.items()}, "l1_nonzero": n_nonzero,
        "design_columns": len(cols), "missing_indicator_columns": len(ind_cols), "results": models,
    }

    # the optimistic comparison: shuffled stratified 5-fold CV ignores time order
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    Dall = design(np.arange(n))
    shuf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(penalty="l1", C=C_best, solver="liblinear", class_weight="balanced", max_iter=2000))
    s_cv = cross_val_predict(shuf, Dall.to_numpy(), y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             method="predict_proba", n_jobs=2)[:, 1]
    hshuf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3, min_samples_leaf=20,
                                           l2_regularization=1.0, class_weight="balanced", random_state=SEED)
    s_cvh = cross_val_predict(hshuf, Dall.to_numpy(), y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              method="predict_proba", n_jobs=2)[:, 1]
    out["models"]["shuffled_cv"] = {
        "l1_roc_auc": r(roc_auc_score(y, s_cv), 3), "l1_pr_auc": r(average_precision_score(y, s_cv), 3),
        "hgb_roc_auc": r(roc_auc_score(y, s_cvh), 3), "hgb_pr_auc": r(average_precision_score(y, s_cvh), 3),
        "fail_rate": r(y.mean(), 4),
    }

    # permutation importance on held-out split
    pi_lr = permutation_importance(lr, Ate, yte, scoring="average_precision", n_repeats=10, random_state=SEED, n_jobs=2)
    pi_hgb = permutation_importance(hgb, Ate, yte, scoring="average_precision", n_repeats=10, random_state=SEED, n_jobs=2)
    pim = pd.DataFrame({"pi_lr": pi_lr.importances_mean, "pi_hgb": pi_hgb.importances_mean,
                        "pi_lr_sd": pi_lr.importances_std, "pi_hgb_sd": pi_hgb.importances_std}, index=cols)

    # ---------- 6. stability: bootstrap the train screen ----------
    rng = np.random.default_rng(11)
    counts = pd.Series(0, index=scr.index)
    Xtr_np = Xk.iloc[tr]
    for b in range(N_BOOT_STAB):
        i = rng.integers(0, len(tr), len(tr))
        sb = screen(Xtr_np.iloc[i].reset_index(drop=True), ytr[i])
        top = sb["p"].nsmallest(TOPK).index
        counts[top] += 1
    scr["stability"] = counts / N_BOOT_STAB

    # ---------- 7. assemble the ranking table ----------
    def row(c):
        s_ = scr.loc[c]
        return {
            "feature": c, "p": r(s_["p"], 8), "q": r(s_["q"], 6), "delta": r(s_["delta"], 3),
            "delta_test": r(s_["delta_test"], 3), "p_test_1s": r(s_["p_test_1s"], 5),
            "miss_fail_pct": r(100 * s_["miss_fail"], 1), "miss_pass_pct": r(100 * s_["miss_pass"], 1),
            "corr_all": r(s_["corr_all"], 3), "stability": r(s_["stability"], 2),
            "pi_lr": r(pim.loc[c, "pi_lr"], 4), "pi_hgb": r(pim.loc[c, "pi_hgb"], 4),
            "rho_time": r(s_["rho_time"], 3), "p_post_cp_1s": r(s_["p_post_cp_1s"], 5),
            "fdr_pass": bool(s_["fdr_pass"]), "holdout_pass": bool(s_["holdout_pass"]),
        }
    top20 = list(scr.index[:20])
    top_corr = list(scr["corr_all"].abs().sort_values(ascending=False).index[:10])
    out["signals"] = {
        "tested": int(scr["p"].notna().sum()), "raw_p_lt_05": n_raw, "fdr_q_lt_05": n_fdr,
        "expected_false_at_raw_05": r(0.05 * scr["p"].notna().sum(), 1),
        "fdr_and_holdout": int((scr["fdr_pass"] & scr["holdout_pass"]).sum()),
        "top20": [row(c) for c in top20],
        "top_correlated": [row(c) for c in top_corr],
        "top_correlated_fail_count": int(sum(not (scr.loc[c, "fdr_pass"] and scr.loc[c, "holdout_pass"]) for c in top_corr)),
        "missingness": [{"feature": c, "p": r(md.loc[c, "p"], 6), "q": r(md.loc[c, "q"], 5),
                         "miss_fail_pct": r(100 * md.loc[c, "miss_fail"], 1),
                         "miss_pass_pct": r(100 * md.loc[c, "miss_pass"], 1)} for c in md.index[:10]],
        "missingness_q_lt_05": int((md["q"] < 0.05).sum()) if len(md) else 0,
        "perm_importance_top": [{"column": c, "pi_hgb": r(pim.loc[c, "pi_hgb"], 4), "sd": r(pim.loc[c, "pi_hgb_sd"], 4),
                                 "pi_lr": r(pim.loc[c, "pi_lr"], 4)}
                                for c in pim["pi_hgb"].sort_values(ascending=False).index[:15]],
    }
    # per-signal detail for the drill-down: values over time + histograms (top 20 + top correlated)
    detail = {}
    for c in dict.fromkeys(top20 + top_corr):
        v = X[c].to_numpy(float)
        ok = ~np.isnan(v)
        lo, hi = np.nanquantile(v, [0.005, 0.995])
        edges = np.linspace(lo, hi if hi > lo else lo + 1, 31)
        hp, _ = np.histogram(np.clip(v[ok & (y == 0)], lo, hi), edges)
        hf, _ = np.histogram(np.clip(v[ok & (y == 1)], lo, hi), edges)
        detail[c] = {
            "edges": [r(e, 5) for e in edges], "hist_pass": hp.tolist(), "hist_fail": hf.tolist(),
            "series": [[int(i), r(v[i], 5), int(y[i])] for i in np.where(ok)[0]],
        }
    out["signal_detail"] = detail
    out["signal_detail_time"] = [str(t.date()) for t in ts]
    out["runtime_s"] = r(time.time() - t0, 1)
    Path(out_json).write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps({k: out[k] for k in ["health"]}, indent=1))
    print(json.dumps(out["yield_time"]["change_point"], indent=1))
    print({k: v for k, v in out["yield_time"]["cusum"].items() if k != "batches"})
    s = out["signals"]
    print({k: s[k] for k in ["tested", "raw_p_lt_05", "fdr_q_lt_05", "fdr_and_holdout", "top_correlated_fail_count", "missingness_q_lt_05"]})
    for rw in s["top20"][:10]:
        print(rw)
    print("top corr:")
    for rw in s["top_correlated"]:
        print(rw["feature"], rw["corr_all"], rw["q"], rw["p_test_1s"], rw["fdr_pass"], rw["holdout_pass"])
    print(s["missingness"][:5])
    for k, v in out["models"]["results"].items():
        print(k, v["roc_auc"], v["roc_auc_ci"], v["pr_auc"], v["pr_auc_ci"], v["precision_at_k"], v["recall_at_top10pct"])
    print({k: v for k, v in out["models"].items() if k != "results"})
    print(s["perm_importance_top"][:8])
    print(out["models"]["shuffled_cv"])
    print("survivors:", [(rw["feature"], rw["rho_time"], rw["p_post_cp_1s"]) for rw in s["top20"] if rw["fdr_pass"] and rw["holdout_pass"]])
    print("S059-like:", [(rw["feature"], rw["rho_time"], rw["p_post_cp_1s"]) for rw in s["top20"][:8]])
    print("fdr signals with |rho_time|>0.3:", sum(1 for rw in s["top20"] if rw["fdr_pass"] and rw["rho_time"] is not None and abs(rw["rho_time"])>0.3))
    print("runtime", out["runtime_s"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
