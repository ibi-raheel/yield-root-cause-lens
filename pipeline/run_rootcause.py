"""Stage 2: what changed with the yield step, a regime-stratified re-screen, and a power /
economics / monitoring plan. Reads the SECOM files and the stage-1 results, adds three sections.

Usage:  python -I pipeline/run_rootcause.py <secom_dir> <results_json>
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import fisher_exact, spearmanr

from run_pipeline import MAX_MISSING, TRAIN_FRAC, one_sided_p, r, screen
from yrcl import rootcause as rc
from yrcl import stats as st
from yrcl.load import load_secom

SEED = 0
MIN_SEG = 50
WIDEN = 50            # units added on each side of the yield change-point location CI
LARGE_DELTA = 0.474   # Romano et al. (2006) "large" Cliff's delta
DRIFT_DELTA = rc.DRIFT_DELTA  # Romano "medium": a level move between regimes worth calling drift
N_PERM_ALIGN = 999
VALUE_WINDOW_DAYS = 90
COSTS = [10, 25, 50, 100, 250, 500]  # $ per failed unit, an explicit assumption (swept)
REF_COST = 100


def cliff(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    U, _ = st.mann_whitney(a, b)
    return st.cliffs_delta(U, len(a), len(b))


def main(secom_dir, res_path):
    t0 = time.time()
    res = json.loads(Path(res_path).read_text())
    X, fail, ts = load_secom(Path(secom_dir))
    y = fail.to_numpy()
    n = len(y)
    n_split = int(round(TRAIN_FRAC * n))
    tr, te = np.arange(n_split), np.arange(n_split, n)
    miss_tr = X.iloc[tr].isna().mean()
    keep = X.columns[(miss_tr <= MAX_MISSING) & ~(X.iloc[tr].nunique(dropna=True) <= 1)]
    Xk = X[keep]
    cp = res["yield_time"]["change_point"]
    tau = cp["unit_index"]
    day = ts.dt.normalize()
    d_lo, d_hi = pd.Timestamp(cp["location_ci_days"][0]), pd.Timestamp(cp["location_ci_days"][1])
    win_lo = max(0, int(np.where(day >= d_lo)[0][0]) - WIDEN)
    win_hi = min(n - 1, int(np.where(day <= d_hi)[0][-1]) + WIDEN)
    days_from_tau = lambda u: round((ts.iloc[u] - ts.iloc[tau]).total_seconds() / 86400, 1)

    # ================= 1. change log: what stepped with yield =================
    rows = []
    for c in keep:
        v = Xk[c].to_numpy(float)
        s = rc.rank_step(v, MIN_SEG)
        if s is None:
            continue
        u = s["unit"]
        pre, post = v[:u], v[u:]
        pre, post = pre[~np.isnan(pre)], post[~np.isnan(post)]
        p = sps.mannwhitneyu(pre, post, method="asymptotic").pvalue
        rows.append(dict(feature=c, unit=u, delta=s["delta"], p=p, med_pre=np.median(pre), med_post=np.median(post)))
    cl = pd.DataFrame(rows).set_index("feature")
    cl["q"] = st.benjamini_hochberg(cl["p"].to_numpy())
    cl["in_window"] = (cl["unit"] >= win_lo) & (cl["unit"] <= win_hi)
    cl["aligned"] = cl["in_window"] & (cl["delta"].abs() >= LARGE_DELTA) & (cl["q"] < 0.05)
    series = [Xk[c].to_numpy(float) for c in cl.index]
    obs, null, p_align = rc.alignment_permutation_p(series, win_lo, win_hi, LARGE_DELTA, N_PERM_ALIGN, MIN_SEG, SEED)
    # secondary, post-hoc: a narrow window of +/-3 days around the yield step
    nar = np.where(np.abs((ts - ts.iloc[tau]).dt.total_seconds() / 86400) <= 3)[0]
    nar_lo, nar_hi = int(nar[0]), int(nar[-1])
    obs_n, null_n, p_n = rc.alignment_permutation_p(series, nar_lo, nar_hi, LARGE_DELTA, N_PERM_ALIGN, MIN_SEG, SEED + 1)
    n_medium = int((cl["in_window"] & (cl["delta"].abs() >= 0.33) & (cl["q"] < 0.05)).sum())
    aligned = cl[cl["aligned"]].reindex(cl[cl["aligned"]]["delta"].abs().sort_values(ascending=False).index)
    log_rows = []
    for i, (c, rw) in enumerate(aligned.iterrows()):
        ci = rc.bootstrap_step_location(Xk[c].to_numpy(float), rw["unit"], 200, MIN_SEG, seed=100 + i) if i < 12 else None
        log_rows.append({
            "feature": c, "kind": "level", "step_unit": int(rw["unit"]), "step_day": str(ts.iloc[int(rw["unit"])].date()),
            "lag_units": int(rw["unit"] - tau), "lag_days": days_from_tau(int(rw["unit"])),
            "step_ci_days": None if ci is None else [str(ts.iloc[ci[0]].date()), str(ts.iloc[ci[1]].date())],
            "median_pre": r(rw["med_pre"], 4), "median_post": r(rw["med_post"], 4), "delta": r(rw["delta"], 3),
            "q": r(rw["q"], 8),
        })
    # missingness / logging changes: group columns with an identical missing pattern
    fam = {}
    for c in X.columns:
        m = X[c].isna().to_numpy()
        if 0.01 < m.mean() < 0.99 and X[c].nunique(dropna=True) > 1:
            fam.setdefault(m.tobytes(), []).append(c)
    mrows = []
    for key, cols in fam.items():
        m = X[cols[0]].isna().to_numpy()
        s = rc.presence_step(m, MIN_SEG)
        u = s["unit"]
        pres = 1 - m
        tab = [[int(pres[:u].sum()), int(u - pres[:u].sum())], [int(pres[u:].sum()), int(n - u - pres[u:].sum())]]
        tab_y = [[int(pres[:tau].sum()), int(tau - pres[:tau].sum())], [int(pres[tau:].sum()), int(n - tau - pres[tau:].sum())]]
        mrows.append(dict(family=cols, unit=u, pre=s["pre"], post=s["post"], p=fisher_exact(tab)[1],
                          pre_y=pres[:tau].mean(), post_y=pres[tau:].mean(), p_y=fisher_exact(tab_y)[1]))
    mf = pd.DataFrame(mrows)
    mf["q"] = st.benjamini_hochberg(mf["p"].to_numpy())
    mf["q_y"] = st.benjamini_hochberg(mf["p_y"].to_numpy())
    mf["in_window"] = (mf["unit"] >= win_lo) & (mf["unit"] <= win_hi)
    mf["aligned"] = mf["in_window"] & (mf["q"] < 0.05) & ((mf["pre"] - mf["post"]).abs() >= 0.10)
    mfa = mf.assign(sz=lambda d: (d["pre_y"] - d["post_y"]).abs()).sort_values("sz", ascending=False)
    fam_rows = [{"feature": "/".join(rw["family"][:4]) + (f" +{len(rw['family']) - 4}" if len(rw["family"]) > 4 else ""),
                 "columns": len(rw["family"]), "kind": "logging", "step_unit": int(rw["unit"]),
                 "step_day": str(ts.iloc[int(rw["unit"])].date()), "lag_units": int(rw["unit"] - tau),
                 "lag_days": days_from_tau(int(rw["unit"])),
                 "own_step_in_window": bool(rw["in_window"]),
                 "present_pre_own_step_pct": r(100 * rw["pre"], 1), "present_post_own_step_pct": r(100 * rw["post"], 1),
                 "present_before_yield_step_pct": r(100 * rw["pre_y"], 1), "present_after_yield_step_pct": r(100 * rw["post_y"], 1),
                 "q_at_yield_step": r(rw["q_y"], 8)}
                for _, rw in mfa.iterrows()]
    res["change_log"] = {
        "yield_step_unit": tau, "yield_step_day": cp["day"],
        "window_units": [win_lo, win_hi], "window_days": [str(ts.iloc[win_lo].date()), str(ts.iloc[win_hi].date())],
        "signals_scanned": int(len(cl)), "signals_with_step_q_lt_05": int((cl["q"] < 0.05).sum()),
        "large_delta": LARGE_DELTA,
        "aligned_large": int(obs), "aligned_medium": n_medium,
        "null_mean": r(null.mean(), 2), "null_95pct": r(np.quantile(null, 0.95), 1), "alignment_p": r(p_align, 4),
        "n_perm": N_PERM_ALIGN,
        "level_steps": log_rows,
        "narrow_window_days": 3, "narrow_window_units": [nar_lo, nar_hi], "aligned_narrow": int(obs_n),
        "null_mean_narrow": r(null_n.mean(), 2), "alignment_p_narrow": r(p_n, 4),
        "families_scanned": int(len(mf)), "families_aligned": int(mf["aligned"].sum()),
        "families_shift_at_yield_q_lt_05": int((mf["q_y"] < 0.05).sum()), "logging_steps": fam_rows,
    }

    # ================= 2. regime-stratified screen =================
    # regimes: yield change point, then one more split inside each segment if permutation-significant
    cuts = [tau]
    seg_tests = []
    for lo_, hi_ in [(0, tau), (tau, n)]:
        seg = y[lo_:hi_].astype(float)
        if len(seg) >= 2 * MIN_SEG + 1:
            k, _ = st.single_change_point(seg, MIN_SEG)
            p = st.change_point_permutation_p(seg, 999, MIN_SEG, seed=SEED + lo_)
            seg_tests.append({"segment": [lo_, hi_], "split_unit": lo_ + k, "day": str(ts.iloc[lo_ + k].date()),
                              "fail_pct_left": r(100 * seg[:k].mean(), 2), "fail_pct_right": r(100 * seg[k:].mean(), 2),
                              "permutation_p": r(p, 4)})
            if p < 0.05:
                cuts.append(lo_ + k)
    cuts = sorted(cuts)
    regime = np.searchsorted(cuts, np.arange(n), side="right")
    scr = screen(Xk.iloc[tr], y[tr])
    leads = list(scr.index[scr["q"] < 0.05])
    strat_z, strat_p, lr_p = {}, {}, {}
    for c in keep:
        v = Xk[c].to_numpy(float)[tr]
        strat_z[c], strat_p[c] = rc.stratified_mwu(v, y[tr], regime[tr])
        lr_p[c] = rc.logistic_lr_p(v, y[tr], regime[tr]) if np.nanstd(v) > 0 else np.nan
    S = pd.DataFrame({"Z": strat_z, "p": strat_p, "lr_p": lr_p})
    S["q"] = st.benjamini_hochberg(S["p"].to_numpy())
    S["lr_q"] = st.benjamini_hochberg(S["lr_p"].to_numpy())
    S["p_test"] = [one_sided_p(Xk[c].to_numpy(float)[te][y[te] == 1], Xk[c].to_numpy(float)[te][y[te] == 0],
                               np.sign(S.loc[c, "Z"])) if not np.isnan(S.loc[c, "Z"]) else np.nan for c in S.index]
    strat_leads = list(S.index[S["q"] < 0.05])
    post_tr = tr[regime[tr] == regime[tau]]  # train units in the post-change regime
    lead_rows = []
    for c in leads:
        v = Xk[c].to_numpy(float)
        s0 = scr.loc[c]
        dirn = np.sign(s0["delta"])
        p_test = one_sided_p(v[te][y[te] == 1], v[te][y[te] == 0], dirn)
        d_test = cliff(v[te][y[te] == 1], v[te][y[te] == 0])
        pas = y == 0
        d_pass_regime = cliff(v[tr][(regime[tr] > 0) & pas[tr]], v[tr][(regime[tr] == 0) & pas[tr]])
        d_pass_test = cliff(v[te][pas[te]], v[post_tr][pas[post_tr]])
        d_post = cliff(v[post_tr][y[post_tr] == 1], v[post_tr][y[post_tr] == 0])
        ok = ~np.isnan(v[tr])
        rho = spearmanr(np.arange(len(tr))[ok], v[tr][ok]).statistic
        holds = p_test < 0.05
        drift = max(abs(d_pass_regime), abs(d_pass_test)) >= DRIFT_DELTA
        bucket = rc.classify_lead(S.loc[c, "q"] < 0.05, np.sign(S.loc[c, "Z"]) == dirn, holds, drift, rho, d_pass_regime)
        thr = {}
        for name, idx in [("pre_change_train", tr[regime[tr] == 0]), ("post_change_train", post_tr), ("test", te)]:
            vv, yy = v[idx], y[idx]
            okk = ~np.isnan(vv)
            thr[name] = {"youden_threshold": r(rc.youden_threshold(vv, yy), 4),
                         "median_fail": r(np.nanmedian(vv[yy == 1]) if (yy[okk] == 1).any() else np.nan, 4),
                         "median_pass": r(np.nanmedian(vv[yy == 0]), 4)}
        lead_rows.append({
            "feature": c, "bucket": bucket, "delta_pooled": r(s0["delta"], 3), "q_pooled": r(s0["q"], 6),
            "Z_strat": r(S.loc[c, "Z"], 2), "q_strat": r(S.loc[c, "q"], 6), "lr_q": r(S.loc[c, "lr_q"], 6),
            "delta_post_change_train": r(d_post, 3), "delta_test": r(d_test, 3), "p_test_1s": r(p_test, 5),
            "rho_time": r(rho, 3), "pass_shift_between_regimes": r(d_pass_regime, 3),
            "pass_shift_train_to_test": r(d_pass_test, 3), "thresholds": thr,
        })
    # simulation: pooled vs stratified false-positive rate under a pure time confound
    fp_pool = fp_strat = fp_lr = 0
    N_SIM = 500
    for i in range(N_SIM):
        v, yy, ss = rc.simulate_confounded(seed=1000 + i)
        fp_pool += st.mann_whitney(v[yy == 1], v[yy == 0])[1] < 0.05
        fp_strat += rc.stratified_mwu(v, yy, ss)[1] < 0.05
        fp_lr += rc.logistic_lr_p(v, yy, ss) < 0.05
    counts = pd.Series([rw["bucket"] for rw in lead_rows]).value_counts().to_dict()
    res["regime_screen"] = {
        "cuts": [{"unit": int(u), "day": str(ts.iloc[u].date())} for u in cuts],
        "segment_tests": seg_tests,
        "regime_fail_pct": [r(100 * y[regime == g].mean(), 2) for g in range(len(cuts) + 1)],
        "regime_units": [int((regime == g).sum()) for g in range(len(cuts) + 1)],
        "pooled_fdr_leads": len(leads),
        "strat_fdr_leads": len(strat_leads),
        "strat_fdr_leads_list": strat_leads,
        "strat_leads_detail": [{"feature": c, "Z_strat": r(S.loc[c, "Z"], 2), "q_strat": r(S.loc[c, "q"], 6),
                                "q_pooled": r(scr.loc[c, "q"], 6), "p_test_1s": r(S.loc[c, "p_test"], 5),
                                "delta_test": r(cliff(Xk[c].to_numpy(float)[te][y[te] == 1], Xk[c].to_numpy(float)[te][y[te] == 0]), 3),
                                "delta_post_change_train": r(cliff(Xk[c].to_numpy(float)[post_tr][y[post_tr] == 1],
                                                                   Xk[c].to_numpy(float)[post_tr][y[post_tr] == 0]), 3)}
                               for c in strat_leads],
        "strat_fdr_and_holdout": int(sum(S.loc[c, "p_test"] < 0.05 for c in strat_leads)),
        "strat_fdr_and_holdout_list": [c for c in strat_leads if S.loc[c, "p_test"] < 0.05],
        "lr_fdr_leads": int((S["lr_q"] < 0.05).sum()),
        "strat_and_lr_agree": int(((S["q"] < 0.05) == (S["lr_q"] < 0.05)).sum()), "signals": int(len(S)),
        "bucket_counts": counts, "leads": lead_rows,
        "simulation": {"runs": N_SIM, "fp_pooled": r(fp_pool / N_SIM, 3), "fp_stratified": r(fp_strat / N_SIM, 3),
                       "fp_logistic_fe": r(fp_lr / N_SIM, 3)},
    }

    # ================= 3. power, economics, monitoring plan =================
    na, nb = int(y[te].sum()), int((1 - y[te]).sum())
    mdd = rc.min_detectable_delta(na, nb)
    span_days = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
    thr_day = n / span_days
    f_post = float(y[tau:].mean())
    # Monte Carlo check of the formula on a real lead's pass distribution
    mc_feat = (res["regime_screen"]["strat_fdr_and_holdout_list"] or strat_leads or leads)[0]
    pv = Xk[mc_feat].to_numpy(float)[te][y[te] == 0]
    mc = []
    for k_, dlt in enumerate([0.0, 0.2, 0.3, 0.4]):
        emp = rc.mc_power(pv, dlt, na, nb, n_sim=1000, seed=50 + k_)
        se0 = rc.se_delta_null(na, nb)
        form = 0.05 if dlt == 0 else float(sps.norm.cdf(dlt / se0 - sps.norm.ppf(0.95)))
        mc.append({"delta": dlt, "power_formula": r(form, 3), "power_mc": r(emp, 3)})
    curve = []
    for dlt in np.arange(0.15, 0.651, 0.05):
        N = rc.noether_total_n(dlt, f_post)
        curve.append({"delta": r(dlt, 2), "units": int(np.ceil(N)), "fails": int(np.ceil(N * f_post)),
                      "days": r(N / thr_day, 1)})
    plan_leads = []
    for rw in lead_rows:
        d_plan = abs(rw["delta_post_change_train"]) if rw["delta_post_change_train"] is not None else np.nan
        if not d_plan or np.isnan(d_plan) or d_plan < 0.05:
            days = None
        else:
            days = rc.noether_total_n(d_plan, f_post) / thr_day
        plan_leads.append({"feature": rw["feature"], "bucket": rw["bucket"], "planning_delta": r(d_plan, 3),
                           "days_to_confirm": None if days is None else r(days, 1),
                           "confirmable_in_window": days is not None and days <= VALUE_WINDOW_DAYS,
                           "replicated_with_26_fails": rw["p_test_1s"] is not None and rw["p_test_1s"] < 0.05})
    # economics
    before, after = y[:tau], y[tau:]
    rng = np.random.default_rng(21)
    bb = np.array([rng.choice(before, len(before)).mean() for _ in range(2000)]) * 1000
    aa = np.array([rng.choice(after, len(after)).mean() for _ in range(2000)]) * 1000
    dd = bb - aa
    per_k = {"before": r(1000 * before.mean(), 1), "after": r(1000 * after.mean(), 1),
             "before_ci": [r(np.quantile(bb, .025), 1), r(np.quantile(bb, .975), 1)],
             "after_ci": [r(np.quantile(aa, .025), 1), r(np.quantile(aa, .975), 1)],
             "avoided": r(dd.mean(), 1), "avoided_ci": [r(np.quantile(dd, .025), 1), r(np.quantile(dd, .975), 1)]}
    units_year = thr_day * 365
    avoid = (before.mean() - after.mean())
    sens = [{"cost_per_fail": c_, "per_1000_units": r(avoid * 1000 * c_, 0), "per_year": r(avoid * units_year * c_, 0)}
            for c_ in COSTS]
    du = pd.Series(1, index=day).groupby(level=0).size()
    t_lo, t_hi = float(du.quantile(0.25)), float(du.quantile(0.75))
    base = avoid * units_year * REF_COST
    tornado = [
        {"factor": "fail-rate shift (95% CI)", "low": r(np.quantile(dd, .025) / 1000 * units_year * REF_COST, 0),
         "high": r(np.quantile(dd, .975) / 1000 * units_year * REF_COST, 0)},
        {"factor": f"throughput (daily IQR {t_lo:.0f}-{t_hi:.0f} units)", "low": r(avoid * t_lo * 365 * REF_COST, 0),
         "high": r(avoid * t_hi * 365 * REF_COST, 0)},
        {"factor": "cost per failed unit ($25-$250)", "low": r(avoid * units_year * 25, 0), "high": r(avoid * units_year * 250, 0)},
    ]
    # Phase II CUSUM: limits fixed from the first 200 post-change units, then run forward
    ref = y[tau:tau + 200]
    p0 = ref.mean()
    B = 50
    rest = y[tau + 200:]
    nb2 = len(rest) // B
    bx = rest[: nb2 * B].reshape(nb2, B).mean(axis=1)
    sig = np.sqrt(p0 * (1 - p0) / B)
    Su, au = st.cusum_upper(bx, p0, sig, 0.5, 5.0)
    Sl, al = st.cusum_lower(bx, p0, sig, 0.5, 5.0)
    batches = [{"start": str(ts.iloc[tau + 200 + i * B].date()), "fail_pct": r(100 * bx[i], 1), "S_hi": r(Su[i], 2),
                "S_lo": r(Sl[i], 2)} for i in range(nb2)]
    # individuals charts for the monitored signals, limits from post-change train units only
    mon = list(dict.fromkeys(res["regime_screen"]["strat_fdr_and_holdout_list"]
                             + [rw["feature"] for rw in lead_rows if rw["bucket"] in ("a_replicating", "c_drifting")]))[:3]
    spc = []
    for c in mon:
        v = Xk[c].to_numpy(float)
        cen, lcl, ucl = rc.individuals_limits(v[post_tr])
        vt, yt = v[te], y[te]
        ok = ~np.isnan(vt)
        out_ = ok & ((vt < lcl) | (vt > ucl))
        inn = ok & ~out_
        spc.append({"feature": c, "centre": r(cen, 4), "lcl": r(lcl, 4), "ucl": r(ucl, 4),
                    "test_units_out": int(out_.sum()), "test_units_in": int(inn.sum()),
                    "fail_pct_out": r(100 * yt[out_].mean(), 1) if out_.any() else None,
                    "fail_pct_in": r(100 * yt[inn].mean(), 1) if inn.any() else None})
    res["power_plan"] = {
        "test_fails": na, "test_passes": nb, "se_delta_null": r(rc.se_delta_null(na, nb), 4),
        "min_detectable_delta_80": r(mdd, 3), "alpha_one_sided": 0.05,
        "leads_below_mdd": int(sum(abs(rw["delta_pooled"]) < mdd for rw in lead_rows)),
        "mc_feature": mc_feat, "mc_check": mc,
        "post_change_fail_rate": r(f_post, 4), "units_per_day": r(thr_day, 2), "value_window_days": VALUE_WINDOW_DAYS,
        "curve": curve, "leads": plan_leads,
        "economics": {"fails_per_1000": per_k, "units_per_year": r(units_year, 0), "cost_assumption_note":
                      "cost per failed unit is an assumption, swept; not measured", "sensitivity": sens,
                      "ref_cost": REF_COST, "ref_per_year": r(base, 0), "tornado": tornado},
        "phase2_cusum": {"ref_units": 200, "ref_fail_pct": r(100 * p0, 2), "batch": B, "k": 0.5, "h": 5.0,
                         "alarm_hi": None if au is None else batches[au]["start"],
                         "alarm_lo": None if al is None else batches[al]["start"], "batches": batches},
        "spc": spc,
    }
    res["runtime_rootcause_s"] = r(time.time() - t0, 1)
    Path(res_path).write_text(json.dumps(res, separators=(",", ":")))
    out = {k: v for k, v in res["change_log"].items()}
    print(json.dumps(out, indent=1)[:6000])
    rs = {k: v for k, v in res["regime_screen"].items() if k != "leads"}
    print(json.dumps(rs, indent=1))
    for rw in lead_rows:
        print({k: v for k, v in rw.items() if k != "thresholds"}, rw["thresholds"]["post_change_train"], rw["thresholds"]["test"])
    pp = {k: v for k, v in res["power_plan"].items() if k not in ("phase2_cusum",)}
    print(json.dumps(pp, indent=1)[:8000])
    print({k: v for k, v in res["power_plan"]["phase2_cusum"].items() if k != "batches"})
    print([(b["start"], b["fail_pct"], b["S_hi"]) for b in batches])
    print("runtime", res["runtime_rootcause_s"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
