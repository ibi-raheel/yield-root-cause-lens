"""Lead triage: give it a line's test log, get the date yield moved and every lead in one of four bins.

    python -I pipeline/triage.py --secom <dir> [--split 0.7] [--out triage.json] [--with-lr]
    python -I pipeline/triage.py --csv <file>  [--split 0.7] [--out triage.json]

CSV format: one row per unit, columns ts, fail (1 = fail, 0 = pass), then one numeric column per signal.
A thin wrapper over the existing pipeline: no new statistics. Bins come from yrcl.rootcause.classify_lead,
the same rule run_rootcause.py writes into public/results.json. See SOLUTION.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_pipeline import MAX_MISSING, one_sided_p, r, screen
from yrcl import rootcause as rc
from yrcl import stats as st
from yrcl.load import load_secom

MIN_SEG = 50
BINS = {
    "a_replicating": "WALK TO THE LINE",
    "b_regime_proxy": "CHECK THE CHANGE RECORDS",
    "e_unconfirmed": "CONFIRM OR DROP BY DAY N",
    "c_drifting": "RE-CHART WEEKLY",
    "d_noise": "DROP",
}
DATED = ("a_replicating", "e_unconfirmed")  # bins where "days to confirm" is a decision date


def load_csv(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return df.drop(columns=["ts", "fail"]).astype(float), df["fail"].astype(int), df["ts"]


def cliff(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return st.cliffs_delta(st.mann_whitney(a, b)[0], len(a), len(b))


def triage(X, fail, ts, split=0.7, with_lr=False):
    y = fail.to_numpy()
    n = len(y)
    tr, te = np.arange(int(round(split * n))), np.arange(int(round(split * n)), n)
    keep = X.columns[(X.iloc[tr].isna().mean() <= MAX_MISSING) & ~(X.iloc[tr].nunique(dropna=True) <= 1)]
    Xk = X[keep]

    # 1. date it
    tau, _ = st.single_change_point(y.astype(float), MIN_SEG)
    p_cp = st.change_point_permutation_p(y.astype(float), 999, MIN_SEG, seed=0)
    before, after = y[:tau], y[tau:]
    ci_b, ci_a = st.bootstrap_mean_ci(1 - before, seed=1), st.bootstrap_mean_ci(1 - after, seed=2)

    # 2. split the history at that date (plus any further permutation-significant split inside a segment)
    cuts = [tau]
    for lo, hi in [(0, tau), (tau, n)]:
        seg = y[lo:hi].astype(float)
        if len(seg) >= 2 * MIN_SEG + 1 and st.change_point_permutation_p(seg, 999, MIN_SEG, seed=lo) < 0.05:
            cuts.append(lo + st.single_change_point(seg, MIN_SEG)[0])
    regime = np.searchsorted(sorted(cuts), np.arange(n), side="right")

    # 3. screen pooled and inside each period (training units only)
    scr = screen(Xk.iloc[tr], y[tr])
    zs, ps = {}, {}
    for c in keep:
        zs[c], ps[c] = rc.stratified_mwu(Xk[c].to_numpy(float)[tr], y[tr], regime[tr])
    S = pd.DataFrame({"Z": zs, "p": ps})
    S["q"] = st.benjamini_hochberg(S["p"].to_numpy())
    if with_lr:
        S["lr_p"] = [rc.logistic_lr_p(Xk[c].to_numpy(float)[tr], y[tr], regime[tr]) for c in S.index]
        S["lr_q"] = st.benjamini_hochberg(S["lr_p"].to_numpy())

    # 4-5. re-test on later units and bin every lead
    na, nb = int(y[te].sum()), int((1 - y[te]).sum())
    mdd = rc.min_detectable_delta(na, nb)
    thr_day = n / ((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400)
    f_post = float(y[tau:].mean())
    post_tr = tr[regime[tr] == regime[tau]]
    pas = y == 0
    pooled = list(scr.index[scr["q"] < 0.05])
    within_only = [c for c in S.index[S["q"] < 0.05] if c not in pooled]
    leads, watch = [], []
    for c in pooled + within_only:
        v = Xk[c].to_numpy(float)
        dirn = np.sign(scr.loc[c, "delta"]) if c in pooled else np.sign(S.loc[c, "Z"])
        p_later = one_sided_p(v[te][y[te] == 1], v[te][y[te] == 0], dirn)
        holds = p_later < 0.05
        if c in within_only and not holds:
            watch.append({"signal": c, "q_within": r(S.loc[c, "q"], 6), "q_pooled": r(scr.loc[c, "q"], 6),
                          "p_later": r(p_later, 5)})
            continue
        d_pass_regime = cliff(v[tr][(regime[tr] > 0) & pas[tr]], v[tr][(regime[tr] == 0) & pas[tr]])
        d_pass_test = cliff(v[te][pas[te]], v[post_tr][pas[post_tr]])
        ok = ~np.isnan(v[tr])
        rho = spearmanr(np.arange(len(tr))[ok], v[tr][ok]).statistic
        drift = max(abs(d_pass_regime), abs(d_pass_test)) >= rc.DRIFT_DELTA
        bucket = rc.classify_lead(S.loc[c, "q"] < 0.05, np.sign(S.loc[c, "Z"]) == dirn, holds, drift, rho, d_pass_regime)
        d_later = cliff(v[te][y[te] == 1], v[te][y[te] == 0])
        d_plan = abs(cliff(v[post_tr][y[post_tr] == 1], v[post_tr][y[post_tr] == 0]))
        days = None if np.isnan(d_plan) or r(d_plan, 3) < 0.05 else rc.noether_total_n(r(d_plan, 3), f_post) / thr_day
        note = ""
        if bucket == "a_replicating":
            cen, lcl, ucl = rc.individuals_limits(v[post_tr])
            note = f"individuals chart, post-change limits {lcl:.4g} to {ucl:.4g}"
            if c in within_only:
                note += "; missed by the pooled screen"
        elif bucket == "b_regime_proxy" and holds and abs(d_later) >= mdd:
            note = "re-test: still separates later fails"
        elif bucket == "c_drifting":
            note = "separated fails after the change, not on later units; re-chart weekly"
        row = {"signal": c, "bin": BINS[bucket], "bucket": bucket, "q_pooled": r(scr.loc[c, "q"], 6),
               "q_within": r(S.loc[c, "q"], 6), "delta_later": r(d_later, 3), "p_later": r(p_later, 5),
               "days_to_confirm": None if days is None else r(days, 1), "note": note}
        if with_lr:
            row["lr_q"] = r(S.loc[c, "lr_q"], 6)
        leads.append(row)
    order = list(BINS)
    leads.sort(key=lambda l: (order.index(l["bucket"]), l["days_to_confirm"] or 0, l["signal"]))
    return {
        "change": {"date": str(ts.iloc[tau].date()), "unit": int(tau),
                   "yield_before_pct": r(100 * (1 - before.mean()), 2), "yield_after_pct": r(100 * (1 - after.mean()), 2),
                   "yield_before_ci_pct": [r(100 * ci_b[0], 2), r(100 * ci_b[1], 2)],
                   "yield_after_ci_pct": [r(100 * ci_a[0], 2), r(100 * ci_a[1], 2)],
                   "fails_per_1000_before": r(1000 * before.mean(), 1), "fails_per_1000_after": r(1000 * after.mean(), 1),
                   "p": r(p_cp, 4)},
        "units": n, "signals": int(X.shape[1]), "signals_screened": int(len(keep)), "later_fails": na,
        "mdd": r(mdd, 3), "units_per_day": r(thr_day, 2),
        "counts": {b: sum(l["bin"] == b for l in leads) for b in BINS.values()},
        "leads": leads, "within_only_watch": watch,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--secom", help="folder with secom.data and secom_labels.data")
    src.add_argument("--csv", help="CSV with columns ts, fail, then one column per signal")
    ap.add_argument("--split", type=float, default=0.7, help="share of units (in time) used to find leads")
    ap.add_argument("--out", default="triage.json")
    ap.add_argument("--with-lr", action="store_true", help="add the logistic fixed-effects cross-check (slower)")
    a = ap.parse_args(argv)
    X, fail, ts = load_secom(Path(a.secom)) if a.secom else load_csv(a.csv)
    out = triage(X, fail, ts, a.split, a.with_lr)
    Path(a.out).write_text(json.dumps(out, indent=1))
    ch = out["change"]
    print(f"Yield moved on {ch['date']}: {ch['fails_per_1000_before']:.0f} -> {ch['fails_per_1000_after']:.0f} "
          f"failing units per 1,000 (permutation p = {ch['p']})")
    print(f"Smallest effect visible with {out['later_fails']} later fails: delta = {out['mdd']:.2f}\n")
    for b, k in out["counts"].items():
        print(f"{b:<26} {k}")
    print(f"\n{'signal':<7} {'bin':<26} {'q within':>9} {'d later':>8} {'p later':>8} {'days':>6}  note")
    for l in out["leads"]:
        days = l["days_to_confirm"]
        fmt = lambda x, f: "-" if x is None else format(x, f)
        print(f"{l['signal']:<7} {l['bin']:<26} {fmt(l['q_within'], '.3f'):>9} {fmt(l['delta_later'], '+.2f'):>8} "
              f"{fmt(l['p_later'], '.3f'):>8} {fmt(days if l['bucket'] in DATED else None, '.1f'):>6}  {l['note']}")
    if out["within_only_watch"]:
        print("\nWithin-period only, not yet on later units (watch):", ", ".join(w["signal"] for w in out["within_only_watch"]))
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
