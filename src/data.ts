// Types for the precomputed results.json and the few pure helpers the page needs.
// Every number on the page is read from results.json; the helpers below only
// re-run rank statistics on the per-unit series that the pipeline already ships.

export type Daily = { day: string; units: number; fails: number; yield_pct: number }
export type SignalDetail = { edges: number[]; hist_pass: number[]; hist_fail: number[]; series: [number, number, number][] }
export type Lead = {
  feature: string; bucket: string; delta_pooled: number; q_pooled: number; Z_strat: number; q_strat: number
  delta_post_change_train: number; delta_test: number; p_test_1s: number; rho_time: number
}
export type StratLead = {
  feature: string; Z_strat: number; q_strat: number; q_pooled: number; p_test_1s: number; delta_test: number
  delta_post_change_train: number
}
export type PowerLead = { feature: string; bucket: string; planning_delta: number; days_to_confirm: number | null }
export type RegimeCut = {
  cut_frac: number; cut_day: string; test_units: number; test_fails: number; auc_regime: number; auc_pooled: number
  auc_single: number; single_feature: string; diff_vs_pooled: number; diff_vs_pooled_ci: number[]
}
export type ModelResult = { roc_auc: number; roc_auc_ci: number[]; pr_auc: number; pr_auc_ci: number[] }

export type Results = {
  health: {
    units: number; features_raw: number; fails: number; passes: number; yield_pct: number; features_kept: number
    first_unit: string; last_unit: string; train_units: number; test_units: number; train_fails: number; test_fails: number
    features_constant: number; features_over_50pct_missing: number; split_time: string
  }
  yield_time: {
    daily: Daily[]
    change_point: {
      unit_index: number; day: string; yield_before_pct: number; yield_after_pct: number; ci_before_pct: number[]
      ci_after_pct: number[]; shift_pts: number; shift_ci_pts: number[]; location_ci_days: string[]; permutation_p: number
      units_before: number; units_after: number
    }
    cusum: { alarm_hi_day: string; alarm_lo_day: string; batch_units: number; k: number; h: number }
  }
  models: {
    test_fail_rate: number; results: Record<string, ModelResult>
    shuffled_cv: { l1_roc_auc: number; hgb_roc_auc: number }
  }
  signals: {
    tested: number; raw_p_lt_05: number; fdr_q_lt_05: number; expected_false_at_raw_05: number; fdr_and_holdout: number
    top_correlated_fail_count: number
  }
  signal_detail: Record<string, SignalDetail>
  signal_detail_time: string[]
  change_log: {
    narrow_window_days: number; aligned_narrow: number; null_mean_narrow: number; alignment_p_narrow: number
    families_scanned: number; families_shift_at_yield_q_lt_05: number; signals_scanned: number
    level_steps: { feature: string; step_day: string; lag_days: number; delta: number }[]
  }
  regime_screen: {
    regime_fail_pct: number[]; regime_units: number[]; pooled_fdr_leads: number; strat_fdr_leads: number
    strat_leads_detail: StratLead[]; strat_fdr_and_holdout_list: string[]; bucket_counts: Record<string, number>
    leads: Lead[]; simulation: { runs: number; fp_pooled: number; fp_stratified: number }
  }
  power_plan: {
    test_fails: number; test_passes: number; min_detectable_delta_80: number; units_per_day: number
    post_change_fail_rate: number; leads_below_mdd: number; mc_feature: string
    mc_check: { delta: number; power_formula: number; power_mc: number }[]
    curve: { delta: number; units: number; fails: number; days: number }[]
    leads: PowerLead[]
    economics: { fails_per_1000: { before: number; after: number; avoided: number; avoided_ci: number[] } }
    spc: { feature: string; centre: number; lcl: number; ucl: number; test_units_out: number; test_units_in: number }[]
  }
  ai: {
    ranker: {
      primary_k: number; n_resamples: number
      pooled: { k: number; ai_mean_delta: number; t_mean_delta: number; mwu_mean_delta: number; diff_ai_vs_t: number; diff_ai_vs_t_ci: number[] }[]
    }
    regime_model: {
      per_cut: RegimeCut[]
      mean_over_cuts: {
        auc_regime: number; auc_pooled: number; auc_single: number; diff_vs_pooled: number; diff_vs_pooled_ci: number[]
        diff_vs_single: number; diff_vs_single_ci: number[]
      }
      solver_seed_sensitivity: { diff_vs_pooled: number }[]
      fairer_pooled_check: {
        per_cut: { cut_frac: number; auc_pooled_with_regime_indicator: number; auc_single_all_history: number }[]
        auc_pooled_with_regime_indicator: number; diff_vs_pooled_with_regime_indicator: number
        diff_vs_pooled_with_regime_indicator_ci: number[]
      }
    }
    anomaly: {
      isolation_forest: { alarm_day: string; delay_units: number; delay_days: number }
      bernoulli_cusum: { alarm_day: string; false_alarm_days: string[] }
      reference_fail_rate: number
      bootstrap: { share_runs_bern_detection_before_change: number; share_if_no_detection: number }
    }
  }
}

export const BUCKET_LABEL: Record<string, string> = {
  a_replicating: 'holds within regime',
  b_regime_proxy: 'regime proxy',
  c_drifting: 'drifting separator',
  d_noise: 'noise',
  e_unconfirmed: 'within regime, unconfirmed',
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
export function fmtDay(iso: string, year = false) {
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number)
  return `${d} ${MONTHS[m - 1]}${year ? ` ${y}` : ''}`
}
export const dayNum = (iso: string) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10)) / 864e5

export const n2 = (x: number, d = 2) => x.toFixed(d)
export const sgn = (x: number, d = 2) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d)
export const ci = (c: number[], d = 2) => `${c[0] < 0 ? '−' : ''}${Math.abs(c[0]).toFixed(d)} to ${c[1] < 0 ? '−' : ''}${Math.abs(c[1]).toFixed(d)}`
export const thousands = (x: number) => x.toLocaleString('en-US')

/** Pass/fail label for every unit, from the union of the shipped per-unit series. */
export function unitLabels(r: Results): Int8Array {
  const lab = new Int8Array(r.health.units).fill(-1)
  for (const d of Object.values(r.signal_detail)) for (const [i, , y] of d.series) lab[i] = y
  return lab
}

function lowerBound(a: Float64Array, x: number) {
  let lo = 0, hi = a.length
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m }
  return lo
}
function upperBound(a: Float64Array, x: number) {
  let lo = 0, hi = a.length
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= x) lo = m + 1; else hi = m }
  return lo
}

/** Cliff's delta P(fail > pass) − P(fail < pass), exact, via sorted passes. */
export function cliff(fails: number[], passes: number[]): { d: number; pairs: number } | null {
  if (!fails.length || !passes.length) return null
  const p = Float64Array.from(passes).sort()
  let s = 0
  for (const f of fails) s += lowerBound(p, f) - (p.length - upperBound(p, f))
  const pairs = fails.length * passes.length
  return { d: s / pairs, pairs }
}

export type Prepared = { feature: string; idx: Int32Array; val: Float64Array; y: Int8Array; pooled: number }

/** Training-window series (units before the 70% time split) for one signal. */
export function prepare(feature: string, d: SignalDetail, trainUnits: number): Prepared {
  const rows = d.series.filter((s) => s[0] < trainUnits)
  const idx = Int32Array.from(rows.map((s) => s[0]))
  const val = Float64Array.from(rows.map((s) => s[1]))
  const y = Int8Array.from(rows.map((s) => s[2]))
  const f: number[] = [], p: number[] = []
  rows.forEach((s) => (s[2] ? f : p).push(s[1]))
  return { feature, idx, val, y, pooled: cliff(f, p)?.d ?? 0 }
}

/** Within-split Cliff's delta: each side separately, pooled with van Elteren-type weights n_f·n_p/(n+1). */
export function withinSplit(s: Prepared, split: number) {
  const fb: number[] = [], pb: number[] = [], fa: number[] = [], pa: number[] = []
  for (let i = 0; i < s.idx.length; i++) {
    const before = s.idx[i] < split
    if (s.y[i]) (before ? fb : fa).push(s.val[i]); else (before ? pb : pa).push(s.val[i])
  }
  let num = 0, den = 0
  const b = cliff(fb, pb), a = cliff(fa, pa)
  if (b) { const w = b.pairs / (fb.length + pb.length + 1); num += b.d * w; den += w }
  if (a) { const w = a.pairs / (fa.length + pa.length + 1); num += a.d * w; den += w }
  return { within: den ? num / den : 0, before: b?.d ?? null, after: a?.d ?? null, nfb: fb.length, nfa: fa.length }
}

/** A signal "keeps its effect" at a split when the within-split delta, signed by the pooled direction, is at least half the pooled delta. */
export const keeps = (pooled: number, within: number) => Math.sign(pooled) * within >= 0.5 * Math.abs(pooled)

/** Benjamini-Hochberg adjusted q values. */
export function bh(p: number[]) {
  const n = p.length
  const order = p.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
  const q = new Array<number>(n)
  let prev = 1
  for (let k = n - 1; k >= 0; k--) {
    const [v, i] = order[k]
    prev = Math.min(prev, (v * n) / (k + 1))
    q[i] = prev
  }
  return q
}
