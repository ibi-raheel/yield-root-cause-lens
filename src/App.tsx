import { useEffect, useState } from 'react'

type Row = {
  feature: string; p: number; q: number; delta: number; delta_test: number | null
  p_test_1s: number | null; stability: number; rho_time: number | null; fdr_pass: boolean; holdout_pass: boolean
}
type ModelResult = { roc_auc: number; roc_auc_ci: number[]; pr_auc: number; pr_auc_ci: number[] }
type Results = {
  health: Record<string, number | string>
  yield_time: { change_point: Record<string, unknown> }
  signals: { tested: number; raw_p_lt_05: number; fdr_q_lt_05: number; fdr_and_holdout: number; top20: Row[] }
  models: { test_fail_rate: number; results: Record<string, ModelResult>; shuffled_cv: Record<string, number> }
  regime_screen?: { leads: { feature: string; bucket: string }[]; strat_fdr_leads: number; strat_fdr_and_holdout_list: string[] }
}
const BUCKET: Record<string, string> = {
  a_replicating: 'within-regime, holds', b_regime_proxy: 'regime proxy', c_drifting: 'drifting separator',
  d_noise: 'noise', e_unconfirmed: 'within-regime, unconfirmed',
}

const f = (x: number | null | undefined, d = 3) => (x == null ? '–' : x.toFixed(d))
const fp = (x: number) => (x < 1e-4 ? x.toExponential(1) : x.toFixed(4))

export function App() {
  const [r, setR] = useState<Results | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    fetch('/results.json').then((x) => x.json()).then(setR).catch((e) => setErr(String(e)))
  }, [])
  if (err) return <p>Could not load results: {err}</p>
  if (!r) return <p>Loading results…</p>
  const cp = r.yield_time.change_point as Record<string, number | string>
  const s = r.signals
  const bucket = Object.fromEntries((r.regime_screen?.leads ?? []).map((l) => [l.feature, BUCKET[l.bucket] ?? l.bucket]))
  return (
    <main>
      <h1>Yield Root-Cause Lens</h1>
      <p>
        {r.health.units} production units, {r.health.features_raw} test measurements (UCI SECOM, CC BY 4.0).
        Yield {r.health.yield_pct}%.
      </p>
      <h2>1. When did yield move?</h2>
      <p>
        Change point at unit {cp.unit_index} ({cp.day}): yield {cp.yield_before_pct}% before, {cp.yield_after_pct}% after
        (permutation p = {cp.permutation_p}).
      </p>
      <h2>2. Which signals separate failing units?</h2>
      <p>
        {s.tested} signals tested on the first 70% of units: {s.raw_p_lt_05} at raw p &lt; 0.05, {s.fdr_q_lt_05} after
        Benjamini-Hochberg, {s.fdr_and_holdout} also hold on the later 30%.
        {r.regime_screen && <> Re-screened within production regime: {r.regime_screen.strat_fdr_leads} pass FDR, and{' '}
          {r.regime_screen.strat_fdr_and_holdout_list.join(', ') || 'none'} also hold on the later 30%.</>}
      </p>
      <div className="wrap">
        <table>
          <thead>
            <tr><th>Signal</th><th>p (train)</th><th>q</th><th>Cliff's δ train</th><th>δ test</th><th>p test (1-sided)</th><th>Top-10 stability</th><th>ρ with time</th><th>Verdict</th><th>Within-regime verdict</th></tr>
          </thead>
          <tbody>
            {s.top20.map((x) => (
              <tr key={x.feature}>
                <td>{x.feature}</td><td>{fp(x.p)}</td><td>{fp(x.q)}</td><td>{f(x.delta)}</td><td>{f(x.delta_test)}</td>
                <td>{f(x.p_test_1s)}</td><td>{f(x.stability, 2)}</td><td>{f(x.rho_time, 2)}</td>
                <td>{x.fdr_pass && x.holdout_pass ? 'holds' : x.fdr_pass ? 'fails held-out' : 'fails FDR'}</td>
                <td>{bucket[x.feature] ?? '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h2>3. Does it predict future units?</h2>
      <div className="wrap">
        <table>
          <thead><tr><th>Model (time-ordered test)</th><th>ROC-AUC</th><th>95% CI</th><th>PR-AUC</th><th>95% CI</th></tr></thead>
          <tbody>
            {Object.entries(r.models.results).map(([k, v]) => (
              <tr key={k}><td>{k}</td><td>{f(v.roc_auc)}</td><td>{v.roc_auc_ci.join('–')}</td><td>{f(v.pr_auc)}</td><td>{v.pr_auc_ci.join('–')}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Shuffled 5-fold CV (ignores time order): L1 ROC-AUC {r.models.shuffled_cv.l1_roc_auc}, boosting{' '}
        {r.models.shuffled_cv.hgb_roc_auc}. See METHOD.md for equations and validation.
      </p>
    </main>
  )
}
