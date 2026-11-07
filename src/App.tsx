import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  BUCKET_LABEL, bh, ci, fmtDay, keeps, n2, prepare, sgn, thousands, unitLabels, withinSplit, type Results,
} from './data'
import {
  CutStrip, Dumbbells, PairGrid, PowerCurve, RunChart, SignalPrintout, SplitScrubber, UnitTape, useActiveStep, type DumbRow,
} from './charts'

const STEPS = [
  { id: 'units', op: '010', short: 'Units ship' },
  { id: 'jump', op: '020', short: 'Yield jumps' },
  { id: 'disguise', op: '030', short: 'Date in disguise' },
  { id: 'model', op: '040', short: 'Model vs baseline' },
  { id: 'lead', op: '050', short: 'The one lead' },
  { id: 'know', op: '060', short: 'How we know' },
]

export function App() {
  const [r, setR] = useState<Results | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    fetch(import.meta.env.BASE_URL + 'results.json')
      .then((x) => { if (!x.ok) throw new Error(`HTTP ${x.status}`); return x.json() })
      .then(setR)
      .catch((e) => setErr(String(e)))
  }, [])
  if (err) return <p className="state">The results file did not load ({err}). Reload the page to try again.</p>
  if (!r) return <p className="state">Loading the test log…</p>
  return <Story r={r} />
}

function Stamp({ kind, children }: { kind: 'fail' | 'pass' | 'hold' | 'void'; children: ReactNode }) {
  return <span className={`stamp stamp-${kind}`}>{children}</span>
}

function Details({ children, label = 'Show details' }: { children: ReactNode; label?: string }) {
  return (
    <details className="details">
      <summary>
        <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 7.5 6 4 9.5" /></svg>
        <span>{label}</span>
      </summary>
      <div className="details-body">{children}</div>
    </details>
  )
}

function Op({ id, op, title, say, children, details }: {
  id: string; op: string; title: string; say: ReactNode; children: ReactNode; details?: ReactNode
}) {
  return (
    <section id={id} className="op" aria-labelledby={`${id}-h`}>
      <header className="op-head">
        <span className="op-no">OP {op}</span>
        <h2 id={`${id}-h`}>{title}</h2>
      </header>
      <p className="say">{say}</p>
      <div className="op-visual">{children}</div>
      {details && <Details>{details}</Details>}
    </section>
  )
}

function Story({ r }: { r: Results }) {
  const h = r.health
  const cp = r.yield_time.change_point
  const dates = r.signal_detail_time
  const labels = useMemo(() => unitLabels(r), [r])
  const active = useActiveStep(STEPS.map((s) => s.id))
  const activeIdx = STEPS.findIndex((s) => s.id === active)

  /* ---- step 1: a few log rows from the real first units ---- */
  const logSignals = ['S059', 'S103', 'S510']
  const logRows = useMemo(() => {
    const val: Record<string, Map<number, number>> = {}
    for (const f of logSignals) val[f] = new Map(r.signal_detail[f].series.map((s) => [s[0], s[1]]))
    const out: { i: number; day: string; y: number; v: (number | undefined)[] }[] = []
    for (let i = 0; out.length < 7 && i < h.units; i++) {
      if (labels[i] < 0) continue
      out.push({ i, day: dates[i], y: labels[i], v: logSignals.map((f) => val[f].get(i)) })
    }
    return out
  }, [r])

  /* ---- step 3: live split ---- */
  const leadSet = new Set(r.regime_screen.leads.map((l) => l.feature))
  const prepared = useMemo(
    () => Object.entries(r.signal_detail).filter(([f]) => leadSet.has(f)).map(([f, d]) => prepare(f, d, h.train_units))
      .sort((a, b) => Math.abs(b.pooled) - Math.abs(a.pooled)),
    [r],
  )
  const bucketOf = Object.fromEntries(r.regime_screen.leads.map((l) => [l.feature, l.bucket]))
  const minSplit = 60, maxSplit = h.train_units - 60
  const [split, setSplit] = useState(cp.unit_index)
  const curve = useMemo(() => {
    const out: { unit: number; keep: number }[] = []
    for (let u = minSplit; u <= maxSplit; u += 8) {
      out.push({ unit: u, keep: prepared.filter((s) => keeps(s.pooled, withinSplit(s, u).within)).length })
    }
    return out
  }, [prepared])
  const rows: DumbRow[] = useMemo(() => prepared.map((s) => {
    const w = withinSplit(s, split)
    const k = keeps(s.pooled, w.within)
    return { feature: s.feature, pooled: s.pooled, within: w.within, keep: k, verdict: k ? 'keeps' : 'gone' }
  }), [prepared, split])
  const keptNow = rows.filter((x) => x.keep).length
  const proxies = r.regime_screen.bucket_counts.b_regime_proxy ?? 0
  const defaultSel = prepared.find((s) => bucketOf[s.feature] === 'b_regime_proxy')?.feature ?? prepared[0].feature
  const [sel, setSel] = useState(defaultSel)
  const selPrep = prepared.find((s) => s.feature === sel) ?? prepared[0]
  const missingUnitLevel = r.regime_screen.leads.length - prepared.length

  /* ---- step 4: model vs baseline ---- */
  const rm = r.ai.regime_model
  const fair = rm.fairer_pooled_check
  const baselines = {
    pooled: {
      label: 'Same model, all history', auc: rm.mean_over_cuts.auc_pooled,
      diff: rm.mean_over_cuts.diff_vs_pooled, diffCi: rm.mean_over_cuts.diff_vs_pooled_ci,
      perCut: rm.per_cut.map((c) => c.auc_pooled),
      note: 'The same L1 logistic model trained on every unit, with no idea the line changed.',
    },
    regime: {
      label: 'All history + regime column', auc: fair.auc_pooled_with_regime_indicator,
      diff: fair.diff_vs_pooled_with_regime_indicator, diffCi: fair.diff_vs_pooled_with_regime_indicator_ci,
      perCut: fair.per_cut.map((c) => c.auc_pooled_with_regime_indicator),
      note: 'The all-history model given one extra column: before or after the change. Added at review.',
    },
    single: {
      label: 'Best single signal', auc: rm.mean_over_cuts.auc_single,
      diff: rm.mean_over_cuts.diff_vs_single, diffCi: rm.mean_over_cuts.diff_vs_single_ci,
      perCut: rm.per_cut.map((c) => c.auc_single),
      note: 'One signal, the best within-regime separator on training units, used as the risk score.',
    },
  }
  type BK = keyof typeof baselines
  const [bk, setBk] = useState<BK>('pooled')
  const b = baselines[bk]
  const beats = b.diffCi[0] > 0
  const cutRows = rm.per_cut.map((c, i) => ({
    label: `Cut ${fmtDay(c.cut_day)}`, sub: `${c.test_fails} later fails`, model: c.auc_regime, base: b.perCut[i],
  }))

  /* ---- step 5: the one lead ---- */
  const pp = r.power_plan
  const strat = r.regime_screen.strat_leads_detail
  const qNine = bh(strat.map((s) => s.p_test_1s))
  const leadName = r.regime_screen.strat_fdr_and_holdout_list[0]
  const leadIdx = strat.findIndex((s) => s.feature === leadName)
  const lead = strat[leadIdx]
  const leadCurve = pp.curve.find((c) => Math.abs(c.delta - lead.delta_post_change_train) < 1e-9)
  const windowDays = Math.round(h.test_units / pp.units_per_day)
  const testDelta = Object.fromEntries(strat.map((s) => [s.feature, s.delta_test]))
  const unconfirmed = pp.leads.filter((l) => l.bucket === 'e_unconfirmed' && l.days_to_confirm != null).map((l) => ({
    feature: l.feature, delta: l.planning_delta, days: l.days_to_confirm as number,
    weak: (l.days_to_confirm as number) < windowDays && (testDelta[l.feature] ?? 0) < 0,
  }))
  const dayRange = [Math.min(...unconfirmed.map((l) => l.days)), Math.max(...unconfirmed.map((l) => l.days))]
  const chanceOne = 1 - Math.pow(0.95, strat.length)

  const ifo = r.ai.anomaly.isolation_forest
  const bern = r.ai.anomaly.bernoulli_cusum
  const rank10 = r.ai.ranker.pooled.find((p) => p.k === r.ai.ranker.primary_k)!
  const models = r.models.results
  const l1 = models['L1 logistic'], hgb = models['Gradient boosting']
  const econ = pp.economics.fails_per_1000
  const cl = r.change_log

  return (
    <>
      <div className="progress" aria-hidden="true"><span style={{ transform: `scaleX(${(activeIdx + 1) / STEPS.length})` }} /></div>
      <div className="frame">
        <nav className="rail" aria-label="Steps">
          <p className="rail-title">Routing</p>
          <ol>
            {STEPS.map((s, i) => (
              <li key={s.id} className={(s.id === active ? 'on ' : '') + (i < activeIdx ? 'done' : '')}>
                <a href={`#${s.id}`} aria-current={s.id === active ? 'step' : undefined}>
                  <span className="rail-op">{s.op}</span><span>{s.short}</span>
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <main className="sheet">
          <header className="card-head">
            <h1>Yield Root-Cause Lens</h1>
            <p className="lede">
              A production line's yield jumped. This walks through which test signals actually point at failing units,
              and which only point at the calendar.
            </p>
            <dl className="fields">
              <div><dt>Units</dt><dd>{thousands(h.units)}</dd></div>
              <div><dt>Signals per unit</dt><dd>{h.features_raw}</dd></div>
              <div><dt>Failed</dt><dd>{h.fails}</dd></div>
              <div><dt>Run</dt><dd>{fmtDay(h.first_unit)}–{fmtDay(h.last_unit, true)}</dd></div>
            </dl>
          </header>

          <Op id="units" op="010" title="A line ships units"
            say={<>Every unit gets {h.features_raw} measurements at end-of-line test and a pass or fail. {thousands(h.units)} units, {h.fails} fails, in time order.</>}
            details={<>
              <p>Of the {h.features_raw} signals, {h.features_constant} never change and {h.features_over_50pct_missing} are missing on more than half the units, which leaves {h.features_kept}. Every choice is made on the first 70% of units in time ({thousands(h.train_units)} units, {h.train_fails} fails); the last 30% ({h.test_units} units, {h.test_fails} fails) are kept back as "later units".</p>
              <p>The signals are anonymised, so a lead is a measurement to inspect, not a named part.</p>
            </>}>
            <div className="log" role="table" aria-label="First units in the test log">
              <div className="log-row log-headrow" role="row">
                <span role="columnheader">Serial</span><span role="columnheader">Tested</span>
                {logSignals.map((f) => <span key={f} role="columnheader" className="num">{f}</span>)}
                <span role="columnheader">Result</span>
              </div>
              {logRows.map((row) => (
                <div className="log-row" role="row" key={row.i}>
                  <span role="cell" className="serial">U-{String(row.i + 1).padStart(4, '0')}</span>
                  <span role="cell">{fmtDay(row.day)}</span>
                  {row.v.map((v, j) => <span key={j} role="cell" className="num">{v == null ? '—' : fmtVal(v)}</span>)}
                  <span role="cell">{row.y === 1 ? <Stamp kind="fail">Fail</Stamp> : <Stamp kind="pass">Pass</Stamp>}</span>
                </div>
              ))}
            </div>
            <p className="caption">Every unit as a tick. Fails stand tall.</p>
            <UnitTape labels={labels} dates={dates} changeUnit={cp.unit_index} />
          </Op>

          <Op id="jump" op="020" title="Yield jumps on one date"
            say={<>Around {fmtDay(cp.day)} yield steps from {n2(cp.yield_before_pct, 1)}% to {n2(cp.yield_after_pct, 1)}%: about {Math.round(econ.before)} failing units per 1,000 down to {Math.round(econ.after)}.</>}
            details={<>
              <table className="spec">
                <tbody>
                  <tr><th>Change point</th><td>unit {cp.unit_index}, {fmtDay(cp.day, true)}</td></tr>
                  <tr><th>Date, 95% interval</th><td>{fmtDay(cp.location_ci_days[0])} to {fmtDay(cp.location_ci_days[1])} (hatched)</td></tr>
                  <tr><th>Yield before / after</th><td>{n2(cp.yield_before_pct, 1)}% ({n2(cp.ci_before_pct[0], 1)}–{n2(cp.ci_before_pct[1], 1)}) / {n2(cp.yield_after_pct, 1)}% ({n2(cp.ci_after_pct[0], 1)}–{n2(cp.ci_after_pct[1], 1)})</td></tr>
                  <tr><th>Shift</th><td>+{n2(cp.shift_pts, 1)} points, 95% CI {n2(cp.shift_ci_pts[0], 1)} to {n2(cp.shift_ci_pts[1], 1)}</td></tr>
                  <tr><th>Permutation test</th><td>p = {cp.permutation_p} (999 reorderings)</td></tr>
                  <tr><th>Fails per 1,000</th><td>{econ.before} before, {econ.after} after ({econ.avoided} fewer, CI {ci(econ.avoided_ci, 0)})</td></tr>
                </tbody>
              </table>
              <p>Method: least-squares single change on the per-unit fail indicator, cross-checked by a CUSUM on batches of {r.yield_time.cusum.batch_units} units (k = {r.yield_time.cusum.k}, h = {r.yield_time.cusum.h}).</p>
            </>}>
            <RunChart daily={r.yield_time.daily} cp={cp} />
            <p className="caption">Daily yield, dot size by units tested. Days below 50% sit as triangles on the floor.</p>
          </Op>

          <Op id="disguise" op="030" title="Most 'root causes' are that date in disguise"
            say={<>Fails bunch up before the change, so any signal that drifted with time also lines up with fails. Drag the split: a real lead keeps separating fails on both sides of it.</>}
            details={<>
              <p>The pipeline's verdicts on all {r.regime_screen.leads.length} leads from the plain fail-vs-pass screen, from a stratified rank test inside each regime (training units only). {proxies} of {r.regime_screen.leads.length} are regime proxies.</p>
              <div className="table-wrap">
                <table className="grid-table">
                  <thead><tr><th>Signal</th><th className="num">Plain δ</th><th className="num">q plain</th><th className="num">q within</th><th className="num">δ later</th><th>Verdict</th></tr></thead>
                  <tbody>
                    {r.regime_screen.leads.map((l) => (
                      <tr key={l.feature}>
                        <td className="serial">{l.feature}</td><td className="num">{sgn(l.delta_pooled)}</td>
                        <td className="num">{fmtQ(l.q_pooled)}</td><td className="num">{fmtQ(l.q_strat)}</td>
                        <td className="num">{sgn(l.delta_test)}</td><td>{BUCKET_LABEL[l.bucket] ?? l.bucket}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p>Why the within-regime test: in {r.regime_screen.simulation.runs} simulated lines where a signal only trends with time, the plain test calls it a cause {n2(r.regime_screen.simulation.fp_pooled * 100, 1)}% of the time; the stratified test {n2(r.regime_screen.simulation.fp_stratified * 100, 1)}%.</p>
              <p>The chart recomputes Cliff's δ in your browser from the shipped unit values (training units; {prepared.length} of the {r.regime_screen.leads.length} leads, {missingUnitLevel} are not shipped at unit level). "Keeps" means the effect inside the split is at least half the plain effect. It is a lens on the pipeline's test, not the test itself.</p>
              <p>Separately, {cl.aligned_narrow} signals stepped within ±{cl.narrow_window_days} days of the jump, about what timing chance gives (expected {n2(cl.null_mean_narrow, 1)}, p = {n2(cl.alignment_p_narrow)}). They are a checklist for the change records, not a verdict.</p>
            </>}>
            <div className="split-readout" aria-live="polite">
              <span className="tally">{keptNow}<small>/{rows.length}</small></span>
              <span>signals still separate fails when split at <b>{fmtDay(dates[split], true)}</b></span>
              {split !== cp.unit_index && <button type="button" className="btn-link" onClick={() => setSplit(cp.unit_index)}>Back to {fmtDay(cp.day)}</button>}
            </div>
            <p className="caption cap-tight">How many signals keep their effect, for every split date. The dashed red line is the yield change.</p>
            <SplitScrubber curve={curve} split={split} setSplit={setSplit} maxUnit={maxSplit} dates={dates} changeUnit={cp.unit_index} total={rows.length} />
            <div className="legend" aria-hidden="true">
              <span><i className="lg-dot pooled" />plain effect, all units</span>
              <span><i className="lg-dot within" />effect inside the split</span>
            </div>
            <Dumbbells rows={rows} selected={sel} onSelect={setSel} />
            <p className="caption">Tap a signal for its printout. Passes print above the line, fails below.</p>
            <div className="printout-head">
              <span className="serial">{sel}</span>
              <span>pipeline verdict: {BUCKET_LABEL[bucketOf[sel]] ?? '—'}</span>
            </div>
            <SignalPrintout s={selPrep} detail={r.signal_detail[sel]} split={split} dates={dates} />
          </Op>

          <Op id="model" op="040" title="The AI model on later units"
            say={<>A model trained only on post-change units scores units it has never seen. It beats the naive baseline and ties the fair ones.</>}
            details={<>
              <table className="spec">
                <tbody>
                  <tr><th>Regime-aware model</th><td>ROC-AUC {n2(rm.mean_over_cuts.auc_regime, 3)}, mean over {rm.per_cut.length} time cuts</td></tr>
                  {(Object.keys(baselines) as BK[]).map((k) => (
                    <tr key={k}><th>vs {baselines[k].label.toLowerCase()}</th><td>{n2(baselines[k].auc, 3)}; difference {sgn(baselines[k].diff)} (95% CI {ci(baselines[k].diffCi)})</td></tr>
                  ))}
                  <tr><th>Solver seeds</th><td>gain over all history {sgn(Math.min(...rm.solver_seed_sensitivity.map((s) => s.diff_vs_pooled)))} to {sgn(Math.max(...rm.solver_seed_sensitivity.map((s) => s.diff_vs_pooled)))}</td></tr>
                </tbody>
              </table>
              <p>Validation is split by whole days and CIs come from a day-block bootstrap. The cuts' test windows are nested, so they are not independent replications.</p>
              <p><b>Two other AI parts lose.</b> A stability-selection ranker's top {r.ai.ranker.primary_k} signals replicate on later units less well than a plain t-test ranking (mean held-out δ {n2(rank10.ai_mean_delta, 3)} vs {n2(rank10.t_mean_delta, 3)}; difference CI {ci(rank10.diff_ai_vs_t_ci)}). An Isolation Forest change detector first alarms on {fmtDay(ifo.alarm_day)}, {thousands(ifo.delay_units)} units after the change; the pass/fail CUSUM it was matched against alarms on {fmtDay(bern.alarm_day)}, two weeks early. Neither detects the step.</p>
              <p><b>Shuffling flatters.</b> All-signal models on the time-ordered split score ROC-AUC {n2(l1.roc_auc)} (L1) and {n2(hgb.roc_auc)} (boosting); with shuffled cross-validation the same models score {n2(r.models.shuffled_cv.l1_roc_auc)} and {n2(r.models.shuffled_cv.hgb_roc_auc)}.</p>
            </>}>
            <div className="toggle" role="radiogroup" aria-label="Baseline to compare against">
              {(Object.keys(baselines) as BK[]).map((k) => (
                <button key={k} type="button" role="radio" aria-checked={bk === k} className={bk === k ? 'on' : ''} onClick={() => setBk(k)}>
                  {baselines[k].label}
                </button>
              ))}
            </div>
            <p className="caption">Each square is one failing and one passing later unit, marked when the score calls the failing one riskier. A coin flip marks 50.</p>
            <div className="pairs">
              <figure>
                <figcaption><span className="who model">Regime-aware model</span><span className="score">{Math.round(rm.mean_over_cuts.auc_regime * 100)}<small>/100</small></span></figcaption>
                <PairGrid auc={rm.mean_over_cuts.auc_regime} label="Regime-aware model" variant="model" />
              </figure>
              <figure>
                <figcaption><span className="who">{b.label}</span><span className="score">{Math.round(b.auc * 100)}<small>/100</small></span></figcaption>
                <PairGrid auc={b.auc} label={b.label} variant="base" />
              </figure>
            </div>
            <div className="verdict">
              {beats ? <Stamp kind="hold">Beats</Stamp> : <Stamp kind="void">Tie</Stamp>}
              <p>
                Difference {sgn(b.diff)}, 95% CI {ci(b.diffCi)}.{' '}
                {beats ? 'The interval clears zero.' : 'The interval includes zero, so this is not a win.'} {b.note}
              </p>
            </div>
            <CutStrip rows={cutRows} />
            <div className="legend" aria-hidden="true">
              <span><i className="lg-dot model" />regime-aware model</span>
              <span><i className="lg-dot pooled" />{b.label.toLowerCase()}</span>
            </div>
          </Op>

          <Op id="lead" op="050" title="One real lead, and the days to confirm it"
            say={<>Inside the regime, {leadName} separates fails, and it is the only lead that does it again on later units. If its effect is real, about {leadCurve ? Math.round(leadCurve.days) : '—'} days of production will confirm it.</>}
            details={<>
              <table className="spec">
                <tbody>
                  <tr><th>{leadName}, training</th><td>within-regime q = {n2(lead.q_strat, 3)}; plain screen q = {n2(lead.q_pooled, 2)}, so the plain list missed it</td></tr>
                  <tr><th>{leadName}, later units</th><td>δ = {n2(lead.delta_test)}, one-sided p = {n2(lead.p_test_1s, 3)}</td></tr>
                  <tr><th>Caveat</th><td>{strat.length} leads were re-tested, so one passing at 0.05 happens by chance about {Math.round(chanceOne * 100)}% of the time. Benjamini-Hochberg across the {strat.length}: q = {n2(qNine[leadIdx])}. A candidate, not a confirmed cause.</td></tr>
                  <tr><th>Smallest visible effect</th><td>δ = {n2(pp.min_detectable_delta_80)} at 80% power with {pp.test_fails} later fails; {pp.leads_below_mdd} of the {r.regime_screen.leads.length} leads sit below it</td></tr>
                  {leadCurve && <tr><th>To confirm {leadName}</th><td>{thousands(leadCurve.units)} units, about {leadCurve.fails} fails, {n2(leadCurve.days, 1)} days at {n2(pp.units_per_day, 1)} units a day</td></tr>}
                </tbody>
              </table>
              <p>The other {unconfirmed.length} within-regime leads need {Math.round(dayRange[0])} to {Math.round(dayRange[1])} days each, assuming their training effect holds. Training effects of selected leads run high: the later-unit window (about {windowDays} days) was already long enough for {unconfirmed.filter((l) => l.weak).map((l) => l.feature).join(' and ')}, and both came out negative (hollow dots).</p>
              <p>Sample size: Noether's formula N = (z<sub>α</sub> + z<sub>β</sub>)² / (3c(1−c)δ²), converted to days at the observed throughput and the post-change fail rate ({n2(pp.post_change_fail_rate * 100, 2)}%), and checked against a Monte Carlo on real {pp.mc_feature} values.</p>
            </>}>
            <PowerCurve curve={pp.curve} leads={unconfirmed}
              focus={{ feature: leadName, delta: lead.delta_post_change_train, days: leadCurve?.days ?? 0 }}
              windowDays={windowDays} mdd={pp.min_detectable_delta_80} testFails={pp.test_fails} />
            <p className="caption">Days of production needed to confirm a lead, by its effect size. The red square is {leadName}.</p>
          </Op>

          <Op id="know" op="060" title="How we know"
            say={<>Each statistic is written out and checked against a hand calculation or a published value. Every number on this page is read from the pipeline's output.</>}
            details={<>
              <ul className="plain">
                <li>Mann-Whitney U, Cliff's δ, Benjamini-Hochberg, the change point and the CUSUM are implemented by hand and checked against SciPy and textbook examples.</li>
                <li>A planted step at unit 300 is recovered exactly, with and without 30% missing values.</li>
                <li>Imputation and scaling are fitted on training rows only; time cuts never split a day.</li>
                <li>The labels are end-of-line pass/fail. There are no defect codes or station IDs, so a lead is a place to look.</li>
              </ul>
            </>}>
            <div className="signoff" role="table" aria-label="Validation checks">
              {[
                ['Yield by hand, 1463 / 1567', '93.364%', `${h.yield_pct}%`],
                ['Rank test vs SciPy, 20 cases', 'equal', 'err < 1e-9'],
                ['CUSUM run length vs published 10.4', '±0.6', 'within'],
                ['Trend-only signal, 500 lines', '≈5% false', `${n2(r.regime_screen.simulation.fp_stratified * 100, 1)}%`],
                ['Power formula vs Monte Carlo', 'few points', 'within 3'],
                ['Fitting sees training rows only', 'always', 'enforced'],
              ].map(([check, want, got]) => (
                <div className="signoff-row" role="row" key={check}>
                  <span role="cell">{check}</span><span role="cell" className="num muted">{want}</span>
                  <span role="cell" className="num">{got}</span><span role="cell"><Stamp kind="pass">OK</Stamp></span>
                </div>
              ))}
            </div>
            <p className="caption">42 automated tests. Equations, assumptions and limits are in the repository's method notes.</p>
          </Op>

          <footer className="foot">
            <p>Data: SECOM dataset, UCI Machine Learning Repository (2008), doi:10.24432/C54305, licensed CC BY 4.0. Real production test data; no synthetic values on this page.</p>
            <p className="more">More work at <a href="https://ibiraheel.com">ibiraheel.com</a></p>
          </footer>
        </main>
      </div>
    </>
  )
}

function fmtVal(v: number) {
  const a = Math.abs(v)
  return a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(2) : v.toFixed(4)
}
function fmtQ(q: number) {
  return q < 0.001 ? '<0.001' : q.toFixed(3)
}
