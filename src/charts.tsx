import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { dayNum, fmtDay, n2, type Daily, type Prepared, type SignalDetail } from './data'

/* ---------- shared plumbing ---------- */

export function useWidth<T extends HTMLElement>(fallback = 640) {
  const ref = useRef<T>(null)
  const [w, setW] = useState(fallback)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.round(e.contentRect.width))))
    ro.observe(el)
    setW(Math.max(260, Math.round(el.getBoundingClientRect().width)))
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

type Tip = { x: number; y: number; body: ReactNode } | null
function TipBox({ tip, width }: { tip: Tip; width: number }) {
  if (!tip) return null
  const left = Math.min(Math.max(tip.x, 8), width - 8)
  const flip = tip.x > width * 0.6
  return (
    <div className="tip" style={{ left, top: tip.y, transform: `translate(${flip ? 'calc(-100% - 12px)' : '12px'}, -50%)` }} role="status">
      {tip.body}
    </div>
  )
}

const lin = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)

/* ---------- 1. unit tape: every unit as a tick, fails tall ---------- */

export function UnitTape({ labels, dates, changeUnit }: { labels: Int8Array; dates: string[]; changeUnit: number }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip>(null)
  const H = 74, top = 14, base = 52
  const n = labels.length
  const x = lin(0, n - 1, 0.5, w - 0.5)
  const { passD, failD } = useMemo(() => {
    let p = '', f = ''
    for (let i = 0; i < n; i++) {
      const xi = x(i).toFixed(1)
      if (labels[i] === 1) f += `M${xi} ${top}V${base}`
      else p += `M${xi} ${base - 14}V${base}`
    }
    return { passD: p, failD: f }
  }, [labels, w])
  const months = useMemo(() => {
    const out: { i: number; label: string }[] = []
    let last = ''
    dates.forEach((d, i) => { const m = d.slice(0, 7); if (m !== last) { out.push({ i, label: fmtDay(d).split(' ')[1] }); last = m } })
    return out.slice(1)
  }, [dates])
  function move(e: React.PointerEvent) {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect()
    const i = Math.round(((e.clientX - r.left) / r.width) * (n - 1))
    if (i < 0 || i >= n) return
    setTip({
      x: e.clientX - r.left, y: 30,
      body: <><b>Unit {String(i + 1).padStart(4, '0')}</b><span>{fmtDay(dates[i], true)}</span><span className={labels[i] === 1 ? 'fail-ink' : ''}>{labels[i] === 1 ? 'Failed end-of-line test' : 'Passed'}</span></>,
    })
  }
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={H} onPointerMove={move} onPointerLeave={() => setTip(null)} role="img"
        aria-label={`All ${n} units in time order; failing units drawn as tall red ticks.`}>
        <path d={passD} className="tick-pass" />
        <path d={failD} className="tick-fail" />
        <line x1={x(changeUnit)} x2={x(changeUnit)} y1={2} y2={base + 4} className="mark-change" />
        <line x1={0} x2={w} y1={base + 0.5} y2={base + 0.5} className="axis" />
        {months.map((m) => (
          <g key={m.i}>
            <line x1={x(m.i)} x2={x(m.i)} y1={base} y2={base + 5} className="axis" />
            <text x={x(m.i) + 3} y={base + 17} className="ax-label">{m.label}</text>
          </g>
        ))}
      </svg>
      <TipBox tip={tip} width={w} />
    </div>
  )
}

/* ---------- 2. yield run chart, printed like a line-side board ---------- */

export function RunChart({ daily, cp }: {
  daily: Daily[]
  cp: { day: string; yield_before_pct: number; yield_after_pct: number; location_ci_days: string[] }
}) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip>(null)
  const [hover, setHover] = useState<number | null>(null)
  const narrow = w < 520
  const H = narrow ? 260 : 320
  const m = { l: 40, r: narrow ? 12 : 64, t: 30, b: 30 }
  const d0 = dayNum(daily[0].day) - 1, d1 = dayNum(daily[daily.length - 1].day) + 1
  const x = lin(d0, d1, m.l, w - m.r)
  const y = lin(50, 100, H - m.b, m.t)
  const cx = x(dayNum(cp.day))
  const maxU = Math.max(...daily.map((d) => d.units))
  const r = (u: number) => 2.2 + 3.2 * Math.sqrt(u / maxU)
  const ticks = [50, 60, 70, 80, 90, 100]
  const monthTicks = daily.filter((d) => d.day.endsWith('-01') || d.day === '2008-08-01' || d.day === '2008-09-01' || d.day === '2008-10-01')
  function move(e: React.PointerEvent) {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect()
    const px = e.clientX - rect.left
    let best = 0, bd = Infinity
    daily.forEach((d, i) => { const dd = Math.abs(x(dayNum(d.day)) - px); if (dd < bd) { bd = dd; best = i } })
    const d = daily[best]
    setHover(best)
    setTip({ x: x(dayNum(d.day)), y: Math.max(40, y(Math.max(50, d.yield_pct))), body: <><b>{fmtDay(d.day, true)}</b><span>{d.units} units, {d.fails} failed</span><span>Yield {n2(d.yield_pct, 1)}%</span></> })
  }
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={H} onPointerMove={move} onPointerLeave={() => { setTip(null); setHover(null) }} role="img"
        aria-label={`Daily yield. Before ${fmtDay(cp.day)} the average is ${cp.yield_before_pct}%, after it ${cp.yield_after_pct}%.`}>
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" className="hatch-line" />
          </pattern>
          <clipPath id="reveal"><rect className="reveal-rect" x={0} y={0} width={w} height={H} /></clipPath>
        </defs>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} className={t === 100 ? 'grid strong' : 'grid'} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" className="ax-label">{t}%</text>
          </g>
        ))}
        {monthTicks.map((d) => (
          <text key={d.day} x={x(dayNum(d.day))} y={H - 8} className="ax-label" textAnchor="middle">{fmtDay(d.day)}</text>
        ))}
        <rect x={x(dayNum(cp.location_ci_days[0]))} width={cx - x(dayNum(cp.location_ci_days[0]))} y={m.t} height={H - m.t - m.b} fill="url(#hatch)" />
        <g clipPath="url(#reveal)">
          <line x1={m.l} x2={cx} y1={y(cp.yield_before_pct)} y2={y(cp.yield_before_pct)} className="level" />
          <line x1={cx} x2={w - m.r} y1={y(cp.yield_after_pct)} y2={y(cp.yield_after_pct)} className="level" />
          <line x1={cx} x2={cx} y1={y(cp.yield_before_pct)} y2={y(cp.yield_after_pct)} className="level" />
          {daily.map((d, i) => {
            const v = d.yield_pct
            const px = x(dayNum(d.day))
            if (v < 50) return <path key={d.day} d={`M${px - 4} ${y(50) - 7}h8l-4 6z`} className={'pt-low' + (hover === i ? ' on' : '')} />
            return <circle key={d.day} cx={px} cy={y(v)} r={r(d.units)} className={'pt' + (hover === i ? ' on' : '')} />
          })}
        </g>
        <line x1={cx} x2={cx} y1={m.t - 12} y2={H - m.b} className="mark-change" />
        <text x={cx + 6} y={m.t - 16} className="ann fail-ink">change {fmtDay(cp.day)}</text>
        {!narrow && <>
          <text x={w - m.r + 6} y={y(cp.yield_after_pct) + 4} className="ann">{n2(cp.yield_after_pct, 1)}%</text>
          <text x={m.l + 6} y={y(cp.yield_before_pct) + 16} className="ann">{n2(cp.yield_before_pct, 1)}%</text>
        </>}
      </svg>
      <TipBox tip={tip} width={w} />
    </div>
  )
}

/* ---------- 3a. survivor curve + split scrubber ---------- */

export function SplitScrubber({ curve, split, setSplit, maxUnit, dates, changeUnit, total }: {
  curve: { unit: number; keep: number }[]; split: number; setSplit: (u: number) => void; maxUnit: number
  dates: string[]; changeUnit: number; total: number
}) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const H = 120, m = { l: 26, r: 10, t: 12, b: 28 }
  const minU = curve[0]?.unit ?? 0, maxU = curve[curve.length - 1]?.unit ?? maxUnit
  const x = lin(minU, maxU, m.l, w - m.r)
  const y = lin(0, total, H - m.b, m.t)
  const path = curve.map((c, i) => `${i ? 'L' : 'M'}${x(c.unit).toFixed(1)} ${y(c.keep).toFixed(1)}`).join('')
  const ticks = curve.filter((c, i) => i > 0 && dates[c.unit].slice(8, 10) === '01' && dates[curve[i - 1].unit].slice(8, 10) !== '01')
  const sx = x(Math.min(Math.max(split, minU), maxU))
  return (
    <div className="scrubber" ref={ref}>
      <svg width={w} height={H} aria-hidden="true">
        {[0, total / 2, total].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={m.l - 5} y={y(t) + 4} textAnchor="end" className="ax-label">{t}</text>
          </g>
        ))}
        {ticks.map((c) => <text key={c.unit} x={x(c.unit)} y={H - 4} textAnchor="middle" className="ax-label">{fmtDay(dates[c.unit])}</text>)}
        <line x1={x(changeUnit)} x2={x(changeUnit)} y1={m.t} y2={H - m.b} className="mark-change thin" />
        <path d={`M${x(changeUnit)} ${H - m.b + 1}l5 7h-10z`} className="change-caret" />
        <path d={path} className="curve" />
        <line x1={sx} x2={sx} y1={m.t - 4} y2={H - m.b} className="cursor" />
      </svg>
      <input
        type="range" min={minU} max={maxU} step={1} value={split}
        onChange={(e) => setSplit(+e.target.value)}
        style={{ left: m.l - 11, width: w - m.l - m.r + 22 }}
        aria-label="Split date"
        aria-valuetext={`Split on ${fmtDay(dates[split], true)}`}
      />
    </div>
  )
}

/* ---------- 3b. dumbbells: pooled effect vs effect inside each side of the split ---------- */

export type DumbRow = { feature: string; pooled: number; within: number; keep: boolean; verdict: string }

export function Dumbbells({ rows, selected, onSelect }: { rows: DumbRow[]; selected: string; onSelect: (f: string) => void }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const labelW = 52, stampW = 64, rowH = 22
  const x = lin(-0.1, 0.6, labelW + 8, w - stampW - 8)
  const H = rows.length * rowH + 28
  return (
    <div className="chart dumb" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Each signal's separation between failing and passing units, pooled and within the split.">
        {[0, 0.2, 0.4, 0.6].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={4} y2={H - 22} className={t === 0 ? 'axis' : 'grid'} />
            <text x={x(t)} y={H - 6} textAnchor="middle" className="ax-label">{t === 0 ? '0' : t.toFixed(1)}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = 12 + i * rowH
          const p = Math.abs(r.pooled), wv = Math.sign(r.pooled) * r.within
          const sel = r.feature === selected
          return (
            <g key={r.feature} className={'drow' + (r.keep ? ' keep' : ' gone') + (sel ? ' sel' : '')}
              onClick={() => onSelect(r.feature)} role="button" tabIndex={0} aria-pressed={sel}
              aria-label={`${r.feature}: pooled ${n2(p)}, within split ${n2(wv)}; ${r.keep ? 'keeps' : 'loses'} its effect`}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(r.feature) } }}>
              <rect x={0} y={cy - rowH / 2} width={w} height={rowH} className="hit" />
              <text x={0} y={cy + 4} className="sig">{r.feature}</text>
              <line x1={x(p)} x2={x(Math.max(-0.1, wv))} y1={cy} y2={cy} className="bar" />
              <circle cx={x(p)} cy={cy} r={4.5} className="dot-pooled" />
              <circle cx={x(Math.max(-0.1, wv))} cy={cy} r={4.5} className="dot-within" />
              <text x={w - stampW + 4} y={cy + 4} className="mini-stamp">{r.keep ? 'KEEPS' : 'GONE'}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/* ---------- 3c. signal printout: distribution of passes and fails, before and after ---------- */

export function SignalPrintout({ s, detail, split, dates }: { s: Prepared; detail: SignalDetail; split: number; dates: string[] }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const edges = detail.edges
  const bins = edges.length - 1
  const side = (before: boolean) => {
    const pass = new Array(bins).fill(0), fail = new Array(bins).fill(0)
    let np = 0, nf = 0
    for (let i = 0; i < s.idx.length; i++) {
      if ((s.idx[i] < split) !== before) continue
      let b = Math.floor(((s.val[i] - edges[0]) / (edges[bins] - edges[0])) * bins)
      b = Math.min(bins - 1, Math.max(0, b))
      if (s.y[i]) { fail[b]++; nf++ } else { pass[b]++; np++ }
    }
    return { pass: pass.map((v) => (np ? v / np : 0)), fail: fail.map((v) => (nf ? v / nf : 0)), np, nf }
  }
  const sides = [side(true), side(false)]
  const peak = Math.max(0.0001, ...sides.flatMap((d) => [...d.pass, ...d.fail]))
  const twoUp = w >= 560
  const pw = twoUp ? (w - 16) / 2 : w
  const H = 150, mid = 74
  const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toPrecision(3))
  const panel = (d: ReturnType<typeof side>, title: string) => {
    const x = lin(0, bins, 6, pw - 6)
    const h = lin(0, peak, 0, mid - 14)
    const bw = Math.max(1, (pw - 12) / bins - 2)
    return (
      <figure className="printout" key={title}>
        <figcaption><span>{title}</span><span className="muted">{d.np} pass · {d.nf} fail</span></figcaption>
        <svg width={pw} height={H} role="img" aria-label={`${s.feature} ${title}: pass distribution above the line, fail below.`}>
          <line x1={0} x2={pw} y1={mid} y2={mid} className="axis" />
          {d.pass.map((v, i) => v > 0 && <rect key={'p' + i} x={x(i) + 1} y={mid - 1 - h(v)} width={bw} height={h(v)} className="bin-pass" rx={1} />)}
          {d.fail.map((v, i) => v > 0 && <rect key={'f' + i} x={x(i) + 1} y={mid + 1} width={bw} height={h(v)} className="bin-fail" rx={1} />)}
          <text x={pw - 6} y={12} textAnchor="end" className="ax-label halo">pass</text>
          <text x={pw - 6} y={H - 22} textAnchor="end" className="ax-label fail-ink halo">fail</text>
          <text x={6} y={H - 4} className="ax-label">{fmt(edges[0])}</text>
          <text x={pw - 6} y={H - 4} textAnchor="end" className="ax-label">{fmt(edges[bins])}</text>
        </svg>
      </figure>
    )
  }
  return (
    <div className={'printouts' + (twoUp ? ' two' : '')} ref={ref}>
      {panel(sides[0], `Before ${fmtDay(dates[split])}`)}
      {panel(sides[1], `From ${fmtDay(dates[split])}`)}
    </div>
  )
}

/* ---------- 4a. 100 fail/pass pairs: the AUC as a traveler grid ---------- */

export function PairGrid({ auc, label, variant = 'model' }: { auc: number; label: string; variant?: 'model' | 'base' }) {
  const right = Math.round(auc * 100)
  return (
    <div className={'pairgrid ' + variant} role="img" aria-label={`${label}: ranks the failing unit riskier in ${right} of 100 pairs.`}>
      {Array.from({ length: 100 }, (_, i) => (
        <span key={i} className={'cell' + (i < right ? ' ok' : '')} style={{ transitionDelay: `${Math.abs(i - 55) * 4}ms` }} />
      ))}
      <span className="coin" aria-hidden="true"><b>50</b></span>
    </div>
  )
}

/* ---------- 4b. per-cut strip: model vs a chosen baseline ---------- */

export function CutStrip({ rows }: { rows: { label: string; sub: string; model: number; base: number }[] }) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const narrow = w < 480
  const labelW = narrow ? 0 : 150
  const rowH = narrow ? 52 : 36
  const x = lin(0.4, 0.8, labelW + 10, w - 14)
  const H = rows.length * rowH + 26
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Held-out ROC-AUC by time cut, trained model against the selected baseline.">
        {[0.4, 0.5, 0.6, 0.7, 0.8].map((t) => (
          <g key={t}>
            {narrow
              ? rows.map((_, i) => {
                const cy = 34 + i * rowH
                return <line key={i} x1={x(t)} x2={x(t)} y1={cy - 9} y2={i === rows.length - 1 ? H - 22 : cy + rowH - 30} className={t === 0.5 ? 'chance' : 'grid'} />
              })
              : <line x1={x(t)} x2={x(t)} y1={0} y2={H - 22} className={t === 0.5 ? 'chance' : 'grid'} />}
            <text x={x(t)} y={H - 6} textAnchor="middle" className="ax-label">{t === 0.5 ? '0.5 chance' : t.toFixed(1)}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = (narrow ? 34 : 18) + i * rowH
          return (
            <g key={r.label}>
              <text x={0} y={narrow ? cy - 14 : cy + 1} className="sig halo">{r.label}{narrow && <tspan className="ax-label" dx={8}>{r.sub}</tspan>}</text>
              {!narrow && <text x={0} y={cy + 14} className="ax-label">{r.sub}</text>}
              <line x1={x(r.base)} x2={x(r.model)} y1={cy} y2={cy} className="bar" />
              <circle cx={x(r.base)} cy={cy} r={5} className="dot-pooled" style={{ transform: 'none' }} />
              <circle cx={x(r.model)} cy={cy} r={5} className="dot-model" />
            </g>
          )
        })}
      </svg>
    </div>
  )
}

/* ---------- 5. days to confirm vs effect size ---------- */

export function PowerCurve({ curve, leads, focus, windowDays, mdd, testFails }: {
  curve: { delta: number; days: number; units: number; fails: number }[]
  leads: { feature: string; delta: number; days: number; weak: boolean }[]
  focus: { feature: string; delta: number; days: number }
  windowDays: number; mdd: number; testFails: number
}) {
  const [ref, w] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip>(null)
  const narrow = w < 520
  const H = narrow ? 260 : 300
  const m = { l: 44, r: 16, t: 16, b: 34 }
  const x = lin(0.15, 0.65, m.l, w - m.r)
  const y = lin(0, 130, H - m.b, m.t)
  const path = monotonePath(curve.map((c) => [x(c.delta), y(c.days)] as [number, number]))
  const groups = useMemo(() => {
    const out: { px: number; py: number; names: string[] }[] = []
    for (const l of [...leads].sort((a, b) => a.delta - b.delta)) {
      const px = x(l.delta), py = y(l.days)
      const g = out.find((o) => Math.abs(o.px - px) < 16 && Math.abs(o.py - py) < 14)
      if (g) g.names.push(l.feature)
      else out.push({ px, py, names: [l.feature] })
    }
    return out
  }, [leads, w])
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={H} role="img" aria-label="Days of production needed to confirm a lead, by its effect size." onPointerLeave={() => setTip(null)}>
        {[0, 30, 60, 90, 120].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" className="ax-label">{t}d</text>
          </g>
        ))}
        {[0.2, 0.3, 0.4, 0.5, 0.6].map((t) => <text key={t} x={x(t)} y={H - 16} textAnchor="middle" className="ax-label">{t.toFixed(1)}</text>)}
        <text x={w - m.r} y={H - 2} textAnchor="end" className="ax-label">effect size, Cliff's δ</text>
        <line x1={m.l} x2={w - m.r} y1={y(windowDays)} y2={y(windowDays)} className="window" />
        <text x={w - m.r} y={y(windowDays) - 5} textAnchor="end" className="ax-label">later-unit window ≈ {windowDays} days</text>
        <path d={path} className="curve" />
        {leads.map((l) => (
          <g key={l.feature} onPointerEnter={() => setTip({ x: x(l.delta), y: y(l.days), body: <><b>{l.feature}</b><span>training δ {n2(l.delta)}</span><span>{n2(l.days, 1)} days to confirm</span>{l.weak && <span>test window already argues against it</span>}</> })}>
            <circle cx={x(l.delta)} cy={y(l.days)} r={10} className="hit" />
            <circle cx={x(l.delta)} cy={y(l.days)} r={4.5} className={l.weak ? 'dot-pooled' : 'dot-within'} />
          </g>
        ))}
        {!narrow && groups.map((g) => {
          const below = g.py > y(windowDays) - 30
          return (
            <text key={g.names.join()} x={g.px + 9} y={below ? g.py + 17 : g.py - 9} className="ax-label halo" pointerEvents="none">
              {g.names.join(' · ')}
            </text>
          )
        })}
        <g onPointerEnter={() => setTip({ x: x(focus.delta), y: y(focus.days), body: <><b>{focus.feature}</b><span>training δ {n2(focus.delta)}</span><span>{n2(focus.days, 1)} days to confirm</span></> })}>
          <circle cx={x(focus.delta)} cy={y(focus.days)} r={12} className="hit" />
          <rect x={x(focus.delta) - 6} y={y(focus.days) - 6} width={12} height={12} className="focus-mark" />
          <text x={x(focus.delta) - 10} y={y(focus.days) + 20} textAnchor="end" className="ann fail-ink halo">{focus.feature}</text>
        </g>
        <line x1={x(mdd)} x2={x(mdd)} y1={m.t} y2={H - m.b} className="mdd" />
        <text x={x(mdd) + 4} y={m.t + 8} className="ax-label">{narrow ? `smallest δ visible` : `smallest δ the ${testFails} later fails can see`}</text>
      </svg>
      <TipBox tip={tip} width={w} />
    </div>
  )
}

/** Monotone cubic (Fritsch-Carlson) through points sorted by x: no overshoot between planned values. */
function monotonePath(pts: [number, number][]) {
  const n = pts.length
  if (n < 3) return pts.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)} ${py.toFixed(1)}`).join('')
  const dx = [], sl = [] as number[]
  for (let i = 0; i < n - 1; i++) { dx.push(pts[i + 1][0] - pts[i][0]); sl.push((pts[i + 1][1] - pts[i][1]) / dx[i]) }
  const t = [sl[0]]
  for (let i = 1; i < n - 1; i++) t.push(sl[i - 1] * sl[i] <= 0 ? 0 : (3 * (dx[i - 1] + dx[i])) / ((2 * dx[i] + dx[i - 1]) / sl[i - 1] + (dx[i] + 2 * dx[i - 1]) / sl[i]))
  t.push(sl[n - 2])
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`
  for (let i = 0; i < n - 1; i++) {
    const h = dx[i] / 3
    d += `C${(pts[i][0] + h).toFixed(1)} ${(pts[i][1] + t[i] * h).toFixed(1)} ${(pts[i + 1][0] - h).toFixed(1)} ${(pts[i + 1][1] - t[i + 1] * h).toFixed(1)} ${pts[i + 1][0].toFixed(1)} ${pts[i + 1][1].toFixed(1)}`
  }
  return d
}

/* ---------- step rail visibility ---------- */

export function useActiveStep(ids: string[]) {
  const [active, setActive] = useState(ids[0])
  useEffect(() => {
    const els = ids.map((id) => document.getElementById(id)).filter(Boolean) as HTMLElement[]
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) setActive(e.target.id)
    }, { rootMargin: '-40% 0px -55% 0px' })
    els.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [ids.join()])
  return active
}
