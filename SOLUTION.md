# Solution: a lead-triage playbook

**Question.** When a line's yield moves, which of hundreds of test signals are worth walking to the line for, and
which are just the date the process changed, in disguise?

**Answer on the public data** (UCI SECOM, 1,567 units, 590 signals; every number below is in `public/results.json`).
Yield jumped 10.1 points on 21 Aug 2008 (95% CI 6.4 to 14.1, permutation p = 0.001), from 145 to 44 failing units
per 1,000. A standard screen flagged 22 signals; 12 of them were tracking that date, not failures. Comparing fails with
passes inside each production period removed them and found one lead (S336) that held up on later units, a lead the
standard screen had missed. Every remaining open lead gets a confirm-or-drop date (22 to 63 days).

## The playbook: "Yield moved. Before anyone walks to the line"

1. **Date it.** Fit one least-squares change point to the per-unit fail series and permutation-test it (999
   reorderings). Report the date, yield before and after with bootstrap CIs, and failing units per 1,000.
2. **Split the history at that date.** Make every later comparison inside a period, never across the date. Comparing
   across it is how the date sneaks in disguised as a root cause. (A further split inside a period is added only if
   its own permutation test gives p < 0.05. On this data none did.)
3. **Screen inside each period.** Use a stratified (van Elteren-type) rank test with Benjamini-Hochberg FDR across all
   screened signals. A lead that is significant only in the pooled screen is a calendar proxy until shown otherwise.
4. **Re-test on later units.** Hold back the last 30% of units in time, and re-test every lead one-sided in the
   direction training gave it.
5. **Put every lead in exactly one bin** (rules below).
6. **Monitor the new normal.** Run a Phase II CUSUM on fail rate, with the reference frozen from the first 200 units
   after the last known change (batches of 50, k = 0.5, h = 5). Put individuals charts on the walk-to-the-line leads,
   with limits from post-change units only.

## Decision rules

Every threshold is either a fixed convention or read from the measured results.

| Bin | Rule | On this data |
|---|---|---|
| **Walk to the line** | within-period BH q < 0.05, same direction as the screen, **and** one-sided p < 0.05 on later units | 1: S336 (q within 0.022, pooled q 0.29; later δ = 0.25, p = 0.018) |
| **Check the change records** | pooled q < 0.05 but not within-period; tracks time (\|Spearman ρ\| with unit order > 0.3) or passes shift by \|δ\| ≥ 0.33 between periods. Flag **re-test** if it still separates later fails (p < 0.05) by at least the minimum detectable effect | 12, of which 2 are re-test: S103 (later δ 0.43), S510 (0.34) |
| **Confirm or drop by day N** | within-period q < 0.05, not yet shown on later units. N = Noether sample size for its post-change effect, divided by units per day | 6: S197 day 22, S333 day 26, S021 day 42, S477 day 42, S205 day 44, S341 day 63 |
| (re-chart weekly) | within-period, but the passes themselves drifted by \|δ\| ≥ 0.33, so the threshold moved | 1: S059 |
| **Drop** | none of the above | 3 |

- **Minimum detectable effect.** With 26 later fails, the smallest effect the later-units test can see at 80% power is
  δ = 0.29. 16 of the 22 standard-screen leads sit below it, so a failed re-test is "untestable", not "disproved".
- **Why trust the within-period test.** On 500 *simulated* lines where a signal only trends with time, the standard
  test called it a cause 94.8% of the time and the stratified test 4.4%. That is a simulation result, not a real-data one.
- **Honest label.** S336 is a candidate, not a confirmed cause: nine within-period leads were re-tested, and BH across
  the nine gives q = 0.16 (δ CI 0.02 to 0.48).
- **Scope.** The bins cover the standard screen's leads plus any within-period lead that holds on later units. A
  within-period-only lead that has not yet held is printed on a watch line (one on this data: S469).
- **Dates are optimistic.** The confirm-or-drop dates assume the training effect is real. Training effects of selected leads
  run high, so S197 and S333, already weak on later units, should be expected to drop early.

## Run the tool

```bash
npm run triage                         # SECOM in ../_downloads/secom, writes triage.json
.venv/bin/python -I pipeline/triage.py --csv line_log.csv --split 0.7 --out triage.json
.venv/bin/python -I pipeline/triage.py --secom <dir> --with-lr   # add the logistic fixed-effects cross-check
```

**Input:** one row per unit with columns `ts` (timestamp), `fail` (1 = fail, 0 = pass) and one numeric column per test
signal. Gaps are allowed. Rows are sorted by `ts`.

**Output:** a printed table and `triage.json` with the change (date, yield before/after with CIs, failing units per
1,000 before/after, p), the minimum detectable effect, units per day, bin counts, and one row per lead:
`signal, bin, q_pooled, q_within, delta_later, p_later, days_to_confirm, note`.

The tool is a thin wrapper. It adds no statistics. It calls the same functions that write `public/results.json`
(`single_change_point`, `screen`, `stratified_mwu`, `benjamini_hochberg`, `classify_lead`, `min_detectable_delta`,
`noether_total_n`, `individuals_limits`). `tests/test_triage.py` checks that on SECOM every bin and every
confirm-by date equal the pipeline's results, and that the CSV loader gives the same answer as the SECOM loader on a
slice of real units.

## Week 1 on a line

One product, one station's test logs, about one quarter of history.

- **Day 1.** Pull the unit-level test log (timestamp, pass/fail, every measurement) and run triage. Out come the date
  yield moved, failing units per 1,000 before and after, and every lead in a bin.
- **Day 1–2.** With the line owner, match the change date against the line's change records for that window: ECOs,
  supplier or lot changes, fixture and firmware swaps. Signals that stepped within a few days of the jump become a
  checklist, not a verdict. On the public data 5 stepped within ±3 days, which is what timing chance gives
  (expected 4.97, p = 0.56).
- **Day 2–3.** Walk to the station for the walk-to-the-line leads. Put each on an individuals chart with limits from
  post-change units only, and tie it to the physical step it measures.
- **Day 3.** Move fail-rate monitoring to a Phase II CUSUM with the reference frozen from the first 200 units after
  the last known change, and set a batch-level investigate trigger.
- **Day 4.** Hand over the lead sheet: every open lead with its confirm-or-drop date, so nobody chases a lead the
  data cannot yet support.
- **Day 5.** Review with the team and leave this one-page playbook, so the next investigation starts at step 1.

Scope: the tool covers the test-log side. Images, BOM and repair data stay with the people on the line.
