# Method

The pipeline turns a wide, gappy, imbalanced table of line test measurements into three answers:
when the line's yield moved and what moved with it, which signals separate failing units inside a production regime
(and which of those leads survive honest checks), how much data it takes to confirm each lead, and whether any model
predicts failures on *future* units. Every number in the UI is
written by `pipeline/run_pipeline.py` and `pipeline/run_rootcause.py` into `public/results.json`. Nothing is hand-entered.

## Data

UCI SECOM (McCann, M. & Johnston, A. (2008). SECOM. UCI Machine Learning Repository,
https://doi.org/10.24432/C54305), licence CC BY 4.0. 1,567 production units from a semiconductor
line, 590 anonymised measurements each, a pass/fail label (104 fails) and a timestamp
(19 Jul to 17 Oct 2008). All rows are used. No synthetic data.

Semiconductor line data stands in here for an electronics assembly line's test logs. The structure
is the same: hundreds of anonymous measurements, about 4.5% of cells missing, and a fail rate of a few percent.

## 1. Cleaning

- Units sorted by timestamp.
- Train = first 70% of units in time (1,097 units, 78 fails); test = last 30% (470 units, 26 fails).
  Every choice (features kept, imputation, model hyperparameters, signal ranking) is made on the train split only.
- Dropped on train: 116 constant columns (zero variance) and 28 columns with more than 50% missing, which leaves 444.
- Model inputs: median imputation fitted on train, plus a 0/1 "was missing" column for every kept
  feature whose train missing rate is between 5% and 50% (28 columns).

## 2. Univariate screen

For each feature, with fails `a` (n_a) and passes `b` (n_b), NaNs dropped:

- **Mann-Whitney U.** Pool both samples, rank with average ranks for ties, and let R_a be the rank sum of the fails.
  `U_a = R_a − n_a(n_a+1)/2`. Under H0, `E[U] = n_a n_b / 2` and
  `Var[U] = n_a n_b/12 · ((n+1) − Σ(t³−t)/(n(n−1)))` (tie correction, t = size of each tie group).
  `z = (|U − E[U]| − 0.5)/√Var`, two-sided p = `2·(1 − Φ(z))`.
- **Cliff's delta** (effect size): `δ = P(a>b) − P(a<b) = 2U_a/(n_a n_b) − 1`, in [−1, 1].
- **Benjamini-Hochberg** across the m features tested: sort p ascending,
  `q_(i) = min_{j≥i} m·p_(j)/j`, capped at 1. A signal "passes FDR" when q < 0.05.
- **Held-out replication**: the same signal on the test split, one-sided MWU in the direction seen in
  train. It "holds" when p_test < 0.05.
- **Time confounding**: the Spearman ρ of each signal against unit order (train), and the one-sided
  MWU re-run on train units after the change point only.
- **Stability**: 50 bootstrap resamples of the train units; the share of resamples in which the signal
  ranks in the top 10 by p.
- **Missingness as a signal**: Fisher's exact test on the 2×2 table (missing / present × fail / pass),
  BH-corrected.

## 3. Change point and CUSUM

- **Least-squares single change** on the per-unit fail indicator y_t (time order):
  choose τ to maximise `G(τ) = τ(n−τ)/n · (ȳ_{<τ} − ȳ_{≥τ})²`, which is the drop in the sum of
  squared errors from fitting two means instead of one (minimum segment 50 units).
  Significance: permutation test with 999 random reorderings of y. Yield before and after: bootstrap
  95% CI (2,000 resamples). Location uncertainty: resample within segments, re-fit τ, and take the 2.5–97.5% quantiles (200 resamples).
- **Tabular CUSUM** on batches of 50 consecutive units, x_t = batch fail rate, reference
  μ0 = overall fail rate, σ = √(μ0(1−μ0)/50):
  `z_t = (x_t − μ0)/σ`, `S⁺_t = max(0, S⁺_{t−1} + z_t − k)`, `S⁻_t = max(0, S⁻_{t−1} − z_t − k)`,
  with k = 0.5 and h = 5 (alarm when S > h).

## 4. Models (time-ordered split)

- **L1 logistic regression** on standardised inputs, class-weighted. C picked from {0.01, 0.03, 0.1, 0.3, 1}
  by forward-chaining 3-fold CV inside train (PR-AUC). C = 0.1 was picked, with 135 of 472 columns non-zero.
- **Histogram gradient boosting**: depth 3, 300 iterations, learning rate 0.05, class-weighted.
- **Baselines**: majority class (PR-AUC = test fail rate) and the single best train feature, signed by its train direction.
- Metrics on the test split: ROC-AUC and PR-AUC with bootstrap 95% CIs (1,000 resamples),
  precision at the top k flagged units, and permutation importance (PR-AUC drop, 10 repeats) on test.
- **Optimistic reference**: the same models scored with shuffled, stratified 5-fold CV over all units,
  which ignores time. Published SECOM results are usually reported this way.

## 5. What changed with the yield step (`pipeline/run_rootcause.py`, section 1)

- **Per-signal step.** For each of the 444 kept signals, take the observed values in time order, replace them with
  their ranks, and fit the same least-squares single change as in §3 (minimum segment 50 observed units). The
  location is mapped back to a unit index. The step size is Cliff's δ of the values after vs before the step,
  computed from the rank sum: `δ = 2U_post/(n_pre n_post) − 1`.
- **Window.** The yield change-point location CI (6–21 Aug) widened by 50 units on each side: units 78–438, 3–23 Aug.
  A signal is "aligned" when its step falls in the window with |δ| ≥ 0.474 (the conventional "large" Cliff's δ).
- **Is the alignment more than chance?** Statistic = number of aligned signals. Null: circularly shift every signal by
  its own uniform random offset (999 times). This keeps each signal's distribution and autocorrelation and destroys
  only its timing relative to yield. `p = (1 + #{null ≥ observed})/1000`. A secondary, post-hoc check uses a narrow
  window of ±3 days around the yield step.
- **Location CI per signal.** Resample observed values within the pre and post segments, re-fit, 200 times.
- **Logging changes.** Columns with an identical missing-value pattern are grouped into families (11 families). For
  each one: the presence rate before vs after the yield step (Fisher's exact test, BH across families), and the
  family's own best presence step.
- **Caveat.** A pre/post test at a step location chosen to maximise the contrast is biased toward small p, so the
  per-signal step p-values (443 of 444 have q < 0.05) say only that almost every signal drifts. The permutation test on
  the timing is the honest test.
- **Validation.** A planted +3σ step at unit 300 is recovered exactly; with 30% of values missing, the step maps back
  to the first observed unit after the true step; a 0/1 logging switch is recovered within one unit; 15 planted
  aligned steps among 15 noise series give permutation p ≤ 0.02, and noise alone gives p > 0.05.

## 6. Regime-stratified screen (`run_rootcause.py`, section 2)

51 of the 104 fails sit in the first 352 units, so a pooled fail-vs-pass test partly rediscovers "early vs late".
The re-screen compares fails with passes **inside** each production regime only.

- **Regimes.** The yield change point (unit 352) splits the run in two. Binary segmentation then tries one more split
  inside each segment and keeps it only if its 999-permutation p < 0.05. Neither extra split passed (p = 0.29 before,
  p = 0.33 after), so there are 2 regimes: 14.5% and 4.4% fail.
- **Stratified rank test (van Elteren-type).** In regime k, compute the signed MWU z-score `z_k` (tie-corrected, no
  continuity correction) and weight `w_k = √(n_fail,k · n_pass,k / (n_k + 1))`. Then
  `Z = Σ w_k z_k / √(Σ w_k²)`, two-sided p, BH over the 444 signals. Run on the train split only.
- **Cross-check.** Logistic regression `logit P(fail) = α_regime + β · rank(signal)` (standardised rank), fitted by
  Newton-Raphson, with a likelihood-ratio test of β = 0 and BH.
- **Buckets for the 22 pooled FDR leads.**
  (a) within-regime separator (stratified q < 0.05, same direction) that also holds on the test split;
  (c) within-regime separator that does not hold on test and whose pass-unit level moves between regimes or from
  train to test by |δ| ≥ 0.33 ("drifting");
  (e) within-regime separator, not confirmed on test, no drift;
  (b) no within-regime effect, and |ρ_time| > 0.3 or the pass level moves between regimes by |δ| ≥ 0.33 ("regime proxy");
  (d) everything else ("noise").
  For each lead the per-regime Youden threshold (max TPR − FPR) and the fail and pass medians are exported, so drift is visible.
- **Validation (simulation, 500 runs).** A signal that is only a time trend plus noise, with the fail rate stepping from
  14.5% to 4.4% at unit 350 (no link to failure inside either regime). Pooled MWU: 94.8% false positives at α = 0.05.
  Stratified test: 4.4%. Logistic with regime fixed effects: 4.4%. With one stratum the stratified p equals
  `scipy.stats.mannwhitneyu(use_continuity=False)`, and the Newton-Raphson log-likelihood matches unpenalised
  scikit-learn to 1e-4.

## 7. Power, economics and the week-1 plan (`run_rootcause.py`, section 3)

- **Minimum detectable effect.** Under H0, `Var(U) = n_a n_b (n+1)/12`, so
  `SE₀(δ) = √((n_a + n_b + 1)/(3 n_a n_b))`. For a one-sided test at α = 0.05 with 80% power,
  `δ_min = (z₀.₉₅ + z₀.₈₀) · SE₀`. With 26 test fails and 444 passes: SE₀ = 0.117, **δ_min = 0.29**.
- **Sample size (Noether 1987).** `N = (z_α + z_β)² / (12 c(1−c)(p′ − ½)²)` with `p′ = (δ+1)/2` and c = fail
  fraction, i.e. `N = (z_α + z_β)² / (3 c(1−c) δ²)`. Units are converted to days at the observed throughput
  (1,567 units over 89.7 days = 17.5 units/day) and the post-change fail rate (4.36%). Each lead's planning effect
  is its δ inside the post-change regime of the train split.
- **Monte Carlo check.** Draw passes from a lead's real test-split pass values and fails from the same values shifted
  by the location that gives a set δ (bisection on a 4,000-draw resample), 1,000 runs per δ, one-sided MWU. On S336
  the empirical power is 0.07 / 0.51 / 0.84 / 0.99 at δ = 0 / 0.2 / 0.3 / 0.4 against 0.05 / 0.53 / 0.82 / 0.96 from
  the formula. On skewed synthetic data the formula is conservative by up to about 8 points, because the variance of U
  shrinks under a shift; the unit test bounds it to between −3 and +10 points.
- **Economics.** Failing units per 1,000 built before and after the yield step, with 2,000-resample bootstrap CIs.
  Money needs a cost per failed unit, which the data does not contain, so it is an **explicit assumption swept from
  $10 to $500**. The tornado varies one factor at a time around $100/unit: the fail-rate shift over its 95% CI, the
  throughput over the interquartile range of units per day (8–25), and the cost from $25 to $250.
- **Phase II CUSUM.** Reference fail rate fixed from the first 200 post-change units (5.0%); then the same tabular
  CUSUM (batches of 50, k = 0.5, h = 5) is run forward on the remaining 1,015 units with those limits frozen.
- **Individuals charts.** Centre = mean, limits = mean ± 3·MR̄/1.128, with MR̄ the mean moving range, set from
  post-change train units only and applied to the test split.

## Validation

| Check | Expected | Result |
|---|---|---|
| Yield by hand: 1463/1567 | 93.364% | 93.363% (pipeline, 3 dp) — test passes |
| U by hand from rank sums (tiny example, with and without ties) | U = 8, U = 5 | exact |
| U recomputed from rank sums for the top real signal | matches pipeline δ and p | matches to 1e-8 |
| Our MWU p vs `scipy.stats.mannwhitneyu` (20 random cases) | equal | rel. error < 1e-9 |
| Cliff's δ vs brute-force over all pairs | equal | exact |
| BH q vs a hand example and vs `scipy.stats.false_discovery_control` | equal | exact |
| CUSUM by hand on a known sequence | S = 0, 0.5, 1, 1.5, 2, 2.5; alarm at t = 5 | exact |
| CUSUM in-control ARL0 (k = 0.5, h = 5), published value ≈ 930 | 700–1200 | passes (simulated) |
| CUSUM ARL1 for a 1σ shift, published value 10.4 | ±0.6 | passes (simulated) |
| Change point on a clean step and on null data | τ exact; null p > 0.01 | passes |
| Models vs majority baseline (PR-AUC 0.055) | must beat it to claim prediction | do not beat it clearly (see results) |
| Planted step (+3σ at unit 300), with and without 30% gaps | exact unit | exact |
| Timing permutation: 15 planted aligned steps vs noise only | p ≤ 0.02 vs p > 0.05 | passes |
| Stratified test, one stratum, vs `scipy.stats.mannwhitneyu` | equal | rel. error < 1e-9 |
| Time trend + fail-rate step, 500 runs (no within-regime link) | stratified ≈ 5% false positives; pooled inflates | 4.4% vs 94.8% |
| Logistic log-likelihood vs unpenalised scikit-learn | equal | within 1e-4 |
| SE₀(δ), δ_min and Noether N by hand (26/444, δ = 0.3, c = 0.0436) | 0.1166, 0.290, 549.1 | exact |
| Monte Carlo power vs formula (S336 pass values, 1,000 runs per δ) | within a few points | within 3.4 points |
| Individuals chart limits by hand | MR̄ = 1.5, 3·MR̄/1.128 = 3.989 | exact |

`pytest` runs 31 tests (`npm run test:py`).

## Results (measured)

- **Yield shift.** One change point at unit 352 (21 Aug 2008): yield 85.5% (95% CI 81.8–88.9) before and
  95.6% (94.5–96.8) after, a shift of +10.1 points (6.4–14.1), permutation p = 0.001. In failing units per 1,000
  built: **145 before and 44 after, about 101 fewer (95% CI 63–142)**. The retrospective CUSUM agrees on the date.
- **What stepped with yield.** 16 signals took a large step (|δ| ≥ 0.474) between 3 and 23 Aug. Timing chance alone
  gives 11 (95th percentile 16), so the alignment p is 0.068: not significant. Within ±3 days of the yield step, five
  signals stepped: S103 (−0.8 days, median −0.006 → −0.011), S247 and S519 (+0.5 days, S519 median 11.4 → 4.7), S290
  (+1.1 days) and S059 (+1.8 days, median 8.7 → 0.0). That is also what chance gives (5.0 expected, p = 0.56). On this
  line almost every signal drifts, so **no signal's timing proves it caused the jump**. These five are the first
  things to check against the line's change records (recipe, tool, maintenance), not a conclusion.
- **Logging changed too, but gradually.** 8 of 11 missing-data families change their presence rate across the yield
  step (BH q < 0.05). The S112/S247/S385/S519 family drops from 78% to 48% present, S072/S073/S345/S346 from 80% to
  40%, and a 12-column family starting at S109 goes from 0% to 45%. But each family's own best step falls 7 to 31 days
  after the yield step, so measurement coverage drifted over weeks rather than switching on 21 Aug.
- **The pooled screen was confounded with the production regime.** Of the 22 pooled FDR leads:
  **12 are regime proxies**. That includes all three earlier "survivors": S103 (ρ_time = −0.64, stratified
  q = 0.995), S510 and S129. They separate early from late production, not fails from passes inside a regime.
  **1 is a drifting separator (S059)**: inside the post-change part of train, fails have a median of 4.62 and passes
  0.56 (Youden threshold 1.00), but by the test window the whole signal has moved (fails −0.98, passes −0.56). The
  lead is real, but its fail threshold moved. **6 separate fails within regime but are not confirmed** on the test split
  (S021, S197, S205, S333, S341, S477; within-regime δ 0.21–0.36). **3 are noise.**
- **Within-regime screen over all 444 signals.** 9 pass BH (q < 0.05), and the logistic fixed-effects test agrees on
  all 444. **One, S336, also holds on the later 30% of units** (test δ = 0.25, one-sided p = 0.018). S336 was not in
  the pooled list at all (pooled q = 0.29), because the regime contrast hid it.
- **Power.** With 26 test fails, the smallest δ the replication test can detect at 80% power is 0.29. 16 of the 22
  pooled leads have a train δ below that, so "fails replication" mostly means "untestable with this data", not
  "spurious". At 17.5 units/day and 4.4% fails, the six within-regime leads each need **22 to 63 days** of new
  production to confirm or drop. S059, with its threshold re-learned on current data, needs about 9 days.
- **Money (assumption-driven).** 101 fewer fails per 1,000 is about 640 fewer failing units a year at this line's
  rate (6,372 units/year). At an assumed $100 per failed unit that is about $65k a year. The tornado shows the cost
  assumption dominates ($16k–$161k for $25–$250/unit); throughput (IQR) gives $30k–$92k and the fail-rate CI $40k–$90k.
- **Phase II monitoring.** With limits frozen from the first 200 post-change units (5.0% fail), the forward CUSUM does
  not alarm. It peaks at S = 4.19 (h = 5) right after a 50-unit batch with 18% fails starting 4 Oct. That batch is the
  closest thing to a new excursion, and it is not significant on its own. Individuals charts on S336 and S059 flag 4 and
  1 test units, none of them failing, so these charts watch for drift and do not screen units.
- **Naive correlation list.** 8 of the 10 signals most correlated with failure over all units fail either the FDR check or the held-out check.
- **Prediction.** On the time-ordered test split, L1 logistic ROC-AUC = 0.55 (0.43–0.67), PR-AUC = 0.070
  (baseline 0.055); gradient boosting ROC-AUC = 0.53. Neither CI clearly excludes chance. The same models under shuffled CV
  score ROC-AUC 0.70 (L1) and 0.71 (boosting), in line with published SECOM numbers. **The gap of about 0.16 AUC is
  the optimism you get from ignoring time order.**

## Week-1 plan on a line like this

1. **Day 1.** Pull the line's change records for 18–24 Aug and match them against the five signals that stepped within
   ±3 days of the yield jump (S103, S247, S519, S290, S059) and the logging families that changed. Timing alone cannot
   separate these from chance, so the change records do the confirming.
2. **Day 1.** Put S336 (the only within-regime lead that replicates) and S059 (the drifting separator) on individuals
   charts, with limits from post-change data only. Re-learn S059's fail threshold weekly until it stops moving.
3. **Day 2.** Move the fail-rate CUSUM to Phase II: reference fixed from the first 200 units after the last known
   process change, batches of 50, k = 0.5, h = 5. Investigate any batch above 15% fails (the 4 Oct batch had 18%).
4. **Days 1–63.** Collect labelled units and re-test the six within-regime leads as data arrives. Confirm or drop each
   by the date the power plan gives: S197 by about day 22, S333 by day 26, S021, S205 and S477 by day 42–44, and S341 by day 63.
   Drop the 12 regime proxies now.
5. **Ongoing.** Report yield as failing units per 1,000, with the cost per failed unit agreed with the customer
   in place of the swept assumption.

## Assumptions and limitations

- The labels are pass/fail at the end of line. The data has no defect codes, station IDs or BOM, so a
  "signal" is a candidate to inspect, not a confirmed cause.
- The measurements are anonymised (S000–S589), so the physical meaning of a signal cannot be named.
- The test split has only 26 fails (minimum detectable δ = 0.29), so a signal failing replication is weak evidence
  that it is spurious; one passing is moderate evidence that it is real.
- Regimes come from the fail series. Binary segmentation found no second significant split, so slower drift inside a
  regime is not removed by stratification. ρ_time and the drift flags are reported for that reason.
- The timing test asks whether more signals step near the yield jump than chance would give. It cannot say which
  single signal caused the jump; that needs the line's change records.
- Cost per failed unit is not in the data. Every dollar figure is a labelled assumption, swept.
- The §3 CUSUM is retrospective; the Phase II CUSUM in §7 has its limits frozen in advance.
