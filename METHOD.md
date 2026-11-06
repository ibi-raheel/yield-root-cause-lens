# Method

The pipeline turns a wide, gappy, imbalanced table of line test measurements into three answers:
when the line's yield moved, which signals separate failing units (and which of those leads survive
honest checks), and whether any model predicts failures on *future* units. Every number in the UI is
written by `pipeline/run_pipeline.py` into `public/results.json`. Nothing is hand-entered.

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

`pytest` runs 19 tests (`npm run test:py`).

## Results (measured)

- **Yield shift.** One change point at unit 352 (21 Aug 2008): yield 85.5% (95% CI 81.8–88.9) before and
  95.6% (94.5–96.8) after, a shift of +10.1 points (6.4–14.1), permutation p = 0.001. The upper CUSUM
  alarms on the batch starting 7 Aug, peaks around 19–21 Aug and then decays, which agrees with the
  least-squares date.
- **Signals.** Of 444 signals, 76 have raw p < 0.05 (22 would be expected by chance alone), 22 survive BH at q < 0.05,
  and only **3 of those 22 (S103, S510, S129) still separate fails from passes on the later 30% of units**.
- **The top-ranked lead is not real on future units.** S059 has the smallest train p (δ = 0.51, in the
  top 10 in 50/50 bootstraps), but on the test split δ = −0.05 (p = 0.68). Stability across bootstraps does not
  protect against drift over time. 12 of the 20 top signals have |ρ| > 0.3 with time, so part of what they
  "explain" is early-vs-late production.
- **Naive correlation list.** 8 of the 10 signals most correlated with failure over all units fail either the FDR check or the held-out check.
- **Missingness.** 8 missing-data patterns pass BH: for example, S072/S073/S345/S346 are missing on 46% of passing
  units but 24% of failing ones. They come in blocks, which points to logging or recipe changes rather than to a single sensor.
- **Prediction.** On the time-ordered test split, L1 logistic ROC-AUC = 0.55 (0.43–0.67), PR-AUC = 0.070
  (baseline 0.055); gradient boosting ROC-AUC = 0.53. Neither CI clearly excludes chance. The same models under shuffled CV
  score ROC-AUC 0.70 (L1) and 0.71 (boosting), in line with published SECOM numbers. **The gap of about 0.16 AUC is
  the optimism you get from ignoring time order.**

## Assumptions and limitations

- The labels are pass/fail at the end of line. The data has no defect codes, station IDs or BOM, so a
  "signal" is a candidate to inspect, not a confirmed cause.
- The measurements are anonymised (S000–S589), so the physical meaning of a signal cannot be named.
- The test split has only 26 fails, so held-out tests have low power. A signal failing replication is weak
  evidence that it is spurious; one passing is moderate evidence that it is real.
- The change-point model assumes a single step in mean. Gradual drift or several steps would need PELT or a Bayesian
  online method.
- The CUSUM uses the overall fail rate as its reference, so it describes this window retrospectively. It is not a
  Phase II monitor with limits set in advance.
- The same signal can pass FDR in train and still be confounded with time. The post-change re-test and ρ are
  reported so that a reader can see it, but they do not remove the confounding.
