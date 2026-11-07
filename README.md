# Yield Root-Cause Lens

Finds the date a production line's yield shifted and what moved with it, then ranks which of 590 test signals
separate failing units inside each production regime, checks every lead for false discovery and against later units,
and says how many days of data each remaining lead needs. It runs on real public line data.

**Live demo:** _pending deploy_

## The problem
When a production line's yield drops, the test logs are wide (hundreds of measurements), full of gaps and
heavily imbalanced (a few percent fails). A "top correlated signals" list looks convincing and sends engineers
to the line chasing leads that were noise or drift. An engineer needs a short list with honest statistics,
the date the process changed, and how long it will take to confirm each lead.

## What it does
1. **When did yield move?** A least-squares change point and a CUSUM on the fail rate. On this data yield
   went from 85.5% to 95.6% around 21 Aug 2008 (location 95% CI 6–21 Aug; +10.1 points, 95% CI 6.4–14.1,
   permutation p = 0.001): 145 failing units per 1,000 before, 44 after.
2. **What moved with it?** Every signal gets its own change point, and a circular-shift permutation test asks whether
   more signals step near the yield jump than timing chance would give. Five signals stepped within ±3 days (S103,
   S247, S519, S290, S059), which is about what chance gives on a line where nearly everything drifts (p = 0.56). That
   makes them a checklist for the change records, not a verdict. Logging coverage also shifted (8 of 11 missing-data
   families), but over 1–8 weeks, not on the day.
3. **Which signals are worth walking to the line for?** A pooled screen (Mann-Whitney U, Benjamini-Hochberg FDR, Cliff's
   delta) finds 22 leads. Half the fails sit in the early, low-yield regime, so the leads are re-screened **within**
   each regime (a van Elteren-type stratified rank test, cross-checked by logistic regression with regime fixed effects).
   In the training data 12 of the 22 behave like regime proxies (no effect inside either regime), although two of them
   (S103, S510) do separate fails in the later test window, so they stay open rather than cleared. One (S059) separated
   fails strongly after the change but not in the test window. The within-regime screen surfaces **one lead (S336) that
   also reaches p = 0.018 on the later 30% of units** (δ = 0.25, 95% CI 0.02–0.48), which the pooled screen had missed.
   With nine within-regime leads tested, that is a candidate, not a confirmation (BH q = 0.16 across the nine).
4. **How long to confirm the rest?** A power analysis (Noether's formula, checked by Monte Carlo on real values) gives
   a minimum detectable effect of δ = 0.29 with 26 test fails, so many "failed replications" are untestable rather
   than disproved. If their training effects are real, the six unconfirmed within-regime leads need 22–63 days of
   production each. Training effects of selected leads are optimistic, and the test window (470 units, about 27 days)
   already argues against two of them (S197 and S333: test δ −0.07, CI excluding their training effect). The week-1
   plan in METHOD.md dates each confirm-or-drop decision.
5. **Does a model predict future failures?** Time-ordered held-out test: L1 logistic ROC-AUC 0.55 and gradient boosting 0.53,
   against 0.70–0.71 under shuffled CV. The gap shows how much a random split overstates performance.
6. **Does a trained model earn its place?** Not clearly, on this data. Three trained models were each scored against a
   simple baseline on held-out time, with validation split by whole days and day-block bootstrap CIs on the difference:
   - **Regime-aware failure model (beats naive all-history training only; ties the simple baselines).** Trained only
     inside the post-change regime, the model reaches held-out ROC-AUC 0.64 vs 0.53 when the same model is trained on all
     history with no regime term (+0.11, 95% CI 0.03–0.20, averaged over three time cuts; +0.08 to +0.11 across solver
     seeds). That baseline is weak: the single best all-history signal alone scores 0.57. Give the all-history model one
     extra column, the regime indicator, and the gap shrinks to +0.06 (CI −0.02 to 0.11), which is not significant. That
     check was added at review, after the main result was known. The model also does **not** beat the single best
     within-regime signal (+0.06, CI −0.03 to 0.18). It does at the latest cut (0.72 vs 0.49, +0.23, CI 0.05–0.45), but
     that is one cut of three. Adding a fourth, earlier cut leaves the verdict unchanged (+0.10 vs all history, CI
     0.02–0.18; +0.04 vs the single signal, CI −0.05 to 0.11). Knowing the regime matters; no trained model here clearly
     beats a simple baseline that also knows it.
   - **Stability-selection root-cause ranker (loses).** L1 logistic over 100 day-blocked half-subsamples. Its top-10 signals
     replicate on later units less well than a plain Welch t ranking (mean signed held-out δ 0.016 vs 0.072, difference
     −0.056, CI −0.13 to 0.01) and a Mann-Whitney ranking (0.095). Fitting either ranker inside the regime helps more than
     going multivariate. Feeding each ranking's top 10 into the same small model does not change that (held-out ROC-AUC
     0.51 vs 0.55).
   - **Isolation Forest change detector (loses; neither detector works).** Both detectors are calibrated to the same
     false-alarm rate. The forest flags nothing until 15 Oct (+1,177 units after the 21 Aug change), because after the
     first 12 days units look more typical, not less. A Bernoulli CUSUM on pass/fail alarms on 5 Aug and again on 7 Aug,
     two weeks **before** the estimated change. The 7 Aug alarm counts as a "detection" only because it falls after the
     lower end of the change-date CI, which was itself estimated from all the data. The reference fail rate (21%) was
     unusually high, and in 99% of resamples the CUSUM alarms before the change, so it is reacting to "lower than the
     reference", not detecting the 21 Aug step.

## How it works
- `pipeline/yrcl/stats.py`: U statistic, Cliff's delta, BH, CUSUM and the change point, each written out and unit-tested
  against hand calculations and SciPy.
- `pipeline/run_pipeline.py`: cleaning, screening, models (scikit-learn), bootstrap stability and permutation
  importance. It writes `public/results.json`.
- `pipeline/yrcl/rootcause.py` + `pipeline/run_rootcause.py`: per-signal change points and the timing permutation test,
  the regime-stratified screen, power and sample size, failing units per 1,000 with a cost sweep, a Phase II CUSUM and
  individuals-chart limits.
- `pipeline/yrcl/ai.py` + `pipeline/run_ai.py`: stability selection, the regime-aware model, the Isolation Forest and
  Bernoulli CUSUMs with matched run lengths, and the day-block bootstrap on every AI-vs-baseline difference.
- `tests/`: 42 pytest checks (hand calcs, library cross-checks, published CUSUM run lengths, planted-step recovery, a confounding simulation, Monte Carlo power, consistency of the real-data results).
- `src/`: a static Vite + React page that reads the precomputed JSON. No backend, no API keys.
- [METHOD.md](METHOD.md): equations, assumptions, validation table and limitations.

## Run locally
```bash
# data (CC BY 4.0) into a separate folder
mkdir -p ../_downloads/secom && curl -L -o ../_downloads/secom/secom.zip \
  https://archive.ics.uci.edu/static/public/179/secom.zip && unzip -o ../_downloads/secom/secom.zip -d ../_downloads/secom
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
npm run pipeline     # writes public/results.json (about 3.5 min)
npm run test:py      # 42 tests
npm install && npm run dev
```

## Data
Real data from UCI SECOM: McCann, M. & Johnston, A. (2008). *SECOM*. UCI Machine Learning Repository.
https://doi.org/10.24432/C54305. Licensed CC BY 4.0. It has 1,567 units from a semiconductor line, 590 anonymised
measurements and pass/fail labels with timestamps. Here it stands in for an assembly line's test logs. The analysis
uses no synthetic data; simulated series appear only in the validation tests.

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
