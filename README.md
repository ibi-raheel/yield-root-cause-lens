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
   went from 85.5% to 95.6% on 21 Aug 2008 (+10.1 points, 95% CI 6.4–14.1, permutation p = 0.001): 145 failing
   units per 1,000 before, 44 after.
2. **What moved with it?** Every signal gets its own change point, and a circular-shift permutation test asks whether
   more signals step near the yield jump than timing chance would give. Five signals stepped within ±3 days (S103,
   S247, S519, S290, S059), which is about what chance gives on a line where nearly everything drifts (p = 0.56). That
   makes them a checklist for the change records, not a verdict. Logging coverage also shifted (8 of 11 missing-data
   families), but over weeks, not on the day.
3. **Which signals are worth walking to the line for?** A pooled screen (Mann-Whitney U, Benjamini-Hochberg FDR, Cliff's
   delta) finds 22 leads. Half the fails sit in the early, low-yield regime, so the leads are re-screened **within**
   each regime (a van Elteren-type stratified rank test, cross-checked by logistic regression with regime fixed effects).
   12 of the 22 turn out to be regime proxies, one is a real separator whose threshold drifted, and the within-regime
   screen surfaces **one lead (S336) that holds on the later 30% of units**, which the pooled screen had missed.
4. **How long to confirm the rest?** A power analysis (Noether's formula, checked by Monte Carlo on real values) gives
   a minimum detectable effect of δ = 0.29 with 26 test fails, so most "failed replications" are simply untestable.
   The six unconfirmed within-regime leads need 22–63 days of production each, and the week-1 plan in METHOD.md
   dates each confirm-or-drop decision.
5. **Does a model predict future failures?** Time-ordered held-out test: L1 logistic ROC-AUC 0.55 and gradient boosting 0.53,
   against 0.70–0.71 under shuffled CV. The gap shows how much a random split overstates performance.

## How it works
- `pipeline/yrcl/stats.py`: U statistic, Cliff's delta, BH, CUSUM and the change point, each written out and unit-tested
  against hand calculations and SciPy.
- `pipeline/run_pipeline.py`: cleaning, screening, models (scikit-learn), bootstrap stability and permutation
  importance. It writes `public/results.json`.
- `pipeline/yrcl/rootcause.py` + `pipeline/run_rootcause.py`: per-signal change points and the timing permutation test,
  the regime-stratified screen, power and sample size, failing units per 1,000 with a cost sweep, a Phase II CUSUM and
  individuals-chart limits.
- `tests/`: 31 pytest checks (hand calcs, library cross-checks, published CUSUM run lengths, planted-step recovery, a confounding simulation, Monte Carlo power, consistency of the real-data results).
- `src/`: a static Vite + React page that reads the precomputed JSON. No backend, no API keys.
- [METHOD.md](METHOD.md): equations, assumptions, validation table and limitations.

## Run locally
```bash
# data (CC BY 4.0) into a separate folder
mkdir -p ../_downloads/secom && curl -L -o ../_downloads/secom/secom.zip \
  https://archive.ics.uci.edu/static/public/179/secom.zip && unzip -o ../_downloads/secom/secom.zip -d ../_downloads/secom
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
npm run pipeline     # writes public/results.json (about 1.5 min)
npm run test:py      # 31 tests
npm install && npm run dev
```

## Data
Real data from UCI SECOM: McCann, M. & Johnston, A. (2008). *SECOM*. UCI Machine Learning Repository.
https://doi.org/10.24432/C54305. Licensed CC BY 4.0. It has 1,567 units from a semiconductor line, 590 anonymised
measurements and pass/fail labels with timestamps. Here it stands in for an assembly line's test logs. No synthetic data.

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
