# Yield Root-Cause Lens

Ranks which of 590 production test signals separate failing units, checks every lead for false discovery
and against later units, and finds the date the line's yield shifted. It runs on real public line data.

**Live demo:** _pending deploy_

## The problem
When a production line's yield drops, the test logs are wide (hundreds of measurements), full of gaps and
heavily imbalanced (a few percent fails). A "top correlated signals" list looks convincing and sends engineers
to the line chasing leads that were noise or drift. An engineer needs a short list with honest statistics,
and the date the process changed.

## What it does
1. **When did yield move?** A least-squares change point and a CUSUM on the fail rate. On this data yield
   went from 85.5% to 95.6% on 21 Aug 2008 (+10.1 points, 95% CI 6.4–14.1, permutation p = 0.001).
2. **Which signals are worth walking to the line for?** Mann-Whitney U, Benjamini-Hochberg FDR and Cliff's delta on
   the first 70% of units, then a replication check on the last 30%. Of 444 signals, 22 pass FDR and **only 3 hold on
   later units**. The top-ranked signal flips direction on future units.
3. **Does a model predict future failures?** Time-ordered held-out test: L1 logistic ROC-AUC 0.55 and gradient boosting 0.53,
   against 0.70–0.71 under shuffled CV. The gap shows how much a random split overstates performance.

## How it works
- `pipeline/yrcl/stats.py`: U statistic, Cliff's delta, BH, CUSUM and the change point, each written out and unit-tested
  against hand calculations and SciPy.
- `pipeline/run_pipeline.py`: cleaning, screening, models (scikit-learn), bootstrap stability and permutation
  importance. It writes `public/results.json`.
- `tests/`: 19 pytest checks (hand calcs, library cross-checks, published CUSUM run lengths, consistency of the real-data results).
- `src/`: a static Vite + React page that reads the precomputed JSON. No backend, no API keys.
- [METHOD.md](METHOD.md): equations, assumptions, validation table and limitations.

## Run locally
```bash
# data (CC BY 4.0) into a separate folder
mkdir -p ../_downloads/secom && curl -L -o ../_downloads/secom/secom.zip \
  https://archive.ics.uci.edu/static/public/179/secom.zip && unzip -o ../_downloads/secom/secom.zip -d ../_downloads/secom
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
npm run pipeline     # writes public/results.json (about 1 min)
npm run test:py      # 19 tests
npm install && npm run dev
```

## Data
Real data from UCI SECOM: McCann, M. & Johnston, A. (2008). *SECOM*. UCI Machine Learning Repository.
https://doi.org/10.24432/C54305. Licensed CC BY 4.0. It has 1,567 units from a semiconductor line, 590 anonymised
measurements and pass/fail labels with timestamps. Here it stands in for an assembly line's test logs. No synthetic data.

---
Built by Ibi Raheel, mechanical engineer building AI tools. More work at [ibiraheel.com](https://ibiraheel.com).
