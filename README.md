# Stock Recommendations (XGBoost)

Recommends 5 stocks per user out of the full universe of publicly traded companies, using an XGBoost model over cross-sectional stock features.

## Overview

The universe is ~11,000 publicly traded companies organized into ~1,000 investment tracks (expanding toward ~1,500). Rather than modeling every stock's time series, the problem is framed as a **recommendation problem**: the model scores each stock from a snapshot of tabular features and returns the top 5 per user.

**V1 scope:** user constraints (budget, risk tolerance) are ignored. Constraint-aware personalization lives in the separate Ranking System project.

## How it works

1. Assemble a feature table — one row per company, columns for fundamentals, momentum, track membership, etc. (`n_stocks × n_features`).
2. Define the target label (e.g. forward return over a fixed horizon).
3. Train the XGBoost model on the feature table.
4. Score every stock and take the top 5 per user.
5. Backtest the top-5 output against held-out data and iterate on features.
6. Serve the top-5 output through the iPick backend.

## Tech stack

- **Model:** XGBoost (gradient-boosted trees)
- **Data / feature engineering:** Python, pandas, NumPy
- **Training & evaluation:** scikit-learn
- **Serving:** existing iPick backend

## Getting started

\`\`\`bash
# clone and enter the repo
git clone <repo-url>
cd stock-recommendations

# set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

> Fill in `requirements.txt` with at least: `xgboost`, `pandas`, `numpy`, `scikit-learn`.

## Suggested project structure

\`\`\`
stock-recommendations/
├── data/              # raw + processed feature tables
├── src/
│   ├── features.py    # build the (n_stocks × n_features) matrix
│   ├── train.py       # train the XGBoost model
│   ├── predict.py     # score stocks, return top 5
│   └── backtest.py    # evaluate top-5 picks on held-out data
├── models/            # saved model artifacts
├── requirements.txt
└── README.md
\`\`\`

## Roadmap

- [ ] Build the feature pipeline
- [ ] Define and validate the target label
- [ ] Train baseline XGBoost model
- [ ] Top-5 scoring + backtest harness
- [ ] Wire output into the iPick backend

## Resources

- XGBoost — Get Started (official docs): https://xgboost.readthedocs.io/en/stable/get_started.html
- StatQuest — XGBoost video series + "XGBoost in Python from Start to Finish": https://www.youtube.com/watch?v=GrJP9FLV3FE
- scikit-learn — Getting Started (official): https://scikit-learn.org/stable/getting_started.html
- pandas / NumPy — freeCodeCamp "Data Analysis with Python" full course: https://www.freecodecamp.org/news/how-to-analyze-data-with-python-pandas/