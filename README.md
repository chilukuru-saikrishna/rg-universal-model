# Universal RG risk model — 89 responsible gambling data points

Plain-Python version of the Colab notebook: trains and compares machine-learning models for
early detection of gambling harm, benchmarks them against a rule-based scorecard, and shows
which data points matter.

> **Simulated data.** Players are generated with built-in patterns, so results demonstrate the
> method, not real-world accuracy (published AUCs on real data are roughly 0.6–0.84).

## Setup (once)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

If `lightgbm`, `xgboost` or `shap` fail to install, the code still runs: it falls back to
scikit-learn's histogram gradient boosting and permutation importance, and says so on screen.

## Run

```bash
python run_demo.py --quick --pause     # live demo: 3,000 players, Enter between steps (~40s compute)
python run_demo.py                     # full run: 8,000 players, includes SVM (a few minutes)
python score_player.py                 # score 3 players (high / middle / low) and explain why
python score_player.py --player P00042 # one specific player
```

Charts and CSVs are saved to `outputs/`.

## What the demo shows

| Step | Output |
|---|---|
| 1 | Player data: 89 data points, simulated or real (`--data file.csv`) |
| 2 | Input preparation: text encoded; PGSI and past RG outcomes excluded (leakage) |
| 3 | Baseline: the Excel M1 scorecard (14 weighted warning signs) |
| 4 | 9–10 models on the same split: AUC, CV AUC ± sd, accuracy, sensitivity, specificity, precision, F1 |
| 5 | Which data points and categories drive risk (SHAP or permutation importance) |
| 6 | Accuracy using only the top 5 / 10 / 15 / 20 / 30 data points |
| 7 | Review-capacity trade-off: flag top x% → cases caught vs flags that are real |
| 8 | Suggested evidence-based weights for the Excel scorecard |

## Project structure

```
run_demo.py          end-to-end pipeline (CLI)
score_player.py      per-player score + reasons (logistic regression on research-backed data points)
rg_model/
  config.py          the 89 data points, exclusions, scorecard definition
  data.py            simulated players / real-data loader / Excel column check
  features.py        encoding and leakage exclusions
  scorecard.py       Excel M1 scorecard baseline
  models.py          model zoo and comparison (LightGBM/XGBoost optional)
  explain.py         SHAP or permutation importance, per data point
  report.py          tables, charts, suggested Excel weights
```

## Using real data

Provide an approved, anonymised CSV/XLSX with the same 89 `RG-xxx ...` column names plus an
`outcome` column (1 = RG case), then `python run_demo.py --data file.csv`. Real use requires a
DPIA, anonymisation, and human review of every flag (GDPR Article 22).
