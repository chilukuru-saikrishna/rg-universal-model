"""Universal RG model demo - runs the whole pipeline end to end.

    python run_demo.py                 # full run, 8,000 simulated players
    python run_demo.py --quick         # faster: 3,000 players, skips SVM
    python run_demo.py --pause         # waits for Enter between steps (live demo)
    python run_demo.py --data my.csv   # approved, anonymised real data with the same columns
"""
import argparse
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from rg_model import config, data, explain, features, models, report, scorecard

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def main():
    ap = argparse.ArgumentParser(description="Universal RG risk model - 89 data points")
    ap.add_argument("--players", type=int, default=8000, help="simulated players (default 8000)")
    ap.add_argument("--quick", action="store_true", help="3,000 players, skip SVM")
    ap.add_argument("--data", help="real data file (CSV/XLSX) with the 89 RG columns + 'outcome'")
    ap.add_argument("--excel", default="RG_Universal_Model_100_Players.xlsx", help="Excel to check columns against")
    ap.add_argument("--out", default="outputs", help="folder for tables and charts")
    ap.add_argument("--pause", action="store_true", help="pause between steps for a live demo")
    a = ap.parse_args()
    out = report.ensure(a.out)
    t_start = time.time()

    def step(n, title):
        if a.pause and n > 1:
            input("\n  [Enter] for next step ...")
        print(f"\n{'=' * 78}\nSTEP {n}  {title}\n{'=' * 78}")

    # 1 ------------------------------------------------------------------ data
    step(1, "Player data - 89 responsible gambling data points")
    if a.data:
        players = data.load_real(a.data)
        print(f"  Loaded REAL data: {a.data}")
    else:
        n = 3000 if a.quick else a.players
        players = data.simulate_players(n)
        print("  SIMULATED players - results show the method, not real-world accuracy")
    print(f"  Players: {len(players):,}   RG cases: {players['outcome'].mean():.1%}")
    if os.path.exists(a.excel):
        print(f"  Excel check: {data.check_excel(a.excel)} of 89 RG columns found in {a.excel}")

    # 2 -------------------------------------------------------------- features
    step(2, "Prepare inputs (encode text, exclude leakage)")
    X, y, info = features.prepare(players)
    for rg, why in config.EXCLUDE.items():
        print(f"  Excluded RG-{rg:03d}: {why}")
    print(f"  Encoded text columns: {len(info['yes_no']) + len(info['categorical'])}")
    print(f"  Model inputs: {info['n_inputs']} columns from {89 - len(config.EXCLUDE)} data points")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)
    print(f"  Train: {len(X_train):,}   Test (held out): {len(X_test):,}")

    # 3 -------------------------------------------------------------- baseline
    step(3, "Baseline - the Excel M1 scorecard")
    sc = scorecard.m1_score(X_test, X_train)
    base = models.evaluate("Excel scorecard (M1)", y_test, sc.values, config.M1_HIGH_CUTOFF)
    print(f"  {len(config.M1_SCORECARD)} weighted warning signs, 'High' if score >= {config.M1_HIGH_CUTOFF}")
    print(f"  Scorecard AUC: {base['Test AUC']:.3f}")

    # 4 ---------------------------------------------------------------- models
    step(4, "Train and compare machine-learning models (same data, same split)")
    if not models.HAS_LGBM or not models.HAS_XGB:
        print("  Note: LightGBM/XGBoost not installed - using scikit-learn equivalents where needed")
    rows, probs, fitted = models.compare(models.get_models(a.quick), X_train, y_train, X_test, y_test)
    probs["Excel scorecard (M1)"] = sc.values
    rows.append(base)
    comparison = pd.DataFrame(rows).sort_values("Test AUC", ascending=False).reset_index(drop=True)
    print("\n" + comparison[["Model", "Test AUC", "CV AUC", "CV sd", "Accuracy", "Sensitivity",
                             "Specificity", "Precision", "F1"]].to_string(index=False))
    report.plot_comparison(comparison, probs, y_test, out)
    best = comparison.iloc[0]["Model"]
    lift = comparison.iloc[0]["Test AUC"] - base["Test AUC"]
    print(f"\n  Best: {best}  (AUC {comparison.iloc[0]['Test AUC']:.3f}, {lift:+.3f} vs Excel scorecard)")

    # 5 ------------------------------------------------------------ importance
    step(5, "Which of the 89 data points matter?")
    tree_name = "LightGBM" if models.HAS_LGBM else "Hist. gradient boosting"
    sample = X_test.sample(min(1500, len(X_test)), random_state=config.RANDOM_STATE)
    vals, method = explain.contributions(fitted[tree_name], sample, X_test, y_test)
    dp_imp = explain.data_point_importance(vals, sample)
    print(f"  Method: {method} on {tree_name}\n")
    print(dp_imp.head(15)[["RG ID", "Data point", "Category", "Direction", "Share", "Cumulative"]].to_string(index=False))
    n80 = int((dp_imp["Cumulative"] < 0.8).sum()) + 1
    print(f"\n  {n80} of {len(dp_imp)} data points explain 80% of the model's importance")
    report.plot_importance(dp_imp, out)

    # 6 ------------------------------------------------------- fewer features
    step(6, "Do we need all 89? Retrain on the top N only")
    res = []
    for top_n in [5, 10, 15, 20, 30, len(dp_imp)]:
        keep = {int(r[3:6]) for r in dp_imp["RG ID"].head(top_n)}
        cols = [c for c in X.columns if config.rg_id(c) in keep]
        m = models.boosted_model().fit(X_train[cols], y_train)
        res.append({"Top data points": top_n, "Columns": len(cols),
                    "Test AUC": models.roc_auc_score(y_test, m.predict_proba(X_test[cols])[:, 1])})
    fewer = pd.DataFrame(res)
    print(fewer.to_string(index=False))
    report.plot_fewer(fewer, out)

    # 7 -------------------------------------------------------------- tradeoff
    step(7, "Review capacity: how many players must the RG team review?")
    tradeoff = report.tradeoff_table(y_test, probs[best])
    print(f"  Model: {best}\n")
    print(tradeoff.to_string(index=False))
    report.plot_tradeoff(tradeoff, out)

    # 8 --------------------------------------------------------------- outputs
    step(8, "Suggested Excel scorecard weights + saved outputs")
    weights = report.excel_weights(dp_imp, X_train)
    print(weights.to_string(index=False))
    comparison.to_csv(f"{out}/model_comparison.csv", index=False)
    dp_imp.to_csv(f"{out}/data_point_importance.csv", index=False)
    fewer.to_csv(f"{out}/fewer_data_points.csv", index=False)
    tradeoff.to_csv(f"{out}/review_capacity_tradeoff.csv", index=False)
    weights.to_csv(f"{out}/suggested_excel_weights.csv", index=False)
    print(f"\n  Saved tables and charts to ./{out}/   (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
