"""Score individual players and explain why.

    python score_player.py                 # 3 example players: highest, middle, lowest risk
    python score_player.py --player P00042
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rg_model import config, data, features, scorecard


# Strong-evidence + regulator-specified data points (see RG_Data_Points_iGaming.xlsx)
EXPLAIN_IDS = [1, 9, 12, 15, 20, 21, 22, 23, 24, 25, 30, 33, 37, 38, 41, 45, 49, 57, 60, 65, 89]


def band(p):
    return "HIGH" if p >= 0.6 else "MEDIUM" if p >= 0.3 else "LOW"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", help="Player ID, e.g. P00042")
    ap.add_argument("--players", type=int, default=5000)
    ap.add_argument("--data", help="real data file instead of simulated")
    a = ap.parse_args()

    players = data.load_real(a.data) if a.data else data.simulate_players(a.players)
    X_all, y, _ = features.prepare(players)
    # A focused, research-backed set keeps the explanation readable (correlated inputs blur reasons)
    X = X_all[[config.col(i) for i in EXPLAIN_IDS]]
    ids = players["Player ID"]
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, ids, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)

    # Logistic regression: each reason is exactly (scaled value x weight), so it is fully explainable
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X_tr, y_tr)
    scaler, lr = model[0], model[-1]
    p_te = pd.Series(model.predict_proba(X_te)[:, 1], index=id_te.values)
    X_all_tr, X_all_te = X_all.loc[X_tr.index], X_all.loc[X_te.index]
    sc = pd.Series(scorecard.m1_score(X_all_te, X_all_tr).values, index=id_te.values)

    if a.player:
        chosen = [a.player]
    else:
        order = p_te.sort_values()
        chosen = [order.index[-1], order.index[len(order) // 2], order.index[0]]

    for pid in chosen:
        if pid not in p_te.index:
            print(f"\n{pid}: not in the held-out test set - try another ID from the test set")
            continue
        row = X_te[id_te.values == pid]
        contrib = pd.Series(scaler.transform(row)[0] * lr.coef_[0], index=X.columns)
        p = p_te[pid]
        actual = int(y_te[id_te.values == pid].iloc[0])
        print(f"\n{'=' * 70}\n{pid}   risk {p:.0%}  ->  {band(p)}"
              f"   (Excel scorecard {sc[pid]:.0f}/100, simulated outcome: {'RG case' if actual else 'not a case'})")
        print("  Raises risk:")
        for c, v in contrib.sort_values(ascending=False).head(5).items():
            if v > 0:
                print(f"    + {c:55s} = {row.iloc[0][c]:>10,.2f}   ({v:+.2f})")
        print("  Lowers risk:")
        for c, v in contrib.sort_values().head(3).items():
            if v < 0:
                print(f"    - {c:55s} = {row.iloc[0][c]:>10,.2f}   ({v:+.2f})")
    print("\nA person reviews every flag before any action is taken.")


if __name__ == "__main__":
    main()
