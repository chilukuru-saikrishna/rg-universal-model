"""Tables and charts for the demo, saved to the outputs folder."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, precision_score, recall_score

from .config import col


def ensure(folder):
    os.makedirs(folder, exist_ok=True)
    return folder


def plot_comparison(comparison, probs, y_test, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in comparison["Model"].head(5).tolist() + ["Excel scorecard (M1)"]:
        RocCurveDisplay.from_predictions(y_test, probs[name], name=name, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("Top 5 models vs Excel scorecard")
    s = comparison.set_index("Model")["Test AUC"].sort_values()
    s.plot.barh(ax=axes[1], color=["#C00000" if m == "Excel scorecard (M1)" else "#2F5597" for m in s.index])
    axes[1].axvline(0.5, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Test AUC (0.5 = guessing)")
    axes[1].set_title("All models (red = Excel scorecard)")
    plt.tight_layout(); fig.savefig(f"{out}/1_model_comparison.png", dpi=160); plt.close(fig)


def plot_importance(dp_imp, out, top=20):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    t = dp_imp.head(top).iloc[::-1]
    axes[0].barh(t["RG ID"] + " " + t["Data point"], t["Share"], color="#7030A0")
    axes[0].set_title(f"Top {top} data points"); axes[0].set_xlabel("Share of importance")
    cat = dp_imp.groupby("Category")["Share"].sum().sort_values()
    axes[1].barh(cat.index, cat.values, color="#548235")
    axes[1].set_title("Importance by category"); axes[1].set_xlabel("Share of importance")
    plt.tight_layout(); fig.savefig(f"{out}/2_data_point_importance.png", dpi=160); plt.close(fig)


def tradeoff_table(y_test, p):
    rows = []
    for share in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        flag = (p >= np.quantile(p, 1 - share)).astype(int)
        rows.append({"Flag top %": f"{share:.0%}", "Reviews per 1,000 players": int(share * 1000),
                     "Cases caught (sensitivity)": recall_score(y_test, flag),
                     "Flags that are real (precision)": precision_score(y_test, flag)})
    return pd.DataFrame(rows)


def plot_tradeoff(tradeoff, out):
    ax = tradeoff.plot(x="Flag top %", y=["Cases caught (sensitivity)", "Flags that are real (precision)"],
                       marker="o", figsize=(7, 4), title="Catch more cases <-> review more players")
    ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(f"{out}/4_review_capacity_tradeoff.png", dpi=160); plt.close()


def plot_fewer(fewer, out):
    ax = fewer.plot(x="Top data points", y="Test AUC", marker="o", legend=False, figsize=(6, 4),
                    title="Accuracy vs number of data points used")
    ax.set_ylabel("Test AUC")
    plt.tight_layout(); plt.savefig(f"{out}/3_fewer_data_points.png", dpi=160); plt.close()


def excel_weights(dp_imp, X_train, top=15):
    t = dp_imp.head(top).copy()

    def threshold(row):
        c = col(int(row["RG ID"][3:]))
        if c not in X_train:
            return "see category"
        q = 0.8 if row["Direction"] == "higher = riskier" else 0.2
        return round(float(X_train[c].quantile(q)), 2)

    t["Rule"] = np.where(t["Direction"] == "higher = riskier", ">=", "<=")
    t["Suggested threshold"] = t.apply(threshold, axis=1)
    t["Suggested weight (1-3)"] = pd.cut(t["importance"], 3, labels=[1, 2, 3]).astype(int)
    return t[["RG ID", "Data point", "Category", "Rule", "Suggested threshold", "Suggested weight (1-3)", "Share"]]
