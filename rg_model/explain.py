"""Which data points matter: SHAP (if installed) or permutation importance."""
import numpy as np
import pandas as pd

from .config import META, RANDOM_STATE, rg_id

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def contributions(model, X_sample: pd.DataFrame, X_test=None, y_test=None):
    """Per-player, per-column contributions. Returns (values, method)."""
    if HAS_SHAP:
        sv = shap.TreeExplainer(model)(X_sample)
        vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
        return vals, "SHAP"
    # Fallback: permutation importance, spread with the sign of each column's correlation
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=3,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = np.clip(pi.importances_mean, 0, None)
    pred = model.predict_proba(X_sample)[:, 1]
    sign = np.array([np.sign(np.corrcoef(X_sample[c], pred)[0, 1]) if X_sample[c].std() > 0 else 0
                     for c in X_sample.columns])
    z = (X_sample - X_sample.mean()) / X_sample.std().replace(0, 1)
    return z.values * imp * sign, "permutation importance (install shap for SHAP)"


def data_point_importance(vals, X_sample: pd.DataFrame) -> pd.DataFrame:
    imp = pd.DataFrame({"column": X_sample.columns, "importance": np.abs(vals).mean(axis=0)})
    imp["RG ID"] = imp["column"].map(lambda c: f"RG-{rg_id(c):03d}")
    imp["Data point"] = imp["column"].map(lambda c: META[rg_id(c)][0])
    imp["Category"] = imp["column"].map(lambda c: META[rg_id(c)][1])
    imp["Direction"] = [
        ("higher = riskier" if np.corrcoef(X_sample[c], vals[:, k])[0, 1] > 0 else "lower = riskier")
        if X_sample[c].std() > 0 and np.std(vals[:, k]) > 0 else "-"
        for k, c in enumerate(X_sample.columns)]
    dp = (imp.groupby(["RG ID", "Data point", "Category"], as_index=False)
             .agg({"importance": "sum", "Direction": "first"})
             .sort_values("importance", ascending=False).reset_index(drop=True))
    dp["Share"] = dp["importance"] / dp["importance"].sum()
    dp["Cumulative"] = dp["Share"].cumsum()
    return dp
