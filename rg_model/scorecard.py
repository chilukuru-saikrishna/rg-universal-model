"""The Excel M1 scorecard rebuilt in Python, used as the baseline to beat."""
import pandas as pd

from .config import M1_SCORECARD, col


def m1_score(X: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """0-100 score = % of weighted warning signs triggered.
    Quantile thresholds come from `reference` (the training data) to avoid leakage."""
    total, weight_sum = 0, 0
    for rg, rule, threshold, weight in M1_SCORECARD:
        c = col(rg)
        t = reference[c].quantile(threshold[1]) if isinstance(threshold, tuple) else threshold
        hit = (X[c] >= t) if rule == ">=" else (X[c] <= t)
        total = total + hit.astype(int) * weight
        weight_sum += weight
    return 100 * total / weight_sum
