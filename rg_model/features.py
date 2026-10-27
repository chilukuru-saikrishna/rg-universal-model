"""Turn raw player data into model inputs."""
import pandas as pd

from .config import COLS, EXCLUDE, col


def prepare(players: pd.DataFrame):
    """Return (X, y, info). Encodes text columns, drops excluded data points."""
    X = players[COLS].copy()
    y = players["outcome"].astype(int)
    X = X.drop(columns=[col(i) for i in EXCLUDE])

    yes_no = [c for c in X.columns
              if set(X[c].dropna().astype(str).unique()) <= {"Yes", "No"}]
    for c in yes_no:
        X[c] = (X[c] == "Yes").astype(int)

    days_se = col(61)                         # blank = never self-excluded
    X[days_se] = pd.to_numeric(X[days_se], errors="coerce").fillna(-1)

    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=cats, dtype=int)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    # XGBoost rejects [, ] and < in column names
    X.columns = [c.replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
    info = {"yes_no": yes_no, "categorical": cats, "n_inputs": X.shape[1]}
    return X, y, info
