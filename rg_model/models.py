"""Model zoo and side-by-side comparison."""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def boosted_model():
    """LightGBM if installed, otherwise scikit-learn's equivalent histogram boosting."""
    if HAS_LGBM:
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE)


def get_models(quick: bool = False) -> dict:
    def scaled(m):
        return make_pipeline(StandardScaler(), m)

    models = {
        "Logistic regression": scaled(LogisticRegression(max_iter=3000)),
        "Decision tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra-Trees": ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "LightGBM" if HAS_LGBM else "Hist. gradient boosting": boosted_model(),
        "Neural network (MLP)": scaled(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                                     early_stopping=True, random_state=RANDOM_STATE)),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0)
    if not quick:
        models["SVM"] = scaled(SVC(probability=True, random_state=RANDOM_STATE))
    return models


def evaluate(name, y_true, score, cutoff) -> dict:
    pred = (np.asarray(score) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"Model": name, "Test AUC": roc_auc_score(y_true, score), "Accuracy": accuracy_score(y_true, pred),
            "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
            "Precision": precision_score(y_true, pred, zero_division=0), "F1": f1_score(y_true, pred)}


def compare(models, X_train, y_train, X_test, y_test, log=print):
    """Cross-validate, fit and test every model on the same split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, probs, fitted = [], {}, {}
    for name, m in models.items():
        t0 = time.time()
        cv_auc = cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        probs[name], fitted[name] = p, m
        row = evaluate(name, y_test, p, 0.5)
        row.update({"CV AUC": cv_auc.mean(), "CV sd": cv_auc.std(), "Seconds": time.time() - t0})
        rows.append(row)
        log(f"  {name:24s} AUC {row['Test AUC']:.3f}   ({row['Seconds']:.0f}s)")
    return rows, probs, fitted
