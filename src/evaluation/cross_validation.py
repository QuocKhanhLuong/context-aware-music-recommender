"""Cross-validation helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score


def run_10_fold_cv(model, X, y, scoring: str = "f1_macro") -> dict[str, float]:
    folds = min(10, pd.Series(y).value_counts().min()) if len(set(y)) > 1 else 2
    cv = StratifiedKFold(n_splits=max(2, int(folds)), shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "folds": int(cv.n_splits)}


def compare_models_with_cv(models: dict[str, object], X, y, scoring: str = "f1_macro") -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        result = run_10_fold_cv(model, X, y, scoring=scoring)
        result["model"] = name
        rows.append(result)
    return pd.DataFrame(rows)[["model", "mean", "std", "folds"]]
