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


def compare_scenario_models_cv(
    df: pd.DataFrame,
    model_types: tuple[str, ...] = ("logistic_regression", "svm", "random_forest"),
    label_column: str = "scenario_label",
    scoring: str = "f1_macro",
) -> pd.DataFrame:
    """Cross-validate each scenario classifier and rank them by mean CV score.

    Uses the same preprocessing pipeline as training, so the comparison reflects
    the real model and supports model/hyperparameter selection.
    """
    from src.models.scenario_classifier import ScenarioClassifier

    rows = []
    for model_type in model_types:
        result = ScenarioClassifier(model_type=model_type).cross_validate(
            df, label_column=label_column, scoring=scoring
        )
        result["model"] = model_type
        rows.append(result)
    ranked = pd.DataFrame(rows)[["model", "mean", "std", "folds"]]
    return ranked.sort_values("mean", ascending=False).reset_index(drop=True)
