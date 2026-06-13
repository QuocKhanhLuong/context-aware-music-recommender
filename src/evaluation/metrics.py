"""Evaluation metrics for classification and recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def diversity_at_k(recommendations: pd.DataFrame, k: int = 10, genre_column: str = "genre") -> float:
    top = recommendations.head(k)
    if top.empty or genre_column not in top:
        return 0.0
    return top[genre_column].nunique() / len(top)


def novelty_at_k(recommendations: pd.DataFrame, k: int = 10, popularity_column: str = "popularity") -> float:
    top = recommendations.head(k)
    if top.empty or popularity_column not in top:
        return 0.0
    popularity = pd.to_numeric(top[popularity_column], errors="coerce").fillna(0)
    if popularity.max() > 1:
        popularity = popularity / 100.0
    return float((1 - popularity.clip(0, 1)).mean())


def scenario_fit_at_k(recommendations: pd.DataFrame, k: int = 10, score_column: str = "scenario_score") -> float:
    top = recommendations.head(k)
    if top.empty or score_column not in top:
        return 0.0
    return float(pd.to_numeric(top[score_column], errors="coerce").fillna(0).clip(0, 1).mean())


def explainability_coverage_at_k(recommendations: pd.DataFrame, k: int = 10, explanation_column: str = "explanations") -> float:
    top = recommendations.head(k)
    if top.empty or explanation_column not in top:
        return 0.0
    covered = top[explanation_column].map(lambda value: bool(value) and value != "[]")
    return float(covered.mean())


def precision_at_k(recommendations: pd.DataFrame, relevant_ids: set[str], k: int = 10, id_column: str = "track_id") -> float:
    if not relevant_ids or id_column not in recommendations:
        return 0.0
    top_ids = set(recommendations.head(k)[id_column].astype(str))
    return len(top_ids & set(map(str, relevant_ids))) / max(1, k)


def recommendation_metrics(recommendations: pd.DataFrame, k: int = 10, relevant_ids: set[str] | None = None) -> dict[str, float]:
    metrics = {
        "diversity_at_k": diversity_at_k(recommendations, k),
        "novelty_at_k": novelty_at_k(recommendations, k),
        "scenario_fit_at_k": scenario_fit_at_k(recommendations, k),
        "explainability_coverage_at_k": explainability_coverage_at_k(recommendations, k),
    }
    if relevant_ids is not None:
        metrics["precision_at_k"] = precision_at_k(recommendations, relevant_ids, k)
    return metrics
