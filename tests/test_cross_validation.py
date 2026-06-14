import numpy as np
import pandas as pd

from src.evaluation.cross_validation import compare_scenario_models_cv, run_10_fold_cv
from src.models.scenario_classifier import DEFAULT_FEATURES, ScenarioClassifier


def _toy_features_df(per_class: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    numeric = [c for c in DEFAULT_FEATURES if c != "genre"]
    rows = []
    for label, base in [("study", 0.2), ("gym", 0.8)]:
        for _ in range(per_class):
            row = {c: float(base + rng.normal(0, 0.05)) for c in numeric}
            row["genre"] = "ambient" if label == "study" else "pop"
            row["scenario_label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def test_classifier_cross_validate_returns_scores():
    result = ScenarioClassifier(model_type="logistic_regression").cross_validate(_toy_features_df())
    assert set(result) == {"mean", "std", "folds"}
    assert 0.0 <= result["mean"] <= 1.0
    assert result["folds"] >= 2


def test_compare_scenario_models_cv_ranks_models():
    ranked = compare_scenario_models_cv(_toy_features_df())
    assert list(ranked.columns) == ["model", "mean", "std", "folds"]
    assert len(ranked) == 3
    # sorted descending by mean
    assert ranked["mean"].is_monotonic_decreasing


def test_run_10_fold_cv_adapts_folds_to_smallest_class():
    df = _toy_features_df(per_class=4)
    X = df[[c for c in DEFAULT_FEATURES if c in df]]
    pipeline = ScenarioClassifier()._build_pipeline(X)
    result = run_10_fold_cv(pipeline, X, df["scenario_label"])
    assert result["folds"] == 4
