import pandas as pd

from src.evaluation.metrics import classification_metrics, recommendation_metrics


def test_classification_metrics_return_expected_keys():
    metrics = classification_metrics(["study", "gym", "gym"], ["study", "study", "gym"])
    assert {"accuracy", "precision", "recall", "f1_score", "micro_f1", "macro_f1", "confusion_matrix"}.issubset(metrics)


def test_recommendation_metrics_return_expected_keys():
    recs = pd.DataFrame(
        {
            "track_id": ["a", "b"],
            "genre": ["ambient", "dance"],
            "popularity": [0.2, 0.8],
            "scenario_score": [0.9, 0.6],
            "explanations": [["calm"], ["dance"]],
        }
    )
    metrics = recommendation_metrics(recs, k=2, relevant_ids={"a"})
    assert {"diversity_at_k", "novelty_at_k", "scenario_fit_at_k", "explainability_coverage_at_k", "precision_at_k"}.issubset(metrics)
