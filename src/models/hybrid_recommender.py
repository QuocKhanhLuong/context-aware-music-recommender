"""Hybrid recommender combining similarity, scenario score, and classifier probability."""

from __future__ import annotations

import pandas as pd

from src.evaluation.rubric import compute_scenario_score, explain_scenario_score
from src.models.baseline_recommender import ContentBasedRecommender


class HybridRecommender:
    def __init__(self, classifier=None, alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.3):
        total = alpha + beta + gamma
        if total <= 0:
            raise ValueError("alpha + beta + gamma must be positive")
        self.alpha = alpha / total
        self.beta = beta / total
        self.gamma = gamma / total
        self.classifier = classifier
        self.content = ContentBasedRecommender()
        self.df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, label_column: str = "scenario_label"):
        self.df = df.reset_index(drop=True).copy()
        self.content.fit(self.df)
        if self.classifier is not None and label_column in self.df:
            self.classifier.fit(self.df, label_column=label_column)
        return self

    def _classifier_probability(self, df: pd.DataFrame, scenario: str) -> pd.Series:
        if self.classifier is None:
            return pd.Series(0.0, index=df.index)
        try:
            probabilities = self.classifier.predict_proba(df)
            estimator = self.classifier.pipeline.best_estimator_ if hasattr(self.classifier.pipeline, "best_estimator_") else self.classifier.pipeline
            classes = list(estimator.named_steps["classifier"].classes_)
            if scenario in classes:
                return pd.Series(probabilities[:, classes.index(scenario)], index=df.index)
        except Exception:
            pass
        return pd.Series(0.0, index=df.index)

    def recommend(self, scenario: str, seed_song_names: list[str] | None = None, top_k: int = 10) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Call fit before recommend")
        candidates = self.content.recommend(seed_song_names=seed_song_names, top_k=max(top_k * 4, top_k))
        candidates["scenario_score"] = compute_scenario_score(candidates, scenario)
        candidates["classifier_probability"] = self._classifier_probability(candidates, scenario)
        candidates["final_score"] = (
            self.alpha * candidates["similarity_score"]
            + self.beta * candidates["scenario_score"]
            + self.gamma * candidates["classifier_probability"]
        ).clip(0, 1)
        candidates["explanations"] = candidates.apply(lambda row: explain_scenario_score(row, scenario), axis=1)
        return candidates.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
