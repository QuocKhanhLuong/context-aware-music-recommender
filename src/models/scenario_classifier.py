"""Trainable scenario classifier wrappers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.evaluation.metrics import classification_metrics


DEFAULT_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "loudness",
    "popularity",
    "sentiment_score",
    "emotional_intensity_score",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "appeared_in_billboard_year_end",
    "best_chart_rank",
    "chart_year_count",
    "latest_chart_year",
    "genre",
]


class ScenarioClassifier:
    def __init__(self, model_type: str = "logistic_regression", features: list[str] | None = None, tune: bool = False):
        self.model_type = model_type
        self.features = features or DEFAULT_FEATURES
        self.tune = tune
        self.pipeline = None

    def _build_estimator(self):
        if self.model_type == "svm":
            return SVC(probability=True, class_weight="balanced")
        if self.model_type == "random_forest":
            return RandomForestClassifier(random_state=42, class_weight="balanced")
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    def _parameter_grid(self):
        if self.model_type == "svm":
            return {"classifier__C": [0.1, 1, 10]}
        if self.model_type == "random_forest":
            return {"classifier__n_estimators": [50, 100], "classifier__max_depth": [None, 5, 10]}
        return {"classifier__C": [0.1, 1, 10]}

    def _build_pipeline(self, X: pd.DataFrame):
        numeric_features = [column for column in self.features if column in X and pd.api.types.is_numeric_dtype(X[column])]
        categorical_features = [column for column in self.features if column in X and column not in numeric_features]
        preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ],
            remainder="drop",
        )
        return Pipeline([("preprocessor", preprocessor), ("classifier", self._build_estimator())])

    def fit(self, df: pd.DataFrame, label_column: str = "scenario_label"):
        X = df[[column for column in self.features if column in df]].copy()
        y = df[label_column]
        pipeline = self._build_pipeline(X)
        if self.tune and y.value_counts().min() >= 2:
            folds = min(10, int(y.value_counts().min()))
            self.pipeline = GridSearchCV(pipeline, self._parameter_grid(), cv=folds, scoring="f1_macro")
        else:
            self.pipeline = pipeline
        self.pipeline.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame):
        X = df[[column for column in self.features if column in df]].copy()
        return self.pipeline.predict(X)

    def predict_proba(self, df: pd.DataFrame):
        X = df[[column for column in self.features if column in df]].copy()
        estimator = self.pipeline.best_estimator_ if hasattr(self.pipeline, "best_estimator_") else self.pipeline
        return estimator.predict_proba(X)

    def evaluate_holdout(self, df: pd.DataFrame, label_column: str = "scenario_label", test_size: float = 0.25):
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df[label_column], random_state=42)
        self.fit(train_df, label_column=label_column)
        predictions = self.predict(test_df)
        return classification_metrics(test_df[label_column], predictions)

    def cross_validate(self, df: pd.DataFrame, label_column: str = "scenario_label", scoring: str = "f1_macro") -> dict[str, float]:
        """Run stratified (up to) 10-fold cross-validation on the full pipeline."""
        from src.evaluation.cross_validation import run_10_fold_cv

        X = df[[column for column in self.features if column in df]].copy()
        y = df[label_column]
        pipeline = self._build_pipeline(X)
        return run_10_fold_cv(pipeline, X, y, scoring=scoring)
