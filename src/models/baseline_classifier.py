"""Baseline classifiers for scenario labels."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MajorityClassBaseline:
    def __init__(self):
        self.model = DummyClassifier(strategy="most_frequent")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LogisticRegressionBaseline:
    def __init__(self, max_iter: int = 1000):
        self.model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", LogisticRegression(max_iter=max_iter, class_weight="balanced")),
            ]
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
