"""Content-based recommender using cosine similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTENT_FEATURES = [
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
    "genre",
]


class ContentBasedRecommender:
    def __init__(self, features: list[str] | None = None):
        self.features = features or CONTENT_FEATURES
        self.df: pd.DataFrame | None = None
        self.preprocessor: ColumnTransformer | None = None
        self.matrix = None

    def fit(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        feature_df = self.df[[column for column in self.features if column in self.df]].copy()
        numeric = [column for column in feature_df if pd.api.types.is_numeric_dtype(feature_df[column])]
        categorical = [column for column in feature_df if column not in numeric]
        self.preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ],
            remainder="drop",
        )
        self.matrix = self.preprocessor.fit_transform(feature_df)
        return self

    def _seed_vector(self, seed_song_names: list[str] | None = None, playlist_df: pd.DataFrame | None = None):
        if self.df is None:
            raise ValueError("Call fit before recommend")
        if playlist_df is not None and not playlist_df.empty:
            seed_names = set(playlist_df["track_name"].str.lower())
        else:
            seed_names = {name.lower() for name in (seed_song_names or [])}
        if seed_names:
            mask = self.df["track_name"].str.lower().isin(seed_names)
            if mask.any():
                return self.matrix[mask].mean(axis=0), set(self.df.loc[mask, "track_name"].str.lower())
        return self.matrix.mean(axis=0), set()

    def recommend(self, seed_song_names: list[str] | None = None, playlist_df: pd.DataFrame | None = None, top_k: int = 10) -> pd.DataFrame:
        seed_vector, seed_names = self._seed_vector(seed_song_names, playlist_df)
        similarities = cosine_similarity(self.matrix, np.asarray(seed_vector).reshape(1, -1)).ravel()
        output = self.df.copy()
        output["similarity_score"] = similarities
        if seed_names:
            output = output[~output["track_name"].str.lower().isin(seed_names)]
        return output.sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True)
