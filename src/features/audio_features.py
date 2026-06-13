"""Audio-feature normalization and scenario helper scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


AUDIO_FEATURES = [
    "tempo",
    "loudness",
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "popularity",
]


def normalize_audio_feature_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    output = df.copy()
    columns = [column for column in (columns or AUDIO_FEATURES) if column in output.columns]
    if not columns:
        return output
    output[columns] = output[columns].apply(pd.to_numeric, errors="coerce")
    output[columns] = output[columns].fillna(output[columns].median(numeric_only=True))
    output[columns] = MinMaxScaler().fit_transform(output[columns])
    for column in columns:
        output[f"normalized_{column}"] = output[column].clip(0, 1)
    return output


def _col(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df:
        return pd.to_numeric(df[column], errors="coerce").fillna(default).clip(0, 1)
    return pd.Series(default, index=df.index, dtype=float)


def add_audio_helper_scores(df: pd.DataFrame) -> pd.DataFrame:
    output = normalize_audio_feature_columns(df)
    tempo = _col(output, "normalized_tempo" if "normalized_tempo" in output else "tempo", 0.5)
    loudness = _col(output, "normalized_loudness" if "normalized_loudness" in output else "loudness", 0.5)
    energy = _col(output, "energy", 0.5)
    valence = _col(output, "valence", 0.5)
    danceability = _col(output, "danceability", 0.5)
    acousticness = _col(output, "acousticness", 0.5)
    speechiness = _col(output, "speechiness", 0.5)

    output["low_tempo_score"] = (1 - tempo).clip(0, 1)
    output["high_tempo_score"] = tempo.clip(0, 1)
    output["calm_audio_score"] = np.mean([1 - energy, 1 - tempo, acousticness], axis=0).clip(0, 1)
    output["energetic_audio_score"] = np.mean([energy, tempo, danceability], axis=0).clip(0, 1)
    output["dance_score"] = danceability.clip(0, 1)
    output["acoustic_score"] = acousticness.clip(0, 1)
    output["low_loudness_score"] = (1 - loudness).clip(0, 1)
    output["low_speechiness_score"] = (1 - speechiness).clip(0, 1)
    output["high_valence_score"] = valence.clip(0, 1)
    output["low_energy_score"] = (1 - energy).clip(0, 1)
    if "popularity" in output:
        output["normalized_popularity"] = _col(output, "normalized_popularity" if "normalized_popularity" in output else "popularity", 0.0)
    return output
