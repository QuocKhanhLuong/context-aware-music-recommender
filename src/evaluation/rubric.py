"""Scenario scoring rubric and explanations."""

from __future__ import annotations

import pandas as pd

from src.features.audio_features import add_audio_helper_scores
from src.features.lyric_features import add_lyric_features


SCENARIO_WEIGHTS = {
    "study": {
        "calm_audio_score": 0.22,
        "low_tempo_score": 0.16,
        "acoustic_score": 0.14,
        "low_emotional_intensity": 0.16,
        "instrumentalness": 0.12,
        "low_speechiness_score": 0.14,
        "noun_ratio": 0.06,
    },
    "gym": {
        "energetic_audio_score": 0.30,
        "high_tempo_score": 0.22,
        "dance_score": 0.18,
        "positive_sentiment": 0.15,
        "topic_motivation": 0.15,
    },
    "sleep": {
        "calm_audio_score": 0.25,
        "low_tempo_score": 0.22,
        "acoustic_score": 0.18,
        "low_loudness_score": 0.15,
        "low_speechiness_score": 0.12,
        "low_emotional_intensity": 0.08,
    },
    "party": {
        "dance_score": 0.25,
        "energetic_audio_score": 0.22,
        "high_valence_score": 0.17,
        "normalized_popularity": 0.14,
        "topic_party": 0.12,
        "appeared_in_billboard_year_end": 0.10,
    },
    "sad_healing": {
        "sad_sentiment": 0.22,
        "calm_audio_score": 0.20,
        "acoustic_score": 0.18,
        "low_energy_score": 0.16,
        "topic_breakup": 0.14,
        "topic_relaxing": 0.10,
    },
}


def get_scenario_weights(scenario: str) -> dict[str, float]:
    if scenario not in SCENARIO_WEIGHTS:
        raise ValueError(f"Unknown scenario: {scenario}")
    return SCENARIO_WEIGHTS[scenario]


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = add_audio_helper_scores(df)
    if "sentiment_score" not in output:
        output = add_lyric_features(output)
    defaults = {
        "appeared_in_billboard_year_end": 0.0,
        "normalized_popularity": output.get("popularity", pd.Series(0.0, index=output.index)),
        "instrumentalness": 0.0,
        "noun_ratio": 0.0,
        "adjective_ratio": 0.0,
    }
    for column, value in defaults.items():
        if column not in output:
            output[column] = value
    return output


def compute_scenario_score(df: pd.DataFrame, scenario: str) -> pd.Series:
    output = _ensure_feature_columns(df)
    weights = get_scenario_weights(scenario)
    score = pd.Series(0.0, index=output.index)
    weight_sum = sum(weights.values())
    for feature, weight in weights.items():
        values = pd.to_numeric(output.get(feature, 0.0), errors="coerce").fillna(0).clip(0, 1)
        score += values * weight
    return (score / weight_sum).clip(0, 1)


def explain_scenario_score(row: pd.Series, scenario: str, top_n: int = 3) -> list[str]:
    weights = get_scenario_weights(scenario)
    contributions = []
    for feature, weight in weights.items():
        value = float(pd.to_numeric(pd.Series([row.get(feature, 0.0)]), errors="coerce").fillna(0).iloc[0])
        contributions.append((feature, max(0.0, min(1.0, value)) * weight))
    top = sorted(contributions, key=lambda item: item[1], reverse=True)[:top_n]
    return [f"{feature.replace('_', ' ')} contributes {contribution:.2f}" for feature, contribution in top if contribution > 0]
