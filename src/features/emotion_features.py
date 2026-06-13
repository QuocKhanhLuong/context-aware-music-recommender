"""Mood and weak scenario label features."""

from __future__ import annotations

import pandas as pd


MOOD_TO_SCENARIO = {
    "energetic": "gym",
    "happy": "party",
    "calm": "study",
    "relaxed": "sleep",
    "sad": "sad_healing",
}


def encode_existing_mood_labels(df: pd.DataFrame, mood_column: str = "mood") -> pd.DataFrame:
    output = df.copy()
    if mood_column in output:
        output["mood_encoded"] = output[mood_column].astype("category").cat.codes
        output["scenario_label"] = output[mood_column].str.lower().map(MOOD_TO_SCENARIO).fillna(output.get("scenario_label", "study"))
    return output


def infer_mood_from_audio(row: pd.Series) -> str:
    valence = float(row.get("valence", 0.5))
    energy = float(row.get("energy", 0.5))
    danceability = float(row.get("danceability", 0.5))
    acousticness = float(row.get("acousticness", 0.0))
    tempo = float(row.get("tempo", 0.5))
    if high(energy) and high(danceability):
        return "energetic"
    if high(valence) and high(energy):
        return "happy"
    if high(acousticness) and low(tempo):
        return "calm"
    if low(valence) and low(energy):
        return "sad"
    return "calm"


def high(value: float, threshold: float = 0.65) -> bool:
    return value >= threshold


def low(value: float, threshold: float = 0.4) -> bool:
    return value <= threshold


def create_weak_scenario_label(df: pd.DataFrame) -> pd.DataFrame:
    output = encode_existing_mood_labels(df)
    if "scenario_label" in output and output["scenario_label"].notna().any():
        output["scenario_label"] = output["scenario_label"].fillna("study")
        return output
    output["inferred_mood"] = output.apply(infer_mood_from_audio, axis=1)
    output["scenario_label"] = output["inferred_mood"].map(MOOD_TO_SCENARIO).fillna("study")
    party_mask = (output.get("energy", 0) >= 0.65) & (output.get("danceability", 0) >= 0.65)
    sleep_mask = (output.get("acousticness", 0) >= 0.65) & (output.get("tempo", 1) <= 0.4)
    output.loc[party_mask, "scenario_label"] = "party"
    output.loc[sleep_mask, "scenario_label"] = "sleep"
    return output
