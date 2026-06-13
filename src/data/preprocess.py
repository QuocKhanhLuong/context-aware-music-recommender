"""Cleaning and preprocessing utilities."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


AUDIO_COLUMNS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "loudness",
    "popularity",
]


def clean_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            cleaned[column] = cleaned[column].fillna("")
    return cleaned


def detect_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for column in numeric_columns:
        if column not in df:
            continue
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags[f"{column}_outlier"] = (df[column] < lower) | (df[column] > upper)
    return flags


def cap_outliers_iqr(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    capped = df.copy()
    for column in numeric_columns:
        if column not in capped:
            continue
        q1 = capped[column].quantile(0.25)
        q3 = capped[column].quantile(0.75)
        iqr = q3 - q1
        capped[column] = capped[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return capped


def normalize_audio_features(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    normalized = df.copy()
    columns = [column for column in (columns or AUDIO_COLUMNS) if column in normalized.columns]
    if not columns:
        return normalized
    normalized[columns] = normalized[columns].apply(pd.to_numeric, errors="coerce")
    normalized[columns] = normalized[columns].fillna(normalized[columns].median(numeric_only=True))
    scaler = MinMaxScaler()
    normalized[columns] = scaler.fit_transform(normalized[columns])
    for column in columns:
        normalized[f"normalized_{column}"] = normalized[column]
    return normalized


def create_track_id_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "track_id" not in output.columns:
        base = output.get("track_name", pd.Series("", index=output.index)).astype(str)
        artist = output.get("artist", pd.Series("", index=output.index)).astype(str)
        output["track_id"] = (base + "::" + artist).map(lambda value: re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-"))
    return output


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    subset = [column for column in ["track_id", "track_name", "artist"] if column in df.columns]
    return df.drop_duplicates(subset=subset or None).reset_index(drop=True)


def preprocess_tracks(df: pd.DataFrame) -> pd.DataFrame:
    output = handle_missing_values(df)
    for column in ["track_name", "artist", "lyrics", "genre"]:
        if column in output:
            output[column] = output[column].map(clean_text if column == "lyrics" else lambda x: "" if pd.isna(x) else str(x).strip())
    output = create_track_id_if_missing(output)
    output = remove_duplicates(output)
    numeric = [column for column in AUDIO_COLUMNS if column in output.columns]
    output = cap_outliers_iqr(output, numeric)
    return normalize_audio_features(output, numeric)
