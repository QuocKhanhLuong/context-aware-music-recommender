"""Load and validate offline music datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "track_name": "track_name",
    "name": "track_name",
    "song_name": "track_name",
    "title": "track_name",
    "artists": "artist",
    "artist": "artist",
    "artist_name": "artist",
    "lyrics": "lyrics",
    "lyric": "lyrics",
    "text": "lyrics",
    "genre": "genre",
    "genres": "genre",
    "popularity": "popularity",
    "track_popularity": "popularity",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, snake-case, and map common aliases to the project schema."""
    renamed = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        renamed[column] = COLUMN_ALIASES.get(normalized, normalized)
    return df.rename(columns=renamed)


def validate_required_columns(df: pd.DataFrame, required: tuple[str, ...] = ("track_name", "artist")) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _load_csv(path: str | Path, required: tuple[str, ...]) -> pd.DataFrame:
    df = normalize_column_names(pd.read_csv(path))
    validate_required_columns(df, required)
    return df


def load_tracks_dataset(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ("track_name", "artist"))


def load_mood_dataset(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ("track_name", "artist"))


def load_scraped_chart_dataset(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ("year", "rank", "track_name", "artist", "source_url", "scraped_at"))
