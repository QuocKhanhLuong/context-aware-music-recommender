"""Integrate base music data with scraped chart metadata."""

from __future__ import annotations

from difflib import SequenceMatcher
import re

import pandas as pd


def _key(value: object) -> str:
    value = "" if pd.isna(value) else str(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _match_score(track_a: str, artist_a: str, track_b: str, artist_b: str) -> float:
    track = SequenceMatcher(None, _key(track_a), _key(track_b)).ratio()
    artist = SequenceMatcher(None, _key(artist_a), _key(artist_b)).ratio()
    return 0.7 * track + 0.3 * artist


def fuzzy_match_track_artist(tracks_df: pd.DataFrame, chart_df: pd.DataFrame, threshold: float = 0.88) -> pd.DataFrame:
    """Attach chart rows using conservative title/artist fuzzy matching."""
    matches: list[dict[str, object]] = []
    chart_records = chart_df.to_dict("records")
    for track_idx, track in tracks_df.reset_index().iterrows():
        best_row: dict[str, object] | None = None
        best_score = 0.0
        for chart in chart_records:
            score = _match_score(track.get("track_name", ""), track.get("artist", ""), chart.get("track_name", ""), chart.get("artist", ""))
            if score > best_score:
                best_score = score
                best_row = chart
        if best_row and best_score >= threshold:
            match = {"track_index": track["index"], "match_score": best_score}
            match.update({f"chart_{key}": value for key, value in best_row.items()})
            matches.append(match)
    return pd.DataFrame(matches)


def merge_music_and_chart_data(tracks_df: pd.DataFrame, chart_df: pd.DataFrame) -> pd.DataFrame:
    """Merge exact normalized keys first, then chart features."""
    tracks = tracks_df.copy()
    chart = chart_df.copy()
    tracks["_track_key"] = tracks["track_name"].map(_key)
    tracks["_artist_key"] = tracks["artist"].map(_key)
    chart["_track_key"] = chart["track_name"].map(_key)
    chart["_artist_key"] = chart["artist"].map(_key)
    chart_features = create_chart_features(chart)
    merged = tracks.merge(chart_features, on=["_track_key", "_artist_key"], how="left")
    merged = merged.drop(columns=["_track_key", "_artist_key"])
    return merged


def create_chart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregate Billboard chart features for each title/artist key."""
    chart = df.copy()
    if "_track_key" not in chart:
        chart["_track_key"] = chart["track_name"].map(_key)
    if "_artist_key" not in chart:
        chart["_artist_key"] = chart["artist"].map(_key)
    chart["rank"] = pd.to_numeric(chart["rank"], errors="coerce")
    chart["year"] = pd.to_numeric(chart["year"], errors="coerce")
    grouped = chart.groupby(["_track_key", "_artist_key"], as_index=False).agg(
        appeared_in_billboard_year_end=("rank", lambda values: int(values.notna().any())),
        best_chart_rank=("rank", "min"),
        chart_year_count=("year", "nunique"),
        latest_chart_year=("year", "max"),
    )
    return grouped
