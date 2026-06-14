import pandas as pd

from src.data.integrate_data import (
    CHART_FEATURE_DEFAULTS,
    attach_chart_features,
    fill_missing_chart_features,
    merge_music_and_chart_data,
)


def _chart_df():
    return pd.DataFrame(
        {
            "year": [2018, 2019],
            "rank": [1, 5],
            "track_name": ["God's Plan", "Old Town Road"],
            "artist": ["Drake", "Lil Nas X"],
        }
    )


def test_attach_chart_features_matches_known_hit():
    tracks = pd.DataFrame(
        {
            "track_name": ["God's Plan", "Some Unknown Demo"],
            "artist": ["Drake", "Nobody"],
        }
    )
    merged, matched = attach_chart_features(tracks, _chart_df())
    assert matched == 1
    assert set(CHART_FEATURE_DEFAULTS).issubset(merged.columns)
    hit = merged.loc[merged["track_name"] == "God's Plan"].iloc[0]
    assert hit["appeared_in_billboard_year_end"] == 1
    assert hit["best_chart_rank"] == 1


def test_fill_missing_chart_features_has_no_nulls():
    tracks = pd.DataFrame({"track_name": ["Unmatched"], "artist": ["Nobody"]})
    merged = merge_music_and_chart_data(tracks, _chart_df())
    filled = fill_missing_chart_features(merged)
    for column, default in CHART_FEATURE_DEFAULTS.items():
        assert filled[column].notna().all()
        assert filled.iloc[0][column] == default
