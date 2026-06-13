import pandas as pd

from src.data.load_data import load_default_tracks_dataset, normalize_column_names, prepare_tracks_dataframe, validate_required_columns
from src.data.preprocess import handle_missing_values, normalize_audio_features, remove_duplicates


def test_normalize_column_names_and_validation():
    df = normalize_column_names(pd.DataFrame({"Title": ["A"], "Artist Name": ["B"], "Lyric": ["hello"]}))
    assert {"track_name", "artist", "lyrics"}.issubset(df.columns)
    validate_required_columns(df)


def test_preprocess_missing_values_and_duplicates():
    df = pd.DataFrame({"track_name": ["A", "A"], "artist": ["B", "B"], "energy": [None, 0.5]})
    filled = handle_missing_values(df)
    assert filled["energy"].isna().sum() == 0
    assert len(remove_duplicates(filled)) == 1


def test_audio_features_are_normalized():
    df = pd.DataFrame({"tempo": [60, 180], "energy": [0.1, 0.9]})
    normalized = normalize_audio_features(df)
    assert normalized["tempo"].between(0, 1).all()
    assert normalized["energy"].between(0, 1).all()


def test_prepare_tracks_dataframe_maps_legacy_columns():
    df = pd.DataFrame({"artist_name": ["Artist"], "track_name": ["Song"], "genres": ["pop"], "track_pop": [88]})
    prepared = prepare_tracks_dataframe(df)
    assert prepared.loc[0, "artist"] == "Artist"
    assert prepared.loc[0, "genre"] == "pop"
    assert prepared.loc[0, "popularity"] == 88
    assert "lyrics" in prepared.columns


def test_prepare_tracks_dataframe_coalesces_duplicate_aliases():
    df = pd.DataFrame({"name": ["Wrong playlist name"], "track_name": ["Correct song"], "artist_name": ["Artist"]})
    prepared = prepare_tracks_dataframe(df)
    assert list(prepared.columns).count("track_name") == 1
    assert prepared.loc[0, "track_name"] == "Correct song"


def test_load_default_tracks_dataset_returns_available_dataset():
    df, path = load_default_tracks_dataset(max_rows=2)
    assert len(df) == 2
    assert path.exists()
