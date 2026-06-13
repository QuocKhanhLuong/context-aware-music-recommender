import pandas as pd

from src.data.load_data import normalize_column_names, validate_required_columns
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
