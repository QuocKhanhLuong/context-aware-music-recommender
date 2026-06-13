import pandas as pd

from src.features.audio_features import add_audio_helper_scores


def test_audio_helper_scores_are_between_zero_and_one():
    df = pd.DataFrame({"tempo": [70, 150], "loudness": [-22, -4], "energy": [0.2, 0.9], "valence": [0.3, 0.8], "danceability": [0.4, 0.9], "acousticness": [0.8, 0.1], "speechiness": [0.04, 0.1]})
    scored = add_audio_helper_scores(df)
    score_columns = [column for column in scored.columns if column.endswith("_score")]
    assert score_columns
    for column in score_columns:
        assert scored[column].between(0, 1).all()
