import pandas as pd

from src.data.preprocess import preprocess_tracks
from src.features.audio_features import add_audio_helper_scores
from src.features.emotion_features import create_weak_scenario_label
from src.features.lyric_features import add_lyric_features
from src.features.pos_hmm_viterbi import add_pos_features
from src.models.baseline_recommender import ContentBasedRecommender
from src.models.hybrid_recommender import HybridRecommender


def _sample():
    df = pd.read_csv("examples/sample_tracks.csv")
    df = preprocess_tracks(df)
    df = add_audio_helper_scores(df)
    df = add_lyric_features(df)
    df = add_pos_features(df)
    return create_weak_scenario_label(df)


def test_content_based_recommender_returns_top_k_rows():
    df = _sample()
    recs = ContentBasedRecommender().fit(df).recommend(seed_song_names=["Library Rain"], top_k=4)
    assert len(recs) == 4
    assert "similarity_score" in recs


def test_hybrid_recommender_returns_scores_and_explanations():
    df = _sample()
    recs = HybridRecommender(classifier=None).fit(df).recommend("party", seed_song_names=["Midnight Dance"], top_k=3)
    assert len(recs) == 3
    assert {"final_score", "scenario_score", "similarity_score", "explanations"}.issubset(recs.columns)
    assert recs["explanations"].map(len).min() > 0
