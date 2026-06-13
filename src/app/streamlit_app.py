"""Streamlit demo for context-aware music recommendation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.load_data import load_default_tracks_dataset, prepare_tracks_dataframe
from src.data.preprocess import preprocess_tracks
from src.features.audio_features import add_audio_helper_scores
from src.features.emotion_features import create_weak_scenario_label
from src.features.lyric_features import add_lyric_features
from src.features.pos_hmm_viterbi import add_pos_features
from src.models.baseline_recommender import ContentBasedRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.models.scenario_classifier import ScenarioClassifier
from src.models.scenario_ranker import ScenarioRanker
from src.evaluation.metrics import classification_metrics, recommendation_metrics
from src.visualization.eda_plots import chi_square_test, pearson_correlation_matrix


SAMPLE_PATH = Path("examples/sample_tracks.csv")
SCENARIOS = ["study", "gym", "sleep", "party", "sad_healing"]


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    output = preprocess_tracks(df)
    output = add_audio_helper_scores(output)
    output = add_lyric_features(output)
    output = add_pos_features(output)
    output = create_weak_scenario_label(output)
    return output


def _load_streamlit_dataset(max_rows: int):
    return load_default_tracks_dataset(max_rows=max_rows)


def main() -> None:
    import streamlit as st

    cached_prepare_dataset = st.cache_data(show_spinner="Preparing features")(_prepare_dataset)
    cached_load_dataset = st.cache_data(show_spinner="Loading dataset")(_load_streamlit_dataset)

    st.set_page_config(page_title="Context-Aware Music Recommender", layout="wide")
    st.title("Context-Aware Music Evaluation and Recommendation System")

    uploaded = st.sidebar.file_uploader("Load CSV", type=["csv"])
    max_rows = st.sidebar.number_input("Max rows to load", min_value=10, max_value=100000, value=5000, step=500)
    if uploaded is not None:
        raw_df = prepare_tracks_dataframe(pd.read_csv(uploaded, nrows=int(max_rows)))
        source_path = "uploaded CSV"
    else:
        raw_df, source_path = cached_load_dataset(max_rows=int(max_rows))
    st.sidebar.caption(f"Dataset: {source_path}")
    df = cached_prepare_dataset(raw_df)

    page = st.sidebar.radio(
        "Page",
        ["Dataset Overview", "EDA Dashboard", "Model Training & Evaluation", "Music Recommendation Demo"],
    )

    if page == "Dataset Overview":
        st.subheader("Dataset Overview")
        st.write({"rows": int(df.shape[0]), "columns": int(df.shape[1])})
        st.dataframe(df.head(20))
        st.write("Missing values")
        st.dataframe(df.isna().sum().rename("missing").reset_index())
        st.write("Numeric summary")
        st.dataframe(df.describe(numeric_only=True).T)

    elif page == "EDA Dashboard":
        st.subheader("EDA Dashboard")
        numeric = [column for column in ["tempo", "energy", "valence", "popularity"] if column in df]
        for column in numeric:
            st.bar_chart(df[column])
        if {"energy", "valence"}.issubset(df.columns):
            st.scatter_chart(df, x="energy", y="valence", color="scenario_label")
        st.write("Pearson correlation")
        st.dataframe(pearson_correlation_matrix(df))
        if {"genre", "scenario_label"}.issubset(df.columns):
            st.write("Genre counts")
            st.bar_chart(df["genre"].value_counts())
            st.write("Chi-square: genre vs scenario_label")
            st.write(chi_square_test(df, "genre", "scenario_label"))

    elif page == "Model Training & Evaluation":
        st.subheader("Model Training & Evaluation")
        model_type = st.selectbox("Classifier", ["logistic_regression", "svm", "random_forest"])
        classifier = ScenarioClassifier(model_type=model_type, tune=False)
        if df["scenario_label"].nunique() > 1 and df["scenario_label"].value_counts().min() >= 2:
            metrics = classifier.evaluate_holdout(df)
            st.write(metrics)
        else:
            classifier.fit(df)
            predictions = classifier.predict(df)
            st.write("Sample-only fallback: reporting in-sample metrics because class counts are too small for a holdout split.")
            st.write(classification_metrics(df["scenario_label"], predictions))

    else:
        st.subheader("Music Recommendation Demo")
        scenario = st.selectbox("Scenario", SCENARIOS)
        model_choice = st.selectbox("Model", ["scenario_ranker", "baseline", "hybrid"])
        seed_text = st.text_input("Seed songs for baseline/hybrid", value="")
        top_k = st.slider("Top K", min_value=3, max_value=20, value=5)
        seed_songs = [name.strip() for name in seed_text.split(",") if name.strip()]

        if model_choice == "baseline":
            recommender = ContentBasedRecommender().fit(df)
            recommendations = recommender.recommend(seed_song_names=seed_songs, top_k=top_k)
        elif model_choice == "hybrid":
            classifier = ScenarioClassifier(model_type="logistic_regression", tune=False)
            recommender = HybridRecommender(classifier=classifier).fit(df)
            recommendations = recommender.recommend(scenario=scenario, seed_song_names=seed_songs, top_k=top_k)
        else:
            recommender = ScenarioRanker().fit(df)
            recommendations = recommender.recommend(scenario=scenario, top_k=top_k)
        st.dataframe(recommendations)
        st.write("Recommendation metrics")
        st.write(recommendation_metrics(recommendations, k=top_k))


if __name__ == "__main__":
    main()
