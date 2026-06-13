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
SCENARIO_LABELS = {
    "study": "Study",
    "gym": "Gym",
    "sleep": "Sleep",
    "party": "Party",
    "sad_healing": "Sad healing",
}
DISPLAY_COLUMNS = [
    "track_name",
    "artist",
    "genre",
    "scenario_score",
    "final_score",
    "similarity_score",
    "classifier_probability",
    "energy",
    "tempo",
    "valence",
    "acousticness",
    "popularity",
    "explanations",
]


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    output = preprocess_tracks(df)
    output = add_audio_helper_scores(output)
    output = add_lyric_features(output)
    output = add_pos_features(output)
    output = create_weak_scenario_label(output)
    return output


def _load_streamlit_dataset(max_rows: int):
    return load_default_tracks_dataset(max_rows=max_rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric summary without relying on newer pandas describe args."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T


def dataset_overview(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(len(df.select_dtypes(include="number").columns)),
        "scenario_count": int(df["scenario_label"].nunique()) if "scenario_label" in df else 0,
    }


def display_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DISPLAY_COLUMNS if column in recommendations.columns]
    output = recommendations[columns].copy()
    for column in ["scenario_score", "final_score", "similarity_score", "classifier_probability"]:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce").round(3)
    if "explanations" in output:
        output["explanations"] = output["explanations"].map(lambda value: "; ".join(value) if isinstance(value, list) else value)
    return output


def apply_app_style(st) -> None:
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.5rem; max-width: 1280px; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 0.75rem 1rem;
            border-radius: 8px;
        }
        .section-caption { color: #64748b; font-size: 0.92rem; margin-top: -0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    import streamlit as st

    cached_prepare_dataset = st.cache_data(show_spinner="Preparing features")(_prepare_dataset)
    cached_load_dataset = st.cache_data(show_spinner="Loading dataset")(_load_streamlit_dataset)

    st.set_page_config(page_title="Context-Aware Music Recommender", layout="wide")
    apply_app_style(st)
    st.title("Context-Aware Music Evaluation and Recommendation System")
    st.caption("Offline scenario scoring and recommendation for study, gym, sleep, party, and sad-healing contexts.")

    st.sidebar.header("Controls")
    uploaded = st.sidebar.file_uploader("Load CSV", type=["csv"])
    max_rows = st.sidebar.number_input("Max rows to load", min_value=10, max_value=100000, value=5000, step=500)
    if uploaded is not None:
        raw_df = prepare_tracks_dataframe(pd.read_csv(uploaded, nrows=int(max_rows)))
        source_path = "uploaded CSV"
    else:
        raw_df, source_path = cached_load_dataset(max_rows=int(max_rows))
    st.sidebar.caption(f"Dataset: {source_path}")
    df = cached_prepare_dataset(raw_df)
    overview = dataset_overview(df)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{overview['rows']:,}")
    metric_cols[1].metric("Columns", f"{overview['columns']:,}")
    metric_cols[2].metric("Numeric", f"{overview['numeric_columns']:,}")
    metric_cols[3].metric("Scenarios", f"{overview['scenario_count']:,}")

    page = st.sidebar.radio(
        "Page",
        ["Dataset Overview", "EDA Dashboard", "Model Training & Evaluation", "Music Recommendation Demo"],
    )

    if page == "Dataset Overview":
        st.subheader("Dataset Overview")
        st.markdown('<p class="section-caption">Inspect schema, missing values, and numeric distributions for the active dataset.</p>', unsafe_allow_html=True)
        left, right = st.columns([2, 1])
        with left:
            st.dataframe(df.head(25), use_container_width=True)
        with right:
            st.write("Missing values")
            missing = df.isna().sum().rename("missing").reset_index().rename(columns={"index": "column"})
            st.dataframe(missing.sort_values("missing", ascending=False).head(30), use_container_width=True)
        st.write("Numeric summary")
        summary = numeric_summary(df)
        if summary.empty:
            st.info("No numeric columns available.")
        else:
            st.dataframe(summary, use_container_width=True)

    elif page == "EDA Dashboard":
        st.subheader("EDA Dashboard")
        st.markdown('<p class="section-caption">Quick visual checks for feature distributions and scenario relationships.</p>', unsafe_allow_html=True)
        numeric = [column for column in ["tempo", "energy", "valence", "popularity"] if column in df]
        chart_columns = st.columns(2)
        for index, column in enumerate(numeric):
            with chart_columns[index % 2]:
                st.write(column)
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
        st.markdown('<p class="section-caption">Train a classifier over the prepared scenario labels and inspect holdout metrics.</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="section-caption">Rank songs by context fit, optional content similarity, and model probability.</p>', unsafe_allow_html=True)
        scenario = st.selectbox("Scenario", SCENARIOS, format_func=lambda value: SCENARIO_LABELS[value])
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
        st.dataframe(display_recommendations(recommendations), use_container_width=True)
        st.write("Recommendation metrics")
        st.write(recommendation_metrics(recommendations, k=top_k))


if __name__ == "__main__":
    main()
