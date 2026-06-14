"""Streamlit dashboard for context-aware music recommendation."""

from __future__ import annotations

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.app.ui_components import (
    apply_dark_theme,
    format_table_for_display,
    get_numeric_columns,
    has_columns,
    important_columns,
    missing_columns,
    render_header,
    render_info_box,
    render_metric_card,
    render_section_title,
    safe_plot_message,
    sample_for_plotting,
    schema_dataframe,
    score_column_config,
    truncate_label,
)
from src.data.load_data import load_default_tracks_dataset, prepare_tracks_dataframe
from src.data.preprocess import preprocess_tracks
from src.evaluation.cross_validation import compare_scenario_models_cv
from src.evaluation.metrics import classification_metrics, recommendation_metrics
from src.features.audio_features import add_audio_helper_scores
from src.features.emotion_features import create_weak_scenario_label
from src.features.lyric_features import add_lyric_features
from src.features.pos_hmm_viterbi import add_pos_features
from src.models.baseline_recommender import ContentBasedRecommender
from src.models.hybrid_recommender import HybridRecommender
from src.models.scenario_classifier import DEFAULT_FEATURES, ScenarioClassifier
from src.models.scenario_ranker import ScenarioRanker
from src.visualization.eda_plots import scenario_parallel_coordinates_data


SCENARIOS = ["study", "gym", "sleep", "party", "sad_healing"]
SCENARIO_LABELS = {
    "study": "Study",
    "gym": "Gym",
    "sleep": "Sleep",
    "party": "Party",
    "sad_healing": "Sad healing",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
}
RECOMMENDATION_COLUMNS = [
    "rank",
    "track_name",
    "artist",
    "genre",
    "final_score",
    "scenario_score",
    "similarity_score",
    "classifier_probability",
    "energy",
    "valence",
    "tempo",
    "popularity",
    "explanations",
]
PLOT_FEATURES = ["tempo", "energy", "valence", "popularity", "acousticness"]
PROFILE_FEATURES = ["energy", "danceability", "valence", "acousticness", "speechiness", "popularity"]
CORRELATION_FEATURES = [
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "loudness",
    "scenario_score",
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
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T.round(3)


def dataset_overview(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(len(get_numeric_columns(df))),
        "scenario_count": int(df["scenario_label"].nunique()) if "scenario_label" in df else 0,
    }


def display_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in RECOMMENDATION_COLUMNS if column in recommendations.columns]
    output = recommendations[columns].copy()
    return format_table_for_display(output)


def _plot_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#171B26",
        font_color="#F8FAFC",
        margin=dict(l=20, r=20, t=45, b=20),
        legend_title_text="",
    )
    return fig


def _plot_sample_notice(st, sampled: bool) -> None:
    if sampled:
        st.caption("Charts use a sample for responsiveness.")


def _classification_report_dataframe(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "class": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1-score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )
    return pd.DataFrame(rows).round(3)


def train_model_for_app(df: pd.DataFrame, model_type: str, max_rows: int = 5000) -> dict[str, object]:
    if "scenario_label" not in df:
        raise ValueError("scenario_label is required for model training")
    model_df = df.dropna(subset=["scenario_label"]).copy()
    if len(model_df) > max_rows:
        model_df = model_df.sample(max_rows, random_state=42)
    if model_df["scenario_label"].nunique() < 2 or model_df["scenario_label"].value_counts().min() < 2:
        raise ValueError("Need at least two scenarios with at least two rows each for holdout evaluation")

    class_count = int(model_df["scenario_label"].nunique())
    test_rows = max(class_count, math.ceil(len(model_df) * 0.25))
    if len(model_df) - test_rows < class_count:
        raise ValueError("Not enough rows per scenario to create a stratified holdout split")
    train_df, test_df = train_test_split(model_df, test_size=test_rows, stratify=model_df["scenario_label"], random_state=42)
    classifier = ScenarioClassifier(model_type=model_type, tune=False)
    classifier.fit(train_df)
    predictions = classifier.predict(test_df)
    metrics = classification_metrics(test_df["scenario_label"], predictions)
    labels = sorted(pd.Series(test_df["scenario_label"]).unique())
    report_df = _classification_report_dataframe(test_df["scenario_label"], predictions)
    return {
        "classifier": classifier,
        "metrics": metrics,
        "report": report_df,
        "labels": labels,
        "test_df": test_df,
        "predictions": predictions,
    }


def _feature_importance_frame(classifier: ScenarioClassifier) -> pd.DataFrame:
    pipeline = classifier.pipeline.best_estimator_ if hasattr(classifier.pipeline, "best_estimator_") else classifier.pipeline
    estimator = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = DEFAULT_FEATURES
    clean_names = [str(name).replace("numeric__", "").replace("categorical__", "") for name in names]

    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame({"feature": clean_names, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False)
    if hasattr(estimator, "coef_"):
        coefs = estimator.coef_
        values = coefs[0] if coefs.ndim == 2 else coefs
        return pd.DataFrame({"feature": clean_names[: len(values)], "importance": values}).assign(abs_importance=lambda frame: frame["importance"].abs()).sort_values("abs_importance", ascending=False)
    return pd.DataFrame()


def render_dataset_overview(st, df: pd.DataFrame) -> None:
    render_section_title(st, "Dataset Overview", "Inspect schema, missing values, and descriptive statistics of the prepared music dataset.")
    preview_tab, schema_tab, missing_tab, stats_tab = st.tabs(["Preview", "Schema", "Missing Values", "Summary Statistics"])

    with preview_tab:
        default_columns = important_columns(df) or list(df.columns[:10])
        selected = st.multiselect("Columns to display", list(df.columns), default=default_columns)
        preview = format_table_for_display(df[selected].head(200) if selected else df.head(200), rename=False)
        st.dataframe(preview, use_container_width=True)

    with schema_tab:
        st.dataframe(schema_dataframe(df), use_container_width=True)

    with missing_tab:
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(25)
        if missing.empty:
            st.success("No missing values detected in the prepared dataset.")
        else:
            missing_df = missing.rename("missing_count").reset_index().rename(columns={"index": "column"})
            fig = px.bar(missing_df, x="missing_count", y="column", orientation="h", title="Top missing columns")
            st.plotly_chart(_plot_layout(fig), use_container_width=True)

    with stats_tab:
        summary = numeric_summary(df)
        if summary.empty:
            st.info("No numeric columns available.")
        else:
            st.dataframe(summary, use_container_width=True)


def render_eda_dashboard(st, df: pd.DataFrame) -> None:
    render_section_title(st, "EDA Dashboard", "Explore how audio, lyrics, popularity, and scenario labels relate to each other.")
    plot_df, sampled = sample_for_plotting(df)
    _plot_sample_notice(st, sampled)
    distributions, relationships, scenario_tab, genre_tab, correlation_tab = st.tabs(
        ["Distributions", "Relationships", "Scenario Insights", "Genre & Popularity", "Correlation"]
    )

    with distributions:
        available = [feature for feature in PLOT_FEATURES if feature in plot_df]
        if not available:
            safe_plot_message(st, PLOT_FEATURES)
        else:
            feature = st.selectbox("Feature", available)
            left, right = st.columns(2)
            with left:
                fig = px.histogram(plot_df, x=feature, nbins=40, title=f"{feature} distribution")
                st.plotly_chart(_plot_layout(fig), use_container_width=True)
            with right:
                fig = px.box(plot_df, y=feature, points=False, title=f"{feature} spread")
                st.plotly_chart(_plot_layout(fig), use_container_width=True)

    with relationships:
        if has_columns(plot_df, ["energy", "valence"]):
            fig = px.scatter(
                plot_df,
                x="energy",
                y="valence",
                color="scenario_label" if "scenario_label" in plot_df else None,
                hover_data=[column for column in ["track_name", "artist", "genre"] if column in plot_df],
                title="Energy vs Valence",
            )
            st.plotly_chart(_plot_layout(fig), use_container_width=True)
        else:
            safe_plot_message(st, missing_columns(plot_df, ["energy", "valence"]))

        if has_columns(plot_df, ["danceability", "popularity"]):
            color = "scenario_label" if "scenario_label" in plot_df else ("genre" if "genre" in plot_df else None)
            fig = px.scatter(
                plot_df,
                x="danceability",
                y="popularity",
                color=color,
                hover_data=[column for column in ["track_name", "artist", "genre"] if column in plot_df],
                title="Danceability vs Popularity",
            )
            st.plotly_chart(_plot_layout(fig), use_container_width=True)
        else:
            safe_plot_message(st, missing_columns(plot_df, ["danceability", "popularity"]))

    with scenario_tab:
        if "scenario_label" not in plot_df:
            safe_plot_message(st, ["scenario_label"])
        else:
            counts = plot_df["scenario_label"].value_counts().reset_index()
            counts.columns = ["scenario_label", "count"]
            fig = px.bar(counts, x="scenario_label", y="count", title="Scenario label counts")
            st.plotly_chart(_plot_layout(fig), use_container_width=True)

            box_features = [feature for feature in ["energy", "valence", "tempo", "acousticness"] if feature in plot_df]
            if box_features:
                box_feature = st.selectbox("Scenario boxplot feature", box_features)
                fig = px.box(plot_df, x="scenario_label", y=box_feature, color="scenario_label", title=f"{box_feature} by scenario")
                st.plotly_chart(_plot_layout(fig), use_container_width=True)
            else:
                safe_plot_message(st, ["energy, valence, tempo, or acousticness"])

            profile_features = [feature for feature in PROFILE_FEATURES if feature in plot_df]
            if profile_features:
                profile = plot_df.groupby("scenario_label")[profile_features].mean().reset_index()
                long_profile = profile.melt(id_vars="scenario_label", var_name="feature", value_name="mean")
                fig = px.bar(long_profile, x="feature", y="mean", color="scenario_label", barmode="group", title="Average scenario feature profile")
                st.plotly_chart(_plot_layout(fig), use_container_width=True)

            pc_data = scenario_parallel_coordinates_data(plot_df)
            pc_dims = [column for column in pc_data.columns if column != "scenario_label"]
            if len(pc_dims) >= 3 and "scenario_label" in pc_data:
                codes, uniques = pd.factorize(pc_data["scenario_label"])
                numeric = pc_data[pc_dims].apply(pd.to_numeric, errors="coerce")
                pc_plot = numeric.assign(scenario=codes).dropna()
                if not pc_plot.empty:
                    fig = px.parallel_coordinates(
                        pc_plot,
                        dimensions=pc_dims,
                        color="scenario",
                        color_continuous_scale=px.colors.diverging.Tealrose,
                        title="Scenario feature profiles (parallel coordinates)",
                    )
                    st.plotly_chart(_plot_layout(fig), use_container_width=True)
                    st.caption("Line color encodes scenario: " + ", ".join(f"{i} = {label}" for i, label in enumerate(uniques)))

    with genre_tab:
        if "genre" not in plot_df:
            safe_plot_message(st, ["genre"])
        else:
            genre_series = plot_df["genre"].fillna("unknown").astype(str)
            top_counts = genre_series.value_counts().head(15).rename_axis("genre").reset_index(name="count")
            top_counts["genre_short"] = top_counts["genre"].map(truncate_label)
            fig = px.bar(top_counts, x="count", y="genre_short", orientation="h", hover_data={"genre": True}, title="Top 15 genres")
            st.plotly_chart(_plot_layout(fig), use_container_width=True)

            if "popularity" in plot_df:
                genre_pop = plot_df.assign(genre_short=genre_series.map(truncate_label)).groupby(["genre", "genre_short"], as_index=False)["popularity"].mean()
                genre_pop = genre_pop.sort_values("popularity", ascending=False).head(15)
                fig = px.bar(genre_pop, x="popularity", y="genre_short", orientation="h", hover_data={"genre": True}, title="Average popularity by top genres")
                st.plotly_chart(_plot_layout(fig), use_container_width=True)

            fig = px.treemap(top_counts, path=["genre_short"], values="count", hover_data={"genre": True}, title="Genre distribution treemap")
            st.plotly_chart(_plot_layout(fig), use_container_width=True)

    with correlation_tab:
        columns = [column for column in CORRELATION_FEATURES if column in plot_df and pd.api.types.is_numeric_dtype(plot_df[column])]
        if len(columns) < 2:
            safe_plot_message(st, ["at least two numeric correlation features"])
        else:
            corr = plot_df[columns].corr(method="pearson")
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson correlation heatmap")
            st.plotly_chart(_plot_layout(fig), use_container_width=True)


def render_model_training_page(st, df: pd.DataFrame) -> None:
    render_section_title(st, "Model Training & Evaluation", "Train classifiers to predict listening scenarios and evaluate them using holdout metrics.")
    render_info_box(st, "This page evaluates whether the model can predict a song's suitable listening scenario from audio, lyrics, metadata, and engineered features.")
    model_type = st.selectbox("Classifier", list(MODEL_LABELS.keys()), format_func=lambda value: MODEL_LABELS[value])
    max_train_rows = st.slider("Maximum rows for training", 500, min(10000, max(500, len(df))), min(5000, max(500, len(df))), step=500)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model and evaluating holdout split..."):
            try:
                result = train_model_for_app(df, model_type=model_type, max_rows=max_train_rows)
            except Exception as exc:
                st.error(str(exc))
                return

        metrics = result["metrics"]
        cols = st.columns(6)
        for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score", "Micro F1", "Macro F1"], ["accuracy", "precision", "recall", "f1_score", "micro_f1", "macro_f1"]):
            with col:
                render_metric_card(st, label, f"{metrics[key]:.3f}")

        labels = result["labels"]
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
        fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Reds", title="Confusion matrix")
        st.plotly_chart(_plot_layout(fig), use_container_width=True)

        st.write("Classification report")
        st.dataframe(result["report"], use_container_width=True)

        importance = _feature_importance_frame(result["classifier"])
        if not importance.empty:
            value_column = "abs_importance" if "abs_importance" in importance else "importance"
            top = importance.head(15)
            fig = px.bar(top, x=value_column, y="feature", orientation="h", title="Top feature signals")
            st.plotly_chart(_plot_layout(fig), use_container_width=True)

        with st.expander("Raw metrics"):
            st.write(metrics)
    else:
        st.info("Choose a classifier and click Train Model to evaluate a holdout split.")

    st.markdown("---")
    st.subheader("10-Fold Cross-Validation Comparison")
    render_info_box(st, "Compare classifiers with stratified 10-fold cross-validation (macro F1). This is more reliable than a single holdout split and is used to select the best model.")
    if st.button("Run cross-validation"):
        with st.spinner("Running 10-fold cross-validation for each classifier..."):
            try:
                cv_df = compare_scenario_models_cv(df.dropna(subset=["scenario_label"]))
            except Exception as exc:
                st.error(str(exc))
                return
        cv_display = cv_df.assign(
            **{"mean macro F1": cv_df["mean"].round(3), "std": cv_df["std"].round(3)}
        )[["model", "mean macro F1", "std", "folds"]]
        st.dataframe(cv_display, use_container_width=True)
        best = cv_df.iloc[0]
        st.success(f"Best model by mean macro F1: {best['model']} ({best['mean']:.3f} ± {best['std']:.3f}, {int(best['folds'])} folds)")


def _apply_recommendation_filters(st, df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    left, right, third = st.columns(3)
    with left:
        if "genre" in filtered:
            genres = ["All"] + sorted(filtered["genre"].dropna().astype(str).unique().tolist())[:300]
            selected_genre = st.selectbox("Genre filter", genres)
            if selected_genre != "All":
                filtered = filtered[filtered["genre"].astype(str) == selected_genre]
        else:
            st.caption("Genre filter unavailable.")
    with right:
        if "popularity" in filtered:
            min_popularity = st.slider("Minimum popularity", 0.0, 1.0, 0.0, 0.05)
            filtered = filtered[pd.to_numeric(filtered["popularity"], errors="coerce").fillna(0) >= min_popularity]
        else:
            st.caption("Popularity filter unavailable.")
    with third:
        if "explicit" in filtered:
            exclude_explicit = st.checkbox("Exclude explicit", value=False)
            if exclude_explicit:
                filtered = filtered[~filtered["explicit"].astype(bool)]
        else:
            st.caption("Explicit filter unavailable.")
    return filtered


def _render_recommendation_cards(st, recommendations: pd.DataFrame) -> None:
    for _, row in recommendations.iterrows():
        title = row.get("track_name", "Unknown track")
        artist = row.get("artist", "Unknown artist")
        rank = row.get("rank", "")
        scenario_score = row.get("scenario_score", row.get("final_score", None))
        st.markdown('<div class="rec-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-title">#{rank} {title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-meta">{artist} · {row.get("genre", "unknown")}</div>', unsafe_allow_html=True)
        if scenario_score is not None and not pd.isna(scenario_score):
            st.markdown(f'<span class="badge">Scenario fit {float(scenario_score):.3f}</span>', unsafe_allow_html=True)
        pills = []
        for feature in ["energy", "valence", "tempo", "acousticness"]:
            if feature in row and not pd.isna(row[feature]):
                value = float(row[feature])
                pills.append(f'<span class="pill">{feature}: {value:.3f}</span>')
        if pills:
            st.markdown("".join(pills), unsafe_allow_html=True)
        explanations = row.get("explanations", [])
        if isinstance(explanations, str):
            explanations = [part.strip() for part in explanations.split(";") if part.strip()]
        if explanations:
            st.markdown("\n".join(f"- {reason}" for reason in explanations[:3]))
        st.markdown("</div>", unsafe_allow_html=True)


def render_recommendation_demo(st, df: pd.DataFrame) -> None:
    render_section_title(st, "Music Recommendation Demo", "Rank songs by scenario fit, content similarity, and model probability.")
    filtered = _apply_recommendation_filters(st, df)
    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    left, right = st.columns(2)
    with left:
        scenario = st.selectbox("Scenario", SCENARIOS, format_func=lambda value: SCENARIO_LABELS[value])
        model_choice = st.selectbox("Model", ["scenario_ranker", "baseline", "hybrid"], format_func=lambda value: value.replace("_", " ").title())
        top_k = st.slider("Top K", min_value=3, max_value=20, value=5)
    with right:
        seed_options = sorted(filtered["track_name"].dropna().astype(str).unique().tolist())[:500] if "track_name" in filtered else []
        seed_songs = st.multiselect("Seed songs", seed_options)
        alpha = st.slider("Hybrid alpha", 0.0, 1.0, 0.3, 0.05)
        beta = st.slider("Hybrid beta", 0.0, 1.0, 0.4, 0.05)
        gamma = st.slider("Hybrid gamma", 0.0, 1.0, 0.3, 0.05)

    if model_choice == "baseline":
        recommender = ContentBasedRecommender().fit(filtered)
        recommendations = recommender.recommend(seed_song_names=seed_songs, top_k=top_k)
    elif model_choice == "hybrid":
        label_counts = filtered["scenario_label"].value_counts() if "scenario_label" in filtered else pd.Series(dtype=int)
        classifier = ScenarioClassifier(model_type="logistic_regression", tune=False) if len(label_counts) > 1 and label_counts.min() >= 2 else None
        if classifier is None:
            st.info("Classifier probability is skipped because the filtered dataset does not have enough scenario-label variety.")
        recommender = HybridRecommender(classifier=classifier, alpha=alpha, beta=beta, gamma=gamma).fit(filtered)
        recommendations = recommender.recommend(scenario=scenario, seed_song_names=seed_songs, top_k=top_k)
    else:
        recommender = ScenarioRanker().fit(filtered)
        recommendations = recommender.recommend(scenario=scenario, top_k=top_k)

    recommendations = recommendations.reset_index(drop=True)
    recommendations["rank"] = range(1, len(recommendations) + 1)
    table_tab, card_tab, metrics_tab = st.tabs(["Compact Table", "Card View", "Metrics"])
    with table_tab:
        st.dataframe(display_recommendations(recommendations), use_container_width=True, column_config=score_column_config(st))
    with card_tab:
        _render_recommendation_cards(st, recommendations)
    with metrics_tab:
        metric_values = recommendation_metrics(recommendations, k=top_k)
        cols = st.columns(len(metric_values))
        for col, (label, value) in zip(cols, metric_values.items()):
            with col:
                render_metric_card(st, label.replace("_", " ").title(), f"{value:.3f}")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Context-Aware Music Recommender", layout="wide")
    apply_dark_theme(st)

    cached_prepare_dataset = st.cache_data(show_spinner="Preparing features")(_prepare_dataset)
    cached_load_dataset = st.cache_data(show_spinner="Loading dataset")(_load_streamlit_dataset)

    st.sidebar.header("Dashboard Controls")
    uploaded = st.sidebar.file_uploader("Load CSV", type=["csv"])
    max_rows = st.sidebar.number_input("Max rows to load", min_value=10, max_value=100000, value=5000, step=500)
    if uploaded is not None:
        raw_df = prepare_tracks_dataframe(pd.read_csv(uploaded, nrows=int(max_rows)))
        source_path = "uploaded CSV"
    else:
        raw_df, source_path = cached_load_dataset(max_rows=int(max_rows))
    df = cached_prepare_dataset(raw_df)
    st.sidebar.caption(f"Dataset: {source_path}")
    st.sidebar.caption(f"Rows in memory: {len(df):,}")

    render_header(st, df)
    page = st.sidebar.radio(
        "Page",
        ["Dataset Overview", "EDA Dashboard", "Model Training & Evaluation", "Music Recommendation Demo"],
    )

    if page == "Dataset Overview":
        render_dataset_overview(st, df)
    elif page == "EDA Dashboard":
        render_eda_dashboard(st, df)
    elif page == "Model Training & Evaluation":
        render_model_training_page(st, df)
    else:
        render_recommendation_demo(st, df)


if __name__ == "__main__":
    main()
