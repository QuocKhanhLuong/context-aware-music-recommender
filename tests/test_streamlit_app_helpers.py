import pandas as pd

from src.app.ui_components import kpi_values, sample_for_plotting, schema_dataframe
from src.app.streamlit_app import _prepare_dataset, dataset_overview, display_recommendations, numeric_summary, train_model_for_app


def test_numeric_summary_uses_selected_numeric_columns_only():
    df = pd.DataFrame({"track_name": ["A", "B"], "energy": [0.2, 0.8], "tempo": [80, 140]})
    summary = numeric_summary(df)
    assert list(summary.index) == ["energy", "tempo"]
    assert "mean" in summary.columns


def test_numeric_summary_handles_no_numeric_columns():
    df = pd.DataFrame({"track_name": ["A"], "artist": ["B"]})
    assert numeric_summary(df).empty


def test_dataset_overview_counts_rows_columns_and_scenarios():
    df = pd.DataFrame({"track_name": ["A", "B"], "energy": [0.2, 0.8], "scenario_label": ["study", "gym"]})
    overview = dataset_overview(df)
    assert overview == {"rows": 2, "columns": 3, "numeric_columns": 1, "scenario_count": 2}


def test_display_recommendations_formats_scores_and_explanations():
    recs = pd.DataFrame(
        {
            "track_name": ["A"],
            "artist": ["B"],
            "scenario_score": [0.12345],
            "explanations": [["calm audio", "low tempo"]],
            "unused": ["x"],
        }
    )
    display = display_recommendations(recs)
    assert "unused" not in display.columns
    assert display.loc[0, "Scenario Score"] == 0.123
    assert display.loc[0, "Explanation"] == "calm audio; low tempo"


def test_schema_dataframe_reports_missing_and_unique_counts():
    df = pd.DataFrame({"track_name": ["A", None], "energy": [0.2, 0.8]})
    schema = schema_dataframe(df)
    track_row = schema[schema["column"] == "track_name"].iloc[0]
    assert track_row["missing_count"] == 1
    assert track_row["unique_count"] == 1


def test_sample_for_plotting_caps_large_dataframes():
    df = pd.DataFrame({"x": range(20)})
    sampled, was_sampled = sample_for_plotting(df, max_rows=5)
    assert was_sampled is True
    assert len(sampled) == 5


def test_kpi_values_fallbacks_to_na_for_missing_columns():
    df = pd.DataFrame({"track_name": ["A"], "energy": [0.1]})
    values = kpi_values(df)
    assert values["Total Songs"] == "1"
    assert values["Unique Artists"] == "N/A"
    assert values["Unique Genres"] == "N/A"


def test_train_model_for_app_returns_metrics_and_report():
    raw = pd.read_csv("examples/sample_tracks.csv")
    df = _prepare_dataset(raw)
    result = train_model_for_app(df, "logistic_regression", max_rows=20)
    assert {"accuracy", "precision", "recall", "f1_score", "micro_f1", "macro_f1", "confusion_matrix"}.issubset(result["metrics"])
    assert {"class", "precision", "recall", "f1-score", "support"}.issubset(result["report"].columns)
