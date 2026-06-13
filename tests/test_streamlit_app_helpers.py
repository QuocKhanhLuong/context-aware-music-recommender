import pandas as pd

from src.app.streamlit_app import dataset_overview, display_recommendations, numeric_summary


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
    assert display.loc[0, "scenario_score"] == 0.123
    assert display.loc[0, "explanations"] == "calm audio; low tempo"
