"""Presentation-oriented Streamlit UI helpers."""

from __future__ import annotations

import html
from typing import Iterable

import pandas as pd


IMPORTANT_COLUMNS = [
    "track_name",
    "artist",
    "genre",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "scenario_label",
]

DISPLAY_RENAMES = {
    "rank": "Rank",
    "track_name": "Track",
    "artist": "Artist",
    "genre": "Genre",
    "scenario_label": "Scenario",
    "scenario_score": "Scenario Score",
    "final_score": "Final Score",
    "similarity_score": "Similarity",
    "classifier_probability": "Classifier Prob.",
    "energy": "Energy",
    "valence": "Valence",
    "tempo": "Tempo",
    "acousticness": "Acousticness",
    "popularity": "Popularity",
    "explanations": "Explanation",
}


def apply_dark_theme(st) -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp {
            background: #0E1117;
            color: #F8FAFC;
        }
        .main .block-container {
            max-width: 1320px;
            padding-top: 1.3rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: #111522;
            border-right: 1px solid #2A2F3A;
        }
        h1, h2, h3 {
            color: #F8FAFC;
            letter-spacing: 0;
        }
        .dashboard-title {
            font-size: 2.1rem;
            line-height: 1.15;
            font-weight: 780;
            color: #F8FAFC;
            margin-bottom: 0.2rem;
        }
        .dashboard-subtitle,
        .section-subtitle {
            color: #94A3B8;
            font-size: 0.98rem;
            margin-bottom: 1rem;
        }
        .kpi-card {
            background: linear-gradient(180deg, #1B1F2A 0%, #171B26 100%);
            border: 1px solid #2A2F3A;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            min-height: 94px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        }
        .kpi-label {
            color: #94A3B8;
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            color: #F8FAFC;
            font-size: 1.75rem;
            line-height: 1;
            font-weight: 780;
        }
        .kpi-help {
            color: #64748B;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }
        .section-title {
            margin-top: 1rem;
            margin-bottom: 0.15rem;
            font-size: 1.35rem;
            font-weight: 760;
            color: #F8FAFC;
        }
        .info-box {
            background: #171B26;
            border: 1px solid #2A2F3A;
            border-left: 4px solid #FF4B4B;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            color: #CBD5E1;
            margin-bottom: 1rem;
        }
        .rec-card {
            background: #171B26;
            border: 1px solid #2A2F3A;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .rec-title {
            color: #F8FAFC;
            font-weight: 750;
            font-size: 1.02rem;
        }
        .rec-meta {
            color: #94A3B8;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }
        .badge {
            display: inline-block;
            background: rgba(255, 75, 75, 0.14);
            color: #FCA5A5;
            border: 1px solid rgba(255, 75, 75, 0.35);
            border-radius: 999px;
            padding: 0.15rem 0.5rem;
            font-size: 0.78rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .pill {
            display: inline-block;
            background: #202636;
            color: #CBD5E1;
            border: 1px solid #2A2F3A;
            border-radius: 999px;
            padding: 0.15rem 0.48rem;
            font-size: 0.76rem;
            margin-right: 0.25rem;
            margin-bottom: 0.25rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #2A2F3A;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def has_columns(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(column in df.columns for column in columns)


def missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def safe_plot_message(st, missing: Iterable[str]) -> None:
    missing = list(missing)
    if missing:
        st.warning("Skipped this chart because required columns are missing: " + ", ".join(missing))


def sample_for_plotting(df: pd.DataFrame, max_rows: int = 10000, random_state: int = 42) -> tuple[pd.DataFrame, bool]:
    if len(df) <= max_rows:
        return df.copy(), False
    return df.sample(max_rows, random_state=random_state).copy(), True


def format_number(value: object, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    return str(value)


def render_metric_card(st, label: str, value: object, help_text: str | None = None) -> None:
    help_html = f'<div class="kpi-help">{html.escape(help_text)}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(str(value) if value not in [None, ""] else "N/A")}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(st, title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def render_info_box(st, text: str) -> None:
    st.markdown(f'<div class="info-box">{html.escape(text)}</div>', unsafe_allow_html=True)


def kpi_values(df: pd.DataFrame) -> dict[str, str]:
    total = len(df)
    missing_pct = None
    if df.size:
        missing_pct = float(df.isna().sum().sum() / df.size * 100)
    return {
        "Total Songs": f"{total:,}" if total else "N/A",
        "Unique Artists": f"{df['artist'].nunique():,}" if "artist" in df else "N/A",
        "Unique Genres": f"{df['genre'].nunique():,}" if "genre" in df else "N/A",
        "Missing Values %": f"{missing_pct:.2f}%" if missing_pct is not None else "N/A",
    }


def render_kpi_cards(st, df: pd.DataFrame) -> None:
    values = kpi_values(df)
    cols = st.columns(4)
    help_text = {
        "Total Songs": "Rows loaded after preprocessing",
        "Unique Artists": "Distinct artist values",
        "Unique Genres": "Distinct genre values",
        "Missing Values %": "Share of missing cells",
    }
    for column, (label, value) in zip(cols, values.items()):
        with column:
            render_metric_card(st, label, value, help_text[label])


def render_header(st, df: pd.DataFrame) -> None:
    st.markdown('<div class="dashboard-title">Context-Aware Music Evaluation and Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dashboard-subtitle">Offline scenario scoring and recommendation for study, gym, sleep, party, and sad-healing contexts.</div>',
        unsafe_allow_html=True,
    )
    render_kpi_cards(st, df)


def schema_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing_count": missing_count,
                "missing_pct": round(missing_count / max(1, len(df)) * 100, 3),
                "unique_count": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def important_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in IMPORTANT_COLUMNS if column in df.columns]


def format_table_for_display(df: pd.DataFrame, rename: bool = True) -> pd.DataFrame:
    output = df.copy()
    for column in output.select_dtypes(include="number").columns:
        output[column] = output[column].round(3)
    if "explanations" in output:
        output["explanations"] = output["explanations"].map(lambda value: "; ".join(value) if isinstance(value, list) else value)
    return output.rename(columns=DISPLAY_RENAMES) if rename else output


def score_column_config(st) -> dict:
    config = {}
    for label in ["Final Score", "Scenario Score", "Similarity", "Classifier Prob.", "Energy", "Valence", "Popularity"]:
        config[label] = st.column_config.NumberColumn(label, format="%.3f")
    config["Tempo"] = st.column_config.NumberColumn("Tempo", format="%.1f")
    return config


def truncate_label(value: object, max_len: int = 34) -> str:
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "..."
