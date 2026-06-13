"""Reusable EDA chart helpers."""

from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency


def pearson_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes("number").corr(method="pearson")


def chi_square_test(df: pd.DataFrame, column_a: str, column_b: str) -> dict[str, float]:
    table = pd.crosstab(df[column_a], df[column_b])
    chi2, p_value, dof, _ = chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p_value), "degrees_of_freedom": int(dof)}


def distribution_data(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").dropna()


def genre_counts(df: pd.DataFrame, genre_column: str = "genre") -> pd.Series:
    return df[genre_column].fillna("unknown").value_counts()


def scenario_parallel_coordinates_data(df: pd.DataFrame, scenario_column: str = "scenario_label") -> pd.DataFrame:
    columns = [
        scenario_column,
        "tempo",
        "energy",
        "valence",
        "danceability",
        "acousticness",
        "popularity",
    ]
    return df[[column for column in columns if column in df]].copy()
