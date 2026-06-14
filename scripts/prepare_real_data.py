"""Prepare a real local music CSV for the Streamlit app.

This script does not download or commit large datasets. It normalizes an
existing CSV from `data/raw/` or the local legacy CSVs into
`data/processed/tracks_app_ready.csv`, which the app loads automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import CANDIDATE_TRACK_DATASETS, find_best_tracks_dataset, prepare_tracks_dataframe
from src.data.integrate_data import attach_chart_features

DEFAULT_CHART_DATA = ROOT / "examples" / "billboard_year_end_clean.csv"


def prepare_real_data(
    input_path: str | Path | None,
    output_path: str | Path,
    max_rows: int | None = None,
    chart_data: str | Path | None = None,
) -> Path:
    source = Path(input_path) if input_path else find_best_tracks_dataset(CANDIDATE_TRACK_DATASETS[1:])
    df = pd.read_csv(source, nrows=max_rows)
    prepared = prepare_tracks_dataframe(df)

    chart_path = Path(chart_data) if chart_data else DEFAULT_CHART_DATA
    if chart_path.exists():
        chart_df = pd.read_csv(chart_path)
        prepared, matched = attach_chart_features(prepared, chart_df)
        print(f"Chart data: {chart_path} ({len(chart_df)} rows, {matched} tracks matched)")
    else:
        print(f"Chart data: {chart_path} not found, skipping integration")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    print(f"Source: {source}")
    print(f"Rows: {len(prepared)}")
    print(f"Columns: {len(prepared.columns)}")
    print(f"Saved: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a real music dataset for the app.")
    parser.add_argument("--input", help="Raw CSV path. If omitted, auto-detects data/raw/*.csv or local legacy CSVs.")
    parser.add_argument("--output", default="data/processed/tracks_app_ready.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for faster classroom demos.")
    parser.add_argument("--chart-data", help="Scraped Billboard CSV. Defaults to examples/billboard_year_end_clean.csv.")
    args = parser.parse_args()
    prepare_real_data(args.input, args.output, args.max_rows, args.chart_data)


if __name__ == "__main__":
    main()
