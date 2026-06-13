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


def prepare_real_data(input_path: str | Path | None, output_path: str | Path, max_rows: int | None = None) -> Path:
    source = Path(input_path) if input_path else find_best_tracks_dataset(CANDIDATE_TRACK_DATASETS[1:])
    df = pd.read_csv(source, nrows=max_rows)
    prepared = prepare_tracks_dataframe(df)
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
    args = parser.parse_args()
    prepare_real_data(args.input, args.output, args.max_rows)


if __name__ == "__main__":
    main()
