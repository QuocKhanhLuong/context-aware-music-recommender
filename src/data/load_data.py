"""Load and validate offline music datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_TRACK_DATASETS = [
    PROJECT_ROOT / "data/processed/tracks_app_ready.csv",
    PROJECT_ROOT / "data/raw/spotify_550k_tracks.csv",
    PROJECT_ROOT / "data/raw/tracks.csv",
    PROJECT_ROOT / "data/allsong_data.csv",
    PROJECT_ROOT / "data/processed_data.csv",
    PROJECT_ROOT / "examples/sample_tracks.csv",
]

COLUMN_ALIASES = {
    "track_name": "track_name",
    "name": "track_name",
    "song_name": "track_name",
    "title": "track_name",
    "artists": "artist",
    "artist": "artist",
    "artist_name": "artist",
    "lyrics": "lyrics",
    "lyric": "lyrics",
    "text": "lyrics",
    "genre": "genre",
    "genres": "genre",
    "popularity": "popularity",
    "track_popularity": "popularity",
    "track_pop": "popularity",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, snake-case, and map common aliases to the project schema."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        target = COLUMN_ALIASES.get(normalized, normalized)
        priority = 0 if normalized == target else 1
        groups.setdefault(target, []).append((priority, column))
    output = pd.DataFrame(index=df.index)
    for target, columns in groups.items():
        ordered = [column for _, column in sorted(columns, key=lambda item: item[0])]
        same_name = df.loc[:, ordered]
        merged = same_name.bfill(axis=1).iloc[:, 0]
        output[target] = merged
    return output


def validate_required_columns(df: pd.DataFrame, required: tuple[str, ...] = ("track_name", "artist")) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _load_csv(path: str | Path, required: tuple[str, ...]) -> pd.DataFrame:
    df = normalize_column_names(pd.read_csv(path))
    validate_required_columns(df, required)
    return df


def load_tracks_dataset(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ("track_name", "artist"))


def load_mood_dataset(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ("track_name", "artist"))


def load_scraped_chart_dataset(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, ("year", "rank", "track_name", "artist", "source_url", "scraped_at"))


def find_best_tracks_dataset(paths: list[str | Path] | None = None) -> Path:
    """Return the first loadable dataset path, preferring processed real data."""
    candidates = [_resolve_dataset_path(path) for path in (paths or CANDIDATE_TRACK_DATASETS)]
    skipped: list[str] = []
    for path in candidates:
        if not path.exists() or not path.is_file():
            skipped.append(f"{path}: missing")
            continue
        try:
            sample = pd.read_csv(path, nrows=5)
            prepared = prepare_tracks_dataframe(sample)
            validate_required_columns(prepared, ("track_name", "artist"))
            return path
        except Exception as exc:
            skipped.append(f"{path}: {exc}")
    raise FileNotFoundError("No loadable track dataset found. Checked: " + "; ".join(skipped))


def _resolve_dataset_path(path: str | Path) -> Path:
    """Resolve user-provided paths from cwd first, then project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / candidate


def prepare_tracks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw track dataframe into the app/model schema."""
    output = normalize_column_names(df)
    validate_required_columns(output, ("track_name", "artist"))
    if "lyrics" not in output:
        output["lyrics"] = ""
    if "genre" not in output:
        output["genre"] = "unknown"
    if "popularity" not in output:
        output["popularity"] = 0
    keep_first = [
        "track_name",
        "artist",
        "lyrics",
        "genre",
        "popularity",
        "danceability",
        "energy",
        "valence",
        "tempo",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "loudness",
        "mood",
        "polarity",
        "subjectivity",
    ]
    ordered = [column for column in keep_first if column in output.columns]
    remaining = [column for column in output.columns if column not in ordered]
    return output[ordered + remaining]


def load_default_tracks_dataset(max_rows: int | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the best available track dataset and return dataframe plus source path."""
    path = find_best_tracks_dataset()
    df = pd.read_csv(path, nrows=max_rows)
    return prepare_tracks_dataframe(df), path
