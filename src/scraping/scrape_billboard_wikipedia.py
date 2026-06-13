"""Scrape Billboard Year-End Hot 100 tables from Wikipedia.

The scraper stores source URLs and retrieval dates so the collected snapshot is
reproducible and auditable. Tests use local sample HTML instead of the network.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
USER_AGENT = (
    "context-aware-music-recommender/1.0 "
    "(course project; polite educational scraping)"
)


def build_billboard_year_end_urls(start_year: int, end_year: int) -> list[tuple[int, str]]:
    """Build Wikipedia Billboard Year-End page URLs for an inclusive year range."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")
    return [(year, BASE_URL.format(year=year)) for year in range(start_year, end_year + 1)]


def _clean_cell_text(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", str(text))
    text = text.replace("\xa0", " ")
    text = text.strip().strip('"')
    return re.sub(r"\s+", " ", text)


def _rank_from_text(text: str) -> int | None:
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else None


RANK_HEADERS = ("rank", "no.", "no", "#", "position", "pos")
TITLE_HEADERS = ("title", "song", "single")
ARTIST_HEADERS = ("artist", "artists", "artist(s)", "performer", "performer(s)")


def _find_column_indices(headers: list[str]) -> dict[str, int] | None:
    """Locate rank/title/artist columns by header name (robust to layout changes)."""
    lowered = [h.lower() for h in headers]

    def find(candidates: tuple[str, ...]) -> int | None:
        for idx, head in enumerate(lowered):
            if any(head == cand or head.startswith(cand) for cand in candidates):
                return idx
        return None

    rank_idx = find(RANK_HEADERS)
    title_idx = find(TITLE_HEADERS)
    artist_idx = find(ARTIST_HEADERS)
    if rank_idx is None or title_idx is None or artist_idx is None:
        return None
    return {"rank": rank_idx, "track_name": title_idx, "artist": artist_idx}


def parse_song_table(html: str, year: int, source_url: str | None = None) -> pd.DataFrame:
    """Parse a Wikipedia song table into year/rank/title/artist rows.

    Wikipedia Year-End tables label the rank column "No." rather than "Rank",
    so columns are located by header name instead of a fixed position.
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select("table.wikitable")
    records: list[dict[str, object]] = []

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        headers = [_clean_cell_text(cell.get_text(" ")) for cell in rows[0].find_all(["th", "td"])]
        columns = _find_column_indices(headers)
        if columns is None:
            continue

        max_idx = max(columns.values())
        for row in rows[1:]:
            cells = [_clean_cell_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
            if len(cells) <= max_idx:
                continue
            rank = _rank_from_text(cells[columns["rank"]])
            if rank is None:
                continue
            records.append(
                {
                    "year": int(year),
                    "rank": rank,
                    "track_name": cells[columns["track_name"]],
                    "artist": cells[columns["artist"]],
                    "source_url": source_url or BASE_URL.format(year=year),
                    "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
        if records:
            break

    return pd.DataFrame(records, columns=["year", "rank", "track_name", "artist", "source_url", "scraped_at"])


def scrape_year_end_page(url: str, year: int, timeout: int = 20) -> pd.DataFrame:
    """Fetch and parse one Billboard Year-End page."""
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return parse_song_table(response.text, year=year, source_url=url)


def save_raw_scraped_data(df: pd.DataFrame, path: str | Path) -> None:
    """Save raw scraped rows as CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def clean_scraped_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and deduplicate scraped chart rows."""
    cleaned = df.copy()
    for col in ["track_name", "artist", "source_url"]:
        if col in cleaned:
            cleaned[col] = cleaned[col].fillna("").map(_clean_cell_text)
    if "rank" in cleaned:
        cleaned["rank"] = pd.to_numeric(cleaned["rank"], errors="coerce").astype("Int64")
    if "year" in cleaned:
        cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce").astype("Int64")
    cleaned = cleaned.dropna(subset=["year", "rank", "track_name", "artist"])
    cleaned = cleaned.drop_duplicates(subset=["year", "rank", "track_name", "artist"])
    return cleaned.sort_values(["year", "rank"]).reset_index(drop=True)


def scrape_years(start_year: int, end_year: int) -> pd.DataFrame:
    """Scrape multiple years and return concatenated raw rows."""
    frames = [scrape_year_end_page(url, year) for year, url in build_billboard_year_end_urls(start_year, end_year)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scrape Wikipedia Billboard Year-End Hot 100 pages.")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output", default="data/raw/billboard_year_end_raw.csv")
    parser.add_argument("--clean-output", default="data/processed/billboard_year_end_clean.csv")
    args = parser.parse_args(list(argv) if argv is not None else None)

    raw = scrape_years(args.start_year, args.end_year)
    save_raw_scraped_data(raw, args.output)
    clean = clean_scraped_chart_data(raw)
    save_raw_scraped_data(clean, args.clean_output)
    print(f"Saved {len(raw)} raw rows to {args.output}")
    print(f"Saved {len(clean)} cleaned rows to {args.clean_output}")


if __name__ == "__main__":
    main()
