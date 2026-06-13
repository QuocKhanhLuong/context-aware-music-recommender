# Data Sources

Summary of the datasets used in this project and how they were acquired, for the
course data-collection requirement.

## 1. Track audio/metadata dataset (base data)

- **What:** Per-track audio features (danceability, energy, valence, tempo,
  acousticness, etc.), genre, popularity, and optional lyrics.
- **Source:** A public Kaggle-style Spotify tracks CSV placed in `data/raw/`, or
  the bundled `examples/sample_tracks.csv` for a fresh clone.
- **Acquisition:** Downloaded manually or via `scripts/download_kaggle_dataset.py`.
  Large datasets are gitignored and not committed.
- **Schema:** see the "Recommended schema" section in `README.md`.

## 2. Billboard Year-End Hot 100 chart data (web-scraped)

- **What:** Year-end rank, title, and artist for the Billboard Hot 100.
- **Source:** Wikipedia "Billboard Year-End Hot 100 singles of {year}" pages.
- **Acquisition:** Scraped with `src/scraping/scrape_billboard_wikipedia.py`
  (requests + BeautifulSoup). Each row keeps its `source_url` and `scraped_at`
  timestamp for auditability.
- **Snapshot committed:** `examples/billboard_year_end_clean.csv` — 600 rows,
  years 2018–2023 (100 per year), scraped 2026-06-13.
- **Re-run:**

  ```bash
  python -m src.scraping.scrape_billboard_wikipedia \
    --start-year 2018 --end-year 2023 \
    --output data/raw/billboard_year_end_raw.csv \
    --clean-output examples/billboard_year_end_clean.csv
  ```

### How the two sources are integrated

`src/data/integrate_data.py` fuzzy-matches each track to the chart snapshot on
title + artist (`SequenceMatcher`, weighted 0.7 title / 0.3 artist, threshold
0.88) and derives chart features: `appeared_in_billboard_year_end`,
`best_chart_rank`, `chart_year_count`, `latest_chart_year`. Tracks with no match
get neutral defaults (rank 101 = "not charted"). Integration runs in the notebook
(section 3b) and in `scripts/prepare_real_data.py`.

## Ethics / licensing notes

- Scraping is limited to small, reproducible educational runs; source URLs and
  retrieval dates are stored.
- Check `robots.txt` and Wikimedia usage guidance before larger runs.
- Lyrics are not scraped or redistributed; only derived lyric features are used.
- Billboard chart rank measures popularity, not listening-scenario suitability.
