# Billboard Scraper — Implementation Notes

How `src/scraping/scrape_billboard_wikipedia.py` collects the Billboard Year-End
Hot 100 chart data used as an external data source for this project.

## Goal

Acquire raw, real-world chart data (year, rank, title, artist) by scraping it from
the web, rather than downloading a ready-made dataset. The output is integrated
into the tracks dataset to produce chart features (see `docs/data_sources.md`).

## Source

Wikipedia "Billboard Year-End Hot 100 singles of {year}" pages, one page per year:

```
https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}
```

Each page contains a single `table.wikitable` with ~100 ranked rows.

## Libraries

- `requests` — HTTP fetch.
- `beautifulsoup4` (`html.parser`) — HTML parsing.
- `pandas` — assemble rows into a DataFrame and write CSV.

No Scrapy/Selenium: the pages are static server-rendered HTML, so a plain request
plus an HTML parser is enough and keeps the dependency surface small.

## Pipeline

The scraper is a small set of single-purpose functions:

1. **`build_billboard_year_end_urls(start_year, end_year)`**
   Builds the list of `(year, url)` pairs for an inclusive year range.

2. **`scrape_year_end_page(url, year)`**
   Fetches one page with a descriptive `User-Agent` and a request timeout, then
   hands the HTML to the parser.

3. **`parse_song_table(html, year, source_url)`**
   Selects `table.wikitable`, locates the rank/title/artist columns, and extracts
   one record per data row. Columns are found **by header name**, not by fixed
   position (see "Column detection" below).

4. **`clean_scraped_chart_data(df)`**
   Normalizes text, coerces `rank`/`year` to integers, drops rows missing any key
   field, removes duplicate `(year, rank, title, artist)` rows, and sorts by
   year then rank.

5. **`scrape_years(start_year, end_year)`**
   Loops the URL list and concatenates the per-year frames.

6. **`save_raw_scraped_data(df, path)`**
   Writes a CSV, creating parent directories as needed.

The `main()` CLI ties these together:

```bash
python -m src.scraping.scrape_billboard_wikipedia \
  --start-year 2018 --end-year 2023 \
  --output data/raw/billboard_year_end_raw.csv \
  --clean-output examples/billboard_year_end_clean.csv
```

It writes both the raw scrape (`--output`) and the cleaned snapshot
(`--clean-output`).

## Column detection (the key design decision)

The Year-End tables label the rank column **"No."**, not "Rank", and the artist
column **"Artist(s)"**. A naive parser that hard-codes a "Rank" header or fixed
column indices silently matches nothing.

`_find_column_indices(headers)` instead scans the header cells and maps each role
to its column index using candidate name lists:

```python
RANK_HEADERS   = ("rank", "no.", "no", "#", "position", "pos")
TITLE_HEADERS  = ("title", "song", "single")
ARTIST_HEADERS = ("artist", "artists", "artist(s)", "performer", "performer(s)")
```

A header matches a role if it equals or starts with one of the candidates. Rows
are then read by the discovered indices, so the scraper tolerates layout/label
differences across years. A table is skipped unless all three roles are found.

## Cleaning details

- `_clean_cell_text` strips Wikipedia citation markers (`[1]`), non-breaking
  spaces, surrounding quotes, and collapses whitespace.
- `_rank_from_text` pulls the first integer out of a rank cell.
- `clean_scraped_chart_data` deduplicates and sorts so the committed snapshot is
  stable and diff-friendly.

## Reproducibility & ethics

- Every row records its `source_url` and a UTC `scraped_at` timestamp, so the
  snapshot is auditable.
- A descriptive `User-Agent` identifies the scraper as an educational project.
- Runs are small (a handful of pages) and the result is committed once at
  `examples/billboard_year_end_clean.csv` so re-scraping is not needed to run the
  project.
- Only chart metadata (rank/title/artist) is collected; lyrics are not scraped.

## Testing

`tests/test_scraper_parser.py` runs the parser against a local HTML fixture
(`examples/sample_billboard_page.html`) and checks deduplication, so the parsing
logic is verified without any network access.
