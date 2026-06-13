from pathlib import Path

from src.scraping.scrape_billboard_wikipedia import clean_scraped_chart_data, parse_song_table


def test_parse_song_table_from_sample_html():
    html = Path("examples/sample_billboard_page.html").read_text()
    df = parse_song_table(html, year=2023, source_url="https://example.test/source")
    assert list(df.columns) == ["year", "rank", "track_name", "artist", "source_url", "scraped_at"]
    assert len(df) == 2
    assert df.loc[0, "rank"] == 1
    assert df.loc[0, "track_name"] == "Example Hit"


def test_clean_scraped_chart_data_deduplicates():
    html = Path("examples/sample_billboard_page.html").read_text()
    df = parse_song_table(html, year=2023)
    doubled = clean_scraped_chart_data(df._append(df, ignore_index=True))
    assert len(doubled) == 2
