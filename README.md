# Context-Aware Music Evaluation and Recommendation System

## A Data Science Pipeline for Context-Aware Music Evaluation using Web-Scraped Chart Data, Lyrics NLP, Audio Features, and Scenario-Based Recommendation

This repository is an independent IT4142 final project. It transforms an old Spotify playlist recommender idea into an offline, reproducible data science pipeline for evaluating and recommending music for explicit listening scenarios.

Core scenarios:

- `study`
- `gym`
- `sleep`
- `party`
- `sad_healing`

The main question is: given a song's audio features, lyrics-derived NLP features, POS patterns, genre/popularity metadata, and scraped chart metadata, which listening scenario is the song most suitable for, and why?

## Motivation

Music recommendation should not rely only on title, artist, genre, or lyrics. The same song can create different feelings for different listeners and contexts. This project addresses teacher feedback by making listening context explicit: the user chooses a scenario, the system scores scenario suitability, ranks songs, and explains the top reasons for each recommendation.

## Course Scope Mapping

| Requirement | Implementation |
| --- | --- |
| Problem definition | Scenario-aware music suitability prediction and recommendation |
| Data collection | Offline Kaggle-style tracks plus Wikipedia Billboard Year-End scraping |
| Data cleaning | `src/data/preprocess.py` handles text cleanup, missing values, duplicates, outliers, and normalized audio columns |
| EDA | `src/visualization/eda_plots.py` plus notebooks for distributions, correlation, chi-square, scatter, bar, treemap, and parallel-coordinate analysis |
| ML modeling | Baselines, logistic regression, SVM, random forest, holdout split, and GridSearchCV support |
| Advanced NLP | HMM POS tagging with Viterbi decoding in `src/features/pos_hmm_viterbi.py` |
| Evaluation | Classification metrics, top-k recommendation metrics, diversity, novelty, scenario fit, and explainability coverage |
| Big data considerations | `docs/big_data_considerations.md` discusses 500K+ song scaling with chunked Pandas, Dask, Spark, HDFS, and MapReduce |

## Why No Live Spotify API

The project does not depend on live Spotify Web API endpoints such as Audio Features, Audio Analysis, or Recommendations. Those endpoints are unsuitable for a reproducible offline course project and introduce API access, policy, and availability risks. Use offline datasets and self-collected public chart metadata instead.

## Data Sources

Recommended base dataset:

- 550K Spotify Songs: Audio, Lyrics & Genres or an equivalent offline CSV
- Expected columns include `track_name`, `artist`, `lyrics`, `genre`, `popularity`, `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `speechiness`, `instrumentalness`, and `loudness`

Optional dataset:

- Mood/emotion-labeled music dataset with labels such as `happy`, `sad`, `energetic`, and `calm`

Scraped data:

- Wikipedia Billboard Year-End Hot 100 singles pages
- Output fields: `year`, `rank`, `track_name`, `artist`, `source_url`, `scraped_at`

Ethical note: the scraper is intended for polite, reproducible educational use. Check `robots.txt`, Wikimedia API etiquette, and terms of use before large runs. Do not scrape or redistribute copyrighted lyrics unless the source explicitly allows it.

Large raw datasets should not be committed. Keep `data/raw/.gitkeep` and `data/processed/.gitkeep`; use `examples/` for small demo CSVs.

## Folder Structure

```text
README.md
requirements.txt
data/raw/.gitkeep
data/processed/.gitkeep
examples/sample_tracks.csv
examples/sample_billboard_scraped.csv
notebooks/
src/
  scraping/
  data/
  features/
  models/
  evaluation/
  visualization/
  app/
tests/
docs/report_outline.md
docs/big_data_considerations.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Scraping

```bash
python -m src.scraping.scrape_billboard_wikipedia \
  --start-year 2010 \
  --end-year 2023 \
  --output data/raw/billboard_year_end_raw.csv \
  --clean-output data/processed/billboard_year_end_clean.csv
```

If online scraping is unavailable, use `examples/sample_billboard_scraped.csv` for tests and demos.

## Run Notebooks

Use one consolidated notebook for the final report workflow:

- `notebooks/context_aware_music_pipeline.ipynb`

Open it with Jupyter and run against the prepared real dataset first. If no real dataset is prepared, it falls back to the sample CSV.

## Prepare Real Data

Large datasets are ignored by git. Put downloaded CSV files under `data/raw/`, then normalize them for the app:

```bash
python scripts/prepare_real_data.py \
  --input data/raw/spotify_550k_tracks.csv \
  --output data/processed/tracks_app_ready.csv
```

For a faster classroom demo:

```bash
python scripts/prepare_real_data.py \
  --input data/raw/spotify_550k_tracks.csv \
  --output data/processed/tracks_app_ready.csv \
  --max-rows 5000
```

If you already have local CSVs from the old reference project, run auto-detection:

```bash
python scripts/prepare_real_data.py --max-rows 5000
```

Optional Kaggle CLI download flow:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# put kaggle.json from your Kaggle account in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_kaggle_dataset.py --dataset OWNER/DATASET_SLUG
```

Then inspect `data/raw/` and pass the downloaded CSV path to `scripts/prepare_real_data.py`.

## Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

The app loads `data/processed/tracks_app_ready.csv` first when available. Otherwise it falls back to raw/local CSV candidates and finally `examples/sample_tracks.csv`. The sidebar shows which dataset source is active and lets you limit rows for faster demos.

## Run Tests

```bash
python -m pytest
```

## Models

- `MajorityClassBaseline`: most-frequent scenario baseline
- `LogisticRegressionBaseline`: simple supervised baseline
- `ScenarioClassifier`: logistic regression, SVM, or random forest with optional GridSearchCV
- `ContentBasedRecommender`: cosine similarity over audio, metadata, sentiment, topic, POS, and chart features
- `ScenarioRanker`: ranks by explicit scenario rubric
- `HybridRecommender`: combines similarity, scenario score, and classifier probability

Hybrid score:

```text
final_score = 0.3 * similarity_score + 0.4 * scenario_score + 0.3 * classifier_probability
```

## Scenario Scoring Rubric

- `study`: calm audio, low tempo, acousticness, low emotional intensity, instrumentalness, low speechiness, noun/adjective signal
- `gym`: energetic audio, high tempo, danceability, positive sentiment, motivation topic
- `sleep`: calm audio, low tempo, acousticness, low loudness, low speechiness, low emotional intensity
- `party`: danceability, energetic audio, high valence, popularity, party topic, Billboard appearance
- `sad_healing`: sad sentiment, calm audio, acousticness, low energy, breakup topic, relaxing topic

Scenario scores are clipped to `[0, 1]`, and each recommendation includes top explanation reasons.

## Evaluation Metrics

Classification:

- Accuracy
- Precision
- Recall
- F1-score
- Micro F1
- Macro F1
- Confusion matrix

Recommendation:

- Diversity@K
- Novelty@K
- Scenario fit@K
- Explainability coverage@K
- Precision@K when ground truth is available

## Scenario Labels

If a dataset includes mood labels, they are mapped to scenarios. If explicit labels are missing, weak labels are generated from a documented audio/lyrics heuristic and stored in `scenario_label`. These weak labels are useful for coursework and baselines, but they are not human preference ground truth.

## Limitations

- Weak scenario labels are heuristic and may encode bias.
- Lyrics licensing can limit redistribution of raw text.
- Billboard chart data reflects popularity, not listener context or suitability.
- Offline metrics do not fully measure real listener satisfaction.
- The HMM POS tagger is educational and intentionally simple.

## Future Work

- Add human evaluation for scenario fit.
- Use licensed lyric datasets or derived lyric features only.
- Add Dask/Spark pipelines for 500K+ songs.
- Add experiment tracking and model cards.
- Improve matching with MusicBrainz IDs or other open identifiers.

## Attribution

This independent project structure was inspired by a public Spotify playlist recommender reference repository, but this repository is not a fork and the implementation has been rebuilt around offline data, context-aware scenarios, reproducible scraping, and IT4142 final project requirements.
