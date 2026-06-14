# Context-Aware Music Evaluation and Recommendation System

## A Data Science Pipeline for Context-Aware Music Evaluation using Web-Scraped Chart Data, Lyrics NLP, Audio Features, and Scenario-Based Recommendation

This is an independent IT4142 final project. The system evaluates and recommends songs for explicit listening scenarios instead of relying only on title, artist, genre, or live Spotify API calls.

Core scenarios:

- `study`
- `gym`
- `sleep`
- `party`
- `sad_healing`

## Important Data Note

`data/raw/` only contains `.gitkeep` in GitHub on purpose. Large datasets are ignored and should not be committed.

On the original local machine, there may be ignored CSV files such as:

- `data/allsong_data.csv`
- `data/raw_data.csv`
- `data/processed_data.csv`
- `data/complete_feature.csv`
- `data/test_playlist.csv`
- `data/processed/tracks_app_ready.csv`

These local files are enough to run a real-data demo on that machine, but they are not included when someone else clones the repository.

For a fresh clone, the project still runs immediately with sample data under `examples/`. To run with a larger real dataset, download or copy a CSV into `data/raw/`, then run the preparation script.

## Course Scope Mapping

| Requirement | Implementation |
| --- | --- |
| Problem definition | Scenario-aware music suitability prediction and recommendation |
| Data collection | Offline CSV datasets plus Wikipedia Billboard Year-End scraping (snapshot committed at `examples/billboard_year_end_clean.csv`; see `docs/data_sources.md`) |
| Data integration and cleaning | `src/data/load_data.py`, `src/data/integrate_data.py`, `src/data/preprocess.py`; chart features are merged into tracks in the notebook (section 3b) and `scripts/prepare_real_data.py` |
| EDA and visualization | `src/visualization/eda_plots.py` and `notebooks/context_aware_music_pipeline.ipynb`, including a parallel-coordinates view of scenario feature profiles |
| Machine learning modeling | Baselines, logistic regression, SVM, random forest, holdout split, and stratified 10-fold cross-validation for model selection |
| Advanced NLP | HMM POS tagging with Viterbi decoding in `src/features/pos_hmm_viterbi.py` |
| Scientific evaluation | Classification metrics, recommendation metrics, diversity, novelty, scenario fit, explainability coverage |
| Big data discussion | `docs/big_data_considerations.md` |

## Why No Live Spotify API

The project does not depend on live Spotify Web API endpoints such as Audio Features, Audio Analysis, or Recommendations. The app uses offline CSV data so the demo is reproducible and does not require Spotify credentials.

## Fresh Clone Setup

Clone the repository:

```bash
git clone https://github.com/QuocKhanhLuong/context-aware-music-recommender.git
cd context-aware-music-recommender
git checkout upgrade/context-aware-system
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest
```

Expected result:

```text
17 passed
```

## Run With Sample Data

This works immediately after cloning. It uses `examples/sample_tracks.csv` and creates an app-ready CSV under `data/processed/`.

```bash
python scripts/prepare_real_data.py --max-rows 5000
```

Then run the app:

```bash
streamlit run src/app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

In the sidebar, check the active dataset. On a fresh clone, it will eventually fall back to sample data unless you add a real dataset.

## Run With Local Real Data

If you already have local CSV files in `data/`, such as `data/allsong_data.csv`, you do not need to download more data just to run the app.

Prepare the local data:

```bash
python scripts/prepare_real_data.py --max-rows 5000
```

This auto-detects available datasets in this priority order:

1. `data/raw/spotify_550k_tracks.csv`
2. `data/raw/tracks.csv`
3. `data/allsong_data.csv`
4. `data/processed_data.csv`
5. `examples/sample_tracks.csv`

It writes:

```text
data/processed/tracks_app_ready.csv
```

Then run:

```bash
streamlit run src/app/streamlit_app.py
```

The app loads `data/processed/tracks_app_ready.csv` first.

## Run With A Downloaded Dataset

Download a Kaggle-style music CSV manually or with the Kaggle CLI, then place it in `data/raw/`.

Recommended schema:

- `track_name`
- `artist` or `artist_name`
- `lyrics` if available
- `genre` or `genres`
- `popularity` or `track_pop`
- `danceability`
- `energy`
- `valence`
- `tempo`
- `acousticness`
- `speechiness`
- `instrumentalness`
- `loudness`

Prepare it:

```bash
python scripts/prepare_real_data.py \
  --input data/raw/spotify_550k_tracks.csv \
  --output data/processed/tracks_app_ready.csv \
  --max-rows 5000
```

Remove `--max-rows 5000` if you want to prepare the full dataset.

Optional Kaggle CLI flow:

```bash
pip install kaggle
mkdir -p ~/.kaggle
```

Put your Kaggle API token at:

```text
~/.kaggle/kaggle.json
```

Then:

```bash
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_kaggle_dataset.py --dataset OWNER/DATASET_SLUG
```

After download, inspect `data/raw/`, find the CSV file name, and run `scripts/prepare_real_data.py --input ...`.

## Run Billboard Scraping

The Billboard scraper is the project's web-scraping data-collection deliverable. A
committed snapshot (`examples/billboard_year_end_clean.csv`, 600 rows, 2018–2023)
ships with the repo, and its chart features are integrated into the tracks dataset
(see `docs/data_sources.md`). To refresh or extend the snapshot:

```bash
python -m src.scraping.scrape_billboard_wikipedia \
  --start-year 2018 \
  --end-year 2023 \
  --output data/raw/billboard_year_end_raw.csv \
  --clean-output examples/billboard_year_end_clean.csv
```

Ethical scraping note:

- Use the scraper for small, reproducible educational runs.
- Keep source URLs and scraping dates.
- Check `robots.txt`, Wikimedia usage guidance, and terms of use before larger runs.
- Do not scrape or redistribute copyrighted lyrics unless explicitly allowed.

## Export Report Artifacts

Generate the figures and experiment results used in the written report:

```bash
python scripts/export_report_artifacts.py --max-rows 5000
```

This writes to `reports/` (committed example outputs are generated from the
Kaggle Spotify Tracks dataset, ~6k-row sample):

- `reports/figures/*.png` — correlation heatmap, audio-feature distributions,
  energy-vs-valence scatter, scenario label counts, scenario feature profile,
  top genres, parallel coordinates, confusion matrix, and the 10-fold CV
  comparison bar chart.
- `reports/results/*` — `classification_metrics.json`, `confusion_matrix.csv`,
  `cv_comparison.csv` (10-fold cross-validation model comparison), and
  `recommendation_metrics.json`.

On a tiny sample dataset the holdout/CV step is skipped with a note; run it on a
real dataset for full results.

## Run The Notebook

There is one consolidated notebook for the full Data Science workflow:

```bash
jupyter notebook notebooks/context_aware_music_pipeline.ipynb
```

Run cells from top to bottom. It covers:

1. Setup
2. Load real or sample data
3. Data cleaning
4. EDA and relationship analysis
5. Audio, lyrics, POS HMM/Viterbi, and scenario-label features
6. Machine learning model evaluation
7. Scenario-based recommendation demo
8. Hybrid recommendation demo
9. Notes for the final report

The notebook is independent from the Streamlit app. You can run both at the same time in two terminals:

Terminal 1:

```bash
streamlit run src/app/streamlit_app.py
```

Terminal 2:

```bash
jupyter notebook notebooks/context_aware_music_pipeline.ipynb
```

Streamlit usually runs on port `8501`; Jupyter usually runs on port `8888`. They can run in parallel because both read local CSV files. If you use the full dataset in both at the same time, preprocessing can be slow, so use `--max-rows 5000` or `--max-rows 10000` for a stable demo.

## App Demo Flow

After running:

```bash
streamlit run src/app/streamlit_app.py
```

Use these pages:

1. `Dataset Overview`: check dataset source, shape, columns, missing values, and numeric summary.
2. `EDA Dashboard`: inspect distributions, scatter plot, correlation matrix, genre counts, and chi-square result.
3. `Model Training & Evaluation`: choose logistic regression, SVM, or random forest.
4. `Music Recommendation Demo`: choose scenario and recommendation model.

Suggested demo:

1. Choose `Music Recommendation Demo`.
2. Scenario: `study`.
3. Model: `scenario_ranker`.
4. Top K: `5`.
5. Show scores and explanations.
6. Switch scenario to `gym` or `party` and show that rankings change.

## Folder Structure

```text
README.md
requirements.txt
data/
  raw/.gitkeep
  processed/.gitkeep
examples/
  sample_tracks.csv
  sample_billboard_scraped.csv
notebooks/
  context_aware_music_pipeline.ipynb
scripts/
  prepare_real_data.py
  download_kaggle_dataset.py
src/
  app/
  data/
  evaluation/
  features/
  models/
  scraping/
  visualization/
tests/
docs/
```

## Models

- `MajorityClassBaseline`: most-frequent scenario baseline
- `LogisticRegressionBaseline`: simple supervised baseline
- `ScenarioClassifier`: logistic regression, SVM, or random forest
- `ContentBasedRecommender`: cosine similarity over audio, metadata, sentiment, topic, POS, and chart features
- `ScenarioRanker`: explicit scenario scoring rubric
- `HybridRecommender`: combines similarity, scenario score, and classifier probability

Hybrid score:

```text
final_score = 0.3 * similarity_score + 0.4 * scenario_score + 0.3 * classifier_probability
```

## Scenario Scoring Rubric

- `study`: calm audio, low tempo, acousticness, low emotional intensity, instrumentalness, low speechiness
- `gym`: energetic audio, high tempo, danceability, positive sentiment, motivation topic
- `sleep`: calm audio, low tempo, acousticness, low loudness, low speechiness, low emotional intensity
- `party`: danceability, energetic audio, high valence, popularity, party topic, Billboard appearance
- `sad_healing`: sad sentiment, calm audio, acousticness, low energy, breakup topic, relaxing topic

Scenario scores are clipped to `[0, 1]`, and recommendations include explanation reasons.

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
- Precision@K when ground truth exists

## Scenario Labels

If a dataset includes mood labels, they are mapped to scenarios. If explicit labels are missing, weak labels are generated from audio and lyrics heuristics and stored in `scenario_label`.

Weak labels are useful for a course project and baseline evaluation, but they are not human preference ground truth.

## Limitations

- Local ignored CSVs are not available to fresh clones.
- Weak scenario labels are heuristic.
- Lyrics may be missing or restricted by copyright.
- Billboard chart data measures popularity, not listener suitability.
- Offline metrics do not fully measure real listener satisfaction.
- The HMM POS tagger is educational, not state-of-the-art.

## Future Work

- Add human evaluation for scenario fit.
- Use a licensed lyrics dataset or store only derived lyric features.
- Add a dataset card with exact Kaggle/source citation.
- Add Dask/Spark processing for 500K+ songs.
- Improve entity matching with MusicBrainz IDs or other open identifiers.

## Attribution

This independent project was inspired by a public Spotify playlist recommender reference repository, but this repository is not a fork. The implementation has been rebuilt around offline data, explicit listening scenarios, reproducible scraping, and IT4142 final project requirements.
