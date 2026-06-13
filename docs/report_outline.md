# Report Outline

## 1. Problem Definition

Define scenario-aware music evaluation and recommendation. Explain why song suitability depends on listening context.

## 2. Data Collection

Describe offline base datasets and the Wikipedia Billboard Year-End scraper. Include source URLs, scraping date, and ethical scraping notes.

## 3. Data Integration and Cleaning

Explain column normalization, missing-value handling, duplicate removal, outlier capping, normalized audio features, and chart metadata joins.

## 4. EDA and Visualization

Include distribution plots, correlation heatmap, chi-square tests, genre/scenario counts, scatter plots, treemap, and parallel coordinates.

## 5. Feature Engineering

Cover audio helper scores, VADER/fallback sentiment, emotional intensity, topic keyword detection, metadata features, and chart features.

## 6. Machine Learning Modeling

Compare majority baseline, logistic regression, SVM, and random forest. Use holdout evaluation and 10-fold cross-validation where class counts allow it.

## 7. Advanced NLP with HMM POS Tagging and Viterbi

Explain HMM states, transition probabilities, emission probabilities, tag priors, Viterbi decoding, and POS ratio features.

## 8. Recommendation System

Describe content-based similarity, scenario ranking, and hybrid scoring with classifier probability.

## 9. Evaluation

Report classification metrics and recommendation metrics: diversity@K, novelty@K, scenario_fit@K, explainability_coverage@K, and precision@K if labels exist.

## 10. Limitations and Future Work

Discuss weak labels, lyric licensing, chart-data bias, offline evaluation limits, and future human evaluation.

## 11. Big Data Discussion

Summarize how the project could scale to 500K+ songs with chunked Pandas, Dask, Spark, HDFS, or MapReduce.
