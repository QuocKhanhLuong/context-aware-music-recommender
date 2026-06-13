# Big Data Considerations

This project can involve 500K+ songs when using large audio, lyric, genre, and chart datasets.

## Volume

Hundreds of thousands of songs with lyrics, audio features, sparse topic indicators, POS features, and chart joins can exceed memory on small laptops. A first scaling step is chunked Pandas reads and incremental feature export to Parquet.

## Variety

The system combines structured audio features, semi-structured chart tables, free-text lyrics, categorical genres, and weak scenario labels. A scalable design should keep clear schemas for each intermediate dataset.

## Veracity

Lyrics may be missing, chart artist names may differ from dataset names, and scenario labels may be weak labels. Larger scale increases the need for validation reports, fuzzy-match confidence thresholds, and source metadata.

## Value

The useful output is not just model accuracy. The system should produce scenario rankings, explanations, diversity, novelty, and reproducible evaluation artifacts.

## Scaling Options

- Chunked Pandas: process CSV files in batches and write feature partitions.
- Dask: scale Pandas-like processing across local cores or a cluster.
- Spark: use distributed DataFrames for feature generation and joins.
- HDFS or object storage: store raw, processed, and feature datasets separately.
- MapReduce: tokenize lyrics, count topic keywords, and aggregate POS/topic features in distributed jobs.

Full big-data implementation is optional for this course project, but the pipeline is structured so each stage can be replaced by a distributed equivalent.
