"""Lightweight lyric NLP features with VADER fallback."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


TOPIC_KEYWORDS = {
    "love": {"love", "heart", "kiss", "baby", "darling"},
    "breakup": {"break", "goodbye", "alone", "cry", "tears", "miss"},
    "motivation": {"rise", "strong", "win", "fight", "power", "run"},
    "party": {"party", "dance", "club", "tonight", "drink", "dj"},
    "loneliness": {"lonely", "alone", "empty", "silence", "lost"},
    "confidence": {"boss", "shine", "proud", "bold", "fearless"},
    "relaxing": {"sleep", "dream", "breathe", "calm", "slow", "peace"},
}
POSITIVE_WORDS = {"love", "happy", "bright", "win", "dance", "strong", "shine", "peace"}
NEGATIVE_WORDS = {"sad", "cry", "alone", "lost", "break", "tears", "dark", "hurt"}
INTENSITY_WORDS = {"very", "never", "always", "forever", "scream", "cry", "fire", "wild", "heart"}


def clean_lyrics(text: object) -> str:
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", clean_lyrics(text))


def fallback_sentiment_score(text: object) -> float:
    tokens = _tokens(str(text))
    if not tokens:
        return 0.0
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    return max(-1.0, min(1.0, (positive - negative) / max(1, positive + negative)))


def vader_sentiment_score(text: object) -> float:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return float(analyzer.polarity_scores(str(text))["compound"])
    except Exception:
        return fallback_sentiment_score(text)


def emotional_intensity_score(text: object) -> float:
    tokens = _tokens(str(text))
    if not tokens:
        return 0.0
    exclamations = str(text).count("!")
    keyword_hits = sum(token in INTENSITY_WORDS for token in tokens)
    return min(1.0, (keyword_hits + exclamations) / max(4, len(tokens) * 0.35))


def detect_topic_keywords(text: object, topics: dict[str, set[str]] | None = None) -> dict[str, float]:
    tokens = set(_tokens(str(text)))
    topics = topics or TOPIC_KEYWORDS
    return {f"topic_{topic}": float(bool(tokens & keywords)) for topic, keywords in topics.items()}


def add_lyric_features(df: pd.DataFrame, lyrics_column: str = "lyrics") -> pd.DataFrame:
    output = df.copy()
    lyrics = output[lyrics_column] if lyrics_column in output else pd.Series("", index=output.index)
    output["clean_lyrics"] = lyrics.map(clean_lyrics)
    output["sentiment_score"] = lyrics.map(vader_sentiment_score)
    output["positive_sentiment"] = output["sentiment_score"].clip(lower=0)
    output["sad_sentiment"] = (-output["sentiment_score"]).clip(lower=0)
    output["emotional_intensity_score"] = lyrics.map(emotional_intensity_score)
    output["low_emotional_intensity"] = (1 - output["emotional_intensity_score"]).clip(0, 1)
    topic_rows = lyrics.map(detect_topic_keywords)
    for topic in TOPIC_KEYWORDS:
        column = f"topic_{topic}"
        output[column] = topic_rows.map(lambda row, c=column: row[c])
    return output
