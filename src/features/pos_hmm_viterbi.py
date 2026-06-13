"""Educational HMM POS tagger with Viterbi decoding."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

import pandas as pd


TOY_TAGGED_CORPUS = [
    [("i", "PRON"), ("love", "VERB"), ("calm", "ADJ"), ("music", "NOUN")],
    [("we", "PRON"), ("dance", "VERB"), ("tonight", "ADV")],
    [("sad", "ADJ"), ("songs", "NOUN"), ("heal", "VERB"), ("slowly", "ADV")],
    [("strong", "ADJ"), ("beats", "NOUN"), ("move", "VERB"), ("fast", "ADV")],
    [("sleep", "VERB"), ("under", "ADP"), ("quiet", "ADJ"), ("stars", "NOUN")],
]
OPEN_CLASS_TAGS = {"NOUN", "VERB", "ADJ", "ADV", "PRON"}


def simple_tokenize(text: object) -> list[str]:
    return re.findall(r"[a-z']+", "" if pd.isna(text) else str(text).lower())


def get_training_corpus() -> list[list[tuple[str, str]]]:
    try:
        import nltk

        return [[(word.lower(), tag[:2]) for word, tag in sent] for sent in nltk.corpus.brown.tagged_sents(categories="news")[:500]]
    except Exception:
        return TOY_TAGGED_CORPUS


def train_hmm_pos_tagger(tagged_sentences: list[list[tuple[str, str]]] | None = None):
    tagged_sentences = tagged_sentences or get_training_corpus()
    transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
    emission_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tag_counts: Counter[str] = Counter()
    prior_counts: Counter[str] = Counter()

    for sentence in tagged_sentences:
        previous = "<START>"
        for i, (word, tag) in enumerate(sentence):
            word = word.lower()
            if i == 0:
                prior_counts[tag] += 1
            transition_counts[previous][tag] += 1
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1
            previous = tag
        transition_counts[previous]["<END>"] += 1

    tags = list(tag_counts)
    vocab = {word for counter in emission_counts.values() for word in counter}

    transition_probs = {
        tag: {
            next_tag: (count + 1) / (sum(counter.values()) + len(tags) + 1)
            for next_tag, count in {**{t: 0 for t in tags + ["<END>"]}, **counter}.items()
        }
        for tag, counter in transition_counts.items()
    }
    emission_probs = {
        tag: {
            word: (emission_counts[tag][word] + 1) / (tag_counts[tag] + len(vocab) + 1)
            for word in vocab
        }
        for tag in tags
    }
    for tag in tags:
        emission_probs[tag]["<UNK>"] = 1 / (tag_counts[tag] + len(vocab) + 1)
    tag_priors = {tag: (prior_counts[tag] + 1) / (sum(prior_counts.values()) + len(tags)) for tag in tags}
    return transition_probs, emission_probs, tag_priors


def viterbi_decode(tokens, transition_probs, emission_probs, tag_priors):
    tokens = [token.lower() for token in tokens]
    if not tokens:
        return []
    tags = list(tag_priors)
    states: list[dict[str, tuple[float, list[str]]]] = []

    first = {}
    for tag in tags:
        emission = emission_probs[tag].get(tokens[0], emission_probs[tag].get("<UNK>", 1e-9))
        first[tag] = (math.log(tag_priors[tag]) + math.log(emission), [tag])
    states.append(first)

    for token in tokens[1:]:
        current = {}
        for tag in tags:
            emission = emission_probs[tag].get(token, emission_probs[tag].get("<UNK>", 1e-9))
            candidates = []
            for previous_tag in tags:
                transition = transition_probs.get(previous_tag, {}).get(tag, 1e-9)
                score, path = states[-1][previous_tag]
                candidates.append((score + math.log(transition) + math.log(emission), path + [tag]))
            current[tag] = max(candidates, key=lambda item: item[0])
        states.append(current)
    return max(states[-1].values(), key=lambda item: item[0])[1]


def pos_tag_lyrics(lyrics: object) -> list[tuple[str, str]]:
    tokens = simple_tokenize(lyrics)
    transition_probs, emission_probs, tag_priors = train_hmm_pos_tagger()
    tags = viterbi_decode(tokens, transition_probs, emission_probs, tag_priors)
    return list(zip(tokens, tags))


def extract_pos_features(pos_tags: list[tuple[str, str]]) -> dict[str, float]:
    total = max(1, len(pos_tags))
    counts = Counter(tag for _, tag in pos_tags)

    def ratio(prefixes: tuple[str, ...]) -> float:
        return sum(count for tag, count in counts.items() if tag in prefixes or tag.startswith(prefixes)) / total

    return {
        "noun_ratio": ratio(("NOUN", "NN")),
        "verb_ratio": ratio(("VERB", "VB")),
        "adjective_ratio": ratio(("ADJ", "JJ")),
        "adverb_ratio": ratio(("ADV", "RB")),
        "pronoun_ratio": ratio(("PRON", "PRP")),
        "unique_pos_count": float(len(counts)),
    }


def add_pos_features(df: pd.DataFrame, lyrics_column: str = "lyrics") -> pd.DataFrame:
    output = df.copy()
    lyrics = output[lyrics_column] if lyrics_column in output else pd.Series("", index=output.index)
    features = lyrics.map(lambda text: extract_pos_features(pos_tag_lyrics(text)))
    for column in ["noun_ratio", "verb_ratio", "adjective_ratio", "adverb_ratio", "pronoun_ratio", "unique_pos_count"]:
        output[column] = features.map(lambda row, c=column: row[c])
    return output
