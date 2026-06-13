"""Rank songs by scenario suitability score."""

from __future__ import annotations

import pandas as pd

from src.evaluation.rubric import compute_scenario_score, explain_scenario_score
from src.features.audio_features import add_audio_helper_scores
from src.features.lyric_features import add_lyric_features
from src.features.pos_hmm_viterbi import add_pos_features


class ScenarioRanker:
    def __init__(self):
        self.df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame):
        output = add_audio_helper_scores(df)
        output = add_lyric_features(output)
        missing_pos = {"noun_ratio", "verb_ratio", "adjective_ratio", "adverb_ratio", "pronoun_ratio"} - set(output.columns)
        if missing_pos:
            output = add_pos_features(output)
        self.df = output
        return self

    def recommend(self, scenario: str, top_k: int = 10) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Call fit before recommend")
        output = self.df.copy()
        output["scenario_score"] = compute_scenario_score(output, scenario)
        output["explanations"] = output.apply(lambda row: explain_scenario_score(row, scenario), axis=1)
        return output.sort_values("scenario_score", ascending=False).head(top_k).reset_index(drop=True)
