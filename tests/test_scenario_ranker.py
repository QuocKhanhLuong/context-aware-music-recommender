import pandas as pd

from src.evaluation.rubric import compute_scenario_score, explain_scenario_score
from src.models.scenario_ranker import ScenarioRanker


def test_scenario_scores_between_zero_and_one():
    df = pd.read_csv("examples/sample_tracks.csv")
    scores = compute_scenario_score(df, "study")
    assert scores.between(0, 1).all()


def test_scenario_ranker_returns_top_k_and_explanations():
    df = pd.read_csv("examples/sample_tracks.csv")
    recs = ScenarioRanker().fit(df).recommend("sleep", top_k=3)
    assert len(recs) == 3
    assert recs["scenario_score"].between(0, 1).all()
    assert recs["explanations"].map(len).min() > 0
    assert explain_scenario_score(recs.iloc[0], "sleep")
