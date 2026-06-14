"""Export figures and experiment results for the written report.

Runs the data + feature pipeline once, then writes:
  reports/figures/*.png   - EDA visualizations (matplotlib/seaborn)
  reports/results/*.{json,csv} - experiment results (metrics, CV, confusion matrix)

Figures use matplotlib so export needs no extra dependencies. Run:

    python scripts/export_report_artifacts.py --max-rows 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.integrate_data import attach_chart_features
from src.data.load_data import load_default_tracks_dataset
from src.data.preprocess import preprocess_tracks
from src.evaluation.cross_validation import compare_scenario_models_cv
from src.features.audio_features import add_audio_helper_scores
from src.features.emotion_features import create_weak_scenario_label
from src.features.lyric_features import add_lyric_features
from src.features.pos_hmm_viterbi import add_pos_features
from src.models.scenario_classifier import ScenarioClassifier
from src.models.scenario_ranker import ScenarioRanker
from src.evaluation.metrics import recommendation_metrics
from src.visualization.eda_plots import (
    pearson_correlation_matrix,
    scenario_parallel_coordinates_data,
)

PROFILE_FEATURES = ["energy", "valence", "tempo", "danceability", "acousticness"]


def build_features(max_rows: int | None) -> pd.DataFrame:
    raw, source = load_default_tracks_dataset(max_rows=max_rows)
    print(f"Dataset source: {source} ({len(raw)} rows)")
    clean = preprocess_tracks(raw)
    chart_path = ROOT / "examples" / "billboard_year_end_clean.csv"
    if chart_path.exists():
        clean, matched = attach_chart_features(clean, pd.read_csv(chart_path))
        print(f"Chart integration: {matched} tracks matched")
    clean = add_audio_helper_scores(clean)
    clean = add_lyric_features(clean)
    clean = add_pos_features(clean)
    clean = create_weak_scenario_label(clean)
    return clean


def export_figures(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # Correlation heatmap (drop constant columns, e.g. lyric features when the
    # dataset has no lyrics, so the heatmap has no blank NaN rows)
    corr = pearson_correlation_matrix(df).dropna(how="all").dropna(axis=1, how="all")
    if corr.shape[0] >= 2:
        side = max(10, 0.5 * corr.shape[0])
        fig, ax = plt.subplots(figsize=(side, side))
        sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=False, square=True, ax=ax)
        ax.set_title("Pearson correlation heatmap")
        fig.tight_layout()
        path = fig_dir / "correlation_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    # Distributions of key audio features (histograms)
    dist_features = [c for c in ["energy", "valence", "tempo", "danceability", "acousticness", "popularity"] if c in df]
    if dist_features:
        ncols = 3
        nrows = (len(dist_features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        for ax, feature in zip(axes.ravel(), dist_features):
            values = pd.to_numeric(df[feature], errors="coerce").dropna()
            ax.hist(values, bins=40, color="#4C78A8")
            ax.set_title(feature)
        for ax in axes.ravel()[len(dist_features):]:
            ax.axis("off")
        fig.suptitle("Audio feature distributions")
        fig.tight_layout()
        path = fig_dir / "feature_distributions.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    # Scatter: energy vs valence colored by scenario
    if {"energy", "valence", "scenario_label"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(9, 6))
        for label, group in df.groupby("scenario_label"):
            ax.scatter(
                pd.to_numeric(group["energy"], errors="coerce"),
                pd.to_numeric(group["valence"], errors="coerce"),
                s=10, alpha=0.5, label=label,
            )
        ax.set_xlabel("energy")
        ax.set_ylabel("valence")
        ax.set_title("Energy vs Valence by scenario")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / "energy_vs_valence.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    # Scenario label counts
    if "scenario_label" in df:
        counts = df["scenario_label"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="bar", color="#72B7B2", ax=ax)
        ax.set_title("Scenario label counts")
        ax.set_ylabel("count")
        fig.tight_layout()
        path = fig_dir / "scenario_label_counts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    # Scenario feature profile
    profile_features = [c for c in PROFILE_FEATURES if c in df]
    if "scenario_label" in df and profile_features:
        profile = df.groupby("scenario_label")[profile_features].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        profile.plot(kind="bar", ax=ax)
        ax.set_title("Average feature profile by scenario")
        ax.set_ylabel("mean value")
        fig.tight_layout()
        path = fig_dir / "scenario_feature_profile.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    # Top genres
    if "genre" in df:
        top = df["genre"].fillna("unknown").astype(str).value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        top.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Top 15 genres")
        ax.set_xlabel("count")
        fig.tight_layout()
        path = fig_dir / "top_genres.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    # Parallel coordinates (issue #5)
    pc = scenario_parallel_coordinates_data(df)
    dims = [c for c in pc.columns if c != "scenario_label"]
    if "scenario_label" in pc and len(dims) >= 3:
        numeric = pc[dims].apply(pd.to_numeric, errors="coerce")
        # normalize each axis to [0, 1] so axes are comparable
        normalized = (numeric - numeric.min()) / (numeric.max() - numeric.min()).replace(0, 1)
        plot_df = normalized.assign(scenario_label=pc["scenario_label"].values).dropna()
        # subsample for legibility: thousands of overlapping lines are unreadable
        if len(plot_df) > 400:
            plot_df = plot_df.groupby("scenario_label", group_keys=False).apply(
                lambda g: g.sample(min(len(g), 100), random_state=42)
            )
        if not plot_df.empty and plot_df["scenario_label"].nunique() >= 1:
            fig, ax = plt.subplots(figsize=(11, 6))
            parallel_coordinates(plot_df, "scenario_label", colormap="tab10", ax=ax)
            ax.set_title("Scenario feature profiles (parallel coordinates, normalized)")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            path = fig_dir / "scenario_parallel_coordinates.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(str(path))

    return saved


def _can_evaluate(df: pd.DataFrame) -> bool:
    if "scenario_label" not in df:
        return False
    counts = df["scenario_label"].value_counts()
    n_classes = counts.size
    # A 0.25 holdout must leave at least one row per class on both sides, so the
    # dataset needs >= 4 * n_classes rows (and >= 2 rows/class for CV folds).
    return n_classes >= 2 and counts.min() >= 2 and len(df) >= 4 * n_classes


def export_results(df: pd.DataFrame, results_dir: Path, fig_dir: Path) -> list[str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if not _can_evaluate(df):
        note = {"note": "Too few rows per scenario for holdout/CV; run on the full dataset."}
        path = results_dir / "evaluation_skipped.json"
        path.write_text(json.dumps(note, indent=2))
        print("Evaluation skipped (small sample); wrote a note instead.")
        return [str(path)]

    # Holdout classification metrics + confusion matrix
    classifier = ScenarioClassifier(model_type="logistic_regression")
    metrics = classifier.evaluate_holdout(df)
    confusion = metrics.pop("confusion_matrix")
    labels = sorted(pd.Series(df["scenario_label"]).unique())
    path = results_dir / "classification_metrics.json"
    path.write_text(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))
    saved.append(str(path))
    cm_df = pd.DataFrame(confusion, index=labels, columns=labels)
    cm_path = results_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    saved.append(str(cm_path))

    # Confusion matrix heatmap figure
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (logistic regression holdout)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    cm_fig = fig_dir / "confusion_matrix.png"
    fig.savefig(cm_fig, dpi=150)
    plt.close(fig)
    saved.append(str(cm_fig))

    # 10-fold CV model comparison (issue #4)
    cv_df = compare_scenario_models_cv(df)
    cv_path = results_dir / "cv_comparison.csv"
    cv_df.to_csv(cv_path, index=False)
    saved.append(str(cv_path))
    best = cv_df.iloc[0]
    print(f"Best model by 10-fold CV macro F1: {best['model']} ({best['mean']:.3f} ± {best['std']:.3f})")

    # CV comparison bar chart with std error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cv_df["model"], cv_df["mean"], yerr=cv_df["std"], capsize=5, color="#54A24B")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean macro F1")
    ax.set_title("10-fold cross-validation comparison")
    fig.tight_layout()
    cv_fig = fig_dir / "cv_comparison.png"
    fig.savefig(cv_fig, dpi=150)
    plt.close(fig)
    saved.append(str(cv_fig))

    # Recommendation metrics for each scenario
    ranker = ScenarioRanker().fit(df)
    rec_metrics = {}
    for scenario in ["study", "gym", "sleep", "party", "sad_healing"]:
        recs = ranker.recommend(scenario, top_k=10)
        rec_metrics[scenario] = recommendation_metrics(recs, k=10)
    rec_path = results_dir / "recommendation_metrics.json"
    rec_path.write_text(json.dumps(rec_metrics, indent=2))
    saved.append(str(rec_path))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report figures and experiment results.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for faster runs.")
    parser.add_argument("--out-dir", default="reports", help="Output directory (default: reports/).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    df = build_features(args.max_rows)
    figures = export_figures(df, out_dir / "figures")
    results = export_results(df, out_dir / "results", out_dir / "figures")

    print(f"\nSaved {len(figures)} figures:")
    for path in figures:
        print(f"  {path}")
    print(f"Saved {len(results)} result files:")
    for path in results:
        print(f"  {path}")


if __name__ == "__main__":
    main()
