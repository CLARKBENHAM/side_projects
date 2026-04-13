from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis_core.movie_rt_followup_questions import (  # noqa: E402
    build_data_quality_candidate_table,
    build_taste_numeric_holdout_table,
    build_unique_movie_prediction_frame,
    evaluate_holdout_blends,
    loo_linear_feature_set_metrics,
    parse_rt_genres_from_html,
)


def test_loo_linear_feature_set_metrics_returns_expected_shapes() -> None:
    df = pd.DataFrame(
        [
            {
                "analysis_rating": 4.0,
                "rt_critic_on_10": 4.0,
                "rt_audience_on_10": 4.0,
                "imdb_score": 4.0,
            },
            {
                "analysis_rating": 6.0,
                "rt_critic_on_10": 6.0,
                "rt_audience_on_10": 6.0,
                "imdb_score": 6.0,
            },
            {
                "analysis_rating": 8.0,
                "rt_critic_on_10": 8.0,
                "rt_audience_on_10": 8.0,
                "imdb_score": 8.0,
            },
        ]
    )

    metrics_df, coefficient_df = loo_linear_feature_set_metrics(
        df,
        subset_name="toy",
        feature_sets={
            "rt_imdb": ["rt_critic_on_10", "rt_audience_on_10", "imdb_score"]
        },
    )

    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["rows"] == 3.0
    assert len(coefficient_df) == 12
    assert set(coefficient_df["feature"]) == {
        "intercept",
        "rt_critic_on_10",
        "rt_audience_on_10",
        "imdb_score",
    }


def test_build_taste_numeric_holdout_table_aligns_unique_predictions() -> None:
    second_sheet_predictions = pd.DataFrame(
        [
            {
                "movie_title": "Movie A",
                "imdb_release_year": 2024,
                "imdb_url": "https://example.com/a",
                "analysis_rating": 7.0,
                "cv_linear_rt_prediction": 6.0,
                "cv_linear_rt_imdb_prediction": 6.5,
                "cv_ridge_context_prediction": 7.5,
            },
            {
                "movie_title": "Movie A",
                "imdb_release_year": 2024,
                "imdb_url": "https://example.com/a",
                "analysis_rating": 8.0,
                "cv_linear_rt_prediction": 6.2,
                "cv_linear_rt_imdb_prediction": 6.8,
                "cv_ridge_context_prediction": 7.8,
            },
            {
                "movie_title": "Movie B",
                "imdb_release_year": 2025,
                "imdb_url": "https://example.com/b",
                "analysis_rating": 5.0,
                "cv_linear_rt_prediction": 5.2,
                "cv_linear_rt_imdb_prediction": 5.4,
                "cv_ridge_context_prediction": 5.6,
            },
        ]
    )
    unique_predictions = build_unique_movie_prediction_frame(second_sheet_predictions)
    synopsis_holdout = pd.DataFrame(
        [
            {
                "movie_title": "Movie A",
                "release_year": 2024,
                "analysis_rating": 7.0,
            },
            {
                "movie_title": "Movie B",
                "release_year": 2025,
                "analysis_rating": 5.0,
            },
        ]
    )
    taste_predictions = pd.DataFrame(
        [
            {
                "movie_title": "Movie A",
                "release_year": 2024,
                "predicted_rating": 7.5,
            },
            {
                "movie_title": "Movie B",
                "release_year": 2025,
                "predicted_rating": 5.5,
            },
        ]
    )

    merged = build_taste_numeric_holdout_table(
        unique_predictions=unique_predictions,
        synopsis_holdout=synopsis_holdout,
        taste_predictions=taste_predictions,
    )

    assert len(merged) == 2
    assert (
        merged.loc[
            merged["movie_title"] == "Movie A", "cv_ridge_context_prediction"
        ].iloc[0]
        == 7.5
    )


def test_evaluate_holdout_blends_reports_expected_baselines() -> None:
    holdout_df = pd.DataFrame(
        [
            {
                "analysis_rating": 7.0,
                "predicted_rating": 7.0,
                "cv_linear_rt_imdb_prediction": 6.0,
                "cv_ridge_context_prediction": 7.0,
            },
            {
                "analysis_rating": 5.0,
                "predicted_rating": 5.0,
                "cv_linear_rt_imdb_prediction": 4.0,
                "cv_ridge_context_prediction": 5.0,
            },
        ]
    )

    metrics_df, alpha_df = evaluate_holdout_blends(
        holdout_df,
        numeric_prediction_columns=[
            "cv_linear_rt_imdb_prediction",
            "cv_ridge_context_prediction",
        ],
    )

    taste_row = metrics_df.loc[metrics_df["model"] == "taste_agent"].iloc[0]
    ridge_row = metrics_df.loc[
        metrics_df["model"] == "cv_ridge_context_prediction"
    ].iloc[0]
    assert taste_row["mae"] == 0.0
    assert ridge_row["mae"] == 0.0
    assert set(alpha_df["numeric_model"]) == {
        "cv_linear_rt_imdb_prediction",
        "cv_ridge_context_prediction",
    }


def test_parse_rt_genres_from_html_reads_metadata_and_fallback() -> None:
    assert parse_rt_genres_from_html(
        'metadataGenres:["Sci-Fi","Adventure","Drama"]'
    ) == ["Sci-Fi", "Adventure", "Drama"]
    assert parse_rt_genres_from_html('cag[genre]:"Comedy|Romance"') == [
        "Comedy",
        "Romance",
    ]


def test_build_data_quality_candidate_table_prioritizes_flags_and_disagreement() -> (
    None
):
    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "movie_title": "Stable Row",
                "watch_status": "finished",
                "analysis_rating": 7.0,
                "quality_flags": "",
                "rt_critic_on_10": 7.0,
                "rt_audience_on_10": 7.0,
                "imdb_score": 7.0,
                "rt_average_on_10": 7.0,
                "external_mean_on_10": 7.0,
                "cv_ridge_context_prediction": 7.0,
                "rt_matched_title": "Stable Row",
                "rt_release_year": 2025,
                "rt_url": "https://example.com/rt1",
                "imdb_matched_title": "Stable Row",
                "imdb_release_year": 2025,
                "imdb_url": "https://example.com/imdb1",
            },
            {
                "date": "2025-01-02",
                "movie_title": "Suspicious Row",
                "watch_status": "unfinished",
                "analysis_rating": 2.0,
                "quality_flags": "unfinished_watch;rt_tv_url",
                "rt_critic_on_10": 9.0,
                "rt_audience_on_10": 2.0,
                "imdb_score": 8.5,
                "rt_average_on_10": 5.5,
                "external_mean_on_10": 6.5,
                "cv_ridge_context_prediction": 7.5,
                "rt_matched_title": "Wrong Match",
                "rt_release_year": 2026,
                "rt_url": "https://example.com/rt2",
                "imdb_matched_title": "Wrong Match",
                "imdb_release_year": 2026,
                "imdb_url": "https://example.com/imdb2",
            },
        ]
    )

    review = build_data_quality_candidate_table(
        df,
        prediction_column="cv_ridge_context_prediction",
    )

    assert review.iloc[0]["movie_title"] == "Suspicious Row"
