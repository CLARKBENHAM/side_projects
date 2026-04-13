from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import analysis_core.movie_rt_second_pass_analysis as second_pass_analysis  # noqa: E402
from analysis_core.movie_rt_second_pass_analysis import (  # noqa: E402
    LOCAL_TZ,
    assign_life_period,
    build_cleanup_overlap,
    build_score_cutoff_tradeoff,
    build_synopsis_dataset,
    extract_linear_pipeline_coefficients,
    fit_full_cv_ridge_context_model,
    load_second_sheet_dataset,
)


def test_assign_life_period_uses_expected_regimes() -> None:
    assert assign_life_period(pd.Timestamp("2021-09-01")) == ("early_hive", "high")
    assert assign_life_period(pd.Timestamp("2024-07-01")) == ("mats", "high")
    assert assign_life_period(pd.Timestamp("2025-03-01")) == ("hadrian_vllm", "medium")
    assert assign_life_period(pd.Timestamp("2025-12-20")) == (
        "post_diesl_transition",
        "low",
    )


def test_load_second_sheet_dataset_assigns_unfinished_movies_a_two(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text(
        "\n".join(
            [
                "date,datetime_local,movie_title,clark_rating,normalized_title,watch_status,where_seen,saw_in_home,saw_in_theater,drink_before_movie,completion_basis,completion_confidence,calendar_summary,calendar_location,rt_matched_title,rt_release_year,rt_audience_score,rt_critic_score,rt_url,imdb_matched_title,imdb_release_year,imdb_score,imdb_rating_count,imdb_url,quality_flags,rt_imdb_year_gap,external_mean_rating,rating_gap_vs_external_mean,abs_rating_gap_vs_external_mean",
                "2025-03-01,2025-03-01T20:00-08:00,Finished Pick,7,finished pick,finished,home,True,False,False,long_single_event,medium,Movie: Finished Pick,,Finished Pick,2024,80,70,https://example.com/rt1,Finished Pick,2024,7.1,1000,https://example.com/imdb1,,,7.40,-0.40,0.40",
                "2025-03-02,2025-03-02T20:00-08:00,Started Pick,na,started pick,unfinished,home,True,False,False,started_only_event,low,Started movie: Started Pick,,Started Pick,2024,60,50,https://example.com/rt2,Started Pick,2024,6.0,500,https://example.com/imdb2,unfinished_watch;low_watch_confidence,0,5.67,,",
            ]
        )
        + "\n"
    )
    df = load_second_sheet_dataset(
        csv_path,
        source_name="test_source",
        include_unfinished_as_two=True,
        calendar_dir=tmp_path / "empty_calendar",
        as_of_local=datetime(2026, 4, 4, 23, 59, 59, tzinfo=LOCAL_TZ),
    )

    assert len(df) == 2
    started = df.loc[df["movie_title"] == "Started Pick"].iloc[0]
    assert started["analysis_rating"] == 2.0
    assert bool(started["assigned_unfinished_two"]) is True
    assert started["life_period"] == "hadrian_vllm"


def test_build_cleanup_overlap_matches_on_analysis_row_key() -> None:
    raw = pd.DataFrame(
        [
            {
                "analysis_row_key": "2025-03-01|pick",
                "movie_title": "Pick",
                "analysis_rating": 7.0,
                "rt_average_on_10": 7.0,
                "rt_critic_on_10": 6.5,
                "rt_audience_on_10": 7.5,
                "external_mean_on_10": 7.0,
            }
        ]
    )
    clean = pd.DataFrame(
        [
            {
                "analysis_row_key": "2025-03-01|pick",
                "movie_title": "Pick Cleaned",
                "analysis_rating": 7.0,
                "rt_average_on_10": 7.5,
                "rt_critic_on_10": 7.0,
                "rt_audience_on_10": 8.0,
                "external_mean_on_10": 7.5,
            }
        ]
    )
    overlap = build_cleanup_overlap(raw, clean)

    assert len(overlap) == 1
    assert overlap.iloc[0]["raw_movie_title"] == "Pick"
    assert overlap.iloc[0]["clean_movie_title"] == "Pick Cleaned"


def test_build_synopsis_dataset_falls_back_to_wikipedia(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def raise_imdb_error(imdb_url: str) -> str:
        raise RuntimeError("blocked")

    monkeypatch.setattr(
        second_pass_analysis,
        "fetch_short_imdb_synopsis",
        raise_imdb_error,
    )
    monkeypatch.setattr(
        second_pass_analysis,
        "fetch_wikipedia_summary",
        lambda movie_title, release_year: (
            f"{movie_title} summary from {int(release_year)}"
        ),
    )
    df = pd.DataFrame(
        [
            {
                "movie_title": "Example Movie",
                "imdb_release_year": 2024,
                "analysis_rating": 7.0,
                "watch_status": "finished",
                "where_seen": "home",
                "life_period": "hadrian_vllm",
                "likely_with_amelia": False,
                "imdb_url": "https://www.imdb.com/title/tt1234567/",
            }
        ]
    )

    synopsis_df = build_synopsis_dataset(
        df, cache_path=tmp_path / "synopsis_cache.json"
    )

    assert len(synopsis_df) == 1
    assert synopsis_df.iloc[0]["synopsis"] == "Example Movie summary from 2024"


def test_extract_linear_pipeline_coefficients_returns_named_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "analysis_rating": 4.0,
                "rt_critic_on_10": 4.0,
                "rt_audience_on_10": 5.0,
                "imdb_score": 5.0,
                "start_hour": 19.0,
                "month_index": 0.0,
                "saw_in_theater": False,
                "drink_before_movie": False,
                "assigned_unfinished_two": False,
                "likely_with_amelia": False,
                "is_weekend": False,
                "life_period": "pre_hive",
                "pressure_tier": "low",
                "data_source": "second_clean",
            },
            {
                "analysis_rating": 7.0,
                "rt_critic_on_10": 7.0,
                "rt_audience_on_10": 7.5,
                "imdb_score": 7.0,
                "start_hour": 21.0,
                "month_index": 1.0,
                "saw_in_theater": True,
                "drink_before_movie": True,
                "assigned_unfinished_two": False,
                "likely_with_amelia": True,
                "is_weekend": True,
                "life_period": "mats",
                "pressure_tier": "high",
                "data_source": "second_clean",
            },
            {
                "analysis_rating": 2.0,
                "rt_critic_on_10": 3.0,
                "rt_audience_on_10": 4.0,
                "imdb_score": 4.5,
                "start_hour": 20.0,
                "month_index": 2.0,
                "saw_in_theater": False,
                "drink_before_movie": False,
                "assigned_unfinished_two": True,
                "likely_with_amelia": False,
                "is_weekend": False,
                "life_period": "diesl",
                "pressure_tier": "high",
                "data_source": "second_clean",
            },
        ]
    )
    pipeline = fit_full_cv_ridge_context_model(df)
    coefficients = extract_linear_pipeline_coefficients(
        pipeline,
        numeric_features=[
            "rt_critic_on_10",
            "rt_audience_on_10",
            "imdb_score",
            "start_hour",
            "month_index",
        ],
        binary_features=[
            "saw_in_theater",
            "drink_before_movie",
            "assigned_unfinished_two",
            "likely_with_amelia",
            "is_weekend",
        ],
        categorical_features=["life_period", "pressure_tier", "data_source"],
    )

    assert "intercept" in set(coefficients["original_feature"])
    assert "rt_critic_on_10" in set(coefficients["original_feature"])
    assert "saw_in_theater" in set(coefficients["original_feature"])
    assert "life_period" in set(
        coefficients.loc[
            coefficients["feature_group"] == "categorical", "original_feature"
        ]
    )


def test_build_score_cutoff_tradeoff_uses_score_thresholds() -> None:
    df = pd.DataFrame(
        [
            {"analysis_rating": 8.0, "rt_average_rating": 90.0},
            {"analysis_rating": 6.0, "rt_average_rating": 70.0},
            {"analysis_rating": 4.0, "rt_average_rating": 50.0},
            {"analysis_rating": 2.0, "rt_average_rating": 30.0},
        ]
    )

    tradeoff = build_score_cutoff_tradeoff(
        df,
        score_column="rt_average_rating",
        score_label="rt_average_rating",
    )

    assert tradeoff.iloc[0]["score_cutoff"] >= tradeoff.iloc[-1]["score_cutoff"]
    assert tradeoff.iloc[0]["mean_rating"] >= tradeoff.iloc[-1]["mean_rating"]
