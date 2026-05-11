from __future__ import annotations

from pathlib import Path

import ai_books_tracking.goodreads_followup_analysis as followup
import pandas as pd
import pytest
from ai_books_tracking.goodreads_followup_analysis import (
    PRIMARY_TARGET_SPECS,
    classify_missing_goodreads_reason,
    derive_analysis_columns,
    linear_fit_summary,
    simulate_filter_effects,
    summarize_group_relationships,
    summarize_inferred_sources,
    sweep_cutoff_policies,
    transform_target_utility,
)


def test_derive_analysis_columns_builds_averages_and_canonical_author() -> None:
    df = pd.DataFrame(
        [
            {
                "title": "Example Book",
                "author": "by",
                "author_ratings2": "Ted Chiang",
                "goodreads_author": "Wrong Author",
                "earliest_modified": "2024-01-01",
                "latest_modified": "2024-01-03",
                "earliest_modified_ratings2": None,
                "latest_modified_ratings2": None,
                "Long Term Effects": "Some notes",
                "gb_page_count": 199,
                "pub_year": 2019,
                "Enjoyment (/5)": 4.0,
                "Enjoyment (/5)_ratings2": 3.0,
                "Usefulness /5 to Me": 2.0,
                "Usefulness /5 to Me_ratings2": 4.0,
            }
        ]
    )

    derived = derive_analysis_columns(df)
    row = derived.iloc[0]

    assert row["avg_enjoyment"] == 3.5
    assert row["avg_usefulness"] == 3.0
    assert row["avg_enjoyment_utility"] == pytest.approx((3.5 - 1.0) ** 1.3)
    assert row["avg_usefulness_utility"] == pytest.approx(3.0)
    assert row["enjoyment_label_gap"] == 1.0
    assert row["usefulness_label_gap"] == 2.0
    assert row["canonical_author"] == "ted chiang"
    assert row["reading_days"] == 2
    assert row["year_finished"] == 2024
    assert row["note_length"] == len("Some notes")


def test_classify_missing_goodreads_reason_flags_bad_import_patterns() -> None:
    row = pd.Series({"title": "p1eav352oa122c19jjuuavncabr4", "author": "by"})

    reasons = classify_missing_goodreads_reason(row).split("|")

    assert "opaque_id" in reasons
    assert "missing_or_placeholder_author" in reasons
    assert "very_short_or_ambiguous_title" in reasons


def test_classify_missing_goodreads_reason_detects_filename_noise() -> None:
    row = pd.Series(
        {
            "title": "Intro Reinforcement Learning from Human Feedback.pdf",
            "author": "Unknown",
        }
    )

    reasons = classify_missing_goodreads_reason(row).split("|")

    assert "filename_or_file_suffix" in reasons
    assert "missing_or_placeholder_author" in reasons


def test_linear_fit_summary_applies_range_filter() -> None:
    df = pd.DataFrame(
        {
            "goodreads_rating_raw_best": [3.0, 3.4, 4.0, 4.9],
            "avg_enjoyment": [2.0, 3.0, 4.0, 5.0],
            "Bookshelf": ["A", "A", "B", "B"],
        }
    )

    subset, fit = linear_fit_summary(
        df,
        x_col="goodreads_rating_raw_best",
        y_col="avg_enjoyment",
        x_range=(3.2, 4.8),
    )

    assert len(subset) == 2
    assert subset["goodreads_rating_raw_best"].min() == 3.4
    assert subset["goodreads_rating_raw_best"].max() == 4.0
    assert fit["n"] == 2.0


def test_sweep_cutoff_policies_finds_supported_thresholds() -> None:
    ratings = [3.2] * 15 + [4.2] * 15 + [4.5] * 15
    enjoyment = [2.0] * 15 + [4.0] * 15 + [4.5] * 15
    df = pd.DataFrame(
        {
            "goodreads_rating_raw_best": ratings,
            "avg_enjoyment": enjoyment,
        }
    )

    sweep = sweep_cutoff_policies(df, target_col="avg_enjoyment")

    keep_ge = sweep[sweep["rule"] == "keep_if_ge"].sort_values(
        "delta_keep_vs_all", ascending=False
    )
    best = keep_ge.iloc[0]

    assert best["threshold_lo"] == 4.3
    assert best["n_keep"] == 15
    assert best["keep_mean"] == 4.5


def test_summarize_group_relationships_filters_small_groups_and_range() -> None:
    df = pd.DataFrame(
        {
            "Bookshelf": ["A", "A", "A", "B", "B", "C"],
            "goodreads_rating_raw_best": [3.1, 3.4, 4.0, 4.1, 4.4, 4.2],
            "avg_enjoyment": [2.0, 3.0, 4.0, 4.2, 4.8, 3.5],
            "avg_usefulness": [1.0, 1.5, 2.0, 2.2, 2.7, 1.2],
        }
    )

    summary = summarize_group_relationships(
        df,
        group_col="Bookshelf",
        target_specs=PRIMARY_TARGET_SPECS,
        min_n=2,
        x_range=(3.2, 4.8),
        range_label="restricted",
    )

    assert set(summary["group_name"]) == {"A", "B"}
    row = summary[
        (summary["group_name"] == "A") & (summary["target"] == "avg_enjoyment")
    ].iloc[0]
    assert row["n"] == 2
    assert row["mean_goodreads"] == pytest.approx(3.7)
    assert row["mean_target"] == pytest.approx(3.5)


def test_summarize_inferred_sources_uses_matched_subset_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = pd.DataFrame(
        {
            "title": ["a1", "a2", "a3", "b1", "b2"],
            "Bookshelf": ["Lit", "Lit", "Lit", "CS", "CS"],
            "inferred_source": [
                "Source A",
                "Source A",
                "Source A",
                "Source B",
                "Source B",
            ],
            "goodreads_rating_raw_best": [4.0, 4.2, None, 3.8, 3.9],
            "avg_enjoyment": [3.0, 3.5, 5.0, 2.0, 2.5],
            "avg_usefulness": [1.0, 1.5, 4.5, 2.0, 2.5],
        }
    )

    monkeypatch.setattr(followup, "SOURCE_SUMMARY_CSV", tmp_path / "source_summary.csv")
    summary = summarize_inferred_sources(df, min_n=2)

    source_a = summary[summary["inferred_source"] == "Source A"].iloc[0]
    assert source_a["matched_goodreads_n"] == 2
    assert source_a["source_total_n"] == 3
    assert source_a["mean_avg_enjoyment"] == pytest.approx(3.25)
    assert source_a["mean_avg_usefulness"] == pytest.approx(1.25)


def test_simulate_filter_effects_tracks_mean_and_kept_correlation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = pd.DataFrame(
        {
            "goodreads_rating_raw_best": [3.2, 3.4, 3.6, 4.0, 4.2, 4.4],
            "avg_enjoyment": [2.0, 2.5, 3.0, 4.0, 4.5, 5.0],
            "avg_usefulness": [1.0, 1.2, 1.5, 2.0, 2.2, 2.5],
        }
    )

    monkeypatch.setattr(followup, "FILTER_EFFECT_CSV", tmp_path / "filter_effects.csv")
    effects = simulate_filter_effects(df, min_keep_n=2)
    row = effects[
        (effects["target"] == "avg_enjoyment") & (effects["threshold"] == 4.0)
    ].iloc[0]

    assert row["n_keep"] == 3
    assert row["keep_mean"] == pytest.approx(4.5)
    assert row["delta_keep_vs_all"] > 0
    assert row["delta_keep_utility_vs_all"] > row["delta_keep_vs_all"]
    assert row["keep_spearman_rho"] == pytest.approx(1.0)


def test_transform_target_utility_matches_assumed_profiles() -> None:
    ratings = pd.Series([1.0, 3.0, 5.0])

    enjoyment = transform_target_utility(ratings, "avg_enjoyment")
    usefulness = transform_target_utility(ratings, "avg_usefulness")

    assert enjoyment.iloc[0] == 0.0
    assert enjoyment.iloc[1] == pytest.approx(2.0**1.3)
    assert usefulness.iloc[0] == 0.0
    assert usefulness.iloc[1] == pytest.approx(3.0)
    assert usefulness.iloc[2] == pytest.approx(15.0)
