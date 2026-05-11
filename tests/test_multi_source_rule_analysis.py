from __future__ import annotations

import math

import pandas as pd

from ai_books_tracking.multi_source_rule_analysis import (
    build_drop_curve,
    category_bucket,
    prepare_rule_frames,
)


def test_category_bucket_groups_main_buckets() -> None:
    assert category_bucket("Computer Science") == "Technical"
    assert category_bucket("Math") == "Technical"
    assert category_bucket("Machine Learning") == "Technical"
    assert category_bucket("Literature") == "Literature"
    assert category_bucket("fiction") == "Fiction"
    assert category_bucket("Business, management") == "General/Business"
    assert category_bucket("Histories") == "General/Business"


def test_prepare_rule_frames_complete_case_drops_missing_rows() -> None:
    train = pd.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [5.0, 6.0, 7.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "a": [2.0, None],
            "b": [8.0, 9.0],
        }
    )

    prepared_train, prepared_holdout, feature_cols = prepare_rule_frames(
        train, holdout, ["a", "b"], "complete_case"
    )

    assert feature_cols == ["a", "b"]
    assert len(prepared_train) == 2
    assert len(prepared_holdout) == 1


def test_prepare_rule_frames_median_mode_keeps_rows_and_adds_flags() -> None:
    train = pd.DataFrame(
        {
            "a": [1.0, None, 5.0],
            "b": [2.0, 4.0, None],
        }
    )
    holdout = pd.DataFrame(
        {
            "a": [None],
            "b": [10.0],
        }
    )

    prepared_train, prepared_holdout, feature_cols = prepare_rule_frames(
        train, holdout, ["a", "b"], "median_with_flags"
    )

    assert len(prepared_train) == 3
    assert len(prepared_holdout) == 1
    assert "a__available" in feature_cols
    assert "b__available" in feature_cols
    assert math.isclose(prepared_holdout.iloc[0]["a"], 3.0)
    assert prepared_holdout.iloc[0]["a__available"] == 0.0
    assert prepared_holdout.iloc[0]["b__available"] == 1.0


def test_build_drop_curve_uses_drop_fraction_as_bottom_cut() -> None:
    scored = pd.DataFrame(
        {
            "title": list("ABCDEFGHIJ"),
            "score": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "avg_enjoyment": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    curve = build_drop_curve(scored, "avg_enjoyment", "rule", "mode")
    row = curve[curve["drop_fraction"] == 0.4].iloc[0]

    assert row["keep_n"] == 6
    assert math.isclose(row["mean_kept_rating"], 7.5)
    assert math.isclose(row["rating_uplift"], 2.0)
