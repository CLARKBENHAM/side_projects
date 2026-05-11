from __future__ import annotations

import pandas as pd
import pytest
import ai_books_tracking.future_prediction_evaluation as future_prediction_evaluation
from ai_books_tracking.future_prediction_evaluation import (
    FeatureSpec,
    add_author_history_features,
    apply_policy_to_predictions,
    evaluate_year_split,
    prepare_feature_frames,
    threshold_policy_summary,
)


def test_add_author_history_features_uses_train_only_and_loo() -> None:
    train = pd.DataFrame(
        {
            "canonical_author": ["a", "a", "b", ""],
            "avg_enjoyment": [4.0, 2.0, 5.0, 3.0],
        }
    )
    test = pd.DataFrame(
        {
            "canonical_author": ["a", "b", "c", ""],
            "avg_enjoyment": [None, None, None, None],
        }
    )

    train_hist, test_hist = add_author_history_features(train, test, "avg_enjoyment")

    assert train_hist.loc[0, "author_target_mean_hist"] == pytest.approx(2.0)
    assert train_hist.loc[1, "author_target_mean_hist"] == pytest.approx(4.0)
    assert train_hist.loc[2, "author_target_mean_hist"] == pytest.approx(3.5)
    assert test_hist.loc[0, "author_target_mean_hist"] == pytest.approx(3.0)
    assert test_hist.loc[1, "author_target_mean_hist"] == pytest.approx(5.0)
    assert test_hist.loc[2, "author_target_mean_hist"] == pytest.approx(3.5)
    assert test_hist.loc[3, "author_book_count_hist"] == 0


def test_prepare_feature_frames_adds_goodreads_features_and_imputes_missing() -> None:
    train = pd.DataFrame(
        {
            "canonical_author": ["a", "b"],
            "Bookshelf": ["Lit", "CS"],
            "year_finished": [2024, 2024],
            "log_pages": [5.0, 6.0],
            "book_age": [10.0, 5.0],
            "avg_usefulness": [2.0, 4.0],
            "goodreads_rating_raw_best": [4.1, None],
            "goodreads_rating_count_raw_best": [100.0, None],
        }
    )
    test = pd.DataFrame(
        {
            "canonical_author": ["a", "c"],
            "Bookshelf": ["Lit", "CS"],
            "year_finished": [2025, 2025],
            "log_pages": [5.5, 6.5],
            "book_age": [8.0, 4.0],
            "avg_usefulness": [3.0, 2.0],
            "goodreads_rating_raw_best": [None, 4.5],
            "goodreads_rating_count_raw_best": [None, 50.0],
        }
    )

    train_frame, test_frame, numeric, categorical = prepare_feature_frames(
        train,
        test,
        target_col="avg_usefulness",
        spec=FeatureSpec("raw", include_goodreads="raw_best"),
    )

    assert "goodreads_available" in numeric
    assert "goodreads_rating_feature" in numeric
    assert "goodreads_log_count_feature" in numeric
    assert categorical == ["Bookshelf"]
    assert train_frame["goodreads_available"].tolist() == [1.0, 0.0]
    assert test_frame.loc[0, "goodreads_rating_feature"] == pytest.approx(4.1)
    assert test_frame.loc[0, "goodreads_log_count_feature"] == pytest.approx(2.00432137)


def test_threshold_policy_summary_and_apply_policy_find_balanced_rule() -> None:
    predictions = pd.DataFrame(
        {
            "prediction": [2.0] * 20 + [4.0] * 20,
            "avg_enjoyment": [2.0] * 20 + [4.5] * 20,
        }
    )

    sweep, best = threshold_policy_summary(predictions, "avg_enjoyment")
    balanced = best[best["selection"] == "best_balanced_utility"].iloc[0]
    realized = apply_policy_to_predictions(
        predictions,
        target_col="avg_enjoyment",
        threshold=float(balanced["threshold"]),
        selection="best_balanced_utility",
    )

    assert not sweep.empty
    assert 2.0 < balanced["threshold"] <= 4.0
    assert realized["policy_keep_n"] == 20
    assert realized["policy_keep_mean"] == pytest.approx(4.5)
    assert realized["policy_delta_keep_vs_all"] > 1.0


def test_evaluate_year_split_handles_avg_target_without_duplicate_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(future_prediction_evaluation, "MIN_TRAIN_ROWS", 2)
    df = pd.DataFrame(
        {
            "title": ["a", "b", "c"],
            "Bookshelf": ["Lit", "Lit", "Lit"],
            "canonical_author": ["x", "y", "z"],
            "year_finished": [2023, 2024, 2025],
            "log_pages": [5.0, 5.1, 5.2],
            "book_age": [10.0, 9.0, 8.0],
            "avg_enjoyment": [3.0, 4.0, 5.0],
            "avg_usefulness": [2.0, 3.0, 4.0],
        }
    )

    result = evaluate_year_split(
        df=df,
        target_col="avg_enjoyment",
        spec=FeatureSpec("preread_base"),
        model_name="Global mean",
        test_year=2025,
        prediction_split="test",
    )

    assert len(result) == 1
    assert result["prediction_error"].iloc[0] == pytest.approx(-1.5)
