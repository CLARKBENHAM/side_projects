from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import ai_books_tracking.new_books_to_rate_analysis as new_books_to_rate_analysis
from ai_books_tracking.new_books_to_rate_analysis import (
    apply_policy_threshold,
    assign_policy_split,
    build_centered_average,
    evaluate_target_variant,
    load_enriched_holdout,
    normalize_new_holdout,
    requested_utility,
    select_policy_rows,
    summarize_best_policy_rules,
    summarize_centering_effect,
)


def test_build_centered_average_equalizes_pass_means_and_preserves_pooled_mean() -> (
    None
):
    first = pd.Series([4.0, 3.0, 2.0])
    second = pd.Series([3.0, 2.0, 1.0])

    centered_first, centered_second, centered_average, summary = build_centered_average(
        first, second
    )

    assert summary.first_mean == pytest.approx(3.0)
    assert summary.second_mean == pytest.approx(2.0)
    assert summary.pooled_mean == pytest.approx(2.5)
    assert centered_first.mean() == pytest.approx(2.5)
    assert centered_second.mean() == pytest.approx(2.5)
    assert centered_average.mean() == pytest.approx(2.5)
    assert centered_average.tolist() == pytest.approx([3.5, 2.5, 1.5])


def test_normalize_new_holdout_standardizes_second_pass_columns(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "new_books.csv"
    pd.DataFrame(
        {
            "title": ["Example"],
            "date_finished": ["2026-03-11"],
            "Enjoyment (/5)": [4.0],
            "Usefulness /5 to Me": [2.0],
            "Enjoyment (/5) 2nd": [3.5],
            "Usefulness /5 to Me.1": [1.5],
            "Bookshelf": ["Literature"],
            "Long Term Effects": [None],
        }
    ).to_csv(csv_path, index=False)

    normalized = normalize_new_holdout(csv_path)

    assert "Enjoyment (/5)_ratings2" in normalized.columns
    assert "Usefulness /5 to Me_ratings2" in normalized.columns
    assert normalized.loc[0, "Long Term Effects"] == ""
    assert normalized.loc[0, "author"] == ""
    assert str(normalized.loc[0, "latest_modified"].date()) == "2026-03-11"


def test_load_enriched_holdout_parses_numeric_goodreads_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "holdout_enriched.csv"
    pd.DataFrame(
        {
            "title": ["Example"],
            "Enjoyment (/5)": ["4.0"],
            "Usefulness /5 to Me": ["2.5"],
            "Enjoyment (/5)_ratings2": ["3.5"],
            "Usefulness /5 to Me_ratings2": ["2.0"],
            "goodreads_rating": ["4.2"],
            "goodreads_rating_count": ["123"],
            "goodreads_rating_raw_best": ["4.2"],
            "goodreads_rating_count_raw_best": ["123"],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_enriched_holdout(csv_path)

    assert loaded.loc[0, "goodreads_rating"] == pytest.approx(4.2)
    assert loaded.loc[0, "goodreads_rating_count_raw_best"] == pytest.approx(123.0)
    assert loaded.loc[0, "Enjoyment (/5)_ratings2"] == pytest.approx(3.5)


def test_summarize_centering_effect_compares_raw_vs_centered_metrics() -> None:
    results = pd.DataFrame(
        [
            {
                "target": "avg_enjoyment",
                "feature_spec": "preread_base",
                "model": "Ridge",
                "family": "enjoyment",
                "mae": 0.80,
                "rmse": 1.00,
                "spearman_rho": 0.20,
            },
            {
                "target": "avg_enjoyment_centered",
                "feature_spec": "preread_base",
                "model": "Ridge",
                "family": "enjoyment",
                "mae": 0.70,
                "rmse": 0.90,
                "spearman_rho": 0.30,
            },
            {
                "target": "avg_usefulness",
                "feature_spec": "preread_base",
                "model": "Ridge",
                "family": "usefulness",
                "mae": 0.90,
                "rmse": 1.10,
                "spearman_rho": 0.10,
            },
            {
                "target": "avg_usefulness_centered",
                "feature_spec": "preread_base",
                "model": "Ridge",
                "family": "usefulness",
                "mae": 0.95,
                "rmse": 1.15,
                "spearman_rho": 0.05,
            },
        ]
    )

    comparison = summarize_centering_effect(results)

    enjoy = comparison[comparison["family"] == "enjoyment"].iloc[0]
    useful = comparison[comparison["family"] == "usefulness"].iloc[0]

    assert enjoy["mae_delta_centered_minus_raw"] == pytest.approx(-0.10)
    assert enjoy["rho_delta_centered_minus_raw"] == pytest.approx(0.10)
    assert useful["mae_delta_centered_minus_raw"] == pytest.approx(0.05)


def test_evaluate_target_variant_handles_average_target_without_duplicate_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(new_books_to_rate_analysis, "MIN_TRAIN_ROWS", 2)
    monkeypatch.setattr(
        new_books_to_rate_analysis,
        "model_predictions_for_split",
        lambda **_: pd.Series([3.0]).to_numpy(),
    )

    train = pd.DataFrame(
        {
            "Bookshelf": ["Lit", "Lit"],
            "canonical_author": ["a", "b"],
            "avg_enjoyment": [3.0, 4.0],
            "avg_enjoyment_centered": [3.0, 4.0],
            "avg_usefulness": [2.0, 3.0],
            "avg_usefulness_centered": [2.0, 3.0],
            "goodreads_rating": [4.0, 4.1],
        }
    )
    holdout = pd.DataFrame(
        {
            "title": ["Example"],
            "Bookshelf": ["Lit"],
            "date_finished": ["2026-03-11"],
            "canonical_author": ["c"],
            "avg_enjoyment": [4.0],
            "avg_enjoyment_centered": [4.0],
            "avg_usefulness": [2.5],
            "avg_usefulness_centered": [2.5],
            "goodreads_rating": [4.2],
        }
    )

    result, detail = evaluate_target_variant(
        train_df=train,
        holdout_df=holdout,
        target_col="avg_enjoyment",
        spec=new_books_to_rate_analysis.ACTIONABLE_FEATURE_SPECS[0],
        model_name="Global mean",
    )

    assert result["mae"] == pytest.approx(1.0)
    assert detail.columns.is_unique
    assert detail.loc[0, "prediction_error"] == pytest.approx(-1.0)


def test_requested_utility_starts_at_zero_and_uses_requested_bases() -> None:
    values = pd.Series([1.0, 3.0, 5.0])

    enjoy = requested_utility(values, "avg_enjoyment")
    useful = requested_utility(values, "avg_usefulness")

    assert enjoy.tolist() == pytest.approx([0.0, 0.69, 1.8561])
    assert useful.tolist() == pytest.approx([0.0, 2.24, 9.4976], rel=1e-6)


def test_policy_selection_uses_validation_thresholds_and_returns_uplift() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [1.0] * 6 + [4.0] * 6,
            "avg_usefulness": [1.0] * 6 + [4.0] * 6,
        }
    )

    selected = select_policy_rows(frame, "avg_usefulness")
    best = selected[selected["selection"] == "best_validation_utility"].iloc[0]
    realized = apply_policy_threshold(frame, "avg_usefulness", float(best["threshold"]))

    assert float(best["threshold"]) > 1.0
    assert int(best["n_keep"]) == 6
    assert realized["n_keep"] == 6
    assert realized["delta_keep_vs_all"] == pytest.approx(1.5)
    assert realized["delta_keep_utility_vs_all"] == pytest.approx(2.416)


def test_assign_policy_split_filters_to_goodreads_and_splits_chronologically() -> None:
    frame = pd.DataFrame(
        {
            "title": list("abcde"),
            "date_finished": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
            ],
            "goodreads_rating": [4.0, 4.1, None, 4.2, 4.3],
        }
    )

    split = assign_policy_split(frame)

    assert split["title"].tolist() == ["a", "b", "d", "e"]
    assert split["policy_split"].tolist() == [
        "validation",
        "validation",
        "test",
        "test",
    ]


def test_summarize_best_policy_rules_preserves_summary_selection_labels() -> None:
    split_results = pd.DataFrame(
        [
            {
                "target": "avg_enjoyment",
                "selection": "best_validation_utility",
                "validation_delta_keep_utility_vs_all": 0.2,
                "validation_delta_keep_vs_all": 0.1,
                "test_delta_keep_utility_vs_all": 0.0,
                "test_delta_keep_vs_all": 0.0,
            },
            {
                "target": "avg_enjoyment",
                "selection": "best_validation_balanced_utility",
                "validation_delta_keep_utility_vs_all": 0.1,
                "validation_delta_keep_vs_all": 0.05,
                "test_delta_keep_utility_vs_all": 0.3,
                "test_delta_keep_vs_all": 0.2,
            },
        ]
    )
    full_results = pd.DataFrame(
        [
            {
                "target": "avg_enjoyment",
                "selection": "best_full_holdout_utility",
                "delta_keep_utility_vs_all": 0.4,
                "delta_keep_vs_all": 0.25,
            }
        ]
    )

    split_best, full_best = summarize_best_policy_rules(split_results, full_results)

    assert split_best["selection"].tolist() == [
        "best_validation_rule",
        "best_realized_test_rule",
    ]
    assert full_best["selection"].tolist() == ["best_full_holdout_rule"]
