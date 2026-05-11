from __future__ import annotations

import math

import pandas as pd

from ai_books_tracking.multi_source_category_diagnostics import (
    best_complete_case_rule,
    category_correlation_rows,
    missing_counts_by_year_category,
)


def test_best_complete_case_rule_prefers_higher_correlation_then_lower_mae() -> None:
    summary = pd.DataFrame(
        [
            {
                "target": "avg_enjoyment",
                "rule_name": "a",
                "mode": "complete_case",
                "spearman_rho": 0.2,
                "mae": 0.7,
            },
            {
                "target": "avg_enjoyment",
                "rule_name": "b",
                "mode": "median_with_flags",
                "spearman_rho": 0.9,
                "mae": 0.1,
            },
            {
                "target": "avg_enjoyment",
                "rule_name": "c",
                "mode": "complete_case",
                "spearman_rho": 0.2,
                "mae": 0.6,
            },
            {
                "target": "avg_enjoyment",
                "rule_name": "d",
                "mode": "complete_case",
                "spearman_rho": 0.3,
                "mae": 0.8,
            },
        ]
    )

    row = best_complete_case_rule(summary, "avg_enjoyment")

    assert row["rule_name"] == "d"


def test_category_correlation_rows_summarizes_per_category() -> None:
    holdout = pd.DataFrame(
        {
            "target": ["avg_enjoyment"] * 4,
            "rule_name": ["rule"] * 4,
            "mode": ["complete_case"] * 4,
            "category": ["A", "A", "B", "B"],
            "avg_enjoyment": [1.0, 2.0, 2.0, 4.0],
            "score": [1.0, 3.0, 2.5, 3.5],
        }
    )

    summary = category_correlation_rows(
        holdout, "avg_enjoyment", "rule", "complete_case"
    )

    assert set(summary["category"]) == {"A", "B"}
    assert set(summary["n"]) == {2}


def test_missing_counts_by_year_category_counts_na_by_source() -> None:
    merged = pd.DataFrame(
        {
            "estimated_finish": ["2025-01-01", "2025-02-01", "2026-01-01"],
            "category": ["Histories", "Histories", "Literature"],
            "goodreads_rating_verified": [4.0, math.nan, math.nan],
            "ol_rating_consensus": [4.0, 4.2, math.nan],
            "amazon_rating_consensus": [math.nan, 4.5, 4.7],
        }
    )

    missing = missing_counts_by_year_category(merged)

    row = missing[
        (missing["source"] == "goodreads")
        & (missing["year_label"] == "2025")
        & (missing["category"] == "Histories")
    ].iloc[0]
    assert row["missing_count"] == 1
    assert row["total_count"] == 2
