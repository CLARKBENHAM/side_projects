from __future__ import annotations

import math

import pandas as pd

from ai_books_tracking.scripts.temp import (
    model_recommendation_divergence_analysis as div,
)


def test_rank_percentile_assigns_higher_scores_higher_percentiles() -> None:
    series = pd.Series([1.0, 2.0, 4.0, 3.0], index=["a", "b", "c", "d"])
    pct = div.rank_percentile(series)

    assert pct["c"] > pct["d"] > pct["b"] > pct["a"]
    assert math.isclose(float(pct.max()), 1.0)


def test_compute_pairwise_divergence_reports_full_overlap_for_identical_models() -> (
    None
):
    frame = pd.DataFrame(
        {
            "avg_enjoyment": [1.0, 2.0, 3.0, 4.0],
            "avg_usefulness": [1.0, 2.0, 3.0, 4.0],
            "baseline_gbm_enjoyment": [1.0, 2.0, 3.0, 4.0],
            "baseline_ridge_enjoyment": [1.0, 2.0, 3.0, 4.0],
            "all10_enjoyment_mean": [1.0, 2.0, 3.0, 4.0],
            "prompt2_enjoyment_mean": [1.0, 2.0, 3.0, 4.0],
            "prompt3_enjoyment_mean": [1.0, 2.0, 3.0, 4.0],
            "family3_enjoyment_mean": [1.0, 2.0, 3.0, 4.0],
            "family3_enjoyment_median": [1.0, 2.0, 3.0, 4.0],
            "all13_enjoyment_mean": [1.0, 2.0, 3.0, 4.0],
            "simple_rf_usefulness": [1.0, 2.0, 3.0, 4.0],
            "baseline_gbm_usefulness": [1.0, 2.0, 3.0, 4.0],
            "all10_usefulness_mean": [1.0, 2.0, 3.0, 4.0],
            "prompt2_usefulness_mean": [1.0, 2.0, 3.0, 4.0],
            "prompt3_usefulness_mean": [1.0, 2.0, 3.0, 4.0],
            "family3_usefulness_mean": [1.0, 2.0, 3.0, 4.0],
            "all13_usefulness_mean": [1.0, 2.0, 3.0, 4.0],
            "all13_usefulness_median": [1.0, 2.0, 3.0, 4.0],
        }
    )

    pairwise = div.compute_pairwise_divergence(frame)
    row = pairwise[
        pairwise["target"].eq("enjoyment")
        & pairwise["model_a"].eq("numeric_gbm")
        & pairwise["model_b"].eq("numeric_ridge")
    ].iloc[0]

    assert math.isclose(row["spearman_between_predictions"], 1.0)
    assert math.isclose(row["mean_abs_score_diff"], 0.0)
    assert math.isclose(row["keep20_jaccard"], 1.0)


def test_shared_misses_for_representatives_finds_shared_over_and_under() -> None:
    rows = []
    for model, gaps in [
        ("numeric_gbm", [0.4, -0.5, 0.1]),
        ("llm_family3_median", [0.3, -0.4, -0.2]),
        ("numeric_simple_rf", [0.5, -0.3, 0.0]),
        ("llm_all13_mean", [0.4, -0.2, 0.1]),
    ]:
        target = (
            "enjoyment"
            if model in {"numeric_gbm", "llm_family3_median"}
            else "usefulness"
        )
        for idx, gap in enumerate(gaps, start=1):
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "book_id": f"B{idx:03d}",
                    "key": f"k{idx}",
                    "display_title": f"Book {idx}",
                    "display_author": "A",
                    "display_category": "fiction",
                    "actual_rating": float(idx),
                    "predicted_rating": float(idx) + gap,
                    "actual_rank_desc": float(idx),
                    "predicted_rank_desc": float(idx),
                    "actual_pct": 0.5,
                    "predicted_pct": 0.5 + gap,
                    "recommendation_gap_pct": gap,
                    "abs_recommendation_gap_pct": abs(gap),
                    "score_error": gap,
                }
            )
    per_book = pd.DataFrame(rows)

    shared, rep = div.shared_misses_for_representatives(per_book)

    assert not shared.empty
    assert {"shared_overrecommended", "shared_underrecommended"}.issubset(
        set(shared["slice"])
    )
    assert set(rep["target"]) == {"enjoyment", "usefulness"}
