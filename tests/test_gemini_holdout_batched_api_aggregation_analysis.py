from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ai_books_tracking.scripts.temp import (
    gemini_holdout_batched_api_aggregation_analysis as agg,
)


def make_wide_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "book_id": ["B001", "B002", "B003", "B004"],
            "key": ["a", "b", "c", "d"],
            "display_title": ["A", "B", "C", "D"],
            "display_author": ["AA", "BB", "CC", "DD"],
            "display_category": ["fiction", "fiction", "history", "history"],
            "avg_enjoyment": [1.0, 2.0, 3.0, 4.0],
            "avg_usefulness": [1.5, 2.5, 3.5, 4.5],
        }
    )
    for sample_id in ("p1_r01", "p1_r02", "p2_r01"):
        frame[f"enjoyment_{sample_id}"] = frame["avg_enjoyment"]
        frame[f"usefulness_{sample_id}"] = frame["avg_usefulness"]
    return frame


def make_long_frame() -> pd.DataFrame:
    rows = []
    for prompt_id, sample_id in ((1, "p1_r01"), (1, "p1_r02"), (2, "p2_r01")):
        for idx, (actual_enjoy, actual_useful) in enumerate(
            zip([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5], strict=False),
            start=1,
        ):
            rows.append(
                {
                    "book_id": f"B{idx:03d}",
                    "prompt_id": prompt_id,
                    "sample_id": sample_id,
                    "avg_enjoyment": actual_enjoy,
                    "avg_usefulness": actual_useful,
                    "predicted_enjoyment": actual_enjoy,
                    "predicted_usefulness": actual_useful,
                }
            )
    return pd.DataFrame(rows)


def test_drop_curve_rows_returns_expected_kept_mean() -> None:
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    score = pd.Series([1.0, 2.0, 3.0, 4.0])

    rows = agg.drop_curve_rows(actual, score, drop_pcts=(50,))

    assert rows[0]["n_kept"] == 2
    assert math.isclose(rows[0]["baseline_mean"], 2.5)
    assert math.isclose(rows[0]["kept_mean"], 3.5)
    assert math.isclose(rows[0]["gain_vs_all"], 1.0)


def test_exact_subset_metrics_match_perfect_predictions() -> None:
    wide_df = make_wide_frame()

    accuracy_df, drop_df = agg.exact_subset_metrics(wide_df, drop_pcts=(50,))
    summary = agg.summarize_subset_accuracy(accuracy_df)
    drop_summary = agg.summarize_subset_drop(drop_df)

    assert len(accuracy_df) == 2 * (3 + 3 + 1)
    assert np.allclose(summary["spearman_rho_mean"], 1.0)
    assert np.allclose(summary["mae_mean"], 0.0)
    gain_rows = drop_summary[drop_summary["drop_pct"].eq(50)]
    assert np.allclose(gain_rows["gain_mean"], 1.0)
    assert set(np.round(gain_rows["kept_mean_mean"], 6)) == {3.5, 4.0}


def test_prompt_scatter_stats_include_prompt_means_and_all10() -> None:
    wide_df = make_wide_frame().copy()
    wide_df["prompt1_enjoyment_mean"] = wide_df["avg_enjoyment"]
    wide_df["prompt2_enjoyment_mean"] = wide_df["avg_enjoyment"]
    wide_df["all10_enjoyment_mean"] = wide_df["avg_enjoyment"]
    wide_df["prompt1_usefulness_mean"] = wide_df["avg_usefulness"]
    wide_df["prompt2_usefulness_mean"] = wide_df["avg_usefulness"]
    wide_df["all10_usefulness_mean"] = wide_df["avg_usefulness"]

    stats = agg.prompt_scatter_stats(make_long_frame(), wide_df)

    expected_series = {
        "Prompt 1 pooled points",
        "Prompt 2 pooled points",
        "Prompt 1 mean",
        "Prompt 2 mean",
        "All 10 mean",
    }
    assert set(stats["series"]) == expected_series
    assert np.allclose(stats["spearman_rho"], 1.0)
