from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_books_tracking.new_books_to_rate_bootstrap import (
    evaluate_rule_arrays,
    rank_matrix,
    summarize_best_rules,
)


def test_evaluate_rule_arrays_returns_expected_observed_uplift() -> None:
    prediction_values = np.array([1.0, 1.0, 4.0, 4.0], dtype=float)
    actual = np.array([1.0, 2.0, 4.0, 5.0], dtype=float)
    utility = np.array([0.0, 1.0, 4.0, 9.0], dtype=float)

    candidate = evaluate_rule_arrays(
        prediction_values=prediction_values,
        actual=actual,
        utility=utility,
        threshold=3.0,
        min_side_n=2,
    )

    assert candidate is not None
    assert candidate.n_keep == 2
    assert candidate.keep_share == pytest.approx(0.5)
    assert candidate.observed_delta_keep_vs_all == pytest.approx(1.5)
    assert candidate.observed_delta_keep_utility_vs_all == pytest.approx(3.0)


def test_rank_matrix_assigns_best_rank_to_largest_values() -> None:
    metric_matrix = np.array(
        [
            [1.0, 3.0],
            [2.0, 1.0],
            [0.5, 2.0],
        ]
    )

    ranks = rank_matrix(metric_matrix)

    assert ranks[:, 0].tolist() == [2.0, 1.0, 3.0]
    assert ranks[:, 1].tolist() == [1.0, 3.0, 2.0]


def test_summarize_best_rules_preserves_summary_selection_labels() -> None:
    summary = pd.DataFrame(
        [
            {
                "target": "avg_enjoyment",
                "subset": "all",
                "selection": "old",
                "feature_spec": "spec_a",
                "model": "Ridge",
                "threshold": 3.5,
                "observed_delta_keep_vs_all": 0.1,
                "observed_delta_keep_utility_vs_all": 0.2,
                "bootstrap_mean_delta_keep_vs_all": 0.15,
                "bootstrap_mean_delta_keep_utility_vs_all": 0.25,
                "bootstrap_prob_utility_best": 0.6,
            },
            {
                "target": "avg_enjoyment",
                "subset": "all",
                "selection": "old",
                "feature_spec": "spec_b",
                "model": "GBM",
                "threshold": 3.7,
                "observed_delta_keep_vs_all": 0.3,
                "observed_delta_keep_utility_vs_all": 0.4,
                "bootstrap_mean_delta_keep_vs_all": 0.2,
                "bootstrap_mean_delta_keep_utility_vs_all": 0.22,
                "bootstrap_prob_utility_best": 0.3,
            },
        ]
    )

    best = summarize_best_rules(summary)

    assert set(best["selection"]) == {
        "best_observed_utility",
        "best_bootstrap_mean_utility",
        "highest_prob_utility_best",
        "best_bootstrap_mean_rating",
    }
