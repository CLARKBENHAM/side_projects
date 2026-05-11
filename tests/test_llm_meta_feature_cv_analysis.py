from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ai_books_tracking.scripts.temp import llm_meta_feature_cv_analysis as analysis


def test_manual_cv_predictions_recovers_simple_linear_signal() -> None:
    x_frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_series = pd.Series([1.2, 2.1, 3.0, 3.9, 4.8])
    estimator = analysis.model_builders()["Ridge"]

    predictions = analysis.manual_cv_predictions(estimator, x_frame, y_series)

    assert len(predictions) == len(y_series)
    assert np.corrcoef(predictions, y_series)[0, 1] > 0.95
    assert np.mean(np.abs(predictions - y_series.to_numpy())) < 0.35


def test_feature_sets_include_ai_columns_for_overlap() -> None:
    enjoyment_features = analysis.feature_sets_for_dataset("overlap20", "enjoyment")
    usefulness_features = analysis.feature_sets_for_dataset("overlap20", "usefulness")

    assert "best_numeric_plus_6ai" in enjoyment_features
    assert "claude_pred_enjoyment" in enjoyment_features["best_numeric_plus_6ai"]
    assert "codex_pred_usefulness" in usefulness_features["best_numeric_plus_6ai"]


def test_extract_linear_coefficients_returns_intercept() -> None:
    estimator = Pipeline(
        [
            ("imputer", analysis.SimpleImputer(strategy="median")),
            ("scaler", analysis.StandardScaler()),
            ("model", analysis.Ridge(alpha=1.0)),
        ]
    )
    x_frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    y_series = pd.Series([1.5, 2.5, 3.5])
    estimator.fit(x_frame, y_series)

    rows = analysis.extract_linear_coefficients(estimator, ["a", "b"])
    features = {row["feature"] for row in rows}

    assert features == {"a", "b", "intercept"}


def test_drop_gain_rows_matches_expected_gain() -> None:
    rows = analysis.drop_gain_rows(
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        drop_pcts=(50,),
    )

    assert math.isclose(rows[0]["baseline_mean"], 2.5)
    assert math.isclose(rows[0]["kept_mean"], 3.5)
    assert math.isclose(rows[0]["gain_vs_all"], 1.0)
