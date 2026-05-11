from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "external_score_scheme_analysis.py"
)


def load_module():
    spec = spec_from_file_location("external_score_scheme_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weighted_available_mean_respects_missingness():
    module = load_module()
    frame = pd.DataFrame({"gr": [4.0, np.nan, 5.0], "amz": [5.0, 4.0, np.nan]})

    score = module.weighted_available_mean(frame, "gr", "amz", 0.6, 0.4)

    assert np.isclose(score.iloc[0], 4.4)
    assert np.isclose(score.iloc[1], 4.0)
    assert np.isclose(score.iloc[2], 5.0)


def test_add_score_schemes_builds_expected_columns():
    module = load_module()
    frame = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "gr": [4.0, 4.2, 3.8, 4.1],
            "amz": [4.6, 4.7, 4.5, 4.4],
            "avg_enjoyment": [3.0, 4.0, 2.5, 3.5],
            "avg_usefulness": [2.0, 2.5, 1.5, 2.0],
        }
    )

    scored = module.add_score_schemes(frame, frame)

    expected = {
        "sum_zero_fill",
        "sum_mean_fill",
        "mean_available",
        "gr_heavy_mean",
        "amz_heavy_mean",
        "overall_z_mean",
        "category_z_mean",
    }
    assert expected.issubset(scored.columns)
    assert scored["category_z_mean"].notna().all()


def test_mean_filled_sum_uses_reference_means_for_missing_values():
    module = load_module()
    reference = pd.DataFrame({"gr": [4.0, 5.0], "amz": [4.2, 4.8]})
    frame = pd.DataFrame({"gr": [4.0, np.nan], "amz": [np.nan, 5.0]})

    score = module.mean_filled_sum(reference, frame, "gr", "amz")

    assert np.isclose(score.iloc[0], 4.0 + 4.5)
    assert np.isclose(score.iloc[1], 4.5 + 5.0)


def test_export_zscore_coefficients_matches_two_source_linear_formula():
    module = load_module()
    reference = pd.DataFrame({"gr": [4.0, 5.0], "amz": [4.2, 4.8]})
    gr_mean = float(reference["gr"].mean())
    amz_mean = float(reference["amz"].mean())
    gr_std = float(reference["gr"].std())
    amz_std = float(reference["amz"].std())

    coeffs = module.export_zscore_coefficients(reference)

    gr_row = coeffs[coeffs["component"] == "gr"].iloc[0]
    amz_row = coeffs[coeffs["component"] == "amz"].iloc[0]
    formula_row = coeffs[coeffs["component"] == "overall_z_mean"].iloc[0]
    assert np.isclose(gr_row["linear_coef_two_source_mean_z"], 0.5 / gr_std)
    assert np.isclose(amz_row["linear_coef_two_source_mean_z"], 0.5 / amz_std)
    assert np.isclose(
        formula_row["linear_intercept_component"],
        -0.5 * gr_mean / gr_std - 0.5 * amz_mean / amz_std,
    )


def test_gain_at_keep_pct_uses_top_fraction():
    module = load_module()
    frame = pd.DataFrame(
        {
            "score": [10.0, 9.0, 8.0, 7.0],
            "avg_enjoyment": [5.0, 4.0, 2.0, 1.0],
        }
    )

    gain = module.gain_at_keep_pct(frame, "score", 50, "avg_enjoyment")

    assert np.isclose(gain, 1.5)
