from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ai_books_tracking.scripts.temp import (
    gemini_holdout_blend_stability_analysis as analysis,
)


def test_build_full68_candidates_includes_prompt_and_numeric_blends() -> None:
    frame = pd.DataFrame(
        {
            "prompt1_enjoyment_mean": [2.0, 4.0],
            "prompt2_enjoyment_mean": [3.0, 5.0],
            "all10_enjoyment_mean": [2.5, 4.5],
            "baseline_ridge_enjoyment": [1.0, 3.0],
            "baseline_gbm_enjoyment": [1.5, 3.5],
        }
    )

    candidates = analysis.build_full68_candidates(frame, "enjoyment")

    assert "All10 + baseline_gbm_enjoyment avg" in candidates
    assert "Prompt1 + Prompt2 + baseline_ridge_enjoyment avg" in candidates
    assert np.allclose(
        candidates["Prompt1 + Prompt2 + baseline_ridge_enjoyment avg"],
        [2.0, 4.0],
    )


def test_trimmed_mean_drops_min_and_max() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [2.0, 3.0],
            "c": [3.0, 4.0],
            "d": [4.0, 5.0],
            "e": [5.0, 6.0],
            "f": [9.0, 9.0],
        }
    )

    result = analysis.trimmed_mean(frame, ["a", "b", "c", "d", "e", "f"])

    assert np.allclose(result, [3.5, 4.5])


def test_pairwise_prompt_stability_detects_identical_runs() -> None:
    frame = pd.DataFrame(
        {
            "enjoyment_p1_r01": [1.0, 2.0, 3.0],
            "enjoyment_p1_r02": [1.0, 2.0, 3.0],
            "enjoyment_p1_r03": [1.0, 2.0, 3.0],
            "enjoyment_p1_r04": [1.0, 2.0, 3.0],
            "enjoyment_p1_r05": [1.0, 2.0, 3.0],
            "enjoyment_p2_r01": [1.0, 2.0, 3.0],
            "enjoyment_p2_r02": [1.0, 2.0, 3.0],
            "enjoyment_p2_r03": [1.0, 2.0, 3.0],
            "enjoyment_p2_r04": [1.0, 2.0, 3.0],
            "enjoyment_p2_r05": [1.0, 2.0, 3.0],
            "usefulness_p1_r01": [1.5, 2.5, 3.5],
            "usefulness_p1_r02": [1.5, 2.5, 3.5],
            "usefulness_p1_r03": [1.5, 2.5, 3.5],
            "usefulness_p1_r04": [1.5, 2.5, 3.5],
            "usefulness_p1_r05": [1.5, 2.5, 3.5],
            "usefulness_p2_r01": [1.5, 2.5, 3.5],
            "usefulness_p2_r02": [1.5, 2.5, 3.5],
            "usefulness_p2_r03": [1.5, 2.5, 3.5],
            "usefulness_p2_r04": [1.5, 2.5, 3.5],
            "usefulness_p2_r05": [1.5, 2.5, 3.5],
        }
    )

    _, summary = analysis.pairwise_prompt_stability(frame)

    assert np.allclose(summary["pairwise_rho_mean"], 1.0)
    assert np.allclose(summary["pairwise_mae_mean"], 0.0)
    assert np.allclose(summary["per_book_unique_mean"], 1.0)
    assert np.allclose(summary["per_book_std_mean"], 0.0)


def test_normalize_key_handles_punctuation() -> None:
    result = analysis.normalize_key("The Power Broker!", "Robert A. Caro")

    assert result == "the power broker || robert a caro"
    assert math.isclose(
        analysis.drop_gain_rows(
            pd.Series([1.0, 2.0, 3.0, 4.0]),
            pd.Series([1.0, 2.0, 3.0, 4.0]),
            drop_pcts=(50,),
        )[0]["gain_vs_all"],
        1.0,
    )
