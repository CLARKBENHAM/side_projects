from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "improved_standardized_plots.py"
)


def load_module():
    spec = spec_from_file_location("improved_standardized_plots", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_optimal_weighted_thresholds_cover_three_to_five():
    module = load_module()

    assert np.isclose(module.OPTIMAL_WEIGHTED_THRESHOLDS[0], 3.0)
    assert np.isclose(module.OPTIMAL_WEIGHTED_THRESHOLDS[-1], 5.0)


def test_standard_error_of_mean_is_deterministic():
    module = load_module()

    series = pd.Series([3.0, 4.0, 5.0, 4.0])
    expected = float(series.std(ddof=1) / np.sqrt(len(series)))

    assert module.standard_error_of_mean(series) == expected


def test_format_pct_label_zero_pads():
    module = load_module()

    assert module.format_pct_label(1, 100) == "01"
    assert module.format_pct_label(9, 10) == "90"
    assert module.format_pct_label(0, 10) == "00"
    assert module.format_pct_label(10, 10) == "99"


def test_summarize_kept_books_uses_mean_gains_and_sem():
    module = load_module()
    kept = pd.DataFrame(
        {
            "avg_enjoyment": [4.0, 5.0, 3.0],
            "avg_usefulness": [2.0, 4.0, 3.0],
        }
    )

    summary = module.summarize_kept_books(
        kept,
        baseline_enjoy=3.5,
        baseline_useful=2.5,
    )

    assert summary["enjoy_gain"] == 0.5
    assert summary["useful_gain"] == 0.5
    assert summary["enjoy_se"] == float(
        kept["avg_enjoyment"].std(ddof=1) / np.sqrt(len(kept))
    )
    assert summary["useful_se"] == float(
        kept["avg_usefulness"].std(ddof=1) / np.sqrt(len(kept))
    )
