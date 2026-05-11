from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "improved_standardized_plots_fixed.py"
)


def load_module():
    spec = spec_from_file_location("improved_standardized_plots_fixed", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reasonable_subset_respects_threshold_and_keep_bounds():
    module = load_module()
    summary = pd.DataFrame(
        {
            "threshold": [8.6, 8.7, 8.9, 9.2, 9.3],
            "pct_kept": [55.0, 50.0, 30.0, 6.0, 9.0],
            "enjoy_gain": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    subset = module.reasonable_subset(summary)

    assert subset["threshold"].tolist() == [8.7, 8.9]


def test_select_reasonable_threshold_uses_best_gain():
    module = load_module()
    summary = pd.DataFrame(
        {
            "threshold": [8.7, 8.8, 8.9],
            "pct_kept": [50.0, 40.0, 30.0],
            "enjoy_gain": [0.20, 0.35, 0.30],
            "enjoy_bootstrap_sd": [0.05, 0.04, 0.03],
        }
    )

    assert module.select_reasonable_threshold(summary, "enjoy_gain") == 8.8


def test_select_reasonable_threshold_one_se_prefers_lower_threshold():
    module = load_module()
    summary = pd.DataFrame(
        {
            "threshold": [8.7, 8.8, 8.9],
            "pct_kept": [50.0, 40.0, 30.0],
            "enjoy_bootstrap_mean": [0.31, 0.34, 0.33],
            "enjoy_bootstrap_sd": [0.02, 0.05, 0.03],
        }
    )

    selected = module.select_reasonable_threshold(
        summary,
        "enjoy_bootstrap_mean",
        se_col="enjoy_bootstrap_sd",
        one_se=True,
    )

    assert np.isclose(selected, 8.7)


def test_format_pct_label_is_zero_padded():
    module = load_module()

    assert module.format_pct_label(1, 100) == "01"
    assert module.format_pct_label(10, 10) == "99"
