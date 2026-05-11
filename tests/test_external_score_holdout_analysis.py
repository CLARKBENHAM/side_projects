from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "external_score_holdout_analysis.py"
)


def load_module():
    spec = spec_from_file_location("external_score_holdout_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_train_holdout_uses_source_column():
    module = load_module()
    frame = pd.DataFrame(
        {
            "source": ["Play Export", "Holdout 2026", "Play Export"],
            "gr": [4.0, 4.1, 4.2],
            "amz": [4.5, 4.6, 4.7],
        }
    )

    train, holdout = module.split_train_holdout(frame)

    assert len(train) == 2
    assert len(holdout) == 1
    assert holdout.iloc[0]["source"] == "Holdout 2026"


def test_histories_borrows_general_reading_train_stats_for_category_z():
    module = load_module()
    train = pd.DataFrame(
        {
            "category": ["General Reading", "General Reading"],
            "gr": [4.0, 4.4],
            "amz": [4.6, 4.8],
        }
    )
    holdout = pd.DataFrame(
        {
            "category": ["Histories"],
            "gr": [4.2],
            "amz": [4.7],
        }
    )

    scored = module.add_holdout_score_schemes(train, holdout)

    gr_mean = float(train["gr"].mean())
    gr_std = float(train["gr"].std())
    amz_mean = float(train["amz"].mean())
    amz_std = float(train["amz"].std())
    expected = ((4.2 - gr_mean) / gr_std + (4.7 - amz_mean) / amz_std) / 2
    assert np.isclose(scored.loc[0, "category_z_mean"], expected)


def test_select_best_keep_pct_uses_train_only():
    module = load_module()
    train = pd.DataFrame(
        {
            "scheme": list(range(10, 0, -1)),
            "avg_enjoyment": [3.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "scheme": list(range(10, 0, -1)),
            "avg_enjoyment": [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )

    keep_pct, train_gain = module.select_best_keep_pct(train, "scheme", "avg_enjoyment")
    holdout_gain = module.gain_at_keep_pct(holdout, "scheme", keep_pct, "avg_enjoyment")

    assert keep_pct == 50
    assert np.isclose(train_gain, 1.8)
    assert np.isclose(holdout_gain, -2.0)


def test_summarize_split_correlation_tracks_split_and_counts():
    module = load_module()
    frame = pd.DataFrame(
        {
            "sum_mean_fill": [8.0, 8.5],
            "avg_enjoyment": [3.0, 4.0],
        }
    )

    summary = module.summarize_split_correlation(
        frame, "Overall", "enjoyment", "holdout"
    )

    assert summary["category"] == "Overall"
    assert summary["split_name"] == "holdout"
    assert summary["n_books"] == 2
    assert np.isnan(summary["pearson_r"])
