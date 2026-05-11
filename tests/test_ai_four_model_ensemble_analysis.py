from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "ai_four_model_ensemble_analysis.py"
)


def load_module():
    spec = spec_from_file_location("ai_four_model_ensemble_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_choose_match_key_prefers_left_but_falls_back_right():
    module = load_module()
    sample_keys = {"left key", "right key"}
    assert module.choose_match_key("left key", "right key", sample_keys) == "left key"
    assert module.choose_match_key("missing", "right key", sample_keys) == "right key"
    assert module.choose_match_key("missing", "other", sample_keys) is None


def test_median_prediction_uses_middle_two_for_even_count():
    module = load_module()
    frame = pd.DataFrame({"a": [1.0], "b": [4.0], "c": [2.0], "d": [5.0]})
    median = module.median_prediction(frame, ["a", "b", "c", "d"])
    assert median.iloc[0] == 3.0


def test_consensus_count_counts_points_near_median():
    module = load_module()
    frame = pd.DataFrame(
        {
            "a": [3.0],
            "b": [3.5],
            "c": [2.5],
            "d": [4.5],
            "median_col": [3.25],
        }
    )
    count = module.consensus_count(frame, ["a", "b", "c", "d"], "median_col", 0.5)
    assert count.iloc[0] == 2
