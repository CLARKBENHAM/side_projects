from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "gemini_holdout_blend_analysis.py"
)


def load_module():
    spec = spec_from_file_location("gemini_holdout_blend_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rank_average_respects_mean_rank():
    module = load_module()
    frame = pd.DataFrame({"a": [1, 3, 2], "b": [2, 1, 3]})
    blended = module.rank_average(frame, ["a", "b"])
    assert blended.iloc[2] > blended.iloc[1]
    assert blended.iloc[1] > blended.iloc[0]


def test_drop_gain_matches_manual_keep_set():
    module = load_module()
    actual = pd.Series([1.0, 2.0, 3.0, 5.0, 4.0])
    score = pd.Series([0.1, 0.2, 0.3, 0.5, 0.4])
    gain = module.drop_gain(actual, score, drop_pct=60)
    assert gain["n_kept"] == 2
    assert gain["kept_mean"] == 4.5
    assert gain["gain_vs_all"] == 1.5
