from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "gemini_two_run_average_analysis.py"
)


def load_module():
    spec = spec_from_file_location("gemini_two_run_average_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agreement_mask_is_target_specific():
    module = load_module()
    frame = pd.DataFrame(
        {
            "llm_pred_enjoyment_run_a": [3.0, 4.0, 2.5],
            "llm_pred_enjoyment_run_b": [3.25, 5.0, 3.5],
            "llm_pred_usefulness_run_a": [2.0, 2.0, 2.0],
            "llm_pred_usefulness_run_b": [2.5, 2.25, 3.5],
        }
    )
    enjoyment_mask = module.agreement_mask(frame, "enjoyment", 0.5)
    usefulness_mask = module.agreement_mask(frame, "usefulness", 0.5)
    assert enjoyment_mask.tolist() == [True, False, False]
    assert usefulness_mask.tolist() == [True, True, False]


def test_drop_gain_rows_uses_floor_drop_count():
    module = load_module()
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    score = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    rows = module.drop_gain_rows(actual, score)
    sixty = [row for row in rows if row["drop_pct"] == 60][0]
    assert sixty["n_kept"] == 2
    assert sixty["kept_mean"] == 4.5
    assert sixty["gain_vs_all"] == 1.5
