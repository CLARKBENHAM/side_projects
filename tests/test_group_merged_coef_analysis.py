from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "group_merged_coef_analysis.py"
)


def load_module():
    spec = spec_from_file_location("group_merged_coef_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_map_group_matches_expected_categories():
    module = load_module()

    assert module.map_group("Business, management") == "business"
    assert module.map_group("fiction") == "fiction"
    assert module.map_group("Computer Science") == "technical"
    assert module.map_group("Literature") is None


def test_add_group_proxy_scores_uses_supplied_coefficients():
    module = load_module()
    frame = pd.DataFrame(
        {
            "category": [
                "Business, management",
                "fiction",
                "Computer Science",
                "Literature",
            ],
            "gr": [4.2, 4.6, 4.8, 4.1],
            "amz": [4.4, 4.7, 4.3, 4.2],
        }
    )

    scored = module.add_group_proxy_scores(frame)

    business = -0.081924 + 0.803572 * 4.2 + 0.064256 * 4.4
    fiction = -3.258460 + 0.448996 * 4.6 + 0.637394 * 4.7
    technical = 1.916578 + 0.780538 * 4.8 - 0.466253 * 4.3
    assert np.isclose(scored.loc[0, "merged_proxy_enjoyment"], business)
    assert np.isclose(scored.loc[1, "merged_proxy_usefulness"], fiction)
    assert np.isclose(scored.loc[2, "merged_proxy_enjoyment"], technical)
    assert np.isnan(scored.loc[3, "merged_proxy_enjoyment"])


def test_prepare_analysis_frame_filters_to_supported_complete_rows():
    module = load_module()
    frame = pd.DataFrame(
        {
            "category": ["fiction", "Literature", "Business, management"],
            "gr": [4.1, 4.2, np.nan],
            "amz": [4.5, 4.0, 4.3],
            "avg_enjoyment": [4.0, 3.0, 5.0],
            "avg_usefulness": [3.5, 2.5, 4.0],
        }
    )

    scored = module.add_score_schemes(frame, frame)
    scored = module.add_group_proxy_scores(scored)
    covered = scored[scored["coef_group"].notna()].copy()
    covered = covered[covered["gr"].notna() & covered["amz"].notna()].copy()

    assert len(covered) == 1
    assert covered.iloc[0]["category"] == "fiction"
