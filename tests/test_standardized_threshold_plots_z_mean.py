from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "standardized_threshold_plots_z_mean.py"
)


def load_module():
    spec = spec_from_file_location("standardized_threshold_plots_z_mean", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_threshold_grid_spans_observed_values_on_tenths():
    module = load_module()
    frame = pd.DataFrame({"overall_z_mean": [-1.23, -0.01, 0.47, 1.08]})

    thresholds = module.build_threshold_grid(frame, "overall_z_mean")

    assert np.isclose(thresholds[0], -1.3)
    assert np.isclose(thresholds[-1], 1.1)
    assert np.allclose(np.diff(thresholds), 0.1)
