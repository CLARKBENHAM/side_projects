from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "temporal_chunk_cv_experiments.py"
)


def load_module():
    spec = spec_from_file_location("temporal_chunk_cv_experiments", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_temporal_folds_merges_small_adjacent_semesters():
    module = load_module()
    frame = pd.DataFrame(
        {
            "estimated_finish": [
                "2020-01-10",
                "2020-03-10",
                "2020-07-10",
                "2020-09-10",
                "2021-01-10",
                "2021-02-10",
                "2021-03-10",
                "2021-04-10",
                "2021-07-10",
                "2021-08-10",
                "2021-09-10",
                "2021-10-10",
            ],
            "target": np.arange(12, dtype=float),
        }
    )

    folds, eval_frame, fold_table = module.build_temporal_folds(
        frame, "toy_family", "target", min_test_books=4, min_train_books=4
    )

    assert len(eval_frame) == 12
    assert fold_table.loc[0, "semester_labels"] == "2020-H1, 2020-H2"
    assert not bool(fold_table.loc[0, "evaluable"])
    assert bool(fold_table.loc[1, "evaluable"])
    assert [fold["fold_id"] for fold in folds] == ["target_02", "target_03"]


def test_score_metrics_handles_rank_and_scale_outputs():
    module = load_module()
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    predicted = pd.Series([1.2, 2.1, 2.9, 3.8])

    scale_metrics = module.score_metrics(actual, predicted, prediction_scale=True)
    rank_metrics = module.score_metrics(actual, predicted, prediction_scale=False)

    assert scale_metrics["rho"] > 0.9
    assert scale_metrics["r2"] > 0.9
    assert scale_metrics["mae"] < 0.3
    assert np.isnan(rank_metrics["r2"])
    assert np.isnan(rank_metrics["mae"])


def test_policy_metrics_returns_valid_keep_pct_from_grid():
    module = load_module()
    train_score = pd.Series(np.linspace(0, 1, 20))
    train_target = pd.Series(np.linspace(0, 2, 20))
    test_score = pd.Series(np.linspace(0, 1, 20))
    test_target = pd.Series(np.linspace(0, 2, 20))

    metrics = module.policy_metrics(train_score, train_target, test_score, test_target)

    assert metrics["selected_keep_pct"] in set(module.EXT_SCHEME.KEEP_PCTS.tolist())
    assert np.isfinite(metrics["policy_gain"])
