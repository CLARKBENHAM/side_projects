from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "predicted_score_threshold_analysis.py"
)


def load_module():
    spec = spec_from_file_location("predicted_score_threshold_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_train_holdout_uses_holdout_2026_only():
    module = load_module()
    frame = pd.DataFrame(
        {
            "source": ["Play Export", "Holdout 2026", "Play Export (unverified)"],
            "value": [1, 2, 3],
        }
    )

    train, holdout = module.split_train_holdout(frame)

    assert train["value"].tolist() == [1, 3]
    assert holdout["value"].tolist() == [2]


def test_build_cutoff_curve_uses_predicted_cutoff_and_gain():
    module = load_module()
    scored = pd.DataFrame(
        {
            "category": ["fiction"] * 10,
            "title": [f"Book {i}" for i in range(10)],
            "avg_enjoyment": [5, 5, 4, 4, 3, 3, 2, 2, 1, 1],
            "pred_avg_enjoyment": [5.0, 4.8, 4.4, 4.1, 3.8, 3.4, 3.0, 2.5, 2.0, 1.5],
        }
    )

    curve = module.build_cutoff_curve(scored, "fiction", "avg_enjoyment")
    row_50 = curve.loc[curve["drop_percent"] == 50].iloc[0]
    row_90 = curve.loc[curve["drop_percent"] == 90].iloc[0]

    assert np.isclose(row_50["cutoff_score"], 3.8)
    assert np.isclose(row_50["gain"], 1.2)
    assert int(row_50["n_kept"]) == 5
    assert np.isclose(row_90["cutoff_score"], 5.0)
    assert np.isclose(row_90["gain"], 2.0)


def test_select_policy_cutoffs_prefers_lower_one_se_drop():
    module = load_module()
    curve = pd.DataFrame(
        {
            "drop_percent": [40, 50, 60, 70, 80, 90],
            "cutoff_score": [2.8, 3.0, 3.2, 3.4, 3.6, 3.8],
            "gain": [0.2, 0.4, 0.45, 0.5, 0.48, 0.43],
            "bootstrap_mean": [0.18, 0.36, 0.42, 0.47, 0.49, 0.40],
            "bootstrap_sd": [0.03, 0.03, 0.04, 0.05, 0.09, 0.04],
        }
    )

    markers = module.select_policy_cutoffs(curve)

    assert np.isclose(markers["in_sample_best"], 3.4)
    assert np.isclose(markers["reasonable_bootstrap_best"], 3.6)
    assert np.isclose(markers["reasonable_bootstrap_one_se"], 3.2)


def test_build_coefficient_difference_table_flags_bootstrap_membership():
    module = load_module()
    all_coefficients = pd.DataFrame(
        {
            "fit_name": ["all_data", "all_data"],
            "group": ["G", "G"],
            "target": ["enjoyment", "enjoyment"],
            "feature": ["goodreads_rating_raw", "intercept"],
            "coefficient": [0.9, -1.2],
        }
    )
    train_coefficients = pd.DataFrame(
        {
            "fit_name": ["train_only", "train_only"],
            "group": ["G", "G"],
            "target": ["enjoyment", "enjoyment"],
            "feature": ["goodreads_rating_raw", "intercept"],
            "coefficient": [0.7, -1.0],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "group": ["G", "G"],
            "target": ["enjoyment", "enjoyment"],
            "feature": ["goodreads_rating_raw", "intercept"],
            "bootstrap_mean": [0.72, -1.02],
            "bootstrap_sd": [0.1, 0.2],
            "bootstrap_p2_5": [0.5, -1.5],
            "bootstrap_p97_5": [0.85, -0.6],
        }
    )

    comparison = module.build_coefficient_difference_table(
        all_coefficients, train_coefficients, bootstrap
    )
    gr_row = comparison.loc[comparison["feature"] == "goodreads_rating_raw"].iloc[0]
    intercept_row = comparison.loc[comparison["feature"] == "intercept"].iloc[0]

    assert np.isclose(gr_row["coef_delta_all_minus_train"], 0.2)
    assert bool(gr_row["all_data_within_train_bootstrap"]) is False
    assert bool(intercept_row["all_data_within_train_bootstrap"]) is True


def test_build_drop_gain_difference_table_compares_all_vs_holdout_band():
    module = load_module()
    curves_all = pd.DataFrame(
        {
            "category": ["Overall", "Overall"],
            "target": ["enjoyment", "enjoyment"],
            "drop_percent": [50, 60],
            "cutoff_score": [3.2, 3.4],
            "gain": [0.40, 0.55],
            "pct_kept": [50.0, 40.0],
            "n_kept": [10, 8],
            "n_total": [20, 20],
        }
    )
    curves_holdout = pd.DataFrame(
        {
            "category": ["Overall", "Overall"],
            "target": ["enjoyment", "enjoyment"],
            "drop_percent": [50, 60],
            "cutoff_score": [3.1, 3.3],
            "gain": [0.30, 0.65],
            "pct_kept": [50.0, 40.0],
            "n_kept": [5, 4],
            "n_total": [10, 10],
            "bootstrap_mean": [0.32, 0.60],
            "bootstrap_sd": [0.05, 0.07],
            "bootstrap_p10": [0.20, 0.40],
            "bootstrap_p90": [0.35, 0.70],
        }
    )

    comparison = module.build_drop_gain_difference_table(curves_all, curves_holdout)
    row_50 = comparison.loc[comparison["drop_percent"] == 50].iloc[0]
    row_60 = comparison.loc[comparison["drop_percent"] == 60].iloc[0]

    assert np.isclose(row_50["gain_delta_all_minus_holdout"], 0.10)
    assert bool(row_50["all_gain_within_holdout_bootstrap_band"]) is False
    assert bool(row_60["all_gain_within_holdout_bootstrap_band"]) is True


def test_adjusted_r2_returns_nan_when_n_too_small():
    module = load_module()

    assert np.isnan(module.adjusted_r2(0.2, 3, 2))
    assert np.isclose(module.adjusted_r2(0.2, 10, 2), -0.028571428571428692)


def test_openlibrary_ablation_wide_merges_and_computes_deltas():
    module = load_module()
    long_df = pd.DataFrame(
        {
            "scope_type": ["overall", "overall"],
            "scope_name": ["Overall", "Overall"],
            "target": ["usefulness", "usefulness"],
            "eval_name": ["all_data", "holdout"],
            "with_ol_n_eval": [100, 20],
            "with_ol_r2": [0.30, 0.10],
            "with_ol_adjusted_r2": [0.28, -0.01],
            "with_ol_rho": [0.50, 0.40],
            "without_ol_n_eval": [100, 20],
            "without_ol_r2": [0.25, 0.08],
            "without_ol_adjusted_r2": [0.24, 0.00],
            "without_ol_rho": [0.47, 0.35],
        }
    )

    wide = module.openlibrary_ablation_wide(long_df)
    row = wide.iloc[0]

    assert np.isclose(row["delta_r2_all_data"], 0.05)
    assert np.isclose(row["delta_adjusted_r2_holdout"], -0.01)
    assert np.isclose(row["delta_rho_holdout"], 0.05)


def test_tree_ablation_feature_sets_include_base_goodreads_and_amazon():
    module = load_module()
    feature_sets = module.tree_ablation_feature_sets(
        [
            "year_finished",
            "log_pages",
            "book_age",
            "goodreads_available",
            "goodreads_rating_feature",
            "goodreads_log_count_feature",
            "amazon_available",
            "amazon_rating_feature",
            "amazon_log_count_feature",
        ]
    )

    assert feature_sets["BASE_ONLY"] == ["year_finished", "log_pages", "book_age"]
    assert feature_sets["GR_ONLY"] == [
        "goodreads_available",
        "goodreads_rating_feature",
        "goodreads_log_count_feature",
    ]
    assert feature_sets["AMZN_ONLY"] == [
        "amazon_available",
        "amazon_rating_feature",
        "amazon_log_count_feature",
    ]
    assert feature_sets["BASE_PLUS_AMZN"] == [
        "year_finished",
        "log_pages",
        "book_age",
        "amazon_available",
        "amazon_rating_feature",
        "amazon_log_count_feature",
    ]


def test_attach_tree_ablation_deltas_tracks_full_and_drop_effects():
    module = load_module()
    raw = pd.DataFrame(
        {
            "model": ["GBM", "GBM", "GBM", "GBM"],
            "target": ["usefulness"] * 4,
            "ablation": [
                "FULL",
                "CATEGORY_ONLY",
                "BASE_PLUS_AMZN",
                "drop_goodreads_rating_feature",
            ],
            "rho_holdout": [0.52, 0.46, 0.64, 0.59],
            "r2_holdout": [0.10, 0.02, 0.20, -0.05],
            "mae_holdout": [0.58, 0.61, 0.51, 0.56],
        }
    )

    enriched = module.attach_tree_ablation_deltas(raw)
    best_row = enriched.loc[enriched["ablation"] == "BASE_PLUS_AMZN"].iloc[0]
    drop_row = enriched.loc[
        enriched["ablation"] == "drop_goodreads_rating_feature"
    ].iloc[0]

    assert np.isclose(best_row["delta_rho_vs_full"], 0.12)
    assert np.isclose(best_row["delta_rho_vs_category_only"], 0.18)
    assert np.isnan(best_row["feature_value_in_full_rho"])
    assert np.isclose(drop_row["delta_rho_vs_full"], 0.07)
    assert np.isclose(drop_row["feature_value_in_full_rho"], -0.07)
    assert np.isclose(drop_row["feature_value_in_full_mae"], -0.02)


def test_aggregate_tree_feature_importances_collapses_bookshelf_dummies():
    module = load_module()
    aggregated = module.aggregate_tree_feature_importances(
        [
            "num__year_finished",
            "num__amazon_rating_feature",
            "cat__Bookshelf_fiction",
            "cat__Bookshelf_Literature",
        ],
        np.array([0.2, 0.5, 0.1, 0.2]),
    )

    top_row = aggregated.iloc[0]
    bookshelf_row = aggregated.loc[aggregated["original_feature"] == "Bookshelf"].iloc[
        0
    ]

    assert top_row["original_feature"] == "amazon_rating_feature"
    assert np.isclose(top_row["importance"], 0.5)
    assert np.isclose(bookshelf_row["importance"], 0.3)
    assert int(bookshelf_row["rank"]) == 2


def test_targeted_tree_variant_specs_include_requested_comparisons():
    module = load_module()
    specs = module.targeted_tree_variant_specs(
        [
            "year_finished",
            "log_pages",
            "book_age",
            "goodreads_available",
            "goodreads_rating_feature",
            "goodreads_log_count_feature",
            "amazon_available",
            "amazon_rating_feature",
            "amazon_log_count_feature",
        ]
    )
    names = [spec["variant_name"] for spec in specs]

    assert "BASE_PLUS_AMZN" in names
    assert "BASE_PLUS_AMZN_NO_BOOK_AGE" in names
    assert "BASE_PLUS_AMZN_PLUS_GR_RATING" in names
    assert "BASE_PLUS_AMZN_NO_AMZN_LOG_COUNT" in names
    assert "BASE_PLUS_AMZN_AMZN_LOG_COUNT_FROM_GR_LOG_COUNT" in names


def test_apply_amazon_log_count_from_goodreads_uses_regression_then_fallback():
    module = load_module()
    train_raw = pd.DataFrame(
        {
            "goodreads_rating_count_verified": [9, 99, 999, 9999, 49, 199, 499, 799],
            "amazon_reviews_consensus": [19, 199, 1999, 19999, 99, 399, 999, 1599],
        }
    )
    model, fallback = module.fit_amazon_log_count_from_goodreads(train_raw)
    assert model is not None
    assert fallback > 0

    raw_frame = pd.DataFrame(
        {
            "goodreads_rating_count_verified": [99, np.nan],
            "amazon_reviews_consensus": [np.nan, np.nan],
        },
        index=[10, 11],
    )
    prepared = pd.DataFrame(
        {"amazon_log_count_feature": [0.0, 0.0]},
        index=[10, 11],
    )

    adjusted = module.apply_amazon_log_count_from_goodreads(
        raw_frame, prepared, model, fallback
    )

    assert adjusted.loc[10, "amazon_log_count_feature"] != fallback
    assert np.isclose(adjusted.loc[11, "amazon_log_count_feature"], fallback)
