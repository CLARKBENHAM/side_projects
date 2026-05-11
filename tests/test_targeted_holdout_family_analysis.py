from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "targeted_holdout_family_analysis.py"
)


def load_module():
    spec = spec_from_file_location("targeted_holdout_family_analysis", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_coarse_bucket_collapses_to_three_groups():
    module = load_module()

    assert module.coarse_bucket("fiction") == "Fiction/Literature"
    assert module.coarse_bucket("Literature") == "Fiction/Literature"
    assert module.coarse_bucket("Computer Science") == "Technical"
    assert module.coarse_bucket("Business, management") == "History/Biz/General"


def test_tree_variants_include_requested_bookshelf_and_coarse_comparisons():
    module = load_module()
    names = {row["variant_name"] for row in module.tree_variants()}

    assert "TREE_REQUESTED_BOOKSHELF" in names
    assert "TREE_REQUESTED_COARSE_BUCKET" in names
    assert "TREE_REQUESTED_NO_BOOK_AGE_BOOKSHELF" in names
    assert "TREE_REQUESTED_NO_GR_RATING_BOOKSHELF" in names
    assert "TREE_REQUESTED_NO_AMZN_LOG_COUNT_BOOKSHELF" in names
    assert "TREE_REQUESTED_AMZN_LOG_REGFILL_BOOKSHELF" in names


def test_linear_variants_include_requested_bookshelf_and_coarse_comparisons():
    module = load_module()
    names = {row["variant_name"] for row in module.linear_variants()}

    assert "LINEAR_REQUESTED_BOOKSHELF" in names
    assert "LINEAR_REQUESTED_COARSE_BUCKET" in names
    assert "LINEAR_REQUESTED_NO_BOOK_AGE_BOOKSHELF" in names
    assert "LINEAR_REQUESTED_NO_GR_RATING_BOOKSHELF" in names
    assert "LINEAR_REQUESTED_NO_AMZN_LOG_COUNT_BOOKSHELF" in names
    assert "LINEAR_REQUESTED_AMZN_LOG_REGFILL_BOOKSHELF" in names


def test_apply_amazon_log_count_from_goodreads_uses_model_then_fallback():
    module = load_module()
    train_raw = pd.DataFrame(
        {
            "goodreads_rating_count_verified": [9, 19, 39, 79, 159, 319, 639, 1279],
            "amazon_reviews_consensus": [19, 39, 79, 159, 319, 639, 1279, 2559],
        }
    )
    model, fallback = module.fit_amazon_log_count_from_goodreads(train_raw)
    assert model is not None
    assert fallback > 0

    raw_frame = pd.DataFrame(
        {
            "goodreads_rating_count_verified": [159, np.nan],
            "amazon_reviews_consensus": [np.nan, np.nan],
        },
        index=[4, 5],
    )
    prepared = pd.DataFrame(
        {
            "amazon_log_count_feature": [0.0, 0.0],
            "target_values": [2.0, 2.0],
        },
        index=[4, 5],
    )

    adjusted = module.apply_amazon_log_count_from_goodreads(
        raw_frame, prepared, model, fallback
    )

    assert adjusted.loc[4, "amazon_log_count_feature"] != fallback
    assert np.isclose(adjusted.loc[5, "amazon_log_count_feature"], fallback)


def test_book_age_over_1800_is_zero_for_typical_book_ages():
    module = load_module()
    ages = pd.Series([np.nan, -2, 0, 27, 823, 1900])

    transformed = module.book_age_over_1800(ages)

    assert np.allclose(transformed[:5], 0.0)
    assert transformed[5] == 100.0


def test_requested_simple_specs_cover_with_without_year_and_goodreads():
    module = load_module()
    specs = module.requested_simple_specs()
    names = {row["variant_name"] for row in specs}

    assert names == {
        "AMZN_BOOKSHELF_NO_YEAR",
        "AMZN_GR_BOOKSHELF_NO_YEAR",
        "AMZN_BOOKSHELF_WITH_YEAR",
        "AMZN_GR_BOOKSHELF_WITH_YEAR",
    }
