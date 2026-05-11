from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_books_tracking.multi_source_cleaning_analysis import (
    SOURCE_EXTERNAL_COLUMNS,
    build_suspicious_rows,
    linear_terms_in_raw_space,
    merge_master_sources,
    parse_combined_field,
)


def test_parse_combined_field_handles_urls_and_refusal_text() -> None:
    rating, count, url, bad = parse_combined_field("4.7|248|https://example.com/book")
    assert rating == 4.7
    assert count == 248
    assert url == "https://example.com/book"
    assert bad is False

    rating, count, url, bad = parse_combined_field(
        "I'm still learning and can't help with that. Do you need help with anything else?"
    )
    assert math.isnan(rating)
    assert math.isnan(count)
    assert url == ""
    assert bad is True


def test_merge_master_sources_prefers_filename_match() -> None:
    golden = pd.DataFrame(
        [
            {
                "title": "Short Buckley",
                "author": "Sam Tanenhaus",
                "category": "General Reading",
                "estimated_finish": "2026-01-01",
                "source": "Holdout 2026",
                "filename": "Buckley.html",
            }
        ]
    )
    master_row = {
        "title": "Buckley: The Life and the Revolution That Changed America",
        "filename": "Buckley.html",
        "source": "Holdout 2026",
    }
    for column in SOURCE_EXTERNAL_COLUMNS:
        master_row[column] = "4.2|100" if column == "Amazon no links" else pd.NA
    master = pd.DataFrame([master_row])

    merged = merge_master_sources(golden, master)

    assert bool(merged.loc[0, "external_master_match_found"]) is True
    assert merged.loc[0, "external_master_match_method"] == "filename"
    assert merged.loc[0, "Amazon no links"] == "4.2|100"


def test_build_suspicious_rows_returns_empty_with_stable_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "title": "Book A",
                "author": "Author",
                "source": "Play Export",
                "category": "General Reading",
                "goodreads_rating_verified": 4.0,
                "goodreads_review_count_verified": 100,
                "ol_rating_consensus": 4.0,
                "ol_reviews_consensus": 50,
                "amazon_rating_consensus": 4.1,
                "amazon_reviews_consensus": 80,
                "amazon_link_bad_text": False,
                "amazon_nolink_bad_text": False,
                "ol_link_bad_text": False,
                "ol_nolink_bad_text": False,
                "amazon_rating_variant_abs_diff": 0.0,
                "ol_rating_variant_abs_diff": 0.0,
                "amazon_review_variant_ratio": 1.0,
                "ol_review_variant_ratio": 1.0,
                "amazon_rating_resid_z_link": 0.0,
                "amazon_rating_resid_z_nolink": 0.0,
                "amazon_reviews_resid_z_link": 0.0,
                "amazon_reviews_resid_z_nolink": 0.0,
                "ol_rating_resid_z_link": 0.0,
                "ol_rating_resid_z_nolink": 0.0,
                "ol_reviews_resid_z_link": 0.0,
                "ol_reviews_resid_z_nolink": 0.0,
            }
        ]
    )

    suspicious = build_suspicious_rows(frame)

    assert suspicious.empty
    assert "issues" in suspicious.columns


def test_linear_terms_in_raw_space_removes_standardization() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "Bookshelf": ["A", "B", "A"],
        }
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["x"]),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                ),
                ["Bookshelf"],
            ),
        ]
    )
    preprocessor.fit(frame)

    raw_intercept, raw_coefficients = linear_terms_in_raw_space(
        preprocessor=preprocessor,
        numeric_features=["x"],
        transformed_coefficients=np.array([2.0, 3.0]),
        transformed_intercept=10.0,
    )

    assert math.isclose(raw_coefficients["x"], 2.0 / np.std([1.0, 2.0, 3.0], ddof=0))
    category_terms = {
        key: value for key, value in raw_coefficients.items() if key != "x"
    }
    assert len(category_terms) == 1
    assert math.isclose(next(iter(category_terms.values())), 3.0)
    expected_intercept = 10.0 - raw_coefficients["x"] * 2.0
    assert math.isclose(raw_intercept, expected_intercept)
