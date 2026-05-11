from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_books_tracking.reconcile_cleaned_goodreads_metadata import (
    DATASET_SPECS,
    apply_curated_values,
    match_curated_rows,
    normalize_compact_key,
    normalize_filename_key,
    normalize_title_key,
    prepare_cleaned_frame,
    prepare_target_frame,
)


def test_match_curated_rows_handles_filename_cleanup_compact_titles_and_aliases() -> (
    None
):
    cleaned = pd.DataFrame(
        [
            {
                "source": "Play Export",
                "title": "Good Old Neon",
                "corrected_author": "David Foster Wallace",
                "filename": "GoodOldNeon(1).pdf",
                "ratings": 4.53,
                "number reviews": 171,
                "number ratings": 1059,
            },
            {
                "source": "Play Export",
                "title": "e unibus pluram television and u.s. fiction",
                "corrected_author": "David Foster Wallace",
                "filename": "DFW_TV(1).pdf",
                "ratings": 4.34,
                "number reviews": 73,
                "number ratings": 358,
            },
        ]
    )
    for column, fn in [
        ("title_key", normalize_title_key),
        ("title_compact_key", normalize_compact_key),
        ("filename_key", normalize_filename_key),
    ]:
        source_col = "title" if "title" in column else "filename"
        cleaned[column] = cleaned[source_col].map(fn)

    target = pd.DataFrame(
        {
            "title": ["GoodOldNeon.pdf", "DFW_TV.pdf"],
            "filename_ratings2": ["GoodOldNeon.pdf", ""],
        }
    )
    prepared = prepare_target_frame(target, DATASET_SPECS[0])

    matched = match_curated_rows(prepared, cleaned, DATASET_SPECS[0])

    assert matched["goodreads_cleaned_match_method"].tolist() == [
        "filename_key",
        "manual_alias",
    ]
    assert matched["goodreads_title_cleaned"].tolist() == [
        "Good Old Neon",
        "e unibus pluram television and u.s. fiction",
    ]
    assert matched["goodreads_cleaned_has_value"].tolist() == [True, True]


def test_apply_curated_values_preserves_scraped_columns_and_replaces_analysis_fields() -> (
    None
):
    frame = pd.DataFrame(
        [
            {
                "title": "Example",
                "goodreads_rating": 3.8,
                "goodreads_rating_count": 200,
                "goodreads_rating_raw": 3.7,
                "goodreads_rating_count_raw": 180,
                "goodreads_rating_raw_best": 3.7,
                "goodreads_rating_count_raw_best": 180,
                "goodreads_status": "review",
                "goodreads_match_method": "goodreads_autocomplete",
                "goodreads_title": "Wrong Example",
                "goodreads_author": "Wrong Author",
                "goodreads_match_score": 0.42,
                "goodreads_cleaned_row_found": True,
                "goodreads_cleaned_has_value": True,
                "goodreads_cleaned_status": "matched",
                "goodreads_cleaned_match_method": "manual_alias",
                "goodreads_title_cleaned": "Example",
                "goodreads_author_cleaned": "Correct Author",
                "goodreads_rating_cleaned": 4.2,
                "goodreads_review_count_cleaned": 10,
                "goodreads_rating_count_cleaned": 999,
                "goodreads_cleaned_source": "Play Export",
            }
        ]
    )

    curated = apply_curated_values(frame, DATASET_SPECS[0]).iloc[0]

    assert curated["goodreads_rating_old_scraped"] == 3.8
    assert curated["goodreads_rating_raw_best_old_scraped"] == 3.7
    assert curated["goodreads_status_old_scraped"] == "review"
    assert curated["goodreads_rating"] == 4.2
    assert curated["goodreads_rating_raw_best"] == 4.2
    assert curated["goodreads_rating_count_raw_best"] == 999
    assert curated["goodreads_status"] == "matched"
    assert curated["goodreads_match_method"] == "manual_alias"
    assert curated["goodreads_raw_best_source"] == "master_book_metadata_cleaned"


def test_prepare_cleaned_frame_accepts_original_goodreads_column_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cleaned.csv"
    pd.DataFrame(
        [
            {
                "title": "Example",
                "source": "Play Export",
                "filename": "Example.pdf",
                "corrected_author": "Author",
                "goodread ratings": "4.2",
                "goodreads number reviews": "10",
                "goodreads number ratings": "999",
            }
        ]
    ).to_csv(path, index=False)

    prepared = prepare_cleaned_frame(path)

    assert prepared.loc[0, "ratings"] == 4.2
    assert prepared.loc[0, "number reviews"] == 10
    assert prepared.loc[0, "number ratings"] == 999
