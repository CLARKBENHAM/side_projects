from __future__ import annotations

import pandas as pd

from ai_books_tracking.compare_master_metadata_versions import (
    canonicalize_columns,
    compare_shared_columns,
    summarize_added_columns,
)


def test_compare_shared_columns_treats_renamed_goodreads_fields_as_equivalent() -> None:
    original = pd.DataFrame(
        [
            {
                "title": "Book A",
                "source": "Play Export",
                "filename": "book_a.pdf",
                "goodread ratings": 4.2,
                "goodreads number reviews": 12,
                "goodreads number ratings": 345,
            }
        ]
    )
    final = pd.DataFrame(
        [
            {
                "title": "Book A",
                "source": "Play Export",
                "filename": "book_a.pdf",
                "ratings": 4.2,
                "number reviews": 12,
                "number ratings": 345,
                "avg_enjoyment": 3.5,
            }
        ]
    )

    diffs = compare_shared_columns(
        canonicalize_columns(original),
        canonicalize_columns(final),
    )

    assert diffs.empty


def test_summarize_added_columns_reports_non_null_coverage() -> None:
    original = pd.DataFrame(
        [{"title": "Book A", "source": "Play Export", "filename": ""}]
    )
    final = pd.DataFrame(
        [
            {
                "title": "Book A",
                "source": "Play Export",
                "filename": "",
                "avg_enjoyment": 3.5,
                "avg_usefulness": None,
            }
        ]
    )

    summary = summarize_added_columns(original, final)
    summary = summary.set_index("column")

    assert summary.loc["avg_enjoyment", "non_null_rows"] == 1
    assert summary.loc["avg_enjoyment", "coverage_pct"] == 1.0
    assert summary.loc["avg_usefulness", "non_null_rows"] == 0
