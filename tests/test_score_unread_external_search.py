from __future__ import annotations

import pandas as pd

from ai_books_tracking.score_unread_external_search import (
    GOODREADS_RF,
    load_unread_catalog,
    prepare_goodreads_candidates,
    prepare_multisource_candidates,
    build_recommendation_lines,
)


def test_load_unread_catalog_prefers_cleaned_metadata_and_marks_started(
    tmp_path,
) -> None:
    unread = pd.DataFrame(
        [
            {
                "title": "Raw Title.pdf",
                "author": "",
                "cleaned_title": "Clean Title",
                "cleaned_author": "Clean Author",
                "Bookshelf": "Literature",
                "play_status": "unfinished",
                "earliest_modified": "",
                "latest_modified": "",
                "filename": "raw_title.pdf",
            },
            {
                "title": "Finished Book",
                "author": "Original Author",
                "cleaned_title": "",
                "cleaned_author": "",
                "Bookshelf": "General Reading",
                "play_status": "finished",
                "earliest_modified": "",
                "latest_modified": "",
                "filename": "finished_book.epub",
            },
            {
                "title": "Started Book",
                "author": "Original Author",
                "cleaned_title": "",
                "cleaned_author": "",
                "Bookshelf": "Business, management",
                "play_status": "unfinished",
                "earliest_modified": "2026-03-01 10:00:00",
                "latest_modified": "2026-03-02 10:00:00",
                "filename": "started_book.epub",
            },
        ]
    )
    csv_path = tmp_path / "unread.csv"
    unread.to_csv(csv_path, index=False)

    loaded = load_unread_catalog(csv_path)

    assert loaded["display_title"].tolist() == [
        "Clean Title",
        "Finished Book",
        "Started Book",
    ]
    assert loaded["display_author"].tolist() == [
        "Clean Author",
        "Original Author",
        "Original Author",
    ]
    assert loaded["started"].tolist() == [False, True, True]
    assert loaded["started_status"].tolist() == [
        "not_started",
        "started",
        "started",
    ]


def test_prepare_candidate_frames_use_external_rating_columns() -> None:
    unread = pd.DataFrame(
        {
            "display_title": ["Book A", "Book B"],
            "display_author": ["Author A", "Author B"],
            "Bookshelf": ["Literature", "Computer Science"],
            "earliest_modified": [
                pd.Timestamp("2026-03-01 10:00:00"),
                pd.NaT,
            ],
            "latest_modified": [
                pd.Timestamp("2026-03-02 10:00:00"),
                pd.NaT,
            ],
            "filename": ["book_a.pdf", "book_b.pdf"],
            "goodread ratings": [4.2, 3.8],
            "goodreads number reviews": [120.0, 9.0],
            "goodreads number ratings": [3000.0, 150.0],
            "open library combined": [
                "4.1|18|https://openlibrary.org/books/OL1M",
                "N/A|N/A|N/A",
            ],
            "open library ratings": [pd.NA, 3.9],
            "open library number reviews": [pd.NA, 11.0],
            "Amazon combined with links": [
                "4.6|88|https://amazon.com/a",
                "4.2|12|https://amazon.com/b",
            ],
            "Unnamed: 24": [pd.NA, 4.4],
            "Unnamed: 25": [pd.NA, 40.0],
        }
    )

    goodreads_candidates = prepare_goodreads_candidates(unread)
    multisource_candidates = prepare_multisource_candidates(unread)

    assert goodreads_candidates["title"].tolist() == ["Book A", "Book B"]
    assert goodreads_candidates["goodreads_rating"].tolist() == [4.2, 3.8]
    assert goodreads_candidates["year_finished"].iloc[0] == 2026
    assert pd.isna(goodreads_candidates["year_finished"].iloc[1])

    assert multisource_candidates["goodreads_rating_verified"].tolist() == [4.2, 3.8]
    assert multisource_candidates["ol_rating_consensus"].tolist() == [4.1, 3.9]
    assert multisource_candidates["ol_reviews_consensus"].tolist() == [18.0, 11.0]
    assert multisource_candidates["amazon_rating_consensus"].tolist() == [4.6, 4.4]
    assert multisource_candidates["amazon_reviews_consensus"].tolist() == [88.0, 40.0]


def test_build_recommendation_lines_splits_keep_and_discard_by_started_status() -> None:
    scored = pd.DataFrame(
        {
            "display_title": ["Started Keep", "Started Drop", "Fresh Keep"],
            "display_author": ["A", "B", "C"],
            "Bookshelf": ["Literature", "Literature", "General Reading"],
            "started_status": ["started", "started", "not_started"],
            GOODREADS_RF.pred_enjoyment_col: [4.2, 3.1, 4.0],
            GOODREADS_RF.pred_usefulness_col: [2.7, 1.8, 2.4],
            GOODREADS_RF.keep_col: [True, False, True],
        }
    )

    lines = build_recommendation_lines(scored, GOODREADS_RF)
    text = "\n".join(lines)

    assert "Started: 2 books" in text
    assert "Not Started: 1 books" in text
    assert "Started Keep | A | Literature | enjoy=4.20 | useful=2.70" in text
    assert "Started Drop | B | Literature | enjoy=3.10 | useful=1.80" in text
    assert "Fresh Keep | C | General Reading | enjoy=4.00 | useful=2.40" in text
