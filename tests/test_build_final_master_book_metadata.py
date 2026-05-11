from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_books_tracking.build_final_master_book_metadata import build_final_master


def test_build_final_master_merges_personal_ratings_from_historical_and_holdout(
    tmp_path: Path,
) -> None:
    master_csv = tmp_path / "master.csv"
    historical_csv = tmp_path / "historical.csv"
    holdout_csv = tmp_path / "holdout.csv"

    pd.DataFrame(
        [
            {
                "title": "Good Old Neon",
                "author(old and wrong)": "by",
                "filename": "GoodOldNeon(1).pdf",
                "source": "Play Export",
                "corrected_author": "David Foster Wallace",
                "ratings good reads gsheets": 4.53,
                "good reads combined": "4.53|171|1059",
                "goodread ratings": 4.53,
                "goodreads number reviews": 171,
                "goodreads number ratings": 1059,
            },
            {
                "title": "Shape Up Stop Running in Circles and Ship Work that Matters",
                "author(old and wrong)": "Ryan Singer",
                "filename": "",
                "source": "Holdout 2026",
                "corrected_author": "Ryan Singer",
                "ratings good reads gsheets": 4.26,
                "good reads combined": "4.26|317|2908",
                "goodread ratings": 4.26,
                "goodreads number reviews": 317,
                "goodreads number ratings": 2908,
            },
        ]
    ).to_csv(master_csv, index=False)

    pd.DataFrame(
        [
            {
                "title": "GoodOldNeon.pdf",
                "filename_ratings2": "GoodOldNeon.pdf",
                "Bookshelf": "Literature",
                "latest_modified": "2021-02-13",
                "Enjoyment (/5)": 4.5,
                "Usefulness /5 to Me": 2.0,
                "Enjoyment (/5)_ratings2": 4.0,
                "Usefulness /5 to Me_ratings2": 1.5,
                "goodreads_cleaned_status": "matched",
            }
        ]
    ).to_csv(historical_csv, index=False)

    pd.DataFrame(
        [
            {
                "title": "shaping up",
                "date_finished": "2025-12-05",
                "Bookshelf": "Computer Science",
                "Enjoyment (/5)": 4.5,
                "Usefulness /5 to Me": 3.5,
                "Enjoyment (/5)_ratings2": 4.0,
                "Usefulness /5 to Me_ratings2": 3.0,
                "goodreads_cleaned_status": "matched",
            }
        ]
    ).to_csv(holdout_csv, index=False)

    final_master = build_final_master(
        master_csv=master_csv,
        historical_csv=historical_csv,
        holdout_csv=holdout_csv,
    )

    historical_row = final_master[final_master["source"] == "Play Export"].iloc[0]
    holdout_row = final_master[final_master["source"] == "Holdout 2026"].iloc[0]

    assert historical_row["personal_ratings_match_found"] is True
    assert historical_row["personal_ratings_match_method"] == "filename_key"
    assert historical_row["avg_enjoyment"] == pytest.approx(4.25)
    assert historical_row["finished_date"] == "2021-02-13"

    assert holdout_row["personal_ratings_match_found"] is True
    assert holdout_row["personal_ratings_match_method"] == "manual_alias"
    assert holdout_row["bookshelf"] == "Computer Science"
    assert holdout_row["avg_usefulness"] == pytest.approx(3.25)
    assert "gr=4.26" in holdout_row["all_ratings_summary"]
    assert historical_row["goodread ratings"] == pytest.approx(4.53)
