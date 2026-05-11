from __future__ import annotations

import json

import pandas as pd

from ai_books_tracking.merge_goodreads_into_enriched import merge_into_copy


def test_merge_into_copy_joins_goodreads_on_title_only(tmp_path) -> None:
    enriched = pd.DataFrame(
        [
            {
                "title": "Ted Chiang - Exhalation-Knopf (2019).epub",
                "author": "by",
                "Bookshelf": "fiction",
                "Enjoyment (/5)": 4.0,
                "Usefulness /5 to Me": 1.0,
            },
            {
                "title": "Author Only",
                "author": "by",
                "Bookshelf": "other",
                "Enjoyment (/5)": 2.0,
                "Usefulness /5 to Me": 2.0,
            },
            {
                "title": "Rescue Me",
                "author": "Author Name",
                "Bookshelf": "other",
                "Enjoyment (/5)": 3.0,
                "Usefulness /5 to Me": 3.0,
            },
            {
                "title": "Zero Rating",
                "author": "by",
                "Bookshelf": "other",
                "Enjoyment (/5)": 3.0,
                "Usefulness /5 to Me": 3.0,
            },
        ]
    )
    goodreads = pd.DataFrame(
        [
            {
                "title": "Ted Chiang - Exhalation-Knopf (2019).epub",
                "author": "Ted Chiang",
                "goodreads_status": "matched",
                "goodreads_match_score": 0.95,
                "goodreads_rating": 4.27,
                "goodreads_rating_count": 113201,
                "goodreads_url": "https://www.goodreads.com/book/show/41160292-exhalation",
            },
            {
                "title": "Author Only",
                "author": "Unknown",
                "goodreads_status": "review",
                "goodreads_match_score": 0.73,
                "goodreads_rating": 3.14,
                "goodreads_rating_count": 42,
                "goodreads_url": "https://www.goodreads.com/book/show/42-bad-match",
            },
            {
                "title": "Rescue Me",
                "author": "Author Name",
                "goodreads_status": "matched",
                "goodreads_match_score": 0.88,
                "goodreads_rating": 0.0,
                "goodreads_rating_count": 0,
                "goodreads_url": "https://www.goodreads.com/book/show/0-wrapper",
                "goodreads_title": "Rescue Me (Proof)",
                "goodreads_author": "Author Name",
                "goodreads_candidates_json": json.dumps(
                    [
                        {
                            "url": "https://www.goodreads.com/book/show/0-wrapper",
                            "page_title": "Rescue Me (Proof)",
                            "page_author": "Author Name",
                            "rating_value": 0,
                            "rating_count": 0,
                            "title_similarity": 0.95,
                            "author_similarity": 1.0,
                            "score": 0.88,
                        },
                        {
                            "url": "https://www.goodreads.com/book/show/123-rescue-me",
                            "page_title": "Rescue Me",
                            "page_author": "Author Name",
                            "rating_value": 4.2,
                            "rating_count": 2500,
                            "title_similarity": 0.8,
                            "author_similarity": 1.0,
                            "score": 0.72,
                        },
                    ]
                ),
            },
            {
                "title": "Zero Rating",
                "author": "Unknown",
                "goodreads_status": "matched",
                "goodreads_match_score": 0.99,
                "goodreads_rating": 0.0,
                "goodreads_rating_count": 0,
                "goodreads_url": "https://www.goodreads.com/book/show/0-zero-rating",
            },
        ]
    )
    ratings2 = pd.DataFrame(
        [
            {
                "title": "Ted Chiang - Exhalation-Knopf (2019).epub",
                "author": "Ted Chiang",
                "Enjoyment (/5)": 4.5,
                "Usefulness /5 to Me": 1.5,
            },
            {
                "title": "Author Only",
                "author": "Unknown",
                "Enjoyment (/5)": 2.5,
                "Usefulness /5 to Me": 2.5,
            },
            {
                "title": "Rescue Me",
                "author": "Author Name",
                "Enjoyment (/5)": 3.5,
                "Usefulness /5 to Me": 3.5,
            },
            {
                "title": "Zero Rating",
                "author": "Unknown",
                "Enjoyment (/5)": 3.5,
                "Usefulness /5 to Me": 3.5,
            },
        ]
    )

    enriched_path = tmp_path / "books_enriched.csv"
    goodreads_path = tmp_path / "books_goodreads.csv"
    ratings2_path = tmp_path / "ratings2.csv"
    output_path = tmp_path / "output.csv"
    enriched.to_csv(enriched_path, index=False)
    goodreads.to_csv(goodreads_path, index=False)
    ratings2.to_csv(ratings2_path, index=False)

    merged = merge_into_copy(
        enriched_csv=enriched_path,
        goodreads_csv=goodreads_path,
        ratings2_csv=ratings2_path,
        output_csv=output_path,
    )

    assert len(merged) == 4

    matched = merged.loc[
        merged["title"] == "Ted Chiang - Exhalation-Knopf (2019).epub"
    ].iloc[0]
    assert matched["author"] == "by"
    assert matched["goodreads_rating"] == 4.27
    assert matched["goodreads_rating_raw"] == 4.27
    assert matched["goodreads_rating_raw_best"] == 4.27
    assert matched["goodreads_raw_best_source"] == "chosen_candidate"
    assert matched["Enjoyment (/5)_ratings2"] == 4.5

    review = merged.loc[merged["title"] == "Author Only"].iloc[0]
    assert pd.isna(review["goodreads_rating"])
    assert review["goodreads_rating_raw"] == 3.14
    assert review["goodreads_rating_raw_best"] == 3.14
    assert review["Enjoyment (/5)_ratings2"] == 2.5

    rescued = merged.loc[merged["title"] == "Rescue Me"].iloc[0]
    assert pd.isna(rescued["goodreads_rating"])
    assert pd.isna(rescued["goodreads_rating_raw"])
    assert rescued["goodreads_rating_raw_best"] == 4.2
    assert rescued["goodreads_rating_count_raw_best"] == 2500
    assert rescued["goodreads_title_raw_best"] == "Rescue Me"
    assert rescued["goodreads_raw_best_source"] == "rescued_candidate"
    assert rescued["Enjoyment (/5)_ratings2"] == 3.5

    zero_rating = merged.loc[merged["title"] == "Zero Rating"].iloc[0]
    assert pd.isna(zero_rating["goodreads_rating"])
    assert pd.isna(zero_rating["goodreads_rating_raw"])
    assert pd.isna(zero_rating["goodreads_rating_count"])
    assert pd.isna(zero_rating["goodreads_rating_count_raw"])
    assert pd.isna(zero_rating["goodreads_rating_raw_best"])
    assert zero_rating["goodreads_raw_best_source"] == "missing"
    assert zero_rating["Enjoyment (/5)_ratings2"] == 3.5
