from __future__ import annotations

import pandas as pd

from ai_books_tracking.goodreads_ratings import (
    build_title_query_variants,
    build_canonical_books,
    clean_title_for_search,
    extract_goodreads_page_data,
    score_candidate,
)


def test_clean_title_for_search_strips_noise_and_leading_author() -> None:
    cleaned = clean_title_for_search(
        "Ted Chiang - Exhalation-Knopf (2019).epub",
        author="Ted Chiang",
    )
    assert cleaned == "Exhalation"


def test_build_canonical_books_prefers_cleaner_second_csv_author(tmp_path) -> None:
    play = pd.DataFrame(
        [
            {
                "title": "Ted Chiang - Exhalation-Knopf (2019).epub",
                "author": "by",
                "Bookshelf": "fiction",
                "filename": "Ted Chiang - Exhalation-Knopf (2019).epub",
                "Enjoyment (/5)": 4.0,
                "Usefulness /5 to Me": 1.0,
            }
        ]
    )
    rerate = pd.DataFrame(
        [
            {
                "title": "Ted Chiang - Exhalation-Knopf (2019).epub",
                "author": "Ted Chiang",
                "Bookshelf": "fiction",
                "filename": "Exhalation.html",
                "Enjoyment (/5)": 4.5,
                "Usefulness /5 to Me": 1.5,
            }
        ]
    )
    play_path = tmp_path / "play.csv"
    rerate_path = tmp_path / "rerate.csv"
    play.to_csv(play_path, index=False)
    rerate.to_csv(rerate_path, index=False)

    books = build_canonical_books(play_path, rerate_path)

    assert len(books) == 1
    row = books.iloc[0]
    assert row["author"] == "Ted Chiang"
    assert row["search_title"] == "Exhalation"
    assert row["enjoyment_play"] == 4.0
    assert row["enjoyment_ratings2"] == 4.5


def test_build_title_query_variants_adds_shorter_hyphen_fallback() -> None:
    row = pd.Series(
        {
            "search_title": "A Deepness in the Sky-Vernor Vinge",
            "title": "A Deepness in the Sky-Vernor Vinge - Tor Books (2000).epub",
            "author": "",
            "filename_ratings2": "",
            "filename_play": "A Deepness in the Sky-Vernor Vinge - Tor Books (2000).epub",
        }
    )

    variants = build_title_query_variants(row)

    assert "A Deepness in the Sky-Vernor Vinge" in variants
    assert "A Deepness in the Sky" in variants


def test_extract_goodreads_page_data_reads_json_ld_rating() -> None:
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Book",
          "name": "Exhalation",
          "author": {"@type": "Person", "name": "Ted Chiang"},
          "aggregateRating": {
            "@type": "AggregateRating",
            "ratingValue": "4.21",
            "ratingCount": "65234"
          }
        }
        </script>
      </head>
    </html>
    """

    parsed = extract_goodreads_page_data(
        html, "https://www.goodreads.com/book/show/41160292-exhalation"
    )

    assert parsed["page_title"] == "Exhalation"
    assert parsed["page_author"] == "Ted Chiang"
    assert parsed["rating_value"] == 4.21
    assert parsed["rating_count"] == 65234


def test_score_candidate_prefers_right_title_and_author() -> None:
    book = pd.Series(
        {
            "search_title": "Exhalation",
            "search_author": "Ted Chiang",
        }
    )
    item = {
        "title": "Exhalation by Ted Chiang | Goodreads",
        "snippet": "Stories by Ted Chiang.",
        "link": "https://www.goodreads.com/book/show/41160292-exhalation",
    }
    page_data = {
        "page_title": "Exhalation",
        "page_author": "Ted Chiang",
        "rating_value": 4.21,
        "rating_count": 65234,
    }

    candidate = score_candidate(
        book, 'site:goodreads.com/book/show "Exhalation" "Ted Chiang"', item, page_data
    )

    assert candidate.title_similarity > 0.95
    assert candidate.author_similarity > 0.95
    assert candidate.score > 0.9
