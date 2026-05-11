from __future__ import annotations

from pathlib import Path

from structured_summaries.metadata_selection import (
    build_catalog_metadata_links,
    select_high_signal_read_links,
    select_unread_local_links,
)
from structured_summaries.models import BookRecord


def _book(
    *,
    tmp_path: Path,
    book_id: str,
    title: str,
    filename: str,
    author: str = "",
) -> BookRecord:
    book_dir = tmp_path / book_id
    book_dir.mkdir()
    primary_path = book_dir / filename
    primary_path.write_text("sample", encoding="utf-8")
    return BookRecord(
        book_id=book_id,
        title=title,
        author=author,
        book_dir=book_dir,
        primary_path=primary_path,
        primary_format="pdf",
        companion_html_path=None,
        all_paths=(primary_path,),
        search_title=title,
    )


def test_build_catalog_metadata_links_prefers_filename_match(tmp_path: Path) -> None:
    matching_book = _book(
        tmp_path=tmp_path,
        book_id="pragmatic",
        title="The Pragmatic Programmer",
        filename="The Pragmatic Programmer - Andrew Hunt.pdf",
        author="Andrew Hunt",
    )
    other_book = _book(
        tmp_path=tmp_path,
        book_id="other",
        title="Another Book",
        filename="Another Book.pdf",
    )
    metadata_rows = [
        {
            "title": "Wrong title variant",
            "personal_rating_title": "",
            "filename": "The Pragmatic Programmer - Andrew Hunt.pdf",
            "corrected_author": "Andrew Hunt",
            "author(old and wrong)": "",
            "bookshelf": "Computer Science",
            "finished_date": "2024-01-01",
            "avg_enjoyment": "4.5",
            "avg_usefulness": "4.75",
        }
    ]

    links = build_catalog_metadata_links([matching_book, other_book], metadata_rows)
    by_id = {link.book.book_id: link for link in links}

    assert by_id["pragmatic"].is_matched
    assert by_id["pragmatic"].match_method == "filename_key"
    assert not by_id["other"].is_matched


def test_selection_splits_read_candidates_from_unread_queue(tmp_path: Path) -> None:
    read_book = _book(
        tmp_path=tmp_path,
        book_id="high-signal",
        title="High Signal Book",
        filename="High Signal Book.pdf",
        author="Author One",
    )
    unread_book = _book(
        tmp_path=tmp_path,
        book_id="unread",
        title="Unread Book",
        filename="Unread Book.pdf",
        author="Author Two",
    )
    metadata_rows = [
        {
            "title": "High Signal Book",
            "personal_rating_title": "",
            "filename": "High Signal Book.pdf",
            "corrected_author": "Author One",
            "author(old and wrong)": "",
            "bookshelf": "Business, management",
            "finished_date": "2024-01-01",
            "avg_enjoyment": "4.0",
            "avg_usefulness": "4.75",
        },
        {
            "title": "Unread Book",
            "personal_rating_title": "",
            "filename": "Unread Book.pdf",
            "corrected_author": "Author Two",
            "author(old and wrong)": "",
            "bookshelf": "Business, management",
            "finished_date": "",
            "avg_enjoyment": "",
            "avg_usefulness": "",
        },
    ]

    links = build_catalog_metadata_links([read_book, unread_book], metadata_rows)
    read_links = select_high_signal_read_links(links, min_signal_score=4.0)
    unread_links = select_unread_local_links(links)

    assert [link.book.book_id for link in read_links] == ["high-signal"]
    assert [link.book.book_id for link in unread_links] == ["unread"]


def test_build_catalog_metadata_links_fuzzy_matches_hyphenated_title(
    tmp_path: Path,
) -> None:
    book = _book(
        tmp_path=tmp_path,
        book_id="jackal",
        title="The-Day-of-the-Jackal",
        filename="The-Day-of-the-Jackal.pdf",
    )
    metadata_rows = [
        {
            "title": "day of the jackal",
            "personal_rating_title": "",
            "filename": "",
            "corrected_author": "Frederick Forsyth",
            "author(old and wrong)": "",
            "bookshelf": "fiction",
            "finished_date": "2025-09-23",
            "avg_enjoyment": "2.5",
            "avg_usefulness": "1.0",
        }
    ]

    links = build_catalog_metadata_links([book], metadata_rows)

    assert links[0].is_matched
    assert links[0].is_read
    assert links[0].match_method in {"title_key", "fuzzy_title", "fuzzy_title_author"}


def test_build_catalog_metadata_links_fuzzy_matches_prefix_title(
    tmp_path: Path,
) -> None:
    book = _book(
        tmp_path=tmp_path,
        book_id="kelly",
        title="Kelly",
        filename="Kelly.epub",
    )
    metadata_rows = [
        {
            "title": "Kelly: More Than My Share of It All",
            "personal_rating_title": "Kelly my share of it all",
            "filename": "",
            "corrected_author": "Clarence L. Johnson",
            "author(old and wrong)": "",
            "bookshelf": "General Reading",
            "finished_date": "2025-01-28",
            "avg_enjoyment": "3.75",
            "avg_usefulness": "1.5",
        },
        {
            "title": "Robinson Crusoe",
            "personal_rating_title": "Robinson Crusoe James Kelly edition",
            "filename": "",
            "corrected_author": "Daniel Defoe",
            "author(old and wrong)": "",
            "bookshelf": "Literature",
            "finished_date": "2020-03-11",
            "avg_enjoyment": "3.25",
            "avg_usefulness": "1.0",
        },
    ]

    links = build_catalog_metadata_links([book], metadata_rows)

    assert links[0].is_matched
    assert links[0].metadata_row is not None
    assert links[0].metadata_row["corrected_author"] == "Clarence L. Johnson"


def test_build_catalog_metadata_links_applies_manual_read_override(
    tmp_path: Path,
) -> None:
    book = _book(
        tmp_path=tmp_path,
        book_id="trading",
        title="Trading at the Speed of Light",
        filename="Trading at the Speed of Light.epub",
    )

    links = build_catalog_metadata_links(
        [book],
        [],
        read_overrides={"trading": "user_confirmed_read"},
    )
    unread_links = select_unread_local_links(links)

    assert links[0].is_matched
    assert links[0].is_read
    assert links[0].match_method == "manual_override"
    assert links[0].override_reason == "user_confirmed_read"
    assert unread_links == []


def test_build_catalog_metadata_links_prefers_manual_override_over_fuzzy_match(
    tmp_path: Path,
) -> None:
    book = _book(
        tmp_path=tmp_path,
        book_id="churchill",
        title="A History of the English-Speaking Peoples Collecti",
        filename="A History of the English-Speaking Peoples Collecti.epub",
    )
    metadata_rows = [
        {
            "title": "the history of the English speaking peoples volume 4",
            "personal_rating_title": "",
            "filename": "",
            "corrected_author": "Winston S. Churchill",
            "author(old and wrong)": "",
            "bookshelf": "Histories",
            "finished_date": "2026-01-11",
            "avg_enjoyment": "3.0",
            "avg_usefulness": "2.0",
        }
    ]

    links = build_catalog_metadata_links(
        [book],
        metadata_rows,
        read_overrides={"churchill": "series_alias_override"},
    )

    assert links[0].is_matched
    assert links[0].match_method == "manual_override"
    assert links[0].override_reason == "series_alias_override"


def test_build_catalog_metadata_links_matches_truncated_prefix_title(
    tmp_path: Path,
) -> None:
    book = _book(
        tmp_path=tmp_path,
        book_id="selfish",
        title="Selfish Reasons to Have More Kids Why Being a Gre",
        filename="Selfish Reasons to Have More Kids Why Being a Gre.epub",
    )
    metadata_rows = [
        {
            "title": (
                "Selfish Reasons to Have More Kids: Why Being a Great Parent "
                "Is Less Work and More Fun Than You Think"
            ),
            "personal_rating_title": "",
            "filename": "",
            "corrected_author": "Bryan Caplan",
            "author(old and wrong)": "",
            "bookshelf": "General Reading",
            "finished_date": "2025-07-12",
            "avg_enjoyment": "2.5",
            "avg_usefulness": "2.0",
        }
    ]

    links = build_catalog_metadata_links([book], metadata_rows)

    assert links[0].is_matched
    assert links[0].match_method in {"fuzzy_title", "fuzzy_title_author"}
