from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ai_books_tracking.book_word_counts import (
    LocalBookFile,
    build_word_count_outputs,
    classify_body_section,
    propagate_epub_back_matter_categories,
    repair_all_excluded_epub_body,
    score_file_match,
)
from ai_books_tracking.book_speed_analysis import build_speed_analysis
from ai_books_tracking.book_wpm_calendar_notes import normalize_title


def make_local_book_file(path: Path) -> LocalBookFile:
    return LocalBookFile(
        path=path,
        name_norm=normalize_title(path.name),
        stem_norm=normalize_title(path.stem),
    )


def test_score_file_match_rejects_short_title_substring_false_positives() -> None:
    dune_messiah = make_local_book_file(
        Path("(Dune Chronicles volume 2) Frank Herbert - Dune Messiah.epub")
    )
    open_source = make_local_book_file(
        Path("The Architecture of Open Source Applications 2.pdf")
    )
    world_crisis_volume_3 = make_local_book_file(
        Path("The World Crisis - Churchill Winston - Volume 3_ 1916-1918.epub")
    )

    assert score_file_match("Dune", "", dune_messiah) < 70
    assert score_file_match("Open", "", open_source) < 70
    assert score_file_match("the world crisis volume 2", "", world_crisis_volume_3) < 70

    assert score_file_match("Dune Messiah", "", dune_messiah) >= 90


def test_score_file_match_accepts_real_initialism() -> None:
    rlhf = make_local_book_file(
        Path(
            "Reinforcement Learning from Human Feedback_ "
            "A short introduction to RLHF and post-training focused on language models.pdf"
        )
    )

    assert score_file_match("rlhf", "", rlhf) >= 70


def test_build_word_count_outputs_prefers_local_text_over_online_count(
    tmp_path: Path,
) -> None:
    book_root = tmp_path / "books"
    output_dir = tmp_path / "outputs"
    book_root.mkdir()
    (book_root / "Example Book.txt").write_text("one two three four", encoding="utf-8")
    sources = tmp_path / "sources.csv"
    pd.DataFrame(
        [
            {
                "title": "Example Book",
                "source_type": "web_count",
                "word_count": 10,
                "url": "https://example.com/example-book-word-count",
                "confidence": "medium",
                "notes": "test online estimate",
            }
        ]
    ).to_csv(sources, index=False)
    titles = pd.DataFrame(
        [
            {
                "finish_id": "finish_0001",
                "title": "Example Book",
                "finish_date": "2026-01-01",
            }
        ]
    )

    result = build_word_count_outputs(
        titles=titles,
        output_dir=output_dir,
        book_roots=[book_root],
        word_source_csv=sources,
    )

    row = result.iloc[0]
    assert row["local_file_word_count"] == 4
    assert row["external_word_count"] == 10
    assert row["chosen_word_count"] == 4
    assert row["word_count_source"] == "local_file_word_count"
    assert row["online_error_rate_vs_local"] == 1.5
    assert (output_dir / "book_word_count_online_error_rates.png").exists()


def test_build_word_count_outputs_uses_epub_body_count_and_audits_exclusions(
    tmp_path: Path,
) -> None:
    book_root = tmp_path / "books"
    output_dir = tmp_path / "outputs"
    book_root.mkdir()
    epub = book_root / "Example Book.epub"
    with ZipFile(epub, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf"
                  media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>""",
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="2.0">
              <manifest>
                <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
                <item id="index" href="index.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="chapter"/>
                <itemref idref="index"/>
              </spine>
            </package>""",
        )
        archive.writestr(
            "OEBPS/chapter.xhtml",
            "<html><body><h1>Chapter 1</h1><p>one two three four</p></body></html>",
        )
        archive.writestr(
            "OEBPS/index.xhtml",
            "<html><body><h1>Index</h1><p>alpha beta gamma delta</p></body></html>",
        )
    titles = pd.DataFrame(
        [
            {
                "finish_id": "finish_0001",
                "title": "Example Book",
                "finish_date": "2026-01-01",
            }
        ]
    )

    result = build_word_count_outputs(
        titles=titles,
        output_dir=output_dir,
        book_roots=[book_root],
        word_source_csv=None,
    )

    row = result.iloc[0]
    assert row["local_file_word_count"] == 6
    assert row["local_raw_word_count"] == 11
    assert row["local_excluded_word_count"] == 5
    section_audit = pd.read_csv(output_dir / "book_word_count_local_section_audit.csv")
    assert set(section_audit["body_section_category"]) == {"body", "index"}


def test_body_section_classifier_does_not_exclude_body_prose_false_positives() -> None:
    assert (
        classify_body_section(
            "OEBPS/ch20.xhtml",
            "XX",
            "XX My dear Wormwood, I note with great displeasure that the Enemy...",
            "The Screwtape Letters",
        )
        == "body"
    )
    assert (
        classify_body_section(
            "Red_Plenty_split_036.html",
            "Red Plenty | Ladies, Cover Your Ears! 1965",
            "Red Plenty Ladies, Cover Your Ears! Emil splashed his head with water.",
            "Red Plenty",
        )
        == "body"
    )
    assert (
        classify_body_section("notes.xhtml", "Notes", "Notes Chapter 1", "Book")
        == "notes"
    )
    assert (
        classify_body_section(
            "OEBPS/Text/footnotes.xhtml",
            "",
            "1 This is note text that should not count as main reading prose.",
            "Book",
        )
        == "notes"
    )
    assert (
        classify_body_section(
            "OEBPS/html/08_NOTES_ON_MYSELF.xhtml",
            "IMPRO Improvisation and the Theatre",
            "IMPRO Improvisation and the Theatre Notes on Myself As I grew up...",
            "IMPRO Improvisation and the Theatre",
        )
        == "body"
    )
    assert (
        classify_body_section(
            "EPUB/xhtml/note_author.xhtml",
            "A Note from the Authors",
            "A Note from the Authors This is part of the book's introduction.",
            "Downfall",
        )
        == "body"
    )
    assert (
        classify_body_section(
            "index_split_131.html",
            "Churchill: Walking with Destiny",
            "Churchill: Walking with Destiny Notes ABBREVIATIONS General AP Avon Papers",
            "Churchill: Walking with Destiny",
        )
        == "notes"
    )


def test_epub_back_matter_category_propagates_to_unmarked_continuation() -> None:
    rows: list[dict[str, object]] = [
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 10,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 9,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 8,
        },
        {
            "body_section_category": "notes",
            "included_in_reading_count": False,
            "word_count": 5,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 7,
        },
        {
            "body_section_category": "index",
            "included_in_reading_count": False,
            "word_count": 3,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 11,
        },
    ]

    propagate_epub_back_matter_categories(rows)

    assert rows[4]["body_section_category"] == "notes"
    assert rows[4]["included_in_reading_count"] is False
    assert rows[6]["body_section_category"] == "front_back_matter"
    assert rows[6]["included_in_reading_count"] is False


def test_epub_back_matter_category_does_not_propagate_from_early_note() -> None:
    rows: list[dict[str, object]] = [
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 10,
        },
        {
            "body_section_category": "notes",
            "included_in_reading_count": False,
            "word_count": 5,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 7,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 11,
        },
    ]

    propagate_epub_back_matter_categories(rows)

    assert rows[2]["body_section_category"] == "body"
    assert rows[2]["included_in_reading_count"] is True


def test_epub_back_matter_category_excludes_body_after_late_index() -> None:
    rows: list[dict[str, object]] = [
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 10,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 11,
        },
        {
            "body_section_category": "index",
            "included_in_reading_count": False,
            "word_count": 5,
        },
        {
            "body_section_category": "body",
            "included_in_reading_count": True,
            "word_count": 7,
        },
    ]

    propagate_epub_back_matter_categories(rows)

    assert rows[3]["body_section_category"] == "front_back_matter"
    assert rows[3]["included_in_reading_count"] is False


def test_all_excluded_epub_body_repair_reclassifies_malformed_index_sections() -> None:
    rows: list[dict[str, object]] = [
        {
            "body_section_category": "front_back_matter",
            "included_in_reading_count": False,
            "word_count": 1,
        },
        {
            "body_section_category": "index",
            "included_in_reading_count": False,
            "word_count": 100,
        },
    ]

    warning = repair_all_excluded_epub_body(rows)

    assert warning == "all_non_front_sections_reclassified_as_body"
    assert rows[1]["body_section_category"] == "body"
    assert rows[1]["included_in_reading_count"] is True


def test_speed_analysis_uses_only_local_file_word_counts_for_wpm(
    tmp_path: Path,
) -> None:
    time_csv = tmp_path / "time.csv"
    words_csv = tmp_path / "words.csv"
    output_dir = tmp_path / "outputs"
    pd.DataFrame(
        [
            {
                "finish_id": "local",
                "title": "Local Book",
                "matched_finish_date": "2026-01-01",
                "primary_first_pass_minutes": 100,
                "first_pass_wall_clock_minutes": 100,
                "reading_calendar_minutes": 100,
                "reading_wall_clock_minutes": 100,
                "audiobook_overlap_policy_minutes": 0,
                "audiobook_wall_clock_minutes": 0,
                "finished_calendar_minutes": 0,
                "finished_wall_clock_minutes": 0,
                "n_included_events": 2,
                "is_reread": False,
            },
            {
                "finish_id": "metadata",
                "title": "Metadata Book",
                "matched_finish_date": "2026-01-02",
                "primary_first_pass_minutes": 100,
                "first_pass_wall_clock_minutes": 100,
                "reading_calendar_minutes": 100,
                "reading_wall_clock_minutes": 100,
                "audiobook_overlap_policy_minutes": 0,
                "audiobook_wall_clock_minutes": 0,
                "finished_calendar_minutes": 0,
                "finished_wall_clock_minutes": 0,
                "n_included_events": 2,
                "is_reread": False,
            },
        ]
    ).to_csv(time_csv, index=False)
    pd.DataFrame(
        [
            {
                "finish_id": "local",
                "chosen_word_count": 50_000,
                "word_count_source": "local_file_word_count",
                "word_count_confidence": "high",
                "local_file_word_count": 50_000,
                "metadata_page_count": None,
                "metadata_word_estimate": None,
            },
            {
                "finish_id": "metadata",
                "chosen_word_count": 60_000,
                "word_count_source": "metadata_pages_x_words_per_page",
                "word_count_confidence": "medium",
                "local_file_word_count": None,
                "metadata_page_count": 240,
                "metadata_word_estimate": 60_000,
            },
        ]
    ).to_csv(words_csv, index=False)

    result = build_speed_analysis(
        calendar_time_csv=time_csv,
        word_counts_csv=words_csv,
        output_dir=output_dir,
    )

    assert result.set_index("finish_id").loc["local", "usable_for_speed"]
    assert not result.set_index("finish_id").loc["metadata", "usable_for_speed"]
    aggregate = pd.read_csv(output_dir / "book_speed_aggregate_wpm.csv")
    assert (
        aggregate.set_index("cohort").loc[
            "first_reads_with_local_file_word_count", "n_finished_read_instances"
        ]
        == 1
    )
