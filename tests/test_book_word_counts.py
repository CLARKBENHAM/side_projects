from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from ai_books_tracking.book_word_counts import (
    LocalBookFile,
    LocalTextAudit,
    add_full_finished_projection_columns,
    build_projection_metric_summary,
    build_word_count_outputs,
    classify_body_section,
    first_clean_value,
    iter_book_files,
    local_file_hints_for_title,
    propagate_epub_back_matter_categories,
    repair_all_excluded_epub_body,
    score_file_match,
)
from ai_books_tracking import book_word_counts as book_word_counts_module
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


def test_score_file_match_rejects_full_title_initial_collision() -> None:
    discovery = make_local_book_file(Path("The Discovery of France.epub"))

    assert score_file_match("the Dark Forrest", "", discovery) < 70
    assert score_file_match("tdf", "", discovery) >= 70


def test_score_file_match_accepts_real_initialism() -> None:
    rlhf = make_local_book_file(
        Path(
            "Reinforcement Learning from Human Feedback_ "
            "A short introduction to RLHF and post-training focused on language models.pdf"
        )
    )

    assert score_file_match("rlhf", "", rlhf) >= 70


def test_local_file_hints_use_finish_date_for_short_calendar_refs() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "Open_ An Autobiography-Andre Agassi -Vintage (2010).pdf",
                "hint_filename": "Open_ An Autobiography-Andre Agassi -Vintage (2.pdf",
                "hint_finish_date": "2021-10-23",
            },
            {
                "hint_title": "The Architecture of Open Source Applications",
                "hint_filename": "The Architecture of Open Source Applications.pdf",
                "hint_finish_date": "2024-01-01",
            },
        ]
    )

    result = local_file_hints_for_title(hints, "Open", "Open", finish_date="2021-10-23")

    assert result[0] == "Open_ An Autobiography-Andre Agassi -Vintage (2.pdf"
    assert "The Architecture of Open Source Applications.pdf" not in result


def test_local_file_hints_reject_short_title_prefix_without_date_match() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "Dune Messiah",
                "hint_filename": "Dune Messiah.epub",
                "hint_finish_date": "2022-02-19",
            }
        ]
    )

    assert local_file_hints_for_title(hints, "Dune", "Dune", "2022-02-15") == []


def test_local_file_hints_reject_unsafe_short_title_prefix_with_date_match() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "Dune Messiah",
                "hint_filename": "Dune Messiah.epub",
                "hint_finish_date": "2022-02-19",
            }
        ]
    )

    assert local_file_hints_for_title(hints, "Dune", "Dune", "2022-02-16") == []


def test_local_file_hints_preserve_volume_numbers() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "The World Crisis",
                "hint_filename": "The World Crisis.epub",
                "hint_finish_date": "2025-01-01",
            },
            {
                "hint_title": "The World Crisis, Vol. 4",
                "hint_filename": "The World Crisis, Vol. 4.html",
                "hint_finish_date": "2025-01-02",
            },
        ]
    )

    assert (
        local_file_hints_for_title(
            hints,
            "The World Crisis Volume 2",
            "The World Crisis Volume 2",
            "2025-01-01",
        )
        == []
    )


def test_local_file_hints_reject_numbered_volume_for_generic_title() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "The World Crisis, Vol. 2",
                "hint_filename": "The World Crisis, Vol. 2.html",
                "hint_finish_date": "2025-01-01",
            }
        ]
    )

    assert local_file_hints_for_title(hints, "The World Crisis", "", "2025-01-01") == []


def test_local_file_hints_return_only_specific_matched_hint_value() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "The World Crisis",
                "hint_filename": "The World Crisis, Vol. 2.html",
                "hint_finish_date": "2025-01-01",
            }
        ]
    )

    assert local_file_hints_for_title(
        hints, "The World Crisis Volume 2", "", "2025-01-01"
    ) == ["The World Crisis, Vol. 2.html"]


def test_local_file_hints_reject_wrong_volume_even_with_part_number_overlap() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "The World Crisis, Vol. 3 Part 1 and Part 2",
                "hint_filename": "The World Crisis, Vol. 3 Part 1 and Part 2.epub",
                "hint_finish_date": "2025-01-01",
            },
            {
                "hint_title": "The World Crisis, Vol. 2",
                "hint_filename": "The World Crisis, Vol. 2.epub",
                "hint_finish_date": "2025-01-01",
            },
        ]
    )

    assert local_file_hints_for_title(
        hints, "The World Crisis Volume 2", "", "2025-01-01"
    ) == ["The World Crisis, Vol. 2.epub", "The World Crisis, Vol. 2"]


def test_local_file_hints_match_digit_title_to_number_word_export() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "Sixteen Ways to Defend a Walled City",
                "hint_filename": "Sixteen Ways to Defend a Walled City.epub",
                "hint_finish_date": "2023-01-01",
            }
        ]
    )

    assert local_file_hints_for_title(
        hints, "16 ways to defend a walled city", "", "2023-01-01"
    ) == [
        "Sixteen Ways to Defend a Walled City.epub",
        "Sixteen Ways to Defend a Walled City",
    ]


def test_local_file_hints_use_date_bounded_overlap_for_minor_title_typo() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "The Dark Forest (Remembrance of Earth's Past)",
                "hint_filename": "The Dark Forest (Remembrance of Earth_s Past).html",
                "hint_finish_date": "2022-12-27",
            }
        ]
    )

    assert (
        local_file_hints_for_title(hints, "the Dark Forrest", "tdf", "2022-12-27")[0]
        == "The Dark Forest (Remembrance of Earth_s Past).html"
    )


def test_local_file_hints_do_not_use_shared_series_word_as_typo_match() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "Dune Messiah",
                "hint_filename": "Dune Messiah.epub",
                "hint_finish_date": "2022-02-19",
            }
        ]
    )

    assert local_file_hints_for_title(hints, "Children of Dune", "", "2022-02-22") == []


def test_local_file_hints_do_not_mix_ref_matches_when_title_matches() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "[Hyperion 1] Dan Simmons - Hyperion-Saga 1_ Hyperion",
                "hint_filename": "[Hyperion 1] Dan Simmons - Hyperion-Saga 1_ Hyp.pdf",
                "hint_finish_date": "2026-03-09",
            },
            {
                "hint_title": "Hyperion Cantos [02] - The Fall of Hyperion",
                "hint_filename": "Hyperion Cantos [02] - The Fall of Hyperion.epub",
                "hint_finish_date": "2026-03-11",
            },
        ]
    )

    assert local_file_hints_for_title(
        hints,
        "[Hyperion 1] Dan Simmons - Hyperion-Saga 1_ Hyperion (1990)",
        "Hyperion",
        "2026-03-09",
    ) == [
        "[Hyperion 1] Dan Simmons - Hyperion-Saga 1_ Hyp.pdf",
        "[Hyperion 1] Dan Simmons - Hyperion-Saga 1_ Hyperion",
    ]


def test_local_file_hints_allow_repeated_single_word_export_with_date_match() -> None:
    hints = pd.DataFrame(
        [
            {
                "hint_title": "Frank Herbert - Dune 1 - Dune.pdf",
                "hint_filename": "Frank Herbert - Dune 1 - Dune(1).pdf",
                "hint_finish_date": "2022-02-16",
            }
        ]
    )

    assert (
        local_file_hints_for_title(hints, "Dune", "Dune", "2022-02-16")[0]
        == "Frank Herbert - Dune 1 - Dune(1).pdf"
    )


def test_first_clean_value_skips_empty_pandas_values() -> None:
    assert first_clean_value(pd.NA, float("nan"), "", "2022-02-16") == "2022-02-16"


def test_iter_book_files_skips_google_play_notes_html(tmp_path: Path) -> None:
    google_play = tmp_path / "Google Play Books" / "Example"
    google_play.mkdir(parents=True)
    (google_play / "Example.html").write_text("<html>notes</html>", encoding="utf-8")
    (google_play / "Notes.pdf").write_text("<?xml version='1.0'?>", encoding="utf-8")
    (google_play / "Example.epub").write_text("fake epub", encoding="utf-8")
    regular = tmp_path / "Books"
    regular.mkdir()
    (regular / "Local.html").write_text("<html>book</html>", encoding="utf-8")
    (regular / "Local.pdf").write_bytes(b"%PDF-1.7\n")

    paths = {item.path.name for item in iter_book_files([tmp_path])}

    assert "Example.html" not in paths
    assert "Notes.pdf" not in paths
    assert "Example.epub" in paths
    assert "Local.html" in paths
    assert "Local.pdf" in paths


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


def test_build_word_count_outputs_rejects_low_text_pdf_extract(
    tmp_path: Path, monkeypatch
) -> None:
    book_root = tmp_path / "books"
    output_dir = tmp_path / "outputs"
    book_root.mkdir()
    (book_root / "DFW_TV.pdf").write_bytes(b"%PDF-1.7\n")

    def fake_local_text_audit(path: Path, title: str) -> LocalTextAudit:
        return LocalTextAudit(
            raw_word_count=21,
            reading_word_count=21,
            method="pdf_repeated_header_footer_body_pages",
            warning="",
            section_rows=[],
        )

    monkeypatch.setattr(
        book_word_counts_module, "local_text_audit", fake_local_text_audit
    )
    titles = pd.DataFrame(
        [
            {
                "finish_id": "finish_0001",
                "title": "DFW_TV.pdf",
                "finish_date": "2026-01-01",
                "page_count": 407,
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
    assert pd.isna(row["local_file_word_count"])
    assert row["local_raw_word_count"] == 21
    assert row["local_file_error"] == "insufficient_pdf_text_extracted: 21 words"
    assert row["word_count_source"] == "metadata_pages_x_words_per_page"


def test_full_finished_projection_uses_local_calibration_before_mean_imputation() -> (
    None
):
    projection = add_full_finished_projection_columns(
        pd.DataFrame(
            [
                {
                    "title": "Local Pair",
                    "local_file_word_count": 1000,
                    "metadata_page_count": 10,
                    "external_word_count": 500,
                    "chosen_word_count": 1000,
                    "word_count_confidence": "high",
                },
                {
                    "title": "Page Only",
                    "local_file_word_count": None,
                    "metadata_page_count": 5,
                    "external_word_count": None,
                    "chosen_word_count": 1375,
                    "word_count_confidence": "medium",
                },
                {
                    "title": "External Only",
                    "local_file_word_count": None,
                    "metadata_page_count": None,
                    "external_word_count": 250,
                    "chosen_word_count": 250,
                    "word_count_confidence": "medium",
                },
                {
                    "title": "No Inputs",
                    "local_file_word_count": None,
                    "metadata_page_count": None,
                    "external_word_count": None,
                    "chosen_word_count": None,
                    "word_count_confidence": "low",
                },
            ]
        ),
        words_per_page=275,
    ).set_index("title")

    assert projection.loc["Page Only", "projected_word_count"] == 500
    assert projection.loc["External Only", "projected_word_count"] == 500
    assert projection.loc["No Inputs", "projected_word_count"] == 1000
    assert (
        projection.loc["No Inputs", "projected_word_count_method"]
        == "global_mean_local_file_word_count_imputation"
    )
    summary = build_projection_metric_summary(projection.reset_index())
    metrics = summary.set_index("metric")["value"]
    assert metrics["finished_rows"] == 4
    assert metrics["projected_total_words"] == 3000


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
