from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ai_books_tracking.book_calendar_time import (
    build_calendar_time_outputs,
    classify_processed_calendar,
)
from ai_books_tracking.book_wpm_calendar_notes import (
    DocxParagraph,
    MetadataRecord,
    abbreviation_matches_title,
    aggregate_books,
    build_finish_timeline,
    build_time_pages_outliers,
    classify_summary_part,
    note_stats_from_paragraphs,
    parse_ical_datetime,
    resolve_reading_events,
    resolve_title,
    title_initial_variants,
    titles_match,
    unfold_ics_lines,
)


def test_unfold_ics_lines_joins_continuations() -> None:
    text = "SUMMARY:Book: The Path to\n  Power\nDTSTART:20250504T100000Z\n"

    assert unfold_ics_lines(text)[0] == "SUMMARY:Book: The Path to Power"


def test_parse_ical_datetime_converts_utc_to_display_timezone() -> None:
    parsed = parse_ical_datetime(
        "20250504T100000Z",
        {},
        display_tz=ZoneInfo("America/Los_Angeles"),
    )

    assert parsed == datetime(2025, 5, 4, 3, 0, tzinfo=ZoneInfo("America/Los_Angeles"))


def test_classify_summary_part_handles_misnamed_finished_book() -> None:
    event_type, ref = classify_summary_part(
        "Finished book: lyndon johnson the path to power",
        date(2025, 5, 4),
    )

    assert event_type == "finished"
    assert ref == "The Path to Power: The Years of Lyndon Johnson I"


def test_classify_summary_part_accepts_read_and_listen_entries() -> None:
    event_type, ref = classify_summary_part(
        "read: The Checklist Manifesto", date(2024, 3, 19)
    )
    assert event_type == "reading"
    assert ref == "The Checklist Manifesto"

    event_type, ref = classify_summary_part("listen to tgd", date(2025, 1, 10))
    assert event_type == "audiobook"
    assert ref == "tgd"


def test_classify_summary_part_keeps_bare_books_generic() -> None:
    for text in ["Books", "books", "cs books"]:
        event_type, ref = classify_summary_part(text, date(2025, 1, 1))
        assert event_type == "generic"
        assert ref == ""


def test_title_matching_and_abbreviations_cover_articles() -> None:
    assert titles_match("shape up", "shaping up")
    assert abbreviation_matches_title(
        "tpv",
        "The Price of Victory",
    )
    assert abbreviation_matches_title(
        "tptp",
        "The Path to Power: The Years of Lyndon Johnson I",
    )
    assert "sc" in title_initial_variants("Six Crises")
    assert abbreviation_matches_title("sc", "Six Crises")
    assert "5ebgo" in title_initial_variants("50 essays by george orwell")
    assert abbreviation_matches_title("5ebgo", "50 essays by george orwell")
    assert abbreviation_matches_title(
        "hotesp",
        "the history of the English speaking peoples volume 4",
    )
    assert abbreviation_matches_title("nvc", "nonviolent communication")
    assert abbreviation_matches_title("tinad", "there is no anti-memetic division")
    assert abbreviation_matches_title(
        "su", "Shape Up Stop Running in Circles and Ship Work that Matters"
    )
    assert abbreviation_matches_title(
        "tptp", "The Path to Power: The Years of Lyndon Johnson I"
    )
    assert not abbreviation_matches_title(
        "pp", "The Path to Power: The Years of Lyndon Johnson I"
    )
    assert not abbreviation_matches_title("dune", "Dune Messiah")


def test_resolve_title_keeps_path_to_power_aliases_canonical() -> None:
    known_titles = [
        "the path to power",
        "The Path to Power: The Years of Lyndon Johnson I",
    ]

    assert (
        resolve_title("tptp", known_titles)
        == "The Path to Power: The Years of Lyndon Johnson I"
    )
    assert (
        resolve_title("the path to power", known_titles)
        == "The Path to Power: The Years of Lyndon Johnson I"
    )
    assert resolve_title("pp", known_titles) == "pp"


def test_overlap_policy_restores_only_overlaps_longer_than_15_minutes() -> None:
    processed = pd.DataFrame(
        [
            {
                "event_id": "book_walk",
                "event_name": "Book: Example",
                "calendar_name": "Things",
                "start_time": pd.Timestamp("2025-01-01T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-01T11:00:00Z"),
                "duration": 0.5,
                "calendar_analysis_duration_hours": 0.5,
                "wall_clock_duration_hours": 1.0,
            },
            {
                "event_id": "walk",
                "event_name": "walk",
                "calendar_name": "Things",
                "start_time": pd.Timestamp("2025-01-01T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-01T11:00:00Z"),
                "duration": 0.5,
                "calendar_analysis_duration_hours": 0.5,
                "wall_clock_duration_hours": 1.0,
            },
            {
                "event_id": "book_job",
                "event_name": "Book: Work Example",
                "calendar_name": "Things",
                "start_time": pd.Timestamp("2025-01-02T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-02T11:00:00Z"),
                "duration": 0.5,
                "calendar_analysis_duration_hours": 0.5,
                "wall_clock_duration_hours": 1.0,
            },
            {
                "event_id": "job",
                "event_name": "Job: Hive",
                "calendar_name": "Work",
                "start_time": pd.Timestamp("2025-01-02T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-02T11:00:00Z"),
                "duration": 0.5,
                "calendar_analysis_duration_hours": 0.5,
                "wall_clock_duration_hours": 1.0,
            },
            {
                "event_id": "audio_gym",
                "event_name": "Audiobook: Example",
                "calendar_name": "Things",
                "start_time": pd.Timestamp("2025-01-03T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-03T11:00:00Z"),
                "duration": 0.5,
                "calendar_analysis_duration_hours": 0.5,
                "wall_clock_duration_hours": 1.0,
            },
            {
                "event_id": "gym",
                "event_name": "gym",
                "calendar_name": "Things",
                "start_time": pd.Timestamp("2025-01-03T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-03T11:00:00Z"),
                "duration": 0.5,
                "calendar_analysis_duration_hours": 0.5,
                "wall_clock_duration_hours": 1.0,
            },
            {
                "event_id": "book_short",
                "event_name": "Book: Short Example",
                "calendar_name": "Things",
                "start_time": pd.Timestamp("2025-01-04T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-04T10:15:00Z"),
                "duration": 0.125,
                "calendar_analysis_duration_hours": 0.125,
                "wall_clock_duration_hours": 0.25,
            },
            {
                "event_id": "short_overlap",
                "event_name": "walk",
                "calendar_name": "Things",
                "start_time": pd.Timestamp("2025-01-04T10:00:00Z"),
                "end_time": pd.Timestamp("2025-01-04T10:15:00Z"),
                "duration": 0.125,
                "calendar_analysis_duration_hours": 0.125,
                "wall_clock_duration_hours": 0.25,
            },
        ]
    )

    classified = classify_processed_calendar(processed)

    walk_book = classified[classified["source_row_id"].eq("book_walk")].iloc[0]
    job_book = classified[classified["source_row_id"].eq("book_job")].iloc[0]
    audio_gym = classified[classified["source_row_id"].eq("audio_gym")].iloc[0]
    short_book = classified[classified["source_row_id"].eq("book_short")].iloc[0]
    assert walk_book["overlap_policy_duration_hours"] == 1.0
    assert walk_book["overlap_policy_rule"] == "full_duration_overlap_gt_15m"
    assert job_book["overlap_policy_duration_hours"] == 1.0
    assert job_book["overlap_policy_rule"] == "full_duration_overlap_gt_15m"
    assert audio_gym["overlap_policy_duration_hours"] == 1.0
    assert audio_gym["overlap_policy_rule"] == "full_duration_overlap_gt_15m"
    assert short_book["overlap_policy_duration_hours"] == 0.125
    assert (
        short_book["overlap_policy_rule"]
        == "calendar_analysis_duration_overlap_lte_15m"
    )
    assert short_book["overlap_total_minutes"] == 15.0


def test_finished_and_started_audiobook_events_count_as_audio_minutes(
    tmp_path,
) -> None:
    calendar_tsv = tmp_path / "calendar_analysis.txt"
    pd.DataFrame(
        [
            {
                "event_name": "Started audiobook: Example Audio",
                "calendar_name": "Things",
                "start_time": "2026-01-01T10:00:00Z",
                "end_time": "2026-01-01T10:30:00Z",
                "duration": 0.5,
            },
            {
                "event_name": "Finished audiobook: Example Audio",
                "calendar_name": "Things",
                "start_time": "2026-01-02T10:00:00Z",
                "end_time": "2026-01-02T11:00:00Z",
                "duration": 1.0,
            },
        ]
    ).to_csv(calendar_tsv, sep="\t", index=False)

    totals, resolved, _ = build_calendar_time_outputs(
        calendar_analysis_tsv=calendar_tsv,
        output_dir=tmp_path / "outputs",
        notes_dir=tmp_path / "missing_notes",
    )

    row = totals.iloc[0]
    assert row["primary_first_pass_minutes"] == 90
    assert row["audiobook_overlap_policy_minutes"] == 90
    assert row["reading_overlap_policy_minutes"] == 0
    assert row["finished_overlap_policy_minutes"] == 60
    assert resolved["is_audiobook_time"].tolist() == [True, True]


def test_build_finish_timeline_keeps_rereads_as_separate_instances() -> None:
    events = pd.DataFrame(
        [
            {
                "event_type": "finished",
                "date": date(2025, 1, 1),
                "start": pd.Timestamp("2025-01-01"),
                "book_ref": "Example Book",
                "calendar_name": "test",
            },
            {
                "event_type": "finished",
                "date": date(2025, 1, 1),
                "start": pd.Timestamp("2025-01-01 01:00"),
                "book_ref": "Example Book",
                "calendar_name": "test",
            },
            {
                "event_type": "finished",
                "date": date(2025, 6, 1),
                "start": pd.Timestamp("2025-06-01"),
                "book_ref": "Example Book",
                "calendar_name": "test",
            },
        ]
    )

    finishes = build_finish_timeline(events, ["Example Book"])

    assert len(finishes) == 2
    assert finishes["read_instance"].tolist() == [1, 2]
    assert finishes["is_reread"].tolist() == [False, True]


def test_note_stats_separates_highlight_text_from_user_notes() -> None:
    paragraphs = [
        DocxParagraph("Example Book", styles=("Heading1",)),
        DocxParagraph("Example Author", colors=("424242",)),
        DocxParagraph("Annotations by color", styles=("Heading1",)),
        DocxParagraph("Yellow", styles=("Heading2",)),
        DocxParagraph("Highlighted passage.", fills=("fde096",)),
        DocxParagraph("My separate note.", colors=("424242",)),
        DocxParagraph("May 4, 2025", colors=("757575",)),
        DocxParagraph("123", colors=("1565c0",)),
    ]

    stats = note_stats_from_paragraphs(paragraphs, fallback_title="fallback")

    assert stats.title == "Example Book"
    assert stats.highlight_count == 1
    assert stats.highlight_words == 2
    assert stats.note_count == 1
    assert stats.note_words == 3
    assert stats.max_page == 123


def test_aggregate_books_adjusts_print_wpm_for_audiobook_words() -> None:
    finishes = pd.DataFrame(
        [
            {
                "finish_id": "finish_0001",
                "finish_date": date(2025, 1, 2),
                "cal_ref": "Mixed Book",
                "title": "Mixed Book",
                "read_instance": 1,
                "is_reread": False,
                "previous_finish_date": pd.NaT,
                "days_since_previous_finish": pd.NA,
            }
        ]
    )
    resolved_events = pd.DataFrame(
        [
            {
                "date": date(2025, 1, 1),
                "title": "Mixed Book",
                "finish_id": "finish_0001",
                "event_type": "reading",
                "duration_hours": 1.0,
                "match_method": "test",
            },
            {
                "date": date(2025, 1, 1),
                "title": "Mixed Book",
                "finish_id": "finish_0001",
                "event_type": "audiobook",
                "duration_hours": 1.0,
                "match_method": "test",
            },
        ]
    )

    books = aggregate_books(
        finishes,
        resolved_events,
        [MetadataRecord("Mixed Book", page_count=120)],
        [],
        words_per_page=275,
        audiobook_wpm=350,
    )

    row = books.iloc[0]
    assert row["estimated_words"] == 33000
    assert row["observed_total_wpm"] == 275
    assert row["audiobook_words_assumed"] == 21000
    assert row["print_words_after_audiobook_assumption"] == 12000
    assert row["wpm"] == 200
    assert row["wpm_basis"] == "print_words_after_audiobook_assumption"


def test_resolve_reading_events_lets_manual_abbrev_bypass_candidate_narrowing() -> None:
    finishes = pd.DataFrame(
        [
            {
                "finish_id": "finish_0001",
                "finish_date": date(2026, 1, 11),
                "cal_ref": "the history of the English speaking peoples volume 4",
                "title": "the history of the English speaking peoples volume 4",
                "read_instance": 1,
                "is_reread": False,
                "previous_finish_date": pd.NaT,
                "days_since_previous_finish": pd.NA,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 10),
                "start": pd.Timestamp("2026-01-10"),
                "book_ref": "tgd",
                "event_type": "audiobook",
                "duration_hours": 1.0,
            }
        ]
    )

    resolved = resolve_reading_events(events, finishes)

    assert resolved.iloc[0]["title"] == (
        "the history of the English speaking peoples volume 4"
    )
    assert resolved.iloc[0]["finish_id"] == "finish_0001"
    assert resolved.iloc[0]["match_method"] == "ref_to_finished_title"


def test_resolve_reading_events_counts_started_book_entries() -> None:
    finishes = pd.DataFrame(
        [
            {
                "finish_id": "finish_0001",
                "finish_date": date(2026, 5, 14),
                "cal_ref": "managing the professional services firm",
                "title": "managing the professional services firm",
                "read_instance": 1,
                "is_reread": False,
                "previous_finish_date": pd.NaT,
                "days_since_previous_finish": pd.NA,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 4, 28),
                "start": pd.Timestamp("2026-04-28"),
                "book_ref": "managing the professional service firm",
                "event_type": "started",
                "duration_hours": 0.5,
            }
        ]
    )

    resolved = resolve_reading_events(events, finishes)

    assert resolved.iloc[0]["title"] == "managing the professional services firm"
    assert resolved.iloc[0]["finish_id"] == "finish_0001"


def test_resolve_reading_events_uses_unique_title_fragment_candidates() -> None:
    finishes = pd.DataFrame(
        [
            {
                "finish_id": "finish_0001",
                "finish_date": date(2025, 11, 13),
                "cal_ref": "and call me conrad",
                "title": "and call me conrad",
                "read_instance": 1,
                "is_reread": False,
                "previous_finish_date": pd.NaT,
                "days_since_previous_finish": pd.NA,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "date": date(2025, 11, 1),
                "start": pd.Timestamp("2025-11-01"),
                "book_ref": "Conrad",
                "event_type": "reading",
                "duration_hours": 1.0,
            }
        ]
    )

    resolved = resolve_reading_events(events, finishes)

    assert resolved.iloc[0]["title"] == "and call me conrad"
    assert resolved.iloc[0]["finish_id"] == "finish_0001"


def test_build_time_pages_outliers_flags_short_time_large_page_rows() -> None:
    books = pd.DataFrame(
        [
            {
                "title": "A",
                "finish_date": date(2025, 1, 1),
                "pages": 100,
                "total_reading_hours": 2,
                "print_reading_hours": 2,
                "audiobook_hours": 0,
                "wpm": 229,
                "observed_total_wpm": 229,
                "page_source": "test",
                "wpm_basis": "test",
            },
            {
                "title": "B",
                "finish_date": date(2025, 1, 2),
                "pages": 200,
                "total_reading_hours": 4,
                "print_reading_hours": 4,
                "audiobook_hours": 0,
                "wpm": 229,
                "observed_total_wpm": 229,
                "page_source": "test",
                "wpm_basis": "test",
            },
            {
                "title": "C",
                "finish_date": date(2025, 1, 3),
                "pages": 300,
                "total_reading_hours": 6,
                "print_reading_hours": 6,
                "audiobook_hours": 0,
                "wpm": 229,
                "observed_total_wpm": 229,
                "page_source": "test",
                "wpm_basis": "test",
            },
            {
                "title": "Fast",
                "finish_date": date(2025, 1, 4),
                "pages": 400,
                "total_reading_hours": 0.5,
                "print_reading_hours": 0.5,
                "audiobook_hours": 0,
                "wpm": 3667,
                "observed_total_wpm": 3667,
                "page_source": "test",
                "wpm_basis": "test",
            },
        ]
    )

    outliers = build_time_pages_outliers(books)

    assert outliers.iloc[0]["title"] == "Fast"
    assert outliers.iloc[0]["time_pages_outlier_direction"] == (
        "faster_than_page_count_predicts"
    )
