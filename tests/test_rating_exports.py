from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from Self_Tracking.rating_exports import (
    BookRecord,
    LegacyMovieRating,
    MovieEvent,
    build_book_exports,
    build_movie_identity_conflicts_qc,
    build_movie_exports,
    load_book_title_overrides,
    normalize_book_title,
    normalize_movie_title,
    parse_legacy_movie_entries,
    read_movie_events,
    reconcile_legacy_movie_ratings,
)


def _write_ics(path: Path, events: list[dict[str, str]]) -> None:
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0"]
    for index, event in enumerate(events):
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:event-{index}",
                f"DTSTART:{event.get('start', '20260701T020000Z')}",
                f"DTEND:{event.get('end', '20260701T040000Z')}",
                f"STATUS:{event.get('status', 'CONFIRMED')}",
                f"SUMMARY:{event['summary']}",
            ]
        )
        if event.get("description"):
            lines.append(f"DESCRIPTION:{event['description']}")
        if event.get("creator"):
            lines.append(f"X-APPLE-CREATOR-IDENTITY:{event['creator']}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_movie_parser_accepts_confirmed_amc_and_continuations(tmp_path: Path) -> None:
    ics_path = tmp_path / "Waste Time.ics"
    _write_ics(
        ics_path,
        [
            {
                "summary": "Kiki's Delivery Service 4K",
                "creator": "com.vmbc.amc",
                "description": "The event start time is when trailers begin.",
            },
            {
                "summary": "Young Washington",
                "creator": "com.vmbc.amc",
                "status": "CANCELLED",
            },
            {"summary": "Continued movie: Specter"},
            {"summary": "continued move: roofman"},
        ],
    )

    events = read_movie_events(ics_path, as_of=date(2026, 7, 22))

    assert [(event.title, event.progress) for event in events] == [
        ("Kiki’s Delivery Service", "finished"),
        ("Spectre", "continued"),
        ("Roofman", "continued"),
    ]


def test_movie_export_is_one_row_per_title_and_reuses_old_rating() -> None:
    events = [
        MovieEvent(
            uid="example-old",
            title="The Example Film",
            normalized_title=normalize_movie_title("The Example Film"),
            progress="finished",
            watched_on=date(2024, 1, 10),
            duration_hours=2,
            where_seen="theater",
            source="calendar_explicit",
            summary="Finished movie: The Example Film",
        ),
        MovieEvent(
            uid="example-new",
            title="Example Film",
            normalized_title=normalize_movie_title("Example Film"),
            progress="finished",
            watched_on=date(2026, 4, 10),
            duration_hours=2,
            where_seen="theater",
            source="calendar_explicit",
            summary="Movie: Example Film",
        ),
        MovieEvent(
            uid="new",
            title="Example New Movie",
            normalized_title=normalize_movie_title("Example New Movie"),
            progress="finished",
            watched_on=date(2026, 4, 11),
            duration_hours=0.5,
            where_seen="home",
            source="manual_user_confirmation",
            summary="Finished movie: Example New Movie",
        ),
    ]

    all_rows, new_rows, _ = build_movie_exports(
        events,
        old_ratings={"example film": {"rating": "4", "dates": ["2024-01-10"]}},
        new_since=date(2026, 3, 12),
    )

    assert len(all_rows) == 2
    example = next(row for row in all_rows if row["normalized_title"] == "example film")
    assert example["watch_count"] == 2
    assert example["existing_rating_1_10"] == "4"
    assert [row["normalized_title"] for row in new_rows] == ["example new"]


def test_old_na_label_also_suppresses_repeat_rating() -> None:
    events = [
        MovieEvent(
            uid="unrated-example",
            title="Unrated Example",
            normalized_title=normalize_movie_title("Unrated Example"),
            progress="finished",
            watched_on=date(2026, 4, 15),
            duration_hours=2,
            where_seen="home",
            source="calendar_long_event",
            summary="Movie: Unrated Example",
        )
    ]

    all_rows, new_rows, _ = build_movie_exports(
        events,
        old_ratings={"unrated example": {"rating": "na", "dates": ["2024-01-18"]}},
        new_since=date(2026, 3, 12),
    )

    assert all_rows[0]["existing_rating_1_10"] == "na"
    assert all_rows[0]["needs_rating"] == "false"
    assert new_rows == []


def test_movie_title_selection_has_a_deterministic_case_tiebreaker() -> None:
    title_groups = [
        ("Batman Begins", ["Batman begins", "Batman Begins"]),
        ("A Few Good Men", ["A few Good men", "A few good men"]),
        ("Father of the Bride", ["Father of the Bride", "Father of the bride"]),
    ]
    events = [
        MovieEvent(
            uid=f"movie-{group_index}-{variant_index}",
            title=title,
            normalized_title=normalize_movie_title(title),
            progress="finished",
            watched_on=date(2020, group_index + 1, variant_index + 1),
            duration_hours=2,
            where_seen="home",
            source="calendar_long_event",
            summary=f"Movie: {title}",
        )
        for group_index, (_, variants) in enumerate(title_groups)
        for variant_index, title in enumerate(variants)
    ]

    all_rows, _, _ = build_movie_exports(
        events,
        old_ratings={},
        new_since=date(2026, 3, 12),
    )

    assert {row["movie_title"] for row in all_rows} == {
        expected for expected, _ in title_groups
    }


def test_parse_legacy_movie_entries_preserves_multiline_reviews() -> None:
    entries = parse_legacy_movie_entries(
        "Example Movie Alpha 4; synthetic first line\n"
        "Synthetic continuation.\n\n"
        "Example Movie Beta 8; synthetic second review\n"
    )

    assert [(entry.source_title, entry.rating) for entry in entries] == [
        ("Example Movie Alpha", "4"),
        ("Example Movie Beta", "8"),
    ]
    assert entries[0].review == "synthetic first line\nSynthetic continuation."


def test_legacy_reconciliation_uses_identity_and_quarantines_ambiguous_titles() -> None:
    current_events = [
        MovieEvent(
            uid="western",
            title="Example Western",
            normalized_title=normalize_movie_title("Example Western"),
            progress="finished",
            watched_on=date(2020, 1, 1),
            duration_hours=2,
            where_seen="home",
            source="calendar_long_event",
            summary="Movie: Example Western",
        ),
        MovieEvent(
            uid="ambiguous",
            title="The Example Heat",
            normalized_title=normalize_movie_title("The Example Heat"),
            progress="finished",
            watched_on=date(2020, 1, 2),
            duration_hours=2,
            where_seen="home",
            source="calendar_long_event",
            summary="Movie: The Example Heat",
        ),
        MovieEvent(
            uid="identity-conflict",
            title="Example Epic",
            normalized_title=normalize_movie_title("Example Epic"),
            progress="finished",
            watched_on=date(2020, 1, 3),
            duration_hours=2,
            where_seen="theater",
            source="calendar_confirmed_amc",
            summary="Example Epic",
        ),
    ]
    old_ratings = {
        "example epic": {
            "rating": "8",
            "dates": ["2020-01-03"],
            "identity_title": "Example Epic",
            "identity_year": "2021",
            "identity_source": "calendar_rt_match",
        }
    }
    all_rows, _, movie_qc = build_movie_exports(
        current_events,
        old_ratings=old_ratings,
        new_since=date(2026, 3, 12),
    )
    legacy = [
        LegacyMovieRating(
            source_index=0,
            source_title="Example Western (genre note)",
            normalized_title=normalize_movie_title("Example Western"),
            rating="4",
            review="synthetic review A",
            matched_title="Example Western",
            release_year="2007",
            rt_url="https://example.test/western",
        ),
        LegacyMovieRating(
            source_index=1,
            source_title="Example Heat (performer note)",
            normalized_title=normalize_movie_title("Example Heat"),
            rating="3",
            review="synthetic review B",
            matched_title="Example Heat",
            release_year="1995",
            rt_url="https://example.test/ambiguous",
        ),
        LegacyMovieRating(
            source_index=2,
            source_title="Example Epic (1984)",
            normalized_title=normalize_movie_title("Example Epic"),
            rating="5",
            review="synthetic review C",
            matched_title="Example Epic",
            release_year="1984",
            rt_url="https://example.test/identity-conflict",
        ),
        LegacyMovieRating(
            source_index=3,
            source_title="Legacy Only Example",
            normalized_title=normalize_movie_title("Legacy Only Example"),
            rating="7",
            review="synthetic review D",
            matched_title="Legacy Only Example",
            release_year="2000",
            rt_url="https://example.test/legacy-only",
        ),
    ]

    reconciled, new_rows, legacy_qc = reconcile_legacy_movie_ratings(
        all_rows,
        legacy,
        movie_qc_rows=movie_qc,
    )

    western = next(
        row for row in reconciled if row["normalized_title"] == "example western"
    )
    assert western["existing_rating_1_10"] == "4"
    assert western["existing_review"] == "synthetic review A"
    assert western["movie_identity_key"] == "example western|2007"
    ambiguous = next(
        row for row in reconciled if row["movie_title"] == "The Example Heat"
    )
    assert ambiguous["existing_rating_1_10"] == ""
    epic = next(row for row in reconciled if row["movie_title"] == "Example Epic")
    assert epic["existing_rating_1_10"] == "8"
    legacy_only = next(
        row for row in reconciled if row["movie_title"] == "Legacy Only Example"
    )
    assert legacy_only["existing_rating_1_10"] == "7"
    assert legacy_only["completion_sources"] == "legacy_review_notes"
    statuses = {
        row["legacy_source_title"]: row["reconciliation_status"] for row in legacy_qc
    }
    assert statuses["Example Western (genre note)"] == "auto_joined_exact_title"
    assert statuses["Example Heat (performer note)"] == "quarantined_ambiguous_title"
    assert statuses["Example Epic (1984)"] == "quarantined_identity_conflict"
    assert statuses["Legacy Only Example"] == "legacy_only_added"
    assert new_rows == []


def test_confirmed_legacy_adjudications_canonicalize_movie_identities() -> None:
    adjudications = [
        (
            "American Pickle",
            "An American Pickle",
            "An American Pickle",
            "2020",
        ),
        (
            "The Heat",
            "Heat (val Kilmer, Al pacino, 1995)",
            "Heat",
            "1995",
        ),
        (
            "Boondock Saints",
            "The Boondock Saints",
            "The Boondock Saints",
            "1999",
        ),
        (
            "don’t let the devil know you’re dead",
            "Before the Devil knows you’re dead",
            "Before the Devil Knows You're Dead",
            "2007",
        ),
        (
            "the lover and the gentleman",
            "A Lover and a Gentleman",
            "An Officer and a Gentleman",
            "1982",
        ),
        (
            "League of Extraordinary Gentlemen",
            "the league of extraordinary gentlemen",
            "The League of Extraordinary Gentlemen",
            "2003",
        ),
    ]
    current_events = [
        MovieEvent(
            uid=f"movie-{index}",
            title=candidate_title,
            normalized_title=normalize_movie_title(candidate_title),
            progress="finished",
            watched_on=date(2021, 5, 5),
            duration_hours=2,
            where_seen="home",
            source="calendar_long_event",
            summary=f"Movie: {candidate_title}",
        )
        for index, (candidate_title, _, _, _) in enumerate(adjudications)
    ]
    all_rows, _, movie_qc = build_movie_exports(
        current_events,
        old_ratings={},
        new_since=date(2026, 3, 12),
    )
    legacy = [
        LegacyMovieRating(
            source_index=index,
            source_title=source_title,
            normalized_title=normalize_movie_title(source_title),
            rating="1",
            review="synthetic adjudication review",
            matched_title=canonical_title,
            release_year=release_year,
            rt_url=f"https://example.test/movie-{index}",
        )
        for index, (_, source_title, canonical_title, release_year) in enumerate(
            adjudications
        )
    ]

    reconciled, _, legacy_qc = reconcile_legacy_movie_ratings(
        all_rows,
        legacy,
        movie_qc_rows=movie_qc,
    )

    rows_by_title = {str(row["movie_title"]): row for row in reconciled}
    qc_by_source = {str(row["legacy_source_title"]): row for row in legacy_qc}
    for _, source_title, canonical_title, release_year in adjudications:
        row = rows_by_title[canonical_title]
        assert row["existing_rating_1_10"] == "1"
        assert str(row["movie_identity_key"]).endswith(f"|{release_year}")
        assert row["identity_source"] == ("legacy_rt_enrichment_manual_adjudication")
        assert (
            qc_by_source[source_title]["reconciliation_status"]
            == "auto_joined_manual_adjudication"
        )


def test_duplicate_movie_identities_are_reported_without_merging() -> None:
    rows = [
        {
            "movie_title": "Example Movie",
            "normalized_title": "example movie",
            "movie_identity_key": "example movie|2001",
            "identity_title": "Example Movie",
            "identity_year": "2001",
            "identity_source": "calendar_rt_match",
            "existing_rating_1_10": "3",
            "watch_dates": "2020-01-01",
        },
        {
            "movie_title": "Example Movie Two",
            "normalized_title": "example movie two",
            "movie_identity_key": "example movie|2001",
            "identity_title": "Example Movie",
            "identity_year": "2001",
            "identity_source": "calendar_rt_match",
            "existing_rating_1_10": "8",
            "watch_dates": "2021-01-01",
        },
    ]

    qc_rows = build_movie_identity_conflicts_qc(rows)

    assert len(rows) == 2
    assert len(qc_rows) == 1
    assert qc_rows[0]["row_count"] == 2
    assert qc_rows[0]["issue"] == "duplicate_identity_rating_conflict"
    assert qc_rows[0]["movie_titles"] == "Example Movie | Example Movie Two"


def test_movie_aliases_cover_user_corrections() -> None:
    assert normalize_movie_title("Ricky stalicky") == "ricky stanley"
    assert normalize_movie_title("Ricky Stanley") == "ricky stanley"
    assert normalize_movie_title("the sheep detective") == "sheep detectives"
    assert normalize_movie_title("The Sheep Detectives") == "sheep detectives"
    assert normalize_movie_title("specter") == "spectre"


def test_book_finished_calendar_is_superset_and_rereads_are_retained() -> None:
    prior = [
        BookRecord(
            title="Example Memoir",
            normalized_title=normalize_book_title("Example Memoir"),
            author="Example Author",
            finished_on=date(2025, 1, 10),
            completion_source="prior_holdout",
        ),
        BookRecord(
            title="Earlier Example Book",
            normalized_title=normalize_book_title("Earlier Example Book"),
            author="Example Author",
            finished_on=date(2026, 3, 11),
            completion_source="prior_holdout",
        ),
    ]
    play = [
        BookRecord(
            title="Example Technical Book",
            normalized_title=normalize_book_title("Example Technical Book"),
            author="Example Author",
            finished_on=None,
            completion_source="play_unfinished",
            play_finished=False,
        )
    ]
    calendar = [
        BookRecord(
            title="Example Memoir",
            normalized_title=normalize_book_title("Example Memoir"),
            author="",
            finished_on=date(2026, 4, 1),
            completion_source="calendar_finished",
        ),
        BookRecord(
            title="Example Technical Book",
            normalized_title=normalize_book_title("Example Technical Book"),
            author="",
            finished_on=date(2026, 4, 2),
            completion_source="calendar_finished",
        ),
    ]

    all_rows, new_rows, qc_rows = build_book_exports(
        prior_records=prior,
        play_records=play,
        calendar_finished=calendar,
        new_since=date(2026, 3, 12),
    )

    assert len(all_rows) == 4
    assert {row["title"] for row in new_rows} == {
        "Example Memoir",
        "Example Technical Book",
    }
    memoir = next(row for row in new_rows if row["title"] == "Example Memoir")
    assert memoir["is_reread"] == "true"
    technical = next(
        row for row in new_rows if row["title"] == "Example Technical Book"
    )
    assert technical["play_finished"] == "false"
    assert technical["completion_source"] == "calendar_finished"
    assert any(row["issue"] == "calendar_finished_play_unfinished" for row in qc_rows)


def test_book_title_aliases_load_from_local_override(tmp_path: Path) -> None:
    override_path = tmp_path / "book_title_overrides.json"
    override_path.write_text(
        """
{
  "aliases": {"short example": "canonical example"},
  "prefix_aliases": {"long example": "canonical example"},
  "preferred_titles": {"canonical example": "Canonical Example"}
}
""".strip(),
        encoding="utf-8",
    )

    aliases, prefix_aliases, preferred_titles = load_book_title_overrides(override_path)

    assert normalize_book_title("Short Example", aliases=aliases) == "canonical example"
    assert (
        normalize_book_title(
            "Long Example: A Subtitle",
            prefix_aliases=prefix_aliases,
        )
        == "canonical example"
    )
    assert preferred_titles["canonical example"] == "Canonical Example"


def test_csv_fixture_import_is_not_required() -> None:
    """Keep csv imported: generated output is deliberately ordinary CSV."""
    assert csv.excel.delimiter == ","
