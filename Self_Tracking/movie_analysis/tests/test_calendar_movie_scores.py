from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis_core.calendar_movie_scores import (  # noqa: E402
    CalendarEvent,
    build_candidate_movies,
    candidate_cache_key,
    canonicalize_movie_title,
    extract_movie_events,
    filter_new_ambiguities,
    imdb_query_for_title,
    infer_movie_title,
    normalize_tracking_title,
    rt_direct_match_override_for_title,
    rt_query_for_title,
)


def _event(
    summary: str,
    *,
    start: str,
    end: str,
    calendar_name: str = "Waste Time",
) -> CalendarEvent:
    tz = ZoneInfo("America/Los_Angeles")
    start_dt = datetime.fromisoformat(start).replace(tzinfo=tz)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=tz)
    return CalendarEvent(
        calendar_name=calendar_name,
        summary=summary,
        description="",
        location="",
        start_local=start_dt,
        end_local=end_dt,
        duration_hours=(end_dt - start_dt).total_seconds() / 3600.0,
    )


def test_infer_movie_title_parses_home_and_theater_summaries() -> None:
    assert infer_movie_title("Movie: Licorice Pizza", "", "") == (
        "Licorice Pizza",
        "home",
        "default",
    )
    assert infer_movie_title(
        "Dune: Part Two at AMC Metreon 16",
        "",
        "AMC Metreon 16",
    ) == ("Dune: Part Two", "theater", "default")
    assert infer_movie_title(
        "Spider-Man: Across the Spider-Verse at Century San Francisco Centre 9 and XD",
        "",
        "845 Market Street",
    ) == ("Spider-Man: Across the Spider-Verse", "theater", "default")
    assert infer_movie_title("Finish movie: The Nice Guys", "", "") == (
        "The Nice Guys",
        "home",
        "finished",
    )


def test_normalize_tracking_title_handles_existing_aliases() -> None:
    assert normalize_tracking_title("Kill Bill II") == "kill bill 2"
    assert normalize_tracking_title("Movie: Mad Max (1980)") == "max max"
    assert (
        normalize_tracking_title("Movie: don’t let the devil know you’re dead")
        == "before devil knows you re dead"
    )
    assert (
        normalize_tracking_title("Movie: Charlie Wilson’s wall")
        == "charlie wilson s war"
    )
    assert normalize_tracking_title("Finished movie: wolf’s") == "wolfs"


def test_canonicalize_movie_title_fixes_known_calendar_typos() -> None:
    assert canonicalize_movie_title("Vengance") == "Vengeance"
    assert (
        canonicalize_movie_title("Spider-Man In to the Multiverse")
        == "Spider-Man: Across the Spider-Verse"
    )
    assert canonicalize_movie_title("mission impossible 4") == "mission impossible 5"


def test_query_overrides_cover_known_ambiguous_titles() -> None:
    assert rt_query_for_title("Dune 1") == "Dune 2021"
    assert rt_query_for_title("golden eye") == "GoldenEye"
    assert rt_direct_match_override_for_title("golden eye") == {
        "rt_url": "https://www.rottentomatoes.com/m/goldeneye",
        "release_year": 1995,
    }
    assert rt_query_for_title("goldfinger") == "Goldfinger"
    assert rt_direct_match_override_for_title("goldfinger") == {
        "rt_url": "https://www.rottentomatoes.com/m/goldfinger",
        "release_year": 1964,
    }
    assert rt_query_for_title("Sicario 2") == "Sicario: Day of the Soldado"
    assert rt_query_for_title("Aquaman 2") == "Aquaman and the Lost Kingdom"
    assert rt_query_for_title("run away jury") == "Runaway Jury"
    assert rt_query_for_title("fall guy") == "The Fall Guy 2024"
    assert rt_query_for_title("The Naked Gun") == "The Naked Gun 2025"
    assert rt_query_for_title("sense and sensibility") == "Sense and Sensibility 2026"
    assert rt_direct_match_override_for_title("made") == {
        "rt_url": "https://www.rottentomatoes.com/m/made",
        "release_year": 2001,
    }
    assert imdb_query_for_title("bad boys 3") == "Bad Boys for Life"
    assert imdb_query_for_title("argyle") == "Argylle"
    assert imdb_query_for_title("mission impossible 7") == (
        "Mission: Impossible - Dead Reckoning Part One"
    )
    assert imdb_query_for_title("license to kill") == "Licence to Kill"
    assert imdb_query_for_title("Team America") == "Team America: World Police"
    assert imdb_query_for_title("what happened to the Morgan’s") == (
        "Did You Hear About the Morgans?"
    )
    assert imdb_query_for_title("the hitman") == "Hit Man"
    assert imdb_query_for_title("The Naked Gun") == "The Naked Gun 2025"
    assert imdb_query_for_title("Vengance") == "Vengeance 2022"


def test_candidate_cache_key_distinguishes_same_normalized_title_rows() -> None:
    movie_events = extract_movie_events(
        [
            _event(
                "Movie: Naked gun",
                start="2023-09-15T11:00:00",
                end="2023-09-15T13:00:00",
            ),
            _event(
                "The Naked Gun at AMC Century City 15",
                start="2025-08-08T18:45:00",
                end="2025-08-08T21:00:00",
            ),
        ]
    )

    candidates, ambiguities = build_candidate_movies(movie_events, [])

    assert not ambiguities
    assert len(candidates) == 2
    assert candidate_cache_key(candidates[0]) != candidate_cache_key(candidates[1])


def test_build_candidate_movies_prefers_finish_marker_and_flags_prior_drink() -> None:
    movie_events = extract_movie_events(
        [
            _event(
                "Started movie: Licorice Pizza",
                start="2025-09-29T19:00:00",
                end="2025-09-29T19:45:00",
            ),
            _event(
                "Finished movie: Licorice Pizza",
                start="2026-02-16T20:00:00",
                end="2026-02-16T22:00:00",
            ),
            _event(
                "Movie: Robocop",
                start="2025-09-29T21:00:00",
                end="2025-09-29T23:00:00",
            ),
        ]
    )
    drink_events = [
        _event(
            "Drink",
            start="2026-02-16T18:00:00",
            end="2026-02-16T18:30:00",
        )
    ]
    candidates, ambiguities = build_candidate_movies(movie_events, drink_events)

    assert not ambiguities
    licorice = next(
        candidate
        for candidate in candidates
        if candidate.normalized_title == "licorice pizza"
    )
    assert licorice.completion_basis == "explicit_finish_marker"
    assert licorice.date_local.date().isoformat() == "2026-02-16"
    assert licorice.drink_before_movie is True

    robocop = next(
        candidate for candidate in candidates if candidate.normalized_title == "robocop"
    )
    assert robocop.completion_basis == "long_single_event"
    assert robocop.drink_before_movie is False


def test_build_candidate_movies_splits_rewatches_into_separate_clusters() -> None:
    movie_events = extract_movie_events(
        [
            _event(
                "Movie: Dune",
                start="2024-01-01T19:00:00",
                end="2024-01-01T21:30:00",
            ),
            _event(
                "Movie: Dune",
                start="2024-03-15T19:00:00",
                end="2024-03-15T21:30:00",
            ),
        ]
    )

    candidates, ambiguities = build_candidate_movies(movie_events, [])

    assert not ambiguities
    assert len(candidates) == 2
    assert [candidate.date_local.date().isoformat() for candidate in candidates] == [
        "2024-01-01",
        "2024-03-15",
    ]


def test_build_candidate_movies_leaves_started_only_entries_ambiguous() -> None:
    movie_events = extract_movie_events(
        [
            _event(
                "Started movie: The Eagle Has Landed",
                start="2026-03-06T19:00:00",
                end="2026-03-06T21:00:00",
            )
        ]
    )
    candidates, ambiguities = build_candidate_movies(movie_events, [])

    assert not candidates
    assert ambiguities[0]["calendar_title"] == "The Eagle Has Landed"

    candidates, ambiguities = build_candidate_movies(
        movie_events,
        [],
        include_unfinished_started=True,
    )

    assert not ambiguities
    assert candidates[0].watch_status == "unfinished"
    assert candidates[0].completion_basis == "started_only_event"


def test_filter_new_ambiguities_keeps_only_post_cutoff_untracked_titles() -> None:
    movie_events = extract_movie_events(
        [
            _event(
                "Started movie: Old Pick",
                start="2025-06-10T19:00:00",
                end="2025-06-10T20:00:00",
            ),
            _event(
                "Started movie: New Pick",
                start="2025-06-20T19:00:00",
                end="2025-06-20T20:00:00",
            ),
            _event(
                "Started movie: Tracked Pick",
                start="2025-06-21T19:00:00",
                end="2025-06-21T20:00:00",
            ),
        ]
    )
    ambiguities = [
        {
            "normalized_title": "old pick",
            "calendar_title": "Old Pick",
            "reason": "Could not confirm completion from calendar durations",
            "summaries": "Started movie: Old Pick",
        },
        {
            "normalized_title": "new pick",
            "calendar_title": "New Pick",
            "reason": "Could not confirm completion from calendar durations",
            "summaries": "Started movie: New Pick",
        },
        {
            "normalized_title": "tracked pick",
            "calendar_title": "Tracked Pick",
            "reason": "Could not confirm completion from calendar durations",
            "summaries": "Started movie: Tracked Pick",
        },
    ]

    filtered = filter_new_ambiguities(
        ambiguities,
        movie_events,
        {"tracked pick"},
        datetime.fromisoformat("2025-06-14T19:30:00").replace(
            tzinfo=ZoneInfo("America/Los_Angeles")
        ),
    )

    assert [row["normalized_title"] for row in filtered] == ["new pick"]

    filtered_with_min_date = filter_new_ambiguities(
        ambiguities,
        movie_events,
        {"tracked pick"},
        datetime.fromisoformat("2025-06-14T19:30:00").replace(
            tzinfo=ZoneInfo("America/Los_Angeles")
        ),
        min_date_local=datetime.fromisoformat("2025-06-21T00:00:00").replace(
            tzinfo=ZoneInfo("America/Los_Angeles")
        ),
    )

    assert filtered_with_min_date == []
