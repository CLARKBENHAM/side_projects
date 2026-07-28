#!/usr/bin/env python3
"""Build timestamped, review-safe book and movie rating exports.

The command combines prior labeled data with a new Calendar Takeout and Google
Play Books export. It does not fetch external ratings and never writes to a
previous run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from Self_Tracking.play_books_to_csv import process_google_books_exports


def _system_local_zone() -> ZoneInfo:
    configured = os.environ.get("TZ", "").strip()
    if configured:
        return ZoneInfo(configured)
    localtime_parts = Path("/etc/localtime").resolve().parts
    if "zoneinfo" in localtime_parts:
        zone_index = localtime_parts.index("zoneinfo") + 1
        return ZoneInfo("/".join(localtime_parts[zone_index:]))
    return ZoneInfo("UTC")


LOCAL_TZ = _system_local_zone()
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALENDAR_DIR = DEFAULT_ROOT / "data/takeout_07_21_26/Calendar"
DEFAULT_PLAY_DIR = (
    DEFAULT_ROOT / "data/play_books_export_small_failure_07_23_2026/Google Play Books"
)
DEFAULT_MOVIE_LABELS = (
    DEFAULT_ROOT
    / "Self_Tracking/movie_analysis/data/summaries"
    / "Movie Ratings - new_calendar_movies_features-labeled.csv"
)
DEFAULT_LEGACY_MOVIE_NOTES = (
    DEFAULT_ROOT / "Self_Tracking/movie_analysis/data/summaries/movie_rt_notes.txt"
)
DEFAULT_LEGACY_MOVIE_DETAILS = (
    DEFAULT_ROOT
    / "Self_Tracking/movie_analysis/data/summaries/movie_rt_analysis"
    / "movie_rt_scores_detailed.csv"
)
DEFAULT_BOOK_TRAINING = (
    DEFAULT_ROOT / "data/Books Read and their effects - Play Export.csv"
)
DEFAULT_BOOK_HOLDOUT = (
    DEFAULT_ROOT / "data/Books Read and their effects - new_books_to_rate 2026.csv"
)
DEFAULT_MANUAL_MOVIE_COMPLETIONS = (
    DEFAULT_ROOT / "data/rating_export_manual_movie_completions.json"
)
DEFAULT_BOOK_TITLE_OVERRIDES = (
    DEFAULT_ROOT / "data/rating_export_book_title_overrides.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "data/rating_runs"

MOVIE_ALIASES = {
    "charlie wilson s wall": "charlie wilson s war",
    "don t let devil know you re dead": "before devil knows you re dead",
    "f1 the": "f1",
    "mission impossible 4": "mission impossible 5",
    "ricky stalicky": "ricky stanley",
    "ricky stanicky": "ricky stanley",
    "spider man in to multiverse": "spider man across spider verse",
    "specter": "spectre",
    "the sheep detective": "sheep detectives",
    "sheep detective": "sheep detectives",
    "vengance": "vengeance",
}
MOVIE_PREFERRED_TITLES = {
    "father of bride": "Father of the Bride",
    "few good men": "A Few Good Men",
    "great outdoors": "The Great Outdoors",
    "kiki s delivery service": "Kiki’s Delivery Service",
    "ricky stanley": "Ricky Stanley",
    "roofman": "Roofman",
    "sheep detectives": "The Sheep Detectives",
    "spectre": "Spectre",
}
MOVIE_EXCLUSIONS = (
    "best picture showcase",
    "movie marathon",
)
GENERIC_MOVIE_TITLES = {
    "find",
    "search",
    "search for",
    "searching for",
    "started",
}
LEGACY_MOVIE_ENTRY_PATTERN = re.compile(
    r"^(?P<title>.+?)\s+(?P<rating>\d+)\s*;\s*(?P<review>.*)$"
)
LEGACY_MOVIE_ADJUDICATIONS: dict[str, tuple[str, str, str]] = {
    "An American Pickle": (
        "american pickle",
        "An American Pickle",
        "2020",
    ),
    "Heat (val Kilmer, Al pacino, 1995)": (
        "heat",
        "Heat",
        "1995",
    ),
    "The Boondock Saints": (
        "boondock saints",
        "The Boondock Saints",
        "1999",
    ),
    "Before the Devil knows you’re dead": (
        "before devil knows you re dead",
        "Before the Devil Knows You're Dead",
        "2007",
    ),
    "A Lover and a Gentleman": (
        "lover and gentleman",
        "An Officer and a Gentleman",
        "1982",
    ),
    "the league of extraordinary gentlemen": (
        "league of extraordinary gentlemen",
        "The League of Extraordinary Gentlemen",
        "2003",
    ),
}
BOOK_ALIASES: dict[str, str] = {}
BOOK_PREFIX_ALIASES: dict[str, str] = {}
BOOK_PREFERRED_TITLES: dict[str, str] = {}


@dataclass(frozen=True)
class MovieEvent:
    uid: str
    title: str
    normalized_title: str
    progress: str
    watched_on: date
    duration_hours: float
    where_seen: str
    source: str
    summary: str


@dataclass(frozen=True)
class LegacyMovieRating:
    source_index: int
    source_title: str
    normalized_title: str
    rating: str
    review: str
    matched_title: str = ""
    release_year: str = ""
    rt_url: str = ""


@dataclass(frozen=True)
class BookRecord:
    title: str
    normalized_title: str
    author: str
    finished_on: date | None
    completion_source: str
    bookshelf: str = ""
    filename: str = ""
    enjoyment_1: str = ""
    usefulness_1: str = ""
    enjoyment_2: str = ""
    usefulness_2: str = ""
    long_term_effects: str = ""
    play_finished: bool | None = None


def _plain_tokens(value: str) -> list[str]:
    value = html.unescape(value).lower().replace("’", "'").replace("&", " and ")
    value = value.replace("'", " ")
    return re.findall(r"[a-z0-9]+", value)


def normalize_movie_title(title: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", title)
    value = re.sub(
        r"^(?:started|finished|finish|middle|continued)\s+(?:movie|move)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:movies?|amc movie)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+at\s+amc.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:4k|movie)$", "", value, flags=re.IGNORECASE)
    tokens = [
        token for token in _plain_tokens(value) if token not in {"the", "a", "an"}
    ]
    normalized = " ".join(tokens)
    return MOVIE_ALIASES.get(normalized, normalized)


def normalize_movie_identity_title(title: str) -> str:
    """Normalize spelling/punctuation while retaining identity-bearing articles."""
    value = re.sub(r"\([^)]*\)", " ", title)
    value = re.sub(
        r"^(?:started|finished|finish|middle|continued)\s+(?:movie|move)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:movies?|amc movie)\s*:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+at\s+amc.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+(?:4k|movie)$", "", value, flags=re.IGNORECASE)
    return " ".join(_plain_tokens(value))


def _clean_release_year(value: str) -> str:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    return match.group(1) if match else ""


def _movie_identity_key(title: str, release_year: str) -> str:
    normalized = normalize_movie_identity_title(title)
    return f"{normalized}|{release_year}" if normalized and release_year else ""


def parse_legacy_movie_entries(raw_text: str) -> list[LegacyMovieRating]:
    """Parse the original ``title rating; review`` notes, including continuations.

    This preserves the behavior of the legacy ``parse_movie_entries`` function
    without importing its plotting, scraping, and model-training dependencies.
    """
    entries: list[LegacyMovieRating] = []
    current_title: str | None = None
    current_rating: str | None = None
    current_review: list[str] = []

    def flush() -> None:
        if current_title is None or current_rating is None:
            return
        entries.append(
            LegacyMovieRating(
                source_index=len(entries),
                source_title=current_title,
                normalized_title=normalize_movie_title(current_title),
                rating=current_rating,
                review="\n".join(line for line in current_review if line).strip(),
            )
        )

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LEGACY_MOVIE_ENTRY_PATTERN.match(line)
        if match:
            flush()
            current_title = match.group("title").strip()
            current_rating = match.group("rating").strip()
            current_review = [match.group("review").strip()]
        elif current_title is not None:
            current_review.append(line)
    flush()
    return entries


def load_legacy_movie_ratings(
    notes_path: Path, details_path: Path
) -> list[LegacyMovieRating]:
    parsed = parse_legacy_movie_entries(notes_path.read_text(encoding="utf-8"))
    with details_path.open(newline="", encoding="utf-8-sig") as handle:
        details = {row["movie_title"].strip(): row for row in csv.DictReader(handle)}

    enriched: list[LegacyMovieRating] = []
    for entry in parsed:
        detail = details.get(entry.source_title, {})
        detailed_rating = detail.get("my_rating", "").strip()
        if detailed_rating and float(detailed_rating) != float(entry.rating):
            raise ValueError(
                f"Legacy rating mismatch for {entry.source_title!r}: "
                f"notes={entry.rating}, details={detailed_rating}"
            )
        enriched.append(
            LegacyMovieRating(
                source_index=entry.source_index,
                source_title=entry.source_title,
                normalized_title=entry.normalized_title,
                rating=entry.rating,
                review=entry.review,
                matched_title=detail.get("matched_title", "").strip(),
                release_year=_clean_release_year(detail.get("rt_release_year", "")),
                rt_url=detail.get("rt_url", "").strip(),
            )
        )
    return enriched


def load_book_title_overrides(
    path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if not path.exists():
        return {}, {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Book title overrides must be a JSON object: {path}")

    sections: list[dict[str, str]] = []
    for section_name in ("aliases", "prefix_aliases", "preferred_titles"):
        section = payload.get(section_name, {})
        if not isinstance(section, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in section.items()
        ):
            raise ValueError(
                f"Book title override `{section_name}` must map strings to strings: "
                f"{path}"
            )
        sections.append(
            {
                key.strip(): value.strip()
                for key, value in section.items()
                if key.strip() and value.strip()
            }
        )
    return sections[0], sections[1], sections[2]


def normalize_book_title(
    title: str,
    *,
    aliases: dict[str, str] | None = None,
    prefix_aliases: dict[str, str] | None = None,
) -> str:
    value = re.sub(r"\.(?:pdf|epub|html|txt)$", "", title, flags=re.IGNORECASE)
    value = re.sub(r"\([^)]*\)", "", value)
    tokens = _plain_tokens(value)
    tokens = ["volume" if token == "vol" else token for token in tokens]
    if tokens and tokens[0] in {"the", "a", "an"}:
        tokens = tokens[1:]
    normalized = " ".join(tokens)
    for key, canonical in (BOOK_ALIASES if aliases is None else aliases).items():
        if normalized == key:
            return canonical
    for key, canonical in (
        BOOK_PREFIX_ALIASES if prefix_aliases is None else prefix_aliases
    ).items():
        if normalized.startswith(key):
            return canonical
    return normalized


def _preferred_movie_title(normalized: str, variants: Iterable[str]) -> str:
    if normalized in MOVIE_PREFERRED_TITLES:
        return MOVIE_PREFERRED_TITLES[normalized]
    choices = sorted(
        {value.strip() for value in variants if value.strip()},
        key=lambda value: (value.islower(), len(value), value.lower(), value),
    )
    return choices[0] if choices else normalized.title()


def _preferred_book_title(normalized: str, variants: Iterable[str]) -> str:
    if normalized in BOOK_PREFERRED_TITLES:
        return BOOK_PREFERRED_TITLES[normalized]
    choices = sorted(
        {value.strip() for value in variants if value.strip()},
        key=lambda value: (value.islower(), len(value), value.lower(), value),
    )
    return choices[0] if choices else normalized.title()


def _unfold_ics(text: str) -> list[str]:
    result: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith((" ", "\t")) and result:
            result[-1] += raw_line[1:]
        else:
            result.append(raw_line)
    return result


def _read_ics(path: Path) -> list[dict[str, list[str]]]:
    events: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    for line in _unfold_ics(path.read_text(encoding="utf-8", errors="replace")):
        if line == "BEGIN:VEVENT":
            current = {}
        elif line == "END:VEVENT":
            if current is not None:
                events.append(current)
            current = None
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current.setdefault(key, []).append(value)
    return events


def _property(event: dict[str, list[str]], name: str) -> tuple[str, str] | None:
    for key, values in event.items():
        if key.split(";", 1)[0] == name:
            return key, values[0]
    return None


def _text_property(event: dict[str, list[str]], name: str) -> str:
    item = _property(event, name)
    if item is None:
        return ""
    value = item[1]
    value = value.replace("\\N", "\n").replace("\\n", "\n")
    value = value.replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
    return html.unescape(value).strip()


def _parse_ics_datetime(key: str, value: str) -> datetime:
    timezone_name = ""
    for part in key.split(";")[1:]:
        if part.startswith("TZID="):
            timezone_name = part.split("=", 1)[1]
    if "T" not in value:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=LOCAL_TZ)
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    event_tz = ZoneInfo(timezone_name) if timezone_name else LOCAL_TZ
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=event_tz)


def _infer_movie(
    event: dict[str, list[str]], summary: str
) -> tuple[str, str, str] | None:
    normalized_summary = normalize_movie_title(summary)
    if any(marker in normalized_summary for marker in MOVIE_EXCLUSIONS):
        return None

    creator = _text_property(event, "X-APPLE-CREATOR-IDENTITY").lower()
    description = _text_property(event, "DESCRIPTION").lower()
    is_amc = creator == "com.vmbc.amc" or (
        "the event start time is when trailers" in description
        and "amc" in _text_property(event, "LOCATION").lower()
    )
    if is_amc:
        title = re.sub(r"\s+at\s+amc.*$", "", summary, flags=re.IGNORECASE).strip()
        return title, "finished", "theater"

    patterns: tuple[tuple[str, str], ...] = (
        (r"^(?:finished|finish)\s+movie\s*:\s*(.+)$", "finished"),
        (r"^started\s+movie\s*:\s*(.+)$", "started"),
        (r"^(?:continued)\s+(?:movie|move)\s*:\s*(.+)$", "continued"),
        (r"^middle\s+movie\s*:\s*(.+)$", "continued"),
        (r"^(?:movies?|amc movie)\s*:\s*(.+)$", "default"),
        (r"^(.+?)\s+at\s+amc\b.*$", "finished"),
        (r"^([^/,]+?)\s+movie$", "default"),
    )
    for pattern, progress in patterns:
        match = re.match(pattern, summary.strip(), flags=re.IGNORECASE)
        if match:
            where_seen = "theater" if "amc" in summary.lower() else "home"
            return match.group(1).strip(), progress, where_seen
    if " at " in summary.lower() and any(
        token in f"{summary} {_text_property(event, 'LOCATION')}".lower()
        for token in ("century", "metreon", "kabuki", "regal", "alamo", "cinemark")
    ):
        return summary.rsplit(" at ", 1)[0].strip(), "finished", "theater"
    return None


def read_movie_events(ics_path: Path, *, as_of: date) -> list[MovieEvent]:
    as_of_end = datetime.combine(as_of, time.max, tzinfo=LOCAL_TZ)
    parsed: dict[tuple[str, datetime, str], MovieEvent] = {}
    for raw in _read_ics(ics_path):
        if _text_property(raw, "STATUS").upper() == "CANCELLED":
            continue
        summary = _text_property(raw, "SUMMARY")
        start_property = _property(raw, "DTSTART")
        if not summary or start_property is None:
            continue
        start = _parse_ics_datetime(*start_property).astimezone(LOCAL_TZ)
        if start > as_of_end:
            continue
        inferred = _infer_movie(raw, summary)
        if inferred is None:
            continue
        title, progress, where_seen = inferred
        normalized = normalize_movie_title(title)
        if not normalized or normalized in GENERIC_MOVIE_TITLES:
            continue
        end_property = _property(raw, "DTEND")
        end = (
            _parse_ics_datetime(*end_property).astimezone(LOCAL_TZ)
            if end_property
            else start + timedelta(hours=1)
        )
        duration = max(0.0, (end - start).total_seconds() / 3600)
        source = "calendar_explicit"
        if progress == "default" and duration >= 1.25:
            progress = "finished"
            source = "calendar_long_event"
        elif progress == "finished" and where_seen == "theater":
            source = "calendar_confirmed_amc"
        uid = _text_property(raw, "UID")
        event = MovieEvent(
            uid=uid,
            title=_preferred_movie_title(normalized, [title]),
            normalized_title=normalized,
            progress=progress,
            watched_on=start.date(),
            duration_hours=duration,
            where_seen=where_seen,
            source=source,
            summary=summary,
        )
        parsed[(uid, start, summary)] = event

    grouped: dict[tuple[str, date], list[MovieEvent]] = defaultdict(list)
    for event in parsed.values():
        grouped[(event.normalized_title, event.watched_on)].append(event)
    completed_events: list[MovieEvent] = []
    for same_day_events in grouped.values():
        total_hours = sum(event.duration_hours for event in same_day_events)
        make_finished = total_hours >= 1.25 and any(
            event.progress == "default" for event in same_day_events
        )
        for event in same_day_events:
            if make_finished and event.progress == "default":
                event = MovieEvent(
                    **{
                        **event.__dict__,
                        "progress": "finished",
                        "source": "calendar_multi_session",
                    }
                )
            completed_events.append(event)
    return sorted(completed_events, key=lambda item: (item.watched_on, item.uid))


def load_manual_movie_completions(
    path: Path,
) -> list[tuple[str, date, str]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manual movie completions must be a JSON list: {path}")

    completions: list[tuple[str, date, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"Manual movie completion {index} must be a JSON object: {path}"
            )
        title = str(item.get("title", "")).strip()
        watched_on = _parse_date(str(item.get("date", "")))
        reason = str(item.get("reason", "manual local override")).strip()
        if not title or watched_on is None:
            raise ValueError(
                f"Manual movie completion {index} needs title and ISO date: {path}"
            )
        completions.append((title, watched_on, reason))
    return completions


def _manual_movie_events(
    as_of: date,
    completions: list[tuple[str, date, str]],
) -> list[MovieEvent]:
    return [
        MovieEvent(
            uid=f"manual-{normalized}-{watched_on.isoformat()}",
            title=title,
            normalized_title=normalized,
            progress="finished",
            watched_on=watched_on,
            duration_hours=0,
            where_seen="home",
            source="manual_user_confirmation",
            summary=reason,
        )
        for title, watched_on, reason in completions
        if watched_on <= as_of
        for normalized in [normalize_movie_title(title)]
    ]


def load_old_movie_labels(
    path: Path,
) -> tuple[dict[str, dict[str, str | list[str]]], list[MovieEvent]]:
    ratings: dict[str, dict[str, str | list[str]]] = {}
    events: list[MovieEvent] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rating_columns = [
            column
            for column in reader.fieldnames or []
            if column == "personal_rating"
            or (
                column.endswith("_rating")
                and not column.startswith(("amazon_", "goodreads_", "imdb_", "rt_"))
            )
        ]
        if len(rating_columns) != 1:
            raise ValueError(
                "Expected exactly one personal rating column ending in `_rating`; "
                f"found {rating_columns} in {path}"
            )
        rating_column = rating_columns[0]
        for row_number, row in enumerate(reader, start=2):
            title = row.get("movie_title", "").strip()
            if not title:
                continue
            normalized = normalize_movie_title(row.get("normalized_title", "") or title)
            raw_date = row.get("date", "")
            try:
                watched_on = date.fromisoformat(raw_date)
            except ValueError:
                continue
            events.append(
                MovieEvent(
                    uid=f"prior-label-{row_number}",
                    title=_preferred_movie_title(normalized, [title]),
                    normalized_title=normalized,
                    progress="finished",
                    watched_on=watched_on,
                    duration_hours=0,
                    where_seen=row.get("where_seen", "") or "unknown",
                    source="prior_manual_label",
                    summary=title,
                )
            )
            rating = row.get(rating_column, "").strip()
            identity_title = (
                row.get("rt_matched_title", "").strip()
                or row.get("imdb_matched_title", "").strip()
            )
            identity_year = _clean_release_year(
                row.get("rt_release_year", "") or row.get("imdb_release_year", "")
            )
            identity_source = (
                "calendar_rt_match"
                if row.get("rt_matched_title", "").strip()
                else ("calendar_imdb_match" if identity_title else "")
            )
            existing = ratings.setdefault(
                normalized,
                {
                    "rating": "",
                    "dates": [],
                    "identity_title": identity_title,
                    "identity_year": identity_year,
                    "identity_source": identity_source,
                    "identity_conflict": "",
                },
            )
            existing_identity = (
                str(existing.get("identity_title", "")),
                str(existing.get("identity_year", "")),
            )
            incoming_identity = (identity_title, identity_year)
            if (
                all(existing_identity)
                and all(incoming_identity)
                and existing_identity != incoming_identity
            ):
                existing["identity_conflict"] = (
                    f"{existing_identity[0]} ({existing_identity[1]}) | "
                    f"{identity_title} ({identity_year})"
                )
                existing["identity_title"] = ""
                existing["identity_year"] = ""
                existing["identity_source"] = ""
            elif identity_title and not existing.get("identity_title"):
                existing["identity_title"] = identity_title
                existing["identity_year"] = identity_year
                existing["identity_source"] = identity_source
            dates = existing["dates"]
            assert isinstance(dates, list)
            dates.append(raw_date)
            normalized_rating = rating.removesuffix(".0")
            if re.fullmatch(r"(?:10|[1-9])", normalized_rating):
                if existing["rating"] not in {"", "na", normalized_rating}:
                    raise ValueError(
                        f"Conflicting old ratings for normalized title {normalized!r}"
                    )
                existing["rating"] = normalized_rating
            elif rating.lower() == "na" and not existing["rating"]:
                existing["rating"] = "na"
    return ratings, events


def build_movie_exports(
    events: list[MovieEvent],
    *,
    old_ratings: dict[str, dict[str, str | list[str]]],
    new_since: date,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[MovieEvent]] = defaultdict(list)
    for event in events:
        grouped[event.normalized_title].append(event)

    all_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for normalized, title_events in grouped.items():
        finished = [event for event in title_events if event.progress == "finished"]
        variants = sorted({event.title for event in title_events})
        if not finished:
            latest_event_date = max(event.watched_on for event in title_events)
            if latest_event_date < new_since:
                continue
            qc_rows.append(
                {
                    "normalized_title": normalized,
                    "title": _preferred_movie_title(normalized, variants),
                    "latest_event_date": latest_event_date,
                    "progress_states": "|".join(
                        sorted({event.progress for event in title_events})
                    ),
                    "calendar_summaries": " || ".join(
                        sorted({event.summary for event in title_events})
                    ),
                    "issue": "no_finished_evidence",
                }
            )
            continue
        watch_dates = sorted({event.watched_on for event in finished})
        recent_dates = [value for value in watch_dates if value >= new_since]
        rating_info = old_ratings.get(normalized, {})
        rating = str(rating_info.get("rating", ""))
        rating_dates = rating_info.get("dates", [])
        identity_title = str(rating_info.get("identity_title", ""))
        identity_year = str(rating_info.get("identity_year", ""))
        all_rows.append(
            {
                "movie_title": _preferred_movie_title(normalized, variants),
                "normalized_title": normalized,
                "movie_identity_key": _movie_identity_key(
                    identity_title, identity_year
                ),
                "identity_title": identity_title,
                "identity_year": identity_year,
                "identity_source": str(rating_info.get("identity_source", "")),
                "identity_conflict": str(rating_info.get("identity_conflict", "")),
                "first_watch_date": watch_dates[0],
                "last_watch_date": watch_dates[-1],
                "watch_count": len(watch_dates),
                "watch_dates": "|".join(value.isoformat() for value in watch_dates),
                "new_watch_dates": "|".join(
                    value.isoformat() for value in recent_dates
                ),
                "title_variants": " | ".join(variants),
                "where_seen": "|".join(
                    sorted({event.where_seen for event in finished})
                ),
                "completion_sources": "|".join(
                    sorted({event.source for event in finished})
                ),
                "existing_rating_1_10": rating,
                "existing_rating_dates": "|".join(str(value) for value in rating_dates),
                "existing_rating_source": ("calendar_manual_csv" if rating else ""),
                "existing_review": "",
                "legacy_source_title": "",
                "legacy_match_status": "",
                "needs_rating": "true" if recent_dates and not rating else "false",
            }
        )
    all_rows.sort(
        key=lambda row: (str(row["first_watch_date"]), str(row["movie_title"]))
    )
    new_rows = [
        row
        for row in all_rows
        if row["new_watch_dates"] and row["needs_rating"] == "true"
    ]
    new_rows.sort(
        key=lambda row: (str(row["new_watch_dates"]), str(row["movie_title"]))
    )
    qc_rows.sort(key=lambda row: (str(row["latest_event_date"]), str(row["title"])))
    return all_rows, new_rows, qc_rows


def reconcile_legacy_movie_ratings(
    movie_rows: list[dict[str, object]],
    legacy_ratings: list[LegacyMovieRating],
    *,
    movie_qc_rows: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Attach legacy ratings only when movie identity evidence is safe."""
    rows_by_normalized = {str(row["normalized_title"]): row for row in movie_rows}
    unfinished_by_normalized = {
        str(row["normalized_title"]): row for row in movie_qc_rows
    }
    reconciliation_rows: list[dict[str, object]] = []

    for legacy in legacy_ratings:
        candidate = rows_by_normalized.get(legacy.normalized_title)
        identity_title = legacy.matched_title or legacy.source_title
        identity_key = _movie_identity_key(identity_title, legacy.release_year)
        status = ""
        reason = ""
        auto_joined = False

        if candidate is None:
            status = (
                "legacy_only_added"
                if identity_key
                else "legacy_only_added_missing_identity"
            )
            if legacy.normalized_title in unfinished_by_normalized:
                reason = (
                    "Legacy notes prove an older completed watch; a current "
                    "unfinished Calendar candidate with the same broad title remains QC."
                )
            elif identity_key:
                reason = (
                    "No finished Calendar/prior-label row; added from legacy notes."
                )
            else:
                reason = (
                    "No finished Calendar/prior-label row and no enriched title/year; "
                    "added from the explicit legacy rating with partial identity."
                )
            candidate = {
                "movie_title": legacy.matched_title or legacy.source_title,
                "normalized_title": legacy.normalized_title,
                "movie_identity_key": identity_key,
                "identity_title": identity_title,
                "identity_year": legacy.release_year,
                "identity_source": (
                    "legacy_rt_enrichment"
                    if legacy.matched_title
                    else "legacy_review_notes"
                ),
                "identity_conflict": "",
                "first_watch_date": "",
                "last_watch_date": "",
                "watch_count": 1,
                "watch_dates": "",
                "new_watch_dates": "",
                "title_variants": legacy.source_title,
                "where_seen": "unknown",
                "completion_sources": "legacy_review_notes",
                "existing_rating_1_10": legacy.rating,
                "existing_rating_dates": "",
                "existing_rating_source": "legacy_review_notes",
                "existing_review": legacy.review,
                "legacy_source_title": legacy.source_title,
                "legacy_match_status": status,
                "needs_rating": "false",
            }
            movie_rows.append(candidate)
            rows_by_normalized[legacy.normalized_title] = candidate
        else:
            candidate_identity_title = str(candidate.get("identity_title", ""))
            candidate_identity_year = str(candidate.get("identity_year", ""))
            candidate_identity_conflict = str(candidate.get("identity_conflict", ""))
            legacy_identity_normalized = normalize_movie_identity_title(identity_title)
            candidate_identity_normalized = normalize_movie_identity_title(
                candidate_identity_title
            )
            candidate_variants = {
                normalize_movie_identity_title(value)
                for value in str(candidate.get("title_variants", "")).split(" | ")
                if value
            }
            legacy_variants = {
                normalize_movie_identity_title(legacy.source_title),
                legacy_identity_normalized,
            }
            exact_title = bool(candidate_variants & legacy_variants)
            year_conflict = bool(
                candidate_identity_year
                and legacy.release_year
                and candidate_identity_year != legacy.release_year
            )
            title_conflict = bool(
                candidate_identity_normalized
                and legacy_identity_normalized
                and candidate_identity_normalized != legacy_identity_normalized
            )
            identity_exact = bool(
                candidate_identity_normalized
                and candidate_identity_normalized == legacy_identity_normalized
                and not year_conflict
            )
            manual_adjudication = LEGACY_MOVIE_ADJUDICATIONS.get(legacy.source_title)
            manually_adjudicated = manual_adjudication == (
                str(candidate["normalized_title"]),
                identity_title,
                legacy.release_year,
            )
            existing_rating = str(candidate.get("existing_rating_1_10", ""))

            if (
                manually_adjudicated
                and existing_rating
                and existing_rating != legacy.rating
            ):
                status = "quarantined_rating_conflict"
                reason = (
                    f"Current rating {existing_rating} conflicts with legacy "
                    f"rating {legacy.rating}."
                )
            elif manually_adjudicated:
                status = "auto_joined_manual_adjudication"
                reason = (
                    "User confirmed the legacy and Calendar titles are the same movie; "
                    "canonicalized to the verified legacy title/year."
                )
                auto_joined = True
            elif candidate_identity_conflict or year_conflict or title_conflict:
                status = "quarantined_identity_conflict"
                reason = candidate_identity_conflict or (
                    f"Current identity {candidate_identity_title or candidate['movie_title']} "
                    f"({candidate_identity_year or 'unknown year'}) conflicts with "
                    f"legacy identity {identity_title} "
                    f"({legacy.release_year or 'unknown year'})."
                )
            elif not (exact_title or identity_exact):
                status = "quarantined_ambiguous_title"
                reason = (
                    "Broad normalization matches, but cleaned identity-bearing titles "
                    "do not; no matching current title/year evidence."
                )
            elif existing_rating and existing_rating != legacy.rating:
                status = "quarantined_rating_conflict"
                reason = (
                    f"Current rating {existing_rating} conflicts with legacy "
                    f"rating {legacy.rating}."
                )
            else:
                status = (
                    "auto_joined_identity"
                    if identity_exact and not exact_title
                    else (
                        "auto_joined_exact_title"
                        if legacy.release_year
                        else "auto_joined_exact_title_no_year"
                    )
                )
                reason = (
                    "Matched verified title/year identity."
                    if identity_exact
                    else "Matched exact cleaned title; legacy year retained when available."
                )
                auto_joined = True

            if auto_joined:
                candidate["existing_rating_1_10"] = legacy.rating
                candidate["existing_rating_source"] = "legacy_review_notes"
                candidate["existing_review"] = legacy.review
                candidate["legacy_source_title"] = legacy.source_title
                candidate["legacy_match_status"] = status
                candidate["needs_rating"] = "false"
                if manually_adjudicated:
                    candidate["movie_title"] = identity_title
                    candidate["identity_title"] = identity_title
                    candidate["identity_year"] = legacy.release_year
                    candidate["identity_source"] = (
                        "legacy_rt_enrichment_manual_adjudication"
                    )
                    candidate["identity_conflict"] = ""
                    candidate["movie_identity_key"] = identity_key
                elif not candidate_identity_title:
                    candidate["identity_title"] = identity_title
                    candidate["identity_year"] = legacy.release_year
                    candidate["identity_source"] = (
                        "legacy_rt_enrichment"
                        if legacy.matched_title
                        else "legacy_review_notes"
                    )
                    candidate["movie_identity_key"] = identity_key

        reconciliation_rows.append(
            {
                "legacy_index": legacy.source_index,
                "legacy_source_title": legacy.source_title,
                "legacy_normalized_title": legacy.normalized_title,
                "legacy_rating_1_10": legacy.rating,
                "legacy_review": legacy.review,
                "legacy_matched_title": legacy.matched_title,
                "legacy_release_year": legacy.release_year,
                "legacy_rt_url": legacy.rt_url,
                "candidate_movie_title": candidate.get("movie_title", ""),
                "candidate_normalized_title": candidate.get("normalized_title", ""),
                "candidate_identity_title": candidate.get("identity_title", ""),
                "candidate_identity_year": candidate.get("identity_year", ""),
                "reconciliation_status": status,
                "auto_joined": str(auto_joined).lower(),
                "reason": reason,
            }
        )

    movie_rows.sort(
        key=lambda row: (
            str(row.get("first_watch_date", "")),
            str(row["movie_title"]),
        )
    )
    new_rows = [
        row
        for row in movie_rows
        if row.get("new_watch_dates") and row.get("needs_rating") == "true"
    ]
    new_rows.sort(
        key=lambda row: (
            str(row.get("new_watch_dates", "")),
            str(row["movie_title"]),
        )
    )
    return movie_rows, new_rows, reconciliation_rows


def build_movie_identity_conflicts_qc(
    movie_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Report duplicate external identities without changing movie rows."""
    rows_by_identity: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in movie_rows:
        identity_key = str(row.get("movie_identity_key", ""))
        if identity_key:
            rows_by_identity[identity_key].append(row)

    qc_rows: list[dict[str, object]] = []
    for identity_key, rows in rows_by_identity.items():
        if len(rows) < 2:
            continue
        ratings = {
            str(row.get("existing_rating_1_10", ""))
            for row in rows
            if row.get("existing_rating_1_10")
        }
        issue = (
            "duplicate_identity_rating_conflict"
            if len(ratings) > 1
            else "duplicate_identity_same_rating"
        )
        qc_rows.append(
            {
                "movie_identity_key": identity_key,
                "identity_title": str(rows[0].get("identity_title", "")),
                "identity_year": str(rows[0].get("identity_year", "")),
                "row_count": len(rows),
                "movie_titles": " | ".join(str(row["movie_title"]) for row in rows),
                "normalized_titles": " | ".join(
                    str(row["normalized_title"]) for row in rows
                ),
                "ratings": " | ".join(
                    f"{row['movie_title']}={row.get('existing_rating_1_10', '') or '(blank)'}"
                    for row in rows
                ),
                "watch_dates": " | ".join(
                    f"{row['movie_title']}={row.get('watch_dates', '') or '(unknown)'}"
                    for row in rows
                ),
                "identity_sources": " | ".join(
                    sorted(
                        {
                            str(row.get("identity_source", ""))
                            for row in rows
                            if row.get("identity_source")
                        }
                    )
                ),
                "issue": issue,
                "recommended_action": (
                    "Review whether the external identity is wrong or the title rows "
                    "should be merged; no automatic change was made."
                ),
            }
        )
    return sorted(qc_rows, key=lambda row: str(row["movie_identity_key"]))


def _parse_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def read_prior_books(training_path: Path, holdout_path: Path) -> list[BookRecord]:
    records: list[BookRecord] = []
    source_specs = (
        (training_path, "prior_training", "latest_modified"),
        (holdout_path, "prior_holdout", "date_finished"),
    )
    for path, source, date_column in source_specs:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                title = row.get("title", "").strip()
                if not title:
                    continue
                records.append(
                    BookRecord(
                        title=title,
                        normalized_title=normalize_book_title(title),
                        author=row.get("author", "").strip(),
                        finished_on=_parse_date(row.get(date_column, "")),
                        completion_source=source,
                        bookshelf=(
                            row.get("Bookshelf", "") or row.get("Bookshelf", "")
                        ).strip(),
                        filename=row.get("filename", "").strip(),
                        enjoyment_1=row.get("Enjoyment (/5)", "").strip(),
                        usefulness_1=row.get("Usefulness /5 to Me", "").strip(),
                        enjoyment_2=row.get("Enjoyment (/5) 2nd", "").strip(),
                        usefulness_2=(
                            row.get("Usefulness /5 to Me 2nd", "")
                            or (
                                row.get("Usefulness /5 to Me", "")
                                if row.get("Enjoyment (/5) 2nd", "")
                                else ""
                            )
                        ).strip(),
                        long_term_effects=row.get("Long Term Effects", "").strip(),
                    )
                )
    return records


def read_play_books(play_dir: Path) -> list[BookRecord]:
    records: list[BookRecord] = []
    for row in process_google_books_exports(str(play_dir)):
        finished = bool(row["finished"])
        records.append(
            BookRecord(
                title=row["title"],
                normalized_title=normalize_book_title(row["title"]),
                author=re.sub(r"^by\s+", "", row["author"], flags=re.IGNORECASE),
                finished_on=_parse_date(row["latest_modified"]) if finished else None,
                completion_source="play_finished" if finished else "play_unfinished",
                bookshelf=row["bookshelf"],
                filename=row["filename"],
                play_finished=finished,
            )
        )
    return records


def read_calendar_finished_books(
    ics_path: Path, *, start_on: date, as_of: date
) -> list[BookRecord]:
    result: list[BookRecord] = []
    as_of_end = datetime.combine(as_of, time.max, tzinfo=LOCAL_TZ)
    for event in _read_ics(ics_path):
        if _text_property(event, "STATUS").upper() == "CANCELLED":
            continue
        start_property = _property(event, "DTSTART")
        if start_property is None:
            continue
        started = _parse_ics_datetime(*start_property).astimezone(LOCAL_TZ)
        if not (start_on <= started.date() <= as_of_end.date()):
            continue
        summary = _text_property(event, "SUMMARY")
        match = re.search(
            r"(?:finished|end of)\s+(?:audio\s*)?book\s*:\s*(.+?)$",
            summary,
            flags=re.IGNORECASE,
        )
        if match is None:
            match = re.match(r"book\s+finished\s*:\s*(.+)$", summary, re.IGNORECASE)
        if match is None:
            continue
        title = re.sub(
            r"\s*\(day\s*\d+/\d+\)\s*$", "", match.group(1).strip(), flags=re.I
        )
        normalized = normalize_book_title(title)
        result.append(
            BookRecord(
                title=_preferred_book_title(normalized, [title]),
                normalized_title=normalized,
                author="",
                finished_on=started.date(),
                completion_source="calendar_finished",
            )
        )
    return sorted(result, key=lambda row: (row.finished_on or date.min, row.title))


def _book_match_score(left: str, right: str) -> float:
    if left == right:
        return 100
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0
    left_numbers = {token for token in left_tokens if token.isdigit()}
    right_numbers = {token for token in right_tokens if token.isdigit()}
    volume_context = {"volume", "vol", "part"} & set(left_tokens + right_tokens)
    if (
        volume_context
        and left_numbers
        and right_numbers
        and left_numbers != right_numbers
    ):
        return 0
    overlap = len(set(left_tokens) & set(right_tokens))
    smaller = min(len(set(left_tokens)), len(set(right_tokens)))
    larger = max(len(set(left_tokens)), len(set(right_tokens)))
    if smaller >= 2 and overlap / smaller >= 0.8 and smaller / larger >= 0.45:
        return 70 + 20 * (overlap / larger)
    return 0


def _best_book_match(
    normalized: str, candidates: list[BookRecord]
) -> BookRecord | None:
    scored = [
        (_book_match_score(normalized, candidate.normalized_title), candidate)
        for candidate in candidates
    ]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return None
    return max(
        scored,
        key=lambda item: (
            item[0],
            bool(item[1].play_finished),
            item[1].finished_on or date.min,
        ),
    )[1]


def build_book_exports(
    *,
    prior_records: list[BookRecord],
    play_records: list[BookRecord],
    calendar_finished: list[BookRecord],
    new_since: date,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    instances: list[dict[str, object]] = []
    seen_prior: set[tuple[str, date | None]] = set()
    for record in prior_records:
        key = (record.normalized_title, record.finished_on)
        if key in seen_prior:
            continue
        seen_prior.add(key)
        instances.append(
            {
                "record": record,
                "newly_discovered": False,
                "play": _best_book_match(record.normalized_title, play_records),
            }
        )

    for calendar_record in calendar_finished:
        duplicate = next(
            (
                item
                for item in instances
                if item["record"].normalized_title == calendar_record.normalized_title
                and item["record"].finished_on is not None
                and calendar_record.finished_on is not None
                and abs((item["record"].finished_on - calendar_record.finished_on).days)
                <= 7
            ),
            None,
        )
        if duplicate is not None:
            continue
        instances.append(
            {
                "record": calendar_record,
                "newly_discovered": True,
                "play": _best_book_match(
                    calendar_record.normalized_title, play_records
                ),
            }
        )

    represented_titles = {item["record"].normalized_title for item in instances}
    for play_record in play_records:
        if not play_record.play_finished:
            continue
        if any(
            _book_match_score(play_record.normalized_title, normalized) > 0
            for normalized in represented_titles
        ):
            continue
        instances.append(
            {"record": play_record, "newly_discovered": True, "play": play_record}
        )
        represented_titles.add(play_record.normalized_title)

    earlier_dates: dict[str, list[date]] = defaultdict(list)
    for item in instances:
        record = item["record"]
        if not item["newly_discovered"] and record.finished_on:
            earlier_dates[record.normalized_title].append(record.finished_on)

    all_rows: list[dict[str, str]] = []
    qc_rows: list[dict[str, str]] = []
    for item in instances:
        record: BookRecord = item["record"]
        play: BookRecord | None = item["play"]
        title = _preferred_book_title(
            record.normalized_title,
            [record.title, play.title if play else ""],
        )
        reread = bool(
            item["newly_discovered"]
            and record.finished_on
            and any(
                prior_date <= record.finished_on - timedelta(days=30)
                for prior_date in earlier_dates[record.normalized_title]
            )
        )
        row = {
            "title": title,
            "normalized_title": record.normalized_title,
            "date_finished": (
                record.finished_on.isoformat() if record.finished_on else ""
            ),
            "author": (play.author if play and play.author else record.author),
            "bookshelf": (
                play.bookshelf if play and play.bookshelf else record.bookshelf
            ),
            "completion_source": record.completion_source,
            "is_reread": str(reread).lower(),
            "play_title": play.title if play else "",
            "play_finished": (
                str(play.play_finished).lower()
                if play and play.play_finished is not None
                else ""
            ),
            "play_filename": play.filename if play else record.filename,
            "enjoyment_1_5": record.enjoyment_1,
            "usefulness_1_5": record.usefulness_1,
            "enjoyment_2_5": record.enjoyment_2,
            "usefulness_2_5": record.usefulness_2,
            "long_term_effects": record.long_term_effects,
            "needs_rating": str(
                bool(
                    item["newly_discovered"]
                    and record.finished_on
                    and record.finished_on >= new_since
                )
            ).lower(),
        }
        all_rows.append(row)
        if (
            record.completion_source == "calendar_finished"
            and play is not None
            and play.play_finished is False
        ):
            qc_rows.append(
                {
                    "title": title,
                    "date_finished": row["date_finished"],
                    "play_title": play.title,
                    "issue": "calendar_finished_play_unfinished",
                    "resolution": "Calendar completion retained; Play used as metadata only",
                }
            )

    all_rows.sort(key=lambda row: (row["date_finished"], row["title"]))
    new_rows = [row for row in all_rows if row["needs_rating"] == "true"]
    qc_rows.sort(key=lambda row: (row["date_finished"], row["title"]))
    return all_rows, new_rows, qc_rows


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization_text(
    movie_rows: list[dict[str, object]], events: list[MovieEvent]
) -> str:
    event_groups: dict[str, list[MovieEvent]] = defaultdict(list)
    for event in events:
        event_groups[event.normalized_title].append(event)
    lines = [
        "Movie title normalization audit",
        "One block per combined title key; prior labels and Calendar variants are merged.",
        "",
    ]
    for row in sorted(movie_rows, key=lambda item: str(item["normalized_title"])):
        normalized = str(row["normalized_title"])
        group = event_groups[normalized]
        variants = sorted({event.title for event in group})
        if not variants:
            variants = [
                value.strip()
                for value in str(row.get("title_variants", "")).split(" | ")
                if value.strip()
            ]
        lines.extend(
            [
                f"normalized: {normalized}",
                f"canonical: {row['movie_title']}",
                f"identity: {row.get('movie_identity_key', '') or '(unknown)'}",
                "variants: " + " | ".join(variants),
                "summaries: " + " || ".join(sorted({event.summary for event in group})),
                f"watch_dates: {row['watch_dates']}",
                f"old_rating: {row['existing_rating_1_10'] or '(none)'}",
                f"rating_source: {row.get('existing_rating_source', '') or '(none)'}",
                f"legacy_source_title: {row.get('legacy_source_title', '') or '(none)'}",
                f"legacy_match_status: {row.get('legacy_match_status', '') or '(none)'}",
                "",
            ]
        )
    return "\n".join(lines)


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _comparison_report(
    *,
    movie_calendar_events: list[MovieEvent],
    movie_rows: list[dict[str, object]],
    movie_new_rows: list[dict[str, object]],
    legacy_reconciliation_rows: list[dict[str, object]],
    movie_identity_qc_rows: list[dict[str, object]],
    book_rows: list[dict[str, str]],
    book_new_rows: list[dict[str, str]],
    previous_movie_ledger: Path | None,
    previous_movie_queue: Path | None,
    previous_book_holdout: Path,
    new_since: date,
) -> str:
    old_ledger = _read_csv_rows(previous_movie_ledger)
    old_queue = _read_csv_rows(previous_movie_queue)
    old_books = _read_csv_rows(previous_book_holdout)
    old_finished_norms = {
        normalize_movie_title(row["movie_title"])
        for row in old_ledger
        if row.get("watch_status") == "finished"
    }
    calendar_finished_norms = {
        event.normalized_title
        for event in movie_calendar_events
        if event.progress == "finished"
    }
    overlap = old_finished_norms & calendar_finished_norms
    new_movie_norms = {str(row["normalized_title"]) for row in movie_new_rows}
    old_queue_norms = {
        normalize_movie_title(row.get("movie_title", "")) for row in old_queue
    }
    old_book_norms = {normalize_book_title(row.get("title", "")) for row in old_books}
    new_book_norms = {
        normalize_book_title(row.get("title", "")) for row in book_new_rows
    }
    reread_titles = [
        row["title"] for row in book_new_rows if row["is_reread"] == "true"
    ]
    recent_completed_events = {
        (event.normalized_title, event.watched_on)
        for event in movie_calendar_events
        if event.progress == "finished" and event.watched_on >= new_since
    }
    recent_completed_titles = {item[0] for item in recent_completed_events}
    legacy_statuses = Counter(
        str(row["reconciliation_status"]) for row in legacy_reconciliation_rows
    )
    return "\n".join(
        [
            "# Extraction comparison",
            "",
            "## Movies",
            "",
            f"- Previous ledger: {len(old_ledger)} watch-instance rows; "
            f"{len(old_finished_norms)} distinct finished normalized titles.",
            f"- New Calendar parser: {len(movie_calendar_events)} recognized events; "
            f"{len(calendar_finished_norms)} distinct finished normalized titles.",
            f"- Historical normalized-title overlap with the previous finished ledger: "
            f"{len(overlap)} of {len(old_finished_norms)}.",
            f"- Everything export: {len(movie_rows)} one-row-per-title records.",
            f"- Since {new_since}: {len(recent_completed_events)} completed title/date "
            f"pairs and {len(recent_completed_titles)} distinct completed titles.",
            f"- Only-new-to-rate export: {len(movie_new_rows)} titles after old "
            "title-level labels (including `na`) were reused.",
            f"- Legacy review notes: {len(legacy_reconciliation_rows)} ratings; "
            f"{sum(count for status, count in legacy_statuses.items() if status.startswith('auto_joined'))} "
            "auto-joined to existing rows; "
            f"{sum(count for status, count in legacy_statuses.items() if status.startswith('legacy_only_added'))} "
            "legacy-only rows added; "
            f"{sum(count for status, count in legacy_statuses.items() if status.startswith('quarantined'))} "
            "quarantined for review.",
            f"- Duplicate external-identity QC: {len(movie_identity_qc_rows)} "
            "identity groups; no rows or ratings were automatically merged.",
            f"- Previous queue had {len(old_queue)} rows; "
            f"{len(new_movie_norms & old_queue_norms)} current new titles also appeared there.",
            "",
            "The new counts are not expected to equal the old queue: the old queue included "
            "earlier gaps and title-only quarantines, while the new queue is date-bounded and "
            "reuses any prior label for the normalized title.",
            "",
            "## Books",
            "",
            f"- Previous holdout: {len(old_books)} completion rows and "
            f"{len(old_book_norms)} normalized titles.",
            f"- Everything export: {len(book_rows)} completion rows, including rereads.",
            f"- Only-new-to-rate export: {len(book_new_rows)} completion rows.",
            f"- New rows already represented by normalized title in the old holdout: "
            f"{len(new_book_norms & old_book_norms)} (expected rereads).",
            f"- Rows marked as rereads from all prior sources: "
            f"{len(reread_titles)} ({'; '.join(reread_titles)}).",
            "",
            "Calendar finished markers take precedence over Play's finished flag. "
            "Play-only finished titles are included in the everything export only when they "
            "do not match a prior or Calendar title.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> Path:
    calendar_dir = args.calendar_dir.resolve()
    play_dir = args.play_books_dir.resolve()
    manual_movie_completions_path = args.manual_movie_completions.resolve()
    book_title_overrides_path = args.book_title_overrides.resolve()
    source_paths = {
        "movie_calendar": calendar_dir / "Waste Time.ics",
        "book_calendar": calendar_dir / "Things.ics",
        "play_books_dir": play_dir,
        "old_movie_labels": args.old_movie_labels.resolve(),
        "legacy_movie_notes": args.legacy_movie_notes.resolve(),
        "legacy_movie_details": args.legacy_movie_details.resolve(),
        "old_book_training": args.old_book_training.resolve(),
        "old_book_holdout": args.old_book_holdout.resolve(),
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if manual_movie_completions_path.exists():
        source_paths["manual_movie_completions"] = manual_movie_completions_path
    if book_title_overrides_path.exists():
        source_paths["book_title_overrides"] = book_title_overrides_path

    run_id = args.run_id or datetime.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    calendar_movie_events = read_movie_events(
        source_paths["movie_calendar"], as_of=args.as_of
    )
    manual_movie_completions = load_manual_movie_completions(
        manual_movie_completions_path
    )
    manual_movie_events = _manual_movie_events(
        args.as_of,
        manual_movie_completions,
    )
    old_ratings, old_movie_events = load_old_movie_labels(
        source_paths["old_movie_labels"]
    )
    combined_movie_events = (
        calendar_movie_events + manual_movie_events + old_movie_events
    )
    movie_rows, movie_new_rows, movie_qc_rows = build_movie_exports(
        combined_movie_events,
        old_ratings=old_ratings,
        new_since=args.new_since,
    )
    legacy_movie_ratings = load_legacy_movie_ratings(
        source_paths["legacy_movie_notes"],
        source_paths["legacy_movie_details"],
    )
    movie_rows, movie_new_rows, legacy_movie_qc_rows = reconcile_legacy_movie_ratings(
        movie_rows,
        legacy_movie_ratings,
        movie_qc_rows=movie_qc_rows,
    )
    movie_identity_qc_rows = build_movie_identity_conflicts_qc(movie_rows)

    book_aliases, book_prefix_aliases, book_preferred_titles = (
        load_book_title_overrides(book_title_overrides_path)
    )
    BOOK_ALIASES.clear()
    BOOK_ALIASES.update(book_aliases)
    BOOK_PREFIX_ALIASES.clear()
    BOOK_PREFIX_ALIASES.update(book_prefix_aliases)
    BOOK_PREFERRED_TITLES.clear()
    BOOK_PREFERRED_TITLES.update(book_preferred_titles)
    prior_books = read_prior_books(
        source_paths["old_book_training"], source_paths["old_book_holdout"]
    )
    play_books = read_play_books(play_dir)
    calendar_books = read_calendar_finished_books(
        source_paths["book_calendar"],
        start_on=args.new_since - timedelta(days=1),
        as_of=args.as_of,
    )
    book_rows, book_new_rows, book_qc_rows = build_book_exports(
        prior_records=prior_books,
        play_records=play_books,
        calendar_finished=calendar_books,
        new_since=args.new_since,
    )

    movie_columns = [
        "movie_title",
        "normalized_title",
        "movie_identity_key",
        "identity_title",
        "identity_year",
        "identity_source",
        "identity_conflict",
        "first_watch_date",
        "last_watch_date",
        "watch_count",
        "watch_dates",
        "new_watch_dates",
        "title_variants",
        "where_seen",
        "completion_sources",
        "existing_rating_1_10",
        "existing_rating_dates",
        "existing_rating_source",
        "existing_review",
        "legacy_source_title",
        "legacy_match_status",
        "needs_rating",
    ]
    book_columns = [
        "title",
        "normalized_title",
        "date_finished",
        "author",
        "bookshelf",
        "completion_source",
        "is_reread",
        "play_title",
        "play_finished",
        "play_filename",
        "enjoyment_1_5",
        "usefulness_1_5",
        "enjoyment_2_5",
        "usefulness_2_5",
        "long_term_effects",
        "needs_rating",
    ]
    _write_csv(output_dir / "movies_all.csv", movie_rows, movie_columns)
    _write_csv(output_dir / "movies_new_to_rate.csv", movie_new_rows, movie_columns)
    _write_csv(
        output_dir / "movies_unfinished_qc.csv",
        movie_qc_rows,
        [
            "normalized_title",
            "title",
            "latest_event_date",
            "progress_states",
            "calendar_summaries",
            "issue",
        ],
    )
    _write_csv(
        output_dir / "movie_legacy_reconciliation_qc.csv",
        legacy_movie_qc_rows,
        [
            "legacy_index",
            "legacy_source_title",
            "legacy_normalized_title",
            "legacy_rating_1_10",
            "legacy_review",
            "legacy_matched_title",
            "legacy_release_year",
            "legacy_rt_url",
            "candidate_movie_title",
            "candidate_normalized_title",
            "candidate_identity_title",
            "candidate_identity_year",
            "reconciliation_status",
            "auto_joined",
            "reason",
        ],
    )
    _write_csv(
        output_dir / "movie_identity_conflicts_qc.csv",
        movie_identity_qc_rows,
        [
            "movie_identity_key",
            "identity_title",
            "identity_year",
            "row_count",
            "movie_titles",
            "normalized_titles",
            "ratings",
            "watch_dates",
            "identity_sources",
            "issue",
            "recommended_action",
        ],
    )
    _write_csv(output_dir / "books_all.csv", book_rows, book_columns)
    _write_csv(output_dir / "books_new_to_rate.csv", book_new_rows, book_columns)
    _write_csv(
        output_dir / "books_reconciliation_qc.csv",
        book_qc_rows,
        ["title", "date_finished", "play_title", "issue", "resolution"],
    )
    (output_dir / "movie_title_normalizations.txt").write_text(
        _normalization_text(movie_rows, combined_movie_events), encoding="utf-8"
    )
    (output_dir / "comparison_report.md").write_text(
        _comparison_report(
            movie_calendar_events=calendar_movie_events + manual_movie_events,
            movie_rows=movie_rows,
            movie_new_rows=movie_new_rows,
            legacy_reconciliation_rows=legacy_movie_qc_rows,
            movie_identity_qc_rows=movie_identity_qc_rows,
            book_rows=book_rows,
            book_new_rows=book_new_rows,
            previous_movie_ledger=args.previous_movie_ledger,
            previous_movie_queue=args.previous_movie_queue,
            previous_book_holdout=args.old_book_holdout,
            new_since=args.new_since,
        ),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(LOCAL_TZ).isoformat(timespec="seconds"),
        "as_of": args.as_of.isoformat(),
        "new_since_inclusive": args.new_since.isoformat(),
        "output_policy": "unique timestamp directory; creation fails on collision",
        "movie_rating_policy": (
            "one rating per reconciled movie; prior CSV and legacy-note labels reused; "
            "broad-title-only identity matches quarantined unless explicitly adjudicated"
        ),
        "book_completion_policy": (
            "Calendar finished is authoritative; Play finished is a superset source"
        ),
        "manual_movie_completions": [
            {"title": title, "date": watched_on.isoformat(), "reason": reason}
            for title, watched_on, reason in manual_movie_completions
            if watched_on <= args.as_of
        ],
        "counts": {
            "calendar_movie_events": len(calendar_movie_events),
            "movies_all": len(movie_rows),
            "movies_new_to_rate": len(movie_new_rows),
            "movies_unfinished_qc": len(movie_qc_rows),
            "legacy_movie_ratings": len(legacy_movie_ratings),
            "legacy_movie_auto_joined": sum(
                str(row["reconciliation_status"]).startswith("auto_joined")
                for row in legacy_movie_qc_rows
            ),
            "legacy_movie_only_added": sum(
                str(row["reconciliation_status"]).startswith("legacy_only_added")
                for row in legacy_movie_qc_rows
            ),
            "legacy_movie_quarantined": sum(
                str(row["reconciliation_status"]).startswith("quarantined")
                for row in legacy_movie_qc_rows
            ),
            "movie_identity_conflicts_qc": len(movie_identity_qc_rows),
            "play_books": len(play_books),
            "calendar_finished_books_in_window": len(calendar_books),
            "books_all": len(book_rows),
            "books_new_to_rate": len(book_new_rows),
            "books_reconciliation_qc": len(book_qc_rows),
        },
        "sources": {
            name: {
                "path": str(path),
                **({"sha256": _hash_file(path)} if path.is_file() else {}),
            }
            for name, path in source_paths.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def _date_argument(value: str) -> date:
    return date.fromisoformat(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-dir", type=Path, default=DEFAULT_CALENDAR_DIR)
    parser.add_argument("--play-books-dir", type=Path, default=DEFAULT_PLAY_DIR)
    parser.add_argument("--old-movie-labels", type=Path, default=DEFAULT_MOVIE_LABELS)
    parser.add_argument(
        "--legacy-movie-notes",
        type=Path,
        default=DEFAULT_LEGACY_MOVIE_NOTES,
    )
    parser.add_argument(
        "--legacy-movie-details",
        type=Path,
        default=DEFAULT_LEGACY_MOVIE_DETAILS,
    )
    parser.add_argument("--old-book-training", type=Path, default=DEFAULT_BOOK_TRAINING)
    parser.add_argument("--old-book-holdout", type=Path, default=DEFAULT_BOOK_HOLDOUT)
    parser.add_argument(
        "--manual-movie-completions",
        type=Path,
        default=DEFAULT_MANUAL_MOVIE_COMPLETIONS,
        help="Optional ignored local JSON file of confirmed movie completions",
    )
    parser.add_argument(
        "--book-title-overrides",
        type=Path,
        default=DEFAULT_BOOK_TITLE_OVERRIDES,
        help="Optional ignored local JSON file of book title aliases",
    )
    parser.add_argument(
        "--previous-movie-ledger",
        type=Path,
        help="Optional prior movie ledger used only in the comparison report",
    )
    parser.add_argument(
        "--previous-movie-queue",
        type=Path,
        help="Optional prior movie queue used only in the comparison report",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", help="Optional unique run directory name")
    parser.add_argument("--new-since", type=_date_argument, default=date(2026, 3, 12))
    parser.add_argument("--as-of", type=_date_argument, default=date(2026, 7, 22))
    return parser.parse_args(argv)


def main() -> None:
    output_dir = run(parse_args())
    print(output_dir)


if __name__ == "__main__":
    main()
