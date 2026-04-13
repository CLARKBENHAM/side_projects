from __future__ import annotations

import csv
import difflib
import gzip
import html
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
TITLE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
ENTRY_PATTERN = re.compile(r"^(?P<title>.+?)\s+(?P<rating>\d+)\s*;\s*(?P<notes>.*)$")
RT_SEARCH_SECTION_PATTERN = re.compile(
    r'<search-page-result skeleton="panel"'
    r' type="(?P<media_type>movie|tvSeries)"[^>]*>(?P<body>.*?)</search-page-result>',
    re.DOTALL,
)
RT_SEARCH_ROW_PATTERN = re.compile(
    r"<search-page-media-row(?P<attrs>.*?)>"
    r'.*?<a href="(?P<href>https://www\.rottentomatoes\.com/[^"]+)" class="unset"'
    r' data-qa="thumbnail-link" slot="thumbnail">'
    r'.*?<a href="https://www\.rottentomatoes\.com/[^"]+" class="unset" data-qa="info-name"'
    r' slot="title">\s*(?P<title>.*?)\s*</a>',
    re.DOTALL,
)
RT_JSON_SCRIPT_PATTERN = re.compile(
    r'<script[^>]+data-json="(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</script>',
    re.DOTALL,
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "of",
    "part",
    "the",
    "to",
}
ROMAN_NUMERALS = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}
TRACKED_ALIAS_MAP = {
    "before devil knows you re dead": "before devil knows you re dead",
    "charlie wilson s wall": "charlie wilson s war",
    "don t let devil know you re dead": "before devil knows you re dead",
    "fifth element": "5th element",
    "mad max": "max max",
    "inglorious bastards": "inglorious basterds",
    "trainspotting": "trainspotters",
    "oblivian": "oblivion",
    "shawshank redemtion": "shawshank redemption",
    "momento": "memento",
    "justice leage synder cut": "justice league synder cut",
    "tinker tayler solder spy": "tinker tailor soldier spy",
    "wolf s": "wolfs",
}
TITLE_CORRECTIONS = {
    "mission impossible 4": "mission impossible 5",
    "spider man in to multiverse": "Spider-Man: Across the Spider-Verse",
    "vengance": "Vengeance",
}
RT_DIRECT_MATCH_OVERRIDES = {
    "made": {
        "rt_url": "https://www.rottentomatoes.com/m/made",
        "release_year": 2001,
    },
    "golden eye": {
        "rt_url": "https://www.rottentomatoes.com/m/goldeneye",
        "release_year": 1995,
    },
    "goldfinger": {
        "rt_url": "https://www.rottentomatoes.com/m/goldfinger",
        "release_year": 1964,
    },
}
RT_EXACT_TITLE_OVERRIDES = {
    "the naked gun": "The Naked Gun 2025",
}
RT_QUERY_OVERRIDES = {
    "7 psychopaths": "Seven Psychopaths",
    "avatar 2": "Avatar: The Way of Water",
    "aquaman 2": "Aquaman and the Lost Kingdom",
    "argyle": "Argylle",
    "bad boys 3": "Bad Boys for Life",
    "bee keeper": "The Beekeeper",
    "dune": "Dune 2021",
    "dune 1": "Dune 2021",
    "alita": "Alita: Battle Angel",
    "batman vs superman": "Batman v Superman: Dawn of Justice",
    "beverly hills cop 2": "Beverly Hills Cop II",
    "charlie s angels": "Charlie's Angels",
    "charlie wilson s war": "Charlie Wilson's War",
    "dungeons and dragons": "Dungeons & Dragons: Honor Among Thieves",
    "ferris bueller s day off": "Ferris Bueller's Day Off",
    "fall guy": "The Fall Guy 2024",
    "first man": "First Man",
    "glass onion": "Glass Onion: A Knives Out Mystery",
    "golden eye": "GoldenEye",
    "goldfinger": "Goldfinger",
    "good shepard": "The Good Shepherd",
    "hitman": "Hit Man",
    "harry potter 3": "Harry Potter and the Prisoner of Azkaban",
    "harry potter 4": "Harry Potter and the Goblet of Fire",
    "harry potter 5": "Harry Potter and the Order of the Phoenix",
    "harry potter 6": "Harry Potter and the Half-Blood Prince",
    "indian jones": "Indiana Jones and the Dial of Destiny",
    "into spiderverse": "Spider-Man: Into the Spider-Verse",
    "john wick 3": "John Wick: Chapter 3 - Parabellum",
    "licorice pizza": "Licorice Pizza",
    "license to kill": "Licence to Kill",
    "made": "Made 2001",
    "madagascar 2": "Madagascar: Escape 2 Africa",
    "matrix 4": "The Matrix Resurrections",
    "mission impossible 5": "Mission: Impossible - Rogue Nation",
    "mission impossible 7": "Mission: Impossible - Dead Reckoning Part One",
    "naked gun 3": "Naked Gun 33 1/3: The Final Insult",
    "naked gun": "The Naked Gun: From the Files of Police Squad!",
    "naked gun 2": "The Naked Gun 2 1/2: The Smell of Fear",
    "normal life sumo": "A Normal Life: Chronicle of a Sumo Wrestler",
    "nottington hill": "Notting Hill",
    "number 24": "Number 24",
    "oceans 12": "Ocean's Twelve",
    "oceans 13": "Ocean's Thirteen",
    "pacific rim 2": "Pacific Rim: Uprising",
    "pirates of caribbean": "Pirates of the Caribbean: The Curse of the Black Pearl",
    "pirates of caribbean 5": "Pirates of the Caribbean: Dead Men Tell No Tales",
    "robocop": "RoboCop",
    "run away jury": "Runaway Jury",
    "school of rock": "School of Rock 2003",
    "september 5 with cast q and": "September 5",
    "september 5 with cast q a prerecorded": "September 5",
    "sense and sensibility": "Sense and Sensibility 2026",
    "sicario 2": "Sicario: Day of the Soldado",
    "sonic": "Sonic the Hedgehog",
    "team america": "Team America: World Police",
    "water boy": "The Waterboy",
    "vengance": "Vengeance 2022",
    "what happened to morgan s": "Did You Hear About the Morgans?",
    "wrath of man": "Wrath of Man",
    "wolfs": "Wolfs",
    "worlds end": "The World's End",
    "world is not enough": "The World Is Not Enough",
    "you ve got mail": "You've Got Mail",
    "dune 2": "Dune: Part Two",
    "harry potter and goblet of fire 20th anniversary": "Harry Potter and the Goblet of Fire",
}
IMDB_EXACT_TITLE_OVERRIDES = {
    "the naked gun": "The Naked Gun 2025",
}
IMDB_QUERY_OVERRIDES = {
    "7 psychopaths": "Seven Psychopaths",
    "avatar 2": "Avatar: The Way of Water",
    "aquaman 2": "Aquaman and the Lost Kingdom",
    "argyle": "Argylle",
    "bad boys 3": "Bad Boys for Life",
    "bee keeper": "The Beekeeper",
    "dune": "Dune: Part One",
    "dune 1": "Dune: Part One",
    "alita": "Alita: Battle Angel",
    "apollo 13 race for survival": "Apollo 13: Survival",
    "batman vs superman": "Batman v Superman: Dawn of Justice",
    "beverly hills cop 2": "Beverly Hills Cop II",
    "charlie s angels": "Charlie's Angels",
    "charlie wilson s war": "Charlie Wilson's War",
    "dungeons and dragons": "Dungeons & Dragons: Honor Among Thieves",
    "ferris bueller s day off": "Ferris Bueller's Day Off",
    "fall guy": "The Fall Guy 2024",
    "first man": "First Man",
    "glass onion": "Glass Onion: A Knives Out Mystery",
    "golden eye": "GoldenEye",
    "goldfinger": "Goldfinger",
    "good shepard": "The Good Shepherd",
    "hitman": "Hit Man",
    "harry potter 3": "Harry Potter and the Prisoner of Azkaban",
    "harry potter 4": "Harry Potter and the Goblet of Fire",
    "harry potter 5": "Harry Potter and the Order of the Phoenix",
    "harry potter 6": "Harry Potter and the Half-Blood Prince",
    "indian jones": "Indiana Jones and the Dial of Destiny",
    "into spiderverse": "Spider-Man: Into the Spider-Verse",
    "john wick 3": "John Wick: Chapter 3 - Parabellum",
    "k pop demon hunters": "KPop Demon Hunters",
    "license to kill": "Licence to Kill",
    "made": "Made 2001",
    "madagascar 2": "Madagascar: Escape 2 Africa",
    "matrix 4": "The Matrix Resurrections",
    "mission impossible 5": "Mission: Impossible - Rogue Nation",
    "mission impossible 7": "Mission: Impossible - Dead Reckoning Part One",
    "naked gun 3": "Naked Gun 33 1/3: The Final Insult",
    "naked gun": "The Naked Gun: From the Files of Police Squad!",
    "naked gun 2": "The Naked Gun 2 1/2: The Smell of Fear",
    "normal life sumo": "A Normal Life. Chronicle of a Sumo Wrestler",
    "nottington hill": "Notting Hill",
    "number 24": "Nr. 24",
    "oceans 12": "Ocean's Twelve",
    "oceans 13": "Ocean's Thirteen",
    "pacific rim 2": "Pacific Rim: Uprising",
    "pirates of caribbean": "Pirates of the Caribbean: The Curse of the Black Pearl",
    "pirates of caribbean 5": "Pirates of the Caribbean: Dead Men Tell No Tales",
    "run away jury": "Runaway Jury",
    "school of rock": "School of Rock",
    "september 5 with cast q and": "September 5",
    "september 5 with cast q a prerecorded": "September 5",
    "sense and sensibility": "Sense and Sensibility 2026",
    "sicario 2": "Sicario: Day of the Soldado",
    "sonic": "Sonic the Hedgehog",
    "team america": "Team America: World Police",
    "water boy": "The Waterboy",
    "vengance": "Vengeance 2022",
    "what happened to morgan s": "Did You Hear About the Morgans?",
    "wrath of man": "Wrath of Man",
    "wolfs": "Wolfs",
    "worlds end": "The World's End",
    "you ve got mail": "You've Got Mail",
    "dune 2": "Dune: Part Two",
    "harry potter and goblet of fire 20th anniversary": "Harry Potter and the Goblet of Fire",
}
GENERIC_MOVIE_PATTERNS = (
    "movie",
    "movies",
    "find movie",
    "movie search",
    "search for movie",
    "search movies",
    "find movies",
    "look for movie",
    "theater",
    "amc",
    "amc tickets",
)
NEGATIVE_TITLE_PATTERNS = (
    "find movie",
    "movie search",
    "search for movie",
    "look for movie",
)
DRINK_PATTERN = re.compile(
    r"\b(drink|drank|drunk|beer|wine|booze|cocktail|negroni|ipa|whiskey|"
    r"whisky|gin|vodka|tequila|scotch|bourbon)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CalendarEvent:
    calendar_name: str
    summary: str
    description: str
    location: str
    start_local: datetime
    end_local: datetime
    duration_hours: float


@dataclass(frozen=True)
class MovieEvent:
    title: str
    normalized_title: str
    progress: str
    where_seen: str
    start_local: datetime
    end_local: datetime
    duration_hours: float
    calendar_name: str
    summary: str
    description: str
    location: str


@dataclass(frozen=True)
class CandidateMovie:
    title: str
    normalized_title: str
    watch_status: str
    completion_basis: str
    completion_confidence: str
    event_count: int
    total_logged_hours: float
    date_local: datetime
    where_seen: str
    saw_in_home: bool
    saw_in_theater: bool
    drink_before_movie: bool
    calendar_name: str
    summary: str
    description: str
    location: str


@dataclass(frozen=True)
class SearchCandidate:
    media_type: str
    title: str
    url: str
    release_year: int | None
    cast: tuple[str, ...]


@dataclass(frozen=True)
class RtMatch:
    matched_title: str
    rt_url: str
    critic_score: int | None
    audience_score: int | None
    release_year: int | None
    match_confidence: float
    query: str


@dataclass(frozen=True)
class ImdbMatch:
    matched_title: str
    imdb_id: str
    imdb_url: str
    imdb_score: float | None
    imdb_rating_count: int | None
    release_year: int | None
    match_confidence: float
    query: str


def unescape_ics_text(value: str) -> str:
    value = value.replace("\\N", "\n").replace("\\n", "\n")
    value = value.replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
    return html.unescape(value)


def unfold_ics_lines(text: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        if raw_line.startswith((" ", "\t")):
            current += raw_line[1:]
        else:
            if current:
                lines.append(current)
            current = raw_line
    if current:
        lines.append(current)
    return lines


def parse_ics_events(path: Path) -> list[dict[str, list[str]]]:
    events: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    for line in unfold_ics_lines(path.read_text()):
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT":
            if current is not None:
                events.append(current)
            current = None
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current.setdefault(key, []).append(value)
    return events


def get_property(event: dict[str, list[str]], name: str) -> tuple[str, str] | None:
    for key, values in event.items():
        if key.split(";", 1)[0] == name:
            return key, values[0]
    return None


def parse_datetime_property(key: str, value: str) -> datetime:
    tz_name: str | None = None
    for part in key.split(";")[1:]:
        if part.startswith("TZID="):
            tz_name = part.split("=", 1)[1]

    if "T" not in value:
        dt = datetime.strptime(value, "%Y%m%d")
        tz = ZoneInfo(tz_name) if tz_name else LOCAL_TZ
        return dt.replace(tzinfo=tz)

    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

    tz = ZoneInfo(tz_name) if tz_name else LOCAL_TZ
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=tz)


def parse_calendar_directory(
    calendar_dir: Path, *, as_of_local: datetime
) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    as_of_utc = as_of_local.astimezone(timezone.utc)
    for path in sorted(calendar_dir.glob("*.ics")):
        if "Personal Dates" in path.name or "appointment_schedule" in path.name:
            continue
        calendar_name = path.stem
        for raw_event in parse_ics_events(path):
            status_prop = get_property(raw_event, "STATUS")
            if status_prop and status_prop[1].upper() == "CANCELLED":
                continue
            summary_prop = get_property(raw_event, "SUMMARY")
            start_prop = get_property(raw_event, "DTSTART")
            if summary_prop is None or start_prop is None:
                continue
            end_prop = get_property(raw_event, "DTEND")
            start = parse_datetime_property(*start_prop)
            end = (
                parse_datetime_property(*end_prop)
                if end_prop is not None
                else start + timedelta(hours=1)
            )
            start_utc = (
                start.astimezone(timezone.utc)
                if start.tzinfo is not None
                else start.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)
            )
            end_utc = (
                end.astimezone(timezone.utc)
                if end.tzinfo is not None
                else end.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)
            )
            if start_utc > as_of_utc:
                continue
            duration_hours = max(0.0, (end_utc - start_utc).total_seconds() / 3600.0)
            description_prop = get_property(raw_event, "DESCRIPTION")
            location_prop = get_property(raw_event, "LOCATION")
            events.append(
                CalendarEvent(
                    calendar_name=calendar_name,
                    summary=unescape_ics_text(summary_prop[1]).strip(),
                    description=(
                        unescape_ics_text(description_prop[1]).strip()
                        if description_prop is not None
                        else ""
                    ),
                    location=(
                        unescape_ics_text(location_prop[1]).strip()
                        if location_prop is not None
                        else ""
                    ),
                    start_local=start_utc.astimezone(LOCAL_TZ),
                    end_local=end_utc.astimezone(LOCAL_TZ),
                    duration_hours=duration_hours,
                )
            )
    return sorted(events, key=lambda event: event.start_local)


def normalize_tracking_title(title: str) -> str:
    value = title.lower().replace("’", "'").replace("&", " and ")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"^started\s+movie:\s*", "", value)
    value = re.sub(r"^middle\s+movie:\s*", "", value)
    value = re.sub(r"^finished\s+movie:\s*", "", value)
    value = re.sub(r"^finish\s+movie:\s*", "", value)
    value = re.sub(r"^movie:\s*", "", value)
    value = re.sub(r"^movies:\s*", "", value)
    value = re.sub(r"^amc\s+movie:\s*", "", value)
    value = re.sub(r"\s+at\s+amc.*$", "", value)
    value = re.sub(r"\s+movie$", "", value)
    tokens: list[str] = []
    for token in TITLE_TOKEN_PATTERN.findall(value):
        token = ROMAN_NUMERALS.get(token, token)
        if token in {"the", "a", "an"}:
            continue
        tokens.append(token)
    normalized = " ".join(tokens).strip()
    return TRACKED_ALIAS_MAP.get(normalized, normalized)


def parse_notes_titles(notes_path: Path) -> set[str]:
    titles: set[str] = set()
    for raw_line in notes_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = ENTRY_PATTERN.match(line)
        if match:
            titles.add(normalize_tracking_title(match.group("title").strip()))
    return titles


def load_existing_rt_urls(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        return {row["rt_url"] for row in reader if row.get("rt_url")}


def is_drink_event(event: CalendarEvent) -> bool:
    return bool(DRINK_PATTERN.search(event.summary))


def canonicalize_movie_title(title: str) -> str:
    normalized = normalize_tracking_title(title)
    return TITLE_CORRECTIONS.get(normalized, title)


def rt_query_for_title(title: str) -> str:
    exact = title.lower().replace("’", "'").strip()
    if exact in RT_EXACT_TITLE_OVERRIDES:
        return RT_EXACT_TITLE_OVERRIDES[exact]
    normalized = normalize_tracking_title(title)
    return RT_QUERY_OVERRIDES.get(normalized, re.sub(r"\([^)]*\)", "", title)).strip()


def imdb_query_for_title(title: str) -> str:
    exact = title.lower().replace("’", "'").strip()
    if exact in IMDB_EXACT_TITLE_OVERRIDES:
        return IMDB_EXACT_TITLE_OVERRIDES[exact]
    normalized = normalize_tracking_title(title)
    return IMDB_QUERY_OVERRIDES.get(normalized, title)


def rt_direct_match_override_for_title(title: str) -> dict[str, Any] | None:
    normalized = normalize_tracking_title(title)
    return RT_DIRECT_MATCH_OVERRIDES.get(normalized)


def candidate_cache_key(candidate: CandidateMovie) -> str:
    return f"{candidate.date_local.isoformat(timespec='minutes')}|{candidate.title}"


def infer_movie_title(
    summary: str, description: str, location: str
) -> tuple[str | None, str, str]:
    lowered = summary.lower().strip()
    if lowered in GENERIC_MOVIE_PATTERNS:
        return None, "unknown", "generic_summary"

    where_seen = "theater" if "amc" in lowered else "home"
    progress = "default"
    title: str | None = None

    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"^(finished|finish)\s+movie:\s*(.+)$", re.IGNORECASE), "finished"),
        (re.compile(r"^started\s+movie:\s*(.+)$", re.IGNORECASE), "started"),
        (re.compile(r"^middle\s+movie:\s*(.+)$", re.IGNORECASE), "middle"),
        (re.compile(r"^movies?:\s*(.+)$", re.IGNORECASE), "default"),
        (re.compile(r"^amc\s+movie:\s*(.+)$", re.IGNORECASE), "default"),
        (re.compile(r"^(.+?)\s+at\s+amc\b.*$", re.IGNORECASE), "default"),
        (re.compile(r"^([^/,]+?)\s+movie$", re.IGNORECASE), "default"),
    ]
    for pattern, detected_progress in patterns:
        match = pattern.match(summary.strip())
        if match is None:
            continue
        title = match.group(match.lastindex or 1).strip()
        progress = detected_progress
        if "amc" in summary.lower():
            where_seen = "theater"
        break

    if title is None and " at " in summary.lower():
        if any(
            token in f"{summary} {location}".lower()
            for token in ("century", "metreon", "kabuki", "regal", "alamo", "cinemark")
        ):
            title = summary.rsplit(" at ", 1)[0].strip()
            progress = "default"
            where_seen = "theater"

    if title is None:
        return None, where_seen, progress

    title = canonicalize_movie_title(title)
    title = title.strip(" -")
    if not title:
        return None, where_seen, progress

    normalized = normalize_tracking_title(title)
    if not normalized or any(
        pattern in normalized for pattern in NEGATIVE_TITLE_PATTERNS
    ):
        return None, where_seen, progress

    if any(
        normalized == generic or normalized.startswith(f"{generic} ")
        for generic in ("movie", "movies", "search", "find", "started", "middle")
    ):
        return None, where_seen, progress

    if any(
        token in location.lower()
        for token in ("amc", "metreon", "kabuki", "regal", "alamo")
    ):
        where_seen = "theater"

    return title, where_seen, progress


def extract_movie_events(events: list[CalendarEvent]) -> list[MovieEvent]:
    movie_events: list[MovieEvent] = []
    for event in events:
        if "waste time" not in event.calendar_name.lower():
            continue
        parsed = infer_movie_title(event.summary, event.description, event.location)
        title, where_seen, progress = parsed
        if title is None:
            continue
        movie_events.append(
            MovieEvent(
                title=title,
                normalized_title=normalize_tracking_title(title),
                progress=progress,
                where_seen=where_seen,
                start_local=event.start_local,
                end_local=event.end_local,
                duration_hours=event.duration_hours,
                calendar_name=event.calendar_name,
                summary=event.summary,
                description=event.description,
                location=event.location,
            )
        )
    return movie_events


def build_candidate_movies(
    movie_events: list[MovieEvent],
    drink_events: list[CalendarEvent],
    *,
    include_unfinished_started: bool = False,
) -> tuple[list[CandidateMovie], list[dict[str, str]]]:
    grouped: dict[str, list[MovieEvent]] = {}
    for event in movie_events:
        grouped.setdefault(event.normalized_title, []).append(event)

    candidates: list[CandidateMovie] = []
    ambiguous_rows: list[dict[str, str]] = []
    for normalized_title, events in grouped.items():
        events = sorted(events, key=lambda event: event.start_local)
        clusters: list[list[MovieEvent]] = []
        for event in events:
            if not clusters:
                clusters.append([event])
                continue
            if event.start_local - clusters[-1][-1].start_local > timedelta(days=30):
                clusters.append([event])
            else:
                clusters[-1].append(event)

        merged_clusters: list[list[MovieEvent]] = []
        for cluster in clusters:
            if (
                merged_clusters
                and all(
                    event.progress in {"started", "middle"}
                    for event in merged_clusters[-1]
                )
                and any(event.progress == "finished" for event in cluster)
            ):
                merged_clusters[-1].extend(cluster)
            else:
                merged_clusters.append(cluster)

        for cluster in merged_clusters:
            total_logged_hours = round(
                sum(event.duration_hours for event in cluster),
                3,
            )
            representative: MovieEvent | None = None
            watch_status = "finished"
            completion_basis = ""
            confidence = "low"
            saw_in_home = any(event.where_seen == "home" for event in cluster)
            saw_in_theater = any(event.where_seen == "theater" for event in cluster)

            finished_events = [
                event for event in cluster if event.progress == "finished"
            ]
            if finished_events:
                representative = finished_events[0]
                completion_basis = "explicit_finish_marker"
                confidence = "high"
            else:
                theater_events = [
                    event
                    for event in cluster
                    if event.where_seen == "theater" and event.progress == "default"
                ]
                if theater_events:
                    representative = theater_events[0]
                    completion_basis = "theater_ticket_event"
                    confidence = "high"
                else:
                    long_default_events = [
                        event
                        for event in cluster
                        if event.progress == "default" and event.duration_hours >= 1.25
                    ]
                    if long_default_events:
                        representative = long_default_events[0]
                        completion_basis = "long_single_event"
                        confidence = "medium"
                    elif total_logged_hours >= 1.5 and any(
                        event.progress != "started" for event in cluster
                    ):
                        representative = cluster[-1]
                        completion_basis = "multi_session_total"
                        confidence = "medium"

            if representative is None:
                partial_events = [
                    event
                    for event in cluster
                    if event.progress in {"started", "middle"}
                ]
                if (
                    include_unfinished_started
                    and partial_events
                    and all(
                        event.progress in {"started", "middle"} for event in cluster
                    )
                ):
                    representative = partial_events[-1]
                    completion_basis = (
                        "started_only_event"
                        if all(event.progress == "started" for event in cluster)
                        else "partial_only_event"
                    )
                    confidence = "low"
                    watch_status = "unfinished"
                else:
                    ambiguous_rows.append(
                        {
                            "normalized_title": normalized_title,
                            "calendar_title": cluster[0].title,
                            "reason": (
                                "Could not confirm completion from calendar durations"
                            ),
                            "summaries": " | ".join(event.summary for event in cluster),
                        }
                    )
                    continue

            same_day_drinks = [
                event
                for event in drink_events
                if event.start_local.date() == representative.start_local.date()
                and event.start_local < representative.start_local
            ]
            candidates.append(
                CandidateMovie(
                    title=representative.title,
                    normalized_title=normalized_title,
                    watch_status=watch_status,
                    completion_basis=completion_basis,
                    completion_confidence=confidence,
                    event_count=len(cluster),
                    total_logged_hours=total_logged_hours,
                    date_local=representative.start_local,
                    where_seen=representative.where_seen,
                    saw_in_home=saw_in_home,
                    saw_in_theater=saw_in_theater,
                    drink_before_movie=bool(same_day_drinks),
                    calendar_name=representative.calendar_name,
                    summary=representative.summary,
                    description=representative.description,
                    location=representative.location,
                )
            )
    candidates.sort(key=lambda candidate: candidate.date_local)
    return candidates, ambiguous_rows


def filter_new_ambiguities(
    ambiguous_rows: list[dict[str, str]],
    movie_events: list[MovieEvent],
    tracked_titles: set[str],
    cutoff_date: datetime | None,
    min_date_local: datetime | None = None,
) -> list[dict[str, str]]:
    relevant_titles = {
        event.normalized_title
        for event in movie_events
        if event.normalized_title not in tracked_titles
        and (cutoff_date is None or event.start_local > cutoff_date)
        and (min_date_local is None or event.start_local >= min_date_local)
    }
    return [row for row in ambiguous_rows if row["normalized_title"] in relevant_titles]


def http_get_text(url: str) -> str:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "ignore")


def extract_attr(attrs: str, attr_name: str) -> str | None:
    match = re.search(rf'\b{re.escape(attr_name)}="([^"]*)"', attrs)
    if match is None:
        return None
    return html.unescape(match.group(1).strip())


def parse_rt_search_results(search_html: str) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    for section_match in RT_SEARCH_SECTION_PATTERN.finditer(search_html):
        media_type = section_match.group("media_type")
        body = section_match.group("body")
        for row_match in RT_SEARCH_ROW_PATTERN.finditer(body):
            attrs = row_match.group("attrs")
            year_text = extract_attr(attrs, "release-year")
            cast_text = extract_attr(attrs, "cast") or ""
            candidates.append(
                SearchCandidate(
                    media_type=media_type,
                    title=html.unescape(" ".join(row_match.group("title").split())),
                    url=row_match.group("href"),
                    release_year=int(year_text) if year_text else None,
                    cast=tuple(
                        item.strip() for item in cast_text.split(",") if item.strip()
                    ),
                )
            )
    return candidates


def parse_rt_reviews_data(movie_html: str) -> dict[str, Any]:
    json_blobs: dict[str, dict[str, Any]] = {}
    for match in RT_JSON_SCRIPT_PATTERN.finditer(movie_html):
        try:
            json_blobs[match.group("name")] = json.loads(
                html.unescape(match.group("body"))
            )
        except json.JSONDecodeError:
            continue

    reviews_data = json_blobs.get("reviewsData", {}).copy()
    media_scorecard = json_blobs.get("mediaScorecard", {})
    if media_scorecard:
        reviews_data.setdefault(
            "audienceScore", media_scorecard.get("audienceScore", {})
        )
        reviews_data.setdefault("criticsScore", media_scorecard.get("criticsScore", {}))
        for key in ("title", "description"):
            if key not in reviews_data and key in media_scorecard:
                reviews_data[key] = media_scorecard[key]
    if not reviews_data:
        raise ValueError("Could not find Rotten Tomatoes score JSON in page HTML")
    return reviews_data


def normalize_match_text(text: str) -> str:
    lowered = text.lower().replace("&", " and ")
    tokens = [
        ROMAN_NUMERALS.get(token, token)
        for token in TITLE_TOKEN_PATTERN.findall(lowered)
    ]
    return " ".join(tokens)


def significant_tokens(text: str) -> set[str]:
    return {
        ROMAN_NUMERALS.get(token, token)
        for token in TITLE_TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS
    }


def score_rt_candidate(title: str, candidate: SearchCandidate, query: str) -> float:
    normalized_query = normalize_match_text(query)
    normalized_title = normalize_match_text(title)
    normalized_candidate = normalize_match_text(candidate.title)
    similarity = max(
        difflib.SequenceMatcher(None, normalized_query, normalized_candidate).ratio(),
        difflib.SequenceMatcher(None, normalized_title, normalized_candidate).ratio(),
    )
    title_tokens = significant_tokens(title)
    query_tokens = significant_tokens(query)
    candidate_tokens = significant_tokens(candidate.title)
    overlap = 0.0
    if title_tokens | candidate_tokens:
        overlap = len(title_tokens & candidate_tokens) / len(
            title_tokens | candidate_tokens
        )
    if query_tokens | candidate_tokens:
        overlap = max(
            overlap,
            len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens),
        )
    media_bonus = 0.0 if candidate.media_type == "movie" else -0.1
    year_bonus = 0.0
    query_year_match = re.search(r"\b(19|20)\d{2}\b", query)
    if query_year_match is not None and candidate.release_year == int(
        query_year_match.group(0)
    ):
        year_bonus = 0.4
    return similarity + overlap + media_bonus + year_bonus


def fetch_rt_match(title: str) -> RtMatch:
    query = rt_query_for_title(title)
    direct_override = rt_direct_match_override_for_title(title)
    if direct_override is not None:
        movie_html = http_get_text(str(direct_override["rt_url"]))
        reviews_data = parse_rt_reviews_data(movie_html)
        critic_score = reviews_data.get("criticsScore", {}).get("score")
        audience_score = reviews_data.get("audienceScore", {}).get("score")
        matched_title = reviews_data.get("title") or title
        return RtMatch(
            matched_title=str(matched_title),
            rt_url=str(direct_override["rt_url"]),
            critic_score=int(critic_score) if critic_score is not None else None,
            audience_score=int(audience_score) if audience_score is not None else None,
            release_year=int(direct_override["release_year"]),
            match_confidence=2.5,
            query=query,
        )
    search_html = http_get_text(
        f"https://www.rottentomatoes.com/search?search={quote_plus(query)}"
    )
    candidates = parse_rt_search_results(search_html)
    if not candidates:
        raise ValueError(f"No Rotten Tomatoes search candidates found for {title!r}")
    movie_candidates = [
        candidate for candidate in candidates if candidate.media_type == "movie"
    ]
    if movie_candidates:
        candidates = movie_candidates
    scored = sorted(
        (
            (score_rt_candidate(title, candidate, query), candidate)
            for candidate in candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    confidence, best_candidate = scored[0]
    movie_html = http_get_text(best_candidate.url)
    reviews_data = parse_rt_reviews_data(movie_html)
    critic_score = reviews_data.get("criticsScore", {}).get("score")
    audience_score = reviews_data.get("audienceScore", {}).get("score")
    matched_title = reviews_data.get("title") or best_candidate.title
    return RtMatch(
        matched_title=str(matched_title),
        rt_url=best_candidate.url,
        critic_score=int(critic_score) if critic_score is not None else None,
        audience_score=int(audience_score) if audience_score is not None else None,
        release_year=best_candidate.release_year,
        match_confidence=float(confidence),
        query=query,
    )


def score_imdb_candidate(
    title: str, candidate: dict[str, Any], query: str, release_year: int | None
) -> float:
    candidate_title = str(candidate.get("l", ""))
    similarity = max(
        difflib.SequenceMatcher(
            None, normalize_match_text(title), normalize_match_text(candidate_title)
        ).ratio(),
        difflib.SequenceMatcher(
            None, normalize_match_text(query), normalize_match_text(candidate_title)
        ).ratio(),
    )
    query_tokens = significant_tokens(query)
    candidate_tokens = significant_tokens(candidate_title)
    overlap = 0.0
    if query_tokens | candidate_tokens:
        overlap = len(query_tokens & candidate_tokens) / len(
            query_tokens | candidate_tokens
        )
    year_bonus = 0.0
    candidate_year = candidate.get("y")
    if release_year is not None and candidate_year == release_year:
        year_bonus = 0.2
    qid = str(candidate.get("qid") or "")
    media_bonus = 0.0 if qid == "movie" else -0.1
    return similarity + overlap + year_bonus + media_bonus


def parse_imdb_ratings_json(raw_bytes: bytes) -> dict[str, Any]:
    try:
        payload = gzip.decompress(raw_bytes).decode("utf-8", "ignore")
    except OSError:
        payload = raw_bytes.decode("utf-8", "ignore")
    match = re.search(r"imdb\.rating\.run\((?P<body>.*)\)\s*$", payload)
    if match is None:
        raise ValueError("Could not parse IMDb JSONP payload")
    return json.loads(match.group("body"))


def fetch_imdb_match(title: str, release_year: int | None) -> ImdbMatch:
    query = imdb_query_for_title(title)
    first_char = next(
        (character.lower() for character in query if character.isalnum()),
        "a",
    )
    suggestion_url = (
        f"https://v2.sg.media-imdb.com/suggestion/{first_char}/{quote_plus(query)}.json"
    )
    suggestion_text = http_get_text(suggestion_url)
    payload = json.loads(suggestion_text)
    candidates = payload.get("d", [])
    if not candidates:
        raise ValueError(f"No IMDb suggestion candidates found for {title!r}")
    scored = sorted(
        (
            (score_imdb_candidate(title, candidate, query, release_year), candidate)
            for candidate in candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    confidence, best_candidate = scored[0]
    imdb_id = str(best_candidate["id"])
    ratings_url = (
        "https://p.media-imdb.com/static-content/documents/v1/title/"
        f"{imdb_id}/ratings%3Fjsonp=imdb.rating.run:imdb.api.title.ratings/data.json"
    )
    last_http_error: HTTPError | None = None
    for attempt in range(2):
        try:
            request = Request(ratings_url, headers=REQUEST_HEADERS)
            with urlopen(request, timeout=30) as response:
                ratings_payload = parse_imdb_ratings_json(response.read())
            break
        except HTTPError as exc:
            last_http_error = exc
            if exc.code == 403 and attempt == 0:
                time.sleep(0.5)
                continue
            raise
    else:
        assert last_http_error is not None
        raise last_http_error
    resource = ratings_payload.get("resource", {})
    return ImdbMatch(
        matched_title=str(resource.get("title") or best_candidate.get("l") or title),
        imdb_id=imdb_id,
        imdb_url=f"https://www.imdb.com/title/{imdb_id}/",
        imdb_score=(
            float(resource["rating"]) if resource.get("rating") is not None else None
        ),
        imdb_rating_count=(
            int(resource["ratingCount"])
            if resource.get("ratingCount") is not None
            else None
        ),
        release_year=(
            int(resource["year"]) if resource.get("year") is not None else None
        ),
        match_confidence=float(confidence),
        query=query,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def find_new_calendar_movies(
    *,
    calendar_dir: Path,
    notes_path: Path,
    existing_rt_csv_path: Path,
    output_csv_path: Path,
    ambiguity_csv_path: Path,
    cache_path: Path | None = None,
    as_of_local: datetime | None = None,
    after_latest_tracked_date: bool = True,
    min_date_local: datetime | None = None,
    include_unfinished_started: bool = False,
    sleep_seconds: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    as_of = as_of_local or datetime.now(LOCAL_TZ)
    events = parse_calendar_directory(calendar_dir, as_of_local=as_of)
    drink_events = [event for event in events if is_drink_event(event)]
    movie_events = extract_movie_events(events)
    candidates, ambiguous_rows = build_candidate_movies(
        movie_events,
        drink_events,
        include_unfinished_started=include_unfinished_started,
    )

    tracked_titles = parse_notes_titles(notes_path)
    tracked_rt_urls = load_existing_rt_urls(existing_rt_csv_path)
    cutoff_date: datetime | None = None
    if after_latest_tracked_date:
        tracked_events = [
            event for event in movie_events if event.normalized_title in tracked_titles
        ]
        if tracked_events:
            cutoff_date = max(event.start_local for event in tracked_events)
    ambiguous_rows = filter_new_ambiguities(
        ambiguous_rows,
        movie_events,
        tracked_titles,
        cutoff_date,
        min_date_local=min_date_local,
    )
    cache = load_cache(cache_path) if cache_path is not None else {}

    rows: list[dict[str, Any]] = []
    extra_ambiguities: list[dict[str, Any]] = []
    for candidate in candidates:
        if min_date_local is not None and candidate.date_local < min_date_local:
            continue
        if cutoff_date is not None and candidate.date_local <= cutoff_date:
            continue
        if candidate.normalized_title in tracked_titles:
            continue

        cache_key = candidate_cache_key(candidate)
        cached = cache.get(cache_key, {})
        expected_rt_query = rt_query_for_title(candidate.title)
        rt_match: RtMatch | None = None
        if "rt" in cached and cached["rt"].get("query") == expected_rt_query:
            rt_payload = cached["rt"]
            rt_match = RtMatch(
                matched_title=rt_payload["matched_title"],
                rt_url=rt_payload["rt_url"],
                critic_score=rt_payload["critic_score"],
                audience_score=rt_payload["audience_score"],
                release_year=rt_payload["release_year"],
                match_confidence=rt_payload["match_confidence"],
                query=rt_payload["query"],
            )
        else:
            try:
                rt_match = fetch_rt_match(candidate.title)
                cached["rt"] = {
                    "matched_title": rt_match.matched_title,
                    "rt_url": rt_match.rt_url,
                    "critic_score": rt_match.critic_score,
                    "audience_score": rt_match.audience_score,
                    "release_year": rt_match.release_year,
                    "match_confidence": rt_match.match_confidence,
                    "query": rt_match.query,
                }
                if cache_path is not None:
                    cache[cache_key] = cached
                    save_cache(cache_path, cache)
                time.sleep(sleep_seconds)
            except (HTTPError, URLError, ValueError) as exc:
                extra_ambiguities.append(
                    {
                        "normalized_title": candidate.normalized_title,
                        "calendar_title": candidate.title,
                        "reason": f"Rotten Tomatoes lookup failed: {exc}",
                        "summaries": candidate.summary,
                    }
                )
                continue

        if rt_match is not None and rt_match.rt_url in tracked_rt_urls:
            continue

        imdb_match: ImdbMatch | None = None
        expected_imdb_query = imdb_query_for_title(candidate.title)
        if "imdb" in cached and cached["imdb"].get("query") == expected_imdb_query:
            imdb_payload = cached["imdb"]
            imdb_match = ImdbMatch(
                matched_title=imdb_payload["matched_title"],
                imdb_id=imdb_payload["imdb_id"],
                imdb_url=imdb_payload["imdb_url"],
                imdb_score=imdb_payload["imdb_score"],
                imdb_rating_count=imdb_payload["imdb_rating_count"],
                release_year=imdb_payload["release_year"],
                match_confidence=imdb_payload["match_confidence"],
                query=imdb_payload["query"],
            )
        else:
            try:
                imdb_match = fetch_imdb_match(
                    candidate.title,
                    rt_match.release_year if rt_match is not None else None,
                )
                cached["imdb"] = {
                    "matched_title": imdb_match.matched_title,
                    "imdb_id": imdb_match.imdb_id,
                    "imdb_url": imdb_match.imdb_url,
                    "imdb_score": imdb_match.imdb_score,
                    "imdb_rating_count": imdb_match.imdb_rating_count,
                    "release_year": imdb_match.release_year,
                    "match_confidence": imdb_match.match_confidence,
                    "query": imdb_match.query,
                }
                if cache_path is not None:
                    cache[cache_key] = cached
                    save_cache(cache_path, cache)
                time.sleep(sleep_seconds)
            except (HTTPError, URLError, ValueError, OSError) as exc:
                extra_ambiguities.append(
                    {
                        "normalized_title": candidate.normalized_title,
                        "calendar_title": candidate.title,
                        "reason": f"IMDb lookup failed: {exc}",
                        "summaries": candidate.summary,
                    }
                )

        rows.append(
            {
                "date": candidate.date_local.date().isoformat(),
                "datetime_local": candidate.date_local.isoformat(timespec="minutes"),
                "movie_title": candidate.title,
                "normalized_title": candidate.normalized_title,
                "watch_status": candidate.watch_status,
                "where_seen": candidate.where_seen,
                "saw_in_home": candidate.saw_in_home,
                "saw_in_theater": candidate.saw_in_theater,
                "drink_before_movie": candidate.drink_before_movie,
                "completion_basis": candidate.completion_basis,
                "completion_confidence": candidate.completion_confidence,
                "event_count": candidate.event_count,
                "total_logged_hours": candidate.total_logged_hours,
                "calendar_name": candidate.calendar_name,
                "calendar_summary": candidate.summary,
                "calendar_location": candidate.location,
                "rt_matched_title": (
                    rt_match.matched_title if rt_match is not None else ""
                ),
                "rt_release_year": (
                    rt_match.release_year if rt_match is not None else ""
                ),
                "rt_audience_score": (
                    rt_match.audience_score if rt_match is not None else ""
                ),
                "rt_critic_score": (
                    rt_match.critic_score if rt_match is not None else ""
                ),
                "rt_url": rt_match.rt_url if rt_match is not None else "",
                "rt_match_confidence": (
                    f"{rt_match.match_confidence:.3f}" if rt_match is not None else ""
                ),
                "imdb_matched_title": (
                    imdb_match.matched_title if imdb_match is not None else ""
                ),
                "imdb_release_year": (
                    imdb_match.release_year if imdb_match is not None else ""
                ),
                "imdb_score": imdb_match.imdb_score if imdb_match is not None else "",
                "imdb_rating_count": (
                    imdb_match.imdb_rating_count if imdb_match is not None else ""
                ),
                "imdb_url": imdb_match.imdb_url if imdb_match is not None else "",
                "imdb_match_confidence": (
                    f"{imdb_match.match_confidence:.3f}"
                    if imdb_match is not None
                    else ""
                ),
                "notes_cutoff_date": (
                    cutoff_date.date().isoformat() if cutoff_date is not None else ""
                ),
            }
        )

    rows.sort(key=lambda row: row["datetime_local"])
    all_ambiguities = ambiguous_rows + extra_ambiguities
    write_csv(output_csv_path, rows)
    write_csv(ambiguity_csv_path, all_ambiguities)
    return rows, all_ambiguities
