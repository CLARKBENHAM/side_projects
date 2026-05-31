"""Estimate per-book reading WPM from calendar sessions and Play Books notes.

Inputs are local Google Takeout calendar ICS files and Google Play Books notes
DOCX exports. The analysis is intentionally dependency-light: it parses the
small subset of ICS and DOCX XML needed here instead of requiring the old
notebook environment.
"""

from __future__ import annotations

import argparse
import html
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALENDAR_DIR = REPO_ROOT / "data" / "takeout_05_14_26" / "Calendar"
DEFAULT_NOTES_DIR = REPO_ROOT / "data" / "Play_Books_Notes-20260531T040817Z"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "book_wpm_outputs"
DEFAULT_LOCAL_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_ANALYSIS_END = date(2026, 5, 31)
DEFAULT_WORDS_PER_PAGE = 275
DEFAULT_AUDIOBOOK_WPM = 350

DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
DATE_RE = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4}$"
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
COLOR_HEADINGS = {"yellow", "green", "blue", "red"}
HIGHLIGHT_FILLS = {
    "fde096": "yellow",
    "c5e1a5": "green",
    "93e3ed": "blue",
    "ffb8a1": "red",
}
PAGE_LINK_COLOR = "1565c0"
DATE_COLOR = "757575"
NOTE_TEXT_COLOR = "424242"

STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "by"}
ABBREV_STOPWORDS = STOPWORDS | {"its", "with"}
GENERIC_BOOK_REFS = {"", "book", "books", "reading", "read"}

MANUAL_ABBREVS: dict[str, str] = {
    "aeic": "An Experiment in Criticism",
    "asme y14.41": "ASME Y14.41",
    "bftc": "Bobby Fischer Teaches Chess",
    "h": "Hyperion",
    "htiymwtai": "how to improve your marriage without talking about it",
    "mml": "Mathematics for Machine Learning",
    "nvc": "nonviolent communication",
    "ljtptp": "The Path to Power: The Years of Lyndon Johnson I",
    "su": "Shape Up Stop Running in Circles and Ship Work that Matters",
    "tgd": "the great democracies",
    "tinad": "there is no anti-memetic division",
    "toboss": "The Oxford Book of Short Stories",
}

TITLE_ALIASES: list[tuple[str, str]] = [
    ("shape up", "shaping up"),
    ("the history of the english speaking peoples volume 4", "the great democracies"),
    ("night watch", "discworld 29 night watch"),
    ("fall of hyperion", "hyperion cantos 02 the fall of hyperion"),
    ("the mind body prescription", "the mindbody prescription"),
    ("autobiography of ben franklin", "the autobiography of benjamin franklin"),
    ("into the aquarium", "inside the aquarium"),
    ("deaths end", "death's end"),
    ("lyndon johnson the path to power", "the path to power"),
]

TITLE_REWRITES: dict[str, str] = {
    "lyndon johnson the path to power": "The Path to Power: The Years of Lyndon Johnson I",
    "the path to power": "The Path to Power: The Years of Lyndon Johnson I",
    "shaping up": "Shape Up Stop Running in Circles and Ship Work that Matters",
    "finished barbarians at the gate": "Barbarians at the Gate",
}

VOLUME_SPLITS: list[tuple[str, str, str, str]] = [
    ("twc", "2000-01-01", "2025-12-27", "the world crisis"),
    ("twc", "2025-12-28", "2026-01-04", "the world crisis volume 2"),
    ("twc", "2026-01-05", "2026-01-07", "the world crisis volume 3"),
    ("twc", "2026-01-08", "2026-01-10", "the world crisis volume 4"),
]

PAGE_COUNT_CORRECTIONS = {
    "THE 48 LAWS OF POWER - Robert Greene.pdf": 452,
    "Knudsen": 336,
    "Harry Potter and the Methods of Rationality": 1967,
    "Surprised by Joy: The shape of my early life": 238,
    "The Good Research Code Handbook.pdf": 60,
}

PAGE_QUALITY_OVERRIDES = {
    "dfw tv": (
        "low",
        "manual audit: local PDF/page metadata mismatch; 407 pages looks wrong",
    ),
    "bobby fischer teaches chess": (
        "low",
        "manual audit: chess workbook pages are sparse, so pages*words/page is weak",
    ),
    "bleak house": (
        "low",
        "manual audit: 85 pages is not plausible for full Bleak House",
    ),
    "50 essays by george orwell": (
        "low",
        "manual audit: local/notes evidence suggests much longer than metadata pages",
    ),
    "tremendous trifles": (
        "medium",
        "manual audit: local text implies more words than metadata pages",
    ),
    "parkinson's law": (
        "medium",
        "manual audit: possible edition mismatch",
    ),
}

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class CalendarEvent:
    summary: str
    calendar_name: str
    start: datetime
    end: datetime
    duration_hours: float
    uid: str = ""


@dataclass(frozen=True)
class ClassifiedEvent:
    date: date
    start: datetime
    end: datetime
    calendar_name: str
    summary: str
    part_summary: str
    event_type: str
    book_ref: str
    duration_hours: float
    uid: str = ""


@dataclass(frozen=True)
class DocxParagraph:
    text: str
    fills: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()


@dataclass(frozen=True)
class NoteEntry:
    highlight: str
    note: str
    color: str
    page: int | None
    date_text: str


@dataclass
class NoteStats:
    title: str
    author: str = ""
    source_paths: list[Path] = field(default_factory=list)
    entries: list[NoteEntry] = field(default_factory=list)

    @property
    def highlight_count(self) -> int:
        return len({normalize_loose(e.highlight) for e in self.entries if e.highlight})

    @property
    def note_count(self) -> int:
        return sum(1 for e in self.entries if e.note.strip())

    @property
    def highlight_words(self) -> int:
        return sum(count_words(text) for text in self._deduped_highlights())

    @property
    def note_words(self) -> int:
        return sum(count_words(e.note) for e in self.entries if e.note.strip())

    @property
    def max_page(self) -> int | None:
        pages = [e.page for e in self.entries if e.page is not None]
        return max(pages) if pages else None

    def _deduped_highlights(self) -> list[str]:
        seen: set[str] = set()
        highlights: list[str] = []
        for entry in self.entries:
            key = normalize_loose(entry.highlight)
            if not key or key in seen:
                continue
            seen.add(key)
            highlights.append(entry.highlight)
        return highlights


@dataclass(frozen=True)
class MetadataRecord:
    title: str
    author: str = ""
    category: str = ""
    page_count: float | None = None
    source: str = ""


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def normalize_loose(text: object) -> str:
    value = str(text or "").lower().strip()
    value = html.unescape(value)
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = value.replace("\u201c", "").replace("\u201d", "")
    value = re.sub(r"\.(pdf|epub|mobi|html|txt|docx)$", "", value)
    value = re.sub(r"\(\d{4}\)", "", value)
    value = re.sub(r"\s*\(day\s*\d+/\d+\)\s*$", "", value)
    value = re.sub(r"[_:;,\-/.#\[\](){}!?]+", " ", value)
    value = re.sub(r"[^a-z0-9' ]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_title(text: object) -> str:
    value = normalize_loose(text)
    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
    }
    for word, digit in number_words.items():
        value = re.sub(rf"\b{word}\b", digit, value)
    return value


def significant_words(text: object) -> list[str]:
    return [word for word in normalize_title(text).split() if word not in STOPWORDS]


def _containment_match(shorter: str, longer: str) -> bool:
    remainder = longer.replace(shorter, "", 1).strip()
    if re.match(r"(volume|vol|part|book)\s*\d", remainder):
        return False
    if remainder.endswith((" volume", " vol", " part", " book")):
        return False
    return len(shorter) >= len(longer) * 0.45


def titles_match(a: object, b: object) -> bool:
    left = normalize_title(a)
    right = normalize_title(b)
    if not left or not right:
        return False
    if left == right:
        return True
    for alias_left, alias_right in TITLE_ALIASES:
        norm_left = normalize_title(alias_left)
        norm_right = normalize_title(alias_right)
        if (left == norm_left and right == norm_right) or (
            left == norm_right and right == norm_left
        ):
            return True
        if (left.startswith(norm_left + " ") and right.startswith(norm_right)) or (
            right.startswith(norm_left + " ") and left.startswith(norm_right)
        ):
            return True
    if len(left) > 5 and len(right) > 5 and (left in right or right in left):
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if _containment_match(shorter, longer):
            return True

    left_words = [word for word in left.split() if word not in STOPWORDS]
    right_words = [word for word in right.split() if word not in STOPWORDS]
    n = min(len(left_words), len(right_words), 3)
    if n >= 2 and left_words[:n] == right_words[:n]:
        longer_words = left_words if len(left_words) > len(right_words) else right_words
        extra = longer_words[n:]
        if not (extra and extra[0] in {"volume", "vol", "part", "book"}):
            return True

    left_set = set(left_words)
    right_set = set(right_words)
    min_count = min(len(left_set), len(right_set))
    max_count = max(len(left_set), len(right_set))
    if min_count >= 3 and min_count >= max_count * 0.5:
        overlap = len(left_set & right_set)
        if overlap / min_count >= 0.7:
            diff_left = left_set - right_set
            diff_right = right_set - left_set
            has_volume = bool(
                (left_set | right_set) & {"volume", "vol", "part", "book"}
            )
            if not (
                has_volume
                and all(word.isdigit() for word in diff_left)
                and all(word.isdigit() for word in diff_right)
                and diff_left != diff_right
            ):
                return True
    return False


def _split_words_for_initials(
    title: object, *, convert_number_words: bool
) -> list[str]:
    normalizer = normalize_title if convert_number_words else normalize_loose
    value = normalizer(title).replace("'", "")
    return re.findall(r"[a-z0-9]+", value)


def _word_initials(word: str) -> set[str]:
    if word.isdigit():
        return {word, word[0]}
    return {word[0]}


def _join_initial_options(words: Iterable[str]) -> set[str]:
    variants = {""}
    for word in words:
        variants = {
            prefix + initial for prefix in variants for initial in _word_initials(word)
        }
    return variants


def title_initial_variants(title: object) -> set[str]:
    word_sets = [
        _split_words_for_initials(title, convert_number_words=False),
        _split_words_for_initials(title, convert_number_words=True),
    ]
    word_sets = [
        words
        for index, words in enumerate(word_sets)
        if words and words not in word_sets[:index]
    ]
    if not word_sets:
        return set()
    results: set[str] = set()
    for words in word_sets:
        results.update(_join_initial_options(words))
        significant = [word for word in words if word not in ABBREV_STOPWORDS]
        if significant:
            results.update(_join_initial_options(significant))
        article_indices = [i for i, word in enumerate(words) if word in STOPWORDS]
        non_article_indices = [
            i for i, word in enumerate(words) if word not in STOPWORDS
        ]
        for mask in range(1 << len(article_indices)):
            included = set(non_article_indices)
            for bit, idx in enumerate(article_indices):
                if mask & (1 << bit):
                    included.add(idx)
            if included:
                results.update(
                    _join_initial_options(words[i] for i in sorted(included))
                )
    return {value for value in results if value}


def title_prefixes(title: object) -> list[str]:
    value = str(title or "")
    prefixes = [value]
    for separator in [":", " - ", " \u2014 ", "\u00b7"]:
        if separator in value:
            prefixes.append(value.split(separator, 1)[0].strip())
    return [prefix for prefix in prefixes if prefix]


def edit_distance_one(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    short, long = (left, right) if len(left) < len(right) else (right, left)
    skipped = 0
    pos = 0
    for char in long:
        if pos < len(short) and short[pos] == char:
            pos += 1
        else:
            skipped += 1
    return skipped <= 1


def is_abbreviation(ref: object) -> bool:
    value = normalize_title(ref).replace(" ", "")
    return bool(
        re.fullmatch(r"[a-z]{1,10}", value)
        or (re.fullmatch(r"[a-z0-9]{2,10}", value) and re.search(r"[a-z]", value))
    )


def abbreviation_matches_title(ref: object, title: object) -> bool:
    value = normalize_title(ref).replace(" ", "")
    if not value:
        return False
    if value in MANUAL_ABBREVS and titles_match(MANUAL_ABBREVS[value], title):
        return True
    for title_variant in title_prefixes(title):
        if len(value) <= 2:
            strict_variants: set[str] = set()
            for convert in [False, True]:
                words = _split_words_for_initials(
                    title_variant, convert_number_words=convert
                )
                if words:
                    strict_variants.update(_join_initial_options(words))
            if value in strict_variants:
                return True
            continue
        variants = title_initial_variants(title_variant)
        if value in variants:
            return True
        if len(value) >= 4 and any(
            variant.startswith(value) and len(variant) - len(value) <= 3
            for variant in variants
        ):
            return True
        title_norm = normalize_title(title_variant)
        if len(value) >= 3 and (
            title_norm == value
            or (len(value) >= 5 and title_norm.startswith(value + " "))
        ):
            return True
    return False


def ref_matches_title(ref: object, title: object) -> bool:
    ref_norm = normalize_title(ref)
    if ref_norm in MANUAL_ABBREVS and titles_match(MANUAL_ABBREVS[ref_norm], title):
        return True
    if titles_match(ref_norm, title):
        return True
    if is_abbreviation(ref_norm) and abbreviation_matches_title(ref_norm, title):
        return True
    return False


def title_match_score(ref: object, title: object) -> float:
    ref_norm = normalize_title(ref)
    title_norm = normalize_title(title)
    if not ref_norm or not title_norm:
        return 0
    if ref_norm == title_norm:
        return 100
    if ref_norm in MANUAL_ABBREVS and titles_match(MANUAL_ABBREVS[ref_norm], title):
        return 98
    if titles_match(ref_norm, title):
        return 85
    if is_abbreviation(ref_norm) and abbreviation_matches_title(ref_norm, title):
        return 75
    ref_words = set(significant_words(ref_norm))
    title_words = set(significant_words(title_norm))
    if not ref_words or not title_words:
        return 0
    overlap = len(ref_words & title_words)
    return 50 * overlap / max(len(ref_words), len(title_words))


def clean_book_ref(ref: object, event_date: date | None = None) -> str:
    value = str(ref or "").strip()
    value = value.replace("\\,", ",").replace("\\;", ";").replace("\\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*\(day\s*\d+/\d+\)\s*$", "", value, flags=re.IGNORECASE)
    value = value.strip(" :-;")
    norm = normalize_title(value)
    if event_date is not None:
        for split_ref, start, end, canonical in VOLUME_SPLITS:
            if norm == split_ref and start <= str(event_date) <= end:
                return canonical
    return TITLE_REWRITES.get(norm, value)


def canonicalize_title(value: object) -> str:
    text = str(value or "").strip()
    norm = normalize_title(text)
    if norm in TITLE_REWRITES:
        return TITLE_REWRITES[norm]
    if norm in MANUAL_ABBREVS:
        return MANUAL_ABBREVS[norm]
    return text


def unfold_ics_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw_line.startswith((" ", "\t")) and lines:
            lines[-1] += raw_line[1:]
        elif raw_line:
            lines.append(raw_line)
    return lines


def split_ical_line(line: str) -> tuple[str, dict[str, str], str] | None:
    if ":" not in line:
        return None
    left, value = line.split(":", 1)
    parts = left.split(";")
    name = parts[0].upper()
    params: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, param_value = part.split("=", 1)
            params[key.upper()] = param_value
    return name, params, value


def unescape_ical_text(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def parse_ical_datetime(
    value: str,
    params: dict[str, str] | None = None,
    *,
    default_tz: ZoneInfo = DEFAULT_LOCAL_TZ,
    display_tz: ZoneInfo = DEFAULT_LOCAL_TZ,
) -> datetime:
    params = params or {}
    value = value.strip()
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        parsed_date = datetime.strptime(value[:8], "%Y%m%d").date()
        return datetime.combine(parsed_date, time.min, tzinfo=display_tz)
    if value.endswith("Z"):
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return parsed.astimezone(display_tz)
    fmt = "%Y%m%dT%H%M%S" if len(value) >= 15 else "%Y%m%dT%H%M"
    parsed = datetime.strptime(value, fmt)
    tzid = params.get("TZID")
    source_tz = ZoneInfo(tzid) if tzid else default_tz
    return parsed.replace(tzinfo=source_tz).astimezone(display_tz)


def read_ics_events(
    calendar_dir: Path,
    *,
    display_tz: ZoneInfo = DEFAULT_LOCAL_TZ,
    analysis_end: date = DEFAULT_ANALYSIS_END,
) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []
    for ics_path in sorted(calendar_dir.glob("*.ics")):
        if "Personal Dates" in ics_path.name:
            continue
        calendar_name = ics_path.stem
        text = ics_path.read_text(encoding="utf-8", errors="ignore")
        lines = unfold_ics_lines(text)
        default_tz = _calendar_default_tz(lines, display_tz)
        current: list[str] | None = None
        for line in lines:
            if line == "BEGIN:VEVENT":
                current = []
                continue
            if line == "END:VEVENT" and current is not None:
                parsed = _parse_ics_event_lines(
                    current,
                    calendar_name=calendar_name,
                    default_tz=default_tz,
                    display_tz=display_tz,
                    analysis_end=analysis_end,
                )
                events.extend(parsed)
                current = None
                continue
            if current is not None:
                current.append(line)
    return events


def _calendar_default_tz(lines: list[str], fallback: ZoneInfo) -> ZoneInfo:
    for line in lines:
        parsed = split_ical_line(line)
        if parsed and parsed[0] == "X-WR-TIMEZONE":
            try:
                return ZoneInfo(parsed[2])
            except Exception:
                return fallback
    return fallback


def _event_looks_book_related(summary: str) -> bool:
    lowered = summary.lower()
    if "book" in lowered or "audiobook" in lowered or "audio book" in lowered:
        if re.search(r"\b(booked|booking|facebook)\b", lowered):
            return False
        return True
    return bool(
        re.match(r"\s*finished\s*:", lowered)
        or re.match(r"\s*finished\s+", lowered)
        or re.search(r"\s+finished\s*$", lowered)
        or re.match(r"\s*(?:blogs?\s*/\s*)?read\s*:", lowered)
        or re.match(r"\s*read\s+\S", lowered)
        or re.match(r"\s*listen(?:ing)?\s+(?:to\s+)?\S", lowered)
    )


def _parse_ics_event_lines(
    lines: list[str],
    *,
    calendar_name: str,
    default_tz: ZoneInfo,
    display_tz: ZoneInfo,
    analysis_end: date,
) -> list[CalendarEvent]:
    props: dict[str, list[tuple[dict[str, str], str]]] = defaultdict(list)
    for line in lines:
        parsed = split_ical_line(line)
        if parsed is None:
            continue
        name, params, value = parsed
        props[name].append((params, value))
    summary = unescape_ical_text(props.get("SUMMARY", [({}, "")])[0][1]).strip()
    if not summary or not _event_looks_book_related(summary):
        return []
    status = props.get("STATUS", [({}, "")])[0][1].upper()
    if status == "CANCELLED":
        return []
    if "DTSTART" not in props:
        return []
    start_params, start_value = props["DTSTART"][0]
    start = parse_ical_datetime(
        start_value,
        start_params,
        default_tz=default_tz,
        display_tz=display_tz,
    )
    if "DTEND" in props:
        end_params, end_value = props["DTEND"][0]
        end = parse_ical_datetime(
            end_value,
            end_params,
            default_tz=default_tz,
            display_tz=display_tz,
        )
    else:
        end = start + timedelta(hours=1)
    uid = props.get("UID", [({}, "")])[0][1]
    exdates = _parse_exdates(props.get("EXDATE", []), default_tz, display_tz)
    rrules = props.get("RRULE", [])
    if not rrules:
        duration = max((end - start).total_seconds() / 3600.0, 0)
        return [CalendarEvent(summary, calendar_name, start, end, duration, uid)]
    return [
        CalendarEvent(summary, calendar_name, occ_start, occ_end, duration_hours, uid)
        for occ_start, occ_end, duration_hours in _expand_rrule(
            start,
            end,
            rrules[0][1],
            exdates=exdates,
            analysis_end=analysis_end,
            display_tz=display_tz,
        )
    ]


def _parse_exdates(
    raw_values: list[tuple[dict[str, str], str]],
    default_tz: ZoneInfo,
    display_tz: ZoneInfo,
) -> set[datetime]:
    exdates: set[datetime] = set()
    for params, raw_value in raw_values:
        for part in raw_value.split(","):
            try:
                exdates.add(
                    parse_ical_datetime(
                        part,
                        params,
                        default_tz=default_tz,
                        display_tz=display_tz,
                    )
                )
            except ValueError:
                continue
    return exdates


def _parse_rrule_value(raw_value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in raw_value.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            parsed[key.upper()] = value
    return parsed


def _expand_rrule(
    start: datetime,
    end: datetime,
    raw_rrule: str,
    *,
    exdates: set[datetime],
    analysis_end: date,
    display_tz: ZoneInfo,
) -> Iterable[tuple[datetime, datetime, float]]:
    rule = _parse_rrule_value(raw_rrule)
    freq = rule.get("FREQ", "").upper()
    interval = int(rule.get("INTERVAL", "1"))
    count = int(rule.get("COUNT", "0") or 0)
    until = datetime.combine(analysis_end, time.max, tzinfo=display_tz)
    if "UNTIL" in rule:
        try:
            until = min(
                until,
                parse_ical_datetime(
                    rule["UNTIL"],
                    {},
                    default_tz=display_tz,
                    display_tz=display_tz,
                ),
            )
        except ValueError:
            pass
    duration = end - start
    emitted = 0
    byday = {day for day in rule.get("BYDAY", "").split(",") if day}
    weekday_map = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
    by_weekday = {weekday_map[day[-2:]] for day in byday if day[-2:] in weekday_map}

    def maybe_emit(candidate_date: date) -> tuple[datetime, datetime, float] | None:
        nonlocal emitted
        candidate_start = datetime.combine(
            candidate_date, start.timetz().replace(tzinfo=None), tzinfo=start.tzinfo
        )
        if candidate_start < start or candidate_start > until:
            return None
        if any(
            abs((candidate_start - exdate).total_seconds()) < 60 for exdate in exdates
        ):
            return None
        emitted += 1
        candidate_end = candidate_start + duration
        return (
            candidate_start,
            candidate_end,
            max(duration.total_seconds() / 3600.0, 0),
        )

    if freq == "DAILY":
        current = start.date()
        while datetime.combine(current, time.min, tzinfo=display_tz) <= until:
            if not by_weekday or current.weekday() in by_weekday:
                occurrence = maybe_emit(current)
                if occurrence:
                    yield occurrence
                    if count and emitted >= count:
                        return
            current += timedelta(days=interval)
        return

    if freq == "WEEKLY":
        current = start.date()
        while datetime.combine(current, time.min, tzinfo=display_tz) <= until:
            week_delta = (current - start.date()).days // 7
            weekday_ok = (
                current.weekday() in by_weekday
                if by_weekday
                else current.weekday() == start.weekday()
            )
            if week_delta % interval == 0 and weekday_ok:
                occurrence = maybe_emit(current)
                if occurrence:
                    yield occurrence
                    if count and emitted >= count:
                        return
            current += timedelta(days=1)
        return

    if freq == "MONTHLY":
        current = start.date()
        by_monthday = [
            int(part)
            for part in rule.get("BYMONTHDAY", str(start.day)).split(",")
            if re.fullmatch(r"-?\d+", part)
        ]
        month_index = 0
        while datetime.combine(current, time.min, tzinfo=display_tz) <= until:
            if month_index % interval == 0:
                for monthday in by_monthday:
                    try:
                        candidate = current.replace(day=monthday)
                    except ValueError:
                        continue
                    occurrence = maybe_emit(candidate)
                    if occurrence:
                        yield occurrence
                        if count and emitted >= count:
                            return
            month_index += 1
            year = current.year + (current.month // 12)
            month = current.month % 12 + 1
            current = current.replace(year=year, month=month, day=1)
        return

    occurrence = maybe_emit(start.date())
    if occurrence:
        yield occurrence


def split_event_summary(summary: str) -> list[str]:
    parts = [part.strip() for part in summary.split("/")]
    return [part for part in parts if part] or [summary.strip()]


def classify_summary_part(part: str, event_date: date) -> tuple[str, str]:
    text = unescape_ical_text(part).strip()
    text = re.sub(r"\s+", " ", text)
    lowered = text.lower().strip()
    if not lowered:
        return "other", ""
    if re.match(
        r"^(get|search|toil|misc|find|review|track|which|write|luke)\b.*books?\b",
        lowered,
    ):
        return "other", ""
    if re.search(r"\bbook\s+sim\b|\bbooks?\s+blog\b|\bbook\s+analysis\b", lowered):
        return "other", ""
    if re.match(r"^finished\s+(movie|film|show|tv|episode)\b", lowered):
        return "other", ""

    finished_patterns = [
        r"^(?:ai qs\s*)?finished\s+(?:cs\s+)?(?:audio\s*)?book\s*:?\s*(.+)$",
        r"^finished\s+audiobook\s*:?\s*(.+)$",
        r"^finished\s*:\s*(.+)$",
        r"^finished\s+(.+)$",
        r"^end\s+of\s+book\s*:?\s*(.+)$",
        r"^book\s+finished\s*:?\s*(.+)$",
        r"^(.+?)\s+book\s+finished$",
        r"^(.+?)\s+finished$",
    ]
    for pattern in finished_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return "finished", clean_book_ref(match.group(1), event_date)

    started_patterns = [
        r"^started\s+(?:audio\s*)?book\s*:?\s*(.+)$",
        r"^started\s+audiobook\s*:?\s*(.+)$",
        r"^start\s+(?:audio\s*)?book\s*:?\s*(.+)$",
    ]
    for pattern in started_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return "started", clean_book_ref(match.group(1), event_date)

    audio_match = re.match(r"^(?:audio\s*)book\s*:?\s*(.+)$", text, re.IGNORECASE)
    if audio_match:
        ref = clean_book_ref(audio_match.group(1), event_date)
        return ("audiobook" if ref else "generic", ref)

    listen_match = re.match(r"^listen(?:ing)?\s+(?:to\s+)?(.+)$", text, re.IGNORECASE)
    if listen_match:
        ref = clean_book_ref(listen_match.group(1), event_date)
        return ("audiobook" if ref else "generic", ref)

    read_match = re.match(r"^(?:blogs?\s*/\s*)?read\s*:?\s*(.+)$", text, re.IGNORECASE)
    if read_match:
        ref = clean_book_ref(read_match.group(1), event_date)
        return ("reading" if ref else "generic", ref)

    book_match = re.match(
        r"^(?:cs\s+|history\s+|ai histories\s+|ai history\s+|org\s+)?books?\b(?:\s*[;:]\s*(.*)|\s+(.+))?$",
        text,
        re.IGNORECASE,
    )
    if book_match:
        raw_ref = book_match.group(1) or book_match.group(2) or ""
        ref = clean_book_ref(raw_ref, event_date)
        if normalize_title(ref).startswith("finished "):
            return "finished", clean_book_ref(ref[len("finished ") :], event_date)
        return ("reading" if ref else "generic", ref)

    trailing_book = re.match(r"^(.+?)\s+book$", text, re.IGNORECASE)
    if trailing_book:
        ref = clean_book_ref(trailing_book.group(1), event_date)
        if normalize_title(ref) in {"frivolous", "bad", "richard", "richards"}:
            return "generic", ""
        return "reading", ref

    if lowered in GENERIC_BOOK_REFS:
        return "generic", ""
    return "other", ""


def classify_calendar_events(events: list[CalendarEvent]) -> pd.DataFrame:
    rows: list[ClassifiedEvent] = []
    for event in events:
        parts = split_event_summary(event.summary)
        part_duration = event.duration_hours / max(len(parts), 1)
        for part in parts:
            event_type, book_ref = classify_summary_part(part, event.start.date())
            if event_type == "other":
                continue
            if part_duration > 12 and event_type in {"reading", "audiobook", "generic"}:
                continue
            rows.append(
                ClassifiedEvent(
                    date=event.start.date(),
                    start=event.start,
                    end=event.end,
                    calendar_name=event.calendar_name,
                    summary=event.summary,
                    part_summary=part,
                    event_type=event_type,
                    book_ref=book_ref,
                    duration_hours=part_duration,
                    uid=event.uid,
                )
            )
    return pd.DataFrame([row.__dict__ for row in rows])


def parse_docx_paragraphs(path: Path) -> list[DocxParagraph]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[DocxParagraph] = []
    w_ns = DOCX_NS["w"]
    val_key = f"{{{w_ns}}}val"
    fill_key = f"{{{w_ns}}}fill"
    for paragraph in root.findall(".//w:p", DOCX_NS):
        texts: list[str] = []
        fills: set[str] = set()
        colors: set[str] = set()
        styles: set[str] = set()
        p_style = paragraph.find("./w:pPr/w:pStyle", DOCX_NS)
        if p_style is not None and p_style.attrib.get(val_key):
            styles.add(p_style.attrib[val_key])
        for run in paragraph.findall(".//w:r", DOCX_NS):
            for text_node in run.findall(".//w:t", DOCX_NS):
                texts.append(text_node.text or "")
            shading = run.find("./w:rPr/w:shd", DOCX_NS)
            if shading is not None:
                fill = shading.attrib.get(fill_key)
                if fill and fill != "auto":
                    fills.add(fill.lower())
            color = run.find("./w:rPr/w:color", DOCX_NS)
            if color is not None:
                color_value = color.attrib.get(val_key)
                if color_value and color_value != "auto":
                    colors.add(color_value.lower())
        text = "".join(texts).strip()
        if text:
            paragraphs.append(
                DocxParagraph(
                    text=text,
                    fills=tuple(sorted(fills)),
                    colors=tuple(sorted(colors)),
                    styles=tuple(sorted(styles)),
                )
            )
    return paragraphs


def note_title_from_filename(path: Path) -> str:
    stem = path.stem
    match = re.match(r"Notes from _(.+)_$", stem)
    if match:
        return match.group(1).replace("_", ": ")
    return stem


def parse_note_stats(path: Path) -> NoteStats:
    paragraphs = parse_docx_paragraphs(path)
    return note_stats_from_paragraphs(
        paragraphs, fallback_title=note_title_from_filename(path), path=path
    )


def note_stats_from_paragraphs(
    paragraphs: list[DocxParagraph],
    *,
    fallback_title: str,
    path: Path | None = None,
) -> NoteStats:
    title = fallback_title
    author = ""
    for index, paragraph in enumerate(paragraphs):
        if paragraph.text.lower() == "annotations by color":
            prior = [
                p.text
                for p in paragraphs[:index]
                if p.text
                and "this document is overwritten" not in p.text.lower()
                and "you should make a copy" not in p.text.lower()
            ]
            if prior:
                title = prior[0]
            if len(prior) >= 2:
                author = prior[1]
            break

    entries: list[NoteEntry] = []
    current_color = ""
    pending_highlight: list[str] = []
    pending_note: list[str] = []
    pending_date = ""
    pending_page: int | None = None

    def flush() -> None:
        nonlocal pending_highlight, pending_note, pending_date, pending_page
        highlight = " ".join(pending_highlight).strip()
        note = " ".join(pending_note).strip()
        if highlight:
            entries.append(
                NoteEntry(
                    highlight=highlight,
                    note=note,
                    color=current_color,
                    page=pending_page,
                    date_text=pending_date,
                )
            )
        pending_highlight = []
        pending_note = []
        pending_date = ""
        pending_page = None

    in_annotations = False
    for paragraph in paragraphs:
        text = paragraph.text.strip()
        lower = text.lower()
        if lower == "annotations by color":
            in_annotations = True
            continue
        if not in_annotations:
            continue
        if lower in COLOR_HEADINGS:
            flush()
            current_color = lower
            continue
        if lower.startswith("created by ") or lower.endswith(" notes"):
            continue
        fill_colors = [
            HIGHLIGHT_FILLS[fill] for fill in paragraph.fills if fill in HIGHLIGHT_FILLS
        ]
        if fill_colors:
            if pending_highlight and (pending_date or pending_page is not None):
                flush()
            elif pending_highlight and not pending_note:
                flush()
            pending_highlight.append(text)
            if not current_color:
                current_color = fill_colors[0]
            continue
        if DATE_RE.match(text) and DATE_COLOR in paragraph.colors:
            pending_date = text
            continue
        if text.isdigit() and PAGE_LINK_COLOR in paragraph.colors:
            pending_page = int(text)
            flush()
            continue
        if pending_highlight and NOTE_TEXT_COLOR in paragraph.colors:
            pending_note.append(text)
            continue
    flush()
    return NoteStats(
        title=title,
        author=author,
        source_paths=[path] if path else [],
        entries=entries,
    )


def merge_note_stats(note_stats: list[NoteStats]) -> list[NoteStats]:
    merged: list[NoteStats] = []
    for stats in note_stats:
        match = next(
            (item for item in merged if titles_match(item.title, stats.title)), None
        )
        if match is None:
            merged.append(stats)
            continue
        seen = {
            (normalize_loose(entry.highlight), normalize_loose(entry.note))
            for entry in match.entries
        }
        for entry in stats.entries:
            key = (normalize_loose(entry.highlight), normalize_loose(entry.note))
            if key not in seen:
                match.entries.append(entry)
                seen.add(key)
        match.source_paths.extend(stats.source_paths)
        if not match.author and stats.author:
            match.author = stats.author
    return merged


def load_all_note_stats(notes_dir: Path) -> list[NoteStats]:
    stats = []
    for path in sorted(notes_dir.glob("*.docx")):
        try:
            parsed = parse_note_stats(path)
        except Exception as exc:
            print(f"WARN: failed to parse notes {path.name}: {exc}")
            continue
        if parsed.entries:
            stats.append(parsed)
    return merge_note_stats(stats)


def load_metadata_records(notes: list[NoteStats]) -> list[MetadataRecord]:
    records: list[MetadataRecord] = []
    paths = [
        Path(__file__).resolve().parent / "golden_master_multi_source.csv",
        Path(__file__).resolve().parent / "book_reading_times.csv",
        Path(__file__).resolve().parent / "FINAL_CONSOLIDATED_MASTER.csv",
        Path(__file__).resolve().parent / "new_books_to_rate_2026_enriched.csv",
        Path(__file__).resolve().parent / "books_enriched_with_goodreads.csv",
    ]
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            title = str(row.get("title", "") or row.get("original_title", "")).strip()
            if not title or title.lower() == "nan":
                continue
            author = str(
                row.get("author", "")
                or row.get("corrected_author", "")
                or row.get("goodreads_author", "")
            ).strip()
            category = str(
                row.get("category", "")
                or row.get("final_category", "")
                or row.get("Bookshelf", "")
                or row.get("bookshelf", "")
            ).strip()
            page_count = first_number(
                row.get("page_count"),
                row.get("page_count_num"),
                row.get("gb_page_count"),
            )
            corrected = PAGE_COUNT_CORRECTIONS.get(title)
            if corrected:
                page_count = corrected
            records.append(
                MetadataRecord(title, author, category, page_count, path.name)
            )
    return records


def first_number(*values: object) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            return number
    return None


def find_best_title_match(title: str, candidates: Iterable[str]) -> str | None:
    candidate_list = list(candidates)
    narrowed = filter_candidate_titles(title, candidate_list)
    scored = [
        (title_match_score(title, candidate), candidate) for candidate in narrowed
    ]
    scored = [(score, candidate) for score, candidate in scored if score >= 65]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(normalize_title(item[1]))))
    return scored[0][1]


def filter_candidate_titles(
    ref: str, candidates: list[str], *, fallback_to_all: bool = True
) -> list[str]:
    ref_norm = normalize_title(ref)
    if not ref_norm:
        return []
    if is_abbreviation(ref_norm):
        narrowed = [
            title
            for title in candidates
            if normalize_title(title) == ref_norm
            or abbreviation_matches_title(ref_norm, title)
            or normalize_title(title).startswith(ref_norm + " ")
        ]
        return narrowed or (candidates if fallback_to_all else [])

    ref_words = set(significant_words(ref_norm))
    narrowed = []
    for title in candidates:
        title_norm = normalize_title(title)
        if ref_norm == title_norm or ref_norm in title_norm or title_norm in ref_norm:
            narrowed.append(title)
            continue
        title_words = set(significant_words(title_norm))
        if not ref_words or not title_words:
            continue
        overlap = len(ref_words & title_words)
        if overlap and overlap / min(len(ref_words), len(title_words)) >= 0.5:
            narrowed.append(title)
    return narrowed or (candidates if fallback_to_all else [])


def resolve_title(ref: str, known_titles: list[str]) -> str:
    cleaned = clean_book_ref(ref)
    norm = normalize_title(cleaned)
    if norm in TITLE_REWRITES:
        cleaned = TITLE_REWRITES[norm]
    if norm in MANUAL_ABBREVS:
        cleaned = MANUAL_ABBREVS[norm]
    best = find_best_title_match(cleaned, known_titles)
    if best:
        return canonicalize_title(best)
    if is_abbreviation(cleaned):
        abbrev_matches = [
            title
            for title in known_titles
            if abbreviation_matches_title(cleaned, title)
        ]
        if abbrev_matches:
            abbrev_matches.sort(key=lambda title: len(normalize_title(title)))
            return canonicalize_title(abbrev_matches[0])
    return canonicalize_title(cleaned)


def build_known_titles(
    events_df: pd.DataFrame, notes: list[NoteStats], metadata: list[MetadataRecord]
) -> list[str]:
    titles: list[str] = []
    titles.extend(record.title for record in metadata)
    titles.extend(stats.title for stats in notes)
    if not events_df.empty:
        refs = events_df.loc[events_df["book_ref"].astype(bool), "book_ref"].dropna()
        for ref in refs:
            if not is_abbreviation(ref):
                titles.append(str(ref))
    seen: set[str] = set()
    deduped: list[str] = []
    for title in titles:
        norm = normalize_title(title)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(title)
    return deduped


def build_finish_timeline(
    events_df: pd.DataFrame, known_titles: list[str]
) -> pd.DataFrame:
    finished = events_df[events_df["event_type"] == "finished"].copy()
    if finished.empty:
        return pd.DataFrame(
            columns=[
                "finish_id",
                "finish_date",
                "cal_ref",
                "title",
                "calendar_name",
                "read_instance",
                "is_reread",
                "previous_finish_date",
                "days_since_previous_finish",
            ]
        )
    rows: list[dict[str, object]] = []
    for _, row in finished.sort_values(["date", "start"]).iterrows():
        ref = clean_book_ref(row["book_ref"], row["date"])
        if not ref:
            continue
        title = resolve_title(ref, known_titles)
        if normalize_title(title) in {"split peas"}:
            continue
        duplicate = False
        for existing in rows:
            existing_date = existing["finish_date"]
            if (
                isinstance(existing_date, date)
                and abs((row["date"] - existing_date).days) <= 1
                and titles_match(title, existing["title"])
            ):
                duplicate = True
                break
        if duplicate:
            continue
        rows.append(
            {
                "finish_date": row["date"],
                "cal_ref": ref,
                "title": title,
                "calendar_name": row["calendar_name"],
            }
        )
    rows = sorted(rows, key=lambda item: item["finish_date"])
    previous_by_title: dict[str, date] = {}
    read_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        norm = normalize_title(row["title"])
        previous_date = previous_by_title.get(norm)
        read_counts[norm] += 1
        row["finish_id"] = f"finish_{index + 1:04d}"
        row["read_instance"] = read_counts[norm]
        row["is_reread"] = read_counts[norm] > 1
        row["previous_finish_date"] = previous_date if previous_date else pd.NaT
        row["days_since_previous_finish"] = (
            (row["finish_date"] - previous_date).days if previous_date else np.nan
        )
        previous_by_title[norm] = row["finish_date"]
    return pd.DataFrame(rows).sort_values("finish_date").reset_index(drop=True)


def resolve_reading_events(
    events_df: pd.DataFrame, finishes: pd.DataFrame
) -> pd.DataFrame:
    reading = events_df[
        events_df["event_type"].isin(["reading", "audiobook", "started", "generic"])
    ].copy()
    if reading.empty or finishes.empty:
        reading["title"] = None
        reading["finish_id"] = None
        reading["matched_finish_date"] = pd.NaT
        reading["match_method"] = None
        return reading
    finish_rows = finishes.sort_values("finish_date").to_dict("records")
    finish_titles = [str(finish["title"]) for finish in finish_rows]
    finish_refs = [str(finish["cal_ref"]) for finish in finish_rows]
    possible_by_ref: dict[str, list[dict[str, object]]] = {}

    def possible_finishes_for_ref(ref: str) -> list[dict[str, object]]:
        ref_norm = normalize_title(ref)
        if not ref_norm or ref_norm in GENERIC_BOOK_REFS:
            return []
        narrowed_titles = set(
            filter_candidate_titles(ref, finish_titles, fallback_to_all=False)
        )
        narrowed_refs = set(
            filter_candidate_titles(ref, finish_refs, fallback_to_all=False)
        )
        fragment_titles: set[str] = set()
        fragment_refs: set[str] = set()
        if is_abbreviation(ref_norm) and len(ref_norm) >= 5:
            fragment_titles = {
                title for title in finish_titles if ref_norm in significant_words(title)
            }
            fragment_refs = {
                cal_ref
                for cal_ref in finish_refs
                if ref_norm in significant_words(cal_ref)
            }
        possible: list[dict[str, object]] = []
        for finish in finish_rows:
            title = str(finish["title"])
            cal_ref = str(finish["cal_ref"])
            manual_target = MANUAL_ABBREVS.get(ref_norm)
            if manual_target and (
                titles_match(manual_target, title)
                or titles_match(manual_target, cal_ref)
            ):
                possible.append(finish)
                continue
            if ref_matches_title(ref, title) or ref_matches_title(ref, cal_ref):
                possible.append(finish)
                continue
            if title in narrowed_titles or cal_ref in narrowed_refs:
                possible.append(finish)
                continue
            if title in fragment_titles or cal_ref in fragment_refs:
                possible.append(finish)
                continue
            if title not in narrowed_titles and cal_ref not in narrowed_refs:
                if normalize_title(ref) not in {
                    normalize_title(title),
                    normalize_title(cal_ref),
                }:
                    continue
        return possible

    for raw_ref in sorted(set(reading["book_ref"].fillna("").map(str))):
        ref = clean_book_ref(raw_ref)
        possible_by_ref[normalize_title(ref)] = possible_finishes_for_ref(ref)

    def resolve_row(
        row: pd.Series,
    ) -> tuple[str | None, str | None, object, str | None]:
        ref = clean_book_ref(row["book_ref"], row["date"])
        event_date = row["date"]
        if normalize_title(ref) in GENERIC_BOOK_REFS or row["event_type"] == "generic":
            candidates = [
                finish
                for finish in finish_rows
                if finish["finish_date"] >= event_date
                and (finish["finish_date"] - event_date).days <= 45
            ]
            if candidates:
                return (
                    candidates[0]["title"],
                    candidates[0]["finish_id"],
                    candidates[0]["finish_date"],
                    "generic_temporal",
                )
            return None, None, pd.NaT, None
        candidates: list[tuple[int, str, str, object, str]] = []
        ref_key = normalize_title(ref)
        if ref_key not in possible_by_ref:
            possible_by_ref[ref_key] = possible_finishes_for_ref(ref)
        for finish in possible_by_ref.get(ref_key, []):
            title = str(finish["title"])
            if event_date > finish["finish_date"] + timedelta(days=2):
                continue
            distance = abs((finish["finish_date"] - event_date).days)
            candidates.append(
                (
                    distance,
                    title,
                    finish["finish_id"],
                    finish["finish_date"],
                    "ref_to_finished_title",
                )
            )
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return (
                candidates[0][1],
                candidates[0][2],
                candidates[0][3],
                candidates[0][4],
            )
        return None, None, pd.NaT, None

    resolved = reading.apply(resolve_row, axis=1, result_type="expand")
    reading["title"] = resolved[0]
    reading["finish_id"] = resolved[1]
    reading["matched_finish_date"] = resolved[2]
    reading["match_method"] = resolved[3]
    return reading


def find_metadata_record(
    title: str, records: list[MetadataRecord]
) -> MetadataRecord | None:
    candidate_titles = filter_candidate_titles(
        title, [record.title for record in records], fallback_to_all=False
    )
    candidate_norms = {normalize_title(candidate) for candidate in candidate_titles}
    scored: list[tuple[float, MetadataRecord]] = []
    for record in records:
        if normalize_title(record.title) not in candidate_norms:
            continue
        score = title_match_score(title, record.title)
        if score >= 65:
            scored.append((score, record))
    if not scored:
        return None
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].page_count is None,
            item[1].source == "notes_max_page",
            len(normalize_title(item[1].title)),
        )
    )
    return scored[0][1]


def find_note_stats(title: str, notes: list[NoteStats]) -> NoteStats | None:
    candidate_titles = filter_candidate_titles(
        title, [stats.title for stats in notes], fallback_to_all=False
    )
    candidate_norms = {normalize_title(candidate) for candidate in candidate_titles}
    scored = [
        (title_match_score(title, stats.title), stats)
        for stats in notes
        if normalize_title(stats.title) in candidate_norms
    ]
    scored = [(score, stats) for score, stats in scored if score >= 65]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], len(normalize_title(item[1].title))))
    return scored[0][1]


def aggregate_books(
    finishes: pd.DataFrame,
    resolved_events: pd.DataFrame,
    metadata_records: list[MetadataRecord],
    notes: list[NoteStats],
    *,
    words_per_page: int,
    audiobook_wpm: int,
) -> pd.DataFrame:
    aggregates: list[dict[str, object]] = []
    matched = resolved_events[resolved_events["title"].notna()].copy()
    for _, finish in finishes.iterrows():
        title = str(finish["title"])
        finish_id = str(finish["finish_id"])
        book_events = (
            matched[matched["finish_id"].astype(str) == finish_id]
            if not matched.empty and "finish_id" in matched
            else matched
        )
        metadata = find_metadata_record(title, metadata_records)
        note_stats = find_note_stats(title, notes)
        page_count = metadata.page_count if metadata else None
        page_source = metadata.source if metadata and metadata.page_count else ""
        total_hours = (
            float(book_events["duration_hours"].sum()) if not book_events.empty else 0.0
        )
        audiobook_hours = (
            float(
                book_events.loc[
                    book_events["event_type"] == "audiobook", "duration_hours"
                ].sum()
            )
            if not book_events.empty
            else 0.0
        )
        print_hours = total_hours - audiobook_hours
        total_minutes = total_hours * 60
        print_minutes = print_hours * 60
        audiobook_minutes = audiobook_hours * 60
        estimated_words = (
            page_count * words_per_page if page_count and total_minutes > 0 else np.nan
        )
        observed_total_wpm = (
            estimated_words / total_minutes
            if total_minutes > 0 and page_count
            else np.nan
        )
        audiobook_words_assumed = (
            audiobook_minutes * audiobook_wpm
            if page_count and total_hours > 0
            else np.nan
        )
        print_words_after_audio = (
            estimated_words - audiobook_words_assumed
            if page_count and total_hours > 0
            else np.nan
        )
        audio_words_exceed_estimate = bool(
            page_count and audiobook_minutes > 0 and print_words_after_audio < 0
        )
        if not page_count or total_minutes <= 0:
            wpm = np.nan
            wpm_basis = "missing_pages_or_time"
        elif audiobook_minutes <= 0:
            wpm = observed_total_wpm
            wpm_basis = "pages_over_total_calendar_time"
        elif print_minutes <= 0:
            wpm = float(audiobook_wpm)
            wpm_basis = "audiobook_assumption"
        elif audio_words_exceed_estimate:
            wpm = np.nan
            wpm_basis = "audiobook_words_exceed_estimated_book_words"
        else:
            wpm = print_words_after_audio / print_minutes
            wpm_basis = "print_words_after_audiobook_assumption"
        aggregates.append(
            {
                "finish_id": finish_id,
                "title": title,
                "finish_date": finish["finish_date"],
                "calendar_finish_ref": finish["cal_ref"],
                "read_instance": finish["read_instance"],
                "is_reread": finish["is_reread"],
                "previous_finish_date": finish["previous_finish_date"],
                "days_since_previous_finish": finish["days_since_previous_finish"],
                "category": metadata.category if metadata else "",
                "author": (
                    metadata.author
                    if metadata
                    else (note_stats.author if note_stats else "")
                ),
                "total_reading_hours": total_hours if total_hours > 0 else np.nan,
                "total_reading_minutes": total_minutes if total_hours > 0 else np.nan,
                "print_reading_hours": print_hours if total_hours > 0 else np.nan,
                "audiobook_hours": audiobook_hours if total_hours > 0 else np.nan,
                "audiobook_share": (
                    audiobook_hours / total_hours if total_hours > 0 else np.nan
                ),
                "audiobook_wpm_assumed": (
                    audiobook_wpm if audiobook_hours > 0 and page_count else np.nan
                ),
                "n_sessions": int(len(book_events)),
                "first_session": (
                    book_events["date"].min() if not book_events.empty else pd.NaT
                ),
                "last_session": (
                    book_events["date"].max() if not book_events.empty else pd.NaT
                ),
                "pages": page_count,
                "page_source": page_source,
                "words_per_page_assumed": words_per_page if page_count else np.nan,
                "estimated_words": estimated_words,
                "observed_total_wpm": observed_total_wpm,
                "audiobook_words_assumed": audiobook_words_assumed,
                "print_words_after_audiobook_assumption": print_words_after_audio,
                "audio_words_exceed_estimated_words": audio_words_exceed_estimate,
                "wpm": wpm,
                "wpm_basis": wpm_basis,
                "highlighted_words": note_stats.highlight_words if note_stats else 0,
                "my_note_words": note_stats.note_words if note_stats else 0,
                "highlight_count": note_stats.highlight_count if note_stats else 0,
                "my_note_count": note_stats.note_count if note_stats else 0,
                "notes_max_page": note_stats.max_page if note_stats else np.nan,
                "notes_title": note_stats.title if note_stats else "",
                "notes_sources": (
                    "; ".join(str(path.name) for path in note_stats.source_paths)
                    if note_stats
                    else ""
                ),
                "time_match_methods": (
                    ", ".join(sorted(set(book_events["match_method"].dropna())))
                    if not book_events.empty
                    else ""
                ),
            }
        )
    return pd.DataFrame(aggregates).sort_values("finish_date").reset_index(drop=True)


def confidence_value(label: str) -> int:
    return CONFIDENCE_ORDER.get(str(label), 0)


def min_confidence(*labels: str) -> str:
    if not labels:
        return "low"
    return min(labels, key=confidence_value)


def page_quality_override(title: str) -> tuple[str, str] | None:
    title_norm = normalize_title(title)
    if title_norm in PAGE_QUALITY_OVERRIDES:
        return PAGE_QUALITY_OVERRIDES[title_norm]
    for key, value in PAGE_QUALITY_OVERRIDES.items():
        if titles_match(key, title_norm):
            return value
    return None


def assess_word_count_confidence(row: pd.Series) -> tuple[str, str]:
    pages = row.get("pages")
    if pd.isna(pages) or float(pages) <= 0:
        return "low", "missing page/word-count metadata"
    override = page_quality_override(str(row.get("title", "")))
    if override:
        return override
    notes_max_page = row.get("notes_max_page")
    if pd.notna(notes_max_page) and float(notes_max_page) > float(pages) * 1.25:
        return "low", "Play Books max highlighted page exceeds metadata pages by >25%"
    if pd.notna(notes_max_page) and float(notes_max_page) > float(pages) * 1.10:
        return (
            "medium",
            "Play Books max highlighted page modestly exceeds metadata pages",
        )
    return "high", "metadata page count present and no local contradiction found"


def assess_calendar_confidence(row: pd.Series) -> tuple[str, str]:
    total_hours = row.get("total_reading_hours")
    if pd.isna(total_hours) or float(total_hours) <= 0:
        return "low", "no matched calendar reading time"
    reasons: list[str] = []
    confidence = "high"
    methods = {
        method.strip()
        for method in str(row.get("time_match_methods", "")).split(",")
        if method.strip()
    }
    if not methods:
        confidence = "low"
        reasons.append("no explicit match method")
    elif methods == {"generic_temporal"}:
        confidence = "low"
        reasons.append("only generic temporal calendar matching")
    elif "generic_temporal" in methods:
        confidence = min_confidence(confidence, "medium")
        reasons.append("some generic temporal calendar matching")
    if row.get("n_sessions", 0) <= 1:
        confidence = min_confidence(confidence, "medium")
        reasons.append("only one matched session")
    wpm = row.get("wpm")
    if pd.notna(wpm):
        if float(wpm) > 900:
            confidence = "low"
            reasons.append("implausibly high WPM suggests missing sessions")
        elif float(wpm) > 650:
            confidence = min_confidence(confidence, "medium")
            reasons.append("high WPM suggests possible missing sessions")
    if not reasons:
        reasons.append("specific calendar matches and plausible speed")
    return confidence, "; ".join(reasons)


def add_analysis_columns(books: pd.DataFrame) -> pd.DataFrame:
    frame = books.copy()
    word_assessments = frame.apply(assess_word_count_confidence, axis=1)
    calendar_assessments = frame.apply(assess_calendar_confidence, axis=1)
    frame["word_count_confidence"] = [item[0] for item in word_assessments]
    frame["word_count_confidence_reason"] = [item[1] for item in word_assessments]
    frame["calendar_confidence"] = [item[0] for item in calendar_assessments]
    frame["calendar_confidence_reason"] = [item[1] for item in calendar_assessments]
    frame["speed_confidence"] = [
        min_confidence(word, calendar)
        for word, calendar in zip(
            frame["word_count_confidence"], frame["calendar_confidence"]
        )
    ]
    frame["high_confidence_speed"] = (
        frame["speed_confidence"].eq("high") & frame["wpm"].notna()
    )
    frame["high_confidence_first_read"] = frame["high_confidence_speed"] & ~frame[
        "is_reread"
    ].fillna(False).astype(bool)
    frame["reliable_speed"] = (
        frame["speed_confidence"].isin(["high", "medium"]) & frame["wpm"].notna()
    )
    frame["reliable_first_read"] = frame["reliable_speed"] & ~frame["is_reread"].fillna(
        False
    ).astype(bool)
    frame["estimated_word_count"] = frame["estimated_words"]
    frame["highlight_words_per_10k_words"] = (
        frame["highlighted_words"] / frame["estimated_word_count"] * 10000
    )
    frame["note_words_per_10k_words"] = (
        frame["my_note_words"] / frame["estimated_word_count"] * 10000
    )
    frame["highlights_per_100_pages"] = frame["highlight_count"] / frame["pages"] * 100
    frame.loc[frame["estimated_word_count"].isna(), "highlight_words_per_10k_words"] = (
        np.nan
    )
    frame.loc[frame["estimated_word_count"].isna(), "note_words_per_10k_words"] = np.nan
    frame.loc[frame["pages"].isna(), "highlights_per_100_pages"] = np.nan
    return frame


def build_percentile_table(books: pd.DataFrame) -> pd.DataFrame:
    speed = books["wpm"].dropna()
    rows: list[dict[str, object]] = []
    if speed.empty:
        return pd.DataFrame(columns=["percentile", "wpm", "nearest_book"])
    for percentile in range(5, 100, 5):
        value = float(np.percentile(speed, percentile))
        nearest_idx = (books["wpm"] - value).abs().idxmin()
        rows.append(
            {
                "percentile": f"p{percentile}",
                "wpm": value,
                "nearest_book": books.loc[nearest_idx, "title"],
            }
        )
    return pd.DataFrame(rows)


def build_time_pages_outliers(books: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "title",
        "finish_date",
        "pages",
        "total_reading_hours",
        "print_reading_hours",
        "audiobook_hours",
        "wpm",
        "observed_total_wpm",
        "pages_per_hour",
        "time_pages_expected_hours",
        "time_pages_ratio_vs_expected",
        "time_pages_residual_log_hours",
        "time_pages_outlier_direction",
        "page_source",
        "wpm_basis",
    ]
    if books.empty:
        return pd.DataFrame(columns=columns)
    frame = books.dropna(subset=["pages", "total_reading_hours"]).copy()
    frame = frame[(frame["pages"] > 0) & (frame["total_reading_hours"] > 0)]
    if len(frame) < 3:
        return pd.DataFrame(columns=columns)

    log_pages = np.log(frame["pages"].astype(float))
    log_hours = np.log(frame["total_reading_hours"].astype(float))
    slope, intercept = np.polyfit(log_pages, log_hours, 1)
    expected_log_hours = intercept + slope * log_pages
    frame["time_pages_expected_hours"] = np.exp(expected_log_hours)
    frame["time_pages_residual_log_hours"] = log_hours - expected_log_hours
    frame["time_pages_ratio_vs_expected"] = (
        frame["total_reading_hours"] / frame["time_pages_expected_hours"]
    )
    frame["pages_per_hour"] = frame["pages"] / frame["total_reading_hours"]
    frame["time_pages_outlier_direction"] = np.where(
        frame["time_pages_residual_log_hours"] < 0,
        "faster_than_page_count_predicts",
        "slower_than_page_count_predicts",
    )
    frame["_abs_residual"] = frame["time_pages_residual_log_hours"].abs()
    return (
        frame.sort_values("_abs_residual", ascending=False)
        .drop(columns=["_abs_residual"])
        .loc[:, columns]
        .reset_index(drop=True)
    )


def build_reliable_percentile_table(books: pd.DataFrame) -> pd.DataFrame:
    reliable = books[books["reliable_first_read"]].copy()
    return build_percentile_table(reliable)


def build_highlight_effect_tables(
    books: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = books[
        books["reliable_first_read"]
        & books["wpm"].notna()
        & books["highlight_words_per_10k_words"].notna()
    ].copy()
    summary_columns = [
        "n_books",
        "pearson_corr_log_highlight_density_wpm",
        "slope_wpm_per_log1p_highlight_density",
        "predicted_wpm_at_p25_highlight_density",
        "predicted_wpm_at_p75_highlight_density",
        "observational_wpm_change_if_p75_to_p25_density",
        "p25_highlight_words_per_10k_words",
        "p75_highlight_words_per_10k_words",
        "median_wpm",
    ]
    if len(analysis) < 3:
        return pd.DataFrame(columns=summary_columns), pd.DataFrame()

    x = np.log1p(analysis["highlight_words_per_10k_words"].astype(float))
    y = analysis["wpm"].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.nunique() > 1 else np.nan
    p25 = float(np.percentile(analysis["highlight_words_per_10k_words"], 25))
    p75 = float(np.percentile(analysis["highlight_words_per_10k_words"], 75))
    predicted_p25 = float(intercept + slope * np.log1p(p25))
    predicted_p75 = float(intercept + slope * np.log1p(p75))
    summary = pd.DataFrame(
        [
            {
                "n_books": len(analysis),
                "pearson_corr_log_highlight_density_wpm": corr,
                "slope_wpm_per_log1p_highlight_density": float(slope),
                "predicted_wpm_at_p25_highlight_density": predicted_p25,
                "predicted_wpm_at_p75_highlight_density": predicted_p75,
                "observational_wpm_change_if_p75_to_p25_density": (
                    predicted_p25 - predicted_p75
                ),
                "p25_highlight_words_per_10k_words": p25,
                "p75_highlight_words_per_10k_words": p75,
                "median_wpm": float(y.median()),
            }
        ]
    )

    bins = analysis.copy()
    bins["highlight_density_bin"] = "no_highlights"
    positive = bins["highlight_words_per_10k_words"] > 0
    if positive.any():
        positive_values = bins.loc[positive, "highlight_words_per_10k_words"]
        try:
            positive_bins = pd.qcut(
                positive_values,
                q=min(3, positive_values.nunique()),
                labels=["low_positive", "medium_positive", "high_positive"][
                    : min(3, positive_values.nunique())
                ],
                duplicates="drop",
            )
            bins.loc[positive, "highlight_density_bin"] = positive_bins.astype(str)
        except ValueError:
            bins.loc[positive, "highlight_density_bin"] = "positive"
    grouped = (
        bins.groupby("highlight_density_bin", observed=False)
        .agg(
            n_books=("title", "count"),
            median_wpm=("wpm", "median"),
            mean_wpm=("wpm", "mean"),
            median_highlight_words_per_10k_words=(
                "highlight_words_per_10k_words",
                "median",
            ),
            median_highlighted_words=("highlighted_words", "median"),
            median_my_note_words=("my_note_words", "median"),
        )
        .reset_index()
    )
    grouped["highlight_density_bin"] = grouped["highlight_density_bin"].astype(str)
    return summary, grouped


def build_reliable_group_summaries(
    books: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reliable = books[books["reliable_first_read"] & books["wpm"].notna()].copy()
    if reliable.empty:
        empty_group = pd.DataFrame(
            columns=["group", "n_books", "median_wpm", "mean_wpm"]
        )
        empty_trend = pd.DataFrame(
            columns=["n_years", "n_books", "slope_median_wpm_per_year"]
        )
        return empty_group, empty_group, empty_trend
    reliable["finish_year"] = pd.to_datetime(
        reliable["finish_date"], errors="coerce"
    ).dt.year
    by_year = (
        reliable.dropna(subset=["finish_year"])
        .groupby("finish_year")
        .agg(
            n_books=("title", "count"),
            median_wpm=("wpm", "median"),
            mean_wpm=("wpm", "mean"),
            p25_wpm=("wpm", lambda values: float(np.percentile(values, 25))),
            p75_wpm=("wpm", lambda values: float(np.percentile(values, 75))),
        )
        .reset_index()
        .sort_values("finish_year")
    )
    category_frame = reliable.copy()
    category_frame["category"] = category_frame["category"].replace("", np.nan)
    category_frame["category"] = category_frame["category"].fillna("Unknown")
    by_category = (
        category_frame.groupby("category")
        .agg(
            n_books=("title", "count"),
            median_wpm=("wpm", "median"),
            mean_wpm=("wpm", "mean"),
            p25_wpm=("wpm", lambda values: float(np.percentile(values, 25))),
            p75_wpm=("wpm", lambda values: float(np.percentile(values, 75))),
        )
        .reset_index()
        .sort_values(["n_books", "median_wpm"], ascending=[False, False])
    )
    if len(by_year) >= 2:
        slope, intercept = np.polyfit(
            by_year["finish_year"].astype(float), by_year["median_wpm"], 1
        )
    else:
        slope, intercept = np.nan, np.nan
    trend = pd.DataFrame(
        [
            {
                "n_years": len(by_year),
                "n_books": len(reliable),
                "slope_median_wpm_per_year": float(slope),
                "intercept": float(intercept),
                "first_year": int(by_year["finish_year"].min()),
                "last_year": int(by_year["finish_year"].max()),
                "first_year_median_wpm": float(by_year.iloc[0]["median_wpm"]),
                "last_year_median_wpm": float(by_year.iloc[-1]["median_wpm"]),
            }
        ]
    )
    return by_year, by_category, trend


def write_markdown_table(
    df: pd.DataFrame, path: Path, *, max_rows: int | None = None
) -> None:
    frame = df.copy()
    if max_rows:
        frame = frame.head(max_rows)
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = [_markdown_cell(row[column]) for column in frame.columns]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.2f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def plot_violin_by_group(
    frame: pd.DataFrame,
    group_col: str,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    min_group_size: int = 2,
) -> None:
    plot_df = frame.dropna(subset=["wpm", group_col]).copy()
    if plot_df.empty:
        return
    groups = [
        (str(group), values["wpm"].astype(float).to_numpy())
        for group, values in plot_df.groupby(group_col)
        if len(values) >= min_group_size
    ]
    if not groups:
        return
    groups.sort(key=lambda item: item[0])
    labels = [item[0] for item in groups]
    values = [item[1] for item in groups]
    positions = np.arange(1, len(values) + 1)
    plt.figure(figsize=(max(10, len(values) * 0.75), 6))
    violin = plt.violinplot(values, positions=positions, showmedians=True)
    for body in violin["bodies"]:
        body.set_alpha(0.55)
    rng = np.random.default_rng(42)
    for pos, group_values in zip(positions, values):
        jitter = rng.normal(0, 0.035, len(group_values))
        plt.scatter(
            np.full(len(group_values), pos) + jitter,
            group_values,
            color="black",
            alpha=0.55,
            s=16,
            linewidths=0,
        )
    plt.xticks(positions, labels, rotation=35, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel("Estimated WPM")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_outputs(books: pd.DataFrame, output_dir: Path) -> None:
    reliable_df = books[books["reliable_first_read"]].copy()
    plot_df = books.dropna(subset=["wpm"]).copy()
    if not plot_df.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(
            plot_df["wpm"],
            bins=min(25, max(8, int(np.sqrt(len(plot_df))))),
            edgecolor="black",
        )
        plt.xlabel("Estimated words per minute")
        plt.ylabel("Books")
        plt.title("Estimated WPM per Finished Book")
        plt.tight_layout()
        plt.savefig(output_dir / "book_wpm_histogram.png", dpi=180)
        plt.close()

    scatter_df = books.dropna(subset=["wpm"]).copy()
    scatter_df = scatter_df[scatter_df["highlighted_words"] > 0]
    if not scatter_df.empty:
        plt.figure(figsize=(10, 6))
        note_words = scatter_df["my_note_words"].fillna(0)
        marker_sizes = 35 + np.log1p(note_words) * 18
        points = plt.scatter(
            scatter_df["highlighted_words"],
            scatter_df["wpm"],
            c=note_words,
            cmap="viridis",
            s=marker_sizes,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )
        plt.xscale("log")
        plt.xlabel("Highlighted words in Play Books notes (log scale)")
        plt.ylabel("Estimated words per minute")
        plt.title("WPM vs Highlighted Words")
        if note_words.max() > 0:
            colorbar = plt.colorbar(points)
            colorbar.set_label("User-note words")
        for _, row in scatter_df.nlargest(8, "highlighted_words").iterrows():
            plt.annotate(
                str(row["title"])[:28],
                (row["highlighted_words"], row["wpm"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        plt.tight_layout()
        plt.savefig(output_dir / "wpm_vs_highlighted_words.png", dpi=180)
        plt.close()

    time_df = books.dropna(subset=["pages", "total_reading_hours"]).copy()
    time_df = time_df[(time_df["pages"] > 0) & (time_df["total_reading_hours"] > 0)]
    if not time_df.empty:
        outliers = build_time_pages_outliers(books)
        plt.figure(figsize=(10, 6))
        color_values = time_df["wpm"].fillna(time_df["observed_total_wpm"])
        points = plt.scatter(
            time_df["pages"],
            time_df["total_reading_hours"],
            c=color_values,
            cmap="magma",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Estimated pages (log scale)")
        plt.ylabel("Calendar reading time, hours (log scale)")
        plt.title("Reading Time vs Estimated Pages")
        if color_values.notna().any():
            colorbar = plt.colorbar(points)
            colorbar.set_label("Estimated WPM")
        if not outliers.empty:
            expected = outliers.sort_values("pages")
            plt.plot(
                expected["pages"],
                expected["time_pages_expected_hours"],
                color="black",
                linewidth=1,
                alpha=0.5,
            )
            for _, row in outliers.head(12).iterrows():
                plt.annotate(
                    str(row["title"])[:28],
                    (row["pages"], row["total_reading_hours"]),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        plt.tight_layout()
        plt.savefig(output_dir / "time_vs_pages.png", dpi=180)
        plt.close()

    if not reliable_df.empty:
        reliable_df["finish_year"] = pd.to_datetime(
            reliable_df["finish_date"], errors="coerce"
        ).dt.year
        plot_violin_by_group(
            reliable_df,
            "finish_year",
            output_dir / "wpm_violin_by_year.png",
            title="Reliable First-Read WPM by Year",
            xlabel="Finish year",
            min_group_size=2,
        )
        category_df = reliable_df.copy()
        category_df["category_plot"] = category_df["category"].replace("", np.nan)
        category_df["category_plot"] = category_df["category_plot"].fillna("Unknown")
        category_counts = category_df["category_plot"].value_counts()
        category_df = category_df[
            category_df["category_plot"].isin(
                category_counts[category_counts >= 2].index
            )
        ]
        plot_violin_by_group(
            category_df,
            "category_plot",
            output_dir / "wpm_violin_by_category.png",
            title="Reliable First-Read WPM by Category",
            xlabel="Category",
            min_group_size=2,
        )

    highlight_df = reliable_df.dropna(
        subset=["wpm", "highlight_words_per_10k_words"]
    ).copy()
    if len(highlight_df) >= 3:
        x = np.log1p(highlight_df["highlight_words_per_10k_words"].astype(float))
        y = highlight_df["wpm"].astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            highlight_df["highlight_words_per_10k_words"],
            y,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )
        plt.plot(np.expm1(x_line), intercept + slope * x_line, color="black")
        plt.xscale("symlog", linthresh=1)
        plt.xlabel("Highlighted words per 10k estimated book words")
        plt.ylabel("Estimated WPM")
        plt.title("Highlight Density vs Reading Speed")
        for _, row in highlight_df.nlargest(
            8, "highlight_words_per_10k_words"
        ).iterrows():
            plt.annotate(
                str(row["title"])[:28],
                (row["highlight_words_per_10k_words"], row["wpm"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        plt.tight_layout()
        plt.savefig(output_dir / "highlight_density_vs_wpm.png", dpi=180)
        plt.close()


def write_summary(
    books: pd.DataFrame,
    events_df: pd.DataFrame,
    resolved_events: pd.DataFrame,
    notes: list[NoteStats],
    time_pages_outliers: pd.DataFrame,
    highlight_summary: pd.DataFrame,
    trend_summary: pd.DataFrame,
    output_dir: Path,
    *,
    words_per_page: int,
    audiobook_wpm: int,
) -> None:
    has_time = books["total_reading_hours"].notna()
    has_pages = books["pages"].notna()
    has_wpm = books["wpm"].notna()
    high_confidence = books[books["high_confidence_first_read"] & books["wpm"].notna()]
    reliable = books[books["reliable_first_read"] & books["wpm"].notna()]
    unmatched = resolved_events[
        resolved_events["event_type"].isin(["reading", "audiobook", "generic"])
        & resolved_events["title"].isna()
    ]
    top_unmatched = (
        unmatched.groupby("book_ref", dropna=False)["duration_hours"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        if not unmatched.empty
        else pd.Series(dtype=float)
    )
    lines = [
        "# Book WPM Calendar Analysis",
        "",
        f"- Finished-book rows: {len(books)}",
        f"- Books with matched reading time: {int(has_time.sum())}",
        f"- Books with page count estimate: {int(has_pages.sum())}",
        f"- Books with WPM estimate: {int(has_wpm.sum())}",
        f"- Reliable first-read WPM rows: {len(reliable)}",
        f"- Strict high-confidence first-read WPM rows: {len(high_confidence)}",
        f"- Parsed classified calendar rows: {len(events_df)}",
        f"- Parsed Play Books note files with highlights: {len(notes)}",
        f"- WPM assumption: pages * {words_per_page} words/page / reading minutes.",
        f"- Audiobook sessions consume words at {audiobook_wpm} WPM before estimating print-reading WPM.",
        "- `observed_total_wpm` keeps the raw pages/time rate; primary `wpm` is audiobook-adjusted when audio time is present.",
        "- Page counts come from existing book metadata / Google Books fields; Play Books max page is retained as `notes_max_page` audit data, not used as total pages.",
        "- Highlighted words count only highlighted passage text; `my_note_words` counts user-written notes attached to highlights.",
        "- Recurrences are expanded for common DAILY/WEEKLY/MONTHLY RRULEs; exception handling is approximate.",
        "",
    ]
    if has_wpm.any():
        lines.extend(
            [
                "## WPM Distribution",
                "",
                f"- Median WPM: {books.loc[has_wpm, 'wpm'].median():.1f}",
                f"- P5/P95 WPM: {np.percentile(books.loc[has_wpm, 'wpm'], 5):.1f} / {np.percentile(books.loc[has_wpm, 'wpm'], 95):.1f}",
                "",
            ]
        )
    if not reliable.empty:
        lines.extend(
            [
                "## Reliable First Reads",
                "",
                f"- Median WPM: {reliable['wpm'].median():.1f}",
                f"- P5/P95 WPM: {np.percentile(reliable['wpm'], 5):.1f} / {np.percentile(reliable['wpm'], 95):.1f}",
                f"- Years covered: {pd.to_datetime(reliable['finish_date']).dt.year.min()}-{pd.to_datetime(reliable['finish_date']).dt.year.max()}",
                "- Reliable means medium-or-high word-count confidence and medium-or-high calendar confidence, first-read only.",
                "",
            ]
        )
    if not trend_summary.empty:
        row = trend_summary.iloc[0]
        lines.extend(
            [
                "## Year Trend",
                "",
                f"- Median-WPM trend slope: {row['slope_median_wpm_per_year']:.1f} WPM/year.",
                f"- First/last reliable year medians: {row['first_year_median_wpm']:.1f} / {row['last_year_median_wpm']:.1f}.",
                "",
            ]
        )
    if not highlight_summary.empty:
        row = highlight_summary.iloc[0]
        lines.extend(
            [
                "## Highlighting Relationship",
                "",
                f"- Books in model: {int(row['n_books'])}",
                f"- Correlation between log highlight density and WPM: {row['pearson_corr_log_highlight_density_wpm']:.2f}",
                f"- Observed WPM change going from p75 to p25 highlight density: {row['observational_wpm_change_if_p75_to_p25_density']:.1f}",
                "- This is observational correlation, not a causal estimate.",
                "",
            ]
        )
    if not time_pages_outliers.empty:
        lines.extend(["## Time vs Pages Outliers", ""])
        for _, row in time_pages_outliers.head(10).iterrows():
            lines.append(
                f"- {row['title']}: {row['pages']:.0f} pages, "
                f"{row['total_reading_hours']:.1f}h, "
                f"{row['time_pages_ratio_vs_expected']:.2f}x expected time "
                f"({row['time_pages_outlier_direction']})"
            )
        lines.append("")
    if not top_unmatched.empty:
        lines.extend(["## Top Unmatched Calendar Refs", ""])
        for ref, hours in top_unmatched.items():
            display = ref if str(ref).strip() else "(generic)"
            lines.append(f"- {display}: {hours:.1f}h")
        lines.append("")
    output_dir.joinpath("book_wpm_analysis_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_analysis(
    *,
    calendar_dir: Path = DEFAULT_CALENDAR_DIR,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    words_per_page: int = DEFAULT_WORDS_PER_PAGE,
    audiobook_wpm: int = DEFAULT_AUDIOBOOK_WPM,
    analysis_end: date = DEFAULT_ANALYSIS_END,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_events = read_ics_events(calendar_dir, analysis_end=analysis_end)
    events_df = classify_calendar_events(raw_events)
    notes = load_all_note_stats(notes_dir)
    metadata_records = load_metadata_records(notes)
    known_titles = build_known_titles(events_df, notes, metadata_records)
    finishes = build_finish_timeline(events_df, known_titles)
    resolved_events = resolve_reading_events(events_df, finishes)
    books = aggregate_books(
        finishes,
        resolved_events,
        metadata_records,
        notes,
        words_per_page=words_per_page,
        audiobook_wpm=audiobook_wpm,
    )
    books = add_analysis_columns(books)
    percentiles = build_percentile_table(books)
    reliable_percentiles = build_reliable_percentile_table(books)
    time_pages_outliers = build_time_pages_outliers(books)
    highlight_summary, highlight_bins = build_highlight_effect_tables(books)
    by_year, by_category, trend_summary = build_reliable_group_summaries(books)

    events_df.to_csv(
        output_dir / "book_wpm_classified_calendar_events.csv", index=False
    )
    finishes.to_csv(output_dir / "book_wpm_finished_timeline.csv", index=False)
    resolved_events.to_csv(
        output_dir / "book_wpm_resolved_reading_events.csv", index=False
    )
    books.to_csv(output_dir / "book_wpm_by_book.csv", index=False)
    percentiles.to_csv(output_dir / "book_wpm_percentiles.csv", index=False)
    reliable_percentiles.to_csv(
        output_dir / "book_wpm_reliable_percentiles.csv", index=False
    )
    books[books["reliable_first_read"]].to_csv(
        output_dir / "book_wpm_reliable_subset.csv", index=False
    )
    books[books["high_confidence_first_read"]].to_csv(
        output_dir / "book_wpm_high_confidence_subset.csv", index=False
    )
    time_pages_outliers.to_csv(
        output_dir / "book_wpm_time_pages_outliers.csv", index=False
    )
    highlight_summary.to_csv(
        output_dir / "book_wpm_highlight_effect_summary.csv", index=False
    )
    highlight_bins.to_csv(output_dir / "book_wpm_highlight_bins.csv", index=False)
    by_year.to_csv(output_dir / "book_wpm_reliable_by_year.csv", index=False)
    by_category.to_csv(output_dir / "book_wpm_reliable_by_category.csv", index=False)
    trend_summary.to_csv(
        output_dir / "book_wpm_reliable_trend_summary.csv", index=False
    )
    write_markdown_table(
        books[
            [
                "finish_id",
                "title",
                "finish_date",
                "read_instance",
                "is_reread",
                "total_reading_hours",
                "print_reading_hours",
                "audiobook_hours",
                "pages",
                "estimated_word_count",
                "wpm",
                "observed_total_wpm",
                "speed_confidence",
                "reliable_first_read",
                "high_confidence_first_read",
                "word_count_confidence",
                "calendar_confidence",
                "wpm_basis",
                "highlighted_words",
                "highlight_words_per_10k_words",
                "my_note_words",
                "page_source",
                "time_match_methods",
            ]
        ].sort_values("finish_date", ascending=False),
        output_dir / "book_wpm_by_book.md",
    )
    write_markdown_table(percentiles, output_dir / "book_wpm_percentiles.md")
    write_markdown_table(
        reliable_percentiles, output_dir / "book_wpm_reliable_percentiles.md"
    )
    write_markdown_table(
        books[books["reliable_first_read"]].sort_values("finish_date", ascending=False),
        output_dir / "book_wpm_reliable_subset.md",
        max_rows=100,
    )
    write_markdown_table(
        books[books["high_confidence_first_read"]].sort_values(
            "finish_date", ascending=False
        ),
        output_dir / "book_wpm_high_confidence_subset.md",
        max_rows=100,
    )
    write_markdown_table(
        time_pages_outliers,
        output_dir / "book_wpm_time_pages_outliers.md",
        max_rows=50,
    )
    write_markdown_table(
        highlight_summary, output_dir / "book_wpm_highlight_effect_summary.md"
    )
    write_markdown_table(highlight_bins, output_dir / "book_wpm_highlight_bins.md")
    write_markdown_table(by_year, output_dir / "book_wpm_reliable_by_year.md")
    write_markdown_table(by_category, output_dir / "book_wpm_reliable_by_category.md")
    write_markdown_table(
        trend_summary, output_dir / "book_wpm_reliable_trend_summary.md"
    )
    plot_outputs(books, output_dir)
    write_summary(
        books,
        events_df,
        resolved_events,
        notes,
        time_pages_outliers,
        highlight_summary,
        trend_summary,
        output_dir,
        words_per_page=words_per_page,
        audiobook_wpm=audiobook_wpm,
    )
    return books


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-dir", type=Path, default=DEFAULT_CALENDAR_DIR)
    parser.add_argument("--notes-dir", type=Path, default=DEFAULT_NOTES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--words-per-page", type=int, default=DEFAULT_WORDS_PER_PAGE)
    parser.add_argument("--audiobook-wpm", type=int, default=DEFAULT_AUDIOBOOK_WPM)
    parser.add_argument(
        "--analysis-end",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=DEFAULT_ANALYSIS_END,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    books = run_analysis(
        calendar_dir=args.calendar_dir,
        notes_dir=args.notes_dir,
        output_dir=args.output_dir,
        words_per_page=args.words_per_page,
        audiobook_wpm=args.audiobook_wpm,
        analysis_end=args.analysis_end,
    )
    with pd.option_context("display.max_rows", 30, "display.width", 160):
        print(
            books[
                [
                    "title",
                    "finish_date",
                    "total_reading_hours",
                    "pages",
                    "wpm",
                    "speed_confidence",
                    "highlighted_words",
                    "my_note_words",
                ]
            ]
            .sort_values("finish_date", ascending=False)
            .head(30)
            .to_string(index=False)
        )
    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
