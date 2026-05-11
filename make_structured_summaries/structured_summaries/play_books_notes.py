from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


COLOR_HEADINGS = ("Yellow", "Green", "Blue", "Red")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DATE_RE = re.compile(rf"^({'|'.join(MONTHS)})\s+\d{{1,2}},\s+\d{{4}}$")
NOTE_HEADER_RE = re.compile(r'^Notes from ["_](.+?)["_]$')


@dataclass(frozen=True)
class BlueHighlightsSection:
    title: str
    author: str
    highlights: tuple[str, ...]
    source_path: Path | None = None


def _normalize_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    return [line.replace("\t", " ").strip() for line in text.split("\n")]


def _looks_like_metadata(line: str) -> bool:
    if not line:
        return True
    lowered = line.lower()
    if "cover image" in lowered:
        return True
    if DATE_RE.match(line):
        return True
    if line.isdigit():
        return True
    if re.fullmatch(r"\d+\s*[-–]\s*\d+", line):
        return True
    if lowered.startswith("created by "):
        return True
    if lowered.startswith("annotations by color"):
        return True
    if lowered.startswith("limit reached"):
        return True
    if lowered.endswith(" notes"):
        return True
    if lowered == "cover image":
        return True
    if lowered.startswith("out of "):
        return True
    return False


def _dedupe_preserve_order(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return tuple(deduped)


def merge_blue_highlights_sections(
    sections: list[BlueHighlightsSection],
) -> list[BlueHighlightsSection]:
    merged: dict[tuple[str, str], BlueHighlightsSection] = {}
    ordered_keys: list[tuple[str, str]] = []

    for section in sections:
        key = (section.title, section.author)
        if key not in merged:
            merged[key] = section
            ordered_keys.append(key)
            continue

        existing = merged[key]
        merged[key] = BlueHighlightsSection(
            title=existing.title,
            author=existing.author,
            highlights=_dedupe_preserve_order(
                list(existing.highlights) + list(section.highlights)
            ),
            source_path=existing.source_path or section.source_path,
        )

    return [merged[key] for key in ordered_keys]


def _extract_title_author(lines: list[str], fallback_name: str) -> tuple[str, str]:
    content_lines = [line for line in lines if line and not _looks_like_metadata(line)]
    if len(content_lines) >= 2:
        title = content_lines[0]
        author = content_lines[1]
        if title in {
            "This document is overwritten when you make changes in Play Books."
        }:
            title = fallback_name
            author = "Unknown Author"
        return title, author
    return fallback_name, "Unknown Author"


def _section_bounds(lines: list[str], heading: str) -> tuple[int, int] | None:
    start = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if line == heading:
            start = idx + 1
            continue
        if start is not None and line in COLOR_HEADINGS and line != heading:
            end = idx
            break
    if start is None:
        return None
    return start, end


def _parse_blue_entries(section_lines: list[str]) -> tuple[str, ...]:
    entries: list[str] = []
    current: list[str] = []
    in_metadata = False

    for raw_line in section_lines:
        line = raw_line.strip()
        if not line:
            continue
        if DATE_RE.match(line):
            if current:
                entries.append(" ".join(current).strip())
                current = []
            in_metadata = True
            continue
        if in_metadata:
            if _looks_like_metadata(line):
                continue
            in_metadata = False
        if _looks_like_metadata(line):
            continue
        current.append(line)

    if current:
        entries.append(" ".join(current).strip())

    return _dedupe_preserve_order(entries)


def parse_blue_highlights_text(
    text: str,
    *,
    source_path: Path | None = None,
    fallback_name: str | None = None,
) -> BlueHighlightsSection | None:
    lines = _normalize_lines(text)
    fallback = fallback_name or (source_path.stem if source_path else "Unknown Title")
    title, author = _extract_title_author(lines, fallback)
    bounds = _section_bounds(lines, "Blue")
    if bounds is None:
        return None
    start, end = bounds
    highlights = _parse_blue_entries(lines[start:end])
    if not highlights:
        return None
    return BlueHighlightsSection(
        title=title,
        author=author,
        highlights=highlights,
        source_path=source_path,
    )


def parse_blue_highlights_file(path: Path) -> BlueHighlightsSection | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    fallback_name = path.stem
    match = NOTE_HEADER_RE.match(path.stem)
    if match:
        fallback_name = match.group(1)
    return parse_blue_highlights_text(
        text,
        source_path=path,
        fallback_name=fallback_name,
    )


def render_blue_highlights_markdown(sections: list[BlueHighlightsSection]) -> str:
    lines = [
        "# Compiled Blue Highlights",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"## {section.title}",
                "",
                "### Author",
                "",
                section.author,
                "",
                "### Blue Highlights",
                "",
            ]
        )
        for highlight in section.highlights:
            lines.append(f"- {highlight}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
