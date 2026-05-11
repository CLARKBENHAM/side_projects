"""Highlight parsing, review prompts, and Anki export helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .llm_backends import run_prompt
from .prompts import build_alignment_review_prompt, build_anki_prompt
from .utils import normalize_text, parse_json_maybe

PLAY_BOOKS_METADATA_SUBSTRINGS = (
    "this document is overwritten when you make changes in play books",
    "you should make a copy of this document before you edit it",
    "annotations by color",
    "all your annotations",
    "created by ",
    "last synced ",
)
PLAY_BOOKS_COLOR_HEADINGS = {"yellow", "green", "blue", "red"}
PLAY_BOOKS_BOOKMARK_ONLY_MARKER = "bookmarks"
PLAY_BOOKS_COLOR_PRIORITY = {"blue": 4, "red": 3, "green": 2, "yellow": 1, "": 0}
DATE_PATTERN = re.compile(r"^[A-Za-z]+ \d{1,2}, \d{4}$")


@dataclass(frozen=True)
class HighlightEntry:
    text: str
    color: str = ""
    page: int | None = None
    date_text: str = ""
    section_order: int = 0
    progress: float | None = None
    estimated_chunk: int | None = None
    duplicate_count: int = 1
    source_section: str = ""


def _extract_play_books_annotation_body(text: str) -> str:
    lowered = text.lower()
    if (
        PLAY_BOOKS_BOOKMARK_ONLY_MARKER in lowered
        and "annotations by color" not in lowered
    ):
        return ""
    color_match = re.search(
        r"\n\s*(yellow|green|blue|red)\s*\n", text, flags=re.IGNORECASE
    )
    if color_match:
        return text[color_match.end() :]
    return text


def _should_drop_line(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return True
    if lowered in PLAY_BOOKS_COLOR_HEADINGS:
        return True
    if any(marker in lowered for marker in PLAY_BOOKS_METADATA_SUBSTRINGS):
        return True
    if lowered == "cover image":
        return True
    if re.fullmatch(r"[a-z]+ \d{1,2}, \d{4}", lowered):
        return True
    if re.fullmatch(r"\d+", lowered):
        return True
    return False


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = normalize_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _clean_highlight_block(block: str) -> str:
    lines = [
        re.sub(r"^\s*(?:[-*+]|\d+\.)\s*", "", line).strip()
        for line in block.splitlines()
    ]
    kept_lines = [line for line in lines if not _should_drop_line(line)]
    if not kept_lines:
        return ""
    combined = " ".join(kept_lines).strip()
    if len(combined) <= 3:
        return ""
    return combined


def _parse_play_books_segment(
    color: str,
    segment_text: str,
    *,
    source_section: str,
) -> list[HighlightEntry]:
    lines = [line.strip().lstrip("\ufeff") for line in segment_text.splitlines()]
    entries: list[HighlightEntry] = []
    pending_lines: list[str] = []
    last_page: int | None = None
    section_order = 0
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if DATE_PATTERN.fullmatch(line) and pending_lines:
            page: int | None = None
            lookahead = index + 1
            while lookahead < len(lines):
                candidate = lines[lookahead].strip()
                if not candidate:
                    lookahead += 1
                    continue
                if re.fullmatch(r"\d+", candidate):
                    page = int(candidate)
                    break
                if DATE_PATTERN.fullmatch(candidate):
                    break
                lookahead += 1
            text = " ".join(part for part in pending_lines if part).strip()
            if text:
                if page is not None and last_page is not None and page + 20 < last_page:
                    break
                section_order += 1
                entries.append(
                    HighlightEntry(
                        text=text,
                        color=color,
                        page=page,
                        date_text=line,
                        section_order=section_order,
                        source_section=source_section,
                    )
                )
                if page is not None:
                    last_page = page
            pending_lines = []
            index = lookahead + 1
            continue
        if line.lower() in PLAY_BOOKS_COLOR_HEADINGS:
            break
        if _should_drop_line(line):
            index += 1
            continue
        pending_lines.append(line)
        index += 1
    return entries


def _dedupe_entries(entries: list[HighlightEntry]) -> list[HighlightEntry]:
    by_key: dict[str, HighlightEntry] = {}
    for entry in entries:
        key = normalize_text(entry.text)
        if not key:
            continue
        current = by_key.get(key)
        if current is None:
            by_key[key] = entry
            continue
        preferred = current
        if PLAY_BOOKS_COLOR_PRIORITY.get(
            entry.color, 0
        ) > PLAY_BOOKS_COLOR_PRIORITY.get(current.color, 0):
            preferred = entry
        elif current.page is None and entry.page is not None:
            preferred = entry
        elif (
            current.page is not None
            and entry.page is not None
            and entry.page < current.page
        ):
            preferred = entry
        by_key[key] = replace(preferred, duplicate_count=current.duplicate_count + 1)
    return sorted(
        by_key.values(),
        key=lambda entry: (
            entry.page is None,
            entry.page or 10**9,
            -PLAY_BOOKS_COLOR_PRIORITY.get(entry.color, 0),
            entry.section_order,
        ),
    )


def parse_structured_highlights(text: str) -> list[HighlightEntry]:
    raw_text = text.lstrip("\ufeff")
    lowered = raw_text.lower()
    if (
        PLAY_BOOKS_BOOKMARK_ONLY_MARKER in lowered
        and "annotations by color" not in lowered
    ):
        return []

    heading_matches = list(
        re.finditer(r"(?m)^\s*(Yellow|Green|Blue|Red)\s*$", raw_text)
    )
    if not heading_matches:
        return []

    entries: list[HighlightEntry] = []
    for index, match in enumerate(heading_matches):
        color = match.group(1).lower()
        segment_start = match.end()
        segment_end = (
            heading_matches[index + 1].start()
            if index + 1 < len(heading_matches)
            else len(raw_text)
        )
        entries.extend(
            _parse_play_books_segment(
                color,
                raw_text[segment_start:segment_end],
                source_section="color_group",
            )
        )
    return _dedupe_entries(entries)


def annotate_entries_with_progress(
    entries: list[HighlightEntry],
    *,
    total_chunks: int | None = None,
) -> list[HighlightEntry]:
    pages = [entry.page for entry in entries if entry.page is not None]
    max_page = max(pages) if pages else None
    annotated: list[HighlightEntry] = []
    for entry in entries:
        progress = None
        estimated_chunk = None
        if entry.page is not None and max_page and max_page > 0:
            progress = min(max(entry.page / max_page, 0.0), 1.0)
            if total_chunks:
                estimated_chunk = min(
                    total_chunks,
                    (
                        int(progress * total_chunks) + 1
                        if progress < 1.0
                        else total_chunks
                    ),
                )
        annotated.append(
            replace(
                entry,
                progress=progress,
                estimated_chunk=estimated_chunk,
            )
        )
    return annotated


def split_highlights(text: str) -> list[str]:
    structured_entries = parse_structured_highlights(text)
    if structured_entries:
        return [entry.text for entry in structured_entries]

    text = _extract_play_books_annotation_body(text).strip()
    if not text:
        return []

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    highlights: list[str] = []
    for block in blocks:
        raw_lines = [line for line in block.splitlines() if line.strip()]
        if raw_lines and all(
            re.match(r"^\s*(?:[-*+]|>|\d+\.)\s+", line) for line in raw_lines
        ):
            for line in raw_lines:
                normalized = re.sub(r"^\s*(?:[-*+]|\d+\.)\s*", "", line)
                normalized = re.sub(r"^>\s*", "", normalized)
                normalized = normalized.strip()
                if (
                    normalized
                    and not normalized.startswith("#")
                    and not _should_drop_line(normalized)
                ):
                    highlights.append(normalized)
            continue
        if "\n" not in block:
            normalized = re.sub(r"^\s*(?:[-*+]|\d+\.)\s*", "", block)
            normalized = re.sub(r"^>\s*", "", normalized)
            normalized = normalized.strip()
            if (
                normalized
                and not normalized.startswith("#")
                and not _should_drop_line(normalized)
            ):
                highlights.append(normalized)
            continue

        combined = _clean_highlight_block(block)
        if combined:
            highlights.append(combined)
    return _dedupe_preserving_order(highlights)


def load_highlights(path: Path) -> list[str]:
    return split_highlights(path.read_text(encoding="utf-8", errors="ignore"))


def load_highlights_from_paths(paths: list[Path]) -> list[str]:
    combined: list[str] = []
    for path in paths:
        combined.extend(load_highlights(path))
    return _dedupe_preserving_order(combined)


def load_structured_highlights(
    path: Path, *, total_chunks: int | None = None
) -> list[HighlightEntry]:
    entries = parse_structured_highlights(
        path.read_text(encoding="utf-8", errors="ignore")
    )
    return annotate_entries_with_progress(entries, total_chunks=total_chunks)


def load_structured_highlights_from_paths(
    paths: list[Path],
    *,
    total_chunks: int | None = None,
) -> list[HighlightEntry]:
    combined: list[HighlightEntry] = []
    for path in paths:
        combined.extend(
            parse_structured_highlights(
                path.read_text(encoding="utf-8", errors="ignore")
            )
        )
    return annotate_entries_with_progress(
        _dedupe_entries(combined),
        total_chunks=total_chunks,
    )


def parse_anki_cards(
    raw_response: str, *, card_format: str = "qa"
) -> list[dict[str, str]]:
    parsed = parse_json_maybe(raw_response)
    if not isinstance(parsed, list):
        raise ValueError("LLM response did not parse into a JSON list")
    cards: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        tags = item.get("tags", [])
        tag_text = (
            " ".join(str(tag) for tag in tags)
            if isinstance(tags, list)
            else str(tags or "")
        )
        if card_format == "cloze":
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            cards.append(
                {
                    "text": text,
                    "extra": str(item.get("extra", "")).strip(),
                    "tags": tag_text,
                }
            )
            continue

        front = str(item.get("front", "")).strip()
        back = str(item.get("back", "")).strip()
        if not front or not back:
            continue
        cards.append({"front": front, "back": back, "tags": tag_text})
    return cards


def write_anki_csv(
    cards: list[dict[str, str]],
    output_csv: Path,
    *,
    card_format: str = "qa",
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["text", "extra", "tags"]
        if card_format == "cloze"
        else ["front", "back", "tags"]
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cards)
    return output_csv


def convert_highlights_to_cards(
    title: str,
    highlights: list[str],
    *,
    backend: str,
    model: str | None,
    output_csv: Path,
    card_format: str = "qa",
    max_cards: int = 20,
) -> tuple[Path, str]:
    prompt = build_anki_prompt(
        title=title,
        highlights=highlights,
        card_format=card_format,
        max_cards=max_cards,
    )
    raw_response = run_prompt(backend, prompt, model=model)
    cards = parse_anki_cards(raw_response, card_format=card_format)
    write_anki_csv(cards, output_csv, card_format=card_format)
    return output_csv, raw_response


def build_alignment_prompt(title: str, summary_text: str, highlights: list[str]) -> str:
    return build_alignment_review_prompt(title, summary_text, highlights)
