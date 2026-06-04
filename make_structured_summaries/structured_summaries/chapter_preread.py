"""Chapter-level pre-read review helpers."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from typing import Protocol

from .models import BookRecord
from .text_extraction import (
    _ordered_epub_documents,
    clean_extracted_text,
    extract_book_text,
    html_to_text,
)
from .utils import normalize_text


@dataclass(frozen=True)
class ChapterSpan:
    index: int
    title: str
    text: str
    source_label: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ChapterHighlight:
    text: str
    color: str
    page: int | None
    source_labels: tuple[str, ...]
    duplicate_count: int
    method: str


@dataclass(frozen=True)
class ChapterTextRange:
    chapter_index: int
    start: int
    end: int


@dataclass(frozen=True)
class ChapterTextIndex:
    text: str
    ranges: tuple[ChapterTextRange, ...]


class HighlightLike(Protocol):
    text: str
    color: str
    page: int | None
    source_labels: tuple[str, ...]
    duplicate_count: int


NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "twenty-one",
    "twenty-two",
    "twenty-three",
    "twenty-four",
    "twenty-five",
    "twenty-six",
    "twenty-seven",
    "twenty-eight",
    "twenty-nine",
    "thirty",
    "thirty-one",
    "thirty-two",
    "thirty-three",
    "thirty-four",
    "thirty-five",
}


def title_matches_book_title(line: str, book_title: str) -> bool:
    line_key = normalize_text(line)
    title_key = normalize_text(book_title)
    return bool(
        line_key
        and title_key
        and (line_key == title_key or line_key in title_key or title_key in line_key)
    )


def looks_like_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    lowered = stripped.casefold().rstrip(".")
    if lowered in NUMBER_WORDS:
        return True
    if re.fullmatch(r"(chapter|part)\s+[ivxlcdm\d]+\.?", lowered):
        return True
    if re.fullmatch(r"[ivxlcdm]+\.?", lowered):
        return True
    if re.fullmatch(r"\d+\.?", lowered):
        return True
    if stripped.endswith((".", "?", "!")):
        return False
    words = re.findall(r"[A-Za-z]+", stripped)
    if 1 <= len(words) <= 10 and stripped[:1].isupper():
        return True
    return False


def chapter_title_from_text(text: str, *, book_title: str, fallback: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    while lines and title_matches_book_title(lines[0], book_title):
        lines.pop(0)
    heading_lines: list[str] = []
    for line in lines[:8]:
        if not looks_like_heading_line(line):
            break
        heading_lines.append(line)
        if len(" ".join(heading_lines)) >= 80:
            break
    title = " ".join(heading_lines).strip()
    return title or fallback


def _chapters_from_epub(
    book: BookRecord,
    *,
    min_chapter_chars: int,
) -> list[ChapterSpan]:
    chapters: list[ChapterSpan] = []
    cursor = 0
    with zipfile.ZipFile(book.primary_path) as epub:
        for document in _ordered_epub_documents(epub):
            try:
                raw = epub.read(document).decode("utf-8", errors="ignore")
            except KeyError:
                continue
            text = html_to_text(raw)
            if len(text) < min_chapter_chars:
                cursor += len(text) + 2
                continue
            title = chapter_title_from_text(
                text,
                book_title=book.title,
                fallback=f"Chapter {len(chapters) + 1}",
            )
            chapters.append(
                ChapterSpan(
                    index=len(chapters) + 1,
                    title=title,
                    text=text,
                    source_label=document,
                    start_char=cursor,
                    end_char=cursor + len(text),
                )
            )
            cursor += len(text) + 2
    return chapters


def _heading_spans(text: str) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    heading_re = re.compile(
        r"(?m)^(?:chapter\s+[\divxlcdm]+(?:\s+.+)?|[ivxlcdm]+\.\s+.+)$",
        flags=re.IGNORECASE,
    )
    for match in heading_re.finditer(text):
        line = match.group(0).strip()
        if 4 <= len(line) <= 100:
            matches.append((match.start(), line))
    return matches


def _chapters_from_text(
    book: BookRecord,
    *,
    min_chapter_chars: int,
    fallback_chars: int,
) -> list[ChapterSpan]:
    text = extract_book_text(book)
    headings = _heading_spans(text)
    chapters: list[ChapterSpan] = []
    for index, (start, heading) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        chapter_text = clean_extracted_text(text[start:end])
        if len(chapter_text) < min_chapter_chars:
            continue
        chapters.append(
            ChapterSpan(
                index=len(chapters) + 1,
                title=heading,
                text=chapter_text,
                source_label=f"heading:{heading}",
                start_char=start,
                end_char=end,
            )
        )
    if chapters:
        weak_heading_split = len(chapters) < 6 and any(
            len(chapter.text) > fallback_chars * 4 for chapter in chapters
        )
        if not weak_heading_split:
            return chapters

    chunks: list[ChapterSpan] = []
    for start in range(0, len(text), fallback_chars):
        chunk = clean_extracted_text(text[start : start + fallback_chars])
        if len(chunk) < min_chapter_chars:
            continue
        chunks.append(
            ChapterSpan(
                index=len(chunks) + 1,
                title=f"Span {len(chunks) + 1}",
                text=chunk,
                source_label="fixed-span",
                start_char=start,
                end_char=start + len(chunk),
            )
        )
    return chunks


def split_long_chapters(
    chapters: list[ChapterSpan],
    *,
    max_chars: int,
) -> list[ChapterSpan]:
    if max_chars <= 0:
        return chapters

    split_chapters: list[ChapterSpan] = []
    for chapter in chapters:
        if len(chapter.text) <= max_chars:
            split_chapters.append(
                ChapterSpan(
                    index=len(split_chapters) + 1,
                    title=chapter.title,
                    text=chapter.text,
                    source_label=chapter.source_label,
                    start_char=chapter.start_char,
                    end_char=chapter.end_char,
                )
            )
            continue

        start = 0
        part_index = 1
        while start < len(chapter.text):
            end = min(len(chapter.text), start + max_chars)
            if end < len(chapter.text):
                paragraph_break = chapter.text.rfind("\n\n", start + max_chars // 2, end)
                if paragraph_break > start:
                    end = paragraph_break
            part_text = clean_extracted_text(chapter.text[start:end])
            if part_text:
                split_chapters.append(
                    ChapterSpan(
                        index=len(split_chapters) + 1,
                        title=f"{chapter.title} (part {part_index})",
                        text=part_text,
                        source_label=chapter.source_label,
                        start_char=chapter.start_char + start,
                        end_char=chapter.start_char + end,
                    )
                )
                part_index += 1
            start = max(end, start + 1)
    return split_chapters


def extract_chapters(
    book: BookRecord,
    *,
    min_chapter_chars: int = 2_500,
    fallback_chars: int = 60_000,
) -> list[ChapterSpan]:
    if book.primary_format == "epub":
        chapters = _chapters_from_epub(book, min_chapter_chars=min_chapter_chars)
        if chapters:
            return chapters
    return _chapters_from_text(
        book,
        min_chapter_chars=min_chapter_chars,
        fallback_chars=fallback_chars,
    )


def highlight_probes(text: str) -> list[str]:
    normalized = normalize_text(text)
    if len(normalized) < 12:
        return []
    probes = [normalized]
    if len(normalized) > 420:
        probes.extend(
            [
                normalized[:320],
                normalized[:220],
                normalized[-260:],
                normalized[-180:],
            ]
        )
    elif len(normalized) > 220:
        probes.extend([normalized[:180], normalized[-160:]])
    elif len(normalized) > 120:
        probes.append(normalized[:100])
    deduped: list[str] = []
    seen: set[str] = set()
    for probe in probes:
        probe = probe.strip()
        if len(probe) >= 12 and probe not in seen:
            seen.add(probe)
            deduped.append(probe)
    return deduped


def build_chapter_text_index(chapters: list[ChapterSpan]) -> ChapterTextIndex:
    parts: list[str] = []
    ranges: list[ChapterTextRange] = []
    cursor = 0
    for chapter in chapters:
        normalized = normalize_text(chapter.text)
        if not normalized:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(normalized)
        cursor += len(normalized)
        ranges.append(
            ChapterTextRange(
                chapter_index=chapter.index,
                start=start,
                end=cursor,
            )
        )
    return ChapterTextIndex(text="".join(parts), ranges=tuple(ranges))


def chapter_index_for_offset(
    text_index: ChapterTextIndex,
    offset: int,
) -> int | None:
    for text_range in text_index.ranges:
        if text_range.start <= offset < text_range.end:
            return text_range.chapter_index
    return None


def locate_highlight_chapter(
    highlight_text: str,
    chapters: list[ChapterSpan],
    *,
    text_index: ChapterTextIndex | None = None,
) -> tuple[int | None, str]:
    index = text_index or build_chapter_text_index(chapters)
    full_highlight = normalize_text(highlight_text)
    for probe in highlight_probes(highlight_text):
        offset = index.text.find(probe)
        if offset == -1:
            continue
        chapter_index = chapter_index_for_offset(index, offset)
        if chapter_index is None:
            continue
        method = "source text" if probe == full_highlight else "source text probe"
        return chapter_index, method
    return None, "unpositioned"


def page_progress_chapter(
    page: int | None,
    *,
    max_page: int | None,
    chapter_count: int,
) -> int | None:
    if page is None or not max_page or max_page <= 0 or chapter_count <= 0:
        return None
    progress = min(max(page / max_page, 0.0), 1.0)
    if progress >= 1.0:
        return chapter_count
    return min(chapter_count, int(progress * chapter_count) + 1)


def place_highlights_by_chapter(
    highlights: list[HighlightLike],
    chapters: list[ChapterSpan],
) -> dict[int, list[ChapterHighlight]]:
    max_page = max(
        (highlight.page for highlight in highlights if highlight.page is not None),
        default=None,
    )
    placed: dict[int, list[ChapterHighlight]] = {chapter.index: [] for chapter in chapters}
    text_index = build_chapter_text_index(chapters)
    for highlight in highlights:
        chapter_index, method = locate_highlight_chapter(
            highlight.text,
            chapters,
            text_index=text_index,
        )
        if chapter_index is None:
            chapter_index = page_progress_chapter(
                highlight.page,
                max_page=max_page,
                chapter_count=len(chapters),
            )
            method = "page progress" if chapter_index is not None else "unpositioned"
        if chapter_index is None:
            continue
        placed.setdefault(chapter_index, []).append(
            ChapterHighlight(
                text=highlight.text,
                color=highlight.color,
                page=highlight.page,
                source_labels=highlight.source_labels,
                duplicate_count=highlight.duplicate_count,
                method=method,
            )
        )
    return placed


def render_highlights_markdown(highlights: list[ChapterHighlight]) -> str:
    if not highlights:
        return "- No mapped highlights for this chapter."
    lines: list[str] = []
    for index, highlight in enumerate(highlights, start=1):
        page = f"p{highlight.page}" if highlight.page is not None else "no page"
        sources = ", ".join(highlight.source_labels)
        lines.append(
            f"- #{index} [{highlight.color or 'uncolored'} | {page} | "
            f"{highlight.method} | {sources}] {highlight.text}"
        )
    return "\n".join(lines)
