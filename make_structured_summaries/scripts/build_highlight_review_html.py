from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from scripts.evaluate_highlight_coverage import (
    COLOR_LABELS,
    PROJECT_ROOT,
    SIDE_PROJECTS_ROOT,
    CombinedHighlight,
    SourceDocument,
    SummaryTarget,
    combine_highlights,
    load_source_documents,
    load_summary_targets,
    manifest_for_book,
    title_similarity,
)
from structured_summaries.highlights import HighlightEntry, annotate_entries_with_progress
from structured_summaries.utils import normalize_text, slugify


DEFAULT_PRESET = "churchill-price-senate"
DEFAULT_SELECTED_TITLES = (
    "A History of the English-Speaking Peoples Collection",
    "Churchill",
    "Churchill: Walking with Destiny",
    "Great Contemporaries",
    "Master of the Senate",
    "My Early Life",
    "The Price of Victory",
    "The Second World War",
    "The World Crisis",
    "The World Crisis, Vol. 2",
    "The World Crisis, Vol. 3",
    "The World Crisis, Vol. 4",
    "Thoughts and Adventures",
)
CHURCHILL_AUTHOR_MARKERS = ("churchill",)
AI_CATEGORIES = (
    "orientation_facts",
    "arc_events",
    "key_mechanisms",
    "load_bearing_scenes",
    "reader_watchpoints",
    "people_and_terms",
    "likely_low_value_detail",
    "core_models",
    "key_facts_and_mechanisms",
    "legibility_gains",
    "reasoning_methods",
    "practical_transfers",
    "named_tools_and_metrics",
    "best_examples",
    "tactical_wins",
    "non_obvious_claims",
    "pushback_points",
    "checkable_claims",
)


@dataclass(frozen=True)
class AiItem:
    category: str
    text: str
    label: str = ""


@dataclass(frozen=True)
class AiChunk:
    chunk_index: int
    items: tuple[AiItem, ...]
    raw_text: str
    source_text: str = ""


@dataclass(frozen=True)
class AiRunMetadata:
    source_path: Path | None
    source_format: str
    extracted_text_path: Path | None
    chunk_count: int | None
    summary_path: Path | None


@dataclass(frozen=True)
class PositionedHighlight:
    highlight: CombinedHighlight
    estimated_chunk: int | None
    progress: float | None
    method: str


@dataclass(frozen=True)
class BookReview:
    title: str
    author: str
    slug: str
    highlights: list[CombinedHighlight]
    source_paths: list[Path]
    ai_target: SummaryTarget | None
    ai_chunks: list[AiChunk]
    ai_metadata: AiRunMetadata | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build static HTML to compare real Play Books highlights against "
            "AI chunk extractions by book position, without text matching."
        )
    )
    parser.add_argument(
        "--preset",
        choices=(DEFAULT_PRESET, "none"),
        default=DEFAULT_PRESET,
        help="Default title selection. Use none with --title for a custom set.",
    )
    parser.add_argument(
        "--title",
        action="append",
        default=[],
        help="Title to include. Can be repeated.",
    )
    parser.add_argument(
        "--drive-notes-root",
        type=Path,
        default=SIDE_PROJECTS_ROOT / "data" / "Play_Books_Notes-20260531T040817Z",
    )
    parser.add_argument(
        "--tablet-md-root",
        type=Path,
        default=SIDE_PROJECTS_ROOT
        / "data"
        / "play_books_highlights_full_backup_20260530_235315",
    )
    parser.add_argument(
        "--curated-notes-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "my_highlights",
    )
    parser.add_argument(
        "--chunk-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunk_notes",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "summaries",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_reviews" / "highlight_review_html",
    )
    parser.add_argument("--source-match-threshold", type=float, default=0.84)
    parser.add_argument("--ai-target-threshold", type=float, default=0.80)
    parser.add_argument("--max-highlight-rows", type=int)
    return parser.parse_args()


def parse_chunk_json(raw_output: str) -> dict[str, Any] | None:
    candidates = [raw_output.strip()]
    stripped = raw_output.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def join_nonempty(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def item_from_dict(item: dict[str, Any], *, category: str) -> AiItem | None:
    if category == "orientation_facts":
        label = clean_text(item.get("fact"))
        text = join_nonempty(
            [
                clean_text(item.get("fact")),
                f"Why it matters: {clean_text(item.get('why_it_matters'))}"
                if clean_text(item.get("why_it_matters"))
                else "",
            ]
        )
    elif category == "arc_events":
        label = clean_text(item.get("event"))
        text = join_nonempty(
            [
                clean_text(item.get("event")),
                f"Consequence: {clean_text(item.get('consequence'))}"
                if clean_text(item.get("consequence"))
                else "",
            ]
        )
    elif category == "key_mechanisms":
        label = clean_text(item.get("mechanism"))
        text = join_nonempty(
            [
                clean_text(item.get("mechanism")),
                f"Example: {clean_text(item.get('example'))}"
                if clean_text(item.get("example"))
                else "",
            ]
        )
    elif category == "load_bearing_scenes":
        label = clean_text(item.get("label"))
        text = join_nonempty(
            [
                clean_text(item.get("what_happens")),
                f"Why it matters: {clean_text(item.get('why_it_matters'))}"
                if clean_text(item.get("why_it_matters"))
                else "",
            ]
        )
    elif category == "people_and_terms":
        label = clean_text(item.get("name"))
        text = clean_text(item.get("role"))
    elif category == "core_models":
        label = clean_text(item.get("name"))
        text = join_nonempty(
            [
                clean_text(item.get("summary")),
                f"Why it matters: {clean_text(item.get('why_it_matters'))}"
                if clean_text(item.get("why_it_matters"))
                else "",
            ]
        )
    elif category == "named_tools_and_metrics":
        label = clean_text(item.get("name"))
        text = join_nonempty(
            [
                clean_text(item.get("what_it_is")),
                f"When to use: {clean_text(item.get('when_to_use'))}"
                if clean_text(item.get("when_to_use"))
                else "",
            ]
        )
    elif category in {"best_examples", "tactical_wins"}:
        label = clean_text(item.get("label"))
        text = join_nonempty(
            [
                f"Supports: {clean_text(item.get('supports'))}"
                if clean_text(item.get("supports"))
                else "",
                f"Setup: {clean_text(item.get('setup'))}"
                if clean_text(item.get("setup"))
                else "",
                f"Mechanism: {clean_text(item.get('mechanism'))}"
                if clean_text(item.get("mechanism"))
                else "",
                f"Payoff: {clean_text(item.get('payoff'))}"
                if clean_text(item.get("payoff"))
                else "",
                f"Why it matters: {clean_text(item.get('why_it_matters'))}"
                if clean_text(item.get("why_it_matters"))
                else "",
            ]
        )
    elif category == "checkable_claims":
        label = clean_text(item.get("claim"))
        text = join_nonempty(
            [
                clean_text(item.get("claim")),
                f"Why check: {clean_text(item.get('why_check'))}"
                if clean_text(item.get("why_check"))
                else "",
            ]
        )
    else:
        label = ""
        text = ""

    if not text and label:
        text = label
    if not text:
        return None
    return AiItem(category=category, label=label, text=text)


def extract_ai_items(parsed: dict[str, Any]) -> list[AiItem]:
    items: list[AiItem] = []
    for category in AI_CATEGORIES:
        value = parsed.get(category)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str):
                text = clean_text(item)
                if text:
                    items.append(AiItem(category=category, text=text))
            elif isinstance(item, dict):
                parsed_item = item_from_dict(item, category=category)
                if parsed_item is not None:
                    items.append(parsed_item)
    return items


def extract_prompt_source_text(prompt_path: Path) -> str:
    if not prompt_path.exists():
        return ""
    prompt = prompt_path.read_text(encoding="utf-8", errors="ignore")
    marker = "\nChunk text:\n"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].strip()


def extract_ai_chunks(chunk_dir: Path) -> list[AiChunk]:
    chunks: list[AiChunk] = []
    for chunk_file in sorted(chunk_dir.glob("chunk_*_response.txt")):
        match = re.search(r"chunk_(\d+)_response\.txt$", chunk_file.name)
        chunk_index = int(match.group(1)) if match else len(chunks) + 1
        raw_text = chunk_file.read_text(encoding="utf-8", errors="ignore").strip()
        parsed = parse_chunk_json(raw_text)
        items = extract_ai_items(parsed) if parsed is not None else []
        if not items and raw_text:
            items = [AiItem(category="raw_chunk_output", text=clean_text(raw_text))]
        chunks.append(
            AiChunk(
                chunk_index=chunk_index,
                items=tuple(items),
                raw_text=raw_text,
                source_text=extract_prompt_source_text(
                    chunk_dir / f"chunk_{chunk_index:03d}_prompt.txt"
                ),
            )
        )
    return chunks


def selected_title_patterns(args: argparse.Namespace) -> tuple[str, ...]:
    patterns: list[str] = []
    if args.preset == DEFAULT_PRESET:
        patterns.extend(DEFAULT_SELECTED_TITLES)
    patterns.extend(args.title)
    return tuple(patterns)


def is_selected_source(title: str, author: str, patterns: tuple[str, ...]) -> bool:
    title_norm = normalize_text(title)
    author_norm = normalize_text(author)
    if any(marker in author_norm for marker in CHURCHILL_AUTHOR_MARKERS):
        return True
    if "churchill" in title_norm:
        return True
    return any(title_similarity(title, pattern) >= 0.86 for pattern in patterns)


def best_ai_target(
    title: str,
    targets: list[SummaryTarget],
    *,
    threshold: float,
) -> SummaryTarget | None:
    scored = [
        (title_similarity(title, target.title), target)
        for target in targets
        if target.variant == "current"
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < threshold:
        return None
    return scored[0][1]


def ai_metadata_for_target(
    target: SummaryTarget,
    *,
    summary_dir: Path,
) -> AiRunMetadata:
    manifest = manifest_for_book(summary_dir, target.book_id)
    book = manifest.get("book", {})
    paths = manifest.get("paths", {})
    chunk_count = manifest.get("chunk_count")
    if not isinstance(book, dict):
        book = {}
    if not isinstance(paths, dict):
        paths = {}
    return AiRunMetadata(
        source_path=path_or_none(book.get("primary_path")),
        source_format=str(book.get("primary_format") or ""),
        extracted_text_path=path_or_none(paths.get("extracted_text_path")),
        chunk_count=chunk_count if isinstance(chunk_count, int) else None,
        summary_path=path_or_none(paths.get("summary_path")) or target.path,
    )


def path_or_none(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def group_sources_by_title(
    selected_docs: list[SourceDocument],
    *,
    threshold: float,
) -> dict[str, list[SourceDocument]]:
    by_title: dict[str, list[SourceDocument]] = {}
    for source in selected_docs:
        matched = False
        for key, existing_sources in list(by_title.items()):
            if title_similarity(source.document.title, key) >= threshold:
                existing_sources.append(source)
                matched = True
                break
        if not matched:
            by_title[source.document.title] = [source]
    return by_title


def build_reviews(args: argparse.Namespace) -> list[BookReview]:
    patterns = selected_title_patterns(args)
    sources = load_source_documents(
        drive_notes_root=args.drive_notes_root,
        tablet_md_root=args.tablet_md_root,
        curated_notes_root=args.curated_notes_root,
    )
    targets = load_summary_targets(args.summary_dir, include_expanded=False)
    selected_docs = [
        source
        for source in sources
        if is_selected_source(
            source.document.title,
            source.document.author,
            patterns,
        )
    ]
    by_title = group_sources_by_title(
        selected_docs,
        threshold=args.source_match_threshold,
    )

    reviews: list[BookReview] = []
    for title, group in sorted(by_title.items(), key=lambda item: normalize_text(item[0])):
        highlights = combine_highlights(group)
        if not highlights:
            continue
        if args.max_highlight_rows:
            highlights = highlights[: args.max_highlight_rows]
        authors = [
            source.document.author
            for source in group
            if source.document.author and source.document.author != "Unknown Author"
        ]
        author = authors[0] if authors else ""
        target = best_ai_target(
            title,
            targets,
            threshold=args.ai_target_threshold,
        )
        chunks: list[AiChunk] = []
        metadata: AiRunMetadata | None = None
        if target is not None:
            chunks = extract_ai_chunks(args.chunk_root / target.book_id)
            metadata = ai_metadata_for_target(target, summary_dir=args.summary_dir)
        source_paths = sorted(
            {
                source.document.source_path
                for source in group
                if source.document.source_path is not None
            }
        )
        reviews.append(
            BookReview(
                title=title,
                author=author,
                slug=slugify(title),
                highlights=highlights,
                source_paths=source_paths,
                ai_target=target,
                ai_chunks=chunks,
                ai_metadata=metadata,
            )
        )
    return reviews


def total_chunk_count(review: BookReview) -> int:
    manifest_count = review.ai_metadata.chunk_count if review.ai_metadata else None
    observed_count = max((chunk.chunk_index for chunk in review.ai_chunks), default=0)
    return manifest_count or observed_count


def positioned_highlights(
    highlights: list[CombinedHighlight],
    *,
    total_chunks: int,
    ai_chunks: list[AiChunk],
) -> list[PositionedHighlight]:
    entries = [
        HighlightEntry(
            text=highlight.text,
            color=highlight.color,
            page=highlight.page,
        )
        for highlight in highlights
    ]
    annotated = annotate_entries_with_progress(entries, total_chunks=total_chunks)
    chunk_texts = {
        chunk.chunk_index: normalize_text(chunk.source_text)
        for chunk in ai_chunks
        if chunk.source_text
    }
    positioned: list[PositionedHighlight] = []
    for highlight, entry in zip(highlights, annotated, strict=True):
        located_chunk = source_located_chunk(highlight.text, chunk_texts)
        estimated_chunk = located_chunk or entry.estimated_chunk
        if located_chunk is not None:
            method = "source text"
        elif entry.estimated_chunk is not None:
            method = "page progress"
        else:
            method = "unpositioned"
        positioned.append(
            PositionedHighlight(
                highlight=highlight,
                estimated_chunk=estimated_chunk,
                progress=entry.progress,
                method=method,
            )
        )
    return positioned


def source_located_chunk(
    highlight_text: str,
    chunk_texts: dict[int, str],
) -> int | None:
    normalized = normalize_text(highlight_text)
    if len(normalized) < 20:
        return None
    probes = [normalized]
    if len(normalized) > 260:
        probes.extend(
            [
                normalized[:260],
                normalized[:180],
                normalized[-180:],
            ]
        )
    elif len(normalized) > 120:
        probes.append(normalized[:120])
    for probe in probes:
        if len(probe) < 20:
            continue
        for chunk_index, chunk_text in sorted(chunk_texts.items()):
            if probe in chunk_text:
                return chunk_index
    return None


def color_class(color: str) -> str:
    return COLOR_LABELS.get(color, "uncolored")


def render_sources(paths: list[Path]) -> str:
    if not paths:
        return ""
    items = "\n".join(f"<li><code>{escape(str(path))}</code></li>" for path in paths)
    return f"<details><summary>Highlight source files</summary><ul>{items}</ul></details>"


def render_ai_metadata(metadata: AiRunMetadata | None) -> str:
    if metadata is None:
        return """
        <div class="provenance warning">
          <strong>AI provenance</strong>
          <span>No AI summary/extraction output was found for this book.</span>
        </div>
        """
    source = escape(str(metadata.source_path)) if metadata.source_path else "unknown"
    extracted = (
        escape(str(metadata.extracted_text_path))
        if metadata.extracted_text_path
        else "unknown"
    )
    summary = escape(str(metadata.summary_path)) if metadata.summary_path else "unknown"
    file_format = metadata.source_format or "unknown"
    direct_pdf = "yes" if file_format == "pdf" else "no"
    return f"""
    <div class="provenance">
      <strong>AI provenance</strong>
      <span>Local source: <code>{source}</code></span>
      <span>Format: <code>{escape(file_format)}</code>; direct PDF: <code>{direct_pdf}</code></span>
      <span>Extracted text: <code>{extracted}</code></span>
      <span>Final summary: <code>{summary}</code></span>
    </div>
    """


def page_label(highlights: list[PositionedHighlight]) -> str:
    pages = sorted(
        {item.highlight.page for item in highlights if item.highlight.page is not None}
    )
    if not pages:
        return "no page"
    if pages[0] == pages[-1]:
        return f"p{pages[0]}"
    return f"p{pages[0]}-p{pages[-1]}"


def placement_label(highlights: list[PositionedHighlight]) -> str:
    if not highlights:
        return "no highlights"
    counts: dict[str, int] = defaultdict(int)
    for item in highlights:
        counts[item.method] += 1
    return "; ".join(
        f"{method}: {count}" for method, count in sorted(counts.items())
    )


def render_highlight_card(item: PositionedHighlight, index: int) -> str:
    highlight = item.highlight
    color = color_class(highlight.color)
    page = f"p{highlight.page}" if highlight.page is not None else "no page"
    sources = ", ".join(highlight.source_labels)
    return f"""
    <article class="highlight-card {color}">
      <div class="highlight-meta">
        <span>#{index}</span>
        <span>{escape(color)}</span>
        <span>{escape(page)}</span>
        <span>{escape(sources)}</span>
        <span>{highlight.duplicate_count} source hits</span>
        <span>{escape(item.method)}</span>
      </div>
      <p>{escape(highlight.text)}</p>
    </article>
    """


def render_highlight_group(
    highlights: list[PositionedHighlight],
    offset: int = 0,
) -> str:
    if not highlights:
        return '<div class="empty-state">No highlights estimated in this span.</div>'
    return "\n".join(
        render_highlight_card(highlight, offset + index)
        for index, highlight in enumerate(highlights, start=1)
    )


def category_label(category: str) -> str:
    return category.replace("_", " ").title()


def render_ai_item(item: AiItem) -> str:
    label = f"<strong>{escape(item.label)}</strong><br>" if item.label else ""
    return f"<li>{label}{escape(item.text)}</li>"


def render_ai_chunk(chunk: AiChunk) -> str:
    if not chunk.items:
        return '<div class="empty-state">No AI extraction content found for this chunk.</div>'
    by_category: dict[str, list[AiItem]] = defaultdict(list)
    for item in chunk.items:
        by_category[item.category].append(item)
    sections: list[str] = []
    for category in AI_CATEGORIES + ("raw_chunk_output",):
        items = by_category.get(category)
        if not items:
            continue
        rendered_items = "\n".join(render_ai_item(item) for item in items)
        sections.append(
            f"""
            <details open class="ai-category">
              <summary>{escape(category_label(category))} ({len(items)})</summary>
              <ol>{rendered_items}</ol>
            </details>
            """
        )
    return "\n".join(sections)


def render_chunk_row(
    *,
    chunk: AiChunk,
    total_chunks: int,
    highlights: list[PositionedHighlight],
    highlight_offset: int,
) -> str:
    pages = page_label(highlights)
    method_counts = placement_label(highlights)
    return f"""
    <article class="position-row" id="chunk-{chunk.chunk_index:03d}">
      <section class="real-highlights">
        <div class="span-meta">
          <span>{escape(method_counts)}</span>
          <span>{escape(pages)}</span>
          <span>{len(highlights)} highlights</span>
        </div>
        {render_highlight_group(highlights, offset=highlight_offset)}
      </section>
      <section class="ai-output">
        <div class="span-meta">
          <span>AI chunk {chunk.chunk_index} of {total_chunks}</span>
          <span>{len(chunk.items)} extraction items</span>
        </div>
        {render_ai_chunk(chunk)}
      </section>
    </article>
    """


def render_missing_ai_page_row(
    *,
    label: str,
    highlights: list[PositionedHighlight],
    highlight_offset: int,
) -> str:
    return f"""
    <article class="position-row">
      <section class="real-highlights">
        <div class="span-meta">
          <span>{escape(label)}</span>
          <span>{len(highlights)} highlights</span>
          <span>{escape(placement_label(highlights))}</span>
        </div>
        {render_highlight_group(highlights, offset=highlight_offset)}
      </section>
      <section class="ai-output">
        <div class="empty-state">No AI extraction output found for this book.</div>
      </section>
    </article>
    """


def render_book_rows(review: BookReview) -> str:
    total_chunks = total_chunk_count(review)
    if not review.ai_chunks or total_chunks == 0:
        by_page: dict[int | None, list[PositionedHighlight]] = defaultdict(list)
        for highlight in review.highlights:
            by_page[highlight.page].append(
                PositionedHighlight(
                    highlight=highlight,
                    estimated_chunk=None,
                    progress=None,
                    method="page group",
                )
            )
        rows: list[str] = []
        offset = 0
        for page in sorted(by_page, key=lambda item: (item is None, item or 10**9)):
            highlights = by_page[page]
            label = f"p{page}" if page is not None else "no page"
            rows.append(
                render_missing_ai_page_row(
                    label=label,
                    highlights=highlights,
                    highlight_offset=offset,
                )
            )
            offset += len(highlights)
        return "\n".join(rows)

    positioned = positioned_highlights(
        review.highlights,
        total_chunks=total_chunks,
        ai_chunks=review.ai_chunks,
    )
    by_chunk: dict[int, list[PositionedHighlight]] = defaultdict(list)
    unpositioned: list[PositionedHighlight] = []
    for item in positioned:
        if item.estimated_chunk is None:
            unpositioned.append(item)
        else:
            by_chunk[item.estimated_chunk].append(item)

    rows = []
    offset = 0
    chunks_by_index = {chunk.chunk_index: chunk for chunk in review.ai_chunks}
    for chunk_index in range(1, total_chunks + 1):
        chunk = chunks_by_index.get(
            chunk_index,
            AiChunk(chunk_index=chunk_index, items=(), raw_text=""),
        )
        highlights = by_chunk.get(chunk_index, [])
        rows.append(
            render_chunk_row(
                chunk=chunk,
                total_chunks=total_chunks,
                highlights=highlights,
                highlight_offset=offset,
            )
        )
        offset += len(highlights)

    if unpositioned:
        rows.append(
            render_missing_ai_page_row(
                label="highlights without Play Books page",
                highlights=unpositioned,
                highlight_offset=offset,
            )
        )
    return "\n".join(rows)


def direct_pdf_label(metadata: AiRunMetadata | None) -> str:
    if metadata is None:
        return "no AI output"
    if metadata.source_format == "pdf":
        return "yes"
    if metadata.source_format:
        return f"no ({metadata.source_format})"
    return "unknown"


def highlights_with_pages(highlights: list[CombinedHighlight]) -> int:
    return sum(1 for highlight in highlights if highlight.page is not None)


def render_book_page(review: BookReview) -> str:
    total_chunks = total_chunk_count(review)
    ai_label = (
        f"{total_chunks} chunks from {review.ai_metadata.source_format or 'unknown source'}"
        if review.ai_metadata is not None
        else "none"
    )
    page_count = highlights_with_pages(review.highlights)
    summary = f"""
    <div class="summary-grid">
      <div><strong>Real highlights</strong><span>{len(review.highlights)}</span></div>
      <div><strong>Highlights with pages</strong><span>{page_count}</span></div>
      <div><strong>AI output</strong><span>{escape(ai_label)}</span></div>
      <div><strong>Direct PDF</strong><span>{escape(direct_pdf_label(review.ai_metadata))}</span></div>
    </div>
    """
    return render_html_document(
        title=f"{review.title} - positional highlight review",
        body=f"""
        <header class="page-header">
          <a href="index.html">Index</a>
          <h1>{escape(review.title)}</h1>
          <p>{escape(review.author)}</p>
          <p>
            This page is positional, not text-matched. Left-side highlights are
            grouped by Play Books page/progress; right-side AI output is grouped
            by the original AI chunk. The current artifacts do not contain exact
            PDF page ranges for each AI chunk.
          </p>
          {summary}
          {render_ai_metadata(review.ai_metadata)}
          {render_sources(review.source_paths)}
        </header>
        <main class="book-page">
          <div class="columns-label">
            <span>Real highlights from this span</span>
            <span>AI chunk extraction from the corresponding span</span>
          </div>
          {render_book_rows(review)}
        </main>
        """,
    )


def render_index(reviews: list[BookReview]) -> str:
    rows: list[str] = []
    for review in reviews:
        total = len(review.highlights)
        page_count = highlights_with_pages(review.highlights)
        chunk_count = total_chunk_count(review)
        ai_text = (
            f"{chunk_count} chunks"
            if review.ai_metadata is not None
            else "no AI output"
        )
        rows.append(
            f"""
            <tr>
              <td><a href="{escape(review.slug)}.html">{escape(review.title)}</a></td>
              <td>{escape(review.author)}</td>
              <td>{total}</td>
              <td>{page_count}</td>
              <td>{escape(ai_text)}</td>
              <td>{escape(direct_pdf_label(review.ai_metadata))}</td>
            </tr>
            """
        )
    body = f"""
    <header class="page-header">
      <h1>Highlight Review Index</h1>
      <p>
        Positional review pages: real Play Books highlights are grouped on the
        left; AI chunk extractions from the same approximate book span are on
        the right. No text-match scoring is used.
      </p>
      <p>
        Inputs: Drive DOCX notes, tablet markdown exports, curated notes, and
        existing local AI chunk outputs.
      </p>
    </header>
    <main>
      <table class="index-table">
        <thead>
          <tr>
            <th>Book</th>
            <th>Author</th>
            <th>Highlights</th>
            <th>With pages</th>
            <th>AI output</th>
            <th>Direct PDF</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </main>
    """
    return render_html_document(title="Highlight Review Index", body=body)


def render_html_document(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #60646c;
      --line: #d8d8d2;
      --panel: #ffffff;
      --yellow: #fff2a8;
      --green: #d8f3dc;
      --blue: #dbeafe;
      --red: #ffe0e0;
      --uncolored: #f1f3f4;
      --warn: #fff7d6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    a {{ color: #174ea6; }}
    .page-header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 247, 244, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 4px 0; font-size: 24px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; }}
    code {{ font-size: 12px; overflow-wrap: anywhere; }}
    main {{ padding: 18px 24px 40px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 8px;
      margin-top: 12px;
    }}
    .summary-grid div, .provenance {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px;
      border-radius: 6px;
    }}
    .summary-grid strong, .summary-grid span, .provenance strong, .provenance span {{
      display: block;
      font-size: 13px;
    }}
    .summary-grid span, .provenance span {{ color: var(--muted); margin-top: 4px; }}
    .provenance {{ margin-top: 10px; }}
    .provenance.warning {{ background: var(--warn); }}
    .columns-label, .position-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }}
    .columns-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
      padding: 0 2px;
    }}
    .position-row {{
      margin-bottom: 14px;
      align-items: start;
    }}
    .real-highlights, .ai-output {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      min-height: 120px;
    }}
    .span-meta, .highlight-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .span-meta span, .highlight-meta span {{
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      color: var(--muted);
    }}
    .highlight-card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      margin-bottom: 8px;
      background: var(--uncolored);
    }}
    .highlight-card.yellow {{ background: var(--yellow); }}
    .highlight-card.green {{ background: var(--green); }}
    .highlight-card.blue {{ background: var(--blue); }}
    .highlight-card.red {{ background: var(--red); }}
    .highlight-card.uncolored {{ background: var(--uncolored); }}
    .ai-category {{
      border-top: 1px solid var(--line);
      padding-top: 8px;
      margin-top: 8px;
    }}
    .ai-category summary {{
      color: var(--muted);
      cursor: pointer;
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .ai-category ol {{
      margin: 0 0 0 22px;
      padding: 0;
    }}
    .ai-category li {{
      margin-bottom: 8px;
    }}
    .empty-state {{
      color: var(--muted);
      font-style: italic;
    }}
    .index-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    .index-table th, .index-table td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      vertical-align: top;
    }}
    .index-table th {{
      font-size: 13px;
      color: var(--muted);
      background: #fafafa;
    }}
    details {{ margin-top: 10px; }}
    @media (max-width: 900px) {{
      .summary-grid, .columns-label, .position-row {{
        grid-template-columns: 1fr;
      }}
      .page-header {{ position: static; }}
      main {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    reviews = build_reviews(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for review in reviews:
        (args.out_dir / f"{review.slug}.html").write_text(
            render_book_page(review),
            encoding="utf-8",
        )
    index_path = args.out_dir / "index.html"
    index_path.write_text(render_index(reviews), encoding="utf-8")
    print(f"Wrote {len(reviews)} positional review pages to {args.out_dir}")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
