from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from structured_summaries.highlights import (
    HighlightDocument,
    HighlightEntry,
    annotate_entries_with_progress,
    load_highlight_document,
)
from structured_summaries.utils import clean_title_for_search, normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDE_PROJECTS_ROOT = PROJECT_ROOT.parent
COLOR_PRIORITY = {"blue": 4, "red": 3, "green": 2, "yellow": 1, "": 0}
COLOR_LABELS = {
    "yellow": "yellow",
    "green": "green",
    "blue": "blue",
    "red": "red",
    "": "uncolored",
}
STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "almost",
    "also",
    "although",
    "always",
    "among",
    "another",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "doing",
    "done",
    "down",
    "during",
    "each",
    "even",
    "every",
    "from",
    "have",
    "having",
    "here",
    "into",
    "itself",
    "just",
    "like",
    "more",
    "most",
    "much",
    "must",
    "only",
    "other",
    "over",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


@dataclass(frozen=True)
class SourceDocument:
    document: HighlightDocument
    source_label: str
    match_title_key: str


@dataclass(frozen=True)
class CombinedHighlight:
    text: str
    color: str
    page: int | None
    source_labels: tuple[str, ...]
    duplicate_count: int


@dataclass(frozen=True)
class SummaryTarget:
    path: Path
    book_id: str
    variant: str
    title: str
    chunk_model: str
    synthesis_model: str


@dataclass(frozen=True)
class ScoredHighlight:
    highlight: CombinedHighlight
    score: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated book summaries against combined Play Books highlights."
        )
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "summaries",
    )
    parser.add_argument(
        "--chunk-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunk_notes",
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
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_reviews" / "combined_highlight_metrics",
    )
    parser.add_argument("--match-threshold", type=float, default=0.84)
    parser.add_argument("--covered-threshold", type=float, default=0.38)
    parser.add_argument("--partial-threshold", type=float, default=0.22)
    parser.add_argument("--include-expanded", action="store_true")
    return parser.parse_args()


def title_key(title: str) -> str:
    return normalize_text(clean_title_for_search(title))


def title_similarity(left: str, right: str) -> float:
    left_key = title_key(left)
    right_key = title_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    if min(len(left_key), len(right_key)) >= 12 and (
        left_key in right_key or right_key in left_key
    ):
        return 0.95
    return SequenceMatcher(None, left_key, right_key).ratio()


def iter_existing_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def load_source_documents(
    *,
    drive_notes_root: Path,
    tablet_md_root: Path,
    curated_notes_root: Path,
) -> list[SourceDocument]:
    sources: list[tuple[str, list[Path]]] = [
        ("drive_docx", iter_existing_files(drive_notes_root, ("*.docx",))),
        ("tablet_md", iter_existing_files(tablet_md_root, ("*.md",))),
        (
            "curated_notes",
            iter_existing_files(curated_notes_root, ("*.txt", "*.md", "*.docx")),
        ),
    ]

    documents: list[SourceDocument] = []
    for source_label, paths in sources:
        for path in paths:
            document = load_highlight_document(path, source_section=source_label)
            if document is None:
                continue
            documents.append(
                SourceDocument(
                    document=document,
                    source_label=source_label,
                    match_title_key=title_key(document.title),
                )
            )
    return documents


def manifest_for_book(summary_dir: Path, book_id: str) -> dict:
    manifest_path = summary_dir / f"{book_id}.manifest.json"
    if not manifest_path.exists():
        manifest_path = summary_dir / f"{book_id}.pre_prompt_revision.manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def split_summary_stem(stem: str) -> tuple[str, str]:
    for marker in (".pre_", ".expanded"):
        if marker in stem:
            base, suffix = stem.split(marker, 1)
            return base, marker.lstrip(".") + suffix
    return stem, "current"


def load_summary_targets(summary_dir: Path, *, include_expanded: bool) -> list[SummaryTarget]:
    targets: list[SummaryTarget] = []
    for path in sorted(summary_dir.glob("*.md")):
        if path.name.endswith(".critique.md"):
            continue
        if path.name.endswith(".expanded.md") and not include_expanded:
            continue
        book_id, variant = split_summary_stem(path.stem)
        manifest = manifest_for_book(summary_dir, book_id)
        book = manifest.get("book", {})
        config = manifest.get("config", {})
        title = book.get("title") or book_id.replace("-", " ")
        targets.append(
            SummaryTarget(
                path=path,
                book_id=book_id,
                variant=variant,
                title=title,
                chunk_model=str(config.get("chunk_model") or ""),
                synthesis_model=str(config.get("synthesis_model") or ""),
            )
        )
    return targets


def matched_documents(
    target: SummaryTarget,
    sources: list[SourceDocument],
    *,
    threshold: float,
) -> list[tuple[SourceDocument, float]]:
    matches: list[tuple[SourceDocument, float]] = []
    for source in sources:
        score = title_similarity(target.title, source.document.title)
        if score >= threshold:
            matches.append((source, score))
    return sorted(matches, key=lambda item: item[1], reverse=True)


def choose_entry(existing: CombinedHighlight, entry: HighlightEntry) -> tuple[str, int | None]:
    existing_priority = COLOR_PRIORITY.get(existing.color, 0)
    entry_priority = COLOR_PRIORITY.get(entry.color, 0)
    color = existing.color if existing_priority >= entry_priority else entry.color
    if existing.page is None:
        page = entry.page
    elif entry.page is None:
        page = existing.page
    else:
        page = min(existing.page, entry.page)
    return color, page


def combine_highlights(documents: list[SourceDocument]) -> list[CombinedHighlight]:
    combined: dict[str, CombinedHighlight] = {}
    for source in documents:
        for entry in source.document.entries:
            key = normalize_text(entry.text)
            if not key:
                continue
            current = combined.get(key)
            source_labels = {source.source_label}
            if current is None:
                combined[key] = CombinedHighlight(
                    text=entry.text,
                    color=entry.color,
                    page=entry.page,
                    source_labels=tuple(sorted(source_labels)),
                    duplicate_count=entry.duplicate_count,
                )
                continue
            color, page = choose_entry(current, entry)
            source_labels.update(current.source_labels)
            text = current.text if COLOR_PRIORITY.get(current.color, 0) >= COLOR_PRIORITY.get(entry.color, 0) else entry.text
            combined[key] = CombinedHighlight(
                text=text,
                color=color,
                page=page,
                source_labels=tuple(sorted(source_labels)),
                duplicate_count=current.duplicate_count + entry.duplicate_count,
            )
    return sorted(
        combined.values(),
        key=lambda item: (
            item.page is None,
            item.page or 10**9,
            -COLOR_PRIORITY.get(item.color, 0),
            item.text,
        ),
    )


def content_tokens(text: str) -> list[str]:
    tokens = normalize_text(text).split()
    return [
        token
        for token in tokens
        if token not in STOPWORDS and (len(token) >= 3 or token.isdigit())
    ]


def token_windows(tokens: list[str], *, size: int = 90, step: int = 35) -> list[set[str]]:
    if not tokens:
        return []
    if len(tokens) <= size:
        return [set(tokens)]
    windows: list[set[str]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            continue
        windows.append(set(window))
        if start + size >= len(tokens):
            break
    return windows


def has_exact_ngram(highlight_tokens: list[str], text_tokens: list[str]) -> bool:
    n = 6 if len(highlight_tokens) >= 12 else 4
    if len(highlight_tokens) < n or len(text_tokens) < n:
        return False
    text_ngrams = {
        tuple(text_tokens[index : index + n])
        for index in range(0, len(text_tokens) - n + 1)
    }
    return any(
        tuple(highlight_tokens[index : index + n]) in text_ngrams
        for index in range(0, len(highlight_tokens) - n + 1)
    )


def coverage_score(highlight_text: str, comparison_text: str) -> float:
    highlight_tokens = content_tokens(highlight_text)
    comparison_tokens = content_tokens(comparison_text)
    if not highlight_tokens or not comparison_tokens:
        return 0.0

    highlight_set = set(highlight_tokens)
    windows = token_windows(comparison_tokens)
    if not windows:
        return 0.0

    best = 0.0
    denominator = min(len(highlight_set), 24)
    for window in windows:
        overlap = len(highlight_set & window)
        if overlap == 0:
            continue
        capped_recall = overlap / denominator
        cosine = overlap / math.sqrt(max(len(highlight_set), 1) * max(len(window), 1))
        density = min(overlap / 8, 1.0)
        best = max(best, (0.72 * capped_recall) + (0.18 * cosine) + (0.10 * density))

    if has_exact_ngram(highlight_tokens, comparison_tokens):
        best = max(best, 0.95)
    return min(best, 1.0)


def score_highlights(
    highlights: list[CombinedHighlight],
    comparison_text: str,
    *,
    covered_threshold: float,
    partial_threshold: float,
) -> list[ScoredHighlight]:
    scored: list[ScoredHighlight] = []
    for highlight in highlights:
        score = coverage_score(highlight.text, comparison_text)
        if score >= covered_threshold:
            status = "covered"
        elif score >= partial_threshold:
            status = "partial"
        else:
            status = "missed"
        scored.append(ScoredHighlight(highlight=highlight, score=score, status=status))
    return scored


def summarize_scores(scored: list[ScoredHighlight]) -> dict[str, float | int]:
    count = len(scored)
    covered = sum(1 for item in scored if item.status == "covered")
    partial = sum(1 for item in scored if item.status == "partial")
    avg_score = sum(item.score for item in scored) / count if count else 0.0
    return {
        "highlight_count": count,
        "covered_count": covered,
        "partial_count": partial,
        "missed_count": count - covered - partial,
        "covered_rate": covered / count if count else 0.0,
        "covered_or_partial_rate": (covered + partial) / count if count else 0.0,
        "avg_score": avg_score,
    }


def color_counts(highlights: list[CombinedHighlight]) -> Counter[str]:
    return Counter(COLOR_LABELS.get(highlight.color, highlight.color) for highlight in highlights)


def source_counts(highlights: list[CombinedHighlight]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for highlight in highlights:
        for label in highlight.source_labels:
            counts[label] += 1
    return counts


def format_rate(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.1%}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summary_rows(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)

    aggregates: list[dict[str, object]] = []
    for group_key, group_rows in sorted(grouped.items()):
        highlights = sum(int(row["highlight_count"]) for row in group_rows)
        covered = sum(int(row["covered_count"]) for row in group_rows)
        partial = sum(int(row["partial_count"]) for row in group_rows)
        avg_score = (
            sum(float(row["avg_score"]) * int(row["highlight_count"]) for row in group_rows)
            / highlights
            if highlights
            else 0.0
        )
        aggregates.append(
            {
                key: group_key,
                "summary_count": len(group_rows),
                "highlight_count": highlights,
                "covered_count": covered,
                "partial_count": partial,
                "covered_rate": covered / highlights if highlights else 0.0,
                "covered_or_partial_rate": (covered + partial) / highlights
                if highlights
                else 0.0,
                "avg_score": avg_score,
            }
        )
    return aggregates


def aggregate_detail_rows(
    rows: list[dict[str, object]],
    *,
    key: str,
    variant: str | None = None,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if variant is not None and row["variant"] != variant:
            continue
        values = str(row[key]).split(";") if key == "source_labels" else [str(row[key])]
        for value in values:
            grouped[value].append(row)

    aggregates: list[dict[str, object]] = []
    for group_key, group_rows in sorted(grouped.items()):
        count = len(group_rows)
        covered = sum(1 for row in group_rows if row["status"] == "covered")
        partial = sum(1 for row in group_rows if row["status"] == "partial")
        avg_score = (
            sum(float(row["score"]) for row in group_rows) / count if count else 0.0
        )
        aggregates.append(
            {
                key: group_key,
                "highlight_count": count,
                "covered_count": covered,
                "partial_count": partial,
                "covered_rate": covered / count if count else 0.0,
                "covered_or_partial_rate": (covered + partial) / count
                if count
                else 0.0,
                "avg_score": avg_score,
            }
        )
    return aggregates


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def chunk_comparison_text(chunk_root: Path, book_id: str, highlights: list[CombinedHighlight]) -> str:
    chunk_files = sorted((chunk_root / book_id).glob("chunk_*_response.txt"))
    if not chunk_files:
        return ""

    entries = [
        HighlightEntry(text=item.text, color=item.color, page=item.page)
        for item in highlights
    ]
    annotated = annotate_entries_with_progress(entries, total_chunks=len(chunk_files))
    chunk_texts = {
        index: path.read_text(encoding="utf-8", errors="ignore")
        for index, path in enumerate(chunk_files, start=1)
    }
    parts: list[str] = []
    for item, entry in zip(highlights, annotated, strict=True):
        if entry.estimated_chunk is None:
            parts.append("\n".join(chunk_texts.values()))
        else:
            parts.append(chunk_texts.get(entry.estimated_chunk, ""))
    return "\n\n---HIGHLIGHT_COMPARISON_BOUNDARY---\n\n".join(parts)


def score_chunks(
    chunk_root: Path,
    book_id: str,
    highlights: list[CombinedHighlight],
    *,
    covered_threshold: float,
    partial_threshold: float,
) -> list[ScoredHighlight]:
    chunk_files = sorted((chunk_root / book_id).glob("chunk_*_response.txt"))
    if not chunk_files:
        return []

    entries = [
        HighlightEntry(text=item.text, color=item.color, page=item.page)
        for item in highlights
    ]
    annotated = annotate_entries_with_progress(entries, total_chunks=len(chunk_files))
    chunk_texts = {
        index: path.read_text(encoding="utf-8", errors="ignore")
        for index, path in enumerate(chunk_files, start=1)
    }
    all_chunks_text = "\n\n".join(chunk_texts.values())

    scored: list[ScoredHighlight] = []
    for item, entry in zip(highlights, annotated, strict=True):
        comparison_text = (
            chunk_texts.get(entry.estimated_chunk, "")
            if entry.estimated_chunk is not None
            else all_chunks_text
        )
        score = coverage_score(item.text, comparison_text)
        if score >= covered_threshold:
            status = "covered"
        elif score >= partial_threshold:
            status = "partial"
        else:
            status = "missed"
        scored.append(ScoredHighlight(highlight=item, score=score, status=status))
    return scored


def main() -> int:
    args = parse_args()
    sources = load_source_documents(
        drive_notes_root=args.drive_notes_root,
        tablet_md_root=args.tablet_md_root,
        curated_notes_root=args.curated_notes_root,
    )
    targets = load_summary_targets(args.summary_dir, include_expanded=args.include_expanded)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []

    highlights_by_book: dict[str, list[CombinedHighlight]] = {}
    matched_book_ids: set[str] = set()
    for target in targets:
        matches = matched_documents(target, sources, threshold=args.match_threshold)
        if not matches:
            match_rows.append(
                {
                    "book_id": target.book_id,
                    "variant": target.variant,
                    "title": target.title,
                    "matched_source_count": 0,
                    "best_match_title": "",
                    "best_match_score": "",
                }
            )
            continue

        matched_sources = [source for source, _score in matches]
        highlights = combine_highlights(matched_sources)
        highlights_by_book.setdefault(target.book_id, highlights)
        matched_book_ids.add(target.book_id)
        summary_text = target.path.read_text(encoding="utf-8", errors="ignore")
        scored = score_highlights(
            highlights,
            summary_text,
            covered_threshold=args.covered_threshold,
            partial_threshold=args.partial_threshold,
        )
        summary = summarize_scores(scored)
        colors = color_counts(highlights)
        source_counter = source_counts(highlights)
        best_source, best_score = matches[0]
        row = {
            "book_id": target.book_id,
            "variant": target.variant,
            "summary_file": str(target.path.relative_to(PROJECT_ROOT)),
            "title": target.title,
            "chunk_model": target.chunk_model,
            "synthesis_model": target.synthesis_model,
            "matched_source_count": len(matches),
            "best_match_title": best_source.document.title,
            "best_match_score": f"{best_score:.3f}",
            "highlight_count": summary["highlight_count"],
            "covered_count": summary["covered_count"],
            "partial_count": summary["partial_count"],
            "missed_count": summary["missed_count"],
            "covered_rate": f"{summary['covered_rate']:.6f}",
            "covered_or_partial_rate": f"{summary['covered_or_partial_rate']:.6f}",
            "avg_score": f"{summary['avg_score']:.6f}",
            "yellow_count": colors["yellow"],
            "green_count": colors["green"],
            "blue_count": colors["blue"],
            "red_count": colors["red"],
            "uncolored_count": colors["uncolored"],
            "drive_docx_highlights": source_counter["drive_docx"],
            "tablet_md_highlights": source_counter["tablet_md"],
            "curated_notes_highlights": source_counter["curated_notes"],
        }
        summary_rows.append(row)
        match_rows.append(
            {
                "book_id": target.book_id,
                "variant": target.variant,
                "title": target.title,
                "matched_source_count": len(matches),
                "best_match_title": best_source.document.title,
                "best_match_score": f"{best_score:.3f}",
            }
        )
        for index, scored_item in enumerate(scored, start=1):
            highlight = scored_item.highlight
            detail_rows.append(
                {
                    "book_id": target.book_id,
                    "variant": target.variant,
                    "highlight_index": index,
                    "color": COLOR_LABELS.get(highlight.color, highlight.color),
                    "page": highlight.page or "",
                    "source_labels": ";".join(highlight.source_labels),
                    "duplicate_count": highlight.duplicate_count,
                    "score": f"{scored_item.score:.6f}",
                    "status": scored_item.status,
                    "text": highlight.text,
                }
            )

    for book_id, highlights in sorted(highlights_by_book.items()):
        scored_chunks = score_chunks(
            args.chunk_root,
            book_id,
            highlights,
            covered_threshold=args.covered_threshold,
            partial_threshold=args.partial_threshold,
        )
        if not scored_chunks:
            continue
        summary = summarize_scores(scored_chunks)
        colors = color_counts(highlights)
        chunk_rows.append(
            {
                "book_id": book_id,
                "highlight_count": summary["highlight_count"],
                "covered_count": summary["covered_count"],
                "partial_count": summary["partial_count"],
                "missed_count": summary["missed_count"],
                "covered_rate": f"{summary['covered_rate']:.6f}",
                "covered_or_partial_rate": f"{summary['covered_or_partial_rate']:.6f}",
                "avg_score": f"{summary['avg_score']:.6f}",
                "yellow_count": colors["yellow"],
                "green_count": colors["green"],
                "blue_count": colors["blue"],
                "red_count": colors["red"],
                "uncolored_count": colors["uncolored"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "combined_highlight_coverage_summary.csv", summary_rows)
    write_csv(args.out_dir / "combined_highlight_coverage_details.csv", detail_rows)
    write_csv(args.out_dir / "combined_highlight_chunk_coverage.csv", chunk_rows)
    write_csv(args.out_dir / "combined_highlight_source_matches.csv", match_rows)

    variant_aggregates = aggregate_summary_rows(summary_rows, "variant")
    current_color_aggregates = aggregate_detail_rows(
        detail_rows,
        key="color",
        variant="current",
    )
    current_source_aggregates = aggregate_detail_rows(
        detail_rows,
        key="source_labels",
        variant="current",
    )
    comparable_book_ids = {
        book_id
        for book_id, count in Counter(row["book_id"] for row in summary_rows).items()
        if count > 1
    }
    comparable_rows = [
        row for row in summary_rows if str(row["book_id"]) in comparable_book_ids
    ]
    comparable_aggregates = aggregate_summary_rows(comparable_rows, "variant")

    report_lines = [
        "# Combined Highlight Coverage Metrics",
        "",
        "Inputs:",
        f"- Drive DOCX root: `{args.drive_notes_root}`",
        f"- Tablet markdown root: `{args.tablet_md_root}`",
        f"- Curated notes root: `{args.curated_notes_root}`",
        f"- Source documents parsed: {len(sources)}",
        f"- Summary files scanned: {len(targets)}",
        f"- Summary files evaluated with matched highlights: {len(summary_rows)}",
        f"- Summary files without matched highlights: {len(targets) - len(summary_rows)}",
        f"- Matched base books: {len(matched_book_ids)}",
        "",
        "Metric notes:",
        "- `covered` means the best local summary/chunk window crossed the deterministic lexical threshold.",
        "- `partial` means there was material overlap, but below the covered threshold.",
        "- Tablet markdown exports do not carry color/page metadata; tablet-only items are counted as uncolored.",
        "",
        "## Summary Coverage By Variant",
        "",
    ]
    report_lines.extend(
        markdown_table(
            [
                "variant",
                "summaries",
                "highlights",
                "covered",
                "partial+covered",
                "avg score",
            ],
            [
                [
                    row["variant"],
                    row["summary_count"],
                    row["highlight_count"],
                    format_rate(float(row["covered_rate"])),
                    format_rate(float(row["covered_or_partial_rate"])),
                    f"{float(row['avg_score']):.3f}",
                ]
                for row in variant_aggregates
            ],
        )
    )
    report_lines.extend(["", "## Prompt Variant Comparable Subset", ""])
    report_lines.append(
        "This subset only includes books with more than one saved summary variant."
    )
    report_lines.append("")
    report_lines.extend(
        markdown_table(
            [
                "variant",
                "summaries",
                "highlights",
                "covered",
                "partial+covered",
                "avg score",
            ],
            [
                [
                    row["variant"],
                    row["summary_count"],
                    row["highlight_count"],
                    format_rate(float(row["covered_rate"])),
                    format_rate(float(row["covered_or_partial_rate"])),
                    f"{float(row['avg_score']):.3f}",
                ]
                for row in comparable_aggregates
            ],
        )
    )
    report_lines.extend(["", "## Current Summary Coverage By Color", ""])
    report_lines.extend(
        markdown_table(
            [
                "color",
                "highlights",
                "covered",
                "partial+covered",
                "avg score",
            ],
            [
                [
                    row["color"],
                    row["highlight_count"],
                    format_rate(float(row["covered_rate"])),
                    format_rate(float(row["covered_or_partial_rate"])),
                    f"{float(row['avg_score']):.3f}",
                ]
                for row in current_color_aggregates
            ],
        )
    )
    report_lines.extend(["", "## Current Summary Coverage By Source", ""])
    report_lines.extend(
        markdown_table(
            [
                "source",
                "highlights",
                "covered",
                "partial+covered",
                "avg score",
            ],
            [
                [
                    row["source_labels"],
                    row["highlight_count"],
                    format_rate(float(row["covered_rate"])),
                    format_rate(float(row["covered_or_partial_rate"])),
                    f"{float(row['avg_score']):.3f}",
                ]
                for row in current_source_aggregates
            ],
        )
    )
    report_lines.extend(["", "## Book And Variant Detail", ""])
    report_lines.extend(
        markdown_table(
            [
                "book",
                "variant",
                "highlights",
                "covered",
                "partial+covered",
                "avg score",
            ],
            [
                [
                    row["book_id"],
                    row["variant"],
                    row["highlight_count"],
                    format_rate(float(row["covered_rate"])),
                    format_rate(float(row["covered_or_partial_rate"])),
                    f"{float(row['avg_score']):.3f}",
                ]
                for row in sorted(
                    summary_rows,
                    key=lambda item: (str(item["book_id"]), str(item["variant"])),
                )
            ],
        )
    )
    if chunk_rows:
        report_lines.extend(["", "## Chunk Extraction Coverage", ""])
        report_lines.extend(
            markdown_table(
                [
                    "book",
                    "highlights",
                    "covered",
                    "partial+covered",
                    "avg score",
                ],
                [
                    [
                        row["book_id"],
                        row["highlight_count"],
                        format_rate(float(row["covered_rate"])),
                        format_rate(float(row["covered_or_partial_rate"])),
                        f"{float(row['avg_score']):.3f}",
                    ]
                    for row in chunk_rows
                ],
            )
        )

    report_path = args.out_dir / "combined_highlight_coverage_report.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
