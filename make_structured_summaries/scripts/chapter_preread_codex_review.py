from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from html import escape
from pathlib import Path

from scripts.evaluate_highlight_coverage import (
    COLOR_LABELS,
    SIDE_PROJECTS_ROOT,
    SourceDocument,
    combine_highlights,
    iter_existing_files,
    title_key,
)
from structured_summaries.chapter_preread import (
    ChapterHighlight,
    ChapterSpan,
    extract_chapters,
    place_highlights_by_chapter,
    render_highlights_markdown,
    split_long_chapters,
)
from structured_summaries.highlights import load_highlight_document
from structured_summaries.llm_backends import (
    PromptExecution,
    default_model_for_backend,
    run_prompt_with_metadata,
)
from structured_summaries.models import BookRecord
from structured_summaries.prompts import (
    build_chapter_highlight_judge_prompt,
    build_chapter_preread_summary_prompt,
)
from structured_summaries.takeout import SUPPORTED_EXTENSIONS, load_catalog
from structured_summaries.utils import normalize_text, slugify

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TITLE_PATTERNS = (
    "Master of the Senate",
    "The Price of Victory",
    "A History of the English-Speaking Peoples",
    "Great Contemporaries",
    "My Early Life",
    "Thoughts and Adventures",
)


@dataclass(frozen=True)
class ChapterCandidate:
    book: BookRecord
    chapter: ChapterSpan
    highlights: tuple[ChapterHighlight, ...]
    total_chapters: int
    matched_highlight_titles: tuple[str, ...]


@dataclass(frozen=True)
class ChapterRunResult:
    candidate: ChapterCandidate
    sample_slug: str
    sample_dir: Path
    summary_path: Path | None
    judge_path: Path | None
    html_path: Path
    summary_execution: PromptExecution | None
    judge_execution: PromptExecution | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample chapters, summarize them with headless Codex, judge against "
            "the reader's real highlights, and render static review HTML."
        )
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "takeout_catalog.csv",
    )
    parser.add_argument("--book-id", action="append", default=[])
    parser.add_argument("--book-path", action="append", type=Path, default=[])
    parser.add_argument(
        "--book-title",
        action="append",
        default=[],
        help="Title override for each --book-path, in the same order.",
    )
    parser.add_argument(
        "--book-author",
        action="append",
        default=[],
        help="Author override for each --book-path, in the same order.",
    )
    parser.add_argument("--title", action="append", default=[])
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
        default=PROJECT_ROOT / "data" / "eval_reviews" / "chapter_preread_codex_review",
    )
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--min-highlights", type=int, default=2)
    parser.add_argument("--min-chapter-chars", type=int, default=2_500)
    parser.add_argument("--fallback-chars", type=int, default=60_000)
    parser.add_argument("--max-chapter-chars", type=int, default=110_000)
    parser.add_argument("--match-threshold", type=float, default=0.84)
    parser.add_argument("--backend", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--judge-backend")
    parser.add_argument("--judge-model")
    parser.add_argument("--timeout-seconds", type=int, default=1_200)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of chapter prompt jobs to run concurrently.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-empty-highlight-chapters",
        action="store_true",
        help=(
            "Generate summary-only pages for books with no matched reader highlight "
            "exports. Books with matched highlights still honor --min-highlights."
        ),
    )
    parser.add_argument(
        "--include-all-chapters",
        action="store_true",
        help="Generate pages for every selected chapter, regardless of highlight coverage.",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Build candidates for every selected book before sampling.",
    )
    return parser.parse_args()


def selected_title_patterns(args: argparse.Namespace) -> tuple[str, ...]:
    if args.book_id or args.book_path:
        return tuple(args.title)
    return tuple(args.title) if args.title else DEFAULT_TITLE_PATTERNS


def strict_title_similarity(left: str, right: str) -> float:
    left_key = normalize_text(left)
    right_key = normalize_text(right)
    if not left_key or not right_key:
        return 0.0
    left_compare = left_key.removeprefix("the ")
    right_compare = right_key.removeprefix("the ")
    if left_compare == right_compare:
        return 1.0
    if left_key == right_key:
        return 1.0
    if min(len(left_key), len(right_key)) >= 18 and (
        left_key in right_key or right_key in left_key
    ):
        return 0.95
    return SequenceMatcher(None, left_key, right_key).ratio()


def is_low_value_back_matter(chapter: ChapterSpan) -> bool:
    title_key_value = normalize_text(chapter.title)
    back_matter_markers = (
        "author s note",
        "bibliography",
        "chapter references",
        "recommended for further reading",
        "index",
        "list of figures",
        "list of sidebars",
        "back cover",
    )
    return any(marker in title_key_value for marker in back_matter_markers)


def title_selected(record: BookRecord, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    for pattern in patterns:
        if strict_title_similarity(record.title, pattern) >= 0.84:
            return True
    return False


def load_selected_books(args: argparse.Namespace) -> list[BookRecord]:
    records = load_catalog(args.catalog_csv)
    selected: list[BookRecord] = []
    if args.book_id:
        wanted = set(args.book_id)
        selected.extend(record for record in records if record.book_id in wanted)
        missing = wanted - {record.book_id for record in selected}
        if missing:
            raise SystemExit(f"Book id not found in catalog: {', '.join(sorted(missing))}")
    if args.book_path:
        selected.extend(load_path_books(args))
        return selected
    if selected:
        return selected
    patterns = selected_title_patterns(args)
    return [record for record in records if title_selected(record, patterns)]


def title_from_book_path(path: Path) -> str:
    stem = path.stem.replace("_", " ").strip()
    if " - " in stem:
        return stem.split(" - ", 1)[0].strip()
    return " ".join(stem.split())


def load_path_books(args: argparse.Namespace) -> list[BookRecord]:
    if args.book_title and len(args.book_title) != len(args.book_path):
        raise SystemExit("--book-title must be supplied once for each --book-path")
    if args.book_author and len(args.book_author) != len(args.book_path):
        raise SystemExit("--book-author must be supplied once for each --book-path")

    books: list[BookRecord] = []
    for index, raw_path in enumerate(args.book_path):
        path = raw_path.expanduser().resolve()
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise SystemExit(f"Unsupported book path type: {path}")
        title = (
            args.book_title[index].strip()
            if args.book_title
            else title_from_book_path(path)
        )
        author = args.book_author[index].strip() if args.book_author else ""
        books.append(
            BookRecord(
                book_id=slugify(f"{title}-{author}") if author else slugify(title),
                title=title,
                author=author,
                book_dir=path.parent,
                primary_path=path,
                primary_format=SUPPORTED_EXTENSIONS[suffix],
                companion_html_path=None,
                all_paths=(path,),
                search_title=title,
                source_name="book_path",
            )
        )
    return books


def matched_sources_for_book(
    book: BookRecord,
    sources: list,
    *,
    threshold: float,
) -> list:
    matches = [
        source
        for source in sources
        if strict_title_similarity(book.title, source.document.title) >= threshold
    ]
    return sorted(
        matches,
        key=lambda source: strict_title_similarity(book.title, source.document.title),
        reverse=True,
    )


def source_path_matches_book(path: Path, book: BookRecord) -> bool:
    path_key = normalize_text(path.stem)
    title = book.title
    title_key_value = normalize_text(title)
    title_without_article = title_key_value.removeprefix("the ")
    search_title = normalize_text(book.search_title)
    if title_key_value and title_key_value in path_key:
        return True
    if title_without_article and title_without_article in path_key:
        return True
    if search_title and search_title in path_key:
        return True
    if strict_title_similarity(path.stem, title) >= 0.72:
        return True
    if book.search_title and strict_title_similarity(path.stem, book.search_title) >= 0.72:
        return True
    return False


def load_relevant_source_documents(
    args: argparse.Namespace,
    books: list[BookRecord],
) -> list[SourceDocument]:
    source_roots: list[tuple[str, Path, tuple[str, ...]]] = [
        ("drive_docx", args.drive_notes_root, ("*.docx",)),
        ("tablet_md", args.tablet_md_root, ("*.md",)),
        ("curated_notes", args.curated_notes_root, ("*.txt", "*.md", "*.docx")),
    ]
    documents: list[SourceDocument] = []
    for source_label, root, patterns in source_roots:
        for path in iter_existing_files(root, patterns):
            if not args.scan_all and not any(
                source_path_matches_book(path, book) for book in books
            ):
                continue
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


def build_candidates(args: argparse.Namespace) -> list[ChapterCandidate]:
    books = load_selected_books(args)
    if not args.scan_all:
        random.Random(args.seed).shuffle(books)
    sources = load_relevant_source_documents(args, books)
    candidates: list[ChapterCandidate] = []
    candidate_book_ids: set[str] = set()
    for book in books:
        matched_sources = matched_sources_for_book(
            book,
            sources,
            threshold=args.match_threshold,
        )
        if not matched_sources and not args.include_empty_highlight_chapters:
            continue
        highlights = combine_highlights(matched_sources)
        chapters = extract_chapters(
            book,
            min_chapter_chars=args.min_chapter_chars,
            fallback_chars=args.fallback_chars,
        )
        chapters = split_long_chapters(chapters, max_chars=args.max_chapter_chars)
        if not chapters:
            continue
        placed = place_highlights_by_chapter(highlights, chapters)
        matched_titles = tuple(
            sorted({source.document.title for source in matched_sources})
        )
        for chapter in chapters:
            chapter_highlights = tuple(placed.get(chapter.index, []))
            should_include = len(chapter_highlights) >= args.min_highlights
            if args.include_all_chapters:
                should_include = True
            elif args.include_empty_highlight_chapters and not matched_sources:
                should_include = True
                if is_low_value_back_matter(chapter):
                    should_include = False
            if not should_include:
                continue
            candidates.append(
                ChapterCandidate(
                    book=book,
                    chapter=chapter,
                    highlights=chapter_highlights,
                    total_chapters=len(chapters),
                    matched_highlight_titles=matched_titles,
                )
            )
            candidate_book_ids.add(book.book_id)
        if (
            not args.scan_all
            and len(candidate_book_ids) >= args.sample_count
            and len(candidates) >= args.sample_count
        ):
            break
    return candidates


def sample_candidates(
    candidates: list[ChapterCandidate],
    *,
    sample_count: int,
    seed: int,
) -> list[ChapterCandidate]:
    rng = random.Random(seed)
    by_book: dict[str, list[ChapterCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_book[candidate.book.book_id].append(candidate)
    for book_candidates in by_book.values():
        rng.shuffle(book_candidates)

    selected: list[ChapterCandidate] = []
    book_ids = sorted(by_book)
    while len(selected) < sample_count and book_ids:
        progressed = False
        rng.shuffle(book_ids)
        for book_id in list(book_ids):
            if len(selected) >= sample_count:
                break
            book_candidates = by_book[book_id]
            if not book_candidates:
                book_ids.remove(book_id)
                continue
            selected.append(book_candidates.pop())
            progressed = True
        if not progressed:
            break
    return selected


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def maybe_run_prompt(
    *,
    backend: str,
    model: str | None,
    prompt: str,
    prompt_path: Path,
    response_path: Path,
    timeout: int,
    force: bool,
    dry_run: bool,
) -> tuple[str, PromptExecution | None]:
    write_text(prompt_path, prompt)
    if dry_run:
        return "", None
    if response_path.exists() and not force:
        return response_path.read_text(encoding="utf-8"), None
    execution = run_prompt_with_metadata(
        backend,
        prompt,
        model=model,
        timeout=timeout,
    )
    write_text(response_path, execution.output)
    return execution.output, execution


def color_label(color: str) -> str:
    return COLOR_LABELS.get(color, color or "uncolored")


def render_highlights_html(highlights: tuple[ChapterHighlight, ...]) -> str:
    if not highlights:
        return '<p class="empty">No mapped highlights for this chapter.</p>'
    cards: list[str] = []
    for index, highlight in enumerate(highlights, start=1):
        page = f"p{highlight.page}" if highlight.page is not None else "no page"
        sources = ", ".join(highlight.source_labels)
        color = color_label(highlight.color)
        cards.append(
            f"""
            <article class="highlight-card {escape(color)}">
              <div class="meta">
                <span>#{index}</span>
                <span>{escape(color)}</span>
                <span>{escape(page)}</span>
                <span>{escape(highlight.method)}</span>
                <span>{escape(sources)}</span>
              </div>
              <p>{escape(highlight.text)}</p>
            </article>
            """
        )
    return "\n".join(cards)


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    html: list[str] = []
    list_open = False
    ordered_open = False

    def close_lists() -> None:
        nonlocal list_open, ordered_open
        if list_open:
            html.append("</ul>")
            list_open = False
        if ordered_open:
            html.append("</ol>")
            ordered_open = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            close_lists()
            continue
        if line.startswith("### "):
            close_lists()
            html.append(f"<h3>{escape(line[4:])}</h3>")
            continue
        if line.startswith("## "):
            close_lists()
            html.append(f"<h2>{escape(line[3:])}</h2>")
            continue
        if line.startswith("# "):
            close_lists()
            html.append(f"<h1>{escape(line[2:])}</h1>")
            continue
        if line.startswith(("- ", "* ")):
            if ordered_open:
                html.append("</ol>")
                ordered_open = False
            if not list_open:
                html.append("<ul>")
                list_open = True
            html.append(f"<li>{escape(line[2:])}</li>")
            continue
        if len(line) >= 3 and line[0].isdigit() and ". " in line[:5]:
            if list_open:
                html.append("</ul>")
                list_open = False
            if not ordered_open:
                html.append("<ol>")
                ordered_open = True
            html.append(f"<li>{escape(line.split('. ', 1)[1])}</li>")
            continue
        close_lists()
        html.append(f"<p>{escape(line)}</p>")
    close_lists()
    return "\n".join(html)


def render_sample_page(result: ChapterRunResult) -> str:
    candidate = result.candidate
    summary = (
        result.summary_path.read_text(encoding="utf-8")
        if result.summary_path and result.summary_path.exists()
        else ""
    )
    judge = (
        result.judge_path.read_text(encoding="utf-8")
        if result.judge_path and result.judge_path.exists()
        else ""
    )
    judge_empty_text = (
        "No mapped highlights for this chapter, so no judge report was run."
        if not candidate.highlights
        else "Dry run only."
    )
    source_titles = ", ".join(candidate.matched_highlight_titles)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(candidate.book.title)} - {escape(candidate.chapter.title)}</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d9d9d2;
      --panel: #fff;
      --yellow: #fff2a8;
      --green: #d8f3dc;
      --blue: #dbeafe;
      --red: #ffe0e0;
      --uncolored: #f1f3f4;
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
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 247, 244, 0.97);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px;
    }}
    h1 {{ margin: 4px 0; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 18px 0 8px; font-size: 18px; letter-spacing: 0; }}
    h3 {{ margin: 14px 0 6px; font-size: 15px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; }}
    code {{ font-size: 12px; overflow-wrap: anywhere; }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .meta span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      background: rgba(255, 255, 255, 0.7);
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.05fr) minmax(0, 1.05fr);
      gap: 12px;
      padding: 14px 18px 36px;
      align-items: start;
    }}
    section {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 12px;
    }}
    section > h2:first-child {{ margin-top: 0; }}
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
    .empty {{ color: var(--muted); font-style: italic; }}
    ul, ol {{ padding-left: 22px; }}
    li {{ margin-bottom: 6px; }}
    @media (max-width: 1100px) {{
      header {{ position: static; }}
      main {{ grid-template-columns: 1fr; padding: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <a href="index.html">Index</a>
    <h1>{escape(candidate.book.title)}</h1>
    <p>{escape(candidate.chapter.title)}</p>
    <div class="meta">
      <span>chapter {candidate.chapter.index} of {candidate.total_chapters}</span>
      <span>{len(candidate.chapter.text):,} chars</span>
      <span>{len(candidate.highlights)} mapped highlights</span>
      <span>source: {escape(candidate.chapter.source_label)}</span>
    </div>
    <p><code>{escape(str(candidate.book.primary_path))}</code></p>
    <p>Highlight exports matched: {escape(source_titles)}</p>
  </header>
  <main>
    <section>
      <h2>Your Highlights</h2>
      {render_highlights_html(candidate.highlights)}
    </section>
    <section>
      <h2>AI Pre-Read Summary</h2>
      {markdown_to_html(summary) if summary else '<p class="empty">Dry run only.</p>'}
    </section>
    <section>
      <h2>AI Judge Report</h2>
      {markdown_to_html(judge) if judge else f'<p class="empty">{escape(judge_empty_text)}</p>'}
    </section>
  </main>
</body>
</html>
"""


def render_index(results: list[ChapterRunResult]) -> str:
    rows = []
    for result in results:
        candidate = result.candidate
        rows.append(
            f"""
            <tr>
              <td><a href="{escape(result.html_path.name)}">{escape(candidate.book.title)}</a></td>
              <td>{escape(candidate.chapter.title)}</td>
              <td>{candidate.chapter.index} / {candidate.total_chapters}</td>
              <td>{len(candidate.highlights)}</td>
              <td>{len(candidate.chapter.text):,}</td>
              <td><code>{escape(display_path(result.sample_dir))}</code></td>
            </tr>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chapter Pre-Read Codex Review</title>
  <style>
    body {{
      margin: 0;
      background: #f7f7f4;
      color: #202124;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{ padding: 18px 22px; border-bottom: 1px solid #d9d9d2; }}
    main {{ padding: 18px 22px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid #d9d9d2;
    }}
    th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #d9d9d2; vertical-align: top; }}
    th {{ color: #5f6368; font-size: 13px; background: #fafafa; }}
    code {{ font-size: 12px; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>Chapter Pre-Read Codex Review</h1>
    <p>Each page compares one AI chapter pre-read summary against your mapped highlights and an AI judge report.</p>
  </header>
  <main>
    <table>
      <thead>
        <tr>
          <th>Book</th>
          <th>Chapter</th>
          <th>Position</th>
          <th>Highlights</th>
          <th>Chars</th>
          <th>Artifacts</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""


def run_sample(
    candidate: ChapterCandidate,
    *,
    out_dir: Path,
    backend: str,
    model: str | None,
    judge_backend: str,
    judge_model: str | None,
    timeout: int,
    force: bool,
    dry_run: bool,
) -> ChapterRunResult:
    sample_slug = (
        f"{slugify(candidate.book.title)}-chapter-{candidate.chapter.index:03d}"
    )
    sample_dir = out_dir / "artifacts" / sample_slug
    summary_prompt = build_chapter_preread_summary_prompt(
        candidate.book,
        chapter_title=candidate.chapter.title,
        chapter_index=candidate.chapter.index,
        total_chapters=candidate.total_chapters,
        chapter_text=candidate.chapter.text,
    )
    highlights_markdown = render_highlights_markdown(list(candidate.highlights))
    write_text(sample_dir / "highlights.md", highlights_markdown)
    write_text(sample_dir / "chapter_excerpt_for_prompt.txt", candidate.chapter.text)
    summary, summary_execution = maybe_run_prompt(
        backend=backend,
        model=model,
        prompt=summary_prompt,
        prompt_path=sample_dir / "summary_prompt.txt",
        response_path=sample_dir / "summary.md",
        timeout=timeout,
        force=force,
        dry_run=dry_run,
    )
    judge_execution = None
    judge_path = None
    if candidate.highlights:
        judge_prompt = build_chapter_highlight_judge_prompt(
            candidate.book,
            chapter_title=candidate.chapter.title,
            chapter_index=candidate.chapter.index,
            total_chapters=candidate.total_chapters,
            summary_markdown=summary,
            highlights_markdown=highlights_markdown,
        )
        _judge, judge_execution = maybe_run_prompt(
            backend=judge_backend,
            model=judge_model,
            prompt=judge_prompt,
            prompt_path=sample_dir / "judge_prompt.txt",
            response_path=sample_dir / "judge.md",
            timeout=timeout,
            force=force,
            dry_run=dry_run,
        )
        judge_path = None if dry_run else sample_dir / "judge.md"
    result = ChapterRunResult(
        candidate=candidate,
        sample_slug=sample_slug,
        sample_dir=sample_dir,
        summary_path=None if dry_run else sample_dir / "summary.md",
        judge_path=judge_path,
        html_path=out_dir / f"{sample_slug}.html",
        summary_execution=summary_execution,
        judge_execution=judge_execution,
    )
    write_text(result.html_path, render_sample_page(result))
    return result


def execution_record(execution: PromptExecution | None) -> dict[str, object] | None:
    if execution is None:
        return None
    return {
        "command": list(execution.command),
        "started_at": execution.started_at,
        "finished_at": execution.finished_at,
        "duration_seconds": round(execution.duration_seconds, 3),
    }


def chapter_record(chapter: ChapterSpan) -> dict[str, object]:
    return {
        "index": chapter.index,
        "title": chapter.title,
        "source_label": chapter.source_label,
        "start_char": chapter.start_char,
        "end_char": chapter.end_char,
        "char_count": len(chapter.text),
    }


def write_manifest(
    *,
    args: argparse.Namespace,
    candidates: list[ChapterCandidate],
    selected: list[ChapterCandidate],
    results: list[ChapterRunResult],
    index_path: Path,
) -> Path:
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sample_count": args.sample_count,
            "seed": args.seed,
            "min_highlights": args.min_highlights,
            "min_chapter_chars": args.min_chapter_chars,
            "max_chapter_chars": args.max_chapter_chars,
            "backend": args.backend,
            "model": args.model or default_model_for_backend(args.backend),
            "judge_backend": args.judge_backend or args.backend,
            "judge_model": args.judge_model
            or args.model
            or default_model_for_backend(args.judge_backend or args.backend),
            "workers": args.workers,
            "dry_run": args.dry_run,
        },
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "index_path": str(index_path),
        "samples": [
            {
                "book": result.candidate.book.to_row(),
                "chapter": chapter_record(result.candidate.chapter),
                "highlight_count": len(result.candidate.highlights),
                "matched_highlight_titles": result.candidate.matched_highlight_titles,
                "sample_dir": str(result.sample_dir),
                "html_path": str(result.html_path),
                "summary_execution": execution_record(result.summary_execution),
                "judge_execution": execution_record(result.judge_execution),
            }
            for result in results
        ],
    }
    return write_text(args.out_dir / "manifest.json", json.dumps(manifest, indent=2))


def run_samples(
    selected: list[ChapterCandidate],
    *,
    args: argparse.Namespace,
    backend: str,
    model: str | None,
    judge_backend: str,
    judge_model: str | None,
) -> list[ChapterRunResult]:
    if args.workers <= 1 or len(selected) <= 1:
        return [
            run_sample(
                candidate,
                out_dir=args.out_dir,
                backend=backend,
                model=model,
                judge_backend=judge_backend,
                judge_model=judge_model,
                timeout=args.timeout_seconds,
                force=args.force,
                dry_run=args.dry_run,
            )
            for candidate in selected
        ]

    results: list[ChapterRunResult | None] = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_sample,
                candidate,
                out_dir=args.out_dir,
                backend=backend,
                model=model,
                judge_backend=judge_backend,
                judge_model=judge_model,
                timeout=args.timeout_seconds,
                force=args.force,
                dry_run=args.dry_run,
            ): index
            for index, candidate in enumerate(selected)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return [result for result in results if result is not None]


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args)
    if not candidates:
        raise SystemExit(
            "No candidate chapters found. Try lowering --min-highlights, raising "
            "--max-chapter-chars, or passing --book-id for a known local book."
        )
    selected = sample_candidates(
        candidates,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    backend = args.backend
    model = args.model or default_model_for_backend(backend)
    judge_backend = args.judge_backend or backend
    judge_model = (
        args.judge_model
        or args.model
        or default_model_for_backend(judge_backend)
    )
    results = run_samples(
        selected,
        args=args,
        backend=backend,
        model=model,
        judge_backend=judge_backend,
        judge_model=judge_model,
    )
    index_path = write_text(args.out_dir / "index.html", render_index(results))
    manifest_path = write_manifest(
        args=args,
        candidates=candidates,
        selected=selected,
        results=results,
        index_path=index_path,
    )
    print(f"Candidate chapters: {len(candidates)}")
    print(f"Selected chapters: {len(selected)}")
    print(f"Index: {index_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
