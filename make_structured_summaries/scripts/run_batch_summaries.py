from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from structured_summaries.llm_backends import default_model_for_backend
from structured_summaries.models import BookRecord
from structured_summaries.pipeline import (
    SummaryConfig,
    summarize_book,
    summarize_books_async,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _parse_int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "takeout_catalog.csv",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--book-ids", nargs="*")
    parser.add_argument("--chunk-backend", default="gemini")
    parser.add_argument("--chunk-model")
    parser.add_argument("--synthesis-backend", default="claude")
    parser.add_argument("--synthesis-model")
    parser.add_argument("--critique-backend", default="claude")
    parser.add_argument("--critique-model")
    parser.add_argument("--chunk-chars", type=int, default=28_000)
    parser.add_argument("--overlap-chars", type=int, default=1_500)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--chunk-concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--mode",
        choices=("sync", "async"),
        default="sync",
    )
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--book-concurrency", type=int, default=5)
    parser.add_argument("--requests-per-window", type=int, default=16)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-critique", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> list[dict[str, str]]:
    if args.book_ids:
        selected = [row for row in rows if row.get("book_id") in set(args.book_ids)]
        return selected[: args.limit]
    return sorted(
        rows,
        key=lambda row: (
            _parse_int(row.get("goodreads_rating_count")),
            _parse_float(row.get("goodreads_rating")),
            row.get("title", ""),
        ),
        reverse=True,
    )[: args.limit]


def main() -> None:
    args = parse_args()
    rows = select_rows(load_rows(args.catalog_csv), args)
    config = SummaryConfig(
        chunk_backend=args.chunk_backend,
        chunk_model=args.chunk_model or default_model_for_backend(args.chunk_backend),
        synthesis_backend=args.synthesis_backend,
        synthesis_model=args.synthesis_model
        or default_model_for_backend(args.synthesis_backend),
        critique_backend=None if args.skip_critique else args.critique_backend,
        critique_model=(
            None
            if args.skip_critique
            else args.critique_model or default_model_for_backend(args.critique_backend)
        ),
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        max_chunks=args.max_chunks,
        chunk_concurrency=args.chunk_concurrency,
        llm_timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
        force=args.force,
    )
    books = [BookRecord.from_row(row) for row in rows]
    if args.mode == "async":
        results = asyncio.run(
            summarize_books_async(
                books,
                project_root=PROJECT_ROOT,
                config=config,
                max_concurrency=args.max_concurrency,
                requests_per_window=args.requests_per_window,
                window_seconds=args.window_seconds,
                book_concurrency=args.book_concurrency,
            )
        )
        for book, artifacts in results:
            print(f"{book.book_id}: {artifacts.summary_path or 'dry-run only'}")
        return

    for book in books:
        artifacts = summarize_book(book, project_root=PROJECT_ROOT, config=config)
        print(f"{book.book_id}: {artifacts.summary_path or 'dry-run only'}")


if __name__ == "__main__":
    main()
