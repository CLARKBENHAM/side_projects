from __future__ import annotations

import argparse
from pathlib import Path

from structured_summaries.chunk_companion import render_expanded_chunk_notes
from structured_summaries.models import BookRecord
from structured_summaries.takeout import build_record_from_path, load_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "takeout_catalog.csv",
    )
    parser.add_argument("--book-id")
    parser.add_argument("--book-path", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_book(args: argparse.Namespace) -> BookRecord:
    if args.book_path:
        return build_record_from_path(args.book_path.expanduser().resolve())
    if not args.book_id:
        raise SystemExit("Provide either --book-id or --book-path")
    for record in load_catalog(args.catalog_csv):
        if record.book_id == args.book_id:
            return record
    raise SystemExit(f"Book id not found in catalog: {args.book_id}")


def main() -> None:
    args = parse_args()
    book = load_book(args)
    chunk_dir = PROJECT_ROOT / "data" / "chunk_notes" / book.book_id
    if not chunk_dir.exists():
        raise SystemExit(f"Chunk directory not found: {chunk_dir}")
    chunk_files = sorted(chunk_dir.glob("chunk_*_response.txt"))
    if not chunk_files:
        raise SystemExit(f"No chunk response files found in: {chunk_dir}")

    output_path = args.output_file or (
        PROJECT_ROOT / "data" / "summaries" / f"{book.book_id}.expanded.md"
    )
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Output already exists: {output_path}. Re-run with --force to overwrite."
        )

    chunk_outputs = [
        path.read_text(encoding="utf-8", errors="ignore") for path in chunk_files
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_expanded_chunk_notes(book, chunk_outputs),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
