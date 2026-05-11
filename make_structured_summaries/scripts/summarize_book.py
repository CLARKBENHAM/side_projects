from __future__ import annotations

import argparse
from pathlib import Path

from structured_summaries.llm_backends import default_model_for_backend
from structured_summaries.models import BookRecord
from structured_summaries.pipeline import SummaryConfig, summarize_book
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
    parser.add_argument("--chunk-backend", default="gemini")
    parser.add_argument("--chunk-model")
    parser.add_argument("--synthesis-backend", default="claude")
    parser.add_argument("--synthesis-model")
    parser.add_argument("--critique-backend", default="claude")
    parser.add_argument("--critique-model")
    parser.add_argument("--chunk-chars", type=int, default=28_000)
    parser.add_argument("--overlap-chars", type=int, default=1_500)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-critique", action="store_true")
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
        llm_timeout_seconds=args.timeout_seconds,
        force=args.force,
        dry_run=args.dry_run,
    )
    artifacts = summarize_book(book, project_root=PROJECT_ROOT, config=config)
    print(f"Extracted text: {artifacts.extracted_text_path}")
    print(f"Chunk dir: {artifacts.chunk_dir}")
    print(f"Expanded notes path: {artifacts.expanded_notes_path or 'not written'}")
    print(f"Summary path: {artifacts.summary_path or 'dry-run only'}")
    print(f"Critique path: {artifacts.critique_path or 'not written'}")
    print(f"Manifest path: {artifacts.manifest_path}")


if __name__ == "__main__":
    main()
