from __future__ import annotations

import argparse
from pathlib import Path

from structured_summaries.metadata_selection import (
    DEFAULT_FICTION_SHELVES,
    build_catalog_metadata_links,
    load_read_overrides,
    load_master_metadata,
    select_high_signal_read_links,
    select_unread_local_links,
    write_link_rows,
)
from structured_summaries.takeout import load_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "takeout_catalog.csv",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT.parent
        / "data"
        / "Books Read and their effects - master_book_metadata_cleaned_final.csv",
    )
    parser.add_argument(
        "--join-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "catalog_metadata_join.csv",
    )
    parser.add_argument(
        "--read-overrides-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "manual_read_overrides.csv",
    )
    parser.add_argument(
        "--read-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "read_prompt_eval_candidates.csv",
    )
    parser.add_argument(
        "--unread-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "unread_local_processing_queue.csv",
    )
    parser.add_argument("--min-signal-score", type=float, default=4.0)
    parser.add_argument(
        "--exclude-shelf",
        dest="exclude_shelves",
        action="append",
        default=list(DEFAULT_FICTION_SHELVES),
    )
    parser.add_argument("--read-limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_catalog(args.catalog_csv)
    metadata_rows = load_master_metadata(args.metadata_csv)
    read_overrides = load_read_overrides(args.read_overrides_csv)
    links = build_catalog_metadata_links(
        records,
        metadata_rows,
        read_overrides=read_overrides,
    )

    read_links = select_high_signal_read_links(
        links,
        min_signal_score=args.min_signal_score,
        fiction_shelves=tuple(args.exclude_shelves),
    )
    if args.read_limit:
        read_links = read_links[: args.read_limit]
    unread_links = select_unread_local_links(links)

    write_link_rows(links, args.join_output)
    write_link_rows(read_links, args.read_output)
    write_link_rows(unread_links, args.unread_output)

    matched_count = sum(1 for link in links if link.is_matched)
    read_matched_count = sum(1 for link in links if link.is_matched and link.is_read)
    print(f"Catalog records: {len(records)}")
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Manual read overrides: {len(read_overrides)}")
    print(f"Matched catalog records: {matched_count}")
    print(f"Matched read catalog records: {read_matched_count}")
    print(f"High-signal read candidates: {len(read_links)}")
    print(f"Unread local queue size: {len(unread_links)}")
    print(f"Join output: {args.join_output}")
    print(f"Read candidates output: {args.read_output}")
    print(f"Unread queue output: {args.unread_output}")
    if read_links:
        print("Top read candidates:")
        for link in read_links[:10]:
            print(
                f"- {link.book.book_id} | {link.book.title} | "
                f"{link.bookshelf} | enjoyment={link.avg_enjoyment:.3g} | "
                f"usefulness={link.avg_usefulness:.3g}"
            )


if __name__ == "__main__":
    main()
