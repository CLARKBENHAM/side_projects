from __future__ import annotations

import argparse
import os
from pathlib import Path

from structured_summaries.goodreads_adapter import enrich_catalog_with_goodreads
from structured_summaries.takeout import (
    catalog_statistics,
    scan_takeout_root,
    write_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAKEOUT_ROOT = Path(
    os.environ.get(
        "PLAY_BOOKS_TAKEOUT_ROOT",
        PROJECT_ROOT / "data" / "Play Books Takeout" / "Google Play Books",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takeout-root", type=Path, default=DEFAULT_TAKEOUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "takeout_catalog.csv",
    )
    parser.add_argument("--enrich-goodreads", action="store_true")
    parser.add_argument(
        "--goodreads-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "takeout_catalog_goodreads.csv",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-goodreads-refresh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = scan_takeout_root(args.takeout_root)
    if args.limit is not None:
        records = records[: args.limit]
    write_catalog(records, args.output)
    stats = catalog_statistics(records)
    print(f"Saved catalog to {args.output}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")

    if args.enrich_goodreads:
        enrich_catalog_with_goodreads(
            args.output,
            args.goodreads_output,
            limit=args.limit,
            force_refresh=args.force_goodreads_refresh,
            verbose=args.verbose,
        )
        print(f"Saved Goodreads enrichment to {args.goodreads_output}")


if __name__ == "__main__":
    main()
