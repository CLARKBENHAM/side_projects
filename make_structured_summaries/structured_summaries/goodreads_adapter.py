"""Adapter for reusing the existing Goodreads enrichment code."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .takeout import load_catalog

PROJECT_PARENT = Path(__file__).resolve().parents[2]


def enrich_catalog_with_goodreads(
    catalog_csv: Path,
    output_csv: Path,
    *,
    limit: int | None = None,
    force_refresh: bool = False,
    sleep_seconds: float = 0.1,
    verbose: bool = False,
) -> Path:
    if str(PROJECT_PARENT) not in sys.path:
        sys.path.insert(0, str(PROJECT_PARENT))

    import pandas as pd
    from ai_books_tracking.goodreads_ratings import enrich_books_with_goodreads

    records = load_catalog(catalog_csv)
    rows = [record.to_goodreads_row() for record in records]
    if limit is not None:
        rows = rows[:limit]
    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enrich_books_with_goodreads(
        books=frame,
        output_path=output_csv,
        limit=None,
        title_filter=None,
        force_refresh=force_refresh,
        sleep_seconds=sleep_seconds,
        verbose=verbose,
    )
    return output_csv


def merge_goodreads_columns(
    base_catalog_csv: Path,
    goodreads_csv: Path,
    merged_output_csv: Path,
) -> Path:
    with base_catalog_csv.open(encoding="utf-8", newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    with goodreads_csv.open(encoding="utf-8", newline="") as handle:
        goodreads_rows = {row["canonical_key"]: row for row in csv.DictReader(handle)}

    merged_rows: list[dict[str, str]] = []
    for row in base_rows:
        canonical_key = next(
            record.to_goodreads_row()["canonical_key"]
            for record in load_catalog(base_catalog_csv)
            if record.book_id == row["book_id"]
        )
        merged_rows.append({**row, **goodreads_rows.get(canonical_key, {})})

    fieldnames = sorted({key for row in merged_rows for key in row})
    merged_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with merged_output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return merged_output_csv
