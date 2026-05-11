from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ai_books_tracking.goodread_emperical_dist.config import (
    GOODREADS_BOOK_RATING_COUNTS_CSV,
    OVERLAP_BOOKS_CSV,
    PREPARED_PROFILE_BOOKS_CSV,
)
from ai_books_tracking.goodreads_ratings import (
    PAGE_CACHE_FILE,
    fetch_goodreads_page,
    load_cache,
    save_cache,
)


def _book_id_from_goodreads_url(value: object) -> str:
    match = re.search(r"/book/show/(\d+)", str(value))
    return match.group(1) if match else ""


def _rating_count_rows_from_page_cache(page_cache: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cache_key, cached_value in page_cache.items():
        if not isinstance(cached_value, dict):
            continue
        book_id = _book_id_from_goodreads_url(cache_key) or _book_id_from_goodreads_url(
            cached_value.get("url", "")
        )
        if not book_id:
            continue
        rows.append(
            {
                "book_id": book_id,
                "goodreads_rating_count": cached_value.get("rating_count"),
                "goodreads_rating_value": cached_value.get("rating_value"),
                "goodreads_page_title": cached_value.get("page_title", ""),
                "goodreads_page_author": cached_value.get("page_author", ""),
                "source": "goodreads_page_cache",
                "fetched_at_utc": "",
            }
        )
    return _clean_rating_count_rows(pd.DataFrame(rows))


def _clean_rating_count_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cleaned = frame.copy()
    cleaned["book_id"] = cleaned["book_id"].astype(str)
    cleaned["goodreads_rating_count"] = pd.to_numeric(
        cleaned["goodreads_rating_count"], errors="coerce"
    )
    cleaned["goodreads_rating_value"] = pd.to_numeric(
        cleaned.get("goodreads_rating_value"), errors="coerce"
    )
    cleaned = cleaned.dropna(subset=["book_id", "goodreads_rating_count"])
    cleaned = cleaned[cleaned["goodreads_rating_count"].gt(0)]
    cleaned = cleaned.sort_values(
        ["book_id", "source", "goodreads_rating_count"],
        ascending=[True, True, False],
    )
    return cleaned.drop_duplicates(subset=["book_id"], keep="last").reset_index(
        drop=True
    )


def _load_existing_rating_count_cache(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    return _clean_rating_count_rows(pd.read_csv(output_path))


def _candidate_book_ids(candidate_source: str) -> list[str]:
    if candidate_source == "overlap" and OVERLAP_BOOKS_CSV.exists():
        overlap = pd.read_csv(OVERLAP_BOOKS_CSV, usecols=["book_id", "n_raters"])
        return (
            overlap.assign(book_id=lambda frame: frame["book_id"].astype(str))
            .sort_values(["n_raters", "book_id"], ascending=[False, True])["book_id"]
            .drop_duplicates()
            .tolist()
        )

    prepared = pd.read_csv(PREPARED_PROFILE_BOOKS_CSV, usecols=["book_id"])
    prepared["book_id"] = prepared["book_id"].astype(str)
    return (
        prepared.groupby("book_id", as_index=False)
        .size()
        .sort_values(["size", "book_id"], ascending=[False, True])["book_id"]
        .tolist()
    )


def build_goodreads_book_rating_count_cache(
    output_path: Path = GOODREADS_BOOK_RATING_COUNTS_CSV,
    page_cache_path: Path = PAGE_CACHE_FILE,
    fetch_missing_limit: int = 0,
    sleep_seconds: float = 0.1,
    candidate_source: str = "overlap",
    verbose: bool = False,
) -> pd.DataFrame:
    page_cache = load_cache(page_cache_path)
    frames = [
        _load_existing_rating_count_cache(output_path),
        _rating_count_rows_from_page_cache(page_cache),
    ]
    current = _clean_rating_count_rows(pd.concat(frames, ignore_index=True))
    known_book_ids = set(current["book_id"]) if not current.empty else set()

    if fetch_missing_limit > 0:
        session = requests.Session()
        fetched_rows: list[dict[str, object]] = []
        candidates = [
            book_id
            for book_id in _candidate_book_ids(candidate_source)
            if book_id not in known_book_ids
        ][:fetch_missing_limit]
        timestamp = datetime.now(timezone.utc).isoformat()
        for index, book_id in enumerate(candidates, start=1):
            page_data = fetch_goodreads_page(
                f"https://www.goodreads.com/book/show/{book_id}",
                session=session,
                page_cache=page_cache,
                sleep_seconds=sleep_seconds,
            )
            fetched_rows.append(
                {
                    "book_id": book_id,
                    "goodreads_rating_count": page_data.get("rating_count"),
                    "goodreads_rating_value": page_data.get("rating_value"),
                    "goodreads_page_title": page_data.get("page_title", ""),
                    "goodreads_page_author": page_data.get("page_author", ""),
                    "source": "goodreads_page_fetch",
                    "fetched_at_utc": timestamp,
                }
            )
            if verbose and index % 25 == 0:
                print(f"Fetched {index}/{len(candidates)} Goodreads book pages")
            if index % 25 == 0:
                save_cache(page_cache, page_cache_path)
                current = _clean_rating_count_rows(
                    pd.concat(
                        [current, pd.DataFrame(fetched_rows)],
                        ignore_index=True,
                    )
                )
                current.to_csv(output_path, index=False)

        save_cache(page_cache, page_cache_path)
        if fetched_rows:
            current = _clean_rating_count_rows(
                pd.concat([current, pd.DataFrame(fetched_rows)], ignore_index=True)
            )

    current.to_csv(output_path, index=False)
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=GOODREADS_BOOK_RATING_COUNTS_CSV)
    parser.add_argument("--page-cache", type=Path, default=PAGE_CACHE_FILE)
    parser.add_argument("--fetch-missing-limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument(
        "--candidate-source",
        choices=["overlap", "prepared"],
        default="overlap",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    output = build_goodreads_book_rating_count_cache(
        output_path=args.output,
        page_cache_path=args.page_cache,
        fetch_missing_limit=args.fetch_missing_limit,
        sleep_seconds=args.sleep_seconds,
        candidate_source=args.candidate_source,
        verbose=args.verbose,
    )
    count_total = (
        int(output["goodreads_rating_count"].notna().sum())
        if "goodreads_rating_count" in output
        else 0
    )
    print(
        f"Wrote {len(output)} book rating-count rows to {args.output} "
        f"({count_total} with counts)"
    )


if __name__ == "__main__":
    main()
