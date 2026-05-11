"""Merge Goodreads enrichment and second-pass ratings into a copy of books_enriched.csv."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
ENRICHED_CSV = BASE_DIR / "books_enriched.csv"
GOODREADS_CSV = BASE_DIR / "books_goodreads.csv"
RATINGS2_CSV = DATA_DIR / "Books Read and their effects - Ratings 2.csv"
OUTPUT_CSV = BASE_DIR / "books_enriched_with_goodreads.csv"
MIN_CONFIDENT_GOODREADS_SCORE = 0.8
MIN_RESCUE_BASE_SCORE = 0.45
MIN_RESCUE_TITLE_SIMILARITY = 0.30
MIN_RESCUE_AUTHOR_SIMILARITY = 0.50


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def build_ratings2_frame(ratings2_csv: Path) -> pd.DataFrame:
    df = load_csv(ratings2_csv)
    keep = [
        "title",
        "author",
        "Bookshelf",
        "earliest_modified",
        "latest_modified",
        "filename",
        "Enjoyment (/5)",
        "Usefulness /5 to Me",
    ]
    available = [column for column in keep if column in df.columns]
    df = df[available].copy()
    rename_map = {
        "author": "author_ratings2",
        "Bookshelf": "Bookshelf_ratings2",
        "earliest_modified": "earliest_modified_ratings2",
        "latest_modified": "latest_modified_ratings2",
        "filename": "filename_ratings2",
        "Enjoyment (/5)": "Enjoyment (/5)_ratings2",
        "Usefulness /5 to Me": "Usefulness /5 to Me_ratings2",
    }
    return df.rename(columns=rename_map)


def build_goodreads_frame(goodreads_csv: Path) -> pd.DataFrame:
    df = load_csv(goodreads_csv)
    keep = [
        "title",
        "author",
        "search_title",
        "goodreads_status",
        "goodreads_match_method",
        "goodreads_url",
        "goodreads_title",
        "goodreads_author",
        "goodreads_rating",
        "goodreads_rating_count",
        "goodreads_match_score",
        "goodreads_title_similarity",
        "goodreads_author_similarity",
        "goodreads_query",
        "goodreads_candidates_json",
    ]
    available = [column for column in keep if column in df.columns]
    return df[available].copy()


def to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_candidates_json(value: object) -> list[dict[str, object]]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def choose_best_raw_candidate(row: pd.Series) -> pd.Series:
    if pd.notna(row.get("goodreads_rating_raw")) and pd.notna(
        row.get("goodreads_rating_count_raw")
    ):
        return pd.Series(
            {
                "goodreads_rating_raw_best": row.get("goodreads_rating_raw"),
                "goodreads_rating_count_raw_best": row.get(
                    "goodreads_rating_count_raw"
                ),
                "goodreads_url_raw_best": row.get("goodreads_url"),
                "goodreads_title_raw_best": row.get("goodreads_title"),
                "goodreads_author_raw_best": row.get("goodreads_author"),
                "goodreads_raw_best_source": "chosen_candidate",
                "goodreads_raw_best_score": row.get("goodreads_match_score"),
            }
        )

    eligible: list[dict[str, object]] = []
    for candidate in parse_candidates_json(row.get("goodreads_candidates_json")):
        rating_value = to_float(candidate.get("rating_value"))
        rating_count = to_int(candidate.get("rating_count"))
        base_score = to_float(candidate.get("score")) or 0.0
        title_similarity = to_float(candidate.get("title_similarity")) or 0.0
        author_similarity = to_float(candidate.get("author_similarity")) or 0.0

        if not rating_value or rating_value <= 0:
            continue
        if not rating_count or rating_count <= 0:
            continue
        if base_score < MIN_RESCUE_BASE_SCORE:
            continue
        if title_similarity < MIN_RESCUE_TITLE_SIMILARITY:
            continue
        if author_similarity < MIN_RESCUE_AUTHOR_SIMILARITY:
            continue

        rescue_score = base_score + min(math.log10(rating_count + 1) / 8, 0.5)
        eligible.append(
            {
                "goodreads_rating_raw_best": rating_value,
                "goodreads_rating_count_raw_best": rating_count,
                "goodreads_url_raw_best": candidate.get("url"),
                "goodreads_title_raw_best": candidate.get("page_title"),
                "goodreads_author_raw_best": candidate.get("page_author"),
                "goodreads_raw_best_source": "rescued_candidate",
                "goodreads_raw_best_score": round(rescue_score, 4),
                "_rescue_sort": (rescue_score, rating_count, title_similarity),
            }
        )

    if not eligible:
        return pd.Series(
            {
                "goodreads_rating_raw_best": pd.NA,
                "goodreads_rating_count_raw_best": pd.NA,
                "goodreads_url_raw_best": pd.NA,
                "goodreads_title_raw_best": pd.NA,
                "goodreads_author_raw_best": pd.NA,
                "goodreads_raw_best_source": "missing",
                "goodreads_raw_best_score": pd.NA,
            }
        )

    best = max(eligible, key=lambda item: item["_rescue_sort"])
    best.pop("_rescue_sort")
    return pd.Series(best)


def apply_goodreads_confidence_gate(df: pd.DataFrame) -> pd.DataFrame:
    gated = df.copy()
    if "goodreads_rating" in gated.columns:
        gated["goodreads_rating_raw"] = gated["goodreads_rating"]
        gated.loc[gated["goodreads_rating"].fillna(0).le(0), "goodreads_rating"] = pd.NA
    if "goodreads_rating_count" in gated.columns:
        gated["goodreads_rating_count_raw"] = gated["goodreads_rating_count"]
        gated.loc[
            gated["goodreads_rating_count"].fillna(0).le(0), "goodreads_rating_count"
        ] = pd.NA

    if "goodreads_rating_raw" in gated.columns:
        gated.loc[
            gated["goodreads_rating_raw"].fillna(0).le(0), "goodreads_rating_raw"
        ] = pd.NA
    if "goodreads_rating_count_raw" in gated.columns:
        gated.loc[
            gated["goodreads_rating_count_raw"].fillna(0).le(0),
            "goodreads_rating_count_raw",
        ] = pd.NA

    if "goodreads_status" not in gated.columns:
        return gated

    confirmed = gated["goodreads_status"].eq("matched")
    if "goodreads_match_score" in gated.columns:
        confirmed &= (
            gated["goodreads_match_score"].fillna(0).ge(MIN_CONFIDENT_GOODREADS_SCORE)
        )
    for column in ["goodreads_rating", "goodreads_rating_count"]:
        if column in gated.columns:
            gated.loc[~confirmed, column] = pd.NA

    rescued = gated.apply(choose_best_raw_candidate, axis=1)
    gated = pd.concat([gated, rescued], axis=1)
    return gated


def merge_into_copy(
    enriched_csv: Path = ENRICHED_CSV,
    goodreads_csv: Path = GOODREADS_CSV,
    ratings2_csv: Path = RATINGS2_CSV,
    output_csv: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    enriched = load_csv(enriched_csv)
    goodreads = build_goodreads_frame(goodreads_csv)
    ratings2 = build_ratings2_frame(ratings2_csv)

    merged = enriched.merge(
        goodreads,
        on=["title"],
        how="left",
        suffixes=("", "_goodreads"),
    )
    merged = merged.merge(ratings2, on="title", how="left")
    merged = apply_goodreads_confidence_gate(merged)
    merged.to_csv(output_csv, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched-csv", type=Path, default=ENRICHED_CSV)
    parser.add_argument("--goodreads-csv", type=Path, default=GOODREADS_CSV)
    parser.add_argument("--ratings2-csv", type=Path, default=RATINGS2_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_into_copy(
        enriched_csv=args.enriched_csv,
        goodreads_csv=args.goodreads_csv,
        ratings2_csv=args.ratings2_csv,
        output_csv=args.output_csv,
    )
    print(f"Saved merged copy to {args.output_csv}")
    print(
        merged[
            [
                "title",
                "goodreads_status",
                "goodreads_rating",
                "goodreads_rating_raw",
                "goodreads_rating_count",
                "Enjoyment (/5)_ratings2",
                "Usefulness /5 to Me_ratings2",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
