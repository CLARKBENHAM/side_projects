"""Build a final cleaned master metadata CSV with personal ratings attached.

The source `master_book_metadata_cleaned.csv` is treated as immutable. This
script appends matched personal-rating fields into a separate output file
without rewriting the source master.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_books_tracking.goodreads_followup_analysis import derive_analysis_columns
from ai_books_tracking.reconcile_cleaned_goodreads_metadata import (
    DATASET_SPECS,
    build_unique_lookup,
    load_csv,
    normalize_compact_key,
    normalize_filename_key,
    normalize_title_key,
    prepare_target_frame,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"

MASTER_CLEANED_CSV = (
    DATA_DIR / "Books Read and their effects - master_book_metadata_cleaned.csv"
)
HISTORICAL_CURATED_CSV = BASE_DIR / "books_enriched_with_goodreads_curated.csv"
HOLDOUT_CURATED_CSV = BASE_DIR / "new_books_to_rate_2026_enriched_curated.csv"
HISTORICAL_FALLBACK_CSV = BASE_DIR / "books_enriched_with_goodreads.csv"
HOLDOUT_FALLBACK_CSV = BASE_DIR / "new_books_to_rate_2026_enriched.csv"
OUTPUT_CSV = (
    DATA_DIR / "Books Read and their effects - master_book_metadata_cleaned_final.csv"
)

GOODREADS_RATING_COLUMNS = ("ratings", "goodread ratings")
GOODREADS_REVIEW_COLUMNS = ("number reviews", "goodreads number reviews")
GOODREADS_COUNT_COLUMNS = ("number ratings", "goodreads number ratings")

PERSONAL_OUTPUT_COLUMNS = {
    "title": "personal_rating_title",
    "Bookshelf": "bookshelf",
    "finished_date_best": "finished_date",
    "Enjoyment (/5)": "enjoyment_pass1",
    "Usefulness /5 to Me": "usefulness_pass1",
    "Enjoyment (/5)_ratings2": "enjoyment_pass2",
    "Usefulness /5 to Me_ratings2": "usefulness_pass2",
    "avg_enjoyment": "avg_enjoyment",
    "avg_usefulness": "avg_usefulness",
    "enjoyment_label_gap": "enjoyment_label_gap",
    "usefulness_label_gap": "usefulness_label_gap",
    "goodreads_cleaned_status": "goodreads_analysis_status",
}


def prepare_master(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared.columns = prepared.columns.str.strip()
    prepared["title_key"] = prepared["title"].map(normalize_title_key)
    prepared["title_compact_key"] = prepared["title"].map(normalize_compact_key)
    prepared["filename_key"] = prepared["filename"].map(normalize_filename_key)
    return prepared


def resolve_enriched_input(preferred: Path, fallback: Path) -> Path:
    if preferred.exists():
        return preferred
    return fallback


def first_present_value(row: pd.Series, candidates: tuple[str, ...]) -> object:
    for candidate in candidates:
        if candidate in row.index:
            value = row[candidate]
            if pd.notna(value):
                return value
    return pd.NA


def finished_date_best(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    if "date_finished" in frame.columns:
        result = pd.to_datetime(frame["date_finished"], format="mixed", errors="coerce")
    for column in ["latest_modified_best", "latest_modified"]:
        if column in frame.columns:
            result = result.combine_first(
                pd.to_datetime(frame[column], format="mixed", errors="coerce")
            )
    return result


def load_enriched(path: Path, spec_name: str) -> pd.DataFrame:
    spec = next(spec for spec in DATASET_SPECS if spec.name == spec_name)
    df = load_csv(path)
    for column, default in [
        ("Long Term Effects", ""),
        ("author", ""),
        ("author_ratings2", ""),
        ("goodreads_author", ""),
        ("gb_page_count", pd.NA),
        ("pub_year", pd.NA),
    ]:
        if column not in df.columns:
            df[column] = default
    if "earliest_modified" not in df.columns and "latest_modified" in df.columns:
        df["earliest_modified"] = df["latest_modified"]
    if (
        "earliest_modified_ratings2" not in df.columns
        and "latest_modified" in df.columns
    ):
        df["earliest_modified_ratings2"] = df["latest_modified"]
    if "latest_modified_ratings2" not in df.columns and "latest_modified" in df.columns:
        df["latest_modified_ratings2"] = df["latest_modified"]
    if "latest_modified" not in df.columns and "date_finished" in df.columns:
        df["latest_modified"] = df["date_finished"]
    if "earliest_modified" not in df.columns and "date_finished" in df.columns:
        df["earliest_modified"] = df["date_finished"]
    if "earliest_modified_ratings2" not in df.columns and "date_finished" in df.columns:
        df["earliest_modified_ratings2"] = df["date_finished"]
    if "latest_modified_ratings2" not in df.columns and "date_finished" in df.columns:
        df["latest_modified_ratings2"] = df["date_finished"]
    df = derive_analysis_columns(df)
    df = prepare_target_frame(df, spec)
    df["finished_date_best"] = finished_date_best(df).dt.strftime("%Y-%m-%d")
    return df


def build_enriched_lookups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "filename_key": build_unique_lookup(df, "filename_key"),
        "title_key": build_unique_lookup(df, "title_key"),
        "title_compact_key": build_unique_lookup(df, "title_compact_key"),
        "title_compact_key_for_match": build_unique_lookup(
            df, "title_compact_key_for_match"
        ),
    }


def match_personal_row(
    row: pd.Series,
    lookups: dict[str, pd.DataFrame],
    *,
    allow_filename_match: bool,
) -> tuple[pd.Series | None, str]:
    filename_key = row.get("filename_key", "")
    if (
        allow_filename_match
        and filename_key
        and filename_key in lookups["filename_key"].index
    ):
        return lookups["filename_key"].loc[filename_key], "filename_key"

    title_key = row.get("title_key", "")
    if title_key and title_key in lookups["title_key"].index:
        return lookups["title_key"].loc[title_key], "title_key"

    compact_key = row.get("title_compact_key", "")
    if compact_key and compact_key in lookups["title_compact_key"].index:
        return lookups["title_compact_key"].loc[compact_key], "compact_title_key"
    if compact_key and compact_key in lookups["title_compact_key_for_match"].index:
        return (
            lookups["title_compact_key_for_match"].loc[compact_key],
            "manual_alias",
        )
    return None, "unmatched"


def source_personal_ratings(
    master: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    source_name: str,
    allow_filename_match: bool,
) -> pd.DataFrame:
    lookups = build_enriched_lookups(enriched)

    additions: list[dict[str, object]] = []
    for _, row in master.iterrows():
        if row.get("source") != source_name:
            additions.append(
                {
                    "personal_ratings_match_found": pd.NA,
                    "personal_ratings_match_method": pd.NA,
                    **{value: pd.NA for value in PERSONAL_OUTPUT_COLUMNS.values()},
                }
            )
            continue

        matched, match_method = match_personal_row(
            row, lookups, allow_filename_match=allow_filename_match
        )
        row_dict: dict[str, object] = {
            "personal_ratings_match_found": matched is not None,
            "personal_ratings_match_method": match_method,
        }
        for source_col, target_col in PERSONAL_OUTPUT_COLUMNS.items():
            row_dict[target_col] = matched[source_col] if matched is not None else pd.NA
        additions.append(row_dict)

    return pd.DataFrame(additions, index=master.index)


def build_final_master(
    master_csv: Path = MASTER_CLEANED_CSV,
    historical_csv: Path = resolve_enriched_input(
        HISTORICAL_CURATED_CSV, HISTORICAL_FALLBACK_CSV
    ),
    holdout_csv: Path = resolve_enriched_input(
        HOLDOUT_CURATED_CSV, HOLDOUT_FALLBACK_CSV
    ),
) -> pd.DataFrame:
    master = prepare_master(load_csv(master_csv))
    historical = load_enriched(historical_csv, "historical")
    holdout = load_enriched(holdout_csv, "holdout_2026")

    historical_personal = source_personal_ratings(
        master,
        historical,
        source_name="Play Export",
        allow_filename_match=True,
    )
    holdout_personal = source_personal_ratings(
        master,
        holdout,
        source_name="Holdout 2026",
        allow_filename_match=False,
    )
    combined = master.copy()
    for column in historical_personal.columns:
        combined[column] = historical_personal[column].combine_first(
            holdout_personal[column]
        )

    combined["all_ratings_summary"] = combined.apply(
        lambda row: "|".join(
            [
                f"gr={first_present_value(row, GOODREADS_RATING_COLUMNS)}",
                f"gr_reviews={first_present_value(row, GOODREADS_REVIEW_COLUMNS)}",
                f"gr_ratings={first_present_value(row, GOODREADS_COUNT_COLUMNS)}",
                f"e1={row['enjoyment_pass1']}",
                f"u1={row['usefulness_pass1']}",
                f"e2={row['enjoyment_pass2']}",
                f"u2={row['usefulness_pass2']}",
                f"avg_e={row['avg_enjoyment']}",
                f"avg_u={row['avg_usefulness']}",
            ]
        ),
        axis=1,
    )

    helper_columns = [
        "title_key",
        "title_compact_key",
        "filename_key",
    ]
    return combined.drop(
        columns=[col for col in helper_columns if col in combined.columns]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", type=Path, default=MASTER_CLEANED_CSV)
    parser.add_argument(
        "--historical-csv",
        type=Path,
        default=resolve_enriched_input(HISTORICAL_CURATED_CSV, HISTORICAL_FALLBACK_CSV),
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=resolve_enriched_input(HOLDOUT_CURATED_CSV, HOLDOUT_FALLBACK_CSV),
    )
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_master = build_final_master(
        master_csv=args.master_csv,
        historical_csv=args.historical_csv,
        holdout_csv=args.holdout_csv,
    )
    final_master.to_csv(args.output_csv, index=False)

    matched = (
        final_master["personal_ratings_match_found"].astype("boolean").fillna(False)
    )
    print(f"Saved final master metadata to {args.output_csv}")
    print(
        f"Matched personal ratings for {int(matched.sum())}/{len(final_master)} rows "
        f"({matched.mean():.1%})"
    )


if __name__ == "__main__":
    main()
