"""Apply curated Goodreads metadata to the read-book datasets.

This script uses `master_book_metadata_cleaned.csv` as the trusted source for
Goodreads rating, review count, and rating count on already-read books. It:

1. Matches the curated rows onto the historical read dataset and the 2026
   holdout dataset.
2. Preserves the old scraped Goodreads fields for auditability.
3. Writes curated copies plus per-row comparison reports.
4. Optionally backs up and replaces the default enriched CSVs used by the
   downstream analysis scripts.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"

CLEANED_MASTER_CSV = (
    DATA_DIR / "Books Read and their effects - master_book_metadata_cleaned.csv"
)
HISTORICAL_INPUT_CSV = BASE_DIR / "books_enriched_with_goodreads.csv"
HOLDOUT_INPUT_CSV = BASE_DIR / "new_books_to_rate_2026_enriched.csv"

HISTORICAL_CURATED_CSV = BASE_DIR / "books_enriched_with_goodreads_curated.csv"
HOLDOUT_CURATED_CSV = BASE_DIR / "new_books_to_rate_2026_enriched_curated.csv"

HISTORICAL_DIFF_CSV = BASE_DIR / "goodreads_cleaned_diff_historical.csv"
HOLDOUT_DIFF_CSV = BASE_DIR / "goodreads_cleaned_diff_holdout.csv"
SUMMARY_CSV = BASE_DIR / "goodreads_cleaned_diff_summary.csv"

FILE_SUFFIXES = (".pdf", ".epub", ".html", ".txt")
HOLDOUT_SOURCE = "Holdout 2026"
HISTORICAL_SOURCE = "Play Export"
DEFAULT_BACKUP_STAMP = date.today().isoformat()
DEFAULT_TITLE_ALIASES = {
    "buckley": "buckleythelifeandtherevolutionthatchangedamerica",
    "davidfosterwallace": "davidfosterwallacethelastinterview",
    "dfwtv": "eunibuspluramtelevisionandusfiction",
    "kelly": "kellymorethanmyshareofitall",
    "kellymyshareofitall": "kellymorethanmyshareofitall",
    "nowitcanbetold": "nowitcanbetoldthestoryofthemanhattanproject",
    "shapingup": "shapeupstoprunningincirclesandshipworkthatmatters",
}

GOODREADS_COLUMN_ALIASES = {
    "ratings": ("ratings", "goodread ratings"),
    "number reviews": ("number reviews", "goodreads number reviews"),
    "number ratings": ("number ratings", "goodreads number ratings"),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    cleaned_source: str
    input_csv: Path
    curated_csv: Path
    diff_csv: Path
    allow_filename_match: bool
    title_column: str = "title"
    filename_column: str | None = None


DATASET_SPECS = (
    DatasetSpec(
        name="historical",
        cleaned_source=HISTORICAL_SOURCE,
        input_csv=HISTORICAL_INPUT_CSV,
        curated_csv=HISTORICAL_CURATED_CSV,
        diff_csv=HISTORICAL_DIFF_CSV,
        allow_filename_match=True,
        filename_column="filename_ratings2",
    ),
    DatasetSpec(
        name="holdout_2026",
        cleaned_source=HOLDOUT_SOURCE,
        input_csv=HOLDOUT_INPUT_CSV,
        curated_csv=HOLDOUT_CURATED_CSV,
        diff_csv=HOLDOUT_DIFF_CSV,
        allow_filename_match=False,
    ),
)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def strip_file_suffix(value: str) -> str:
    lowered = value
    for suffix in FILE_SUFFIXES:
        if lowered.lower().endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def normalize_title_key(value: object) -> str:
    text = normalize_text(value).replace("’", "'")
    text = strip_file_suffix(text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def normalize_compact_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_title_key(value))


def normalize_filename_key(value: object) -> str:
    text = normalize_text(value).replace("’", "'")
    text = strip_file_suffix(text)
    text = re.sub(r"\(\d+\)$", "", text.strip())
    text = re.sub(r"\[[^\]]+\]$", "", text.strip())
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def similarity(left: object, right: object) -> float:
    left_key = normalize_compact_key(left)
    right_key = normalize_compact_key(right)
    if not left_key or not right_key:
        return 0.0
    return SequenceMatcher(None, left_key, right_key).ratio()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def ensure_cleaned_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for canonical, aliases in GOODREADS_COLUMN_ALIASES.items():
        if canonical in normalized.columns:
            continue
        for alias in aliases:
            if alias in normalized.columns:
                normalized[canonical] = normalized[alias]
                break
    return normalized


def prepare_cleaned_frame(path: Path = CLEANED_MASTER_CSV) -> pd.DataFrame:
    df = ensure_cleaned_numeric_columns(load_csv(path))
    for column in ["ratings", "number reviews", "number ratings"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["title_key"] = df["title"].map(normalize_title_key)
    df["title_compact_key"] = df["title"].map(normalize_compact_key)
    df["filename_key"] = df["filename"].map(normalize_filename_key)
    return df


def prepare_target_frame(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    prepared = df.copy()
    prepared["title_key"] = prepared[spec.title_column].map(normalize_title_key)
    prepared["title_compact_key"] = prepared[spec.title_column].map(
        normalize_compact_key
    )
    if spec.filename_column and spec.filename_column in prepared.columns:
        prepared["filename_key"] = prepared[spec.filename_column].map(
            normalize_filename_key
        )
    else:
        prepared["filename_key"] = ""
    prepared["title_compact_key_for_match"] = prepared["title_compact_key"].replace(
        DEFAULT_TITLE_ALIASES
    )
    return prepared


def build_unique_lookup(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    valid = df[df[key_col].ne("")].copy()
    counts = valid[key_col].value_counts()
    unique_keys = counts[counts == 1].index
    return valid[valid[key_col].isin(unique_keys)].set_index(key_col, drop=False)


def match_curated_rows(
    target: pd.DataFrame, cleaned: pd.DataFrame, spec: DatasetSpec
) -> pd.DataFrame:
    clean_subset = cleaned[cleaned["source"] == spec.cleaned_source].copy()
    by_filename = build_unique_lookup(clean_subset, "filename_key")
    by_title = build_unique_lookup(clean_subset, "title_key")
    by_compact = build_unique_lookup(clean_subset, "title_compact_key")

    matched_rows: list[dict[str, object]] = []
    for row in target.itertuples(index=False):
        matched = None
        match_method = "unmatched"

        filename_key = getattr(row, "filename_key", "")
        if (
            spec.allow_filename_match
            and filename_key
            and filename_key in by_filename.index
        ):
            matched = by_filename.loc[filename_key]
            match_method = "filename_key"
        else:
            title_key = getattr(row, "title_key")
            compact_key = getattr(row, "title_compact_key")
            compact_match_key = getattr(row, "title_compact_key_for_match")
            if title_key and title_key in by_title.index:
                matched = by_title.loc[title_key]
                match_method = "title_key"
            elif compact_key and compact_key in by_compact.index:
                matched = by_compact.loc[compact_key]
                match_method = "compact_title_key"
            elif compact_match_key and compact_match_key in by_compact.index:
                matched = by_compact.loc[compact_match_key]
                match_method = "manual_alias"

        row_dict = {
            "goodreads_cleaned_row_found": matched is not None,
            "goodreads_cleaned_match_method": match_method,
            "goodreads_title_cleaned": pd.NA,
            "goodreads_author_cleaned": pd.NA,
            "goodreads_rating_cleaned": pd.NA,
            "goodreads_review_count_cleaned": pd.NA,
            "goodreads_rating_count_cleaned": pd.NA,
            "goodreads_cleaned_source": spec.cleaned_source,
            "goodreads_cleaned_filename": pd.NA,
        }
        if matched is not None:
            row_dict.update(
                {
                    "goodreads_title_cleaned": matched["title"],
                    "goodreads_author_cleaned": matched["corrected_author"],
                    "goodreads_rating_cleaned": matched["ratings"],
                    "goodreads_review_count_cleaned": matched["number reviews"],
                    "goodreads_rating_count_cleaned": matched["number ratings"],
                    "goodreads_cleaned_filename": matched["filename"],
                }
            )
        matched_rows.append(row_dict)

    matched_frame = pd.DataFrame(matched_rows, index=target.index)
    matched_frame["goodreads_cleaned_has_value"] = matched_frame[
        "goodreads_rating_cleaned"
    ].notna()
    matched_frame["goodreads_cleaned_status"] = np.select(
        [
            matched_frame["goodreads_cleaned_has_value"],
            matched_frame["goodreads_cleaned_row_found"],
        ],
        ["matched", "no_value"],
        default="not_in_clean_file",
    )
    return matched_frame


def preserve_old_scraped_columns(df: pd.DataFrame) -> pd.DataFrame:
    preserved = df.copy()
    rename_pairs = {
        "goodreads_rating": "goodreads_rating_old_scraped",
        "goodreads_rating_count": "goodreads_rating_count_old_scraped",
        "goodreads_rating_raw": "goodreads_rating_raw_old_scraped",
        "goodreads_rating_count_raw": "goodreads_rating_count_raw_old_scraped",
        "goodreads_rating_raw_best": "goodreads_rating_raw_best_old_scraped",
        "goodreads_rating_count_raw_best": "goodreads_rating_count_raw_best_old_scraped",
        "goodreads_status": "goodreads_status_old_scraped",
        "goodreads_match_method": "goodreads_match_method_old_scraped",
        "goodreads_title": "goodreads_title_old_scraped",
        "goodreads_author": "goodreads_author_old_scraped",
        "goodreads_match_score": "goodreads_match_score_old_scraped",
    }
    for src, dst in rename_pairs.items():
        if src in preserved.columns and dst not in preserved.columns:
            preserved[dst] = preserved[src]
    return preserved


def apply_curated_values(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    curated = preserve_old_scraped_columns(df)
    curated["goodreads_rating"] = curated["goodreads_rating_cleaned"]
    curated["goodreads_rating_count"] = curated["goodreads_rating_count_cleaned"]
    curated["goodreads_rating_raw"] = curated["goodreads_rating_cleaned"]
    curated["goodreads_rating_count_raw"] = curated["goodreads_rating_count_cleaned"]
    curated["goodreads_rating_raw_best"] = curated["goodreads_rating_cleaned"]
    curated["goodreads_rating_count_raw_best"] = curated[
        "goodreads_rating_count_cleaned"
    ]
    curated["goodreads_status"] = np.where(
        curated["goodreads_cleaned_has_value"],
        "matched",
        np.where(
            curated["goodreads_cleaned_row_found"],
            "no_value",
            "unmatched",
        ),
    )
    curated["goodreads_raw_best_source"] = np.where(
        curated["goodreads_cleaned_has_value"],
        "master_book_metadata_cleaned",
        "missing",
    )
    curated["goodreads_match_method"] = curated["goodreads_cleaned_match_method"]
    curated["goodreads_title"] = curated["goodreads_title_cleaned"]
    curated["goodreads_author"] = curated["goodreads_author_cleaned"]
    curated["goodreads_match_score"] = np.where(
        curated["goodreads_cleaned_has_value"],
        1.0,
        np.where(curated["goodreads_cleaned_row_found"], 0.5, np.nan),
    )
    curated["goodreads_query"] = np.where(
        curated["goodreads_cleaned_has_value"],
        "master_book_metadata_cleaned",
        curated.get("goodreads_query", pd.Series(index=curated.index, dtype=object)),
    )
    curated["goodreads_url_raw_best"] = pd.NA
    curated["goodreads_title_raw_best"] = curated["goodreads_title_cleaned"]
    curated["goodreads_author_raw_best"] = curated["goodreads_author_cleaned"]
    curated["goodreads_raw_best_score"] = np.where(
        curated["goodreads_cleaned_has_value"],
        1.0,
        pd.NA,
    )
    curated["goodreads_candidates_json"] = pd.NA
    curated["goodreads_curated_dataset"] = spec.name
    return curated


def categorize_difference(row: pd.Series) -> str:
    old_rating = pd.to_numeric(
        row.get("goodreads_rating_raw_best_old_scraped"), errors="coerce"
    )
    old_count = pd.to_numeric(
        row.get("goodreads_rating_count_raw_best_old_scraped"), errors="coerce"
    )
    new_rating = pd.to_numeric(row.get("goodreads_rating_cleaned"), errors="coerce")
    new_count = pd.to_numeric(
        row.get("goodreads_rating_count_cleaned"), errors="coerce"
    )
    cleaned_found = bool(row.get("goodreads_cleaned_row_found"))
    cleaned_has_value = bool(row.get("goodreads_cleaned_has_value"))
    old_status = normalize_text(row.get("goodreads_status_old_scraped")).lower()
    old_score = pd.to_numeric(
        row.get("goodreads_match_score_old_scraped"), errors="coerce"
    )
    title_similarity = similarity(
        row.get("goodreads_title_old_scraped"),
        row.get("goodreads_title_cleaned"),
    )

    if not cleaned_found:
        return "not_in_clean_file"
    if not cleaned_has_value:
        if pd.notna(old_rating):
            return "clean_file_marks_value_missing"
        return "missing_in_both"
    if pd.isna(old_rating):
        return "pipeline_false_negative"

    rating_diff = abs(float(old_rating) - float(new_rating))
    count_diff = (
        abs(float(old_count) - float(new_count))
        if pd.notna(old_count) and pd.notna(new_count)
        else np.nan
    )

    if rating_diff <= 0.05:
        if pd.notna(count_diff) and count_diff > 0:
            return "count_snapshot_or_edition_drift"
        return "agree"

    if old_status != "matched" or (pd.notna(old_score) and old_score < 0.8):
        if title_similarity < 0.6:
            return "wrong_book_low_confidence_match"
        return "low_confidence_match_difference"

    if title_similarity < 0.6:
        return "wrong_book_high_confidence_match"
    return "same_work_rating_difference"


def build_comparison_frame(curated: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    comparison = curated.copy()
    comparison["dataset"] = spec.name
    comparison["old_goodreads_rating"] = pd.to_numeric(
        comparison.get("goodreads_rating_raw_best_old_scraped"),
        errors="coerce",
    )
    comparison["old_goodreads_rating_count"] = pd.to_numeric(
        comparison.get("goodreads_rating_count_raw_best_old_scraped"),
        errors="coerce",
    )
    comparison["new_goodreads_rating"] = pd.to_numeric(
        comparison["goodreads_rating_cleaned"], errors="coerce"
    )
    comparison["new_goodreads_rating_count"] = pd.to_numeric(
        comparison["goodreads_rating_count_cleaned"], errors="coerce"
    )
    comparison["new_goodreads_review_count"] = pd.to_numeric(
        comparison["goodreads_review_count_cleaned"], errors="coerce"
    )
    comparison["rating_diff_abs"] = (
        comparison["old_goodreads_rating"] - comparison["new_goodreads_rating"]
    ).abs()
    comparison["rating_count_diff_abs"] = (
        comparison["old_goodreads_rating_count"]
        - comparison["new_goodreads_rating_count"]
    ).abs()
    comparison["difference_category"] = comparison.apply(categorize_difference, axis=1)
    comparison["significant_rating_change"] = comparison["rating_diff_abs"].gt(0.05)
    keep_cols = [
        col
        for col in [
            "dataset",
            "title",
            "Bookshelf",
            spec.filename_column,
            "goodreads_cleaned_row_found",
            "goodreads_cleaned_has_value",
            "goodreads_cleaned_status",
            "goodreads_cleaned_match_method",
            "goodreads_title_cleaned",
            "goodreads_author_cleaned",
            "new_goodreads_rating",
            "new_goodreads_review_count",
            "new_goodreads_rating_count",
            "goodreads_status_old_scraped",
            "goodreads_match_method_old_scraped",
            "goodreads_match_score_old_scraped",
            "goodreads_title_old_scraped",
            "goodreads_author_old_scraped",
            "old_goodreads_rating",
            "old_goodreads_rating_count",
            "rating_diff_abs",
            "rating_count_diff_abs",
            "difference_category",
            "significant_rating_change",
        ]
        if col in comparison.columns
    ]
    return comparison[keep_cols].copy()


def summarize_comparison(comparisons: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        dataset = str(comparison["dataset"].iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "metric": "rows_total",
                "value": float(len(comparison)),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "cleaned_rows_found",
                "value": float(comparison["goodreads_cleaned_row_found"].sum()),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "cleaned_rows_with_value",
                "value": float(comparison["goodreads_cleaned_has_value"].sum()),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "old_rows_with_value",
                "value": float(comparison["old_goodreads_rating"].notna().sum()),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "new_rows_with_value",
                "value": float(comparison["new_goodreads_rating"].notna().sum()),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "rating_diff_gt_0p05",
                "value": float(comparison["rating_diff_abs"].gt(0.05).sum()),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "pipeline_false_negative",
                "value": float(
                    comparison["difference_category"]
                    .eq("pipeline_false_negative")
                    .sum()
                ),
            }
        )
        for category, count in (
            comparison["difference_category"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "metric": f"category::{category}",
                    "value": float(count),
                }
            )
    return pd.DataFrame(rows)


def backup_and_replace(target_path: Path, replacement_path: Path, stamp: str) -> Path:
    backup_path = target_path.with_name(
        f"{target_path.stem}_scraped_backup_{stamp}{target_path.suffix}"
    )
    if not backup_path.exists():
        shutil.copy2(target_path, backup_path)
    shutil.copy2(replacement_path, target_path)
    return backup_path


def reconcile_dataset(
    cleaned: pd.DataFrame, spec: DatasetSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = prepare_target_frame(load_csv(spec.input_csv), spec)
    matched = match_curated_rows(target, cleaned, spec)
    merged = pd.concat([target, matched], axis=1)
    curated = apply_curated_values(merged, spec)
    comparison = build_comparison_frame(curated, spec)
    return curated, comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-master-csv",
        type=Path,
        default=CLEANED_MASTER_CSV,
    )
    parser.add_argument(
        "--replace-defaults",
        action="store_true",
        help="Backup and overwrite the default enriched CSVs with curated copies.",
    )
    parser.add_argument(
        "--backup-stamp",
        default=DEFAULT_BACKUP_STAMP,
        help="Suffix used for the scraped-data backup copies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned = prepare_cleaned_frame(args.cleaned_master_csv)

    comparisons: list[pd.DataFrame] = []
    backup_rows: list[dict[str, object]] = []

    for spec in DATASET_SPECS:
        curated, comparison = reconcile_dataset(cleaned, spec)
        curated.to_csv(spec.curated_csv, index=False)
        comparison.to_csv(spec.diff_csv, index=False)
        comparisons.append(comparison)

        if args.replace_defaults:
            backup_path = backup_and_replace(
                target_path=spec.input_csv,
                replacement_path=spec.curated_csv,
                stamp=args.backup_stamp,
            )
            backup_rows.append(
                {
                    "dataset": spec.name,
                    "metric": "backup_path",
                    "value": str(backup_path),
                }
            )

    summary = summarize_comparison(comparisons)
    if backup_rows:
        summary = pd.concat([summary, pd.DataFrame(backup_rows)], ignore_index=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    for comparison in comparisons:
        dataset = str(comparison["dataset"].iloc[0])
        print(f"\n[{dataset}]")
        print(comparison["difference_category"].value_counts(dropna=False).to_string())
        changed = comparison[comparison["significant_rating_change"]].copy()
        if not changed.empty:
            print("\nTop significant changes:")
            print(
                changed[
                    [
                        "title",
                        "difference_category",
                        "old_goodreads_rating",
                        "new_goodreads_rating",
                        "old_goodreads_rating_count",
                        "new_goodreads_rating_count",
                    ]
                ]
                .head(12)
                .to_string(index=False)
            )

    print(f"\nSaved summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
