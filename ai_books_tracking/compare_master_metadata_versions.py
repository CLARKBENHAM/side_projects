"""Audit the immutable master metadata against the derived final master.

This script has two jobs:
1. Verify that the derived final master preserves the original verified fields
   from `master_book_metadata_cleaned.csv` and only appends new columns.
2. Surface how the golden master differs from downstream enriched datasets that
   were previously populated from scraped Goodreads data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_books_tracking.build_final_master_book_metadata import (
    MASTER_CLEANED_CSV,
    OUTPUT_CSV as FINAL_MASTER_CSV,
)
from ai_books_tracking.reconcile_cleaned_goodreads_metadata import (
    DATASET_SPECS,
    build_comparison_frame,
    prepare_cleaned_frame,
    reconcile_dataset,
    summarize_comparison,
)

BASE_DIR = Path(__file__).parent

SHARED_DIFF_CSV = BASE_DIR / "master_book_metadata_final_shared_differences.csv"
ADDED_COLUMNS_CSV = BASE_DIR / "master_book_metadata_final_added_columns_summary.csv"
UNMATCHED_PERSONAL_CSV = BASE_DIR / "master_book_metadata_final_unmatched_personal.csv"
GOLDEN_MATCH_AUDIT_CSV = BASE_DIR / "master_book_metadata_golden_match_audit.csv"
GOLDEN_MATCH_SUMMARY_CSV = BASE_DIR / "master_book_metadata_golden_match_summary.csv"
REPORT_MD = BASE_DIR / "master_book_metadata_final_report.md"

KEY_COLUMNS = ["title", "source", "filename"]
CANONICAL_RENAMES = {
    "ratings": "goodread ratings",
    "number reviews": "goodreads number reviews",
    "number ratings": "goodreads number ratings",
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = renamed.columns.str.strip()
    rename_map = {
        src: dst
        for src, dst in CANONICAL_RENAMES.items()
        if src in renamed.columns and dst not in renamed.columns
    }
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def compare_shared_columns(original: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    diff_columns = [
        "title",
        "source",
        "filename",
        "field",
        "original_value",
        "final_value",
    ]
    shared_cols = [
        col
        for col in original.columns
        if col in final.columns and col not in KEY_COLUMNS
    ]
    if not shared_cols:
        return pd.DataFrame(columns=diff_columns)

    original_indexed = original.set_index(KEY_COLUMNS)
    final_indexed = final.set_index(KEY_COLUMNS)
    common_index = original_indexed.index.intersection(final_indexed.index)

    diff_rows: list[dict[str, object]] = []
    for key in common_index:
        orig_row = original_indexed.loc[key]
        final_row = final_indexed.loc[key]
        for col in shared_cols:
            orig_value = orig_row[col]
            final_value = final_row[col]
            if pd.isna(orig_value) and pd.isna(final_value):
                continue
            if str(orig_value) == str(final_value):
                continue
            diff_rows.append(
                {
                    "title": key[0],
                    "source": key[1],
                    "filename": key[2],
                    "field": col,
                    "original_value": orig_value,
                    "final_value": final_value,
                }
            )
    return pd.DataFrame(diff_rows, columns=diff_columns)


def summarize_added_columns(
    original: pd.DataFrame, final: pd.DataFrame
) -> pd.DataFrame:
    added_cols = [col for col in final.columns if col not in original.columns]
    rows: list[dict[str, object]] = []
    for col in added_cols:
        series = final[col]
        non_null = int(series.notna().sum())
        rows.append(
            {
                "column": col,
                "non_null_rows": non_null,
                "coverage_pct": non_null / len(final) if len(final) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["coverage_pct", "column"], ascending=[False, True]
    )


def build_report(
    original: pd.DataFrame,
    final: pd.DataFrame,
    shared_diffs: pd.DataFrame,
    added_summary: pd.DataFrame,
    unmatched_personal: pd.DataFrame,
    golden_summary: pd.DataFrame,
    golden_audit: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Master Metadata Final Report")
    lines.append("")
    lines.append("## Final-vs-Original")
    lines.append(f"- Original rows: {len(original)}")
    lines.append(f"- Final rows: {len(final)}")
    lines.append(f"- Original columns: {len(original.columns)}")
    lines.append(f"- Final columns: {len(final.columns)}")
    lines.append(f"- Added columns in final: {len(added_summary)}")
    lines.append(f"- Shared field differences: {len(shared_diffs)}")
    if shared_diffs.empty:
        lines.append("- Shared canonical fields are unchanged.")
    else:
        lines.append("- Shared canonical fields changed; inspect the diff CSV.")
    lines.append(f"- Personal-rating match gaps in final: {len(unmatched_personal)}")

    lines.append("")
    lines.append("## Added Column Coverage")
    if added_summary.empty:
        lines.append("- No added columns.")
    else:
        for _, row in added_summary.head(12).iterrows():
            lines.append(
                f"- `{row['column']}`: {int(row['non_null_rows'])} rows "
                f"({row['coverage_pct']:.1%})"
            )

    lines.append("")
    lines.append("## Golden-to-Downstream Goodreads Audit")
    if golden_summary.empty:
        lines.append("- No downstream comparison rows.")
    else:
        for dataset in golden_summary["dataset"].dropna().unique():
            subset = golden_summary[golden_summary["dataset"] == dataset]

            def metric(name: str) -> int:
                match = subset[subset["metric"] == name]["value"]
                return int(match.iloc[0]) if not match.empty else 0

            lines.append(
                f"- `{dataset}`: "
                f"cleaned rows found {metric('cleaned_rows_found')}, "
                f"cleaned rows with value {metric('cleaned_rows_with_value')}, "
                f"significant rating changes {metric('rating_diff_gt_0p05')}, "
                f"pipeline false negatives {metric('pipeline_false_negative')}"
            )

    significant = golden_audit[golden_audit["significant_rating_change"]].copy()
    if not significant.empty:
        lines.append("")
        lines.append("## Largest Golden-vs-Old Differences")
        top = significant.sort_values(
            ["rating_diff_abs", "dataset"], ascending=[False, True]
        ).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['dataset']}` | {row['title']}: "
                f"{row['difference_category']} "
                f"(old={row['old_goodreads_rating']}, new={row['new_goodreads_rating']})"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    original = canonicalize_columns(load_csv(MASTER_CLEANED_CSV))
    final = canonicalize_columns(load_csv(FINAL_MASTER_CSV))

    shared_diffs = compare_shared_columns(original, final)
    shared_diffs.to_csv(SHARED_DIFF_CSV, index=False)

    added_summary = summarize_added_columns(original, final)
    added_summary.to_csv(ADDED_COLUMNS_CSV, index=False)

    unmatched_mask = (
        final["personal_ratings_match_found"].astype("boolean").fillna(False)
    )
    unmatched_personal = final.loc[
        ~unmatched_mask, KEY_COLUMNS + ["personal_ratings_match_method"]
    ].copy()
    unmatched_personal.to_csv(UNMATCHED_PERSONAL_CSV, index=False)

    cleaned = prepare_cleaned_frame(MASTER_CLEANED_CSV)
    comparisons: list[pd.DataFrame] = []
    for spec in DATASET_SPECS:
        if spec.curated_csv.exists():
            comparison = build_comparison_frame(load_csv(spec.curated_csv), spec)
        else:
            _, comparison = reconcile_dataset(cleaned, spec)
        comparisons.append(comparison)
    golden_audit = pd.concat(comparisons, ignore_index=True)
    golden_audit.to_csv(GOLDEN_MATCH_AUDIT_CSV, index=False)

    golden_summary = summarize_comparison(comparisons)
    golden_summary.to_csv(GOLDEN_MATCH_SUMMARY_CSV, index=False)

    REPORT_MD.write_text(
        build_report(
            original=original,
            final=final,
            shared_diffs=shared_diffs,
            added_summary=added_summary,
            unmatched_personal=unmatched_personal,
            golden_summary=golden_summary,
            golden_audit=golden_audit,
        ),
        encoding="utf-8",
    )

    print(f"Saved shared diffs to {SHARED_DIFF_CSV}")
    print(f"Saved added-column summary to {ADDED_COLUMNS_CSV}")
    print(f"Saved unmatched-personal rows to {UNMATCHED_PERSONAL_CSV}")
    print(f"Saved golden match audit to {GOLDEN_MATCH_AUDIT_CSV}")
    print(f"Saved golden match summary to {GOLDEN_MATCH_SUMMARY_CSV}")
    print(f"Saved report to {REPORT_MD}")


if __name__ == "__main__":
    main()
