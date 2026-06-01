"""Project word totals for the full raw-ICS finished-book timeline.

The speed pipeline uses the processed calendar-analysis TSV so overlap handling
matches the rest of the self-tracking work. That processed source currently
starts later than the raw ICS export and misses a small number of finish
markers, so this script builds a separate all-finished word-count projection
from the legacy raw-ICS finished timeline and writes an explicit reconciliation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_books_tracking.book_word_counts import (  # noqa: E402
    DEFAULT_BOOK_ROOTS,
    DEFAULT_WORD_SOURCE_CSV,
    _write_markdown_table,
    build_word_count_outputs,
    write_full_finished_projection_outputs,
)
from ai_books_tracking.book_wpm_calendar_notes import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORDS_PER_PAGE,
    normalize_title,
    titles_match,
)


DEFAULT_ALL_FINISHED_CSV = DEFAULT_OUTPUT_DIR / "book_wpm_finished_timeline.csv"
DEFAULT_CURRENT_FINISHED_CSV = (
    DEFAULT_OUTPUT_DIR / "book_calendar_first_pass_finishes.csv"
)
DEFAULT_CURRENT_TIME_CSV = DEFAULT_OUTPUT_DIR / "book_calendar_first_pass_time.csv"
DEFAULT_LEGACY_WPM_CSV = DEFAULT_OUTPUT_DIR / "book_wpm_by_book.csv"
DEFAULT_PROCESSED_CALENDAR_TSV = REPO_ROOT / "data" / "calendar_analysis.txt"


def load_finished_titles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"finish_id", "title", "finish_date"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "calendar_finish_ref" not in frame:
        frame["calendar_finish_ref"] = frame.get("cal_ref", "")
    return frame


def title_overlap_score(left: object, right: object) -> float:
    left_words = set(normalize_title(str(left)).split())
    right_words = set(normalize_title(str(right)).split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def match_current_finish(
    all_finished_row: pd.Series, current_finishes: pd.DataFrame
) -> pd.Series | None:
    if current_finishes.empty:
        return None
    finish_date = pd.to_datetime(all_finished_row.get("finish_date"), errors="coerce")
    if pd.isna(finish_date):
        candidates = current_finishes.copy()
    else:
        candidates = current_finishes[
            (current_finishes["_finish_date"] - finish_date).abs().dt.days.le(2)
        ].copy()
    if candidates.empty:
        return None

    title = str(all_finished_row.get("title", ""))
    ref = str(all_finished_row.get("calendar_finish_ref", ""))
    for _, candidate in candidates.iterrows():
        candidate_title = str(candidate.get("title", ""))
        candidate_ref = str(candidate.get("calendar_finish_ref", ""))
        if (
            titles_match(title, candidate_title)
            or titles_match(title, candidate_ref)
            or titles_match(ref, candidate_title)
            or titles_match(ref, candidate_ref)
        ):
            return candidate

    candidates["_match_score"] = candidates.apply(
        lambda candidate: max(
            title_overlap_score(title, candidate.get("title", "")),
            title_overlap_score(title, candidate.get("calendar_finish_ref", "")),
            title_overlap_score(ref, candidate.get("title", "")),
            title_overlap_score(ref, candidate.get("calendar_finish_ref", "")),
        ),
        axis=1,
    )
    best = candidates.sort_values("_match_score", ascending=False).iloc[0]
    return best if float(best["_match_score"]) >= 0.65 else None


def processed_calendar_start(processed_calendar_tsv: Path) -> object:
    if not processed_calendar_tsv.exists():
        return pd.NaT
    processed = pd.read_csv(processed_calendar_tsv, sep="\t", usecols=["start_time"])
    starts = pd.to_datetime(processed["start_time"], errors="coerce", utc=True)
    if starts.dropna().empty:
        return pd.NaT
    return starts.min().date()


def build_calendar_coverage_reconciliation(
    *,
    all_finished: pd.DataFrame,
    current_finishes: pd.DataFrame,
    current_time: pd.DataFrame,
    legacy_wpm: pd.DataFrame,
    processed_calendar_tsv: Path,
) -> pd.DataFrame:
    current = current_finishes.copy()
    if "calendar_finish_ref" not in current:
        current["calendar_finish_ref"] = current.get("cal_ref", "")
    current["_finish_date"] = pd.to_datetime(current["finish_date"], errors="coerce")
    processed_start = processed_calendar_start(processed_calendar_tsv)
    time_by_id = (
        current_time.set_index("finish_id", drop=False)
        if not current_time.empty and "finish_id" in current_time
        else pd.DataFrame()
    )
    legacy_by_id = (
        legacy_wpm.set_index("finish_id", drop=False)
        if not legacy_wpm.empty and "finish_id" in legacy_wpm
        else pd.DataFrame()
    )

    rows: list[dict[str, object]] = []
    for _, row in all_finished.iterrows():
        match = match_current_finish(row, current)
        finish_date = pd.to_datetime(row.get("finish_date"), errors="coerce")
        current_finish_id = str(match.get("finish_id", "")) if match is not None else ""
        has_current_time = (
            bool(current_finish_id)
            and not time_by_id.empty
            and current_finish_id in time_by_id.index
        )
        if has_current_time:
            status = "in_current_processed_calendar_time"
        elif match is not None:
            status = "in_current_finish_list_missing_current_time"
        elif (
            pd.notna(finish_date)
            and pd.notna(processed_start)
            and (finish_date.date() < processed_start)
        ):
            status = "before_processed_calendar_analysis_start"
        else:
            status = "raw_ics_finish_not_in_processed_calendar_analysis"

        legacy_row = (
            legacy_by_id.loc[row["finish_id"]]
            if not legacy_by_id.empty and row["finish_id"] in legacy_by_id.index
            else pd.Series(dtype=object)
        )
        current_time_row = (
            time_by_id.loc[current_finish_id]
            if has_current_time
            else pd.Series(dtype=object)
        )
        rows.append(
            {
                "all_finished_finish_id": row["finish_id"],
                "all_finished_title": row["title"],
                "finish_date": row["finish_date"],
                "calendar_finish_ref": row.get("calendar_finish_ref", ""),
                "current_finish_id": current_finish_id,
                "current_title": match.get("title", "") if match is not None else "",
                "calendar_coverage_status": status,
                "processed_calendar_start": processed_start,
                "legacy_total_reading_minutes": legacy_row.get(
                    "total_reading_minutes", np.nan
                ),
                "current_primary_first_pass_minutes": current_time_row.get(
                    "primary_first_pass_minutes", np.nan
                ),
                "current_n_included_events": current_time_row.get(
                    "n_included_events", np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_calendar_coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame(columns=["calendar_coverage_status", "rows"])
    summary = (
        coverage.groupby("calendar_coverage_status", dropna=False)
        .agg(
            rows=("all_finished_title", "size"),
            legacy_total_reading_minutes=("legacy_total_reading_minutes", "sum"),
            current_primary_first_pass_minutes=(
                "current_primary_first_pass_minutes",
                "sum",
            ),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    totals = pd.DataFrame(
        [
            {
                "calendar_coverage_status": "TOTAL",
                "rows": len(coverage),
                "legacy_total_reading_minutes": coverage[
                    "legacy_total_reading_minutes"
                ].sum(),
                "current_primary_first_pass_minutes": coverage[
                    "current_primary_first_pass_minutes"
                ].sum(),
            }
        ]
    )
    return pd.concat([summary, totals], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finished-csv", type=Path, default=DEFAULT_ALL_FINISHED_CSV)
    parser.add_argument(
        "--current-finished-csv", type=Path, default=DEFAULT_CURRENT_FINISHED_CSV
    )
    parser.add_argument(
        "--current-time-csv", type=Path, default=DEFAULT_CURRENT_TIME_CSV
    )
    parser.add_argument("--legacy-wpm-csv", type=Path, default=DEFAULT_LEGACY_WPM_CSV)
    parser.add_argument(
        "--processed-calendar-tsv",
        type=Path,
        default=DEFAULT_PROCESSED_CALENDAR_TSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--book-root", action="append", type=Path, default=[])
    parser.add_argument("--word-source-csv", type=Path, default=DEFAULT_WORD_SOURCE_CSV)
    parser.add_argument("--words-per-page", type=int, default=DEFAULT_WORDS_PER_PAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_finished = load_finished_titles(args.finished_csv)
    work_output_dir = args.output_dir / "all_finished_word_count_inputs"
    counts = build_word_count_outputs(
        titles=all_finished,
        output_dir=work_output_dir,
        book_roots=args.book_root or DEFAULT_BOOK_ROOTS,
        word_source_csv=args.word_source_csv,
        words_per_page=args.words_per_page,
    )
    counts.to_csv(
        args.output_dir / "book_word_count_all_finished_counts.csv", index=False
    )
    write_full_finished_projection_outputs(
        counts,
        args.output_dir,
        output_stem="book_word_count_all_finished_projection",
        words_per_page=args.words_per_page,
    )

    current_finishes = (
        load_finished_titles(args.current_finished_csv)
        if args.current_finished_csv.exists()
        else pd.DataFrame()
    )
    current_time = (
        pd.read_csv(args.current_time_csv)
        if args.current_time_csv.exists()
        else pd.DataFrame()
    )
    legacy_wpm = (
        pd.read_csv(args.legacy_wpm_csv)
        if args.legacy_wpm_csv.exists()
        else pd.DataFrame()
    )
    coverage = build_calendar_coverage_reconciliation(
        all_finished=all_finished,
        current_finishes=current_finishes,
        current_time=current_time,
        legacy_wpm=legacy_wpm,
        processed_calendar_tsv=args.processed_calendar_tsv,
    )
    coverage_summary = build_calendar_coverage_summary(coverage)
    coverage.to_csv(
        args.output_dir / "book_all_finished_calendar_coverage.csv", index=False
    )
    coverage_summary.to_csv(
        args.output_dir / "book_all_finished_calendar_coverage_summary.csv",
        index=False,
    )
    _write_markdown_table(
        coverage, args.output_dir / "book_all_finished_calendar_coverage.md"
    )
    _write_markdown_table(
        coverage_summary,
        args.output_dir / "book_all_finished_calendar_coverage_summary.md",
    )
    print(f"Wrote all-finished projection for {len(all_finished)} finish rows")
    print(coverage_summary.to_string(index=False))


if __name__ == "__main__":
    main()
