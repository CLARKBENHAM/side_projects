"""Join calendar-time and word-count inputs into reading-speed outputs."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_books_tracking.book_wpm_calendar_notes import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    normalize_title,
    titles_match,
)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(_markdown_table(df), encoding="utf-8")


def build_speed_analysis(
    *,
    calendar_time_csv: Path,
    word_counts_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_df = pd.read_csv(calendar_time_csv)
    words_df = pd.read_csv(word_counts_csv)
    word_columns = [
        "finish_id",
        "chosen_word_count",
        "word_count_source",
        "word_count_confidence",
        "local_file_path",
        "local_file_match_score",
        "local_file_word_count",
        "local_raw_word_count",
        "local_excluded_word_count",
        "local_word_count_method",
        "local_file_error",
        "external_word_count",
        "external_word_count_source",
        "metadata_page_count",
        "metadata_word_estimate",
        "online_minus_local_words",
        "online_error_rate_vs_local",
        "online_abs_error_rate_vs_local",
    ]
    merged = time_df.merge(
        words_df[[column for column in word_columns if column in words_df.columns]],
        on="finish_id",
        how="left",
    )
    legacy_details = output_dir / "book_wpm_by_book.csv"
    if legacy_details.exists():
        detail_df = pd.read_csv(legacy_details)
        merged = attach_legacy_detail_columns(merged, detail_df)
    for column in ["category", "highlighted_words", "highlight_count", "my_note_words"]:
        if column not in merged:
            merged[column] = np.nan
    if "primary_first_pass_minutes" not in merged:
        merged["primary_first_pass_minutes"] = merged["first_pass_adjusted_minutes"]
        merged["primary_minutes_rule"] = "legacy_first_pass_adjusted_minutes"
    ensure_duration_aliases(merged)
    merged["wpm_first_pass_primary"] = (
        merged["chosen_word_count"] / merged["primary_first_pass_minutes"]
    )
    merged["wpm_first_pass_wall_clock"] = (
        merged["chosen_word_count"] / merged["first_pass_wall_clock_minutes"]
    )
    merged["wpm_first_pass_full_wall_clock"] = merged["wpm_first_pass_wall_clock"]
    audio_minutes = merged.get("audiobook_overlap_policy_minutes")
    if audio_minutes is None:
        audio_minutes = merged.get("audiobook_wall_clock_minutes", 0)
    merged["audiobook_primary_minutes"] = pd.to_numeric(
        audio_minutes, errors="coerce"
    ).fillna(0)
    merged["non_audio_primary_minutes"] = (
        merged["primary_first_pass_minutes"] - merged["audiobook_primary_minutes"]
    ).clip(lower=0)
    merged["audiobook_share_of_primary_minutes"] = (
        merged["audiobook_primary_minutes"] / merged["primary_first_pass_minutes"]
    ).replace([np.inf, -np.inf], np.nan)
    merged["estimated_audio_words_at_350wpm"] = (
        merged["audiobook_primary_minutes"] * 350
    )
    merged["estimated_visual_words_after_audio_350wpm"] = (
        merged["chosen_word_count"] - merged["estimated_audio_words_at_350wpm"]
    ).clip(lower=0)
    merged["wpm_visual_after_audio_350wpm"] = (
        merged["estimated_visual_words_after_audio_350wpm"]
        / merged["non_audio_primary_minutes"]
    ).where(merged["non_audio_primary_minutes"].gt(0))
    merged["audio_dominant"] = (
        merged["audiobook_share_of_primary_minutes"].ge(0.5)
        | (merged["estimated_audio_words_at_350wpm"] / merged["chosen_word_count"]).ge(
            0.5
        )
    ).fillna(False)
    merged["wpm_reading_only_calendar"] = (
        merged["chosen_word_count"] / merged["reading_calendar_minutes"]
    )
    merged["highlight_words_per_10k_words"] = (
        merged["highlighted_words"] / merged["chosen_word_count"] * 10000
    )
    merged["finish_year"] = pd.to_datetime(
        merged["matched_finish_date"], errors="coerce"
    ).dt.year
    merged["word_count_confidence"] = merged["word_count_confidence"].fillna("low")
    merged["uses_local_file_word_count"] = merged["word_count_source"].eq(
        "local_file_word_count"
    )
    merged["usable_for_speed"] = (
        merged["chosen_word_count"].notna()
        & merged["primary_first_pass_minutes"].gt(0)
        & merged["is_reread"].fillna(False).eq(False)
        & merged["uses_local_file_word_count"]
    )
    merged["usable_for_confident_speed"] = merged["usable_for_speed"] & merged[
        "word_count_confidence"
    ].eq("high")
    merged["usable_for_duration_screened_speed"] = merged["usable_for_speed"] & merged[
        "primary_first_pass_minutes"
    ].gt(90)
    merged["usable_for_visual_reading_speed"] = (
        merged["usable_for_duration_screened_speed"]
        & ~merged["audio_dominant"]
        & merged["wpm_visual_after_audio_350wpm"].notna()
    )
    merged["duration_screen_exclusion_reason"] = merged.apply(
        duration_screen_exclusion_reason, axis=1
    )
    add_high_wpm_investigation(merged, output_dir)
    merged.to_csv(output_dir / "book_speed_analysis.csv", index=False)
    _write_markdown_table(
        merged.sort_values("matched_finish_date", ascending=False).head(100),
        output_dir / "book_speed_analysis.md",
    )
    write_aggregate_speed_summary(merged, output_dir)
    write_noise_sensitivity_summary(merged, output_dir)
    write_summaries_and_plots(merged, output_dir)
    write_report(merged, output_dir)
    return merged


def attach_legacy_detail_columns(
    merged: pd.DataFrame, detail_df: pd.DataFrame
) -> pd.DataFrame:
    detail_columns = [
        column
        for column in [
            "category",
            "highlighted_words",
            "highlight_count",
            "my_note_words",
        ]
        if column in detail_df.columns
    ]
    if not detail_columns or detail_df.empty:
        return merged
    details = detail_df.copy()
    details["_finish_date"] = pd.to_datetime(
        details.get("finish_date"), errors="coerce"
    ).dt.date
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        title = str(row.get("title", ""))
        finish_timestamp = pd.to_datetime(
            row.get("matched_finish_date"), errors="coerce"
        )
        finish_date = None if pd.isna(finish_timestamp) else finish_timestamp.date()
        if finish_date is None:
            candidates = details.copy()
        else:
            candidates = details[
                details["_finish_date"].map(
                    lambda value: pd.notna(value)
                    and abs((finish_date - value).days) <= 2
                )
            ].copy()
        if candidates.empty:
            candidates = details
        candidates["_title_match"] = candidates["title"].map(
            lambda value: titles_match(title, value)
        )
        exact = candidates[candidates["_title_match"]]
        if exact.empty:
            candidates["_norm_overlap"] = candidates["title"].map(
                lambda value: title_overlap_score(title, str(value))
            )
            exact = candidates[candidates["_norm_overlap"].ge(0.65)]
        if exact.empty:
            rows.append({column: np.nan for column in detail_columns})
            continue
        if finish_date is None:
            match = exact.iloc[0]
        else:
            match = exact.sort_values(
                by="_finish_date",
                key=lambda values: values.map(
                    lambda value: (
                        abs((finish_date - value).days) if pd.notna(value) else 9999
                    )
                ),
            ).iloc[0]
        rows.append({column: match.get(column, np.nan) for column in detail_columns})
    detail_matches = pd.DataFrame(rows, index=merged.index)
    return pd.concat([merged, detail_matches], axis=1)


def title_overlap_score(left: str, right: str) -> float:
    left_words = set(normalize_title(left).split())
    right_words = set(normalize_title(right).split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def ensure_duration_aliases(frame: pd.DataFrame) -> None:
    aliases = {
        "first_pass_calendar_minutes": "first_pass_adjusted_minutes",
        "first_pass_wall_clock_minutes": "first_pass_raw_minutes",
        "reading_calendar_minutes": "reading_adjusted_minutes",
        "reading_wall_clock_minutes": "reading_raw_minutes",
        "audiobook_calendar_minutes": "audiobook_adjusted_minutes",
        "finished_calendar_minutes": "finished_adjusted_minutes",
    }
    for canonical, fallback in aliases.items():
        if canonical not in frame:
            frame[canonical] = frame[fallback] if fallback in frame else np.nan
    if "calendar_overlap_delta_minutes" not in frame:
        frame["calendar_overlap_delta_minutes"] = (
            frame["first_pass_wall_clock_minutes"]
            - frame["first_pass_calendar_minutes"]
        )


def duration_screen_exclusion_reason(row: pd.Series) -> str:
    if not bool(row.get("usable_for_speed", False)):
        return ""
    if float(row.get("primary_first_pass_minutes", 0) or 0) <= 90:
        return "calendar_time_lte_90_minutes"
    return ""


def likely_high_wpm_cause(row: pd.Series) -> str:
    minutes = float(row.get("primary_first_pass_minutes", 0) or 0)
    event_count = int(row.get("n_included_events", 0) or 0)
    reading_minutes = float(row.get("reading_calendar_minutes", 0) or 0)
    audiobook_minutes = float(
        row.get(
            "audiobook_primary_minutes",
            row.get("audiobook_overlap_policy_minutes", 0),
        )
        or 0
    )
    finished_minutes = float(row.get("finished_calendar_minutes", 0) or 0)
    words = float(row.get("chosen_word_count", 0) or 0)
    wpm = float(row.get("wpm_first_pass_primary", 0) or 0)
    wall_wpm = float(row.get("wpm_first_pass_wall_clock", 0) or 0)
    source = str(row.get("word_count_source", "") or "")
    confidence = str(row.get("word_count_confidence", "") or "")
    local_error = str(row.get("local_file_error", "") or "")
    overlap_delta = float(row.get("calendar_overlap_delta_minutes", 0) or 0)
    if wpm > 900 and wall_wpm <= 900 and overlap_delta > 0:
        return "calendar_analysis_overlap_adjustment_drives_high_wpm"
    if minutes <= 90:
        return "calendar_time_lte_90_minutes"
    if "metadata_pages_x_words_per_page" in source:
        return "word_count_uses_page_estimate_needs_local_file_match"
    if bool(row.get("audio_dominant", False)):
        return "audio_dominant_not_visual_reading_speed"
    if audiobook_minutes > 0 and wpm > 350:
        return "mixed_audio_and_reading_needs_audio_adjusted_wpm"
    if event_count <= 2 and finished_minutes > reading_minutes:
        return "calendar_has_only_finish_or_near-finish_events"
    if reading_minutes == 0 and finished_minutes > 0:
        return "no_matched_reading_sessions_only_finished_duration"
    if words > 250_000 and minutes < 600:
        return "long_book_has_too_little_matched_calendar_time"
    if confidence != "high":
        return "word_count_not_high_confidence"
    if local_error and local_error != "nan":
        return "local_file_extraction_failed_fallback_used"
    if minutes > 0 and overlap_delta / minutes > 0.5:
        return "large_calendar_overlap_adjustment"
    if event_count <= 2:
        return "very_few_calendar_events_matched"
    return "needs_manual_calendar_or_word_count_review"


def add_high_wpm_investigation(merged: pd.DataFrame, output_dir: Path) -> None:
    high = merged[
        merged["usable_for_speed"] & merged["wpm_first_pass_primary"].gt(900)
    ].copy()
    if high.empty:
        pd.DataFrame().to_csv(output_dir / "book_speed_high_wpm_investigation.csv")
        return
    high["likely_cause"] = high.apply(likely_high_wpm_cause, axis=1)
    columns = [
        "finish_id",
        "title",
        "matched_finish_date",
        "primary_first_pass_minutes",
        "first_pass_wall_clock_minutes",
        "calendar_overlap_delta_minutes",
        "reading_calendar_minutes",
        "audiobook_calendar_minutes",
        "finished_calendar_minutes",
        "n_included_events",
        "chosen_word_count",
        "word_count_source",
        "word_count_confidence",
        "local_file_path",
        "local_file_match_score",
        "local_file_word_count",
        "local_file_error",
        "metadata_page_count",
        "metadata_word_estimate",
        "wpm_first_pass_primary",
        "wpm_first_pass_wall_clock",
        "duration_screen_exclusion_reason",
        "likely_cause",
    ]
    existing = [column for column in columns if column in high.columns]
    high = high.sort_values("wpm_first_pass_primary", ascending=False)
    high[existing].to_csv(
        output_dir / "book_speed_high_wpm_investigation.csv", index=False
    )
    _write_markdown_table(
        high[existing], output_dir / "book_speed_high_wpm_investigation.md"
    )
    events_path = output_dir / "book_calendar_first_pass_events.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
        evidence = events[events["finish_id"].isin(high["finish_id"])].copy()
        evidence.to_csv(
            output_dir / "book_speed_high_wpm_event_evidence.csv", index=False
        )


def write_summaries_and_plots(merged: pd.DataFrame, output_dir: Path) -> None:
    screened_cols = [
        "title",
        "matched_finish_date",
        "primary_first_pass_minutes",
        "first_pass_wall_clock_minutes",
        "chosen_word_count",
        "word_count_source",
        "word_count_confidence",
        "wpm_first_pass_primary",
        "wpm_first_pass_full_wall_clock",
        "audio_dominant",
        "audiobook_primary_minutes",
        "wpm_visual_after_audio_350wpm",
        "category",
        "highlighted_words",
        "highlight_words_per_10k_words",
        "duration_screen_exclusion_reason",
    ]
    usable_all = merged[merged["usable_for_speed"]].copy()
    excluded = usable_all[~usable_all["usable_for_duration_screened_speed"]].copy()
    excluded.to_csv(
        output_dir / "book_speed_short_duration_exclusions.csv", index=False
    )
    _write_markdown_table(
        excluded.sort_values("wpm_first_pass_primary", ascending=False)[screened_cols],
        output_dir / "book_speed_short_duration_exclusions.md",
    )
    usable = merged[merged["usable_for_duration_screened_speed"]].copy()
    usable.to_csv(output_dir / "book_speed_duration_screened_subset.csv", index=False)
    _write_markdown_table(
        usable.sort_values("matched_finish_date", ascending=False)[screened_cols],
        output_dir / "book_speed_duration_screened_subset.md",
    )
    fast = usable[usable["wpm_first_pass_primary"].gt(600)].copy()
    if not fast.empty:
        fast["likely_cause"] = fast.apply(likely_high_wpm_cause, axis=1)
    fast_columns = [
        "finish_id",
        *screened_cols,
        "n_included_events",
        "reading_wall_clock_minutes",
        "audiobook_wall_clock_minutes",
        "finished_wall_clock_minutes",
        "local_file_path",
        "local_file_match_score",
        "likely_cause",
    ]
    fast_existing = [column for column in fast_columns if column in fast.columns]
    fast = fast.sort_values("wpm_first_pass_primary", ascending=False)
    fast[fast_existing].to_csv(
        output_dir / "book_speed_gt600_wpm_investigation.csv", index=False
    )
    _write_markdown_table(
        fast[fast_existing], output_dir / "book_speed_gt600_wpm_investigation.md"
    )
    events_path = output_dir / "book_calendar_first_pass_events.csv"
    if events_path.exists() and not fast.empty:
        events = pd.read_csv(events_path)
        events[events["finish_id"].isin(fast["finish_id"])].to_csv(
            output_dir / "book_speed_gt600_wpm_event_evidence.csv", index=False
        )
    if usable.empty:
        return
    visual_usable = merged[merged["usable_for_visual_reading_speed"]].copy()
    visual_usable.to_csv(
        output_dir / "book_speed_visual_reading_subset.csv", index=False
    )
    _write_markdown_table(
        visual_usable.sort_values("matched_finish_date", ascending=False)[
            [column for column in screened_cols if column in visual_usable.columns]
        ],
        output_dir / "book_speed_visual_reading_subset.md",
    )
    audio_dominant = usable[usable["audio_dominant"]].copy()
    audio_dominant.to_csv(
        output_dir / "book_speed_audio_dominant_exclusions.csv", index=False
    )
    _write_markdown_table(
        audio_dominant.sort_values(
            "audiobook_share_of_primary_minutes", ascending=False
        )[[column for column in screened_cols if column in audio_dominant.columns]],
        output_dir / "book_speed_audio_dominant_exclusions.md",
    )
    by_year = (
        usable.groupby("finish_year")
        .agg(
            n_books=("title", "count"),
            median_wpm=("wpm_first_pass_primary", "median"),
            mean_wpm=("wpm_first_pass_primary", "mean"),
        )
        .reset_index()
    )
    by_year.to_csv(output_dir / "book_speed_by_year.csv", index=False)
    _write_markdown_table(by_year, output_dir / "book_speed_by_year.md")
    category_frame = usable.copy()
    category_frame["category_plot"] = category_frame["category"].replace("", np.nan)
    category_frame["category_plot"] = category_frame["category_plot"].fillna("Unknown")
    by_category = (
        category_frame.groupby("category_plot")
        .agg(
            n_books=("title", "count"),
            median_wpm=("wpm_first_pass_primary", "median"),
            mean_wpm=("wpm_first_pass_primary", "mean"),
        )
        .reset_index()
        .sort_values(["n_books", "median_wpm"], ascending=[False, False])
    )
    by_category.to_csv(output_dir / "book_speed_by_category.csv", index=False)
    _write_markdown_table(by_category, output_dir / "book_speed_by_category.md")
    if len(by_year) >= 2:
        slope, intercept = np.polyfit(
            by_year["finish_year"].astype(float), by_year["median_wpm"], 1
        )
    else:
        slope, intercept = np.nan, np.nan
    pd.DataFrame(
        [
            {
                "n_books": len(usable),
                "n_usable_before_screening": len(usable_all),
                "n_confident_books": int(merged["usable_for_confident_speed"].sum()),
                "n_years": len(by_year),
                "median_wpm": usable["wpm_first_pass_primary"].median(),
                "slope_median_wpm_per_year": slope,
                "intercept": intercept,
            }
        ]
    ).to_csv(output_dir / "book_speed_summary.csv", index=False)
    percentiles = pd.DataFrame(
        {
            "percentile": [5, 10, 25, 50, 75, 90, 95],
            "wpm_first_pass_primary": np.percentile(
                usable["wpm_first_pass_primary"].dropna(), [5, 10, 25, 50, 75, 90, 95]
            ),
        }
    )
    percentiles.to_csv(output_dir / "book_speed_percentiles.csv", index=False)
    _write_markdown_table(percentiles, output_dir / "book_speed_percentiles.md")
    plot_violin(
        usable,
        "finish_year",
        "wpm_first_pass_primary",
        output_dir / "book_speed_violin_by_year.png",
        "First-Pass WPM by Year",
    )
    plot_histogram(
        usable,
        "wpm_first_pass_primary",
        output_dir / "book_speed_histogram.png",
        "First-Pass WPM Histogram",
        force_zero_floor=False,
    )
    plot_histogram(
        usable,
        "wpm_first_pass_full_wall_clock",
        output_dir / "book_speed_histogram_full_wall_clock.png",
        "First-Pass WPM Histogram Using Full Wall-Clock Times",
    )
    plot_histogram(
        visual_usable,
        "wpm_visual_after_audio_350wpm",
        output_dir / "book_speed_histogram_visual_after_audio_350wpm.png",
        "Visual Reading WPM After 350 WPM Audiobook Adjustment",
    )
    write_final_best_subset_outputs(visual_usable, output_dir, screened_cols)
    plot_time_vs_words(
        usable,
        output_dir / "book_time_vs_words.png",
        "Reading Time vs Local Extracted Words",
    )
    plot_time_vs_words(
        usable,
        output_dir / "book_time_vs_words_full_wall_clock.png",
        "Full Wall-Clock Reading Time vs Local Extracted Words",
        time_col="first_pass_wall_clock_minutes",
        wpm_col="wpm_first_pass_full_wall_clock",
        x_label="Full wall-clock first-pass time (hours)",
    )
    page_frame = usable[usable["metadata_page_count"].notna()].copy()
    plot_time_vs_pages(
        page_frame,
        output_dir / "book_time_vs_pages.png",
        "Reading Time vs Estimated Pages",
    )
    plot_time_vs_pages(
        page_frame,
        output_dir / "book_time_vs_pages_full_wall_clock.png",
        "Full Wall-Clock Reading Time vs Estimated Pages",
        time_col="first_pass_wall_clock_minutes",
        wpm_col="wpm_first_pass_full_wall_clock",
        x_label="Full wall-clock first-pass time (hours)",
    )
    category_counts = category_frame["category_plot"].value_counts()
    category_plot_frame = category_frame[
        category_frame["category_plot"].isin(
            category_counts[category_counts >= 2].index
        )
    ]
    plot_violin(
        category_plot_frame,
        "category_plot",
        "wpm_first_pass_primary",
        output_dir / "book_speed_violin_by_category.png",
        "First-Pass WPM by Category",
    )
    write_rolling_category_percentiles(usable, output_dir)
    write_confidence_stratified_outputs(usable, output_dir)
    write_highlight_effects_and_plots(usable, output_dir)
    write_cleaned_and_full_highlight_plots(merged, output_dir)


def write_rolling_category_percentiles(
    usable: pd.DataFrame,
    output_dir: Path,
    *,
    window_months: int = 6,
    min_books_per_window: int = 2,
    min_books_per_category: int = 3,
) -> pd.DataFrame:
    frame = usable.copy()
    frame["matched_finish_date"] = pd.to_datetime(
        frame["matched_finish_date"], errors="coerce"
    )
    frame["category_plot"] = frame["category"].replace("", np.nan).fillna("Unknown")
    frame["wpm_first_pass_primary"] = pd.to_numeric(
        frame["wpm_first_pass_primary"], errors="coerce"
    )
    frame = frame.dropna(subset=["matched_finish_date", "wpm_first_pass_primary"])
    category_counts = frame["category_plot"].value_counts()
    categories = sorted(
        category_counts[category_counts.ge(min_books_per_category)].index.tolist()
    )
    output_columns = [
        "window_end",
        "window_start_exclusive",
        "category",
        "rolling_n",
        "wpm_p20",
        "wpm_p50",
        "wpm_p80",
    ]
    if frame.empty or not categories:
        empty = pd.DataFrame(columns=output_columns)
        empty.to_csv(
            output_dir / "book_speed_rolling_6mo_category_percentiles.csv",
            index=False,
        )
        _write_markdown_table(
            empty, output_dir / "book_speed_rolling_6mo_category_percentiles.md"
        )
        return empty

    start = frame["matched_finish_date"].min().to_period("M").to_timestamp("M")
    end = frame["matched_finish_date"].max().to_period("M").to_timestamp("M")
    month_ends = pd.date_range(start=start, end=end, freq="ME")
    rows: list[dict[str, object]] = []
    for window_end in month_ends:
        window_start = window_end - pd.DateOffset(months=window_months)
        for category in categories:
            subset = frame[
                frame["category_plot"].eq(category)
                & frame["matched_finish_date"].gt(window_start)
                & frame["matched_finish_date"].le(window_end)
            ].copy()
            if len(subset) < min_books_per_window:
                continue
            values = subset["wpm_first_pass_primary"].to_numpy()
            p20, p50, p80 = np.percentile(values, [20, 50, 80])
            rows.append(
                {
                    "window_end": window_end.date().isoformat(),
                    "window_start_exclusive": window_start.date().isoformat(),
                    "category": category,
                    "rolling_n": len(subset),
                    "wpm_p20": p20,
                    "wpm_p50": p50,
                    "wpm_p80": p80,
                }
            )
    result = pd.DataFrame(rows, columns=output_columns)
    result.to_csv(
        output_dir / "book_speed_rolling_6mo_category_percentiles.csv",
        index=False,
    )
    _write_markdown_table(
        result, output_dir / "book_speed_rolling_6mo_category_percentiles.md"
    )
    plot_rolling_category_percentiles(
        result,
        output_dir / "book_speed_rolling_6mo_category_percentiles.png",
    )
    return result


def plot_rolling_category_percentiles(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plot_frame["window_end"] = pd.to_datetime(plot_frame["window_end"])
    categories = sorted(plot_frame["category"].unique())
    colors = {
        category: plt.get_cmap("tab10")(index % 10)
        for index, category in enumerate(categories)
    }
    percentile_specs = [
        ("wpm_p20", "20th percentile WPM"),
        ("wpm_p50", "50th percentile WPM"),
        ("wpm_p80", "80th percentile WPM"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        "Six-Month Rolling WPM Percentiles by Category",
        fontsize=15,
    )
    for ax, (column, label) in zip(axes, percentile_specs):
        for category in categories:
            subset = plot_frame[plot_frame["category"].eq(category)].sort_values(
                "window_end"
            )
            if subset.empty:
                continue
            ax.plot(
                subset["window_end"],
                subset[column],
                marker="o",
                linewidth=1.6,
                markersize=3.5,
                color=colors[category],
                label=category,
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Window ending month")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.87, 0.5))
    fig.tight_layout(rect=(0, 0, 0.86, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_final_best_subset_outputs(
    visual_usable: pd.DataFrame, output_dir: Path, screened_cols: list[str]
) -> None:
    final_best = visual_usable[
        visual_usable["wpm_visual_after_audio_350wpm"].le(600)
    ].copy()
    final_best.to_csv(output_dir / "book_speed_final_best_subset.csv", index=False)
    if final_best.empty:
        return
    _write_markdown_table(
        final_best.sort_values("matched_finish_date", ascending=False)[
            [column for column in screened_cols if column in final_best.columns]
        ],
        output_dir / "book_speed_final_best_subset.md",
    )
    by_year = (
        final_best.groupby("finish_year")
        .agg(
            n_books=("title", "count"),
            median_visual_wpm=("wpm_visual_after_audio_350wpm", "median"),
            mean_visual_wpm=("wpm_visual_after_audio_350wpm", "mean"),
        )
        .reset_index()
    )
    by_year.to_csv(output_dir / "book_speed_final_best_by_year.csv", index=False)
    _write_markdown_table(by_year, output_dir / "book_speed_final_best_by_year.md")

    category_frame = final_best.copy()
    category_frame["category_plot"] = category_frame["category"].replace("", np.nan)
    category_frame["category_plot"] = category_frame["category_plot"].fillna("Unknown")
    by_category = (
        category_frame.groupby("category_plot")
        .agg(
            n_books=("title", "count"),
            median_visual_wpm=("wpm_visual_after_audio_350wpm", "median"),
            mean_visual_wpm=("wpm_visual_after_audio_350wpm", "mean"),
        )
        .reset_index()
        .sort_values(["n_books", "median_visual_wpm"], ascending=[False, False])
    )
    by_category.to_csv(
        output_dir / "book_speed_final_best_by_category.csv", index=False
    )
    _write_markdown_table(
        by_category, output_dir / "book_speed_final_best_by_category.md"
    )

    plot_histogram(
        final_best,
        "wpm_visual_after_audio_350wpm",
        output_dir / "book_speed_final_best_histogram.png",
        "Final Best Subset: Visual Reading WPM",
    )
    plot_violin(
        final_best,
        "finish_year",
        "wpm_visual_after_audio_350wpm",
        output_dir / "book_speed_final_best_violin_by_year.png",
        "Final Best Subset: Visual Reading WPM by Year",
    )
    category_counts = category_frame["category_plot"].value_counts()
    category_plot_frame = category_frame[
        category_frame["category_plot"].isin(
            category_counts[category_counts >= 2].index
        )
    ]
    plot_violin(
        category_plot_frame,
        "category_plot",
        "wpm_visual_after_audio_350wpm",
        output_dir / "book_speed_final_best_violin_by_category.png",
        "Final Best Subset: Visual Reading WPM by Category",
    )


def write_aggregate_speed_summary(merged: pd.DataFrame, output_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    local_word_counts = merged["uses_local_file_word_count"].fillna(False)
    cohorts = [
        (
            "all_finished_read_instances_with_local_file_word_count",
            merged["chosen_word_count"].notna()
            & merged["primary_first_pass_minutes"].gt(0),
        ),
        (
            "first_reads_with_local_file_word_count",
            merged["chosen_word_count"].notna()
            & merged["primary_first_pass_minutes"].gt(0)
            & merged["is_reread"].fillna(False).eq(False),
        ),
        (
            "first_reads_gt90m_with_local_file_word_count",
            merged["chosen_word_count"].notna()
            & merged["primary_first_pass_minutes"].gt(90)
            & merged["is_reread"].fillna(False).eq(False),
        ),
        (
            "visual_reading_first_reads_gt90m_excluding_audio_dominant",
            merged["usable_for_visual_reading_speed"],
        ),
    ]
    for cohort, mask in cohorts:
        subset = merged[mask & local_word_counts].copy()
        total_words = float(subset["chosen_word_count"].sum())
        total_minutes = float(subset["primary_first_pass_minutes"].sum())
        total_full_wall_clock_minutes = float(
            subset["first_pass_wall_clock_minutes"].sum()
        )
        rows.append(
            {
                "cohort": cohort,
                "n_finished_read_instances": len(subset),
                "total_estimated_words": total_words,
                "total_primary_minutes": total_minutes,
                "total_primary_hours": total_minutes / 60 if total_minutes else np.nan,
                "total_full_wall_clock_minutes": total_full_wall_clock_minutes,
                "total_full_wall_clock_hours": (
                    total_full_wall_clock_minutes / 60
                    if total_full_wall_clock_minutes
                    else np.nan
                ),
                "aggregate_wpm": (
                    total_words / total_minutes if total_minutes else np.nan
                ),
                "aggregate_visual_after_audio_350wpm": (
                    subset["estimated_visual_words_after_audio_350wpm"].sum()
                    / subset["non_audio_primary_minutes"].sum()
                    if not subset.empty
                    and subset["non_audio_primary_minutes"].sum() > 0
                    else np.nan
                ),
                "aggregate_full_wall_clock_wpm": (
                    total_words / total_full_wall_clock_minutes
                    if total_full_wall_clock_minutes
                    else np.nan
                ),
                "mean_of_book_wpms": subset["wpm_first_pass_primary"].mean(),
                "mean_of_book_full_wall_clock_wpms": subset[
                    "wpm_first_pass_full_wall_clock"
                ].mean(),
                "median_book_wpm": subset["wpm_first_pass_primary"].median(),
                "median_book_visual_after_audio_350wpm": subset[
                    "wpm_visual_after_audio_350wpm"
                ].median(),
                "median_book_full_wall_clock_wpm": subset[
                    "wpm_first_pass_full_wall_clock"
                ].median(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "book_speed_aggregate_wpm.csv", index=False)
    _write_markdown_table(summary, output_dir / "book_speed_aggregate_wpm.md")


def write_noise_sensitivity_summary(merged: pd.DataFrame, output_dir: Path) -> None:
    base = merged["usable_for_duration_screened_speed"].fillna(False)
    cohorts = [
        ("base_first_reads_gt90m", base),
        ("base_excluding_gt600_wpm", base & merged["wpm_first_pass_primary"].le(600)),
        ("base_excluding_gt900_wpm", base & merged["wpm_first_pass_primary"].le(900)),
        (
            "base_excluding_audio_dominant",
            base & ~merged["audio_dominant"].fillna(False),
        ),
        (
            "visual_after_audio_excluding_audio_dominant",
            merged["usable_for_visual_reading_speed"].fillna(False),
        ),
        (
            "visual_after_audio_excluding_audio_dominant_gt600",
            merged["usable_for_visual_reading_speed"].fillna(False)
            & merged["wpm_visual_after_audio_350wpm"].le(600),
        ),
        (
            "high_confidence_first_reads_gt90m",
            base & merged["word_count_confidence"].eq("high"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, mask in cohorts:
        subset = merged[mask].copy()
        total_primary_minutes = float(subset["primary_first_pass_minutes"].sum())
        total_full_minutes = float(subset["first_pass_wall_clock_minutes"].sum())
        total_words = float(subset["chosen_word_count"].sum())
        visual_minutes = float(subset["non_audio_primary_minutes"].sum())
        visual_words = float(subset["estimated_visual_words_after_audio_350wpm"].sum())
        rows.append(
            {
                "cohort": name,
                "n_books": len(subset),
                "total_words": total_words,
                "total_primary_minutes": total_primary_minutes,
                "aggregate_primary_wpm": (
                    total_words / total_primary_minutes
                    if total_primary_minutes
                    else np.nan
                ),
                "median_primary_wpm": subset["wpm_first_pass_primary"].median(),
                "mean_primary_wpm": subset["wpm_first_pass_primary"].mean(),
                "total_full_wall_clock_minutes": total_full_minutes,
                "aggregate_full_wall_clock_wpm": (
                    total_words / total_full_minutes if total_full_minutes else np.nan
                ),
                "audio_dominant_books": int(subset["audio_dominant"].sum()),
                "gt600_books": int(subset["wpm_first_pass_primary"].gt(600).sum()),
                "gt900_books": int(subset["wpm_first_pass_primary"].gt(900).sum()),
                "aggregate_visual_after_audio_350wpm": (
                    visual_words / visual_minutes if visual_minutes else np.nan
                ),
                "median_visual_after_audio_350wpm": subset[
                    "wpm_visual_after_audio_350wpm"
                ].median(),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        base_value = result.loc[
            result["cohort"].eq("base_first_reads_gt90m"), "aggregate_primary_wpm"
        ].iloc[0]
        result["delta_vs_base_aggregate_primary_wpm"] = (
            result["aggregate_primary_wpm"] - base_value
        )
        result["pct_delta_vs_base_aggregate_primary_wpm"] = (
            result["delta_vs_base_aggregate_primary_wpm"] / base_value * 100
        )
    result.to_csv(output_dir / "book_speed_noise_sensitivity.csv", index=False)
    _write_markdown_table(result, output_dir / "book_speed_noise_sensitivity.md")


def write_confidence_stratified_outputs(usable: pd.DataFrame, output_dir: Path) -> None:
    strata = [
        ("high_confidence", usable[usable["word_count_confidence"].eq("high")]),
        (
            "medium_confidence",
            usable[usable["word_count_confidence"].eq("medium")],
        ),
        (
            "local_file_high_confidence",
            usable[
                usable["word_count_confidence"].eq("high")
                & usable["word_count_source"].eq("local_file_word_count")
            ],
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, frame in strata:
        rows.append(
            {
                "confidence_subset": label,
                "n_books": len(frame),
                "median_wpm": (
                    frame["wpm_first_pass_primary"].median()
                    if not frame.empty
                    else np.nan
                ),
                "mean_wpm": (
                    frame["wpm_first_pass_primary"].mean()
                    if not frame.empty
                    else np.nan
                ),
                "aggregate_wpm": (
                    frame["chosen_word_count"].sum()
                    / frame["primary_first_pass_minutes"].sum()
                    if not frame.empty and frame["primary_first_pass_minutes"].sum() > 0
                    else np.nan
                ),
                "aggregate_full_wall_clock_wpm": (
                    frame["chosen_word_count"].sum()
                    / frame["first_pass_wall_clock_minutes"].sum()
                    if not frame.empty
                    and frame["first_pass_wall_clock_minutes"].sum() > 0
                    else np.nan
                ),
            }
        )
        if frame.empty:
            continue
        plot_histogram(
            frame,
            "wpm_first_pass_primary",
            output_dir / f"book_speed_histogram_{label}.png",
            f"First-Pass WPM Histogram: {label.replace('_', ' ').title()}",
        )
        plot_violin(
            frame,
            "finish_year",
            "wpm_first_pass_primary",
            output_dir / f"book_speed_violin_by_year_{label}.png",
            f"First-Pass WPM by Year: {label.replace('_', ' ').title()}",
        )
        category_frame = frame.copy()
        category_frame["category_plot"] = category_frame["category"].replace("", np.nan)
        category_frame["category_plot"] = category_frame["category_plot"].fillna(
            "Unknown"
        )
        plot_violin(
            category_frame,
            "category_plot",
            "wpm_first_pass_primary",
            output_dir / f"book_speed_violin_by_category_{label}.png",
            f"First-Pass WPM by Category: {label.replace('_', ' ').title()}",
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "book_speed_confidence_comparison.csv", index=False)
    _write_markdown_table(summary, output_dir / "book_speed_confidence_comparison.md")


def plot_violin(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    output_path: Path,
    title: str,
) -> None:
    groups = [
        (str(group), values[value_col].dropna().to_numpy())
        for group, values in frame.groupby(group_col)
        if len(values[value_col].dropna()) >= 1
    ]
    if not groups:
        return
    groups.sort(key=lambda item: item[0])
    labels = [f"{item[0]}\nN={len(item[1])}" for item in groups]
    values = [item[1] for item in groups]
    positions = np.arange(1, len(values) + 1)
    plt.figure(figsize=(max(10, len(values) * 0.8), 6))
    plt.violinplot(values, positions=positions, showmedians=True)
    for pos, vals in zip(positions, values):
        plt.scatter(np.full(len(vals), pos), vals, color="black", s=16, alpha=0.55)
    plt.xticks(positions, labels, rotation=35, ha="right")
    plt.ylabel("WPM")
    plt.title(f"{title} (N={sum(len(vals) for vals in values)})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_histogram(
    frame: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    *,
    force_zero_floor: bool = True,
) -> None:
    values = frame[value_col].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    lower = 0 if force_zero_floor else max(0, float(values.min()) - 25)
    upper = float(values.max()) + 25
    if lower >= upper:
        lower = max(0, float(values.min()) - 1)
        upper = float(values.max()) + 1
    bin_start = math.floor(lower / 50) * 50
    bin_end = math.ceil(upper / 50) * 50
    bins = np.arange(bin_start, bin_end + 50, 50)
    plt.figure(figsize=(10, 6))
    plt.hist(values.clip(lower=0), bins=bins, color="#3b82f6")
    plt.xlabel("WPM")
    plt.ylabel("Books")
    plt.title(f"{title} (N={len(values)})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_time_vs_words(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    time_col: str = "primary_first_pass_minutes",
    wpm_col: str = "wpm_first_pass_primary",
    x_label: str = "First-pass calendar time (hours)",
) -> None:
    plot_frame = frame[frame[time_col].gt(0) & frame["chosen_word_count"].gt(0)].copy()
    if plot_frame.empty:
        return
    plt.figure(figsize=(10, 7))
    plt.scatter(
        plot_frame[time_col] / 60,
        plot_frame["chosen_word_count"] / 1000,
        c=plot_frame[wpm_col],
        cmap="viridis",
        s=36,
        alpha=0.85,
    )
    for _, row in plot_frame.iterrows():
        wpm = row[wpm_col]
        if pd.notna(wpm) and (wpm >= plot_frame[wpm_col].quantile(0.9)):
            plt.annotate(
                str(row["title"])[:28],
                (
                    row[time_col] / 60,
                    row["chosen_word_count"] / 1000,
                ),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    plt.colorbar(label="WPM")
    plt.xlabel(x_label)
    plt.ylabel("Local extracted words (thousands)")
    plt.title(f"{title} (N={len(plot_frame)})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_time_vs_pages(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    time_col: str = "primary_first_pass_minutes",
    wpm_col: str = "wpm_first_pass_primary",
    x_label: str = "First-pass calendar time (hours)",
) -> None:
    plot_frame = frame[
        frame[time_col].gt(0) & frame["metadata_page_count"].gt(0)
    ].copy()
    if plot_frame.empty:
        return
    plt.figure(figsize=(10, 7))
    plt.scatter(
        plot_frame[time_col] / 60,
        plot_frame["metadata_page_count"],
        c=plot_frame[wpm_col],
        cmap="plasma",
        s=36,
        alpha=0.85,
    )
    for _, row in plot_frame.iterrows():
        wpm = row[wpm_col]
        if pd.notna(wpm) and (wpm >= plot_frame[wpm_col].quantile(0.9)):
            plt.annotate(
                str(row["title"])[:28],
                (row[time_col] / 60, row["metadata_page_count"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    plt.colorbar(label="WPM")
    plt.xlabel(x_label)
    plt.ylabel("Estimated pages")
    plt.title(f"{title} (N={len(plot_frame)})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_highlight_effects_and_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    highlight_df = prepare_highlight_frame(frame, "wpm_first_pass_primary")
    if len(highlight_df) < 3:
        return
    x = np.log1p(highlight_df["highlight_words_per_10k_words"].astype(float))
    y = highlight_df["wpm_first_pass_primary"].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(highlight_df) >= 2 else np.nan
    p25 = float(np.percentile(highlight_df["highlight_words_per_10k_words"], 25))
    p75 = float(np.percentile(highlight_df["highlight_words_per_10k_words"], 75))
    predicted_p25 = float(intercept + slope * np.log1p(p25))
    predicted_p75 = float(intercept + slope * np.log1p(p75))
    summary = pd.DataFrame(
        [
            {
                "n_books": len(highlight_df),
                "pearson_corr_log_highlight_density_wpm": corr,
                "slope_wpm_per_log1p_highlight_density": float(slope),
                "p25_highlight_words_per_10k_words": p25,
                "p75_highlight_words_per_10k_words": p75,
                "predicted_wpm_at_p25_highlight_density": predicted_p25,
                "predicted_wpm_at_p75_highlight_density": predicted_p75,
                "observational_wpm_change_if_p75_to_p25_density": (
                    predicted_p25 - predicted_p75
                ),
            }
        ]
    )
    summary.to_csv(output_dir / "book_speed_highlight_effect_summary.csv", index=False)
    _write_markdown_table(
        summary, output_dir / "book_speed_highlight_effect_summary.md"
    )
    write_highlight_category_models(highlight_df, output_dir)
    plot_highlight_scatter(
        highlight_df,
        "highlighted_words",
        output_dir / "book_speed_wpm_vs_highlighted_words.png",
        "WPM vs Highlighted Words",
    )
    plot_highlight_scatter(
        highlight_df,
        "highlight_words_per_10k_words",
        output_dir / "book_speed_highlight_density_vs_wpm.png",
        "WPM vs Highlight Density",
        log_x=True,
    )
    plot_highlight_scatter(
        highlight_df,
        "highlight_count_per_page",
        output_dir / "book_speed_highlight_count_per_page_vs_wpm.png",
        "WPM vs Highlights Per Page",
        log_x=True,
    )
    plot_highlight_scatter(
        highlight_df,
        "my_note_words_per_page",
        output_dir / "book_speed_note_words_per_page_vs_wpm.png",
        "WPM vs My Note Words Per Page",
        log_x=True,
    )


def write_cleaned_and_full_highlight_plots(
    merged: pd.DataFrame, output_dir: Path
) -> None:
    cleaned = merged[
        merged["usable_for_visual_reading_speed"].fillna(False)
        & merged["wpm_visual_after_audio_350wpm"].le(600)
    ].copy()
    cleaned_highlights = prepare_highlight_frame(
        cleaned, "wpm_visual_after_audio_350wpm"
    )
    cleaned_highlights.to_csv(
        output_dir / "book_speed_highlight_cleaned_visual_subset.csv", index=False
    )
    if len(cleaned_highlights) >= 3:
        write_highlight_effect_summary_for_frame(
            cleaned_highlights,
            output_dir,
            output_stem="book_speed_highlight_cleaned_visual_effect_summary",
            y_col="wpm_visual_after_audio_350wpm",
        )
        write_highlight_category_models(
            cleaned_highlights,
            output_dir,
            y_col="wpm_visual_after_audio_350wpm",
            output_stem="book_speed_highlight_cleaned_visual_note_category_model",
        )
        plot_highlight_scatter(
            cleaned_highlights,
            "highlight_count_per_page",
            output_dir
            / "book_speed_highlight_count_per_page_vs_visual_wpm_cleaned.png",
            "Cleaned Visual WPM vs Highlights Per Page",
            y_col="wpm_visual_after_audio_350wpm",
            y_label="Visual WPM after 350 WPM audiobook adjustment",
            log_x=True,
        )
        plot_highlight_scatter(
            cleaned_highlights,
            "my_note_words_per_page",
            output_dir / "book_speed_note_words_per_page_vs_visual_wpm_cleaned.png",
            "Cleaned Visual WPM vs My Note Words Per Page",
            y_col="wpm_visual_after_audio_350wpm",
            y_label="Visual WPM after 350 WPM audiobook adjustment",
            log_x=True,
        )
        plot_highlight_scatter(
            cleaned_highlights,
            "highlight_words_per_10k_words",
            output_dir / "book_speed_highlight_density_vs_visual_wpm_cleaned.png",
            "Cleaned Visual WPM vs Highlight Density",
            y_col="wpm_visual_after_audio_350wpm",
            y_label="Visual WPM after 350 WPM audiobook adjustment",
            log_x=True,
        )

    full_available = merged[
        merged["uses_local_file_word_count"].fillna(False)
        & merged["chosen_word_count"].notna()
        & merged["primary_first_pass_minutes"].gt(0)
    ].copy()
    full_highlights = prepare_highlight_frame(full_available, "wpm_first_pass_primary")
    full_highlights.to_csv(
        output_dir / "book_speed_highlight_all_finished_available_subset.csv",
        index=False,
    )
    if len(full_highlights) >= 3:
        write_highlight_effect_summary_for_frame(
            full_highlights,
            output_dir,
            output_stem="book_speed_highlight_all_finished_effect_summary",
            y_col="wpm_first_pass_primary",
        )
        total_finished = len(merged)
        plot_highlight_scatter(
            full_highlights,
            "highlight_count_per_page",
            output_dir / "book_speed_highlight_count_per_page_vs_wpm_all_finished.png",
            "All Finished Books With WPM vs Highlights Per Page",
            log_x=True,
            total_n=total_finished,
        )
        plot_highlight_scatter(
            full_highlights,
            "my_note_words_per_page",
            output_dir / "book_speed_note_words_per_page_vs_wpm_all_finished.png",
            "All Finished Books With WPM vs My Note Words Per Page",
            log_x=True,
            total_n=total_finished,
        )
        plot_highlight_scatter(
            full_highlights,
            "highlight_words_per_10k_words",
            output_dir / "book_speed_highlight_density_vs_wpm_all_finished.png",
            "All Finished Books With WPM vs Highlight Density",
            log_x=True,
            total_n=total_finished,
        )


def prepare_highlight_frame(frame: pd.DataFrame, y_col: str) -> pd.DataFrame:
    highlight_df = frame.dropna(subset=[y_col, "highlight_words_per_10k_words"]).copy()
    highlight_df = highlight_df[highlight_df["highlighted_words"].fillna(0).ge(0)]
    if highlight_df.empty:
        return highlight_df
    highlight_df["estimated_pages_for_density"] = highlight_df[
        "metadata_page_count"
    ].fillna(highlight_df["chosen_word_count"] / 275)
    highlight_df = highlight_df[highlight_df["estimated_pages_for_density"].gt(0)]
    highlight_df["highlight_count_per_page"] = (
        highlight_df["highlight_count"].fillna(0)
        / highlight_df["estimated_pages_for_density"]
    )
    highlight_df["my_note_words_per_page"] = (
        highlight_df["my_note_words"].fillna(0)
        / highlight_df["estimated_pages_for_density"]
    )
    return highlight_df


def write_highlight_effect_summary_for_frame(
    highlight_df: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
    y_col: str,
) -> None:
    if len(highlight_df) < 3:
        return
    x = np.log1p(highlight_df["highlight_words_per_10k_words"].astype(float))
    y = highlight_df[y_col].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(highlight_df) >= 2 else np.nan
    p25 = float(np.percentile(highlight_df["highlight_words_per_10k_words"], 25))
    p75 = float(np.percentile(highlight_df["highlight_words_per_10k_words"], 75))
    predicted_p25 = float(intercept + slope * np.log1p(p25))
    predicted_p75 = float(intercept + slope * np.log1p(p75))
    summary = pd.DataFrame(
        [
            {
                "response_column": y_col,
                "n_books": len(highlight_df),
                "pearson_corr_log_highlight_density_response": corr,
                "slope_response_per_log1p_highlight_density": float(slope),
                "p25_highlight_words_per_10k_words": p25,
                "p75_highlight_words_per_10k_words": p75,
                "predicted_response_at_p25_highlight_density": predicted_p25,
                "predicted_response_at_p75_highlight_density": predicted_p75,
                "observational_response_change_if_p75_to_p25_density": (
                    predicted_p25 - predicted_p75
                ),
            }
        ]
    )
    summary.to_csv(output_dir / f"{output_stem}.csv", index=False)
    _write_markdown_table(summary, output_dir / f"{output_stem}.md")


def write_highlight_category_models(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    y_col: str = "wpm_first_pass_primary",
    output_stem: str = "book_speed_highlight_note_category_model",
) -> None:
    predictors = ["highlight_count_per_page", "my_note_words_per_page"]
    model_frame = frame.dropna(subset=[y_col, *predictors]).copy()
    if len(model_frame) < 8:
        return
    model_frame["category_plot"] = model_frame["category"].replace("", np.nan)
    model_frame["category_plot"] = model_frame["category_plot"].fillna("Unknown")
    rows: list[dict[str, object]] = []
    for predictor in predictors:
        x_name = f"log1p_{predictor}"
        model_frame[x_name] = np.log1p(model_frame[predictor].astype(float))
        for include_category in [False, True]:
            coefficient, intercept, r_squared = fit_linear_model(
                model_frame,
                response=y_col,
                predictor=x_name,
                category_col="category_plot" if include_category else None,
            )
            p25 = float(np.percentile(model_frame[predictor], 25))
            p75 = float(np.percentile(model_frame[predictor], 75))
            rows.append(
                {
                    "model": (
                        "category_fixed_effects" if include_category else "unadjusted"
                    ),
                    "response_column": y_col,
                    "predictor": predictor,
                    "n_books": len(model_frame),
                    "coefficient_wpm_per_log1p_unit": coefficient,
                    "intercept_or_baseline": intercept,
                    "r_squared": r_squared,
                    "p25_predictor": p25,
                    "p75_predictor": p75,
                    "estimated_wpm_change_p75_to_p25": coefficient
                    * (np.log1p(p25) - np.log1p(p75)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"{output_stem}.csv", index=False)
    _write_markdown_table(result, output_dir / f"{output_stem}.md")


def fit_linear_model(
    frame: pd.DataFrame,
    *,
    response: str,
    predictor: str,
    category_col: str | None = None,
) -> tuple[float, float, float]:
    columns = [pd.Series(1.0, index=frame.index, name="intercept"), frame[predictor]]
    if category_col is not None:
        dummies = pd.get_dummies(
            frame[category_col], prefix="category", drop_first=True
        )
        columns.extend([dummies[column].astype(float) for column in dummies.columns])
    x = pd.concat(columns, axis=1).astype(float).to_numpy()
    y = frame[response].astype(float).to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan
    return float(beta[1]), float(beta[0]), r_squared


def plot_highlight_scatter(
    frame: pd.DataFrame,
    x_col: str,
    output_path: Path,
    title: str,
    *,
    log_x: bool = False,
    y_col: str = "wpm_first_pass_primary",
    y_label: str = "WPM",
    total_n: int | None = None,
) -> None:
    plot_frame = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_frame.empty:
        return
    x = plot_frame[x_col].astype(float)
    y = plot_frame[y_col].astype(float)
    fit_x = np.log1p(x) if log_x else x
    slope, intercept = np.polyfit(fit_x, y, 1)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    fit_line_x = np.log1p(x_line) if log_x else x_line
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, s=36, alpha=0.8)
    plt.plot(x_line, intercept + slope * fit_line_x, color="black")
    for _, row in plot_frame.nlargest(8, x_col).iterrows():
        plt.annotate(
            str(row["title"])[:28],
            (row[x_col], row[y_col]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    if log_x:
        plt.xscale("symlog", linthresh=1)
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_label)
    if total_n is None:
        n_label = f"N={len(plot_frame)}"
    else:
        n_label = f"N plotted={len(plot_frame)}, N finished={total_n}"
    plt.title(f"{title} ({n_label})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_report(merged: pd.DataFrame, output_dir: Path) -> None:
    usable_all = merged[merged["usable_for_speed"]].copy()
    usable = merged[merged["usable_for_duration_screened_speed"]].copy()
    visual_usable = merged[merged["usable_for_visual_reading_speed"]].copy()
    short = usable_all[~usable_all["usable_for_duration_screened_speed"]].copy()
    audio_dominant = usable[usable["audio_dominant"]].copy()
    high = usable_all[usable_all["wpm_first_pass_primary"].gt(900)].copy()
    high["likely_cause"] = (
        high.apply(likely_high_wpm_cause, axis=1) if not high.empty else []
    )
    fast = usable[usable["wpm_first_pass_primary"].gt(600)].copy()
    fast["likely_cause"] = (
        fast.apply(likely_high_wpm_cause, axis=1) if not fast.empty else []
    )
    summary = pd.DataFrame(
        [
            {
                "usable_books_before_duration_screen": len(usable_all),
                "duration_screened_books": len(usable),
                "excluded_lte_90_minutes": len(short),
                "gt900_wpm_books_investigated": len(high),
                "gt600_wpm_duration_screened_books": len(fast),
                "audio_dominant_books_excluded_from_visual_wpm": len(audio_dominant),
                "visual_reading_books_after_audio_screen": len(visual_usable),
                "median_duration_screened_wpm": (
                    usable["wpm_first_pass_primary"].median()
                    if not usable.empty
                    else np.nan
                ),
                "median_visual_wpm_after_350wpm_audio_adjustment": (
                    visual_usable["wpm_visual_after_audio_350wpm"].median()
                    if not visual_usable.empty
                    else np.nan
                ),
            }
        ]
    )
    percentiles = pd.read_csv(output_dir / "book_speed_percentiles.csv")
    by_year = pd.read_csv(output_dir / "book_speed_by_year.csv")
    by_category = pd.read_csv(output_dir / "book_speed_by_category.csv")
    highlight_summary_path = output_dir / "book_speed_highlight_effect_summary.csv"
    highlight_summary = (
        pd.read_csv(highlight_summary_path)
        if highlight_summary_path.exists()
        else pd.DataFrame()
    )
    aggregate_path = output_dir / "book_speed_aggregate_wpm.csv"
    aggregate_speed = (
        pd.read_csv(aggregate_path) if aggregate_path.exists() else pd.DataFrame()
    )
    confidence_path = output_dir / "book_speed_confidence_comparison.csv"
    confidence_comparison = (
        pd.read_csv(confidence_path) if confidence_path.exists() else pd.DataFrame()
    )
    highlight_model_path = output_dir / "book_speed_highlight_note_category_model.csv"
    highlight_model = (
        pd.read_csv(highlight_model_path)
        if highlight_model_path.exists()
        else pd.DataFrame()
    )
    reconciliation_path = output_dir / "book_calendar_time_reconciliation.csv"
    reconciliation = (
        pd.read_csv(reconciliation_path)
        if reconciliation_path.exists()
        else pd.DataFrame()
    )
    high_columns = [
        "title",
        "matched_finish_date",
        "primary_first_pass_minutes",
        "first_pass_wall_clock_minutes",
        "chosen_word_count",
        "word_count_source",
        "wpm_first_pass_primary",
        "wpm_first_pass_full_wall_clock",
        "duration_screen_exclusion_reason",
        "likely_cause",
    ]
    short_columns = [
        "title",
        "matched_finish_date",
        "primary_first_pass_minutes",
        "first_pass_wall_clock_minutes",
        "chosen_word_count",
        "wpm_first_pass_primary",
        "wpm_first_pass_full_wall_clock",
        "duration_screen_exclusion_reason",
    ]
    report = [
        "# Book Speed Analysis",
        "",
        "Calendar time starts from `data/calendar_analysis.txt`, the cached final TSV produced by `Self_Tracking/calendar_analysis.py`. The primary minute column starts from that dataframe's overlap-adjusted `duration`, then restores full wall-clock duration for book/audiobook rows whose actual overlapped time is longer than 15 minutes; overlaps of exactly 15 minutes remain split by the calendar analysis. Full wall-clock time from `start_time`/`end_time` is retained separately in `first_pass_wall_clock_minutes` and used for the full-time comparison plots.",
        "",
        "WPM cohorts require `word_count_source == local_file_word_count`, so page estimates and online word counts remain in the word-count audit but do not feed the main speed numbers. Local EPUB/PDF counts use the reading/body-text count where section-level filtering is available, with raw extracted words and excluded notes/index/front-back matter retained in the local text audit. Rows above 900 WPM are not excluded. The only duration screen used for the main subset is `primary_first_pass_minutes > 90`, and every short-duration exclusion is listed below and in `book_speed_short_duration_exclusions.csv`.",
        "",
        "## Core Files",
        "",
        "- Calendar time: `book_calendar_first_pass_time.csv` and `book_calendar_first_pass_events.csv`",
        "- Calendar overlap audit: `book_calendar_overlap_audit.csv`",
        "- Calendar reconciliation: `book_calendar_time_reconciliation.csv` and `book_calendar_unmatched_includable_events.csv`",
        "- Word counts: `book_word_counts.csv`, `book_word_count_source_audit.csv`, `book_word_count_validation_summary.csv`, `book_word_count_local_text_audit.csv`, and `book_word_count_local_section_audit.csv`",
        "- Online-vs-local word-count error: `book_word_count_online_error_rates.csv`, `book_word_count_online_error_outliers.csv`, and `book_word_count_online_error_rates.png`",
        "- Joined analysis: `book_speed_analysis.csv`",
        "- Duration-screened subset: `book_speed_duration_screened_subset.csv`",
        "- Visual-reading subset excluding audio-dominant books: `book_speed_visual_reading_subset.csv`",
        "- Audio-dominant exclusions: `book_speed_audio_dominant_exclusions.csv`",
        "- Short-duration exclusions: `book_speed_short_duration_exclusions.csv`",
        "- Rolling category WPM percentiles: `book_speed_rolling_6mo_category_percentiles.csv` and `book_speed_rolling_6mo_category_percentiles.png`",
        "- High-WPM investigation: `book_speed_high_wpm_investigation.csv` and `book_speed_high_wpm_event_evidence.csv`",
        "- >600 WPM duration-screened investigation: `book_speed_gt600_wpm_investigation.csv` and `book_speed_gt600_wpm_event_evidence.csv`",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "## WPM Percentiles",
        "",
        _markdown_table(percentiles),
        "## Aggregate WPM",
        "",
        "Aggregate WPM is `sum(chosen_word_count) / sum(primary_first_pass_minutes)`, not the mean of per-book WPMs. Full wall-clock aggregate WPM is included for comparison.",
        "",
        _markdown_table(aggregate_speed),
        "## WPM By Year",
        "",
        _markdown_table(by_year),
        "## WPM By Category",
        "",
        _markdown_table(by_category),
        "## Highlight Effect",
        "",
        _markdown_table(highlight_summary),
        "## Highlight And Note Models",
        "",
        _markdown_table(highlight_model),
        "## Confidence Comparison",
        "",
        _markdown_table(confidence_comparison),
        "## Calendar Time Reconciliation",
        "",
        _markdown_table(reconciliation),
        "## High-WPM Investigation",
        "",
        _markdown_table(
            high.sort_values("wpm_first_pass_primary", ascending=False)[high_columns]
        ),
        "## >600 WPM Duration-Screened Investigation",
        "",
        _markdown_table(
            fast.sort_values("wpm_first_pass_primary", ascending=False)[high_columns]
        ),
        "## Short-Duration Exclusions",
        "",
        _markdown_table(
            short.sort_values("wpm_first_pass_primary", ascending=False)[short_columns]
        ),
        "## Plots",
        "",
        "![WPM histogram](book_speed_histogram.png)",
        "",
        "![Full wall-clock WPM histogram](book_speed_histogram_full_wall_clock.png)",
        "",
        "![Visual WPM after audiobook adjustment](book_speed_histogram_visual_after_audio_350wpm.png)",
        "",
        "![WPM by year](book_speed_violin_by_year.png)",
        "",
        "![WPM by category](book_speed_violin_by_category.png)",
        "",
        "![Six-month rolling WPM percentiles by category](book_speed_rolling_6mo_category_percentiles.png)",
        "",
        "![Time vs pages](book_time_vs_pages.png)",
        "",
        "![Full wall-clock time vs pages](book_time_vs_pages_full_wall_clock.png)",
        "",
        "![Time vs words](book_time_vs_words.png)",
        "",
        "![Full wall-clock time vs words](book_time_vs_words_full_wall_clock.png)",
        "",
        "![Online word-count search error vs local extraction](book_word_count_online_error_rates.png)",
        "",
        "![WPM vs highlighted words](book_speed_wpm_vs_highlighted_words.png)",
        "",
        "![Highlight density vs WPM](book_speed_highlight_density_vs_wpm.png)",
        "",
        "![Highlights per page vs WPM](book_speed_highlight_count_per_page_vs_wpm.png)",
        "",
        "![My note words per page vs WPM](book_speed_note_words_per_page_vs_wpm.png)",
        "",
        "![High-confidence histogram](book_speed_histogram_high_confidence.png)",
        "",
        "![Medium-confidence histogram](book_speed_histogram_medium_confidence.png)",
        "",
    ]
    (output_dir / "book_speed_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calendar-time-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "book_calendar_first_pass_time.csv",
    )
    parser.add_argument(
        "--word-counts-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "book_word_counts.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_speed_analysis(
        calendar_time_csv=args.calendar_time_csv,
        word_counts_csv=args.word_counts_csv,
        output_dir=args.output_dir,
    )
    with pd.option_context("display.max_rows", 30, "display.width", 180):
        print(
            result.sort_values("matched_finish_date", ascending=False)
            .head(30)
            .to_string(index=False)
        )
    print(f"\nWrote speed analysis to {args.output_dir / 'book_speed_analysis.csv'}")


if __name__ == "__main__":
    main()
