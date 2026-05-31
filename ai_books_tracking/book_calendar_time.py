"""Build auditable first-pass reading-time tables from calendar_analysis output.

The source of truth for calendar preprocessing is
``Self_Tracking/calendar_analysis.py``. That script writes
``data/calendar_analysis.txt`` after recurrence expansion, slash splitting,
book-title abbreviation, sleep handling, and overlap adjustment. This script
starts from that processed TSV and does only book-specific attribution.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_books_tracking.book_wpm_calendar_notes import (  # noqa: E402
    DEFAULT_ANALYSIS_END,
    DEFAULT_NOTES_DIR,
    build_finish_timeline,
    build_known_titles,
    classify_summary_part,
    filter_candidate_titles,
    load_all_note_stats,
    load_metadata_records,
    normalize_title,
    resolve_reading_events,
    ref_matches_title,
    split_event_summary,
    title_match_score,
    titles_match,
)


DEFAULT_CALENDAR_ANALYSIS_TSV = REPO_ROOT / "data" / "calendar_analysis.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "book_wpm_outputs"
PRINT_READING_EVENT_TYPES = {"reading", "started"}
FIRST_PASS_EVENT_TYPES = {"reading", "started", "audiobook", "finished"}
OVERLAP_RESTORE_EVENT_TYPES = {"reading", "started", "audiobook", "finished"}
COMPATIBLE_READING_OVERLAP_RE = re.compile(
    r"\b(?:walk|walking|commute|travel|transit|train|bus|subway|metro|"
    r"flight|fly|airport|uber|lyft|taxi|drive|driving|car|bike|biking|"
    r"cycling|ride|riding|hike|hiking)\b",
    re.IGNORECASE,
)
COMPATIBLE_AUDIOBOOK_EXTRA_OVERLAP_RE = re.compile(
    r"\b(?:run|running|gym|workout|exercise|errand|chores?|clean|cleaning|"
    r"laundry|dishes|cook|cooking|meal|breakfast|lunch|dinner|eat|eating|"
    r"grocery|shopping|shower)\b",
    re.IGNORECASE,
)


def _finished_duration_policy(event_name: str, book_ref: str = "") -> tuple[bool, str]:
    text = event_name.strip().lower()
    if re.search(r"\b(?:finished|end of)\s+(?:audio\s*)?book\b", text):
        return True, "explicit_finished_book_duration"
    if re.search(r"\bbook\s+finished\b", text):
        return True, "explicit_book_finished_duration"
    if normalize_title(book_ref):
        return True, "bare_finished_marker_with_book_ref_duration"
    return False, "bare_finished_marker_not_counted"


def load_processed_calendar(calendar_analysis_tsv: Path) -> pd.DataFrame:
    frame = pd.read_csv(calendar_analysis_tsv, sep="\t")
    for column in ["start_time", "end_time"]:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame = frame.dropna(subset=["event_name", "start_time", "end_time", "duration"])
    frame["calendar_analysis_duration_hours"] = pd.to_numeric(
        frame["duration"], errors="coerce"
    )
    frame["wall_clock_duration_hours"] = (
        frame["end_time"] - frame["start_time"]
    ).dt.total_seconds() / 3600
    frame = frame[frame["calendar_analysis_duration_hours"].gt(0)].copy()
    frame["event_id"] = [f"calendar_analysis_row_{idx:06d}" for idx in frame.index]
    return frame


def classify_processed_calendar(processed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, event in processed.iterrows():
        parts = split_event_summary(str(event["event_name"]))
        for part_index, part in enumerate(parts):
            event_type, book_ref = classify_summary_part(
                part, event["start_time"].date()
            )
            if event_type == "other":
                continue
            include = event_type in {"reading", "audiobook", "started"}
            inclusion_rule = "explicit_reading_or_audiobook_duration" if include else ""
            if event_type == "started":
                inclusion_rule = "explicit_started_book_duration"
            if event_type == "generic":
                include = False
                inclusion_rule = "generic_temporal_read_not_counted_by_default"
            if event_type == "finished":
                include, inclusion_rule = _finished_duration_policy(part, book_ref)
            divisor = max(len(parts), 1)
            rows.append(
                {
                    "event_id": f"{event['event_id']}_part_{part_index}",
                    "source_row_id": event["event_id"],
                    "date": event["start_time"].date(),
                    "start": event["start_time"],
                    "end": event["end_time"],
                    "calendar_name": event["calendar_name"],
                    "summary": event["event_name"],
                    "part_summary": part,
                    "event_type": event_type,
                    "book_ref": book_ref,
                    "calendar_analysis_duration_hours": (
                        float(event["calendar_analysis_duration_hours"]) / divisor
                    ),
                    "wall_clock_duration_hours": (
                        float(event["wall_clock_duration_hours"]) / divisor
                    ),
                    "include_in_first_pass_time": include,
                    "inclusion_rule": inclusion_rule,
                }
            )
    return attach_overlap_policy(pd.DataFrame(rows), processed)


def _is_compatible_overlap(event_type: str, other_summary: object) -> bool:
    text = str(other_summary or "")
    if COMPATIBLE_READING_OVERLAP_RE.search(text):
        return True
    return event_type == "audiobook" and bool(
        COMPATIBLE_AUDIOBOOK_EXTRA_OVERLAP_RE.search(text)
    )


def attach_overlap_policy(
    classified: pd.DataFrame, processed: pd.DataFrame
) -> pd.DataFrame:
    if classified.empty:
        return classified
    frame = classified.copy()
    frame["overlap_policy_duration_hours"] = frame["calendar_analysis_duration_hours"]
    frame["overlap_policy_rule"] = "calendar_analysis_duration"
    frame["overlap_other_events"] = ""
    frame["overlap_other_calendars"] = ""
    frame["overlap_restored_hours"] = 0.0
    processed_lookup = processed.set_index("event_id", drop=False)
    for idx, row in frame.iterrows():
        if row["event_type"] not in OVERLAP_RESTORE_EVENT_TYPES:
            continue
        source_id = row["source_row_id"]
        if source_id not in processed_lookup.index:
            continue
        source = processed_lookup.loc[source_id]
        wall_hours = float(row["wall_clock_duration_hours"] or 0)
        calendar_hours = float(row["calendar_analysis_duration_hours"] or 0)
        if wall_hours <= calendar_hours + 1e-9:
            continue
        is_minor_overlap = wall_hours - calendar_hours <= 0.25
        overlaps = processed[
            (processed["event_id"] != source_id)
            & (processed["start_time"] < source["end_time"])
            & (processed["end_time"] > source["start_time"])
        ].copy()
        if overlaps.empty and not is_minor_overlap:
            continue
        overlap_seconds = (
            overlaps["end_time"].clip(upper=source["end_time"])
            - overlaps["start_time"].clip(lower=source["start_time"])
        ).dt.total_seconds()
        overlaps = overlaps[overlap_seconds.gt(60)].copy()
        if overlaps.empty and not is_minor_overlap:
            continue
        summaries = overlaps["event_name"].astype(str).tolist()
        calendars = overlaps["calendar_name"].astype(str).tolist()
        frame.loc[idx, "overlap_other_events"] = " | ".join(summaries[:8])
        frame.loc[idx, "overlap_other_calendars"] = " | ".join(sorted(set(calendars)))
        if is_minor_overlap:
            frame.loc[idx, "overlap_policy_duration_hours"] = wall_hours
            frame.loc[idx, "overlap_policy_rule"] = "full_duration_minor_overlap"
            frame.loc[idx, "overlap_restored_hours"] = wall_hours - calendar_hours
            continue
        compatible = all(
            _is_compatible_overlap(str(row["event_type"]), summary)
            for summary in summaries
        )
        if not compatible:
            frame.loc[idx, "overlap_policy_rule"] = (
                "calendar_analysis_duration_incompatible_overlap"
            )
            continue
        frame.loc[idx, "overlap_policy_duration_hours"] = wall_hours
        frame.loc[idx, "overlap_policy_rule"] = "full_duration_compatible_overlap"
        frame.loc[idx, "overlap_restored_hours"] = wall_hours - calendar_hours
    return frame


def _attach_finish_ids_to_finished_events(
    classified: pd.DataFrame, finishes: pd.DataFrame
) -> pd.DataFrame:
    frame = classified.copy()
    if frame.empty or finishes.empty:
        return frame
    finish_lookup = finishes.copy()
    finish_lookup["_date"] = pd.to_datetime(finish_lookup["finish_date"]).dt.date
    for idx, row in frame[frame["event_type"] == "finished"].iterrows():
        ref = str(row["book_ref"])
        candidate_titles = filter_candidate_titles(
            ref, finish_lookup["title"].astype(str).tolist(), fallback_to_all=True
        )
        candidates = finish_lookup[
            (finish_lookup["_date"] == row["date"])
            & (
                finish_lookup["title"].isin(candidate_titles)
                | finish_lookup["cal_ref"].map(lambda value: titles_match(ref, value))
            )
        ]
        if candidates.empty:
            continue
        match = candidates.iloc[0]
        frame.loc[idx, "title"] = match["title"]
        frame.loc[idx, "finish_id"] = match["finish_id"]
        frame.loc[idx, "matched_finish_date"] = match["finish_date"]
        frame.loc[idx, "match_method"] = "finished_event_identity"
    return frame


def _sum_minutes(
    group: pd.DataFrame, column: str, event_types: set[str] | None = None
) -> float:
    subset = (
        group if event_types is None else group[group["event_type"].isin(event_types)]
    )
    return float(subset[column].sum() * 60)


def _write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_calendar_time_outputs(
    *,
    calendar_analysis_tsv: Path = DEFAULT_CALENDAR_ANALYSIS_TSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    analysis_end: object = DEFAULT_ANALYSIS_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del analysis_end
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = load_processed_calendar(calendar_analysis_tsv)
    processed.to_csv(output_dir / "book_calendar_processed_base.csv", index=False)

    classified = classify_processed_calendar(processed)
    notes = load_all_note_stats(notes_dir) if notes_dir.exists() else []
    metadata_records = load_metadata_records(notes)
    known_titles = build_known_titles(classified, notes, metadata_records)
    finishes = build_finish_timeline(classified, known_titles)

    reading_like = classified[
        classified["event_type"].isin(["reading", "audiobook", "started", "generic"])
    ].copy()
    resolved_reading = resolve_reading_events(reading_like, finishes)
    resolved = _attach_finish_ids_to_finished_events(classified, finishes)
    if not resolved_reading.empty:
        update_cols = ["title", "finish_id", "matched_finish_date", "match_method"]
        for col in update_cols:
            if col not in resolved:
                resolved[col] = np.nan
        keyed = resolved_reading.set_index("event_id")
        for event_id, row in keyed.iterrows():
            mask = resolved["event_id"] == event_id
            for col in update_cols:
                resolved.loc[mask, col] = row[col]

    included = resolved[
        resolved["include_in_first_pass_time"] & resolved["finish_id"].notna()
    ].copy()
    total_rows: list[dict[str, object]] = []
    if not included.empty:
        group_cols = ["finish_id", "title", "matched_finish_date"]
        for keys, group in included.groupby(group_cols, dropna=False):
            finish_id, title, matched_finish_date = keys
            calendar_minutes = _sum_minutes(group, "calendar_analysis_duration_hours")
            wall_clock_minutes = _sum_minutes(group, "wall_clock_duration_hours")
            overlap_policy_minutes = _sum_minutes(
                group, "overlap_policy_duration_hours"
            )
            row = {
                "finish_id": finish_id,
                "title": title,
                "matched_finish_date": matched_finish_date,
                "first_pass_calendar_minutes": calendar_minutes,
                "first_pass_wall_clock_minutes": wall_clock_minutes,
                "first_pass_overlap_policy_minutes": overlap_policy_minutes,
                "reading_calendar_minutes": _sum_minutes(
                    group, "calendar_analysis_duration_hours", PRINT_READING_EVENT_TYPES
                ),
                "reading_wall_clock_minutes": _sum_minutes(
                    group, "wall_clock_duration_hours", PRINT_READING_EVENT_TYPES
                ),
                "reading_overlap_policy_minutes": _sum_minutes(
                    group, "overlap_policy_duration_hours", PRINT_READING_EVENT_TYPES
                ),
                "audiobook_calendar_minutes": _sum_minutes(
                    group, "calendar_analysis_duration_hours", {"audiobook"}
                ),
                "audiobook_wall_clock_minutes": _sum_minutes(
                    group, "wall_clock_duration_hours", {"audiobook"}
                ),
                "audiobook_overlap_policy_minutes": _sum_minutes(
                    group, "overlap_policy_duration_hours", {"audiobook"}
                ),
                "finished_calendar_minutes": _sum_minutes(
                    group, "calendar_analysis_duration_hours", {"finished"}
                ),
                "finished_wall_clock_minutes": _sum_minutes(
                    group, "wall_clock_duration_hours", {"finished"}
                ),
                "finished_overlap_policy_minutes": _sum_minutes(
                    group, "overlap_policy_duration_hours", {"finished"}
                ),
                "n_included_events": int(group["event_id"].count()),
                "first_event": group["date"].min(),
                "last_event": group["date"].max(),
                "primary_first_pass_minutes": overlap_policy_minutes,
                "primary_minutes_rule": (
                    "calendar_analysis_duration_with_compatible_overlap_restored"
                ),
            }
            row["calendar_overlap_delta_minutes"] = (
                row["first_pass_wall_clock_minutes"]
                - row["first_pass_calendar_minutes"]
            )
            row["overlap_policy_restored_minutes"] = (
                row["first_pass_overlap_policy_minutes"]
                - row["first_pass_calendar_minutes"]
            )
            total_rows.append(row)
    totals = pd.DataFrame(total_rows)
    if not totals.empty:
        totals = totals.merge(
            finishes[
                [
                    "finish_id",
                    "read_instance",
                    "is_reread",
                    "previous_finish_date",
                    "days_since_previous_finish",
                ]
            ],
            on="finish_id",
            how="left",
        )

    resolved.to_csv(output_dir / "book_calendar_first_pass_events.csv", index=False)
    write_overlap_audit(resolved, output_dir)
    totals.to_csv(output_dir / "book_calendar_first_pass_time.csv", index=False)
    if not totals.empty:
        display_cols = [
            "title",
            "matched_finish_date",
            "primary_first_pass_minutes",
            "first_pass_wall_clock_minutes",
            "first_pass_overlap_policy_minutes",
            "first_pass_calendar_minutes",
            "reading_wall_clock_minutes",
            "reading_overlap_policy_minutes",
            "audiobook_wall_clock_minutes",
            "audiobook_overlap_policy_minutes",
            "finished_wall_clock_minutes",
            "finished_overlap_policy_minutes",
            "n_included_events",
            "overlap_policy_restored_minutes",
            "calendar_overlap_delta_minutes",
        ]
        _write_markdown_table(
            totals.sort_values("matched_finish_date")[
                [column for column in display_cols if column in totals.columns]
            ],
            output_dir / "book_calendar_first_pass_time.md",
        )
    finishes.to_csv(output_dir / "book_calendar_first_pass_finishes.csv", index=False)
    write_reconciliation_outputs(resolved, finishes, output_dir)
    return totals, resolved, finishes


def write_overlap_audit(resolved: pd.DataFrame, output_dir: Path) -> None:
    if resolved.empty or "overlap_policy_duration_hours" not in resolved:
        return
    frame = resolved[
        resolved["event_type"].isin(FIRST_PASS_EVENT_TYPES)
        & resolved["include_in_first_pass_time"]
    ].copy()
    frame["calendar_analysis_minutes"] = frame["calendar_analysis_duration_hours"] * 60
    frame["wall_clock_minutes"] = frame["wall_clock_duration_hours"] * 60
    frame["overlap_policy_minutes"] = frame["overlap_policy_duration_hours"] * 60
    frame["wall_minus_calendar_minutes"] = (
        frame["wall_clock_minutes"] - frame["calendar_analysis_minutes"]
    )
    frame["policy_restored_minutes"] = (
        frame["overlap_policy_minutes"] - frame["calendar_analysis_minutes"]
    )
    frame = frame[frame["wall_minus_calendar_minutes"].gt(1)].copy()
    columns = [
        "date",
        "summary",
        "part_summary",
        "event_type",
        "book_ref",
        "title",
        "matched_finish_date",
        "calendar_analysis_minutes",
        "wall_clock_minutes",
        "overlap_policy_minutes",
        "policy_restored_minutes",
        "overlap_policy_rule",
        "overlap_other_events",
        "overlap_other_calendars",
    ]
    existing = [column for column in columns if column in frame.columns]
    frame = frame.sort_values(["date", "summary"])
    frame[existing].to_csv(output_dir / "book_calendar_overlap_audit.csv", index=False)
    _write_markdown_table(
        frame[existing].head(120), output_dir / "book_calendar_overlap_audit.md"
    )


def write_reconciliation_outputs(
    resolved: pd.DataFrame, finishes: pd.DataFrame, output_dir: Path
) -> None:
    if resolved.empty:
        return
    explicit = resolved[
        resolved["event_type"].isin(FIRST_PASS_EVENT_TYPES)
        & resolved["include_in_first_pass_time"]
    ].copy()
    generic = resolved[resolved["event_type"].eq("generic")].copy()
    bare_finished = resolved[
        resolved["event_type"].eq("finished") & ~resolved["include_in_first_pass_time"]
    ].copy()
    assigned = explicit[explicit["finish_id"].notna()].copy()
    unmatched = explicit[explicit["finish_id"].isna()].copy()
    rows = [
        {
            "metric": "explicit_includable_book_time",
            "events": len(explicit),
            "calendar_analysis_minutes": explicit[
                "calendar_analysis_duration_hours"
            ].sum()
            * 60,
            "wall_clock_minutes": explicit["wall_clock_duration_hours"].sum() * 60,
        },
        {
            "metric": "assigned_to_finished_books",
            "events": len(assigned),
            "calendar_analysis_minutes": assigned[
                "calendar_analysis_duration_hours"
            ].sum()
            * 60,
            "wall_clock_minutes": assigned["wall_clock_duration_hours"].sum() * 60,
        },
        {
            "metric": "unmatched_explicit_book_time",
            "events": len(unmatched),
            "calendar_analysis_minutes": unmatched[
                "calendar_analysis_duration_hours"
            ].sum()
            * 60,
            "wall_clock_minutes": unmatched["wall_clock_duration_hours"].sum() * 60,
        },
        {
            "metric": "generic_temporal_not_counted",
            "events": len(generic),
            "calendar_analysis_minutes": generic[
                "calendar_analysis_duration_hours"
            ].sum()
            * 60,
            "wall_clock_minutes": generic["wall_clock_duration_hours"].sum() * 60,
        },
        {
            "metric": "bare_finished_marker_not_counted",
            "events": len(bare_finished),
            "calendar_analysis_minutes": bare_finished[
                "calendar_analysis_duration_hours"
            ].sum()
            * 60,
            "wall_clock_minutes": bare_finished["wall_clock_duration_hours"].sum() * 60,
        },
    ]
    summary = pd.DataFrame(rows)
    total = summary.loc[
        summary["metric"].eq("explicit_includable_book_time"),
        ["calendar_analysis_minutes", "wall_clock_minutes"],
    ].iloc[0]
    assigned_row = summary.loc[
        summary["metric"].eq("assigned_to_finished_books"),
        ["calendar_analysis_minutes", "wall_clock_minutes"],
    ].iloc[0]
    unmatched_row = summary.loc[
        summary["metric"].eq("unmatched_explicit_book_time"),
        ["calendar_analysis_minutes", "wall_clock_minutes"],
    ].iloc[0]
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [
                    {
                        "metric": "assigned_plus_unmatched_minus_explicit_total",
                        "events": "",
                        "calendar_analysis_minutes": (
                            assigned_row["calendar_analysis_minutes"]
                            + unmatched_row["calendar_analysis_minutes"]
                            - total["calendar_analysis_minutes"]
                        ),
                        "wall_clock_minutes": (
                            assigned_row["wall_clock_minutes"]
                            + unmatched_row["wall_clock_minutes"]
                            - total["wall_clock_minutes"]
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    summary.to_csv(output_dir / "book_calendar_time_reconciliation.csv", index=False)
    unmatched.to_csv(
        output_dir / "book_calendar_unmatched_includable_events.csv", index=False
    )
    write_unmatched_ref_audit(unmatched, finishes, output_dir)


def write_unmatched_ref_audit(
    unmatched: pd.DataFrame, finishes: pd.DataFrame, output_dir: Path
) -> None:
    if unmatched.empty:
        pd.DataFrame().to_csv(output_dir / "book_calendar_unmatched_ref_audit.csv")
        return
    finish_rows = finishes.copy()
    if not finish_rows.empty:
        finish_rows["finish_date"] = pd.to_datetime(finish_rows["finish_date"]).dt.date
    grouped = (
        unmatched.groupby(["event_type", "book_ref"], dropna=False)
        .agg(
            events=("event_id", "count"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            wall_clock_minutes=(
                "wall_clock_duration_hours",
                lambda values: float(values.sum() * 60),
            ),
            calendar_analysis_minutes=(
                "calendar_analysis_duration_hours",
                lambda values: float(values.sum() * 60),
            ),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for _, row in grouped.iterrows():
        ref = "" if pd.isna(row["book_ref"]) else str(row["book_ref"])
        first_date = pd.to_datetime(row["first_date"]).date()
        last_date = pd.to_datetime(row["last_date"]).date()
        candidates: list[tuple[float, object, str, str]] = []
        for _, finish in finish_rows.iterrows():
            if finish["finish_date"] < first_date:
                continue
            score = max(
                title_match_score(ref, str(finish["title"])),
                title_match_score(ref, str(finish["cal_ref"])),
            )
            if (
                score >= 50
                or ref_matches_title(ref, str(finish["title"]))
                or ref_matches_title(ref, str(finish["cal_ref"]))
            ):
                candidates.append(
                    (
                        score,
                        finish["finish_date"],
                        str(finish["title"]),
                        str(finish["cal_ref"]),
                    )
                )
        candidates.sort(key=lambda item: (-item[0], abs((item[1] - last_date).days)))
        best = candidates[0] if candidates else None
        if best is None:
            classification = "no_finished_book_candidate_found"
            best_score = np.nan
            best_date = pd.NaT
            best_title = ""
            best_ref = ""
            days_after_last = np.nan
        else:
            best_score, best_date, best_title, best_ref = best
            days_after_last = (best_date - last_date).days
            classification = (
                "possible_missed_finished_book_match"
                if best_score >= 65
                else "weak_candidate_needs_manual_review"
            )
        rows.append(
            {
                **row.to_dict(),
                "classification": classification,
                "best_candidate_score": best_score,
                "best_candidate_finish_date": best_date,
                "best_candidate_title": best_title,
                "best_candidate_ref": best_ref,
                "days_from_last_event_to_candidate_finish": days_after_last,
            }
        )
    audit = pd.DataFrame(rows).sort_values("wall_clock_minutes", ascending=False)
    audit.to_csv(output_dir / "book_calendar_unmatched_ref_audit.csv", index=False)
    _write_markdown_table(
        audit.head(80), output_dir / "book_calendar_unmatched_ref_audit.md"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calendar-analysis-tsv", type=Path, default=DEFAULT_CALENDAR_ANALYSIS_TSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--notes-dir", type=Path, default=DEFAULT_NOTES_DIR)
    parser.add_argument(
        "--analysis-end",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=DEFAULT_ANALYSIS_END,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    totals, _, _ = build_calendar_time_outputs(
        calendar_analysis_tsv=args.calendar_analysis_tsv,
        output_dir=args.output_dir,
        notes_dir=args.notes_dir,
        analysis_end=args.analysis_end,
    )
    with pd.option_context("display.max_rows", 30, "display.width", 180):
        print(
            totals.sort_values("matched_finish_date", ascending=False)
            .head(30)
            .to_string(index=False)
        )
    print(f"\nWrote calendar time outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
