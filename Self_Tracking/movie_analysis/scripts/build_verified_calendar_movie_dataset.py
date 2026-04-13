from __future__ import annotations

import csv
import difflib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis_core.calendar_movie_scores import (  # noqa: E402
    LOCAL_TZ,
    canonicalize_movie_title,
    find_new_calendar_movies,
    normalize_tracking_title,
    write_csv,
)


def normalized_dataset_title(row: dict[str, Any]) -> str:
    title = str(row.get("movie_title") or row.get("normalized_title") or "").strip()
    return normalize_tracking_title(canonicalize_movie_title(title))


def load_manual_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_norm"] = normalized_dataset_title(row)
    return rows


def match_manual_rows(
    generated_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    used_indices: set[int] = set()
    enriched_rows: list[dict[str, Any]] = []

    for row in generated_rows:
        row = row.copy()
        row["clark_rating"] = "na"
        generated_norm = normalized_dataset_title(row)
        candidates: list[tuple[float, int]] = []
        for index, manual in enumerate(manual_rows):
            if index in used_indices or manual["_norm"] != generated_norm:
                continue

            score = 0.0
            manual_date = manual.get("date", "").strip()
            generated_date = str(row.get("date", "")).strip()
            if manual_date and manual_date == generated_date:
                score += 100.0
            elif manual_date:
                score -= 50.0
            else:
                score += 5.0

            manual_where = manual.get("where_seen", "").strip().lower()
            generated_where = str(row.get("where_seen", "")).strip().lower()
            if manual_where and manual_where == generated_where:
                score += 10.0

            manual_title = manual.get("movie_title", "")
            generated_title = str(row.get("movie_title", ""))
            score += (
                10.0
                * difflib.SequenceMatcher(
                    None,
                    manual_title.lower(),
                    generated_title.lower(),
                ).ratio()
            )
            candidates.append((score, index))

        if candidates:
            _, best_index = max(candidates)
            used_indices.add(best_index)
            row["clark_rating"] = (
                manual_rows[best_index].get("clark_rating", "na") or "na"
            )

        enriched_rows.append(row)

    unmatched = [
        manual_rows[index]
        for index in range(len(manual_rows))
        if index not in used_indices
    ]
    return enriched_rows, unmatched


def float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_quality_flags(row: dict[str, Any]) -> tuple[str, str]:
    flags: list[str] = []
    rt_year = int_or_none(str(row.get("rt_release_year", "")))
    imdb_year = int_or_none(str(row.get("imdb_release_year", "")))
    year_gap = ""
    if rt_year is not None and imdb_year is not None:
        year_gap_value = abs(rt_year - imdb_year)
        year_gap = str(year_gap_value)
        if year_gap_value > 1:
            flags.append("source_year_gap")
    if not row.get("rt_url"):
        flags.append("missing_rt")
    elif "/tv/" in str(row.get("rt_url", "")):
        flags.append("rt_tv_url")
    if not row.get("imdb_url"):
        flags.append("missing_imdb")
    if row.get("watch_status") == "unfinished":
        flags.append("unfinished_watch")
    if row.get("completion_confidence") == "low":
        flags.append("low_watch_confidence")
    return ";".join(flags), year_gap


def external_mean_rating(row: dict[str, Any]) -> float | None:
    components: list[float] = []
    for key in ("rt_audience_score", "rt_critic_score"):
        value = float_or_none(str(row.get(key, "")))
        if value is not None:
            components.append(value / 10.0)
    imdb = float_or_none(str(row.get("imdb_score", "")))
    if imdb is not None:
        components.append(imdb)
    if not components:
        return None
    return sum(components) / len(components)


def add_quality_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        updated = row.copy()
        flags, year_gap = build_quality_flags(updated)
        updated["quality_flags"] = flags
        updated["rt_imdb_year_gap"] = year_gap
        ext_mean = external_mean_rating(updated)
        updated["external_mean_rating"] = (
            f"{ext_mean:.2f}" if ext_mean is not None else ""
        )
        rating = float_or_none(str(updated.get("clark_rating", "")))
        if rating is not None and ext_mean is not None:
            updated["rating_gap_vs_external_mean"] = f"{rating - ext_mean:.2f}"
            updated["abs_rating_gap_vs_external_mean"] = f"{abs(rating - ext_mean):.2f}"
        else:
            updated["rating_gap_vs_external_mean"] = ""
            updated["abs_rating_gap_vs_external_mean"] = ""
        enriched.append(updated)
    return enriched


def build_quality_review(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    for row in rows:
        rating = float_or_none(str(row.get("clark_rating", "")))
        ext_mean = float_or_none(str(row.get("external_mean_rating", "")))
        if rating is None or ext_mean is None:
            continue
        review_rows.append(
            {
                "date": row["date"],
                "movie_title": row["movie_title"],
                "clark_rating": row["clark_rating"],
                "external_mean_rating": row["external_mean_rating"],
                "rating_gap_vs_external_mean": row["rating_gap_vs_external_mean"],
                "abs_rating_gap_vs_external_mean": row[
                    "abs_rating_gap_vs_external_mean"
                ],
                "watch_status": row["watch_status"],
                "where_seen": row["where_seen"],
                "quality_flags": row["quality_flags"],
                "calendar_summary": row["calendar_summary"],
                "calendar_location": row["calendar_location"],
                "rt_matched_title": row["rt_matched_title"],
                "rt_release_year": row["rt_release_year"],
                "rt_url": row["rt_url"],
                "imdb_matched_title": row["imdb_matched_title"],
                "imdb_release_year": row["imdb_release_year"],
                "imdb_url": row["imdb_url"],
            }
        )
    review_rows.sort(
        key=lambda row: (
            -float(row["abs_rating_gap_vs_external_mean"]),
            row["date"],
            row["movie_title"],
        )
    )
    return review_rows


def seed_cache(output_cache_path: Path, source_paths: list[Path]) -> None:
    if output_cache_path.exists():
        return
    merged: dict[str, dict[str, Any]] = {}
    for path in source_paths:
        if path.exists():
            merged.update(json.loads(path.read_text()))
    if merged:
        output_cache_path.parent.mkdir(parents=True, exist_ok=True)
        output_cache_path.write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n"
        )


def main() -> None:
    output_dir = (
        ROOT
        / "data"
        / "summaries"
        / "movie_rt_analysis"
        / "calendar_movies_verified_qcfix_20260404"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "new_calendar_movies_match_cache.json"
    seed_cache(
        cache_path,
        [
            ROOT
            / "data"
            / "summaries"
            / "movie_rt_analysis"
            / "calendar_movies_since_2021_20260404"
            / "new_calendar_movies_match_cache.json",
            ROOT
            / "data"
            / "summaries"
            / "movie_rt_analysis"
            / "calendar_movies_20260331"
            / "new_calendar_movies_match_cache.json",
        ],
    )

    rows, ambiguities = find_new_calendar_movies(
        calendar_dir=ROOT.parents[1] / "data" / "Takeout 5" / "Calendar",
        notes_path=ROOT / "data" / "summaries" / "movie_rt_notes.txt",
        existing_rt_csv_path=ROOT
        / "data"
        / "summaries"
        / "movie_rt_analysis"
        / "movie_rt_scores_detailed.csv",
        output_csv_path=output_dir / "new_calendar_movies_rt_imdb.csv",
        ambiguity_csv_path=output_dir / "new_calendar_movies_ambiguities.csv",
        cache_path=cache_path,
        as_of_local=datetime(2026, 4, 4, 23, 59, 59, tzinfo=LOCAL_TZ),
        after_latest_tracked_date=False,
        min_date_local=datetime(2021, 1, 1, 0, 0, 0, tzinfo=LOCAL_TZ),
        include_unfinished_started=True,
        sleep_seconds=0.01,
    )

    manual_path = (
        ROOT
        / "data"
        / "summaries"
        / "Movie Ratings - new_calendar_movies_features-labeled.csv"
    )
    manual_rows = load_manual_rows(manual_path)
    rated_rows, unmatched_manual_rows = match_manual_rows(rows, manual_rows)
    rated_rows = add_quality_columns(rated_rows)

    features_path = output_dir / "new_calendar_movies_features.csv"
    labeled_path = output_dir / "new_calendar_movies_features_labeled_verified.csv"
    root_labeled_path = (
        ROOT
        / "data"
        / "summaries"
        / "Movie Ratings - new_calendar_movies_features-labeled-verified.csv"
    )
    root_qc_labeled_path = (
        ROOT
        / "data"
        / "summaries"
        / "Movie Ratings - new_calendar_movies_features-labeled-verified-qc.csv"
    )
    review_path = output_dir / "rating_outlier_review.csv"
    unmatched_path = output_dir / "unmatched_manual_rows.csv"

    feature_columns = [
        "date",
        "datetime_local",
        "movie_title",
        "normalized_title",
        "watch_status",
        "where_seen",
        "saw_in_home",
        "saw_in_theater",
        "drink_before_movie",
        "completion_basis",
        "completion_confidence",
        "calendar_summary",
        "calendar_location",
        "rt_matched_title",
        "rt_release_year",
        "rt_audience_score",
        "rt_critic_score",
        "rt_url",
        "imdb_matched_title",
        "imdb_release_year",
        "imdb_score",
        "imdb_rating_count",
        "imdb_url",
        "quality_flags",
        "rt_imdb_year_gap",
    ]
    labeled_columns = [
        "date",
        "datetime_local",
        "movie_title",
        "clark_rating",
        "normalized_title",
        "watch_status",
        "where_seen",
        "saw_in_home",
        "saw_in_theater",
        "drink_before_movie",
        "completion_basis",
        "completion_confidence",
        "calendar_summary",
        "calendar_location",
        "rt_matched_title",
        "rt_release_year",
        "rt_audience_score",
        "rt_critic_score",
        "rt_url",
        "imdb_matched_title",
        "imdb_release_year",
        "imdb_score",
        "imdb_rating_count",
        "imdb_url",
        "quality_flags",
        "rt_imdb_year_gap",
        "external_mean_rating",
        "rating_gap_vs_external_mean",
        "abs_rating_gap_vs_external_mean",
    ]

    write_csv(
        features_path,
        [
            {column: row.get(column, "") for column in feature_columns}
            for row in rated_rows
        ],
    )
    labeled_output_rows = [
        {column: row.get(column, "") for column in labeled_columns}
        for row in rated_rows
    ]
    write_csv(labeled_path, labeled_output_rows)
    write_csv(root_labeled_path, labeled_output_rows)
    write_csv(root_qc_labeled_path, labeled_output_rows)
    write_csv(review_path, build_quality_review(rated_rows))
    write_csv(unmatched_path, unmatched_manual_rows)

    print(f"Wrote {len(rated_rows)} labeled rows")
    print(labeled_path)
    print(f"Wrote {len(ambiguities)} ambiguity rows")
    print(output_dir / "new_calendar_movies_ambiguities.csv")
    print(f"Wrote {len(unmatched_manual_rows)} unmatched manual rows")
    print(unmatched_path)
    print(review_path)


if __name__ == "__main__":
    main()
