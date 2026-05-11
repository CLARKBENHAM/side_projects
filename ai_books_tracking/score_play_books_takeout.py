"""Score Play Books takeout titles with the current Goodreads-aware models."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

from ai_books_tracking.future_prediction_evaluation import (
    FeatureSpec,
    model_predictions_for_split,
)
from ai_books_tracking.goodreads_followup_analysis import (
    derive_analysis_columns,
    load_data,
)
from ai_books_tracking.goodreads_ratings import (
    MATCH_CACHE_FILE,
    clean_title_for_search,
    enrich_books_with_goodreads,
    load_cache,
    normalize_text,
)
from ai_books_tracking.merge_goodreads_into_enriched import (
    apply_goodreads_confidence_gate,
)
from ai_books_tracking.new_books_to_rate_analysis import (
    NEW_BOOKS_ENRICHED,
    add_centered_targets,
)

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR.parent / "data"

FINISHED_CSV = DATA_DIR / "finished_books_2025_03_16.csv"
UNFINISHED_CSV = DATA_DIR / "unfinished_books_2025_03_16.csv"
TAKEOUT_GOODREADS_CSV = OUTPUT_DIR / "takeout_play_books_03_16_25_goodreads.csv"
TAKEOUT_SCORED_CSV = OUTPUT_DIR / "takeout_play_books_03_16_25_scored.csv"
TAKEOUT_SUMMARY_CSV = OUTPUT_DIR / "takeout_play_books_03_16_25_summary.csv"

PLACEHOLDER_AUTHORS = {"", "by", "unknown", "nan", "none"}


@dataclass(frozen=True)
class ScoringRule:
    output_column: str
    target_col: str
    feature_spec: FeatureSpec
    model_name: str
    threshold: float | None = None
    decision_column: str | None = None
    description: str = ""


SCORING_RULES = (
    ScoringRule(
        output_column="pred_avg_enjoyment_conservative_gbm",
        target_col="avg_enjoyment",
        feature_spec=FeatureSpec(
            "preread_plus_goodreads_conservative", include_goodreads="conservative"
        ),
        model_name="GBM",
        threshold=3.8,
        decision_column="decision_enjoyment_gbm_3p8",
        description="Primary stable enjoyment rule",
    ),
    ScoringRule(
        output_column="pred_avg_enjoyment_conservative_gbm_3p7",
        target_col="avg_enjoyment",
        feature_spec=FeatureSpec(
            "preread_plus_goodreads_conservative", include_goodreads="conservative"
        ),
        model_name="GBM",
        threshold=3.7,
        decision_column="decision_enjoyment_gbm_3p7",
        description="More permissive enjoyment rule",
    ),
    ScoringRule(
        output_column="pred_avg_enjoyment_raw_lasso",
        target_col="avg_enjoyment",
        feature_spec=FeatureSpec(
            "preread_plus_goodreads_raw", include_goodreads="raw_best"
        ),
        model_name="Lasso",
        description="Best MAE enjoyment model",
    ),
    ScoringRule(
        output_column="pred_avg_usefulness_conservative_lasso",
        target_col="avg_usefulness",
        feature_spec=FeatureSpec(
            "preread_plus_goodreads_conservative", include_goodreads="conservative"
        ),
        model_name="Lasso",
        threshold=2.4,
        decision_column="decision_usefulness_lasso_2p4",
        description="Balanced usefulness rule",
    ),
    ScoringRule(
        output_column="pred_avg_usefulness_conservative_ridge",
        target_col="avg_usefulness",
        feature_spec=FeatureSpec(
            "preread_plus_goodreads_conservative", include_goodreads="conservative"
        ),
        model_name="Ridge",
        threshold=2.7,
        decision_column="decision_usefulness_ridge_2p7",
        description="Aggressive usefulness rule",
    ),
    ScoringRule(
        output_column="pred_avg_usefulness_raw_lasso",
        target_col="avg_usefulness",
        feature_spec=FeatureSpec(
            "preread_plus_goodreads_raw", include_goodreads="raw_best"
        ),
        model_name="Lasso",
        threshold=2.4,
        decision_column="decision_usefulness_raw_lasso_2p4",
        description="Best realized usefulness split rule",
    ),
    ScoringRule(
        output_column="pred_avg_usefulness_conservative_rf",
        target_col="avg_usefulness",
        feature_spec=FeatureSpec(
            "preread_plus_goodreads_conservative", include_goodreads="conservative"
        ),
        model_name="Random Forest",
        description="Best point-prediction usefulness model",
    ),
)


def normalize_author_text(value: object) -> str:
    text = normalize_text(value)
    return "" if text.lower() in PLACEHOLDER_AUTHORS else text


def load_takeout_catalog(
    finished_csv: Path = FINISHED_CSV,
    unfinished_csv: Path = UNFINISHED_CSV,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, status in [
        (finished_csv, "finished"),
        (unfinished_csv, "unfinished"),
    ]:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        frame = frame.rename(columns={"bookshelf": "Bookshelf"})
        frame["play_status"] = status
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["title"] = combined["title"].astype(str).str.strip()
    combined["author"] = combined["author"].map(normalize_author_text)
    combined["Bookshelf"] = combined["Bookshelf"].fillna("Unknown Shelf").astype(str)
    combined["filename"] = combined["filename"].astype(str).str.strip()
    combined["earliest_modified"] = pd.to_datetime(
        combined["earliest_modified"], format="mixed", errors="coerce"
    )
    combined["latest_modified"] = pd.to_datetime(
        combined["latest_modified"], format="mixed", errors="coerce"
    )
    combined = combined.drop_duplicates(subset=["filename"]).reset_index(drop=True)
    combined["canonical_key"] = combined["title"].map(
        lambda title: normalize_text(f"{clean_title_for_search(title, '')}::{title}")
    )
    return combined


def build_goodreads_lookup_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in catalog.iterrows():
        search_title = clean_title_for_search(
            row.get("title", ""), row.get("author", "")
        )
        rows.append(
            {
                "canonical_key": row["canonical_key"],
                "title": row.get("title", ""),
                "author": row.get("author", ""),
                "Bookshelf": row.get("Bookshelf", ""),
                "search_title": search_title,
                "search_author": normalize_author_text(row.get("author", "")),
                "filename_play": row.get("filename", ""),
                "filename_ratings2": "",
            }
        )
    return pd.DataFrame(rows)


def add_goodreads_to_catalog(
    catalog: pd.DataFrame,
    output_path: Path = TAKEOUT_GOODREADS_CSV,
    force_refresh: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    lookup = build_goodreads_lookup_frame(catalog)
    goodreads = enrich_books_with_goodreads(
        books=lookup,
        output_path=output_path,
        force_refresh=force_refresh,
        sleep_seconds=0.1,
        verbose=verbose,
    )
    merged = catalog.merge(
        goodreads[
            [
                "canonical_key",
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
        ],
        on=["canonical_key", "search_title"],
        how="left",
    )
    merged = apply_goodreads_confidence_gate(merged)
    merged.to_csv(output_path, index=False)
    return merged


def load_match_cache_with_retry(
    cache_path: Path = MATCH_CACHE_FILE, attempts: int = 5
) -> dict[str, object]:
    for _ in range(attempts):
        try:
            return load_cache(cache_path)
        except ValueError:
            continue
    return {}


def add_goodreads_from_existing_cache(
    catalog: pd.DataFrame,
    output_path: Path = TAKEOUT_GOODREADS_CSV,
) -> pd.DataFrame:
    lookup = build_goodreads_lookup_frame(catalog)
    match_cache = load_match_cache_with_retry()

    rows: list[dict[str, object]] = []
    for _, row in lookup.iterrows():
        match = match_cache.get(
            row["canonical_key"],
            {
                "goodreads_status": "unmatched",
                "goodreads_match_method": "cache_only_missing",
                "goodreads_candidates_json": "[]",
            },
        )
        rows.append({**row.to_dict(), **match})

    goodreads = pd.DataFrame(rows)
    required_columns = [
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
        "goodreads_status",
        "goodreads_match_method",
    ]
    for column in required_columns:
        if column not in goodreads.columns:
            goodreads[column] = pd.NA
    merged = catalog.merge(
        goodreads[
            [
                "canonical_key",
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
        ],
        on="canonical_key",
        how="left",
    )
    merged = apply_goodreads_confidence_gate(merged)
    merged.to_csv(output_path, index=False)
    return merged


def prepare_candidate_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    prepared = catalog.copy()
    for column in [
        "Enjoyment (/5)",
        "Enjoyment (/5)_ratings2",
        "Usefulness /5 to Me",
        "Usefulness /5 to Me_ratings2",
    ]:
        if column not in prepared.columns:
            prepared[column] = np.nan

    if "Long Term Effects" not in prepared.columns:
        prepared["Long Term Effects"] = ""
    prepared["Long Term Effects"] = prepared["Long Term Effects"].fillna("")

    prepared["earliest_modified_ratings2"] = prepared["earliest_modified"]
    prepared["latest_modified_ratings2"] = prepared["latest_modified"]
    prepared["author_ratings2"] = prepared["author"]
    prepared["author_goodreads"] = prepared["goodreads_author"].fillna("")

    if "gb_page_count" not in prepared.columns:
        prepared["gb_page_count"] = np.nan
    if "pub_year" not in prepared.columns:
        prepared["pub_year"] = np.nan

    prepared = derive_analysis_columns(prepared)
    prepared = add_centered_targets(prepared)
    return prepared


def load_training_frame(include_new_holdout: bool = True) -> pd.DataFrame:
    historical = add_centered_targets(derive_analysis_columns(load_data()))
    historical["label_source"] = "historical_main"

    frames = [historical]
    if include_new_holdout and NEW_BOOKS_ENRICHED.exists():
        new_holdout = pd.read_csv(NEW_BOOKS_ENRICHED)
        new_holdout.columns = new_holdout.columns.str.strip()
        new_holdout = add_centered_targets(derive_analysis_columns(new_holdout))
        new_holdout["label_source"] = "new_books_holdout_2026"
        frames.append(new_holdout)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_key = (
        combined["title"].astype(str).str.strip().str.lower()
        + "||"
        + combined.get("filename", pd.Series("", index=combined.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    combined = combined.loc[~dedupe_key.duplicated()].reset_index(drop=True)
    return combined


def attach_known_labels(
    candidates: pd.DataFrame, training: pd.DataFrame
) -> pd.DataFrame:
    labeled = training.copy()
    labeled["title_norm"] = labeled["title"].astype(str).str.strip().str.lower()
    labeled["filename_norm"] = (
        labeled.get("filename", pd.Series("", index=labeled.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    candidates = candidates.copy()
    candidates["title_norm"] = candidates["title"].astype(str).str.strip().str.lower()
    candidates["filename_norm"] = (
        candidates["filename"].astype(str).str.strip().str.lower()
    )

    label_lookup = (
        labeled.sort_values("label_source")
        .drop_duplicates(subset=["filename_norm"], keep="last")
        .set_index("filename_norm")
    )
    title_lookup = (
        labeled.sort_values("label_source")
        .drop_duplicates(subset=["title_norm"], keep="last")
        .set_index("title_norm")
    )

    candidates["already_labeled"] = False
    candidates["actual_avg_enjoyment"] = np.nan
    candidates["actual_avg_usefulness"] = np.nan
    candidates["label_source"] = pd.NA

    for index, row in candidates.iterrows():
        match = None
        if row["filename_norm"] and row["filename_norm"] in label_lookup.index:
            match = label_lookup.loc[row["filename_norm"]]
        elif row["title_norm"] in title_lookup.index:
            match = title_lookup.loc[row["title_norm"]]
        if match is None:
            continue
        candidates.at[index, "already_labeled"] = True
        candidates.at[index, "actual_avg_enjoyment"] = match.get("avg_enjoyment")
        candidates.at[index, "actual_avg_usefulness"] = match.get("avg_usefulness")
        candidates.at[index, "label_source"] = match.get("label_source")

    return candidates.drop(columns=["title_norm", "filename_norm"])


def apply_scoring_rules(
    training: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    scored = candidates.copy()
    for rule in SCORING_RULES:
        predictions = model_predictions_for_split(
            train_df=training[training[rule.target_col].notna()].copy(),
            test_df=scored,
            target_col=rule.target_col,
            spec=rule.feature_spec,
            model_name=rule.model_name,
        )
        scored[rule.output_column] = predictions
        if rule.threshold is not None and rule.decision_column:
            scored[rule.decision_column] = scored[rule.output_column] >= rule.threshold

    scored["primary_decision"] = np.where(
        scored["decision_enjoyment_gbm_3p8"], "read", "skip"
    )
    scored["primary_decision_with_usefulness"] = np.select(
        [
            scored["decision_enjoyment_gbm_3p8"]
            & scored["decision_usefulness_lasso_2p4"],
            scored["decision_enjoyment_gbm_3p8"],
            scored["decision_usefulness_lasso_2p4"],
        ],
        ["read_high_priority", "read", "maybe_useful"],
        default="skip",
    )
    return scored


def summarize_scored_catalog(scored: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "n_books", "value": int(len(scored))},
        {
            "metric": "goodreads_matched",
            "value": int(scored["goodreads_status"].eq("matched").sum()),
        },
        {
            "metric": "goodreads_review",
            "value": int(scored["goodreads_status"].eq("review").sum()),
        },
        {
            "metric": "goodreads_unmatched",
            "value": int(scored["goodreads_status"].eq("unmatched").sum()),
        },
        {
            "metric": "primary_read",
            "value": int(scored["primary_decision"].eq("read").sum()),
        },
        {
            "metric": "primary_read_high_priority",
            "value": int(
                scored["primary_decision_with_usefulness"]
                .eq("read_high_priority")
                .sum()
            ),
        },
        {
            "metric": "already_labeled",
            "value": int(scored["already_labeled"].sum()),
        },
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finished-csv", type=Path, default=FINISHED_CSV)
    parser.add_argument("--unfinished-csv", type=Path, default=UNFINISHED_CSV)
    parser.add_argument("--goodreads-output", type=Path, default=TAKEOUT_GOODREADS_CSV)
    parser.add_argument("--output", type=Path, default=TAKEOUT_SCORED_CSV)
    parser.add_argument("--summary-output", type=Path, default=TAKEOUT_SUMMARY_CSV)
    parser.add_argument("--force-goodreads-refresh", action="store_true")
    parser.add_argument("--goodreads-cache-only", action="store_true")
    parser.add_argument("--exclude-new-holdout-from-training", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_takeout_catalog(args.finished_csv, args.unfinished_csv)
    if args.goodreads_cache_only:
        catalog = add_goodreads_from_existing_cache(
            catalog,
            output_path=args.goodreads_output,
        )
    else:
        catalog = add_goodreads_to_catalog(
            catalog,
            output_path=args.goodreads_output,
            force_refresh=args.force_goodreads_refresh,
            verbose=args.verbose,
        )
    candidates = prepare_candidate_frame(catalog)
    training = load_training_frame(
        include_new_holdout=not args.exclude_new_holdout_from_training
    )
    scored = apply_scoring_rules(training, candidates)
    scored = attach_known_labels(scored, training)
    scored = scored.sort_values(
        [
            "primary_decision",
            "primary_decision_with_usefulness",
            "pred_avg_enjoyment_conservative_gbm",
            "pred_avg_usefulness_conservative_lasso",
            "goodreads_rating",
        ],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    scored.to_csv(args.output, index=False)

    summary = summarize_scored_catalog(scored)
    summary.to_csv(args.summary_output, index=False)

    print(f"Scored {len(scored)} takeout books")
    print(summary.to_string(index=False))
    print(f"\nWrote scored catalog to: {args.output}")
    print(f"Wrote Goodreads-enriched catalog to: {args.goodreads_output}")


if __name__ == "__main__":
    main()
