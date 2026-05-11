"""Score unread external-search books with the chosen simple and complex models."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_books_tracking.future_prediction_evaluation import (
    FeatureSpec,
    model_predictions_for_split as goodreads_model_predictions_for_split,
)
from ai_books_tracking.multi_source_cleaning_analysis import (
    MERGED_CSV,
    SourceFeatureSpec,
    model_predictions_for_split as multisource_model_predictions_for_split,
    parse_combined_field,
    prepare_model_frame,
)
from ai_books_tracking.score_play_books_takeout import (
    load_training_frame as load_goodreads_training_frame,
    prepare_candidate_frame as prepare_goodreads_candidate_frame,
)

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR.parent / "data"
DEFAULT_INPUT_CSV = (
    DATA_DIR
    / "Books Read and their effects - takeout_play_books_03_16_25_unread_for_external_search_cleaned.csv"
)
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "unread_external_search_scored.csv"
DEFAULT_SUMMARY_CSV = OUTPUT_DIR / "unread_external_search_started_summary.csv"
GOODREADS_RECOMMENDATIONS_TXT = (
    OUTPUT_DIR / "unread_external_search_goodreads_rf_recommendations.txt"
)
MULTISOURCE_RECOMMENDATIONS_TXT = (
    OUTPUT_DIR / "unread_external_search_all_sources_rf_recommendations.txt"
)


@dataclass(frozen=True)
class ModelRun:
    slug: str
    label: str
    threshold: float
    pred_enjoyment_col: str
    pred_usefulness_col: str
    keep_col: str


GOODREADS_RF = ModelRun(
    slug="goodreads_rf",
    label="Goodreads conservative Random Forest",
    threshold=3.9,
    pred_enjoyment_col="pred_avg_enjoyment_goodreads_rf",
    pred_usefulness_col="pred_avg_usefulness_goodreads_rf",
    keep_col="keep_goodreads_rf",
)

MULTISOURCE_RF = ModelRun(
    slug="all_sources_rf",
    label="All-sources Random Forest",
    threshold=3.7,
    pred_enjoyment_col="pred_avg_enjoyment_all_sources_rf",
    pred_usefulness_col="pred_avg_usefulness_all_sources_rf",
    keep_col="keep_all_sources_rf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--goodreads-recommendations",
        type=Path,
        default=GOODREADS_RECOMMENDATIONS_TXT,
    )
    parser.add_argument(
        "--multisource-recommendations",
        type=Path,
        default=MULTISOURCE_RECOMMENDATIONS_TXT,
    )
    return parser.parse_args()


def _clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.where(cleaned.ne("nan"), "")


def coalesce_text_series(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    primary_clean = _clean_text_series(primary)
    fallback_clean = _clean_text_series(fallback)
    return primary_clean.where(primary_clean.ne(""), fallback_clean)


def parse_datetime_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = frame.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(
                parsed[column], format="mixed", errors="coerce"
            )
    return parsed


def started_mask(frame: pd.DataFrame) -> pd.Series:
    play_status = _clean_text_series(
        frame.get("play_status", pd.Series("", index=frame.index))
    )
    has_started_dates = (
        frame["earliest_modified"].notna() | frame["latest_modified"].notna()
    )
    return has_started_dates | play_status.str.lower().eq("finished")


def load_unread_catalog(path: Path = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    unread = pd.read_csv(path)
    unread.columns = unread.columns.str.strip()
    unread = parse_datetime_columns(unread, ["earliest_modified", "latest_modified"])

    unread["display_title"] = coalesce_text_series(
        unread.get("cleaned_title", pd.Series("", index=unread.index)),
        unread.get("title", pd.Series("", index=unread.index)),
    )
    unread["display_author"] = coalesce_text_series(
        unread.get("cleaned_author", pd.Series("", index=unread.index)),
        unread.get("author", pd.Series("", index=unread.index)),
    )
    unread["Bookshelf"] = _clean_text_series(
        unread.get("Bookshelf", pd.Series("", index=unread.index))
    ).replace("", "Unknown Shelf")
    unread["filename"] = _clean_text_series(
        unread.get("filename", pd.Series("", index=unread.index))
    )
    unread["started"] = started_mask(unread)
    unread["started_status"] = np.where(unread["started"], "started", "not_started")

    for column in [
        "goodread ratings",
        "goodreads number reviews",
        "goodreads number ratings",
        "open library ratings",
        "open library number reviews",
        "Unnamed: 24",
        "Unnamed: 25",
    ]:
        if column in unread.columns:
            unread[column] = pd.to_numeric(unread[column], errors="coerce")

    return unread


def _combined_field_to_series(
    frame: pd.DataFrame, column: str
) -> tuple[pd.Series, pd.Series]:
    parsed = frame.get(column, pd.Series(pd.NA, index=frame.index)).apply(
        parse_combined_field
    )
    rating = parsed.map(lambda value: value[0]).astype(float)
    count = parsed.map(lambda value: value[1]).astype(float)
    return rating, count


def prepare_goodreads_candidates(unread: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.DataFrame(
        {
            "title": unread["display_title"],
            "author": unread["display_author"],
            "Bookshelf": unread["Bookshelf"],
            "earliest_modified": unread["earliest_modified"],
            "latest_modified": unread["latest_modified"],
            "filename": unread["filename"],
            "goodreads_author": unread["display_author"],
            "goodreads_rating": pd.to_numeric(
                unread.get("goodread ratings"), errors="coerce"
            ),
            "goodreads_rating_count": pd.to_numeric(
                unread.get("goodreads number ratings"), errors="coerce"
            ),
            "goodreads_rating_raw_best": pd.to_numeric(
                unread.get("goodread ratings"), errors="coerce"
            ),
            "goodreads_rating_count_raw_best": pd.to_numeric(
                unread.get("goodreads number ratings"), errors="coerce"
            ),
            "gb_page_count": np.nan,
            "pub_year": np.nan,
        }
    )
    return prepare_goodreads_candidate_frame(candidates)


def prepare_multisource_candidates(unread: pd.DataFrame) -> pd.DataFrame:
    ol_rating_from_combined, ol_reviews_from_combined = _combined_field_to_series(
        unread, "open library combined"
    )
    amazon_rating_from_combined, amazon_reviews_from_combined = (
        _combined_field_to_series(unread, "Amazon combined with links")
    )
    estimated_finish = unread["latest_modified"].combine_first(
        unread["earliest_modified"]
    )

    candidates = pd.DataFrame(
        {
            "title": unread["display_title"],
            "author": unread["display_author"],
            "category": unread["Bookshelf"],
            "estimated_finish": estimated_finish,
            "page_count": np.nan,
            "pub_year": np.nan,
            "source": "Unread external search",
            "avg_enjoyment": np.nan,
            "avg_usefulness": np.nan,
            "goodreads_rating_verified": pd.to_numeric(
                unread.get("goodread ratings"), errors="coerce"
            ),
            "goodreads_review_count_verified": pd.to_numeric(
                unread.get("goodreads number reviews"), errors="coerce"
            ),
            "goodreads_rating_count_verified": pd.to_numeric(
                unread.get("goodreads number ratings"), errors="coerce"
            ),
            "ol_rating_consensus": pd.to_numeric(
                unread.get("open library ratings"), errors="coerce"
            ).fillna(ol_rating_from_combined),
            "ol_reviews_consensus": pd.to_numeric(
                unread.get("open library number reviews"), errors="coerce"
            ).fillna(ol_reviews_from_combined),
            "amazon_rating_consensus": pd.to_numeric(
                unread.get("Unnamed: 24"), errors="coerce"
            ).fillna(amazon_rating_from_combined),
            "amazon_reviews_consensus": pd.to_numeric(
                unread.get("Unnamed: 25"), errors="coerce"
            ).fillna(amazon_reviews_from_combined),
        }
    )
    return prepare_model_frame(candidates)


def score_goodreads_random_forest(unread: pd.DataFrame) -> pd.DataFrame:
    candidates = prepare_goodreads_candidates(unread)
    training = load_goodreads_training_frame(include_new_holdout=True)
    spec = FeatureSpec(
        "preread_plus_goodreads_conservative",
        include_goodreads="conservative",
    )

    enjoyment_predictions = goodreads_model_predictions_for_split(
        train_df=training[training["avg_enjoyment"].notna()].copy(),
        test_df=candidates,
        target_col="avg_enjoyment",
        spec=spec,
        model_name="Random Forest",
    )
    usefulness_predictions = goodreads_model_predictions_for_split(
        train_df=training[training["avg_usefulness"].notna()].copy(),
        test_df=candidates,
        target_col="avg_usefulness",
        spec=spec,
        model_name="Random Forest",
    )

    scored = unread.copy()
    scored[GOODREADS_RF.pred_enjoyment_col] = enjoyment_predictions
    scored[GOODREADS_RF.pred_usefulness_col] = usefulness_predictions
    scored[GOODREADS_RF.keep_col] = (
        scored[GOODREADS_RF.pred_enjoyment_col] >= GOODREADS_RF.threshold
    )
    return scored


def load_multisource_training() -> pd.DataFrame:
    training = pd.read_csv(MERGED_CSV)
    training.columns = training.columns.str.strip()
    training = prepare_model_frame(training)
    return training[training["source"].ne("Holdout 2026 (unverified)")].copy()


def score_multisource_random_forest(unread: pd.DataFrame) -> pd.DataFrame:
    candidates = prepare_multisource_candidates(unread)
    training = load_multisource_training()
    spec = SourceFeatureSpec(
        "preread_all_sources",
        include_goodreads=True,
        include_openlibrary=True,
        include_amazon=True,
    )

    enjoyment_predictions = multisource_model_predictions_for_split(
        train_df=training[training["avg_enjoyment"].notna()].copy(),
        test_df=candidates,
        target_col="avg_enjoyment",
        spec=spec,
        model_name="Random Forest",
    )
    usefulness_predictions = multisource_model_predictions_for_split(
        train_df=training[training["avg_usefulness"].notna()].copy(),
        test_df=candidates,
        target_col="avg_usefulness",
        spec=spec,
        model_name="Random Forest",
    )

    scored = unread.copy()
    scored[MULTISOURCE_RF.pred_enjoyment_col] = enjoyment_predictions
    scored[MULTISOURCE_RF.pred_usefulness_col] = usefulness_predictions
    scored[MULTISOURCE_RF.keep_col] = (
        scored[MULTISOURCE_RF.pred_enjoyment_col] >= MULTISOURCE_RF.threshold
    )
    return scored


def summarize_started_groups(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in [GOODREADS_RF, MULTISOURCE_RF]:
        for started_status in ["started", "not_started"]:
            subset = scored[scored["started_status"] == started_status].copy()
            rows.append(
                {
                    "model": model.label,
                    "started_status": started_status,
                    "n_books": int(len(subset)),
                    "keep_n": int(subset[model.keep_col].sum()),
                    "keep_share": float(subset[model.keep_col].mean()),
                    "mean_pred_avg_enjoyment": float(
                        subset[model.pred_enjoyment_col].mean()
                    ),
                    "median_pred_avg_enjoyment": float(
                        subset[model.pred_enjoyment_col].median()
                    ),
                    "mean_pred_avg_usefulness": float(
                        subset[model.pred_usefulness_col].mean()
                    ),
                    "median_pred_avg_usefulness": float(
                        subset[model.pred_usefulness_col].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_recommendation_lines(scored: pd.DataFrame, model: ModelRun) -> list[str]:
    lines = [
        f"Model: {model.label}",
        f"Keep rule: predicted avg_enjoyment >= {model.threshold:.1f}",
        "",
    ]

    for started_status, heading in [
        ("started", "Started"),
        ("not_started", "Not Started"),
    ]:
        subset = scored[scored["started_status"] == started_status].copy()
        keep = subset[subset[model.keep_col]].copy()
        discard = subset[~subset[model.keep_col]].copy()
        keep = keep.sort_values(
            [model.pred_enjoyment_col, "display_title"], ascending=[False, True]
        )
        discard = discard.sort_values(
            [model.pred_enjoyment_col, "display_title"], ascending=[True, True]
        )

        lines.append(f"{heading}: {len(subset)} books")
        lines.append(f"Keep: {len(keep)}")
        for row in keep.itertuples(index=False):
            lines.append(
                "  - "
                f"{row.display_title} | {row.display_author or 'Unknown'} | "
                f"{row.Bookshelf} | enjoy={getattr(row, model.pred_enjoyment_col):.2f} | "
                f"useful={getattr(row, model.pred_usefulness_col):.2f}"
            )
        lines.append(f"Discard: {len(discard)}")
        for row in discard.itertuples(index=False):
            lines.append(
                "  - "
                f"{row.display_title} | {row.display_author or 'Unknown'} | "
                f"{row.Bookshelf} | enjoy={getattr(row, model.pred_enjoyment_col):.2f} | "
                f"useful={getattr(row, model.pred_usefulness_col):.2f}"
            )
        lines.append("")

    return lines


def write_recommendations(
    scored: pd.DataFrame, model: ModelRun, output_path: Path
) -> None:
    output_path.write_text("\n".join(build_recommendation_lines(scored, model)))


def build_output_frame(scored: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "display_title",
        "display_author",
        "title",
        "author",
        "Bookshelf",
        "play_status",
        "started",
        "started_status",
        "earliest_modified",
        "latest_modified",
        "goodread ratings",
        "goodreads number reviews",
        "goodreads number ratings",
        "open library ratings",
        "open library number reviews",
        "Unnamed: 24",
        "Unnamed: 25",
        GOODREADS_RF.pred_enjoyment_col,
        GOODREADS_RF.pred_usefulness_col,
        GOODREADS_RF.keep_col,
        MULTISOURCE_RF.pred_enjoyment_col,
        MULTISOURCE_RF.pred_usefulness_col,
        MULTISOURCE_RF.keep_col,
    ]
    available = [column for column in keep_columns if column in scored.columns]
    return scored[available].sort_values(
        ["started", GOODREADS_RF.pred_enjoyment_col, "display_title"],
        ascending=[False, False, True],
    )


def main() -> None:
    args = parse_args()
    unread = load_unread_catalog(args.input_csv)
    scored = score_goodreads_random_forest(unread)
    scored = score_multisource_random_forest(scored)

    output_frame = build_output_frame(scored)
    summary = summarize_started_groups(scored)

    output_frame.to_csv(args.output_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    write_recommendations(scored, GOODREADS_RF, args.goodreads_recommendations)
    write_recommendations(scored, MULTISOURCE_RF, args.multisource_recommendations)

    print(summary.to_string(index=False))
    print()
    print(f"Scored CSV: {args.output_csv}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Goodreads recommendations: {args.goodreads_recommendations}")
    print(f"All-sources recommendations: {args.multisource_recommendations}")


if __name__ == "__main__":
    main()
