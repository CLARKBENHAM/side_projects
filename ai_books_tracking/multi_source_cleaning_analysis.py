"""Audit multi-source external ratings and evaluate prediction lift."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_books_tracking.build_golden_master import build as build_golden_master
from ai_books_tracking.goodreads_followup_analysis import spearman_summary

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR.parent / "data"

MASTER_CSV = (
    DATA_DIR / "Books Read and their effects - master_book_metadata_cleaned.csv"
)
MERGED_CSV = OUTPUT_DIR / "golden_master_multi_source.csv"
SOURCE_SUMMARY_CSV = OUTPUT_DIR / "multi_source_quality_summary.csv"
SUSPICIOUS_CSV = OUTPUT_DIR / "multi_source_suspicious_rows.csv"
STRUCTURAL_REVIEW_CSV = OUTPUT_DIR / "multi_source_structural_review_rows.csv"
MODEL_RESULTS_CSV = OUTPUT_DIR / "multi_source_model_results.csv"
MODEL_BEST_CSV = OUTPUT_DIR / "multi_source_model_best.csv"
POLICY_RESULTS_CSV = OUTPUT_DIR / "multi_source_policy_results.csv"
POLICY_BEST_CSV = OUTPUT_DIR / "multi_source_policy_best.csv"
HOLDOUT_PREDICTIONS_LONG_CSV = OUTPUT_DIR / "multi_source_holdout_predictions_long.csv"
HOLDOUT_PREDICTIONS_WIDE_CSV = (
    OUTPUT_DIR / "multi_source_holdout_predictions_vs_actual.csv"
)
LINEAR_COEFFICIENTS_CSV = OUTPUT_DIR / "multi_source_linear_model_coefficients.csv"
LINEAR_RULES_CSV = OUTPUT_DIR / "multi_source_linear_rules.csv"
LINEAR_RULES_MD = OUTPUT_DIR / "multi_source_linear_rules.md"
RATINGS_PLOT = OUTPUT_DIR / "multi_source_ratings_vs_targets.png"
COUNTS_PLOT = OUTPUT_DIR / "multi_source_counts_vs_targets.png"
HOLDOUT_PLOT = OUTPUT_DIR / "multi_source_holdout_prediction_scatter.png"
REPORT_MD = OUTPUT_DIR / "multi_source_analysis_report.md"

SOURCE_EXTERNAL_COLUMNS = [
    "good reads combined",
    "goodread ratings",
    "goodreads number reviews",
    "goodreads number ratings",
    "open library combined",
    "open library ratings",
    "open library number reviews",
    "open library books link",
    "Amazon combined with links",
    "Unnamed: 15",
    "Unnamed: 16",
    "Unnamed: 17",
    "Open LIbrary No links",
    "open library rating",
    "open library num reviews",
    "Amazon no links",
    "amazon ratings",
    "amzon number reviews",
]

TRAIN_SOURCE_EXCLUDE = {"Holdout 2026", "Holdout 2026 (unverified)"}
TARGETS = ("avg_enjoyment", "avg_usefulness")
TARGET_LABELS = {
    "avg_enjoyment": "Average enjoyment",
    "avg_usefulness": "Average usefulness",
}
UTILITY_BASES = {
    "avg_enjoyment": 1.3,
    "avg_usefulness": 1.8,
}
THRESHOLDS = [round(value, 1) for value in np.arange(1.0, 5.01, 0.1)]
POLICY_MIN_SIDE_N = 5
BALANCED_KEEP_SHARE_RANGE = (0.25, 0.75)
PREDICTION_MIN = 1.0
PREDICTION_MAX = 5.0
MIN_TRAIN_ROWS = 25


@dataclass(frozen=True)
class SourceFeatureSpec:
    name: str
    include_goodreads: bool = False
    include_openlibrary: bool = False
    include_amazon: bool = False


FEATURE_SPECS = (
    SourceFeatureSpec("preread_base"),
    SourceFeatureSpec("preread_goodreads", include_goodreads=True),
    SourceFeatureSpec("preread_openlibrary", include_openlibrary=True),
    SourceFeatureSpec("preread_amazon", include_amazon=True),
    SourceFeatureSpec(
        "preread_goodreads_openlibrary",
        include_goodreads=True,
        include_openlibrary=True,
    ),
    SourceFeatureSpec(
        "preread_goodreads_amazon",
        include_goodreads=True,
        include_amazon=True,
    ),
    SourceFeatureSpec(
        "preread_all_sources",
        include_goodreads=True,
        include_openlibrary=True,
        include_amazon=True,
    ),
)


def ridge_model():
    return Ridge(alpha=10.0)


def lasso_model():
    return Lasso(alpha=0.05)


def random_forest_model():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=4,
        random_state=42,
    )


def gbm_model():
    return GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


MODEL_BUILDERS = {
    "Category mean": None,
    "Global mean": None,
    "Ridge": ridge_model,
    "Lasso": lasso_model,
    "Random Forest": random_forest_model,
    "GBM": gbm_model,
}

BASE_NUMERIC_FEATURES = [
    "year_finished",
    "log_pages",
    "book_age",
    "author_target_mean_hist",
    "author_book_count_hist",
]
BASE_CATEGORICAL_FEATURES = ["Bookshelf"]
SOURCE_FEATURE_COLUMNS = {
    "goodreads": [
        "goodreads_available",
        "goodreads_rating_feature",
        "goodreads_log_count_feature",
    ],
    "openlibrary": [
        "openlibrary_available",
        "openlibrary_rating_feature",
        "openlibrary_log_count_feature",
    ],
    "amazon": [
        "amazon_available",
        "amazon_rating_feature",
        "amazon_log_count_feature",
    ],
}
LINEAR_MODEL_NAMES = {"Lasso", "Ridge"}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_title(value: object) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\.(pdf|epub|mobi|html|txt)$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_filename(value: object) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\(\d+\)$", "", text)
    text = re.sub(r"\.(pdf|epub|mobi|html|txt)$", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def parse_combined_field(value: object) -> tuple[float, float, str, bool]:
    if pd.isna(value):
        return (math.nan, math.nan, "", False)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return (math.nan, math.nan, "", False)
    if "still learning" in text.lower():
        return (math.nan, math.nan, "", True)

    parts = [part.strip() for part in text.split("|")]
    url = next((part for part in parts if part.startswith("http")), "")
    numeric_parts = [part for part in parts if not part.startswith("http")]
    cleaned: list[float] = []
    for part in numeric_parts[:2]:
        if part in {"", "N/A", "n/a"}:
            cleaned.append(math.nan)
            continue
        try:
            cleaned.append(float(part))
        except ValueError:
            return (math.nan, math.nan, url, True)
    while len(cleaned) < 2:
        cleaned.append(math.nan)
    return (cleaned[0], cleaned[1], url, False)


def average_available(values: list[float]) -> float:
    valid = [value for value in values if pd.notna(value)]
    if not valid:
        return math.nan
    return float(np.mean(valid))


def log_count_ratio(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right) or left <= 0 or right <= 0:
        return math.nan
    larger = max(left, right)
    smaller = min(left, right)
    return float(larger / smaller)


def build_master_source_frame() -> pd.DataFrame:
    master = pd.read_csv(MASTER_CSV)
    master.columns = master.columns.str.strip()
    master["_source_title_key"] = (
        master["source"].astype(str) + "|||" + master["title"].map(normalize_title)
    )
    master["_source_filename_key"] = (
        master["source"].astype(str)
        + "|||"
        + master["filename"].map(normalize_filename)
    )
    return master


def merge_master_sources(golden: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    prepared_master = master.copy()
    if "_source_title_key" not in prepared_master.columns:
        prepared_master["_source_title_key"] = (
            prepared_master["source"].astype(str)
            + "|||"
            + prepared_master["title"].map(normalize_title)
        )
    if "_source_filename_key" not in prepared_master.columns:
        prepared_master["_source_filename_key"] = (
            prepared_master["source"].astype(str)
            + "|||"
            + prepared_master["filename"].map(normalize_filename)
        )

    by_title = (
        prepared_master.sort_values("title")
        .drop_duplicates("_source_title_key")
        .set_index("_source_title_key")
    )
    by_filename = (
        prepared_master[prepared_master["filename"].map(normalize_filename).ne("")]
        .sort_values("title")
        .drop_duplicates("_source_filename_key")
        .set_index("_source_filename_key")
    )

    merged_rows: list[dict[str, object]] = []
    for row in golden.itertuples(index=False):
        title_key = f"{row.source}|||{normalize_title(row.title)}"
        filename_key = (
            f"{row.source}|||{normalize_filename(getattr(row, 'filename', ''))}"
        )
        matched = None
        match_method = "unmatched"
        if filename_key in by_filename.index and normalize_filename(
            getattr(row, "filename", "")
        ):
            matched = by_filename.loc[filename_key]
            match_method = "filename"
        elif title_key in by_title.index:
            matched = by_title.loc[title_key]
            match_method = "title"

        payload = {
            "external_master_match_found": matched is not None,
            "external_master_match_method": match_method,
        }
        for column in SOURCE_EXTERNAL_COLUMNS:
            payload[column] = (
                matched[column]
                if matched is not None and column in matched.index
                else pd.NA
            )
        merged_rows.append(payload)

    merged = golden.copy()
    merged = pd.concat([merged, pd.DataFrame(merged_rows)], axis=1)
    return merged


def fill_numeric_from_candidates(*values: object) -> float:
    numeric: list[float] = []
    for value in values:
        coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(coerced):
            numeric.append(float(coerced))
    if not numeric:
        return math.nan
    return float(np.mean(numeric))


def prepare_external_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    ol_link_parsed = enriched["open library combined"].apply(parse_combined_field)
    enriched["ol_link_rating_combined"] = ol_link_parsed.map(lambda value: value[0])
    enriched["ol_link_reviews_combined"] = ol_link_parsed.map(lambda value: value[1])
    enriched["ol_link_url_combined"] = ol_link_parsed.map(lambda value: value[2])
    enriched["ol_link_bad_text"] = ol_link_parsed.map(lambda value: value[3])

    ol_no_link_parsed = enriched["Open LIbrary No links"].apply(parse_combined_field)
    enriched["ol_nolink_rating_combined"] = ol_no_link_parsed.map(
        lambda value: value[0]
    )
    enriched["ol_nolink_reviews_combined"] = ol_no_link_parsed.map(
        lambda value: value[1]
    )
    enriched["ol_nolink_bad_text"] = ol_no_link_parsed.map(lambda value: value[3])

    amazon_link_parsed = enriched["Amazon combined with links"].apply(
        parse_combined_field
    )
    enriched["amazon_link_rating_combined"] = amazon_link_parsed.map(
        lambda value: value[0]
    )
    enriched["amazon_link_reviews_combined"] = amazon_link_parsed.map(
        lambda value: value[1]
    )
    enriched["amazon_link_url_combined"] = amazon_link_parsed.map(
        lambda value: value[2]
    )
    enriched["amazon_link_bad_text"] = amazon_link_parsed.map(lambda value: value[3])

    amazon_no_link_parsed = enriched["Amazon no links"].apply(parse_combined_field)
    enriched["amazon_nolink_rating_combined"] = amazon_no_link_parsed.map(
        lambda value: value[0]
    )
    enriched["amazon_nolink_reviews_combined"] = amazon_no_link_parsed.map(
        lambda value: value[1]
    )
    enriched["amazon_nolink_bad_text"] = amazon_no_link_parsed.map(
        lambda value: value[3]
    )

    for column in [
        "goodread ratings",
        "goodreads number reviews",
        "goodreads number ratings",
        "open library ratings",
        "open library number reviews",
        "open library rating",
        "open library num reviews",
        "Unnamed: 15",
        "Unnamed: 16",
        "amazon ratings",
        "amzon number reviews",
    ]:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    enriched["goodreads_rating_verified"] = pd.to_numeric(
        enriched["goodread ratings"], errors="coerce"
    )
    enriched["goodreads_review_count_verified"] = pd.to_numeric(
        enriched["goodreads number reviews"], errors="coerce"
    )
    enriched["goodreads_rating_count_verified"] = pd.to_numeric(
        enriched["goodreads number ratings"], errors="coerce"
    )

    enriched["ol_link_rating"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["ol_link_rating_combined"],
            enriched["open library ratings"],
            strict=False,
        )
    ]
    enriched["ol_link_reviews"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["ol_link_reviews_combined"],
            enriched["open library number reviews"],
            strict=False,
        )
    ]
    enriched["ol_link_url"] = (
        enriched["open library books link"]
        .fillna(enriched["ol_link_url_combined"])
        .astype(str)
        .replace("nan", "")
    )

    enriched["ol_nolink_rating"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["ol_nolink_rating_combined"],
            enriched["open library rating"],
            strict=False,
        )
    ]
    enriched["ol_nolink_reviews"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["ol_nolink_reviews_combined"],
            enriched["open library num reviews"],
            strict=False,
        )
    ]

    enriched["amazon_link_rating"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["amazon_link_rating_combined"],
            enriched["Unnamed: 15"],
            strict=False,
        )
    ]
    enriched["amazon_link_reviews"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["amazon_link_reviews_combined"],
            enriched["Unnamed: 16"],
            strict=False,
        )
    ]
    enriched["amazon_link_url"] = (
        enriched["Unnamed: 17"].fillna(enriched["amazon_link_url_combined"]).astype(str)
    ).replace("nan", "")

    enriched["amazon_nolink_rating"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["amazon_nolink_rating_combined"],
            enriched["amazon ratings"],
            strict=False,
        )
    ]
    enriched["amazon_nolink_reviews"] = [
        fill_numeric_from_candidates(left, right)
        for left, right in zip(
            enriched["amazon_nolink_reviews_combined"],
            enriched["amzon number reviews"],
            strict=False,
        )
    ]

    enriched["ol_rating_variant_abs_diff"] = (
        enriched["ol_link_rating"] - enriched["ol_nolink_rating"]
    ).abs()
    enriched["amazon_rating_variant_abs_diff"] = (
        enriched["amazon_link_rating"] - enriched["amazon_nolink_rating"]
    ).abs()
    enriched["ol_review_variant_ratio"] = [
        log_count_ratio(left, right)
        for left, right in zip(
            enriched["ol_link_reviews"],
            enriched["ol_nolink_reviews"],
            strict=False,
        )
    ]
    enriched["amazon_review_variant_ratio"] = [
        log_count_ratio(left, right)
        for left, right in zip(
            enriched["amazon_link_reviews"],
            enriched["amazon_nolink_reviews"],
            strict=False,
        )
    ]
    return enriched


def fit_goodreads_regression(
    df: pd.DataFrame,
    target_col: str,
    *,
    log_target: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    predictors = pd.DataFrame(
        {
            "goodreads_rating": pd.to_numeric(
                df["goodreads_rating_verified"], errors="coerce"
            ),
            "goodreads_log_reviews": np.log1p(
                pd.to_numeric(df["goodreads_review_count_verified"], errors="coerce")
            ),
            "goodreads_log_ratings": np.log1p(
                pd.to_numeric(df["goodreads_rating_count_verified"], errors="coerce")
            ),
        }
    )
    target = pd.to_numeric(df[target_col], errors="coerce")
    train_mask = target.notna() & predictors.notna().all(axis=1)
    if train_mask.sum() < 12:
        fallback = pd.Series(target.mean(), index=df.index, dtype=float)
        return (
            fallback,
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
        )

    X_train = predictors.loc[train_mask]
    y_train = np.log1p(target.loc[train_mask]) if log_target else target.loc[train_mask]
    model = LinearRegression()
    model.fit(X_train, y_train)

    valid_predictors = predictors.notna().all(axis=1)
    predicted = pd.Series(np.nan, index=df.index, dtype=float)
    predicted.loc[valid_predictors] = model.predict(predictors.loc[valid_predictors])
    if log_target:
        predicted = np.maximum(np.expm1(predicted), 0.0)
    else:
        predicted = predicted.clip(lower=1.0, upper=5.0)
    residual = target - predicted
    resid_std = (
        float(residual.loc[train_mask].std(ddof=0)) if train_mask.sum() else math.nan
    )
    if resid_std and not math.isnan(resid_std):
        zscore = residual / resid_std
    else:
        zscore = pd.Series(np.nan, index=df.index, dtype=float)
    return predicted, residual, zscore


def add_source_consensus(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for source in ["ol", "amazon"]:
        for metric in ["rating", "reviews"]:
            link_col = f"{source}_link_{metric}"
            nolink_col = f"{source}_nolink_{metric}"
            log_target = metric == "reviews"

            predicted_link, residual_link, z_link = fit_goodreads_regression(
                enriched, link_col, log_target=log_target
            )
            predicted_nolink, residual_nolink, z_nolink = fit_goodreads_regression(
                enriched, nolink_col, log_target=log_target
            )

            enriched[f"{source}_{metric}_pred_link"] = predicted_link
            enriched[f"{source}_{metric}_pred_nolink"] = predicted_nolink
            enriched[f"{source}_{metric}_resid_link"] = residual_link
            enriched[f"{source}_{metric}_resid_nolink"] = residual_nolink
            enriched[f"{source}_{metric}_resid_z_link"] = z_link
            enriched[f"{source}_{metric}_resid_z_nolink"] = z_nolink

            filled_link = pd.to_numeric(enriched[link_col], errors="coerce").fillna(
                predicted_link
            )
            filled_nolink = pd.to_numeric(enriched[nolink_col], errors="coerce").fillna(
                predicted_nolink
            )
            enriched[f"{source}_{metric}_consensus"] = pd.concat(
                [filled_link, filled_nolink], axis=1
            ).mean(axis=1)
            enriched[f"{source}_{metric}_observed_variants"] = (
                pd.concat(
                    [
                        pd.to_numeric(enriched[link_col], errors="coerce"),
                        pd.to_numeric(enriched[nolink_col], errors="coerce"),
                    ],
                    axis=1,
                )
                .notna()
                .sum(axis=1)
            )

    enriched["open_library_rating"] = enriched["ol_rating_consensus"]
    enriched["open_library_review_count"] = enriched["ol_reviews_consensus"]
    enriched["amazon_rating_consensus"] = enriched["amazon_rating_consensus"]
    enriched["amazon_review_count_consensus"] = enriched["amazon_reviews_consensus"]
    return enriched


def build_suspicious_rows(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "title",
        "author",
        "source",
        "category",
        "goodreads_rating",
        "goodreads_review_count",
        "open_library_rating_consensus",
        "open_library_review_count_consensus",
        "amazon_rating_consensus",
        "amazon_review_count_consensus",
        "issues",
    ]
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        issues: list[str] = []
        if getattr(row, "amazon_link_bad_text"):
            issues.append("amazon_link_bad_text")
        if getattr(row, "amazon_nolink_bad_text"):
            issues.append("amazon_nolink_bad_text")
        if getattr(row, "ol_link_bad_text"):
            issues.append("openlibrary_link_bad_text")
        if getattr(row, "ol_nolink_bad_text"):
            issues.append("openlibrary_nolink_bad_text")

        if (
            pd.notna(getattr(row, "amazon_rating_variant_abs_diff"))
            and getattr(row, "amazon_rating_variant_abs_diff") > 0.25
        ):
            issues.append("amazon_variant_rating_disagreement")
        if (
            pd.notna(getattr(row, "ol_rating_variant_abs_diff"))
            and getattr(row, "ol_rating_variant_abs_diff") > 0.25
        ):
            issues.append("openlibrary_variant_rating_disagreement")
        if (
            pd.notna(getattr(row, "amazon_review_variant_ratio"))
            and getattr(row, "amazon_review_variant_ratio") > 3
        ):
            issues.append("amazon_variant_review_disagreement")
        if (
            pd.notna(getattr(row, "ol_review_variant_ratio"))
            and getattr(row, "ol_review_variant_ratio") > 3
        ):
            issues.append("openlibrary_variant_review_disagreement")

        for column, label in [
            ("amazon_rating_resid_z_link", "amazon_link_rating_outlier"),
            ("amazon_rating_resid_z_nolink", "amazon_nolink_rating_outlier"),
            ("amazon_reviews_resid_z_link", "amazon_link_review_outlier"),
            ("amazon_reviews_resid_z_nolink", "amazon_nolink_review_outlier"),
            ("ol_rating_resid_z_link", "openlibrary_link_rating_outlier"),
            ("ol_rating_resid_z_nolink", "openlibrary_nolink_rating_outlier"),
            ("ol_reviews_resid_z_link", "openlibrary_link_review_outlier"),
            ("ol_reviews_resid_z_nolink", "openlibrary_nolink_review_outlier"),
        ]:
            value = getattr(row, column)
            if pd.notna(value) and abs(value) >= 2.5:
                issues.append(label)

        if not issues:
            continue

        rows.append(
            {
                "title": row.title,
                "author": row.author,
                "source": row.source,
                "category": row.category,
                "goodreads_rating": row.goodreads_rating_verified,
                "goodreads_review_count": row.goodreads_review_count_verified,
                "open_library_rating_consensus": row.ol_rating_consensus,
                "open_library_review_count_consensus": row.ol_reviews_consensus,
                "amazon_rating_consensus": row.amazon_rating_consensus,
                "amazon_review_count_consensus": row.amazon_reviews_consensus,
                "issues": ";".join(sorted(set(issues))),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["source", "title"])


def summarize_source_quality(
    df: pd.DataFrame, suspicious: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (
        source_name,
        rating_col,
        review_col,
        diff_col,
        ratio_col,
        observed_col,
        suspicious_token,
    ) in [
        (
            "Open Library",
            "ol_rating_consensus",
            "ol_reviews_consensus",
            "ol_rating_variant_abs_diff",
            "ol_review_variant_ratio",
            "ol_rating_observed_variants",
            "openlibrary",
        ),
        (
            "Amazon",
            "amazon_rating_consensus",
            "amazon_reviews_consensus",
            "amazon_rating_variant_abs_diff",
            "amazon_review_variant_ratio",
            "amazon_rating_observed_variants",
            "amazon",
        ),
    ]:
        rating = pd.to_numeric(df[rating_col], errors="coerce")
        review = pd.to_numeric(df[review_col], errors="coerce")
        observed_mask = pd.to_numeric(df[observed_col], errors="coerce").fillna(0).gt(0)
        valid_rating = rating.notna() & observed_mask
        goodreads_valid = pd.to_numeric(
            df["goodreads_rating_verified"], errors="coerce"
        ).notna()
        joint_mask = valid_rating & goodreads_valid
        suspicious_count = (
            suspicious["issues"].str.contains(suspicious_token, na=False).sum()
            if not suspicious.empty
            else 0
        )
        rows.extend(
            [
                {
                    "source": source_name,
                    "metric": "rows_with_observed_rating",
                    "value": float(valid_rating.sum()),
                },
                {
                    "source": source_name,
                    "metric": "rows_with_rating",
                    "value": float(valid_rating.sum()),
                },
                {
                    "source": source_name,
                    "metric": "rows_with_reviews",
                    "value": float(review.notna().sum()),
                },
                {
                    "source": source_name,
                    "metric": "mean_variant_rating_abs_diff",
                    "value": float(pd.to_numeric(df[diff_col], errors="coerce").mean()),
                },
                {
                    "source": source_name,
                    "metric": "mean_variant_review_ratio",
                    "value": float(
                        pd.to_numeric(df[ratio_col], errors="coerce").mean()
                    ),
                },
                {
                    "source": source_name,
                    "metric": "spearman_vs_goodreads_rating",
                    "value": float(
                        spearman_summary(
                            rating[joint_mask],
                            pd.to_numeric(
                                df.loc[joint_mask, "goodreads_rating_verified"],
                                errors="coerce",
                            ),
                        )[0]
                    ),
                },
                {
                    "source": source_name,
                    "metric": "spearman_vs_avg_enjoyment",
                    "value": float(
                        spearman_summary(
                            rating[valid_rating],
                            pd.to_numeric(
                                df.loc[valid_rating, "avg_enjoyment"], errors="coerce"
                            ),
                        )[0]
                    ),
                },
                {
                    "source": source_name,
                    "metric": "spearman_vs_avg_usefulness",
                    "value": float(
                        spearman_summary(
                            rating[valid_rating],
                            pd.to_numeric(
                                df.loc[valid_rating, "avg_usefulness"], errors="coerce"
                            ),
                        )[0]
                    ),
                },
                {
                    "source": source_name,
                    "metric": "suspicious_rows",
                    "value": float(suspicious_count),
                },
            ]
        )
    return pd.DataFrame(rows)


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()
    model_df["Bookshelf"] = model_df["category"].fillna("Unknown")
    model_df["estimated_finish"] = pd.to_datetime(
        model_df["estimated_finish"], format="mixed", errors="coerce"
    )
    model_df["year_finished"] = model_df["estimated_finish"].dt.year.astype(float)
    model_df["canonical_author"] = model_df["author"].map(normalize_title)
    model_df["page_count"] = pd.to_numeric(model_df["page_count"], errors="coerce")
    model_df["pub_year"] = pd.to_numeric(model_df["pub_year"], errors="coerce")
    model_df["log_pages"] = np.log10(1 + model_df["page_count"])
    model_df["book_age"] = model_df["year_finished"] - model_df["pub_year"]
    for column in [
        "goodreads_rating_verified",
        "goodreads_rating_count_verified",
        "goodreads_review_count_verified",
        "ol_rating_consensus",
        "ol_reviews_consensus",
        "amazon_rating_consensus",
        "amazon_reviews_consensus",
        "avg_enjoyment",
        "avg_usefulness",
    ]:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")
    return model_df


def add_author_history_features(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    other = other_df.copy()

    target = pd.to_numeric(train[target_col], errors="coerce")
    global_mean = float(target.mean())
    valid_author = train["canonical_author"].ne("")
    valid_target = target.notna()
    valid = valid_author & valid_target

    author_counts = train.loc[valid_author, "canonical_author"].value_counts()
    author_sums = train.loc[valid].groupby("canonical_author")[target_col].sum()
    author_means = author_sums / author_counts.reindex(author_sums.index)

    train["author_target_mean_hist"] = global_mean
    train["author_book_count_hist"] = (
        train["canonical_author"].map(author_counts).fillna(0).astype(float)
    )

    repeat_mask = valid & train["author_book_count_hist"].gt(1)
    if repeat_mask.any():
        repeat_authors = train.loc[repeat_mask, "canonical_author"]
        train.loc[repeat_mask, "author_target_mean_hist"] = (
            author_sums.loc[repeat_authors].to_numpy()
            - pd.to_numeric(
                train.loc[repeat_mask, target_col], errors="coerce"
            ).to_numpy()
        ) / (train.loc[repeat_mask, "author_book_count_hist"].to_numpy() - 1)

    other["author_target_mean_hist"] = (
        other["canonical_author"].map(author_means).fillna(global_mean).astype(float)
    )
    other["author_book_count_hist"] = (
        other["canonical_author"].map(author_counts).fillna(0).astype(float)
    )
    return train, other


def add_source_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prefix: str,
    rating_col: str,
    count_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    for frame in (train, test):
        frame[f"{prefix}_available"] = frame[rating_col].notna().astype(float)
        frame[f"{prefix}_rating_feature"] = pd.to_numeric(
            frame[rating_col], errors="coerce"
        )
        frame[f"{prefix}_log_count_feature"] = np.log10(
            1 + pd.to_numeric(frame[count_col], errors="coerce")
        )
    return train, test


def prepare_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    spec: SourceFeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train, test = add_author_history_features(train_df, test_df, target_col)
    numeric_features = list(BASE_NUMERIC_FEATURES)
    categorical_features = list(BASE_CATEGORICAL_FEATURES)

    if spec.include_goodreads:
        train, test = add_source_features(
            train,
            test,
            "goodreads",
            "goodreads_rating_verified",
            "goodreads_rating_count_verified",
        )
        numeric_features.extend(SOURCE_FEATURE_COLUMNS["goodreads"])
    if spec.include_openlibrary:
        train, test = add_source_features(
            train, test, "openlibrary", "ol_rating_consensus", "ol_reviews_consensus"
        )
        numeric_features.extend(SOURCE_FEATURE_COLUMNS["openlibrary"])
    if spec.include_amazon:
        train, test = add_source_features(
            train,
            test,
            "amazon",
            "amazon_rating_consensus",
            "amazon_reviews_consensus",
        )
        numeric_features.extend(SOURCE_FEATURE_COLUMNS["amazon"])

    for column in numeric_features:
        train[column] = pd.to_numeric(train[column], errors="coerce")
        median = float(train[column].median()) if train[column].notna().any() else 0.0
        train[column] = train[column].fillna(median)
        test[column] = pd.to_numeric(test[column], errors="coerce").fillna(median)

    for column in categorical_features:
        train[column] = train[column].fillna("Unknown")
        test[column] = test[column].fillna("Unknown")

    return train, test, numeric_features, categorical_features


def make_pipeline(
    model,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, PREDICTION_MIN, PREDICTION_MAX)


def linear_terms_in_raw_space(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    transformed_coefficients: np.ndarray,
    transformed_intercept: float,
) -> tuple[float, dict[str, float]]:
    raw_intercept = transformed_intercept
    raw_coefficients: dict[str, float] = {}

    feature_names = preprocessor.get_feature_names_out().tolist()
    numeric_scaler: StandardScaler = preprocessor.named_transformers_["num"]
    numeric_means = dict(zip(numeric_features, numeric_scaler.mean_, strict=False))
    numeric_scales = dict(zip(numeric_features, numeric_scaler.scale_, strict=False))

    for feature_name, coefficient in zip(
        feature_names, transformed_coefficients, strict=False
    ):
        if feature_name.startswith("num__"):
            raw_feature_name = feature_name.removeprefix("num__")
            scale = float(numeric_scales[raw_feature_name])
            mean = float(numeric_means[raw_feature_name])
            if scale == 0:
                raw_coefficient = 0.0
            else:
                raw_coefficient = float(coefficient) / scale
                raw_intercept -= raw_coefficient * mean
        else:
            raw_feature_name = feature_name.removeprefix("cat__")
            raw_coefficient = float(coefficient)
        raw_coefficients[raw_feature_name] = raw_coefficient

    return raw_intercept, raw_coefficients


def fit_pipeline_for_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    spec: SourceFeatureSpec,
    model_name: str,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    builder = MODEL_BUILDERS[model_name]
    model = builder()
    train_frame, test_frame, numeric_features, categorical_features = (
        prepare_feature_frames(train_df, test_df, target_col, spec)
    )
    pipeline = make_pipeline(model, numeric_features, categorical_features)
    y_train = pd.to_numeric(train_frame[target_col], errors="coerce").to_numpy()
    pipeline.fit(train_frame, y_train)
    return pipeline, train_frame, test_frame, numeric_features, categorical_features


def baseline_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    model_name: str,
) -> np.ndarray:
    target = pd.to_numeric(train_df[target_col], errors="coerce")
    global_mean = float(target.mean())
    if model_name == "Global mean":
        return np.full(len(test_df), global_mean)
    grouped = train_df.groupby("Bookshelf")[target_col].mean()
    predictions = test_df["Bookshelf"].map(grouped).fillna(global_mean).to_numpy()
    return clip_predictions(predictions)


def model_predictions_for_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    spec: SourceFeatureSpec,
    model_name: str,
) -> np.ndarray:
    if model_name in {"Category mean", "Global mean"}:
        return baseline_predictions(train_df, test_df, target_col, model_name)
    pipeline, _, test_frame, _, _ = fit_pipeline_for_split(
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        spec=spec,
        model_name=model_name,
    )
    return clip_predictions(pipeline.predict(test_frame))


def utility_series(values: pd.Series, target_col: str) -> pd.Series:
    base = UTILITY_BASES[target_col]
    numeric = pd.to_numeric(values, errors="coerce")
    exponent = np.clip(numeric - 1, a_min=0, a_max=None)
    return pd.Series(np.power(base, exponent) - 1, index=numeric.index)


def build_prediction_rows(
    frame: pd.DataFrame,
    target_col: str,
    spec: SourceFeatureSpec,
    model_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    train_df = frame[
        (~frame["source"].isin(TRAIN_SOURCE_EXCLUDE)) & frame[target_col].notna()
    ].copy()
    holdout_df = frame[
        frame["source"].eq("Holdout 2026") & frame[target_col].notna()
    ].copy()
    if len(train_df) < MIN_TRAIN_ROWS or holdout_df.empty:
        return {}, pd.DataFrame()

    predictions = model_predictions_for_split(
        train_df=train_df,
        test_df=holdout_df,
        target_col=target_col,
        spec=spec,
        model_name=model_name,
    )
    actual = pd.to_numeric(holdout_df[target_col], errors="coerce")
    predicted = pd.Series(predictions, index=holdout_df.index, dtype=float)
    rho, p_value = spearman_summary(predicted, actual)

    detail = holdout_df[
        [
            "title",
            "author",
            "category",
            "estimated_finish",
            target_col,
            "goodreads_rating_verified",
            "ol_rating_consensus",
            "amazon_rating_consensus",
        ]
    ].copy()
    detail["target"] = target_col
    detail["feature_spec"] = spec.name
    detail["model"] = model_name
    detail["prediction"] = predicted.to_numpy()
    detail["prediction_error"] = predicted.to_numpy() - actual.to_numpy()

    return (
        {
            "target": target_col,
            "target_label": TARGET_LABELS[target_col],
            "feature_spec": spec.name,
            "model": model_name,
            "n": int(len(holdout_df)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
            "spearman_rho": rho,
            "spearman_p": p_value,
            "actual_mean": float(actual.mean()),
            "prediction_mean": float(predicted.mean()),
        },
        detail,
    )


def build_holdout_split(frame: pd.DataFrame) -> pd.DataFrame:
    holdout = frame[frame["source"].eq("Holdout 2026")].copy()
    holdout["estimated_finish"] = pd.to_datetime(
        holdout["estimated_finish"], format="mixed", errors="coerce"
    )
    holdout = holdout.sort_values(["estimated_finish", "title"]).reset_index(drop=True)
    midpoint = len(holdout) // 2
    holdout["split"] = np.where(holdout.index < midpoint, "validation", "test")
    return holdout


def build_holdout_prediction_table(
    frame: pd.DataFrame, model_results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = build_holdout_split(frame)
    long_rows: list[pd.DataFrame] = []
    wide = holdout[
        [
            "title",
            "author",
            "category",
            "estimated_finish",
            "split",
            "avg_enjoyment",
            "avg_usefulness",
            "goodreads_rating_verified",
            "ol_rating_consensus",
            "amazon_rating_consensus",
        ]
    ].copy()

    for result in model_results.itertuples(index=False):
        holdout_valid = holdout[holdout[result.target].notna()].copy()
        if holdout_valid.empty:
            continue
        predictions = model_predictions_for_split(
            train_df=frame[
                (~frame["source"].isin(TRAIN_SOURCE_EXCLUDE))
                & frame[result.target].notna()
            ].copy(),
            test_df=holdout_valid,
            target_col=result.target,
            spec=next(
                spec for spec in FEATURE_SPECS if spec.name == result.feature_spec
            ),
            model_name=result.model,
        )
        detail = holdout_valid[
            [
                "title",
                "author",
                "category",
                "estimated_finish",
                "split",
                result.target,
            ]
        ].copy()
        detail["target"] = result.target
        detail["feature_spec"] = result.feature_spec
        detail["model"] = result.model
        detail["prediction"] = predictions
        detail["prediction_error"] = (
            predictions
            - pd.to_numeric(holdout_valid[result.target], errors="coerce").to_numpy()
        )
        long_rows.append(detail)

        column_name = (
            f"pred__{result.target}__{result.feature_spec}__"
            f"{result.model.lower().replace(' ', '_')}"
        )
        error_name = (
            f"err__{result.target}__{result.feature_spec}__"
            f"{result.model.lower().replace(' ', '_')}"
        )
        join_frame = detail[["title", "prediction", "prediction_error"]].rename(
            columns={"prediction": column_name, "prediction_error": error_name}
        )
        wide = wide.merge(join_frame, on="title", how="left")

    long_df = (
        pd.concat(long_rows, ignore_index=True)
        if long_rows
        else pd.DataFrame(
            columns=[
                "title",
                "author",
                "category",
                "estimated_finish",
                "split",
                "target",
                "feature_spec",
                "model",
                "prediction",
                "prediction_error",
            ]
        )
    )
    return long_df, wide


def feature_list_for_spec(spec: SourceFeatureSpec) -> list[str]:
    features = list(BASE_NUMERIC_FEATURES) + list(BASE_CATEGORICAL_FEATURES)
    if spec.include_goodreads:
        features.extend(SOURCE_FEATURE_COLUMNS["goodreads"])
    if spec.include_openlibrary:
        features.extend(SOURCE_FEATURE_COLUMNS["openlibrary"])
    if spec.include_amazon:
        features.extend(SOURCE_FEATURE_COLUMNS["amazon"])
    return features


def export_linear_model_coefficients(
    frame: pd.DataFrame, model_results: pd.DataFrame, policy_results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []

    linear_policies = policy_results[policy_results["model"].isin(LINEAR_MODEL_NAMES)]
    for target_col in TARGETS:
        for selection in ("best_balanced_utility", "best_overall_utility"):
            candidates = linear_policies[
                (linear_policies["target"] == target_col)
                & (linear_policies["selection"] == selection)
            ].copy()
            if candidates.empty:
                continue
            chosen_policy = candidates.sort_values(
                "test_utility_uplift", ascending=False
            ).iloc[0]
            spec = next(
                spec
                for spec in FEATURE_SPECS
                if spec.name == chosen_policy["feature_spec"]
            )
            train_df = frame[
                (~frame["source"].isin(TRAIN_SOURCE_EXCLUDE))
                & frame[target_col].notna()
            ].copy()
            holdout_df = frame[
                frame["source"].eq("Holdout 2026") & frame[target_col].notna()
            ].copy()
            if holdout_df.empty:
                continue
            pipeline, _, _, numeric_features, _ = fit_pipeline_for_split(
                train_df=train_df,
                test_df=holdout_df,
                target_col=target_col,
                spec=spec,
                model_name=str(chosen_policy["model"]),
            )
            preprocessor = pipeline.named_steps["prep"]
            model = pipeline.named_steps["model"]
            feature_names = preprocessor.get_feature_names_out().tolist()
            coefficients = getattr(model, "coef_", None)
            intercept = float(getattr(model, "intercept_", 0.0))
            raw_intercept, raw_coefficients = linear_terms_in_raw_space(
                preprocessor=preprocessor,
                numeric_features=numeric_features,
                transformed_coefficients=np.asarray(coefficients, dtype=float),
                transformed_intercept=intercept,
            )
            for feature_name, coefficient in zip(
                feature_names, coefficients, strict=False
            ):
                raw_feature_name = feature_name.removeprefix("num__").removeprefix(
                    "cat__"
                )
                coefficient_rows.append(
                    {
                        "target": target_col,
                        "target_label": TARGET_LABELS[target_col],
                        "feature_spec": chosen_policy["feature_spec"],
                        "model": chosen_policy["model"],
                        "selection": selection,
                        "threshold": float(chosen_policy["threshold"]),
                        "intercept": intercept,
                        "raw_intercept": raw_intercept,
                        "feature_name": feature_name,
                        "raw_feature_name": raw_feature_name,
                        "coefficient": float(coefficient),
                        "raw_coefficient": float(raw_coefficients[raw_feature_name]),
                    }
                )

            rule_rows.append(
                {
                    "target": target_col,
                    "target_label": TARGET_LABELS[target_col],
                    "feature_spec": chosen_policy["feature_spec"],
                    "model": chosen_policy["model"],
                    "selection": selection,
                    "threshold": float(chosen_policy["threshold"]),
                    "intercept": intercept,
                    "raw_intercept": raw_intercept,
                    "feature_count": len(feature_names),
                    "features_used": ", ".join(feature_list_for_spec(spec)),
                }
            )

    coefficients_df = pd.DataFrame(coefficient_rows)
    rules_df = pd.DataFrame(rule_rows)
    return coefficients_df, rules_df


def plot_ratings_vs_targets(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_specs = [
        (
            "goodreads_rating_verified",
            None,
            "Goodreads",
            None,
        ),
        (
            "ol_rating_consensus",
            "ol_rating_observed_variants",
            "Open Library",
            "imputed",
        ),
        (
            "amazon_rating_consensus",
            "amazon_rating_observed_variants",
            "Amazon",
            "imputed",
        ),
    ]
    targets = [
        ("avg_enjoyment", "Average enjoyment"),
        ("avg_usefulness", "Average usefulness"),
    ]

    for row_idx, (target_col, target_label) in enumerate(targets):
        for col_idx, (rating_col, observed_col, title, show_imputed) in enumerate(
            plot_specs
        ):
            axis = axes[row_idx, col_idx]
            x = pd.to_numeric(df[rating_col], errors="coerce")
            y = pd.to_numeric(df[target_col], errors="coerce")
            mask = x.notna() & y.notna()
            if observed_col is None:
                axis.scatter(x[mask], y[mask], s=24, alpha=0.65, color="#0B3954")
            else:
                observed_mask = mask & pd.to_numeric(
                    df[observed_col], errors="coerce"
                ).fillna(0).gt(0)
                imputed_mask = mask & ~observed_mask
                axis.scatter(
                    x[observed_mask],
                    y[observed_mask],
                    s=24,
                    alpha=0.65,
                    color="#0B3954",
                    label="observed",
                )
                axis.scatter(
                    x[imputed_mask],
                    y[imputed_mask],
                    s=24,
                    alpha=0.45,
                    color="#C81D25",
                    marker="x",
                    label=show_imputed,
                )
                axis.legend(frameon=False, fontsize=8)
            rho, _ = spearman_summary(x[mask], y[mask])
            axis.set_title(
                f"{title} vs {target_label}\nrho={rho:.3f}, n={int(mask.sum())}"
            )
            axis.set_xlabel(f"{title} rating")
            axis.set_ylabel(target_label)
            axis.grid(alpha=0.15)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_counts_vs_targets(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_specs = [
        ("goodreads_rating_count_verified", "Goodreads ratings", None),
        ("ol_reviews_consensus", "Open Library reviews", "ol_rating_observed_variants"),
        (
            "amazon_reviews_consensus",
            "Amazon reviews",
            "amazon_rating_observed_variants",
        ),
    ]
    targets = [
        ("avg_enjoyment", "Average enjoyment"),
        ("avg_usefulness", "Average usefulness"),
    ]

    for row_idx, (target_col, target_label) in enumerate(targets):
        for col_idx, (count_col, title, observed_col) in enumerate(plot_specs):
            axis = axes[row_idx, col_idx]
            x = np.log10(1 + pd.to_numeric(df[count_col], errors="coerce"))
            y = pd.to_numeric(df[target_col], errors="coerce")
            mask = x.notna() & y.notna()
            if observed_col is None:
                axis.scatter(x[mask], y[mask], s=24, alpha=0.65, color="#087E8B")
            else:
                observed_mask = mask & pd.to_numeric(
                    df[observed_col], errors="coerce"
                ).fillna(0).gt(0)
                imputed_mask = mask & ~observed_mask
                axis.scatter(
                    x[observed_mask],
                    y[observed_mask],
                    s=24,
                    alpha=0.65,
                    color="#087E8B",
                    label="observed",
                )
                axis.scatter(
                    x[imputed_mask],
                    y[imputed_mask],
                    s=24,
                    alpha=0.45,
                    color="#FF5A5F",
                    marker="x",
                    label="imputed",
                )
                axis.legend(frameon=False, fontsize=8)
            rho, _ = spearman_summary(x[mask], y[mask])
            axis.set_title(
                f"{title} vs {target_label}\nrho={rho:.3f}, n={int(mask.sum())}"
            )
            axis.set_xlabel(f"log10(1 + {title.lower()})")
            axis.set_ylabel(target_label)
            axis.grid(alpha=0.15)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_holdout_predictions(
    predictions_long: pd.DataFrame, best_models: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()
    if predictions_long.empty or best_models.empty:
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    plotted = 0
    for _, row in best_models.iterrows():
        subset = predictions_long[
            (predictions_long["target"] == row["target"])
            & (predictions_long["feature_spec"] == row["feature_spec"])
            & (predictions_long["model"] == row["model"])
        ].copy()
        if subset.empty or plotted >= len(axes):
            continue
        axis = axes[plotted]
        actual = pd.to_numeric(subset[row["target"]], errors="coerce")
        predicted = pd.to_numeric(subset["prediction"], errors="coerce")
        axis.scatter(actual, predicted, s=28, alpha=0.7, color="#0B3954")
        axis.plot([1, 5], [1, 5], linestyle="--", color="black", linewidth=1)
        axis.set_xlim(1, 5)
        axis.set_ylim(1, 5)
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.set_title(
            f"{row['target_label']} | {row['selection']}\n"
            f"{row['feature_spec']} + {row['model']}"
        )
        axis.grid(alpha=0.15)
        plotted += 1

    for axis in axes[plotted:]:
        axis.axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_linear_rules_markdown(
    rules_df: pd.DataFrame, coefficients_df: pd.DataFrame
) -> str:
    lines = ["# Linear Decision Rules", ""]
    if rules_df.empty:
        lines.append("No linear rules were exported.")
        return "\n".join(lines) + "\n"

    for _, rule in rules_df.iterrows():
        lines.append(
            f"## {rule['target_label']} | {rule['feature_spec']} + {rule['model']}"
        )
        lines.append(f"- Selection: {rule['selection']}")
        lines.append(f"- Read if predicted score >= {rule['threshold']:.3f}")
        lines.append(f"- Raw-space intercept: {rule['raw_intercept']:.6f}")
        lines.append(f"- Raw feature groups: {rule['features_used']}")
        lines.append("- Raw-space non-zero coefficients:")
        subset = coefficients_df[
            (coefficients_df["target"] == rule["target"])
            & (coefficients_df["feature_spec"] == rule["feature_spec"])
            & (coefficients_df["model"] == rule["model"])
            & (coefficients_df["selection"] == rule["selection"])
        ].copy()
        subset = subset[subset["raw_coefficient"].abs() > 1e-9].sort_values(
            "raw_coefficient", key=lambda series: series.abs(), ascending=False
        )
        for _, coefficient_row in subset.iterrows():
            lines.append(
                f"  - {coefficient_row['raw_feature_name']}: "
                f"{coefficient_row['raw_coefficient']:.6f}"
            )
        lines.append("")
    return "\n".join(lines)


def select_threshold_rows(predictions: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        validation = predictions[predictions["split"] == "validation"].copy()
        keep = validation[validation["prediction"] >= threshold]
        skip = validation[validation["prediction"] < threshold]
        if len(keep) < POLICY_MIN_SIDE_N or len(skip) < POLICY_MIN_SIDE_N:
            continue
        keep_share = len(keep) / len(validation)
        utility = utility_series(validation[target_col], target_col)
        keep_utility = utility_series(keep[target_col], target_col)
        rows.append(
            {
                "threshold": threshold,
                "n_validation": len(validation),
                "n_keep_validation": len(keep),
                "keep_share_validation": keep_share,
                "validation_uplift": float(
                    pd.to_numeric(keep[target_col], errors="coerce").mean()
                    - pd.to_numeric(validation[target_col], errors="coerce").mean()
                ),
                "validation_utility_uplift": float(
                    keep_utility.mean() - utility.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def apply_threshold(
    predictions: pd.DataFrame, target_col: str, threshold: float
) -> dict[str, object]:
    test = predictions[predictions["split"] == "test"].copy()
    keep = test[test["prediction"] >= threshold]
    all_mean = float(pd.to_numeric(test[target_col], errors="coerce").mean())
    all_utility = float(utility_series(test[target_col], target_col).mean())
    if keep.empty:
        return {
            "test_keep_n": 0,
            "test_keep_share": 0.0,
            "test_uplift": math.nan,
            "test_utility_uplift": math.nan,
        }
    keep_mean = float(pd.to_numeric(keep[target_col], errors="coerce").mean())
    keep_utility = float(utility_series(keep[target_col], target_col).mean())
    return {
        "test_keep_n": int(len(keep)),
        "test_keep_share": len(keep) / len(test),
        "test_uplift": keep_mean - all_mean,
        "test_utility_uplift": keep_utility - all_utility,
    }


def evaluate_policies(frame: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    holdout = frame[frame["source"].eq("Holdout 2026")].copy()
    holdout["estimated_finish"] = pd.to_datetime(
        holdout["estimated_finish"], format="mixed", errors="coerce"
    )
    holdout = holdout.sort_values(["estimated_finish", "title"]).reset_index(drop=True)
    midpoint = len(holdout) // 2
    holdout["split"] = np.where(holdout.index < midpoint, "validation", "test")

    rows: list[dict[str, object]] = []
    for result in results.itertuples(index=False):
        holdout_valid = holdout[holdout[result.target].notna()].copy()
        if holdout_valid.empty:
            continue
        predictions = model_predictions_for_split(
            train_df=frame[
                (~frame["source"].isin(TRAIN_SOURCE_EXCLUDE))
                & frame[result.target].notna()
            ].copy(),
            test_df=holdout_valid,
            target_col=result.target,
            spec=next(
                spec for spec in FEATURE_SPECS if spec.name == result.feature_spec
            ),
            model_name=result.model,
        )
        prediction_frame = holdout_valid[["title", result.target, "split"]].copy()
        prediction_frame["prediction"] = predictions
        sweep = select_threshold_rows(prediction_frame, result.target)
        if sweep.empty:
            continue
        balanced = sweep[
            sweep["keep_share_validation"].between(
                BALANCED_KEEP_SHARE_RANGE[0], BALANCED_KEEP_SHARE_RANGE[1]
            )
        ].copy()
        for selection_name, selection_frame in [
            ("best_overall_utility", sweep),
            ("best_balanced_utility", balanced),
        ]:
            if selection_frame.empty:
                continue
            best = selection_frame.sort_values(
                "validation_utility_uplift", ascending=False
            ).iloc[0]
            realized = apply_threshold(
                prediction_frame, result.target, float(best["threshold"])
            )
            rows.append(
                {
                    "target": result.target,
                    "target_label": TARGET_LABELS[result.target],
                    "feature_spec": result.feature_spec,
                    "model": result.model,
                    "selection": selection_name,
                    **best.to_dict(),
                    **realized,
                }
            )
    return pd.DataFrame(rows)


def build_report(
    merged: pd.DataFrame,
    suspicious: pd.DataFrame,
    structural_review: pd.DataFrame,
    source_summary: pd.DataFrame,
    best_models: pd.DataFrame,
    best_policies: pd.DataFrame,
) -> str:
    lines = ["# Multi-Source Analysis", ""]
    lines.append("## Coverage")
    lines.append(f"- Rows in merged frame: {len(merged)}")
    lines.append(
        f"- Rows matched back to external-source master: {int(merged['external_master_match_found'].sum())}"
    )
    lines.append(
        f"- Rows with Open Library observed rating: {int(pd.to_numeric(merged['ol_rating_observed_variants'], errors='coerce').fillna(0).gt(0).sum())}"
    )
    lines.append(
        f"- Rows with Open Library consensus/imputed rating: {int(pd.to_numeric(merged['ol_rating_consensus'], errors='coerce').notna().sum())}"
    )
    lines.append(
        f"- Rows with Amazon observed rating: {int(pd.to_numeric(merged['amazon_rating_observed_variants'], errors='coerce').fillna(0).gt(0).sum())}"
    )
    lines.append(
        f"- Rows with Amazon consensus/imputed rating: {int(pd.to_numeric(merged['amazon_rating_consensus'], errors='coerce').notna().sum())}"
    )
    lines.append("")
    lines.append("## Suspicious Source Rows")
    lines.append(f"- Total suspicious rows: {len(suspicious)}")
    lines.append(
        f"- Structural rows worth manual cleanup first: {len(structural_review)}"
    )
    if not suspicious.empty:
        for _, row in suspicious.head(12).iterrows():
            lines.append(f"- `{row['title']}`: {row['issues']}")

    lines.append("")
    lines.append("## Source Summary")
    for source in source_summary["source"].unique():
        subset = source_summary[source_summary["source"] == source]
        rows = {
            metric: value
            for metric, value in zip(subset["metric"], subset["value"], strict=False)
        }
        lines.append(
            f"- `{source}`: rows_with_rating={rows.get('rows_with_rating', math.nan):.0f}, "
            f"suspicious_rows={rows.get('suspicious_rows', math.nan):.0f}, "
            f"rho_vs_goodreads={rows.get('spearman_vs_goodreads_rating', math.nan):.3f}, "
            f"rho_vs_avg_enjoyment={rows.get('spearman_vs_avg_enjoyment', math.nan):.3f}, "
            f"rho_vs_avg_usefulness={rows.get('spearman_vs_avg_usefulness', math.nan):.3f}"
        )

    lines.append("")
    lines.append("## Best Holdout Models")
    for _, row in best_models.iterrows():
        lines.append(
            f"- `{row['target_label']}` | {row['selection']}: "
            f"{row['feature_spec']} + {row['model']} "
            f"(MAE={row['mae']:.3f}, rho={row['spearman_rho']:.3f})"
        )

    lines.append("")
    lines.append("## Best Decision Rules")
    for _, row in best_policies.iterrows():
        lines.append(
            f"- `{row['target_label']}` | {row['selection']}: "
            f"{row['feature_spec']} + {row['model']} @ {row['threshold']:.1f} "
            f"(validation utility uplift={row['validation_utility_uplift']:.3f}, "
            f"test utility uplift={row['test_utility_uplift']:.3f})"
        )
    lines.append("")
    lines.append("## Buckley")
    buckley = merged[
        merged["title"].astype(str).str.contains("Buckley", case=False, na=False)
    ]
    if not buckley.empty:
        row = buckley.iloc[0]
        openlibrary_text = (
            f"OpenLibrary observed={row['ol_rating_consensus']}"
            if row["ol_rating_observed_variants"] > 0
            else f"OpenLibrary missing in source sheet, imputed={row['ol_rating_consensus']:.2f}"
        )
        lines.append(
            f"- `Buckley` is no longer a Goodreads mismatch problem: "
            f"Goodreads={row['goodreads_rating_verified']}, "
            f"Amazon={row['amazon_rating_consensus']}, "
            f"{openlibrary_text}."
        )
        lines.append(
            "- The earlier mismatch came from the old short-title Goodreads scrape, not from the cleaned master."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    golden = build_golden_master()
    master = build_master_source_frame()
    merged = merge_master_sources(golden, master)
    merged = prepare_external_features(merged)
    merged = add_source_consensus(merged)
    merged = prepare_model_frame(merged)
    merged.to_csv(MERGED_CSV, index=False)

    suspicious = build_suspicious_rows(merged)
    suspicious.to_csv(SUSPICIOUS_CSV, index=False)
    structural_review = suspicious[
        suspicious["issues"].str.contains("bad_text|variant_", regex=True, na=False)
    ].copy()
    structural_review.to_csv(STRUCTURAL_REVIEW_CSV, index=False)

    source_summary = summarize_source_quality(merged, suspicious)
    source_summary.to_csv(SOURCE_SUMMARY_CSV, index=False)

    model_rows: list[dict[str, object]] = []
    prediction_details: list[pd.DataFrame] = []
    for target_col in TARGETS:
        for spec in FEATURE_SPECS:
            for model_name in MODEL_BUILDERS:
                if (
                    model_name in {"Category mean", "Global mean"}
                    and spec.name != "preread_base"
                ):
                    continue
                result, detail = build_prediction_rows(
                    merged, target_col, spec, model_name
                )
                if result:
                    model_rows.append(result)
                    prediction_details.append(detail)
    model_results = pd.DataFrame(model_rows).sort_values(
        ["target", "mae", "feature_spec", "model"]
    )
    model_results.to_csv(MODEL_RESULTS_CSV, index=False)

    best_rows: list[dict[str, object]] = []
    for target_col in TARGETS:
        subset = model_results[model_results["target"] == target_col].copy()
        if subset.empty:
            continue
        best_rows.append(
            {
                "target": target_col,
                "target_label": TARGET_LABELS[target_col],
                "selection": "best_mae",
                **subset.sort_values("mae").iloc[0].to_dict(),
            }
        )
        best_rows.append(
            {
                "target": target_col,
                "target_label": TARGET_LABELS[target_col],
                "selection": "best_rank_correlation",
                **subset.sort_values("spearman_rho", ascending=False).iloc[0].to_dict(),
            }
        )
    best_models = pd.DataFrame(best_rows)
    best_models.to_csv(MODEL_BEST_CSV, index=False)

    policy_results = evaluate_policies(merged, model_results)
    policy_results.to_csv(POLICY_RESULTS_CSV, index=False)
    best_policies = (
        policy_results.sort_values(
            ["target", "selection", "test_utility_uplift"],
            ascending=[True, True, False],
        )
        .groupby(["target", "selection"], as_index=False)
        .head(1)
    )
    best_policies.to_csv(POLICY_BEST_CSV, index=False)

    predictions_long = (
        pd.concat(prediction_details, ignore_index=True)
        if prediction_details
        else pd.DataFrame()
    )
    predictions_long.to_csv(HOLDOUT_PREDICTIONS_LONG_CSV, index=False)
    _, predictions_wide = build_holdout_prediction_table(merged, model_results)
    predictions_wide.to_csv(HOLDOUT_PREDICTIONS_WIDE_CSV, index=False)

    coefficients_df, rules_df = export_linear_model_coefficients(
        merged, model_results, policy_results
    )
    coefficients_df.to_csv(LINEAR_COEFFICIENTS_CSV, index=False)
    rules_df.to_csv(LINEAR_RULES_CSV, index=False)
    LINEAR_RULES_MD.write_text(
        build_linear_rules_markdown(rules_df, coefficients_df),
        encoding="utf-8",
    )

    plot_ratings_vs_targets(merged, RATINGS_PLOT)
    plot_counts_vs_targets(merged, COUNTS_PLOT)
    plot_holdout_predictions(predictions_long, best_models, HOLDOUT_PLOT)

    REPORT_MD.write_text(
        build_report(
            merged,
            suspicious,
            structural_review,
            source_summary,
            best_models,
            best_policies,
        ),
        encoding="utf-8",
    )

    print(f"Saved merged frame to {MERGED_CSV}")
    print(f"Saved suspicious rows to {SUSPICIOUS_CSV}")
    print(f"Saved structural review rows to {STRUCTURAL_REVIEW_CSV}")
    print(f"Saved source summary to {SOURCE_SUMMARY_CSV}")
    print(f"Saved model results to {MODEL_RESULTS_CSV}")
    print(f"Saved policy results to {POLICY_RESULTS_CSV}")
    print(f"Saved holdout predictions to {HOLDOUT_PREDICTIONS_WIDE_CSV}")
    print(f"Saved linear coefficients to {LINEAR_COEFFICIENTS_CSV}")
    print(f"Saved report to {REPORT_MD}")


if __name__ == "__main__":
    main()
