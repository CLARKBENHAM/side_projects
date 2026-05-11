"""Explore simple external-rating decision rules with missing-data sensitivity."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_books_tracking.build_golden_master import build as build_golden_master
from ai_books_tracking.goodreads_followup_analysis import spearman_summary
from ai_books_tracking.multi_source_cleaning_analysis import (
    TRAIN_SOURCE_EXCLUDE,
    add_source_consensus,
    build_master_source_frame,
    merge_master_sources,
    prepare_external_features,
    prepare_model_frame,
    utility_series,
)

OUTPUT_DIR = Path(__file__).parent
MISSINGNESS_SUMMARY_CSV = OUTPUT_DIR / "multi_source_rule_missingness_summary.csv"
MODEL_SUMMARY_CSV = OUTPUT_DIR / "multi_source_rule_model_summary.csv"
DROP_CURVES_CSV = OUTPUT_DIR / "multi_source_rule_drop_curves.csv"
DROP_CURVES_RATING_PLOT = OUTPUT_DIR / "multi_source_rule_drop_curves_ratings.png"
DROP_CURVES_UTILITY_PLOT = OUTPUT_DIR / "multi_source_rule_drop_curves_utility.png"
SITE_BUCKET_CORR_CSV = OUTPUT_DIR / "multi_source_rule_site_correlations_by_bucket.csv"
SITE_BUCKET_CORR_PLOT = OUTPUT_DIR / "multi_source_rule_site_correlations_by_bucket.png"
HOLDOUT_SCORES_CSV = OUTPUT_DIR / "multi_source_rule_holdout_scores.csv"
GROUPED_COEFFICIENTS_CSV = OUTPUT_DIR / "multi_source_rule_grouped_coefficients.csv"

SITE_RATING_COLUMNS = {
    "goodreads": "goodreads_rating_verified",
    "openlibrary": "ol_rating_consensus",
    "amazon": "amazon_rating_consensus",
}
SITE_COUNT_COLUMNS = {
    "goodreads": "goodreads_rating_count_verified",
    "openlibrary": "ol_reviews_consensus",
    "amazon": "amazon_reviews_consensus",
}
TARGETS = ("avg_enjoyment", "avg_usefulness")
TARGET_LABELS = {
    "avg_enjoyment": "Average enjoyment",
    "avg_usefulness": "Average usefulness",
}
DROP_FRACTIONS = [round(value, 2) for value in np.arange(0.0, 0.81, 0.05)]
MIN_BUCKET_TRAIN_ROWS = 12
MIN_CURVE_POINTS = 5
PLOT_RULES = [
    "goodreads_rating",
    "amazon_rating",
    "openlibrary_rating",
    "mean_site_rating",
    "pooled_ridge_ratings_median",
    "bucket_ridge_ratings_complete",
]


def category_bucket(category: object) -> str:
    text = "" if pd.isna(category) else str(category).strip()
    if text in {"Computer Science", "Math", "Machine Learning", "Stats"}:
        return "Technical"
    if text == "Literature":
        return "Literature"
    if text == "fiction":
        return "Fiction"
    return "General/Business"


def load_frame() -> pd.DataFrame:
    golden = build_golden_master()
    master = build_master_source_frame()
    merged = merge_master_sources(golden, master)
    merged = prepare_external_features(merged)
    merged = add_source_consensus(merged)
    merged = prepare_model_frame(merged)
    merged["category_bucket"] = merged["category"].map(category_bucket)
    for site, rating_col in SITE_RATING_COLUMNS.items():
        merged[f"{site}_log_count"] = np.log10(
            1 + pd.to_numeric(merged[SITE_COUNT_COLUMNS[site]], errors="coerce")
        )
    merged["mean_site_rating"] = (
        merged[list(SITE_RATING_COLUMNS.values())]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )
    merged["mean_site_log_count"] = (
        merged[[f"{site}_log_count" for site in SITE_COUNT_COLUMNS]]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
    )
    return merged


def external_feature_columns(include_counts: bool) -> list[str]:
    columns = list(SITE_RATING_COLUMNS.values())
    if include_counts:
        columns.extend(f"{site}_log_count" for site in SITE_COUNT_COLUMNS)
    return columns


def prepare_rule_frames(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_columns: list[str],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train_df.copy()
    holdout = holdout_df.copy()
    feature_cols = list(feature_columns)

    if mode == "median_with_flags":
        availability_columns = []
        for column in feature_columns:
            availability = f"{column}__available"
            train[availability] = train[column].notna().astype(float)
            holdout[availability] = holdout[column].notna().astype(float)
            availability_columns.append(availability)
            train[column] = pd.to_numeric(train[column], errors="coerce")
            holdout[column] = pd.to_numeric(holdout[column], errors="coerce")
            median = (
                float(train[column].median()) if train[column].notna().any() else 0.0
            )
            train[column] = train[column].fillna(median)
            holdout[column] = holdout[column].fillna(median)
        feature_cols.extend(availability_columns)
        return train, holdout, feature_cols

    if mode != "complete_case":
        raise ValueError(f"Unsupported mode: {mode}")

    train_numeric = train[feature_columns].apply(pd.to_numeric, errors="coerce")
    holdout_numeric = holdout[feature_columns].apply(pd.to_numeric, errors="coerce")
    train.loc[:, feature_columns] = train_numeric
    holdout.loc[:, feature_columns] = holdout_numeric
    train_mask = train_numeric.notna().all(axis=1)
    holdout_mask = holdout_numeric.notna().all(axis=1)
    return train.loc[train_mask].copy(), holdout.loc[holdout_mask].copy(), feature_cols


def fit_ridge_scores(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    mode: str,
) -> tuple[pd.DataFrame, Pipeline | None, list[str]]:
    train, holdout, feature_cols = prepare_rule_frames(
        train_df, holdout_df, feature_columns, mode
    )
    if train.empty or holdout.empty:
        return holdout.assign(score=np.nan), None, feature_cols

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=3.0)),
        ]
    )
    model.fit(train[feature_cols], pd.to_numeric(train[target_col], errors="coerce"))
    scored_holdout = holdout.copy()
    scored_holdout["score"] = model.predict(holdout[feature_cols])
    return scored_holdout, model, feature_cols


def fit_bucketed_ridge_scores(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
    mode: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pooled_holdout, pooled_model, pooled_features = fit_ridge_scores(
        train_df, holdout_df, target_col, feature_columns, mode
    )
    if pooled_holdout.empty or pooled_model is None:
        return pooled_holdout, []

    scored_parts: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    for bucket, bucket_holdout in pooled_holdout.groupby(
        "category_bucket", dropna=False
    ):
        bucket_train = train_df[train_df["category_bucket"] == bucket].copy()
        if len(bucket_train) < MIN_BUCKET_TRAIN_ROWS:
            bucket_scored = bucket_holdout.copy()
            bucket_scored["score"] = pooled_model.predict(
                bucket_holdout[pooled_features]
            )
            scored_parts.append(bucket_scored)
            continue

        bucket_scored, bucket_model, bucket_features = fit_ridge_scores(
            bucket_train, bucket_holdout, target_col, feature_columns, mode
        )
        if bucket_scored.empty:
            continue
        scored_parts.append(bucket_scored)
        scaler: StandardScaler = bucket_model.named_steps["scale"]
        ridge: Ridge = bucket_model.named_steps["ridge"]
        raw_coefficients = ridge.coef_ / scaler.scale_
        raw_intercept = float(
            ridge.intercept_ - np.sum(raw_coefficients * scaler.mean_)
        )
        for feature, coefficient in zip(
            bucket_features, raw_coefficients, strict=False
        ):
            coefficient_rows.append(
                {
                    "target": target_col,
                    "category_bucket": bucket,
                    "feature_set": (
                        "ratings_plus_counts"
                        if len(feature_columns) > len(SITE_RATING_COLUMNS)
                        else "ratings_only"
                    ),
                    "mode": mode,
                    "feature": feature,
                    "raw_coefficient": float(coefficient),
                    "raw_intercept": raw_intercept,
                }
            )
    if not scored_parts:
        return pooled_holdout.iloc[0:0].copy(), coefficient_rows
    return pd.concat(scored_parts, ignore_index=True), coefficient_rows


def score_site_rule(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = pd.to_numeric(scored[score_column], errors="coerce")
    return scored[scored["score"].notna()].copy()


def score_mean_site_rule(frame: pd.DataFrame) -> pd.DataFrame:
    return score_site_rule(frame, "mean_site_rating")


def evaluate_rule(
    scored_holdout: pd.DataFrame,
    target_col: str,
    rule_name: str,
    mode: str,
) -> dict[str, object]:
    valid = scored_holdout[
        pd.to_numeric(scored_holdout[target_col], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return {
            "target": target_col,
            "rule_name": rule_name,
            "mode": mode,
            "n_holdout": 0,
            "mae": math.nan,
            "spearman_rho": math.nan,
            "spearman_p": math.nan,
        }
    actual = pd.to_numeric(valid[target_col], errors="coerce")
    predicted = pd.to_numeric(valid["score"], errors="coerce")
    rho, pvalue = spearman_summary(actual, predicted)
    return {
        "target": target_col,
        "rule_name": rule_name,
        "mode": mode,
        "n_holdout": int(len(valid)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "spearman_rho": float(rho),
        "spearman_p": float(pvalue),
    }


def build_drop_curve(
    scored_holdout: pd.DataFrame,
    target_col: str,
    rule_name: str,
    mode: str,
) -> pd.DataFrame:
    valid = scored_holdout[
        pd.to_numeric(scored_holdout[target_col], errors="coerce").notna()
    ].copy()
    if len(valid) < MIN_CURVE_POINTS:
        return pd.DataFrame()
    valid = valid.sort_values(["score", "title"], ascending=[False, True]).reset_index(
        drop=True
    )
    utility = utility_series(valid[target_col], target_col)
    rows: list[dict[str, object]] = []
    for drop_fraction in DROP_FRACTIONS:
        drop_n = int(np.floor(len(valid) * drop_fraction))
        kept = valid.iloc[: len(valid) - drop_n].copy()
        kept_utility = utility.iloc[: len(valid) - drop_n]
        if len(kept) < MIN_CURVE_POINTS:
            continue
        rows.append(
            {
                "target": target_col,
                "rule_name": rule_name,
                "mode": mode,
                "drop_fraction": drop_fraction,
                "drop_percent": drop_fraction * 100,
                "keep_n": int(len(kept)),
                "keep_share": len(kept) / len(valid),
                "mean_kept_rating": float(
                    pd.to_numeric(kept[target_col], errors="coerce").mean()
                ),
                "mean_kept_utility": float(kept_utility.mean()),
                "baseline_rating": float(
                    pd.to_numeric(valid[target_col], errors="coerce").mean()
                ),
                "baseline_utility": float(utility.mean()),
                "rating_uplift": float(
                    pd.to_numeric(kept[target_col], errors="coerce").mean()
                    - pd.to_numeric(valid[target_col], errors="coerce").mean()
                ),
                "utility_uplift": float(kept_utility.mean() - utility.mean()),
            }
        )
    return pd.DataFrame(rows)


def site_bucket_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labeled = frame[frame["source"] != "Holdout 2026 (unverified)"].copy()
    for bucket in sorted(labeled["category_bucket"].dropna().unique()):
        bucket_frame = labeled[labeled["category_bucket"] == bucket].copy()
        for site, rating_col in SITE_RATING_COLUMNS.items():
            rating = pd.to_numeric(bucket_frame[rating_col], errors="coerce")
            for target in TARGETS:
                outcome = pd.to_numeric(bucket_frame[target], errors="coerce")
                mask = rating.notna() & outcome.notna()
                rho, pvalue = spearman_summary(rating[mask], outcome[mask])
                rows.append(
                    {
                        "category_bucket": bucket,
                        "site": site,
                        "target": target,
                        "n": int(mask.sum()),
                        "spearman_rho": float(rho),
                        "spearman_p": float(pvalue),
                    }
                )
    return pd.DataFrame(rows)


def plot_drop_curves(
    curves: pd.DataFrame, value_column: str, output_path: Path, title_suffix: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    color_map = {
        "goodreads_rating": "#0B3954",
        "amazon_rating": "#087E8B",
        "openlibrary_rating": "#C81D25",
        "mean_site_rating": "#FF5A5F",
        "pooled_ridge_ratings_median": "#6C5CE7",
        "bucket_ridge_ratings_complete": "#2D6A4F",
    }
    label_map = {
        "goodreads_rating": "Goodreads",
        "amazon_rating": "Amazon",
        "openlibrary_rating": "Open Library",
        "mean_site_rating": "Mean site rating",
        "pooled_ridge_ratings_median": "Pooled ridge + medians",
        "bucket_ridge_ratings_complete": "Bucket ridge + complete-case",
    }
    for axis, target in zip(axes, TARGETS, strict=False):
        subset = curves[
            (curves["target"] == target) & (curves["rule_name"].isin(PLOT_RULES))
        ].copy()
        for rule_name in PLOT_RULES:
            rule_subset = subset[subset["rule_name"] == rule_name].copy()
            if rule_subset.empty:
                continue
            axis.plot(
                rule_subset["drop_percent"],
                rule_subset[value_column],
                label=label_map[rule_name],
                color=color_map[rule_name],
                linewidth=2,
            )
        axis.set_title(f"{TARGET_LABELS[target]} | {title_suffix}")
        axis.set_xlabel("Percent of books dropped")
        axis.set_ylabel(title_suffix)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_site_bucket_correlations(
    correlations: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    site_colors = {
        "goodreads": "#0B3954",
        "amazon": "#087E8B",
        "openlibrary": "#C81D25",
    }
    for axis, target in zip(axes, TARGETS, strict=False):
        subset = correlations[correlations["target"] == target].copy()
        buckets = list(subset["category_bucket"].drop_duplicates())
        x = np.arange(len(buckets))
        width = 0.23
        for offset, site in enumerate(["goodreads", "amazon", "openlibrary"]):
            site_subset = subset[subset["site"] == site].set_index("category_bucket")
            values = [
                site_subset.reindex(buckets)["spearman_rho"].iloc[i]
                for i in range(len(buckets))
            ]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width=width,
                label=site.title(),
                color=site_colors[site],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(buckets, rotation=20, ha="right")
        axis.set_ylabel("Spearman rho")
        axis.set_title(f"{TARGET_LABELS[target]} by bucket")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run() -> None:
    frame = load_frame()
    train = frame[~frame["source"].isin(TRAIN_SOURCE_EXCLUDE)].copy()
    holdout = frame[frame["source"] == "Holdout 2026"].copy()

    missingness_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    holdout_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []

    site_rules = {
        "goodreads_rating": "goodreads_rating_verified",
        "amazon_rating": "amazon_rating_consensus",
        "openlibrary_rating": "ol_rating_consensus",
    }
    for target in TARGETS:
        for rule_name, score_column in site_rules.items():
            scored = score_site_rule(holdout, score_column)
            summary_rows.append(
                evaluate_rule(scored, target, rule_name, "observed_or_consensus")
            )
            curve = build_drop_curve(scored, target, rule_name, "observed_or_consensus")
            if not curve.empty:
                curve_frames.append(curve)
            if not scored.empty:
                scored_export = scored[
                    [
                        "title",
                        "author",
                        "category",
                        "category_bucket",
                        "estimated_finish",
                        target,
                        "score",
                    ]
                ].copy()
                scored_export["target"] = target
                scored_export["rule_name"] = rule_name
                scored_export["mode"] = "observed_or_consensus"
                holdout_frames.append(scored_export)

        mean_scored = score_mean_site_rule(holdout)
        summary_rows.append(
            evaluate_rule(
                mean_scored, target, "mean_site_rating", "observed_or_consensus"
            )
        )
        mean_curve = build_drop_curve(
            mean_scored, target, "mean_site_rating", "observed_or_consensus"
        )
        if not mean_curve.empty:
            curve_frames.append(mean_curve)
        if not mean_scored.empty:
            mean_export = mean_scored[
                [
                    "title",
                    "author",
                    "category",
                    "category_bucket",
                    "estimated_finish",
                    target,
                    "score",
                ]
            ].copy()
            mean_export["target"] = target
            mean_export["rule_name"] = "mean_site_rating"
            mean_export["mode"] = "observed_or_consensus"
            holdout_frames.append(mean_export)

        for include_counts, feature_set in [
            (False, "ratings_only"),
            (True, "ratings_plus_counts"),
        ]:
            feature_columns = external_feature_columns(include_counts)
            for mode in ["median_with_flags", "complete_case"]:
                prepared_train, prepared_holdout, feature_cols = prepare_rule_frames(
                    train, holdout, feature_columns, mode
                )
                missingness_rows.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "mode": mode,
                        "n_train": int(len(prepared_train)),
                        "n_holdout": int(len(prepared_holdout)),
                        "feature_count": len(feature_cols),
                    }
                )

                pooled_scored, _, _ = fit_ridge_scores(
                    train, holdout, target, feature_columns, mode
                )
                pooled_rule = f"pooled_ridge_{'ratings_counts' if include_counts else 'ratings'}_{'median' if mode == 'median_with_flags' else 'complete'}"
                summary_rows.append(
                    evaluate_rule(pooled_scored, target, pooled_rule, mode)
                )
                pooled_curve = build_drop_curve(
                    pooled_scored, target, pooled_rule, mode
                )
                if not pooled_curve.empty:
                    curve_frames.append(pooled_curve)
                if not pooled_scored.empty:
                    pooled_export = pooled_scored[
                        [
                            "title",
                            "author",
                            "category",
                            "category_bucket",
                            "estimated_finish",
                            target,
                            "score",
                        ]
                    ].copy()
                    pooled_export["target"] = target
                    pooled_export["rule_name"] = pooled_rule
                    pooled_export["mode"] = mode
                    holdout_frames.append(pooled_export)

                bucket_scored, bucket_coefficients = fit_bucketed_ridge_scores(
                    train, holdout, target, feature_columns, mode
                )
                coefficient_rows.extend(bucket_coefficients)
                bucket_rule = f"bucket_ridge_{'ratings_counts' if include_counts else 'ratings'}_{'median' if mode == 'median_with_flags' else 'complete'}"
                summary_rows.append(
                    evaluate_rule(bucket_scored, target, bucket_rule, mode)
                )
                bucket_curve = build_drop_curve(
                    bucket_scored, target, bucket_rule, mode
                )
                if not bucket_curve.empty:
                    curve_frames.append(bucket_curve)
                if not bucket_scored.empty:
                    bucket_export = bucket_scored[
                        [
                            "title",
                            "author",
                            "category",
                            "category_bucket",
                            "estimated_finish",
                            target,
                            "score",
                        ]
                    ].copy()
                    bucket_export["target"] = target
                    bucket_export["rule_name"] = bucket_rule
                    bucket_export["mode"] = mode
                    holdout_frames.append(bucket_export)

    missingness_df = pd.DataFrame(missingness_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["target", "spearman_rho", "mae"], ascending=[True, False, True]
    )
    curves_df = (
        pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    )
    holdout_scores_df = (
        pd.concat(holdout_frames, ignore_index=True)
        if holdout_frames
        else pd.DataFrame()
    )
    bucket_corr_df = site_bucket_correlations(frame)
    coefficients_df = pd.DataFrame(coefficient_rows)

    missingness_df.to_csv(MISSINGNESS_SUMMARY_CSV, index=False)
    summary_df.to_csv(MODEL_SUMMARY_CSV, index=False)
    curves_df.to_csv(DROP_CURVES_CSV, index=False)
    holdout_scores_df.to_csv(HOLDOUT_SCORES_CSV, index=False)
    bucket_corr_df.to_csv(SITE_BUCKET_CORR_CSV, index=False)
    coefficients_df.to_csv(GROUPED_COEFFICIENTS_CSV, index=False)

    plot_drop_curves(
        curves_df, "mean_kept_rating", DROP_CURVES_RATING_PLOT, "Mean kept rating"
    )
    plot_drop_curves(
        curves_df, "mean_kept_utility", DROP_CURVES_UTILITY_PLOT, "Mean kept utility"
    )
    plot_site_bucket_correlations(bucket_corr_df, SITE_BUCKET_CORR_PLOT)

    print(f"Saved missingness summary to {MISSINGNESS_SUMMARY_CSV}")
    print(f"Saved rule model summary to {MODEL_SUMMARY_CSV}")
    print(f"Saved drop curves to {DROP_CURVES_CSV}")
    print(f"Saved holdout scores to {HOLDOUT_SCORES_CSV}")
    print(f"Saved bucket correlations to {SITE_BUCKET_CORR_CSV}")
    print(f"Saved grouped coefficients to {GROUPED_COEFFICIENTS_CSV}")


if __name__ == "__main__":
    run()
