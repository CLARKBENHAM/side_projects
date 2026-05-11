"""Evaluate multi-source book selection policies on validation/test holdout splits.

Pipeline:
1. Combine the two Open Library and Amazon rating variants into single rating/counts.
2. Optionally shrink source ratings toward training bucket means using review counts.
3. Category-bucket z-score each source using training statistics.
4. Handle missing source ratings via complete-case dropping or ridge-based imputation.
5. Aggregate z-scores into a consensus signal.
6. Learn monotone mappings from consensus signal to your targets with isotonic regression.
7. Pick the best preprocessing variant on validation, then report policy curves on test.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

OUTPUT_DIR = Path(__file__).parent
INPUT_CSV = OUTPUT_DIR / "golden_master_multi_source.csv"

VARIANT_RESULTS_CSV = OUTPUT_DIR / "selection_policy_variant_results.csv"
VALIDATION_CURVES_CSV = OUTPUT_DIR / "selection_policy_validation_curves.csv"
TEST_CURVES_CSV = OUTPUT_DIR / "selection_policy_test_curves.csv"
SELECTED_METHODS_MD = OUTPUT_DIR / "selection_policy_selected_methods.md"
TEST_MEAN_PLOT = OUTPUT_DIR / "selection_policy_test_mean_ratings.png"
TEST_PROB_PLOT = OUTPUT_DIR / "selection_policy_test_probability_useful_ge2.png"

DROP_FRACTIONS = [round(value, 2) for value in np.arange(0.0, 0.81, 0.05)]
OBJECTIVES = {
    "enjoyment": "mean_actual_enjoyment",
    "usefulness": "mean_actual_usefulness",
    "balanced": "mean_actual_balanced",
    "prob_usefulness_ge_2": "prob_usefulness_ge_2",
}
SCORE_COLUMNS = {
    "enjoyment": "pred_enjoyment",
    "usefulness": "pred_usefulness",
    "balanced": "pred_balanced",
    "prob_usefulness_ge_2": "pred_prob_usefulness_ge_2",
}
SOURCE_SPECS = {
    "goodreads": ("goodreads_rating_raw", "goodreads_count_raw"),
    "openlibrary": ("openlibrary_rating_raw", "openlibrary_count_raw"),
    "amazon": ("amazon_rating_raw", "amazon_count_raw"),
}
SOURCE_LABELS = {
    "goodreads": "Goodreads",
    "openlibrary": "Open Library",
    "amazon": "Amazon",
}
MEAN_LINE_COLORS = {
    "mean_actual_enjoyment": "#1f77b4",
    "mean_actual_usefulness": "#ff7f0e",
}


@dataclass(frozen=True)
class Variant:
    aggregator: str
    missing_mode: str
    shrink: bool

    @property
    def name(self) -> str:
        shrink_label = "shrink" if self.shrink else "no_shrink"
        return f"{self.aggregator}__{self.missing_mode}__{shrink_label}"


class ConstantPredictor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), self.value, dtype=float)


def average_available(*values: object) -> float:
    valid = [float(value) for value in values if pd.notna(value)]
    if not valid:
        return math.nan
    return float(np.mean(valid))


def category_bucket(category: object) -> str:
    text = "" if pd.isna(category) else str(category).strip()
    if text == "fiction":
        return "Fiction"
    if text == "Literature":
        return "Literature"
    if text in {"Computer Science", "Machine Learning", "Math", "Unknown Shelf"}:
        return "Technical/Other"
    return "General/Business"


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_CSV)
    frame.columns = frame.columns.str.strip()
    frame = frame[frame["source"] != "Holdout 2026 (unverified)"].copy()
    frame["estimated_finish"] = pd.to_datetime(frame["estimated_finish"], errors="coerce")
    frame["category_bucket"] = frame["category"].map(category_bucket)
    frame["title"] = frame["title"].fillna("").astype(str)
    frame["avg_enjoyment"] = pd.to_numeric(frame["avg_enjoyment"], errors="coerce")
    frame["avg_usefulness"] = pd.to_numeric(frame["avg_usefulness"], errors="coerce")

    frame["goodreads_rating_raw"] = pd.to_numeric(
        frame["goodreads_rating_verified"], errors="coerce"
    )
    frame["goodreads_count_raw"] = pd.to_numeric(
        frame["goodreads_rating_count_verified"], errors="coerce"
    )
    frame["openlibrary_rating_raw"] = frame.apply(
        lambda row: average_available(row.get("ol_link_rating"), row.get("ol_nolink_rating")),
        axis=1,
    )
    frame["openlibrary_count_raw"] = frame.apply(
        lambda row: average_available(
            row.get("ol_link_reviews"), row.get("ol_nolink_reviews")
        ),
        axis=1,
    )
    frame["amazon_rating_raw"] = frame.apply(
        lambda row: average_available(
            row.get("amazon_link_rating"), row.get("amazon_nolink_rating")
        ),
        axis=1,
    )
    frame["amazon_count_raw"] = frame.apply(
        lambda row: average_available(
            row.get("amazon_link_reviews"), row.get("amazon_nolink_reviews")
        ),
        axis=1,
    )
    return frame


def split_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[pd.Timestamp, pd.Timestamp]]:
    train = frame[~frame["source"].eq("Holdout 2026")].copy()
    holdout = frame[frame["source"].eq("Holdout 2026")].copy()
    holdout = holdout.sort_values(["estimated_finish", "title"], na_position="last").reset_index(
        drop=True
    )
    mid = len(holdout) // 2
    validation = holdout.iloc[:mid].copy()
    test = holdout.iloc[mid:].copy()
    return train, validation, test, (
        validation["estimated_finish"].min(),
        test["estimated_finish"].min(),
    )


def bucket_stat_map(
    train: pd.DataFrame, value_col: str
) -> tuple[dict[str, tuple[float, float, float]], tuple[float, float, float]]:
    stats: dict[str, tuple[float, float, float]] = {}
    global_series = pd.to_numeric(train[value_col], errors="coerce")
    global_mean = float(global_series.mean())
    global_std = float(global_series.std())
    global_var = float(global_series.var())
    if not np.isfinite(global_std) or global_std <= 1e-8:
        global_std = 1.0
    if not np.isfinite(global_var) or global_var <= 1e-8:
        global_var = 1.0

    for bucket, bucket_frame in train.groupby("category_bucket", dropna=False):
        series = pd.to_numeric(bucket_frame[value_col], errors="coerce")
        mean = float(series.mean()) if series.notna().any() else global_mean
        std = float(series.std()) if series.notna().any() else global_std
        var = float(series.var()) if series.notna().any() else global_var
        if not np.isfinite(std) or std <= 1e-8:
            std = global_std
        if not np.isfinite(var) or var <= 1e-8:
            var = global_var
        stats[str(bucket)] = (mean, std, var)
    return stats, (global_mean, global_std, global_var)


def apply_bucket_shrinkage(
    frame: pd.DataFrame,
    rating_col: str,
    count_col: str,
    bucket_stats: dict[str, tuple[float, float, float]],
    global_stats: tuple[float, float, float],
    prior_strength: float,
) -> pd.Series:
    shrunk: list[float] = []
    global_mean = global_stats[0]
    safe_prior = prior_strength if np.isfinite(prior_strength) and prior_strength > 0 else 1.0
    for row in frame.itertuples(index=False):
        rating = getattr(row, rating_col)
        if pd.isna(rating):
            shrunk.append(math.nan)
            continue
        bucket = str(getattr(row, "category_bucket"))
        bucket_mean = bucket_stats.get(bucket, global_stats)[0]
        count = getattr(row, count_col)
        count_value = float(count) if pd.notna(count) and float(count) >= 0 else safe_prior
        weight = count_value / (count_value + safe_prior)
        target_mean = bucket_mean if np.isfinite(bucket_mean) else global_mean
        shrunk.append(float(target_mean + weight * (float(rating) - target_mean)))
    return pd.Series(shrunk, index=frame.index, dtype=float)


def attach_zscores(
    train: pd.DataFrame, frame: pd.DataFrame, variant: Variant
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    train = train.copy()
    transformed = frame.copy()
    source_meta: dict[str, dict[str, object]] = {}

    for source, (rating_col, count_col) in SOURCE_SPECS.items():
        if variant.shrink:
            prior_strength = float(pd.to_numeric(train[count_col], errors="coerce").median())
            train_processed_col = f"{source}_processed_rating"
            transformed_processed_col = f"{source}_processed_rating"
            train_stats_source = train.copy()
            train_stats, global_stats = bucket_stat_map(train_stats_source, rating_col)
            train[train_processed_col] = apply_bucket_shrinkage(
                train,
                rating_col,
                count_col,
                train_stats,
                global_stats,
                prior_strength,
            )
            transformed[transformed_processed_col] = apply_bucket_shrinkage(
                transformed,
                rating_col,
                count_col,
                train_stats,
                global_stats,
                prior_strength,
            )
            stats, global_processed = bucket_stat_map(train, train_processed_col)
            base_col = train_processed_col
            global_meta = global_processed
        else:
            prior_strength = float(pd.to_numeric(train[count_col], errors="coerce").median())
            stats, global_processed = bucket_stat_map(train, rating_col)
            base_col = rating_col
            global_meta = global_processed

        z_col = f"{source}_z"
        for target_frame in [train, transformed]:
            z_values: list[float] = []
            for row in target_frame.itertuples(index=False):
                value = getattr(row, base_col)
                if pd.isna(value):
                    z_values.append(math.nan)
                    continue
                bucket = str(getattr(row, "category_bucket"))
                mean, std, _ = stats.get(bucket, global_meta)
                z_values.append(float((float(value) - mean) / std))
            target_frame[z_col] = pd.Series(z_values, index=target_frame.index, dtype=float)

        source_meta[source] = {
            "bucket_stats": stats,
            "global_stats": global_meta,
            "prior_strength": prior_strength,
            "z_col": z_col,
        }
    return transformed, source_meta


def build_imputation_features(
    frame: pd.DataFrame, target_source: str, bucket_levels: list[str]
) -> pd.DataFrame:
    feature_data: dict[str, pd.Series] = {}
    for source in SOURCE_SPECS:
        if source == target_source:
            continue
        z_col = f"{source}_z"
        feature_data[f"{source}_z"] = pd.to_numeric(frame[z_col], errors="coerce").fillna(0.0)
        feature_data[f"{source}_available"] = frame[z_col].notna().astype(float)
    for bucket in bucket_levels:
        feature_data[f"bucket__{bucket}"] = frame["category_bucket"].eq(bucket).astype(float)
    return pd.DataFrame(feature_data, index=frame.index)


def impute_missing_sources(
    train: pd.DataFrame, frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    transformed = frame.copy()
    bucket_levels = sorted(train["category_bucket"].fillna("NA").astype(str).unique().tolist())
    models: dict[str, object] = {}
    for target_source in SOURCE_SPECS:
        target_col = f"{target_source}_z"
        train_target = pd.to_numeric(train[target_col], errors="coerce")
        target_mask = train_target.notna()
        if target_mask.sum() < 20:
            models[target_source] = None
            transformed[target_col] = transformed[target_col].fillna(0.0)
            continue
        X_train = build_imputation_features(train, target_source, bucket_levels)
        model = Ridge(alpha=1.0)
        model.fit(X_train.loc[target_mask], train_target.loc[target_mask])
        models[target_source] = model

        X_target = build_imputation_features(transformed, target_source, bucket_levels)
        missing_mask = transformed[target_col].isna()
        if missing_mask.any():
            transformed.loc[missing_mask, target_col] = model.predict(X_target.loc[missing_mask])
    return transformed, models


def aggregate_signal(
    frame: pd.DataFrame, source_meta: dict[str, dict[str, object]], variant: Variant
) -> pd.Series:
    values: list[float] = []
    z_cols = [f"{source}_z" for source in SOURCE_SPECS]
    all_present = frame[z_cols].notna().all(axis=1)
    for idx, row in frame.iterrows():
        if variant.missing_mode == "drop" and not bool(all_present.loc[idx]):
            values.append(math.nan)
            continue
        bucket = str(row["category_bucket"])
        weighted_sum = 0.0
        weight_total = 0.0
        raw_values: list[float] = []
        for source in SOURCE_SPECS:
            z_value = row[f"{source}_z"]
            if pd.isna(z_value):
                continue
            if variant.aggregator == "mean_z":
                raw_values.append(float(z_value))
            else:
                _, _, raw_var = source_meta[source]["bucket_stats"].get(
                    bucket, source_meta[source]["global_stats"]
                )
                weight = 1.0 / raw_var
                weighted_sum += weight * float(z_value)
                weight_total += weight
        if variant.aggregator == "mean_z":
            values.append(float(np.mean(raw_values)) if raw_values else math.nan)
        else:
            values.append(weighted_sum / weight_total if weight_total > 0 else math.nan)
    return pd.Series(values, index=frame.index, dtype=float)


def fit_isotonic(train_scores: pd.Series, target: pd.Series, y_bounds: tuple[float, float] | None = None):
    mask = train_scores.notna() & target.notna()
    x = train_scores.loc[mask].to_numpy(dtype=float)
    y = target.loc[mask].to_numpy(dtype=float)
    if len(x) < 12:
        return ConstantPredictor(float(np.nanmean(y) if len(y) else 0.0))
    unique_x = np.unique(x)
    if len(unique_x) < 3:
        return ConstantPredictor(float(np.nanmean(y)))
    kwargs = {"out_of_bounds": "clip"}
    if y_bounds is not None:
        kwargs["y_min"] = y_bounds[0]
        kwargs["y_max"] = y_bounds[1]
    model = IsotonicRegression(**kwargs)
    model.fit(x, y)
    return model


def predict_model(model: object, scores: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=scores.index, dtype=float)
    mask = scores.notna()
    if not mask.any():
        return result
    x = scores.loc[mask].to_numpy(dtype=float)
    predicted = model.predict(x) if hasattr(model, "predict") else np.full(len(x), np.nan)
    result.loc[mask] = predicted
    return result


def build_scored_frame(
    train: pd.DataFrame, target_frame: pd.DataFrame, variant: Variant
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_targets = train.copy()
    frame = target_frame.copy()
    transformed_train, source_meta = attach_zscores(train_targets, train_targets, variant)
    transformed_frame, _ = attach_zscores(train_targets, frame, variant)

    imputation_meta: dict[str, object] | None = None
    if variant.missing_mode == "impute":
        transformed_train, imputation_meta = impute_missing_sources(
            transformed_train, transformed_train
        )
        transformed_frame, _ = impute_missing_sources(transformed_train, transformed_frame)

    transformed_train["combined_score"] = aggregate_signal(
        transformed_train, source_meta, variant
    )
    transformed_frame["combined_score"] = aggregate_signal(
        transformed_frame, source_meta, variant
    )

    transformed_train["usefulness_ge_2"] = (
        pd.to_numeric(transformed_train["avg_usefulness"], errors="coerce") >= 2.0
    ).astype(float)
    enjoy_model = fit_isotonic(
        transformed_train["combined_score"],
        pd.to_numeric(transformed_train["avg_enjoyment"], errors="coerce"),
        y_bounds=(1.0, 5.0),
    )
    useful_model = fit_isotonic(
        transformed_train["combined_score"],
        pd.to_numeric(transformed_train["avg_usefulness"], errors="coerce"),
        y_bounds=(1.0, 5.0),
    )
    useful_prob_model = fit_isotonic(
        transformed_train["combined_score"],
        transformed_train["usefulness_ge_2"],
        y_bounds=(0.0, 1.0),
    )

    transformed_frame["pred_enjoyment"] = predict_model(
        enjoy_model, transformed_frame["combined_score"]
    )
    transformed_frame["pred_usefulness"] = predict_model(
        useful_model, transformed_frame["combined_score"]
    )
    transformed_frame["pred_balanced"] = (
        transformed_frame["pred_enjoyment"] + transformed_frame["pred_usefulness"]
    ) / 2.0
    transformed_frame["pred_prob_usefulness_ge_2"] = predict_model(
        useful_prob_model, transformed_frame["combined_score"]
    )
    transformed_frame["actual_balanced"] = (
        pd.to_numeric(transformed_frame["avg_enjoyment"], errors="coerce")
        + pd.to_numeric(transformed_frame["avg_usefulness"], errors="coerce")
    ) / 2.0

    metadata = {
        "variant": variant,
        "source_meta": source_meta,
        "imputation_meta": imputation_meta,
        "enjoy_model": enjoy_model,
        "useful_model": useful_model,
        "useful_prob_model": useful_prob_model,
    }
    return transformed_frame, metadata


def selection_curve(scored: pd.DataFrame, score_column: str, split_name: str) -> pd.DataFrame:
    valid = scored[
        scored[score_column].notna()
        & pd.to_numeric(scored["avg_enjoyment"], errors="coerce").notna()
        & pd.to_numeric(scored["avg_usefulness"], errors="coerce").notna()
    ].copy()
    if len(valid) < 8:
        return pd.DataFrame()
    valid = valid.sort_values([score_column, "title"], ascending=[False, True]).reset_index(
        drop=True
    )
    rows: list[dict[str, object]] = []
    for drop_fraction in DROP_FRACTIONS:
        drop_n = int(np.floor(len(valid) * drop_fraction))
        keep_n = len(valid) - drop_n
        if keep_n < 5:
            continue
        kept = valid.iloc[:keep_n].copy()
        rows.append(
            {
                "split": split_name,
                "drop_fraction": drop_fraction,
                "drop_percent": drop_fraction * 100,
                "keep_n": keep_n,
                "n_scored": int(len(valid)),
                "cutoff_predicted_value": float(kept[score_column].iloc[-1]),
                "mean_actual_enjoyment": float(
                    pd.to_numeric(kept["avg_enjoyment"], errors="coerce").mean()
                ),
                "mean_actual_usefulness": float(
                    pd.to_numeric(kept["avg_usefulness"], errors="coerce").mean()
                ),
                "mean_actual_balanced": float(
                    pd.to_numeric(kept["actual_balanced"], errors="coerce").mean()
                ),
                "prob_usefulness_ge_2": float(
                    (
                        pd.to_numeric(kept["avg_usefulness"], errors="coerce") >= 2.0
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def objective_summary_row(
    curve: pd.DataFrame, variant: Variant, objective_name: str, split_name: str
) -> dict[str, object]:
    metric_col = OBJECTIVES[objective_name]
    if curve.empty:
        return {
            "variant_name": variant.name,
            "aggregator": variant.aggregator,
            "missing_mode": variant.missing_mode,
            "shrink": variant.shrink,
            "objective": objective_name,
            "split": split_name,
            "policy_score": math.nan,
            "baseline_value": math.nan,
            "best_value": math.nan,
            "best_drop_percent": math.nan,
            "n_scored": 0,
        }
    best_idx = curve[metric_col].idxmax()
    return {
        "variant_name": variant.name,
        "aggregator": variant.aggregator,
        "missing_mode": variant.missing_mode,
        "shrink": variant.shrink,
        "objective": objective_name,
        "split": split_name,
        "policy_score": float(curve[metric_col].mean()),
        "baseline_value": float(curve.loc[curve["drop_percent"].eq(0), metric_col].iloc[0]),
        "best_value": float(curve.loc[best_idx, metric_col]),
        "best_drop_percent": float(curve.loc[best_idx, "drop_percent"]),
        "n_scored": int(curve["n_scored"].iloc[0]),
    }


def describe_variant(selected_row: pd.Series, source_meta: dict[str, dict[str, object]]) -> str:
    variant = selected_row["variant_name"]
    aggregator = selected_row["aggregator"]
    missing_mode = selected_row["missing_mode"]
    shrink = bool(selected_row["shrink"])
    aggregator_text = (
        "simple mean of the three source z-scores"
        if aggregator == "mean_z"
        else "inverse raw-variance weighted mean of the three source z-scores"
    )
    missing_text = (
        "complete-case only: books missing any source stay unscored"
        if missing_mode == "drop"
        else "missing source z-scores imputed from the other source z-scores plus category bucket via ridge regression"
    )
    if shrink:
        shrink_bits = []
        for source, meta in source_meta.items():
            shrink_bits.append(
                f"{SOURCE_LABELS[source]} k={meta['prior_strength']:.1f}"
            )
        shrink_text = (
            "source ratings shrunk toward training category-bucket means with "
            "rating' = mean_bucket + n/(n+k) * (rating - mean_bucket); "
            + ", ".join(shrink_bits)
        )
    else:
        shrink_text = "no review-count shrinkage applied before z-scoring"
    return (
        f"`{variant}`: {aggregator_text}; {missing_text}; {shrink_text}. "
        "Category buckets were `Fiction`, `Literature`, `Technical/Other`, and `General/Business`."
    )


def plot_mean_curves(curves: pd.DataFrame, selected_rows: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes = axes.flatten()
    for axis, (_, row) in zip(axes, selected_rows.iterrows(), strict=False):
        subset = curves[
            (curves["variant_name"] == row["variant_name"])
            & (curves["objective"] == row["objective"])
            & (curves["split"] == "test")
        ].copy()
        if subset.empty:
            axis.axis("off")
            continue
        for metric_col, label in [
            ("mean_actual_enjoyment", "Actual enjoyment"),
            ("mean_actual_usefulness", "Actual usefulness"),
        ]:
            axis.plot(
                subset["drop_percent"],
                subset[metric_col],
                marker="o",
                linewidth=2,
                color=MEAN_LINE_COLORS[metric_col],
                label=label,
            )
        for _, point in subset.iterrows():
            axis.annotate(
                f"{point['cutoff_predicted_value']:.2f}",
                (point["drop_percent"], point["mean_actual_balanced"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="#444444",
            )
        axis.set_title(
            f"{row['objective']} | {row['variant_name']}\n"
            f"n={int(subset['n_scored'].iloc[0])}"
        )
        axis.set_xlabel("% dropped below cutoff")
        axis.set_ylabel("Actual mean rating of kept books")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Test-set mean enjoyment/usefulness if books were selected by predicted cutoff",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_probability_curves(
    curves: pd.DataFrame, selected_rows: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes = axes.flatten()
    for axis, (_, row) in zip(axes, selected_rows.iterrows(), strict=False):
        subset = curves[
            (curves["variant_name"] == row["variant_name"])
            & (curves["objective"] == row["objective"])
            & (curves["split"] == "test")
        ].copy()
        if subset.empty:
            axis.axis("off")
            continue
        axis.plot(
            subset["drop_percent"],
            subset["prob_usefulness_ge_2"],
            marker="o",
            linewidth=2,
            color="#2d6a4f",
        )
        for _, point in subset.iterrows():
            axis.annotate(
                f"{point['cutoff_predicted_value']:.2f}",
                (point["drop_percent"], point["prob_usefulness_ge_2"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="#444444",
            )
        axis.set_ylim(0, 1.05)
        axis.set_title(
            f"{row['objective']} | {row['variant_name']}\n"
            f"n={int(subset['n_scored'].iloc[0])}"
        )
        axis.set_xlabel("% dropped below cutoff")
        axis.set_ylabel("P(actual usefulness >= 2 | kept)")
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Test-set probability usefulness >= 2 if books were selected by predicted cutoff",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def render_method_markdown(
    selected_rows: pd.DataFrame,
    source_meta_map: dict[str, dict[str, dict[str, object]]],
    summary_rows: list[str],
    split_dates: tuple[pd.Timestamp, pd.Timestamp],
) -> str:
    validation_start, test_start = split_dates
    lines = [
        "# Selection Policy Summary",
        "",
        "## Holdout split",
        f"- Validation = first half of `Holdout 2026`, starting `{validation_start.date()}`",
        f"- Test = second half of `Holdout 2026`, starting `{test_start.date()}`",
        "",
        "## Selected methods",
    ]
    for _, row in selected_rows.iterrows():
        lines.append(
            f"- `{row['objective']}`: {describe_variant(row, source_meta_map[row['variant_name']])}"
        )
    lines.extend(["", "## Validation selection summary"])
    lines.extend(summary_rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    frame = load_frame()
    train, validation, test, split_dates = split_frame(frame)
    variants = [
        Variant(aggregator=aggregator, missing_mode=missing_mode, shrink=shrink)
        for aggregator in ["mean_z", "precision_weighted_z"]
        for missing_mode in ["drop", "impute"]
        for shrink in [False, True]
    ]

    summary_records: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    source_meta_map: dict[str, dict[str, dict[str, object]]] = {}

    for variant in variants:
        scored_validation, metadata = build_scored_frame(train, validation, variant)
        scored_test, _ = build_scored_frame(train, test, variant)
        source_meta_map[variant.name] = metadata["source_meta"]

        for split_name, scored in [("validation", scored_validation), ("test", scored_test)]:
            for objective, score_col in SCORE_COLUMNS.items():
                curve = selection_curve(scored, score_col, split_name)
                if not curve.empty:
                    curve["variant_name"] = variant.name
                    curve["aggregator"] = variant.aggregator
                    curve["missing_mode"] = variant.missing_mode
                    curve["shrink"] = variant.shrink
                    curve["objective"] = objective
                    curve_frames.append(curve)
                summary_records.append(
                    objective_summary_row(curve, variant, objective, split_name)
                )

    summary_df = pd.DataFrame(summary_records)
    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    validation_curves = curves_df[curves_df["split"] == "validation"].copy()
    test_curves = curves_df[curves_df["split"] == "test"].copy()

    selected_rows = (
        summary_df[summary_df["split"] == "validation"]
        .sort_values(["objective", "policy_score", "n_scored"], ascending=[True, False, False])
        .groupby("objective", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    plot_mean_curves(curves_df, selected_rows, TEST_MEAN_PLOT)
    plot_probability_curves(curves_df, selected_rows, TEST_PROB_PLOT)

    validation_summary_lines = []
    for _, row in selected_rows.iterrows():
        validation_summary_lines.append(
            f"- `{row['objective']}`: validation policy score = {row['policy_score']:.3f}, "
            f"best at {row['best_drop_percent']:.0f}% dropped, scored books = {int(row['n_scored'])}"
        )

    SELECTED_METHODS_MD.write_text(
        render_method_markdown(
            selected_rows=selected_rows,
            source_meta_map=source_meta_map,
            summary_rows=validation_summary_lines,
            split_dates=split_dates,
        )
    )

    summary_df.to_csv(VARIANT_RESULTS_CSV, index=False)
    validation_curves.to_csv(VALIDATION_CURVES_CSV, index=False)
    test_curves.to_csv(TEST_CURVES_CSV, index=False)

    print(f"Saved {VARIANT_RESULTS_CSV.name}")
    print(f"Saved {VALIDATION_CURVES_CSV.name}")
    print(f"Saved {TEST_CURVES_CSV.name}")
    print(f"Saved {TEST_MEAN_PLOT.name}")
    print(f"Saved {TEST_PROB_PLOT.name}")
    print(f"Saved {SELECTED_METHODS_MD.name}")


if __name__ == "__main__":
    main()
