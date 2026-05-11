"""Evaluate predictions on the newly labeled new_books_to_rate 2026 holdout."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_books_tracking.future_prediction_evaluation import (
    MIN_TRAIN_ROWS,
    MODEL_BUILDERS,
    THRESHOLDS,
    FeatureSpec,
    model_predictions_for_split,
)
from ai_books_tracking.goodreads_followup_analysis import (
    derive_analysis_columns,
    load_data,
    spearman_summary,
)
from ai_books_tracking.goodreads_ratings import (
    clean_title_for_search,
    enrich_books_with_goodreads,
    normalize_text,
)

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR.parent / "data"
NEW_BOOKS_INPUT = DATA_DIR / "Books Read and their effects - new_books_to_rate 2026.csv"
NEW_BOOKS_GOODREADS = OUTPUT_DIR / "new_books_to_rate_2026_goodreads.csv"
NEW_BOOKS_ENRICHED = OUTPUT_DIR / "new_books_to_rate_2026_enriched.csv"
LABEL_DISTRIBUTION_SUMMARY_CSV = (
    OUTPUT_DIR / "new_books_to_rate_2026_label_distribution_summary.csv"
)
LABEL_AGREEMENT_CSV = OUTPUT_DIR / "new_books_to_rate_2026_label_agreement.csv"
CENTERING_COMPARISON_CSV = (
    OUTPUT_DIR / "new_books_to_rate_2026_centering_comparison.csv"
)
PREDICTION_RESULTS_CSV = OUTPUT_DIR / "new_books_to_rate_2026_prediction_results.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "new_books_to_rate_2026_predictions.csv"
BEST_RESULTS_CSV = OUTPUT_DIR / "new_books_to_rate_2026_best_results.csv"
LABEL_DISTRIBUTIONS_PNG = OUTPUT_DIR / "new_books_to_rate_2026_label_distributions.png"
POLICY_FULL_HOLDOUT_CSV = OUTPUT_DIR / "new_books_to_rate_2026_policy_full_holdout.csv"
POLICY_FULL_HOLDOUT_BEST_CSV = (
    OUTPUT_DIR / "new_books_to_rate_2026_policy_full_holdout_best.csv"
)
POLICY_SPLIT_RESULTS_CSV = (
    OUTPUT_DIR / "new_books_to_rate_2026_policy_split_results.csv"
)
POLICY_SPLIT_BEST_CSV = OUTPUT_DIR / "new_books_to_rate_2026_policy_split_best.csv"
PREDICTIONS_SPLIT_CSV = OUTPUT_DIR / "new_books_to_rate_2026_predictions_split.csv"

ACTIONABLE_FEATURE_SPECS = (
    FeatureSpec("preread_base"),
    FeatureSpec(
        "preread_plus_goodreads_raw",
        include_goodreads="raw_best",
    ),
    FeatureSpec(
        "preread_plus_goodreads_conservative",
        include_goodreads="conservative",
    ),
)
POLICY_TARGETS = ("avg_enjoyment", "avg_usefulness")
UTILITY_BASES = {
    "enjoyment": 1.3,
    "usefulness": 1.8,
}
POLICY_MIN_SIDE_N = 5
POLICY_BALANCED_KEEP_SHARE_RANGE = (0.25, 0.75)
TARGET_VARIANTS = (
    "Enjoyment (/5)",
    "Enjoyment (/5)_ratings2",
    "avg_enjoyment",
    "avg_enjoyment_centered",
    "Usefulness /5 to Me",
    "Usefulness /5 to Me_ratings2",
    "avg_usefulness",
    "avg_usefulness_centered",
)
TARGET_LABELS = {
    "Enjoyment (/5)": "Enjoyment pass 1",
    "Enjoyment (/5)_ratings2": "Enjoyment pass 2",
    "avg_enjoyment": "Enjoyment average",
    "avg_enjoyment_centered": "Enjoyment centered average",
    "Usefulness /5 to Me": "Usefulness pass 1",
    "Usefulness /5 to Me_ratings2": "Usefulness pass 2",
    "avg_usefulness": "Usefulness average",
    "avg_usefulness_centered": "Usefulness centered average",
}
TARGET_FAMILIES = {
    "Enjoyment (/5)": "enjoyment",
    "Enjoyment (/5)_ratings2": "enjoyment",
    "avg_enjoyment": "enjoyment",
    "avg_enjoyment_centered": "enjoyment",
    "Usefulness /5 to Me": "usefulness",
    "Usefulness /5 to Me_ratings2": "usefulness",
    "avg_usefulness": "usefulness",
    "avg_usefulness_centered": "usefulness",
}


@dataclass(frozen=True)
class CenteringSummary:
    first_mean: float
    second_mean: float
    pooled_mean: float


def normalize_new_holdout(path: Path = NEW_BOOKS_INPUT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "Enjoyment (/5) 2nd": "Enjoyment (/5)_ratings2",
            "Usefulness /5 to Me.1": "Usefulness /5 to Me_ratings2",
        }
    )
    for column in [
        "Enjoyment (/5)",
        "Usefulness /5 to Me",
        "Enjoyment (/5)_ratings2",
        "Usefulness /5 to Me_ratings2",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    finished = pd.to_datetime(df["date_finished"], format="mixed", errors="coerce")
    df["earliest_modified"] = finished
    df["latest_modified"] = finished
    df["earliest_modified_ratings2"] = finished
    df["latest_modified_ratings2"] = finished
    if "Long Term Effects" not in df.columns:
        df["Long Term Effects"] = ""
    df["Long Term Effects"] = df["Long Term Effects"].fillna("")
    for column in ["author", "author_ratings2", "author_goodreads"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")
    df["gb_page_count"] = pd.to_numeric(df.get("gb_page_count"), errors="coerce")
    df["pub_year"] = pd.to_numeric(df.get("pub_year"), errors="coerce")
    return df


def load_enriched_holdout(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for column in [
        "Enjoyment (/5)",
        "Usefulness /5 to Me",
        "Enjoyment (/5)_ratings2",
        "Usefulness /5 to Me_ratings2",
        "goodreads_rating",
        "goodreads_rating_count",
        "goodreads_rating_raw_best",
        "goodreads_rating_count_raw_best",
        "gb_page_count",
        "pub_year",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_new_goodreads_lookup_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        search_title = clean_title_for_search(
            row.get("title", ""), row.get("author", "")
        )
        canonical_key = normalize_text(f"{search_title}::{row.get('title', '')}")
        rows.append(
            {
                "canonical_key": canonical_key,
                "title": row.get("title", ""),
                "author": row.get("author", ""),
                "Bookshelf": row.get("Bookshelf", ""),
                "search_title": search_title,
                "search_author": row.get("author", ""),
                "filename_play": "",
                "filename_ratings2": "",
            }
        )
    return pd.DataFrame(rows)


def add_goodreads_to_holdout(
    df: pd.DataFrame,
    output_path: Path = NEW_BOOKS_GOODREADS,
    force_refresh: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    lookup = build_new_goodreads_lookup_frame(df)
    goodreads = enrich_books_with_goodreads(
        books=lookup,
        output_path=output_path,
        force_refresh=force_refresh,
        sleep_seconds=0.1,
        verbose=verbose,
    )
    merged = df.merge(
        goodreads[
            [
                "canonical_key",
                "goodreads_status",
                "goodreads_match_method",
                "goodreads_url",
                "goodreads_title",
                "goodreads_author",
                "goodreads_rating",
                "goodreads_rating_count",
                "goodreads_match_score",
                "goodreads_candidates_json",
            ]
        ],
        on="canonical_key",
        how="left",
    )
    merged["goodreads_rating_raw_best"] = pd.to_numeric(
        merged["goodreads_rating"], errors="coerce"
    )
    merged["goodreads_rating_count_raw_best"] = pd.to_numeric(
        merged["goodreads_rating_count"], errors="coerce"
    )
    return merged


def build_centered_average(
    first: pd.Series, second: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, CenteringSummary]:
    first_numeric = pd.to_numeric(first, errors="coerce")
    second_numeric = pd.to_numeric(second, errors="coerce")
    first_mean = float(first_numeric.mean())
    second_mean = float(second_numeric.mean())
    pooled_mean = float(np.nanmean([first_mean, second_mean]))
    centered_first = first_numeric - first_mean + pooled_mean
    centered_second = second_numeric - second_mean + pooled_mean
    centered_average = pd.concat([centered_first, centered_second], axis=1).mean(axis=1)
    return (
        centered_first,
        centered_second,
        centered_average,
        CenteringSummary(
            first_mean=first_mean,
            second_mean=second_mean,
            pooled_mean=pooled_mean,
        ),
    )


def add_centered_targets(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    (
        enriched["Enjoyment (/5)_centered"],
        enriched["Enjoyment (/5)_ratings2_centered"],
        enriched["avg_enjoyment_centered"],
        _,
    ) = build_centered_average(
        enriched["Enjoyment (/5)"],
        enriched["Enjoyment (/5)_ratings2"],
    )
    (
        enriched["Usefulness /5 to Me_centered"],
        enriched["Usefulness /5 to Me_ratings2_centered"],
        enriched["avg_usefulness_centered"],
        _,
    ) = build_centered_average(
        enriched["Usefulness /5 to Me"],
        enriched["Usefulness /5 to Me_ratings2"],
    )
    return enriched


def summarize_label_distributions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in TARGET_VARIANTS:
        series = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "target": column,
                "label": TARGET_LABELS[column],
                "family": TARGET_FAMILIES[column],
                "count": int(series.notna().sum()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_pass_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [
        ("enjoyment", "Enjoyment (/5)", "Enjoyment (/5)_ratings2"),
        (
            "usefulness",
            "Usefulness /5 to Me",
            "Usefulness /5 to Me_ratings2",
        ),
    ]
    for family, first_col, second_col in pairs:
        frame = df[[first_col, second_col]].dropna().copy()
        rho, p_value = spearman_summary(frame[first_col], frame[second_col])
        diff = frame[first_col] - frame[second_col]
        rows.append(
            {
                "family": family,
                "n": int(len(frame)),
                "pass1_mean": float(frame[first_col].mean()),
                "pass2_mean": float(frame[second_col].mean()),
                "mean_diff_pass1_minus_pass2": float(diff.mean()),
                "mean_abs_gap": float(diff.abs().mean()),
                "max_abs_gap": float(diff.abs().max()),
                "pearson_r": float(frame[first_col].corr(frame[second_col])),
                "spearman_rho": rho,
                "spearman_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def plot_label_distributions(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    bins = np.arange(0.75, 5.26, 0.25)
    plot_specs = [
        (
            axes[0, 0],
            "Enjoyment ratings",
            [
                ("Enjoyment (/5)", "pass 1"),
                ("Enjoyment (/5)_ratings2", "pass 2"),
                ("avg_enjoyment", "raw avg"),
                ("avg_enjoyment_centered", "centered avg"),
            ],
        ),
        (
            axes[0, 1],
            "Usefulness ratings",
            [
                ("Usefulness /5 to Me", "pass 1"),
                ("Usefulness /5 to Me_ratings2", "pass 2"),
                ("avg_usefulness", "raw avg"),
                ("avg_usefulness_centered", "centered avg"),
            ],
        ),
    ]
    colors = ["#0B3954", "#C81D25", "#087E8B", "#FF5A5F"]
    for axis, title, columns in plot_specs:
        for (column, label), color in zip(columns, colors, strict=False):
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("Rating")
        axis.set_ylabel("Density")
        axis.legend(frameon=False)

    enjoy_gap = pd.to_numeric(df["Enjoyment (/5)"], errors="coerce") - pd.to_numeric(
        df["Enjoyment (/5)_ratings2"], errors="coerce"
    )
    useful_gap = pd.to_numeric(
        df["Usefulness /5 to Me"], errors="coerce"
    ) - pd.to_numeric(df["Usefulness /5 to Me_ratings2"], errors="coerce")
    for axis, values, title in [
        (axes[1, 0], enjoy_gap, "Enjoyment pass1 - pass2"),
        (axes[1, 1], useful_gap, "Usefulness pass1 - pass2"),
    ]:
        axis.hist(values.dropna(), bins=np.arange(-2.25, 2.51, 0.25), color="#F4D35E")
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Difference")
        axis.set_ylabel("Count")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_target_variant(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    spec: FeatureSpec,
    model_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    train_valid = train_df[train_df[target_col].notna()].copy()
    holdout_valid = holdout_df[holdout_df[target_col].notna()].copy()
    if len(train_valid) < MIN_TRAIN_ROWS or holdout_valid.empty:
        return {}, pd.DataFrame()

    predictions = model_predictions_for_split(
        train_df=train_valid,
        test_df=holdout_valid,
        target_col=target_col,
        spec=spec,
        model_name=model_name,
    )
    actual = pd.to_numeric(holdout_valid[target_col], errors="coerce")
    predicted = pd.Series(predictions, index=holdout_valid.index, dtype=float)
    rho, p_value = spearman_summary(predicted, actual)

    detail_columns = list(
        dict.fromkeys(
            [
                "title",
                "Bookshelf",
                "date_finished",
                target_col,
                "avg_enjoyment",
                "avg_enjoyment_centered",
                "avg_usefulness",
                "avg_usefulness_centered",
                "goodreads_rating",
            ]
        )
    )
    detail = holdout_valid[detail_columns].copy()
    detail["target"] = target_col
    detail["target_label"] = TARGET_LABELS[target_col]
    detail["feature_spec"] = spec.name
    detail["model"] = model_name
    detail["prediction"] = predicted.to_numpy()
    detail["prediction_error"] = detail["prediction"] - actual.to_numpy()

    result = {
        "target": target_col,
        "target_label": TARGET_LABELS[target_col],
        "family": TARGET_FAMILIES[target_col],
        "feature_spec": spec.name,
        "model": model_name,
        "n": int(len(holdout_valid)),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "spearman_rho": rho,
        "spearman_p": p_value,
        "actual_mean": float(actual.mean()),
        "prediction_mean": float(predicted.mean()),
    }
    return result, detail


def evaluate_holdout(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for target_col in TARGET_VARIANTS:
        for spec in ACTIONABLE_FEATURE_SPECS:
            for model_name in MODEL_BUILDERS:
                if (
                    model_name in {"Category mean", "Global mean"}
                    and spec.name != "preread_base"
                ):
                    continue
                result, details = evaluate_target_variant(
                    train_df=train_df,
                    holdout_df=holdout_df,
                    target_col=target_col,
                    spec=spec,
                    model_name=model_name,
                )
                if not result:
                    continue
                result_rows.append(result)
                prediction_rows.append(details)
    return pd.DataFrame(result_rows), pd.concat(prediction_rows, ignore_index=True)


def summarize_centering_effect(results: pd.DataFrame) -> pd.DataFrame:
    comparisons: list[dict[str, object]] = []
    for family, raw_target, centered_target in [
        ("enjoyment", "avg_enjoyment", "avg_enjoyment_centered"),
        ("usefulness", "avg_usefulness", "avg_usefulness_centered"),
    ]:
        raw = results[results["target"] == raw_target].copy()
        centered = results[results["target"] == centered_target].copy()
        merged = raw.merge(
            centered,
            on=["feature_spec", "model", "family"],
            suffixes=("_raw", "_centered"),
        )
        for _, row in merged.iterrows():
            comparisons.append(
                {
                    "family": family,
                    "feature_spec": row["feature_spec"],
                    "model": row["model"],
                    "raw_mae": row["mae_raw"],
                    "centered_mae": row["mae_centered"],
                    "mae_delta_centered_minus_raw": row["mae_centered"]
                    - row["mae_raw"],
                    "raw_rmse": row["rmse_raw"],
                    "centered_rmse": row["rmse_centered"],
                    "rmse_delta_centered_minus_raw": row["rmse_centered"]
                    - row["rmse_raw"],
                    "raw_spearman_rho": row["spearman_rho_raw"],
                    "centered_spearman_rho": row["spearman_rho_centered"],
                    "rho_delta_centered_minus_raw": row["spearman_rho_centered"]
                    - row["spearman_rho_raw"],
                }
            )
    return pd.DataFrame(comparisons).sort_values(
        ["family", "centered_mae", "feature_spec", "model"]
    )


def summarize_best_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_VARIANTS:
        sub = results[results["target"] == target].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "target": target,
                "selection": "best_mae",
                **sub.sort_values("mae").iloc[0].to_dict(),
            }
        )
        rows.append(
            {
                "target": target,
                "selection": "best_rank_correlation",
                **sub.sort_values("spearman_rho", ascending=False).iloc[0].to_dict(),
            }
        )
    return pd.DataFrame(rows)


def requested_utility(values: pd.Series, target_col: str) -> pd.Series:
    family = TARGET_FAMILIES[target_col]
    base = UTILITY_BASES[family]
    numeric = pd.to_numeric(values, errors="coerce")
    exponent = np.clip(numeric - 1, a_min=0, a_max=None)
    return pd.Series(np.power(base, exponent) - 1, index=numeric.index)


def assign_policy_split(df: pd.DataFrame) -> pd.DataFrame:
    split_df = df[df["goodreads_rating"].notna()].copy()
    split_df["date_finished"] = pd.to_datetime(
        split_df["date_finished"], format="mixed", errors="coerce"
    )
    split_df = split_df.sort_values(["date_finished", "title"]).reset_index(drop=True)
    split_df["date_finished"] = split_df["date_finished"].dt.strftime("%Y-%m-%d")
    midpoint = len(split_df) // 2
    split_df["policy_split"] = np.where(split_df.index < midpoint, "validation", "test")
    return split_df


def summarize_threshold_frame(
    frame: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> dict[str, float] | None:
    subset = frame[["prediction", target_col]].dropna().copy()
    subset["prediction"] = pd.to_numeric(subset["prediction"], errors="coerce")
    subset[target_col] = pd.to_numeric(subset[target_col], errors="coerce")
    subset = subset.dropna()
    if subset.empty:
        return None

    keep = subset[subset["prediction"] >= threshold].copy()
    skip = subset[subset["prediction"] < threshold].copy()
    if len(keep) < POLICY_MIN_SIDE_N or len(skip) < POLICY_MIN_SIDE_N:
        return None

    utility = requested_utility(subset[target_col], target_col)
    keep_utility = requested_utility(keep[target_col], target_col)
    skip_utility = requested_utility(skip[target_col], target_col)
    return {
        "threshold": threshold,
        "n_total": len(subset),
        "n_keep": len(keep),
        "n_skip": len(skip),
        "keep_share": len(keep) / len(subset),
        "overall_mean": float(subset[target_col].mean()),
        "overall_utility_mean": float(utility.mean()),
        "keep_mean": float(keep[target_col].mean()),
        "skip_mean": float(skip[target_col].mean()),
        "delta_keep_vs_all": float(keep[target_col].mean() - subset[target_col].mean()),
        "delta_keep_vs_skip": float(keep[target_col].mean() - skip[target_col].mean()),
        "keep_utility_mean": float(keep_utility.mean()),
        "skip_utility_mean": float(skip_utility.mean()),
        "delta_keep_utility_vs_all": float(keep_utility.mean() - utility.mean()),
        "delta_keep_utility_vs_skip": float(keep_utility.mean() - skip_utility.mean()),
    }


def select_policy_rows(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        summary = summarize_threshold_frame(frame, target_col, threshold)
        if summary:
            rows.append(summary)
    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return pd.DataFrame()

    best_rows = [
        {
            "selection": "best_validation_utility",
            **sweep.sort_values(
                ["delta_keep_utility_vs_all", "delta_keep_vs_all"],
                ascending=False,
            )
            .iloc[0]
            .to_dict(),
        }
    ]
    balanced = sweep[
        sweep["keep_share"].between(
            POLICY_BALANCED_KEEP_SHARE_RANGE[0],
            POLICY_BALANCED_KEEP_SHARE_RANGE[1],
        )
    ].copy()
    if not balanced.empty:
        best_rows.append(
            {
                "selection": "best_validation_balanced_utility",
                **balanced.sort_values(
                    ["delta_keep_utility_vs_all", "delta_keep_vs_all"],
                    ascending=False,
                )
                .iloc[0]
                .to_dict(),
            }
        )
    return pd.DataFrame(best_rows)


def apply_policy_threshold(
    frame: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> dict[str, float]:
    subset = frame[["prediction", target_col]].dropna().copy()
    subset["prediction"] = pd.to_numeric(subset["prediction"], errors="coerce")
    subset[target_col] = pd.to_numeric(subset[target_col], errors="coerce")
    subset = subset.dropna()
    if subset.empty:
        return {
            "n_total": 0,
            "n_keep": 0,
            "keep_share": 0.0,
            "overall_mean": float("nan"),
            "overall_utility_mean": float("nan"),
            "keep_mean": float("nan"),
            "delta_keep_vs_all": float("nan"),
            "keep_utility_mean": float("nan"),
            "delta_keep_utility_vs_all": float("nan"),
        }

    keep = subset[subset["prediction"] >= threshold].copy()
    utility = requested_utility(subset[target_col], target_col)
    if keep.empty:
        return {
            "n_total": len(subset),
            "n_keep": 0,
            "keep_share": 0.0,
            "overall_mean": float(subset[target_col].mean()),
            "overall_utility_mean": float(utility.mean()),
            "keep_mean": float("nan"),
            "delta_keep_vs_all": float("nan"),
            "keep_utility_mean": float("nan"),
            "delta_keep_utility_vs_all": float("nan"),
        }

    keep_utility = requested_utility(keep[target_col], target_col)
    return {
        "n_total": len(subset),
        "n_keep": len(keep),
        "keep_share": len(keep) / len(subset),
        "overall_mean": float(subset[target_col].mean()),
        "overall_utility_mean": float(utility.mean()),
        "keep_mean": float(keep[target_col].mean()),
        "delta_keep_vs_all": float(keep[target_col].mean() - subset[target_col].mean()),
        "keep_utility_mean": float(keep_utility.mean()),
        "delta_keep_utility_vs_all": float(keep_utility.mean() - utility.mean()),
    }


def evaluate_policy_rules(
    predictions: pd.DataFrame,
    split_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_split = predictions.merge(
        split_lookup[["title", "date_finished", "policy_split"]],
        on=["title", "date_finished"],
        how="inner",
    )
    predictions_split.to_csv(PREDICTIONS_SPLIT_CSV, index=False)

    split_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    for target_col in POLICY_TARGETS:
        target_predictions = predictions_split[
            predictions_split["target"] == target_col
        ]
        for (feature_spec, model), frame in target_predictions.groupby(
            ["feature_spec", "model"]
        ):
            validation = frame[frame["policy_split"] == "validation"].copy()
            test = frame[frame["policy_split"] == "test"].copy()
            selected_rows = select_policy_rows(validation, target_col)
            for _, selected in selected_rows.iterrows():
                threshold = float(selected["threshold"])
                test_metrics = apply_policy_threshold(test, target_col, threshold)
                split_rows.append(
                    {
                        "target": target_col,
                        "target_label": TARGET_LABELS[target_col],
                        "feature_spec": feature_spec,
                        "model": model,
                        "selection": selected["selection"],
                        "utility_base": UTILITY_BASES[TARGET_FAMILIES[target_col]],
                        "threshold": threshold,
                        "validation_n_total": int(selected["n_total"]),
                        "validation_n_keep": int(selected["n_keep"]),
                        "validation_keep_share": float(selected["keep_share"]),
                        "validation_keep_mean": float(selected["keep_mean"]),
                        "validation_delta_keep_vs_all": float(
                            selected["delta_keep_vs_all"]
                        ),
                        "validation_keep_utility_mean": float(
                            selected["keep_utility_mean"]
                        ),
                        "validation_delta_keep_utility_vs_all": float(
                            selected["delta_keep_utility_vs_all"]
                        ),
                        "test_n_total": int(test_metrics["n_total"]),
                        "test_n_keep": int(test_metrics["n_keep"]),
                        "test_keep_share": float(test_metrics["keep_share"]),
                        "test_keep_mean": float(test_metrics["keep_mean"]),
                        "test_delta_keep_vs_all": float(
                            test_metrics["delta_keep_vs_all"]
                        ),
                        "test_keep_utility_mean": float(
                            test_metrics["keep_utility_mean"]
                        ),
                        "test_delta_keep_utility_vs_all": float(
                            test_metrics["delta_keep_utility_vs_all"]
                        ),
                    }
                )

            full_selected = select_policy_rows(frame, target_col)
            for _, selected in full_selected.iterrows():
                full_rows.append(
                    {
                        "target": target_col,
                        "target_label": TARGET_LABELS[target_col],
                        "feature_spec": feature_spec,
                        "model": model,
                        "selection": selected["selection"].replace(
                            "validation", "full_holdout"
                        ),
                        "utility_base": UTILITY_BASES[TARGET_FAMILIES[target_col]],
                        "threshold": float(selected["threshold"]),
                        "n_total": int(selected["n_total"]),
                        "n_keep": int(selected["n_keep"]),
                        "keep_share": float(selected["keep_share"]),
                        "keep_mean": float(selected["keep_mean"]),
                        "delta_keep_vs_all": float(selected["delta_keep_vs_all"]),
                        "keep_utility_mean": float(selected["keep_utility_mean"]),
                        "delta_keep_utility_vs_all": float(
                            selected["delta_keep_utility_vs_all"]
                        ),
                    }
                )

    return pd.DataFrame(split_rows), pd.DataFrame(full_rows)


def summarize_best_policy_rules(
    split_results: pd.DataFrame,
    full_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_best_rows: list[dict[str, object]] = []
    for target_col in POLICY_TARGETS:
        sub = split_results[split_results["target"] == target_col].copy()
        if sub.empty:
            continue
        split_best_rows.append(
            {
                **sub.sort_values(
                    [
                        "validation_delta_keep_utility_vs_all",
                        "validation_delta_keep_vs_all",
                    ],
                    ascending=False,
                )
                .iloc[0]
                .to_dict(),
                "target": target_col,
                "selection": "best_validation_rule",
            }
        )
        split_best_rows.append(
            {
                **sub.sort_values(
                    ["test_delta_keep_utility_vs_all", "test_delta_keep_vs_all"],
                    ascending=False,
                )
                .iloc[0]
                .to_dict(),
                "target": target_col,
                "selection": "best_realized_test_rule",
            }
        )

    full_best_rows: list[dict[str, object]] = []
    for target_col in POLICY_TARGETS:
        sub = full_results[full_results["target"] == target_col].copy()
        if sub.empty:
            continue
        full_best_rows.append(
            {
                **sub.sort_values(
                    ["delta_keep_utility_vs_all", "delta_keep_vs_all"],
                    ascending=False,
                )
                .iloc[0]
                .to_dict(),
                "target": target_col,
                "selection": "best_full_holdout_rule",
            }
        )
    return pd.DataFrame(split_best_rows), pd.DataFrame(full_best_rows)


def print_summary(
    agreement: pd.DataFrame,
    centering: pd.DataFrame,
    best_results: pd.DataFrame,
    split_policy_best: pd.DataFrame,
    full_policy_best: pd.DataFrame,
) -> None:
    print("=" * 80)
    print("NEW BOOKS TO RATE 2026 ANALYSIS")
    print("=" * 80)
    print("\nPass agreement:")
    print(agreement.to_string(index=False))
    print("\nCentering comparison:")
    print(centering.to_string(index=False))
    print("\nBest results by target:")
    print(
        best_results[
            [
                "target",
                "selection",
                "feature_spec",
                "model",
                "mae",
                "rmse",
                "spearman_rho",
            ]
        ].to_string(index=False)
    )
    print("\nBest split policy rules:")
    if split_policy_best.empty:
        print("No split policy rows")
    else:
        print(
            split_policy_best[
                [
                    "target",
                    "selection",
                    "feature_spec",
                    "model",
                    "threshold",
                    "validation_delta_keep_vs_all",
                    "validation_delta_keep_utility_vs_all",
                    "test_n_keep",
                    "test_delta_keep_vs_all",
                    "test_delta_keep_utility_vs_all",
                ]
            ].to_string(index=False)
        )
    print("\nBest full-holdout policy rules:")
    if full_policy_best.empty:
        print("No full-holdout policy rows")
    else:
        print(
            full_policy_best[
                [
                    "target",
                    "selection",
                    "feature_spec",
                    "model",
                    "threshold",
                    "n_keep",
                    "delta_keep_vs_all",
                    "delta_keep_utility_vs_all",
                ]
            ].to_string(index=False)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=NEW_BOOKS_INPUT)
    parser.add_argument(
        "--enriched-input",
        type=Path,
        help="Use an existing Goodreads-enriched holdout CSV instead of refetching Goodreads.",
    )
    parser.add_argument("--force-goodreads-refresh", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = add_centered_targets(derive_analysis_columns(load_data()))
    if args.enriched_input:
        holdout = load_enriched_holdout(args.enriched_input)
    else:
        holdout = normalize_new_holdout(args.input)
        holdout["canonical_key"] = holdout["title"].map(
            lambda title: normalize_text(
                f"{clean_title_for_search(title, '')}::{title}"
            )
        )
        holdout = add_goodreads_to_holdout(
            holdout,
            output_path=NEW_BOOKS_GOODREADS,
            force_refresh=args.force_goodreads_refresh,
            verbose=args.verbose,
        )
    holdout = add_centered_targets(derive_analysis_columns(holdout))
    holdout.to_csv(NEW_BOOKS_ENRICHED, index=False)

    distributions = summarize_label_distributions(holdout)
    distributions.to_csv(LABEL_DISTRIBUTION_SUMMARY_CSV, index=False)

    agreement = summarize_pass_agreement(holdout)
    agreement.to_csv(LABEL_AGREEMENT_CSV, index=False)

    plot_label_distributions(holdout, LABEL_DISTRIBUTIONS_PNG)

    results, predictions = evaluate_holdout(train_df, holdout)
    results = results.sort_values(
        ["target", "mae", "rmse", "feature_spec", "model"]
    ).reset_index(drop=True)
    results.to_csv(PREDICTION_RESULTS_CSV, index=False)
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    centering = summarize_centering_effect(results)
    centering.to_csv(CENTERING_COMPARISON_CSV, index=False)

    best_results = summarize_best_results(results)
    best_results.to_csv(BEST_RESULTS_CSV, index=False)

    split_lookup = assign_policy_split(holdout)
    split_policy_results, full_holdout_policy_results = evaluate_policy_rules(
        predictions=predictions,
        split_lookup=split_lookup,
    )
    split_policy_results.to_csv(POLICY_SPLIT_RESULTS_CSV, index=False)
    full_holdout_policy_results.to_csv(POLICY_FULL_HOLDOUT_CSV, index=False)
    split_policy_best, full_policy_best = summarize_best_policy_rules(
        split_results=split_policy_results,
        full_results=full_holdout_policy_results,
    )
    split_policy_best.to_csv(POLICY_SPLIT_BEST_CSV, index=False)
    full_policy_best.to_csv(POLICY_FULL_HOLDOUT_BEST_CSV, index=False)

    print_summary(
        agreement,
        centering,
        best_results,
        split_policy_best,
        full_policy_best,
    )


if __name__ == "__main__":
    main()
