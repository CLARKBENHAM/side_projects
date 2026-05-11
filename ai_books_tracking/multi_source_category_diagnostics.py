"""Category-level diagnostics for holdout predictions and source missingness."""

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

from ai_books_tracking.goodreads_followup_analysis import spearman_summary

OUTPUT_DIR = Path(__file__).parent
MODEL_SUMMARY_CSV = OUTPUT_DIR / "multi_source_rule_model_summary.csv"
HOLDOUT_SCORES_CSV = OUTPUT_DIR / "multi_source_rule_holdout_scores.csv"
MERGED_CSV = OUTPUT_DIR / "golden_master_multi_source.csv"

CATEGORY_CORR_CSV = OUTPUT_DIR / "multi_source_holdout_category_correlations.csv"
MISSING_COUNTS_CSV = OUTPUT_DIR / "multi_source_missing_counts_by_year_category.csv"
PREDICTED_ACTUAL_ENJOYMENT_PLOT = (
    OUTPUT_DIR / "multi_source_predicted_vs_actual_by_category_enjoyment.png"
)
PREDICTED_ACTUAL_USEFULNESS_PLOT = (
    OUTPUT_DIR / "multi_source_predicted_vs_actual_by_category_usefulness.png"
)
GOODREADS_MISSING_PLOT = (
    OUTPUT_DIR / "multi_source_missing_goodreads_by_year_category.png"
)
OPENLIBRARY_MISSING_PLOT = (
    OUTPUT_DIR / "multi_source_missing_openlibrary_by_year_category.png"
)
AMAZON_MISSING_PLOT = OUTPUT_DIR / "multi_source_missing_amazon_by_year_category.png"

TARGET_LABELS = {
    "avg_enjoyment": "Average enjoyment",
    "avg_usefulness": "Average usefulness",
}
SOURCE_COLUMNS = {
    "goodreads": "goodreads_rating_verified",
    "openlibrary": "ol_rating_consensus",
    "amazon": "amazon_rating_consensus",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(MODEL_SUMMARY_CSV),
        pd.read_csv(HOLDOUT_SCORES_CSV),
        pd.read_csv(MERGED_CSV),
    )


def best_complete_case_rule(summary: pd.DataFrame, target_col: str) -> pd.Series:
    subset = summary[
        (summary["target"] == target_col) & (summary["mode"] == "complete_case")
    ].copy()
    if subset.empty:
        raise ValueError(f"No complete-case rows for {target_col}")
    return subset.sort_values(["spearman_rho", "mae"], ascending=[False, True]).iloc[0]


def category_correlation_rows(
    holdout_scores: pd.DataFrame, target_col: str, rule_name: str, mode: str
) -> pd.DataFrame:
    subset = holdout_scores[
        (holdout_scores["target"] == target_col)
        & (holdout_scores["rule_name"] == rule_name)
        & (holdout_scores["mode"] == mode)
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for category, category_frame in subset.groupby("category", dropna=False):
        actual = pd.to_numeric(category_frame[target_col], errors="coerce")
        predicted = pd.to_numeric(category_frame["score"], errors="coerce")
        mask = actual.notna() & predicted.notna()
        rho, pvalue = spearman_summary(actual[mask], predicted[mask])
        rows.append(
            {
                "target": target_col,
                "rule_name": rule_name,
                "mode": mode,
                "category": category,
                "n": int(mask.sum()),
                "spearman_rho": float(rho),
                "spearman_p": float(pvalue),
                "actual_mean": float(actual[mask].mean()) if mask.any() else math.nan,
                "predicted_mean": (
                    float(predicted[mask].mean()) if mask.any() else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "category"], ascending=[False, True])


def plot_predicted_vs_actual_small_multiples(
    holdout_scores: pd.DataFrame,
    target_col: str,
    rule_name: str,
    mode: str,
    output_path: Path,
) -> pd.DataFrame:
    subset = holdout_scores[
        (holdout_scores["target"] == target_col)
        & (holdout_scores["rule_name"] == rule_name)
        & (holdout_scores["mode"] == mode)
    ].copy()
    categories = (
        subset["category"]
        .fillna("NA")
        .value_counts()
        .sort_values(ascending=False)
        .index
    )
    if len(categories) == 0:
        fig, _ = plt.subplots(figsize=(8, 6))
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return pd.DataFrame()

    ncols = 3
    nrows = int(np.ceil(len(categories) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).flatten()

    summary = category_correlation_rows(holdout_scores, target_col, rule_name, mode)
    summary_map = summary.set_index("category") if not summary.empty else pd.DataFrame()

    for axis, category in zip(axes, categories, strict=False):
        cat_subset = subset[subset["category"].fillna("NA") == category].copy()
        actual = pd.to_numeric(cat_subset[target_col], errors="coerce")
        predicted = pd.to_numeric(cat_subset["score"], errors="coerce")
        mask = actual.notna() & predicted.notna()
        axis.scatter(actual[mask], predicted[mask], s=36, alpha=0.75, color="#0B3954")
        axis.plot([1, 5], [1, 5], linestyle="--", color="black", linewidth=1)
        axis.set_xlim(1, 5)
        axis.set_ylim(1, 5)
        axis.set_title(str(category))
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.grid(alpha=0.15)
        if category in summary_map.index:
            row = summary_map.loc[category]
            axis.text(
                0.03,
                0.97,
                f"n={int(row['n'])}\nrho={row['spearman_rho']:.2f}",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    for axis in axes[len(categories) :]:
        axis.axis("off")

    fig.suptitle(
        f"{TARGET_LABELS[target_col]} | {rule_name} ({mode})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def missing_counts_by_year_category(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged.copy()
    frame["estimated_finish"] = pd.to_datetime(
        frame["estimated_finish"], format="mixed", errors="coerce"
    )
    frame["year"] = frame["estimated_finish"].dt.year
    frame["year_label"] = frame["year"].map(
        lambda value: str(int(value)) if pd.notna(value) else "Unknown"
    )
    rows: list[dict[str, object]] = []
    for source_name, column in SOURCE_COLUMNS.items():
        grouped = (
            frame.groupby(["year_label", "category"], dropna=False)[column]
            .agg(total_count="size", missing_count=lambda s: int(s.isna().sum()))
            .reset_index()
        )
        grouped["source"] = source_name
        grouped["missing_share"] = grouped["missing_count"] / grouped["total_count"]
        rows.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(rows)


def plot_missing_counts_heatmap(
    missing_df: pd.DataFrame, source_name: str, output_path: Path
) -> None:
    subset = missing_df[missing_df["source"] == source_name].copy()
    if subset.empty:
        fig, _ = plt.subplots(figsize=(8, 6))
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    pivot = (
        subset.pivot(index="category", columns="year_label", values="missing_count")
        .fillna(0)
        .sort_index()
    )
    years = list(pivot.columns)
    categories = list(pivot.index)

    fig, ax = plt.subplots(
        figsize=(1.4 * max(len(years), 5), 0.6 * max(len(categories), 6))
    )
    image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("Year")
    ax.set_ylabel("Category")
    ax.set_title(f"Missing {source_name.title()} ratings by year/category")
    for i in range(len(categories)):
        for j in range(len(years)):
            value = int(pivot.iloc[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(image, ax=ax, shrink=0.85, label="Missing count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run() -> None:
    summary, holdout_scores, merged = load_inputs()

    category_frames: list[pd.DataFrame] = []
    for target_col, plot_path in [
        ("avg_enjoyment", PREDICTED_ACTUAL_ENJOYMENT_PLOT),
        ("avg_usefulness", PREDICTED_ACTUAL_USEFULNESS_PLOT),
    ]:
        best = best_complete_case_rule(summary, target_col)
        category_summary = plot_predicted_vs_actual_small_multiples(
            holdout_scores=holdout_scores,
            target_col=target_col,
            rule_name=str(best["rule_name"]),
            mode=str(best["mode"]),
            output_path=plot_path,
        )
        if not category_summary.empty:
            category_frames.append(category_summary)

    category_df = (
        pd.concat(category_frames, ignore_index=True)
        if category_frames
        else pd.DataFrame()
    )
    category_df.to_csv(CATEGORY_CORR_CSV, index=False)

    missing_df = missing_counts_by_year_category(merged)
    missing_df.to_csv(MISSING_COUNTS_CSV, index=False)
    plot_missing_counts_heatmap(missing_df, "goodreads", GOODREADS_MISSING_PLOT)
    plot_missing_counts_heatmap(missing_df, "openlibrary", OPENLIBRARY_MISSING_PLOT)
    plot_missing_counts_heatmap(missing_df, "amazon", AMAZON_MISSING_PLOT)

    print(f"Saved category correlations to {CATEGORY_CORR_CSV}")
    print(f"Saved missing counts to {MISSING_COUNTS_CSV}")


if __name__ == "__main__":
    run()
