"""Compare grouped full-holdout approaches on the same data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ai_books_tracking.multi_source_selection_policy import (
    DROP_FRACTIONS,
    Variant,
    attach_zscores,
    load_frame,
)
from ai_books_tracking.practical_group_rules import (
    FEATURE_COLS as RAW_FEATURE_COLS,
    fit_imputation_models,
    impute_group_sources,
)
from ai_books_tracking.sequential_group_missingness_model import (
    SequentialModel,
    fit_sequential_model,
)

OUTPUT_DIR = Path(__file__).parent
HOLDOUT_YEAR_TEXT = os.environ.get("HOLDOUT_YEAR", "").strip()
HOLDOUT_YEAR = int(HOLDOUT_YEAR_TEXT) if HOLDOUT_YEAR_TEXT else None


def output_path(stem: str, suffix: str) -> Path:
    if HOLDOUT_YEAR is None:
        return OUTPUT_DIR / f"{stem}_2x4.{suffix}"
    return OUTPUT_DIR / f"{stem}_{HOLDOUT_YEAR}_holdout_2x4.{suffix}"


TRAIN_PLOT = output_path("group_approach_comparison_train", "png")
HOLDOUT_PLOT = output_path("group_approach_comparison_holdout", "png")
SUMMARY_CSV = output_path("group_approach_comparison_summary", "csv")
COMMON_CSV = output_path("group_approach_comparison_common_subset", "csv")
AGGREGATE_CSV = output_path("group_approach_comparison_aggregate_changes", "csv")
SUMMARY_MD = output_path("group_approach_comparison", "md")

GROUP_ORDER = [
    "Business/Histories/General",
    "Fiction",
    "Literature",
    "Technical/Other",
]
TARGET_SPECS = [
    ("avg_enjoyment", "Enjoyment"),
    ("avg_usefulness", "Usefulness"),
]
KEEP_SHARES = [1.0, 0.4, 0.3, 0.2]
METHOD_LABELS = {
    "zscore_complete_case": "Z-score complete-case",
    "raw_ridge_regression_impute": "Raw ridge + regression impute",
    "sequential_missingness": "Sequential missingness",
}
METHOD_COLORS = {
    "zscore_complete_case": "#1f77b4",
    "raw_ridge_regression_impute": "#2ca02c",
    "sequential_missingness": "#d62728",
}
Z_FEATURE_COLS = ["goodreads_z", "openlibrary_z", "amazon_z"]


def category_group(category: object) -> str:
    text = "" if pd.isna(category) else str(category).strip()
    if text == "fiction":
        return "Fiction"
    if text == "Literature":
        return "Literature"
    if text in {"Computer Science", "Machine Learning", "Math", "Unknown Shelf"}:
        return "Technical/Other"
    return "Business/Histories/General"


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def centered_or_zero(series: pd.Series, mean_value: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric - mean_value).fillna(0.0)


def build_base_frame() -> pd.DataFrame:
    frame = load_frame().copy()
    frame["estimated_finish"] = pd.to_datetime(frame["estimated_finish"], errors="coerce")
    frame["year"] = frame["estimated_finish"].dt.year
    if HOLDOUT_YEAR is not None:
        frame = frame[frame["year"].notna()].copy()
    frame["row_id"] = np.arange(len(frame))
    frame["group_name"] = frame["category"].map(category_group)
    if HOLDOUT_YEAR is None:
        frame["dataset"] = np.where(frame["source"].eq("Holdout 2026"), "holdout", "train")
    else:
        frame["dataset"] = np.where(frame["year"].eq(HOLDOUT_YEAR), "holdout", "train")
    return frame


def build_drop_curve(frame: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    valid = frame[frame[target_col].notna() & frame[pred_col].notna()].copy()
    if len(valid) < 8:
        return pd.DataFrame()
    valid = valid.sort_values([pred_col, "title"], ascending=[False, True]).reset_index(drop=True)
    rows: list[dict[str, float]] = []
    for drop_fraction in DROP_FRACTIONS:
        drop_n = int(np.floor(len(valid) * drop_fraction))
        keep_n = len(valid) - drop_n
        if keep_n < 5:
            continue
        kept = valid.iloc[:keep_n].copy()
        rows.append(
            {
                "drop_percent": drop_fraction * 100.0,
                "mean_actual": float(pd.to_numeric(kept[target_col], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(method_name: str, scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name in GROUP_ORDER:
        for target_col, _ in TARGET_SPECS:
            pred_col = f"pred_{target_col}"
            for dataset in ["train", "holdout"]:
                subset = scored[
                    scored["group_name"].eq(group_name) & scored["dataset"].eq(dataset)
                ].copy()
                eval_rows = subset[subset[target_col].notna() & subset[pred_col].notna()].copy()
                corr = correlation(
                    pd.to_numeric(eval_rows[pred_col], errors="coerce").to_numpy(),
                    pd.to_numeric(eval_rows[target_col], errors="coerce").to_numpy(),
                )
                rows.append(
                    {
                        "method": method_name,
                        "group_name": group_name,
                        "target": target_col,
                        "dataset": dataset,
                        "n_eval": int(len(eval_rows)),
                        "correlation": corr,
                    }
                )
    return pd.DataFrame(rows)


def score_zscore_complete_case(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["dataset"] == "train"].copy()
    holdout = frame[frame["dataset"] == "holdout"].copy()
    variant = Variant(aggregator="precision_weighted_z", missing_mode="drop", shrink=False)
    train_scored, _ = attach_zscores(train, train, variant)
    holdout_scored, _ = attach_zscores(train, holdout, variant)
    scored = pd.concat([train_scored, holdout_scored], ignore_index=True)
    scored["all_sources_present"] = scored[Z_FEATURE_COLS].notna().all(axis=1)

    for target_col, _ in TARGET_SPECS:
        scored[f"pred_{target_col}"] = np.nan

    for group_name in GROUP_ORDER:
        group_train = scored[
            (scored["dataset"] == "train") & scored["group_name"].eq(group_name)
        ].copy()
        group_all = scored[scored["group_name"].eq(group_name)].copy()
        for target_col, _ in TARGET_SPECS:
            fit_rows = group_train[
                group_train["all_sources_present"] & group_train[target_col].notna()
            ].copy()
            if len(fit_rows) < 8:
                continue
            model = Ridge(alpha=1.0).fit(
                fit_rows[Z_FEATURE_COLS].to_numpy(),
                pd.to_numeric(fit_rows[target_col], errors="coerce").to_numpy(),
            )
            mask = group_all["all_sources_present"] & group_all[target_col].notna()
            if not mask.any():
                continue
            predictions = model.predict(group_all.loc[mask, Z_FEATURE_COLS].to_numpy())
            scored.loc[group_all.loc[mask].index, f"pred_{target_col}"] = predictions

    scored["pred_balanced"] = (
        scored["pred_avg_enjoyment"] + scored["pred_avg_usefulness"]
    ) / 2.0
    return scored


def score_raw_ridge_regression_impute(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    for target_col, _ in TARGET_SPECS:
        scored[f"pred_{target_col}"] = np.nan

    for group_name in GROUP_ORDER:
        group_train = scored[
            (scored["dataset"] == "train") & scored["group_name"].eq(group_name)
        ].copy()
        group_holdout = scored[
            (scored["dataset"] == "holdout") & scored["group_name"].eq(group_name)
        ].copy()
        imputation_models, fill_values = fit_imputation_models(group_train)
        train_imputed = impute_group_sources(group_train, imputation_models, fill_values)
        holdout_imputed = impute_group_sources(group_holdout, imputation_models, fill_values)

        for target_col, _ in TARGET_SPECS:
            fit_train = train_imputed[train_imputed[target_col].notna()].copy()
            if len(fit_train) < 8:
                continue
            model = Ridge(alpha=1.0).fit(
                fit_train[RAW_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(),
                pd.to_numeric(fit_train[target_col], errors="coerce").to_numpy(),
            )
            for dataset_name, dataset_frame in [
                ("train", train_imputed),
                ("holdout", holdout_imputed),
            ]:
                eval_rows = dataset_frame[dataset_frame[target_col].notna()].copy()
                if eval_rows.empty:
                    continue
                predictions = model.predict(
                    eval_rows[RAW_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
                )
                mask = scored["row_id"].isin(eval_rows["row_id"]) & scored["dataset"].eq(dataset_name)
                scored.loc[mask, f"pred_{target_col}"] = predictions

    scored["pred_balanced"] = (
        scored["pred_avg_enjoyment"] + scored["pred_avg_usefulness"]
    ) / 2.0
    return scored


def predict_sequential_subset(
    subset: pd.DataFrame, model: SequentialModel, target_col: str
) -> pd.Series:
    predictions = pd.Series(np.nan, index=subset.index, dtype=float)
    mask = subset["goodreads_rating_raw"].notna() & subset[target_col].notna()
    if not mask.any():
        return predictions
    gr = pd.to_numeric(subset.loc[mask, "goodreads_rating_raw"], errors="coerce")
    amz_term = centered_or_zero(subset.loc[mask, "amazon_rating_raw"], model.amazon_mean)
    ol_term = centered_or_zero(
        subset.loc[mask, "openlibrary_rating_raw"], model.openlibrary_mean
    )
    predictions.loc[mask] = (
        model.intercept
        + model.beta_gr * gr
        + model.beta_amz * amz_term
        + model.beta_ol * ol_term
    )
    return predictions


def score_sequential_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    for target_col, _ in TARGET_SPECS:
        scored[f"pred_{target_col}"] = np.nan

    for group_name in GROUP_ORDER:
        group_train = scored[
            (scored["dataset"] == "train") & scored["group_name"].eq(group_name)
        ].copy()
        group_holdout = scored[
            (scored["dataset"] == "holdout") & scored["group_name"].eq(group_name)
        ].copy()
        for target_col, _ in TARGET_SPECS:
            model = fit_sequential_model(group_train, target_col)
            if model is None:
                continue
            for dataset_name, dataset_frame in [
                ("train", group_train),
                ("holdout", group_holdout),
            ]:
                predictions = predict_sequential_subset(dataset_frame, model, target_col)
                mask = scored["row_id"].isin(dataset_frame["row_id"]) & scored["dataset"].eq(
                    dataset_name
                )
                scored.loc[mask, f"pred_{target_col}"] = predictions.values

    scored["pred_balanced"] = (
        scored["pred_avg_enjoyment"] + scored["pred_avg_usefulness"]
    ) / 2.0
    return scored


def build_common_subset(
    predictions_by_method: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[int]]:
    common_ids: set[int] | None = None
    for scored in predictions_by_method.values():
        available_ids = set(
            scored[
                (scored["dataset"] == "holdout")
                & scored["pred_balanced"].notna()
                & scored["avg_enjoyment"].notna()
                & scored["avg_usefulness"].notna()
            ]["row_id"].tolist()
        )
        common_ids = available_ids if common_ids is None else common_ids & available_ids
    common_id_list = sorted(common_ids or set())

    rows: list[dict[str, object]] = []
    for method_name, scored in predictions_by_method.items():
        common = scored[
            scored["row_id"].isin(common_id_list)
            & scored["dataset"].eq("holdout")
            & scored["avg_enjoyment"].notna()
            & scored["avg_usefulness"].notna()
        ].copy()
        for group_name in GROUP_ORDER:
            for target_col, _ in TARGET_SPECS:
                pred_col = f"pred_{target_col}"
                subset = common[common["group_name"].eq(group_name)].copy()
                eval_rows = subset[subset[pred_col].notna() & subset[target_col].notna()].copy()
                corr = correlation(
                    pd.to_numeric(eval_rows[pred_col], errors="coerce").to_numpy(),
                    pd.to_numeric(eval_rows[target_col], errors="coerce").to_numpy(),
                )
                rows.append(
                    {
                        "method": method_name,
                        "group_name": group_name,
                        "target": target_col,
                        "dataset": "holdout_common",
                        "n_eval": int(len(eval_rows)),
                        "correlation": corr,
                    }
                )
    return pd.DataFrame(rows), common_id_list


def aggregate_changes(
    method_name: str, scored: pd.DataFrame, holdout_ids: list[int]
) -> pd.DataFrame:
    holdout = scored[
        scored["row_id"].isin(holdout_ids)
        & scored["dataset"].eq("holdout")
        & scored["pred_balanced"].notna()
        & scored["avg_enjoyment"].notna()
        & scored["avg_usefulness"].notna()
    ].copy()
    holdout = holdout.sort_values(["pred_balanced", "title"], ascending=[False, True]).reset_index(
        drop=True
    )
    if holdout.empty:
        return pd.DataFrame()

    baseline_enjoy = float(pd.to_numeric(holdout["avg_enjoyment"], errors="coerce").mean())
    baseline_useful = float(pd.to_numeric(holdout["avg_usefulness"], errors="coerce").mean())
    baseline_balanced = float(((holdout["avg_enjoyment"] + holdout["avg_usefulness"]) / 2.0).mean())
    rows: list[dict[str, object]] = []

    for keep_share in KEEP_SHARES:
        if keep_share >= 1.0:
            kept = holdout.copy()
        else:
            keep_n = max(1, int(round(len(holdout) * keep_share)))
            kept = holdout.iloc[:keep_n].copy()
        avg_enjoy = float(pd.to_numeric(kept["avg_enjoyment"], errors="coerce").mean())
        avg_useful = float(pd.to_numeric(kept["avg_usefulness"], errors="coerce").mean())
        avg_balanced = float(((kept["avg_enjoyment"] + kept["avg_usefulness"]) / 2.0).mean())
        rows.append(
            {
                "method": method_name,
                "keep_share": keep_share,
                "keep_n": int(len(kept)),
                "eval_n": int(len(holdout)),
                "avg_enjoyment": avg_enjoy,
                "avg_usefulness": avg_useful,
                "avg_balanced": avg_balanced,
                "enjoyment_gain": avg_enjoy - baseline_enjoy,
                "usefulness_gain": avg_useful - baseline_useful,
                "balanced_gain": avg_balanced - baseline_balanced,
            }
        )
    return pd.DataFrame(rows)


def plot_method_comparison(
    predictions_by_method: dict[str, pd.DataFrame], dataset_name: str, output_path: Path
) -> None:
    y_values: dict[str, list[float]] = {target: [] for target, _ in TARGET_SPECS}
    curve_store: dict[tuple[str, str, str], pd.DataFrame] = {}

    for method_name, scored in predictions_by_method.items():
        for group_name in GROUP_ORDER:
            subset = scored[
                scored["group_name"].eq(group_name) & scored["dataset"].eq(dataset_name)
            ].copy()
            for target_col, _ in TARGET_SPECS:
                curve = build_drop_curve(subset, target_col, f"pred_{target_col}")
                curve_store[(method_name, group_name, target_col)] = curve
                if not curve.empty:
                    y_values[target_col].extend(curve["mean_actual"].tolist())

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True, sharey="row")
    for row_idx, (target_col, target_label) in enumerate(TARGET_SPECS):
        values = y_values[target_col]
        if values:
            y_min = min(values)
            y_max = max(values)
            pad = max(0.05, (y_max - y_min) * 0.1)
            ylim = (y_min - pad, y_max + pad)
        else:
            ylim = (1.0, 5.0)

        for col_idx, group_name in enumerate(GROUP_ORDER):
            axis = axes[row_idx, col_idx]
            for method_name in METHOD_LABELS:
                curve = curve_store[(method_name, group_name, target_col)]
                if curve.empty:
                    continue
                axis.plot(
                    curve["drop_percent"],
                    curve["mean_actual"],
                    marker="o",
                    linewidth=2,
                    label=METHOD_LABELS[method_name],
                    color=METHOD_COLORS[method_name],
                )
            axis.set_title(group_name, fontsize=11)
            axis.set_xlabel("% dropped")
            axis.set_ylim(*ylim)
            axis.grid(alpha=0.2)
            if col_idx == 0:
                axis.set_ylabel(f"{target_label}\nmean kept actual")
            if row_idx == 0 and col_idx == 0:
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(frameon=False, fontsize=9)

    fig.suptitle(
        (
            f"Grouped approach comparison | {dataset_name.title()} curves"
            if HOLDOUT_YEAR is None
            else f"Grouped approach comparison | {dataset_name.title()} curves | {HOLDOUT_YEAR} holdout"
        ),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_summary_markdown(
    summary_df: pd.DataFrame, common_df: pd.DataFrame, aggregate_df: pd.DataFrame
) -> str:
    title = "# Group Approach Comparison"
    if HOLDOUT_YEAR is not None:
        title = f"# Group Approach Comparison ({HOLDOUT_YEAR} Holdout)"
    lines = [title, ""]

    lines.append("## Native Coverage Holdout R")
    for group_name in GROUP_ORDER:
        lines.append(f"### {group_name}")
        for target_col, target_label in TARGET_SPECS:
            lines.append(f"- {target_label}:")
            subset = summary_df[
                (summary_df["group_name"] == group_name)
                & (summary_df["target"] == target_col)
                & (summary_df["dataset"] == "holdout")
            ].copy()
            for row in subset.itertuples(index=False):
                corr_text = "nan" if pd.isna(row.correlation) else f"{row.correlation:.3f}"
                lines.append(
                    f"  {METHOD_LABELS[row.method]}: `R={corr_text}` (`n={int(row.n_eval)}`)"
                )
        lines.append("")

    lines.append("## Common-Subset Holdout R")
    lines.append(
        "All methods are compared on the exact same holdout rows where every method has a balanced prediction."
    )
    lines.append("")
    for group_name in GROUP_ORDER:
        lines.append(f"### {group_name}")
        for target_col, target_label in TARGET_SPECS:
            lines.append(f"- {target_label}:")
            subset = common_df[
                (common_df["group_name"] == group_name) & (common_df["target"] == target_col)
            ].copy()
            for row in subset.itertuples(index=False):
                corr_text = "nan" if pd.isna(row.correlation) else f"{row.correlation:.3f}"
                lines.append(
                    f"  {METHOD_LABELS[row.method]}: `R={corr_text}` (`n={int(row.n_eval)}`)"
                )
        lines.append("")

    lines.append("## Common-Subset Aggregate Gains")
    for keep_share in KEEP_SHARES:
        subset = aggregate_df[aggregate_df["keep_share"] == keep_share].copy()
        keep_label = f"{int(keep_share * 100)}%" if keep_share < 1.0 else "100%"
        lines.append(f"### Keep {keep_label}")
        for row in subset.itertuples(index=False):
            lines.append(
                f"- {METHOD_LABELS[row.method]}: "
                f"enjoy `{row.enjoyment_gain:+.3f}`, useful `{row.usefulness_gain:+.3f}`, "
                f"balanced `{row.balanced_gain:+.3f}` (`keep_n={int(row.keep_n)}`, `eval_n={int(row.eval_n)}`)"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    frame = build_base_frame()
    predictions_by_method = {
        "zscore_complete_case": score_zscore_complete_case(frame),
        "raw_ridge_regression_impute": score_raw_ridge_regression_impute(frame),
        "sequential_missingness": score_sequential_missingness(frame),
    }

    summary_df = pd.concat(
        [
            summarize_predictions(method_name, scored)
            for method_name, scored in predictions_by_method.items()
        ],
        ignore_index=True,
    )
    common_df, common_holdout_ids = build_common_subset(predictions_by_method)
    aggregate_df = pd.concat(
        [
            aggregate_changes(method_name, scored, common_holdout_ids)
            for method_name, scored in predictions_by_method.items()
        ],
        ignore_index=True,
    )

    plot_method_comparison(predictions_by_method, "train", TRAIN_PLOT)
    plot_method_comparison(predictions_by_method, "holdout", HOLDOUT_PLOT)

    SUMMARY_CSV.write_text(summary_df.to_csv(index=False))
    COMMON_CSV.write_text(common_df.to_csv(index=False))
    AGGREGATE_CSV.write_text(aggregate_df.to_csv(index=False))
    SUMMARY_MD.write_text(build_summary_markdown(summary_df, common_df, aggregate_df) + "\n")

    print(f"Saved {TRAIN_PLOT.name}")
    print(f"Saved {HOLDOUT_PLOT.name}")
    print(f"Saved {SUMMARY_CSV.name}")
    print(f"Saved {COMMON_CSV.name}")
    print(f"Saved {AGGREGATE_CSV.name}")
    print(f"Saved {SUMMARY_MD.name}")


if __name__ == "__main__":
    main()
