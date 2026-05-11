"""Separate prediction problems by category family on full holdout.

Outputs a 2x3 plot:
- row 1: enjoyment
- row 2: usefulness
- columns: Business/Histories/General, Fiction/Literature, Technical/Other

Each subplot shows train and full-holdout drop curves with cutoff labels.
"""

from __future__ import annotations

import os
import re
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

OUTPUT_DIR = Path(__file__).parent
PLOT_PATH = OUTPUT_DIR / "three_group_full_holdout_drop_curves_2x3.png"
SUMMARY_CSV = OUTPUT_DIR / "three_group_full_holdout_drop_curve_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "three_group_full_holdout_drop_curve_summary.md"

GROUP_ORDER = [
    "Business/Histories/General",
    "Fiction/Literature",
    "Technical/Other",
]
TARGET_SPECS = [
    ("avg_enjoyment", "Enjoyment"),
    ("avg_usefulness", "Usefulness"),
]
FEATURE_COLS = ["goodreads_z", "openlibrary_z", "amazon_z"]


def category_group(category: object) -> str:
    text = "" if pd.isna(category) else str(category).strip()
    if text in {"fiction", "Literature"}:
        return "Fiction/Literature"
    if text in {"Computer Science", "Machine Learning", "Math", "Unknown Shelf"}:
        return "Technical/Other"
    return "Business/Histories/General"


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_frame() -> pd.DataFrame:
    frame = load_frame()
    frame["group_name"] = frame["category"].map(category_group)
    train = frame[~frame["source"].eq("Holdout 2026")].copy()
    holdout = frame[frame["source"].eq("Holdout 2026")].copy()
    variant = Variant(aggregator="precision_weighted_z", missing_mode="drop", shrink=False)
    train_scored, _ = attach_zscores(train, train, variant)
    holdout_scored, _ = attach_zscores(train, holdout, variant)
    train_scored["dataset"] = "train"
    holdout_scored["dataset"] = "holdout"
    for frame_part in [train_scored, holdout_scored]:
        frame_part["all_sources_present"] = frame_part[FEATURE_COLS].notna().all(axis=1)
    return pd.concat([train_scored, holdout_scored], ignore_index=True)


def fit_group_model(group_train: pd.DataFrame, target_col: str) -> Ridge | None:
    fit_rows = group_train[
        group_train["all_sources_present"] & group_train[target_col].notna()
    ].copy()
    if len(fit_rows) < 8:
        return None
    X = fit_rows[FEATURE_COLS].to_numpy()
    y = pd.to_numeric(fit_rows[target_col], errors="coerce").to_numpy()
    return Ridge(alpha=1.0).fit(X, y)


def add_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    for target_col, _ in TARGET_SPECS:
        scored[f"pred_{target_col}"] = np.nan
    for group_name in GROUP_ORDER:
        group_train = scored[
            (scored["dataset"] == "train") & scored["group_name"].eq(group_name)
        ].copy()
        group_all = scored[scored["group_name"].eq(group_name)].copy()
        for target_col, _ in TARGET_SPECS:
            model = fit_group_model(group_train, target_col)
            if model is None:
                continue
            mask = group_all["all_sources_present"] & group_all[target_col].notna()
            if not mask.any():
                continue
            predictions = model.predict(group_all.loc[mask, FEATURE_COLS].to_numpy())
            scored.loc[group_all.loc[mask].index, f"pred_{target_col}"] = predictions
    return scored


def build_drop_curve(frame: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    valid = frame[frame[target_col].notna() & frame[pred_col].notna()].copy()
    if len(valid) < 8:
        return pd.DataFrame()
    valid = valid.sort_values([pred_col, "title"], ascending=[False, True]).reset_index(
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
                "drop_percent": drop_fraction * 100,
                "keep_n": keep_n,
                "mean_actual": float(pd.to_numeric(kept[target_col], errors="coerce").mean()),
                "cutoff_pred": float(pd.to_numeric(kept[pred_col], errors="coerce").iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def annotate_curve(axis: plt.Axes, curve: pd.DataFrame, y_col: str, color: str) -> None:
    for _, row in curve.iterrows():
        axis.annotate(
            f"{row['cutoff_pred']:.2f}",
            (row["drop_percent"], row[y_col]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color=color,
        )


def plot_curves(scored: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    curve_store: dict[tuple[str, str, str], pd.DataFrame] = {}
    y_limits: dict[str, list[float]] = {target: [] for target, _ in TARGET_SPECS}

    for group_name in GROUP_ORDER:
        group_rows = scored[scored["group_name"].eq(group_name)].copy()
        for target_col, _ in TARGET_SPECS:
            pred_col = f"pred_{target_col}"
            for dataset in ["train", "holdout"]:
                subset = group_rows[group_rows["dataset"].eq(dataset)].copy()
                curve = build_drop_curve(subset, target_col, pred_col)
                curve_store[(group_name, target_col, dataset)] = curve
                if not curve.empty:
                    y_limits[target_col].extend(curve["mean_actual"].tolist())
                eval_rows = subset[subset[target_col].notna() & subset[pred_col].notna()].copy()
                corr = correlation(
                    pd.to_numeric(eval_rows[pred_col], errors="coerce").to_numpy(),
                    pd.to_numeric(eval_rows[target_col], errors="coerce").to_numpy(),
                )
                summary_rows.append(
                    {
                        "group_name": group_name,
                        "target": target_col,
                        "dataset": dataset,
                        "n_eval": int(len(eval_rows)),
                        "correlation": corr,
                    }
                )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True, sharey="row")
    colors = {"train": "#1f77b4", "holdout": "#d62728"}
    labels = {"train": "Train", "holdout": "Holdout 2026"}

    for row_idx, (target_col, target_label) in enumerate(TARGET_SPECS):
        y_values = y_limits[target_col]
        if y_values:
            y_min = min(y_values)
            y_max = max(y_values)
            pad = max(0.05, (y_max - y_min) * 0.1)
            shared_ylim = (y_min - pad, y_max + pad)
        else:
            shared_ylim = (1.0, 5.0)

        for col_idx, group_name in enumerate(GROUP_ORDER):
            axis = axes[row_idx, col_idx]
            subset_summary = pd.DataFrame(summary_rows)
            title_bits = []
            for dataset in ["train", "holdout"]:
                row = subset_summary[
                    (subset_summary["group_name"] == group_name)
                    & (subset_summary["target"] == target_col)
                    & (subset_summary["dataset"] == dataset)
                ].iloc[0]
                corr_text = "nan" if pd.isna(row["correlation"]) else f"{row['correlation']:.2f}"
                title_bits.append(f"{labels[dataset]} R={corr_text} n={int(row['n_eval'])}")

            for dataset in ["train", "holdout"]:
                curve = curve_store[(group_name, target_col, dataset)]
                if curve.empty:
                    continue
                axis.plot(
                    curve["drop_percent"],
                    curve["mean_actual"],
                    marker="o",
                    linewidth=2,
                    color=colors[dataset],
                    label=labels[dataset],
                )
                annotate_curve(axis, curve, "mean_actual", colors[dataset])

            axis.set_title(f"{group_name}\n" + " | ".join(title_bits), fontsize=10)
            axis.set_xlabel("% dropped")
            axis.set_ylim(*shared_ylim)
            axis.grid(alpha=0.2)
            if col_idx == 0:
                axis.set_ylabel(f"{target_label}\nmean kept actual")
            if row_idx == 0:
                axis.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Separate prediction problems by category family on full holdout",
        fontsize=16,
    )
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def build_summary_markdown(summary_df: pd.DataFrame) -> str:
    lines = ["# Three-group full-holdout drop curve summary", ""]
    for group_name in GROUP_ORDER:
        lines.append(f"## {group_name}")
        for target_col, target_label in TARGET_SPECS:
            train_row = summary_df[
                (summary_df["group_name"] == group_name)
                & (summary_df["target"] == target_col)
                & (summary_df["dataset"] == "train")
            ].iloc[0]
            holdout_row = summary_df[
                (summary_df["group_name"] == group_name)
                & (summary_df["target"] == target_col)
                & (summary_df["dataset"] == "holdout")
            ].iloc[0]
            train_corr = "nan" if pd.isna(train_row["correlation"]) else f"{train_row['correlation']:.3f}"
            holdout_corr = "nan" if pd.isna(holdout_row["correlation"]) else f"{holdout_row['correlation']:.3f}"
            lines.append(
                f"- {target_label}: train `R={train_corr}` (`n={int(train_row['n_eval'])}`), "
                f"holdout `R={holdout_corr}` (`n={int(holdout_row['n_eval'])}`)"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    scored = add_predictions(build_frame())
    summary_df = plot_curves(scored)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_MD.write_text(build_summary_markdown(summary_df))
    print(f"Saved {PLOT_PATH.name}")
    print(f"Saved {SUMMARY_CSV.name}")
    print(f"Saved {SUMMARY_MD.name}")


if __name__ == "__main__":
    main()
