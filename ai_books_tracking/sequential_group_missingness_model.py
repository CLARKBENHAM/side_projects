"""Sequential residual model by category group.

Stage 1: fit target ~ intercept + beta_gr * Goodreads
Stage 2: fit residuals ~ beta_amz * centered Amazon (no intercept), only where Amazon exists
Stage 3: fit residuals ~ beta_ol * centered Open Library (no intercept), only where OL exists

Rows are only dropped if Goodreads is missing.
Missing Amazon/Open Library contribute 0 after mean-centering.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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
from sklearn.linear_model import LinearRegression

from ai_books_tracking.multi_source_selection_policy import DROP_FRACTIONS, load_frame

OUTPUT_DIR = Path(__file__).parent
PLOT_PATH = OUTPUT_DIR / "sequential_group_missingness_drop_curves_2x4.png"
COEF_CSV = OUTPUT_DIR / "sequential_group_missingness_coefficients_2x4.csv"
SUMMARY_MD = OUTPUT_DIR / "sequential_group_missingness_summary_2x4.md"
PREDICTIONS_CSV = OUTPUT_DIR / "sequential_group_missingness_predictions_2x4.csv"

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


@dataclass
class SequentialModel:
    intercept: float
    beta_gr: float
    amazon_mean: float
    beta_amz: float
    openlibrary_mean: float
    beta_ol: float


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
    centered = numeric - mean_value
    return centered.fillna(0.0)


def fit_sequential_model(train_df: pd.DataFrame, target_col: str) -> SequentialModel | None:
    fit_rows = train_df[
        train_df["goodreads_rating_raw"].notna() & train_df[target_col].notna()
    ].copy()
    if len(fit_rows) < 8:
        return None

    gr_model = LinearRegression().fit(
        fit_rows[["goodreads_rating_raw"]].to_numpy(),
        pd.to_numeric(fit_rows[target_col], errors="coerce").to_numpy(),
    )
    fit_rows["stage1_pred"] = gr_model.predict(fit_rows[["goodreads_rating_raw"]].to_numpy())
    fit_rows["resid1"] = pd.to_numeric(fit_rows[target_col], errors="coerce") - fit_rows["stage1_pred"]

    amazon_mean = float(pd.to_numeric(fit_rows["amazon_rating_raw"], errors="coerce").mean())
    if not np.isfinite(amazon_mean):
        amazon_mean = 0.0
    amazon_rows = fit_rows[fit_rows["amazon_rating_raw"].notna()].copy()
    if len(amazon_rows) >= 5:
        amazon_centered = centered_or_zero(amazon_rows["amazon_rating_raw"], amazon_mean).to_numpy().reshape(-1, 1)
        amazon_target = amazon_rows["resid1"].to_numpy()
        amazon_model = LinearRegression(fit_intercept=False).fit(amazon_centered, amazon_target)
        beta_amz = float(amazon_model.coef_[0])
    else:
        beta_amz = 0.0

    fit_rows["stage2_pred"] = fit_rows["stage1_pred"] + beta_amz * centered_or_zero(
        fit_rows["amazon_rating_raw"], amazon_mean
    )
    fit_rows["resid2"] = pd.to_numeric(fit_rows[target_col], errors="coerce") - fit_rows["stage2_pred"]

    openlibrary_mean = float(pd.to_numeric(fit_rows["openlibrary_rating_raw"], errors="coerce").mean())
    if not np.isfinite(openlibrary_mean):
        openlibrary_mean = 0.0
    openlibrary_rows = fit_rows[fit_rows["openlibrary_rating_raw"].notna()].copy()
    if len(openlibrary_rows) >= 5:
        ol_centered = centered_or_zero(openlibrary_rows["openlibrary_rating_raw"], openlibrary_mean).to_numpy().reshape(-1, 1)
        ol_target = openlibrary_rows["resid2"].to_numpy()
        ol_model = LinearRegression(fit_intercept=False).fit(ol_centered, ol_target)
        beta_ol = float(ol_model.coef_[0])
    else:
        beta_ol = 0.0

    return SequentialModel(
        intercept=float(gr_model.intercept_),
        beta_gr=float(gr_model.coef_[0]),
        amazon_mean=amazon_mean,
        beta_amz=beta_amz,
        openlibrary_mean=openlibrary_mean,
        beta_ol=beta_ol,
    )


def predict_with_model(frame: pd.DataFrame, model: SequentialModel, target_col: str) -> pd.DataFrame:
    scored = frame.copy()
    pred_col = f"pred_{target_col}"
    scored[pred_col] = np.nan
    mask = scored["goodreads_rating_raw"].notna() & scored[target_col].notna()
    if not mask.any():
        return scored
    gr = pd.to_numeric(scored.loc[mask, "goodreads_rating_raw"], errors="coerce")
    amz_term = centered_or_zero(scored.loc[mask, "amazon_rating_raw"], model.amazon_mean)
    ol_term = centered_or_zero(scored.loc[mask, "openlibrary_rating_raw"], model.openlibrary_mean)
    scored.loc[mask, pred_col] = (
        model.intercept
        + model.beta_gr * gr
        + model.beta_amz * amz_term
        + model.beta_ol * ol_term
    )
    return scored


def build_drop_curve(frame: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    valid = frame[frame[target_col].notna() & frame[pred_col].notna()].copy()
    if len(valid) < 8:
        return pd.DataFrame()
    valid = valid.sort_values([pred_col, "title"], ascending=[False, True]).reset_index(drop=True)
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


def annotate_curve(axis: plt.Axes, curve: pd.DataFrame, color: str) -> None:
    for _, row in curve.iterrows():
        axis.annotate(
            f"{row['cutoff_pred']:.2f}",
            (row["drop_percent"], row["mean_actual"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color=color,
        )


def main() -> None:
    frame = load_frame()
    frame["group_name"] = frame["category"].map(category_group)
    frame["dataset"] = np.where(frame["source"].eq("Holdout 2026"), "holdout", "train")

    coefficient_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    curve_store: dict[tuple[str, str, str], pd.DataFrame] = {}
    y_values_by_target: dict[str, list[float]] = {target: [] for target, _ in TARGET_SPECS}
    predictions_df = frame.copy()
    for target_col, _ in TARGET_SPECS:
        predictions_df[f"pred_{target_col}"] = np.nan

    for group_name in GROUP_ORDER:
        train_group = frame[(frame["dataset"] == "train") & frame["group_name"].eq(group_name)].copy()
        holdout_group = frame[(frame["dataset"] == "holdout") & frame["group_name"].eq(group_name)].copy()
        combined_group = pd.concat([train_group, holdout_group], ignore_index=True)

        for target_col, target_label in TARGET_SPECS:
            model = fit_sequential_model(train_group, target_col)
            if model is None:
                summary_rows.extend(
                    [
                        {"group_name": group_name, "target": target_col, "dataset": "train", "n_eval": 0, "correlation": np.nan},
                        {"group_name": group_name, "target": target_col, "dataset": "holdout", "n_eval": 0, "correlation": np.nan},
                    ]
                )
                continue

            coefficient_rows.append(
                {
                    "group_name": group_name,
                    "target": target_label,
                    "intercept": model.intercept,
                    "beta_goodreads": model.beta_gr,
                    "amazon_mean": model.amazon_mean,
                    "beta_amazon_centered": model.beta_amz,
                    "openlibrary_mean": model.openlibrary_mean,
                    "beta_openlibrary_centered": model.beta_ol,
                }
            )

            scored_combined = predict_with_model(combined_group, model, target_col)
            pred_col = f"pred_{target_col}"
            for row in scored_combined.itertuples(index=False):
                mask = (
                    predictions_df["title"].eq(row.title)
                    & predictions_df["source"].eq(row.source)
                    & predictions_df["group_name"].eq(row.group_name)
                )
                predictions_df.loc[mask, pred_col] = getattr(row, pred_col)
            for dataset in ["train", "holdout"]:
                subset = scored_combined[scored_combined["dataset"].eq(dataset)].copy()
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
                curve = build_drop_curve(subset, target_col, pred_col)
                curve_store[(group_name, target_col, dataset)] = curve
                if not curve.empty:
                    y_values_by_target[target_col].extend(curve["mean_actual"].tolist())

    coef_df = pd.DataFrame(coefficient_rows)
    summary_df = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True, sharey="row")
    colors = {"train": "#1f77b4", "holdout": "#d62728"}
    labels = {"train": "Train", "holdout": "Holdout 2026"}

    for row_idx, (target_col, target_label) in enumerate(TARGET_SPECS):
        values = y_values_by_target[target_col]
        if values:
            y_min = min(values)
            y_max = max(values)
            pad = max(0.05, (y_max - y_min) * 0.1)
            ylim = (y_min - pad, y_max + pad)
        else:
            ylim = (1.0, 5.0)

        for col_idx, group_name in enumerate(GROUP_ORDER):
            axis = axes[row_idx, col_idx]
            title_bits = []
            for dataset in ["train", "holdout"]:
                row = summary_df[
                    (summary_df["group_name"] == group_name)
                    & (summary_df["target"] == target_col)
                    & (summary_df["dataset"] == dataset)
                ]
                if row.empty:
                    title_bits.append(f"{labels[dataset]} R=nan n=0")
                else:
                    item = row.iloc[0]
                    corr_text = "nan" if pd.isna(item["correlation"]) else f"{item['correlation']:.2f}"
                    title_bits.append(f"{labels[dataset]} R={corr_text} n={int(item['n_eval'])}")

            for dataset in ["train", "holdout"]:
                curve = curve_store.get((group_name, target_col, dataset), pd.DataFrame())
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
                annotate_curve(axis, curve, colors[dataset])

            axis.set_title(f"{group_name}\n" + " | ".join(title_bits), fontsize=10)
            axis.set_xlabel("% dropped")
            axis.set_ylim(*ylim)
            axis.grid(alpha=0.2)
            if col_idx == 0:
                axis.set_ylabel(f"{target_label}\nmean kept actual")
            if row_idx == 0:
                handles, labels_list = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Sequential Goodreads -> Amazon residual -> Open Library residual model",
        fontsize=16,
    )
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)

    summary_lines = ["# Sequential Group Missingness Model", ""]
    for _, row in coef_df.iterrows():
        summary_lines.append(f"## {row['group_name']} | {row['target']}")
        summary_lines.append(
            f"- `prediction = {row['intercept']:.4f} + {row['beta_goodreads']:.4f}*GR "
            f"+ {row['beta_amazon_centered']:.4f}*(AMZ - {row['amazon_mean']:.4f}, else 0) "
            f"+ {row['beta_openlibrary_centered']:.4f}*(OL - {row['openlibrary_mean']:.4f}, else 0)`"
        )
        summary_lines.append("")

    coef_df.to_csv(COEF_CSV, index=False)
    SUMMARY_MD.write_text("\n".join(summary_lines) + "\n")
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)

    print(f"Saved {PLOT_PATH.name}")
    print(f"Saved {COEF_CSV.name}")
    print(f"Saved {SUMMARY_MD.name}")
    print(f"Saved {PREDICTIONS_CSV.name}")


if __name__ == "__main__":
    main()
