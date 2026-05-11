"""Run utility-transformed three-source z-score models separately by category group."""

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
from sklearn.linear_model import LinearRegression

from ai_books_tracking.multi_source_selection_policy import (
    DROP_FRACTIONS,
    Variant,
    attach_zscores,
    load_frame,
    split_frame,
)

OUTPUT_DIR = Path(__file__).parent
COEF_CSV = OUTPUT_DIR / "three_source_zscore_utility_by_group_coefficients.csv"
SUMMARY_MD = OUTPUT_DIR / "three_source_zscore_utility_by_group_summary.md"

GROUP_ORDER = [
    "Business/Histories/General",
    "Fiction/Literature",
    "Technical/Other",
]
Z_COLS = ["goodreads_z", "openlibrary_z", "amazon_z"]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denom)


def zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
    safe_std = std if np.isfinite(std) and std > 1e-8 else 1.0
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric - mean) / safe_std


def utility_from_z(series: pd.Series, power: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.power(np.maximum(0.0, numeric), power), index=series.index)


def percentile_rank(value: float, reference: pd.Series) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or not np.isfinite(value):
        return float("nan")
    return float(100.0 * np.mean(ref <= value))


def add_target_utilities(train_scored: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    train_enjoy = pd.to_numeric(train_scored["avg_enjoyment"], errors="coerce")
    train_useful = pd.to_numeric(train_scored["avg_usefulness"], errors="coerce")
    enjoy_mean, enjoy_std = float(train_enjoy.mean()), float(train_enjoy.std())
    useful_mean, useful_std = float(train_useful.mean()), float(train_useful.std())

    scored = frame.copy()
    scored["enjoyment_target_z"] = zscore(scored["avg_enjoyment"], enjoy_mean, enjoy_std)
    scored["usefulness_target_z"] = zscore(scored["avg_usefulness"], useful_mean, useful_std)
    scored["util_enjoyment_z"] = utility_from_z(scored["enjoyment_target_z"], 1.5)
    scored["util_usefulness_z"] = utility_from_z(scored["usefulness_target_z"], 2.0)
    scored["util_total_equal"] = scored["util_enjoyment_z"] + scored["util_usefulness_z"]
    scored["util_total_weighted"] = scored["util_enjoyment_z"] + 2.0 * scored["util_usefulness_z"]
    return scored


def fit_models(
    train_scored: pd.DataFrame,
) -> tuple[LinearRegression, LinearRegression, pd.DataFrame]:
    fit_frame = train_scored[
        train_scored["all_sources_present"]
        & train_scored["util_enjoyment_z"].notna()
        & train_scored["util_usefulness_z"].notna()
    ].copy()
    X = fit_frame[Z_COLS].to_numpy()
    enjoy_model = LinearRegression().fit(X, fit_frame["util_enjoyment_z"].to_numpy())
    useful_model = LinearRegression().fit(X, fit_frame["util_usefulness_z"].to_numpy())
    coef_df = pd.DataFrame(
        {
            "feature": ["intercept", *Z_COLS],
            "enjoyment_utility_coef": [enjoy_model.intercept_, *enjoy_model.coef_.tolist()],
            "usefulness_utility_coef": [useful_model.intercept_, *useful_model.coef_.tolist()],
        }
    )
    return enjoy_model, useful_model, coef_df


def add_predictions(
    frame: pd.DataFrame, enjoy_model: LinearRegression, useful_model: LinearRegression
) -> pd.DataFrame:
    scored = frame.copy()
    mask = (
        scored["all_sources_present"]
        & scored["util_enjoyment_z"].notna()
        & scored["util_usefulness_z"].notna()
    )
    for column in [
        "pred_util_enjoyment_own",
        "pred_util_usefulness_own",
        "pred_util_enjoyment_swapped",
        "pred_util_usefulness_swapped",
        "pred_total_equal_own",
        "pred_total_equal_swapped",
        "pred_total_weighted_own",
        "pred_total_weighted_swapped",
    ]:
        scored[column] = np.nan
    if mask.any():
        X = scored.loc[mask, Z_COLS].to_numpy()
        pred_enjoy = enjoy_model.predict(X)
        pred_useful = useful_model.predict(X)
        pred_enjoy_swapped = useful_model.predict(X)
        pred_useful_swapped = enjoy_model.predict(X)
        scored.loc[mask, "pred_util_enjoyment_own"] = pred_enjoy
        scored.loc[mask, "pred_util_usefulness_own"] = pred_useful
        scored.loc[mask, "pred_util_enjoyment_swapped"] = pred_enjoy_swapped
        scored.loc[mask, "pred_util_usefulness_swapped"] = pred_useful_swapped
        scored.loc[mask, "pred_total_equal_own"] = pred_enjoy + pred_useful
        scored.loc[mask, "pred_total_equal_swapped"] = pred_enjoy_swapped + pred_useful_swapped
        scored.loc[mask, "pred_total_weighted_own"] = pred_enjoy + 2.0 * pred_useful
        scored.loc[mask, "pred_total_weighted_swapped"] = (
            pred_enjoy_swapped + 2.0 * pred_useful_swapped
        )
    return scored


def filtering_curve_percentile(
    frame: pd.DataFrame,
    target_col: str,
    score_col: str,
    reference_utility: pd.Series,
) -> pd.DataFrame:
    valid = frame[frame[target_col].notna() & frame[score_col].notna()].copy()
    if len(valid) < 8:
        return pd.DataFrame()
    valid = valid.sort_values([score_col, "title"], ascending=[False, True]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for drop_fraction in DROP_FRACTIONS:
        drop_n = int(np.floor(len(valid) * drop_fraction))
        keep_n = len(valid) - drop_n
        if keep_n < 5:
            continue
        kept = valid.iloc[:keep_n]
        mean_utility = float(pd.to_numeric(kept[target_col], errors="coerce").mean())
        rows.append(
            {
                "drop_percent": drop_fraction * 100,
                "keep_n": keep_n,
                "mean_utility": mean_utility,
                "utility_percentile": percentile_rank(mean_utility, reference_utility),
            }
        )
    return pd.DataFrame(rows)


def plot_scatter(
    axis: plt.Axes,
    frame: pd.DataFrame,
    pred_col: str,
    target_col: str,
    title: str,
) -> None:
    valid = frame[frame[pred_col].notna() & frame[target_col].notna()].copy()
    if valid.empty:
        axis.axis("off")
        return
    x = pd.to_numeric(valid[pred_col], errors="coerce").to_numpy()
    y = pd.to_numeric(valid[target_col], errors="coerce").to_numpy()
    upper = max(np.nanmax(x), np.nanmax(y), 0.1)
    axis.scatter(x, y, alpha=0.7, s=34, color="#0B3954")
    axis.plot([0, upper], [0, upper], linestyle="--", color="black", linewidth=1)
    axis.set_xlim(0, upper * 1.05)
    axis.set_ylim(0, upper * 1.05)
    axis.set_xlabel("Predicted utility")
    axis.set_ylabel("Actual utility")
    axis.grid(alpha=0.15)
    axis.set_title(f"{title}\nR={correlation(x, y):.3f}, n={len(valid)}")


def plot_progress(
    axis: plt.Axes,
    frame: pd.DataFrame,
    split_name: str,
    reference_map: dict[str, pd.Series],
) -> None:
    line_specs = [
        ("util_enjoyment_z", "pred_util_enjoyment_own", "Enjoy utility by enjoy coefs", "#1f77b4"),
        ("util_enjoyment_z", "pred_util_enjoyment_swapped", "Enjoy utility by useful coefs", "#6baed6"),
        ("util_usefulness_z", "pred_util_usefulness_own", "Useful utility by useful coefs", "#ff7f0e"),
        ("util_usefulness_z", "pred_util_usefulness_swapped", "Useful utility by enjoy coefs", "#fdae6b"),
    ]
    for target_col, score_col, label, color in line_specs:
        curve = filtering_curve_percentile(frame, target_col, score_col, reference_map[target_col])
        if curve.empty:
            continue
        axis.plot(
            curve["drop_percent"],
            curve["utility_percentile"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axis.set_xlabel("% dropped")
    axis.set_ylabel("Utility percentile vs prior books")
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.2)
    axis.set_title(f"{split_name}: utility-percentile progress")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8)


def plot_main_2x3(
    group_name: str,
    train_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    reference_map: dict[str, pd.Series],
) -> Path:
    output_path = OUTPUT_DIR / f"three_source_zscore_utility_train_test_2x3_{slugify(group_name)}.png"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    plot_scatter(
        axes[0, 0],
        train_scored,
        "pred_util_enjoyment_own",
        "util_enjoyment_z",
        f"Train enjoyment utility model\n{group_name}",
    )
    plot_scatter(
        axes[0, 1],
        train_scored,
        "pred_util_usefulness_own",
        "util_usefulness_z",
        f"Train usefulness utility model\n{group_name}",
    )
    plot_progress(axes[0, 2], train_scored, "Train", reference_map)
    plot_scatter(
        axes[1, 0],
        test_scored,
        "pred_util_enjoyment_own",
        "util_enjoyment_z",
        f"Test enjoyment utility model\n{group_name}",
    )
    plot_scatter(
        axes[1, 1],
        test_scored,
        "pred_util_usefulness_own",
        "util_usefulness_z",
        f"Test usefulness utility model\n{group_name}",
    )
    plot_progress(axes[1, 2], test_scored, "Test", reference_map)
    fig.suptitle(
        f"Three source z-scores -> utility targets | {group_name}",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_total_progress(
    group_name: str,
    train_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    reference_map: dict[str, pd.Series],
) -> Path:
    output_path = OUTPUT_DIR / f"three_source_zscore_total_utility_percentiles_{slugify(group_name)}.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plot_specs = [
        ("pred_total_equal_own", "pred_total_equal_swapped", "util_total_equal", "Total utility = enjoy + useful"),
        ("pred_total_weighted_own", "pred_total_weighted_swapped", "util_total_weighted", "Total utility = enjoy + 2*useful"),
    ]
    for row_idx, (split_name, frame) in enumerate([("Train", train_scored), ("Test", test_scored)]):
        for col_idx, (own_col, swapped_col, target_col, title) in enumerate(plot_specs):
            axis = axes[row_idx, col_idx]
            own_curve = filtering_curve_percentile(frame, target_col, own_col, reference_map[target_col])
            swapped_curve = filtering_curve_percentile(frame, target_col, swapped_col, reference_map[target_col])
            if not own_curve.empty:
                axis.plot(
                    own_curve["drop_percent"],
                    own_curve["utility_percentile"],
                    marker="o",
                    linewidth=2,
                    color="#2d6a4f",
                    label="Own coefficient mapping",
                )
            if not swapped_curve.empty:
                axis.plot(
                    swapped_curve["drop_percent"],
                    swapped_curve["utility_percentile"],
                    marker="o",
                    linewidth=2,
                    color="#95d5b2",
                    label="Swapped coefficient mapping",
                )
            axis.set_ylim(0, 100)
            axis.set_xlabel("% dropped")
            axis.set_ylabel("Utility percentile vs prior books")
            axis.grid(alpha=0.2)
            axis.set_title(f"{split_name}: {title}")
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Total-utility percentile progress curves | {group_name}", fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def build_group_frames(
    group_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_frame()
    frame["category_bucket"] = frame["category"].map(category_group)
    frame = frame[frame["category_bucket"] == group_name].copy()
    train_all, _validation, test_all, _ = split_frame(frame)
    variant = Variant(aggregator="precision_weighted_z", missing_mode="drop", shrink=False)
    train_scored, _ = attach_zscores(train_all, train_all, variant)
    test_scored, _ = attach_zscores(train_all, test_all, variant)
    for frame_part in [train_scored, test_scored]:
        frame_part["all_sources_present"] = frame_part[Z_COLS].notna().all(axis=1)
    return train_scored.copy(), test_scored.copy()


def main() -> None:
    coef_frames: list[pd.DataFrame] = []
    summary_lines = ["# Three-source z-score utility models by category group", ""]

    for group_name in GROUP_ORDER:
        train_scored, test_scored = build_group_frames(group_name)
        train_scored = add_target_utilities(train_scored, train_scored)
        test_scored = add_target_utilities(train_scored, test_scored)

        fit_rows = train_scored[
            train_scored["all_sources_present"]
            & train_scored["util_enjoyment_z"].notna()
            & train_scored["util_usefulness_z"].notna()
        ]
        if len(fit_rows) < 8:
            summary_lines.extend([f"## {group_name}", "- Not enough complete-case training rows.", ""])
            continue

        enjoy_model, useful_model, coef_df = fit_models(train_scored)
        coef_df["group_name"] = group_name
        coef_frames.append(coef_df)

        train_scored = add_predictions(train_scored, enjoy_model, useful_model)
        test_scored = add_predictions(test_scored, enjoy_model, useful_model)

        reference_map = {
            "util_enjoyment_z": train_scored["util_enjoyment_z"],
            "util_usefulness_z": train_scored["util_usefulness_z"],
            "util_total_equal": train_scored["util_total_equal"],
            "util_total_weighted": train_scored["util_total_weighted"],
        }

        main_plot = plot_main_2x3(group_name, train_scored, test_scored, reference_map)
        total_plot = plot_total_progress(group_name, train_scored, test_scored, reference_map)

        train_valid = train_scored[train_scored["pred_util_enjoyment_own"].notna()].copy()
        test_valid = test_scored[test_scored["pred_util_enjoyment_own"].notna()].copy()
        enjoy_coef = enjoy_model.coef_
        useful_coef = useful_model.coef_

        summary_lines.extend(
            [
                f"## {group_name}",
                f"- Train rows: `{len(train_scored)}`, test rows: `{len(test_scored)}`",
                f"- Complete-case train rows used: `{len(fit_rows)}`",
                f"- Cosine similarity of enjoyment/usefulness coefficients: `{cosine_similarity(enjoy_coef, useful_coef):.4f}`",
                f"- Train enjoyment utility R: `{correlation(train_valid['pred_util_enjoyment_own'].to_numpy(), train_valid['util_enjoyment_z'].to_numpy()):.4f}`",
                f"- Train usefulness utility R: `{correlation(train_valid['pred_util_usefulness_own'].to_numpy(), train_valid['util_usefulness_z'].to_numpy()):.4f}`",
                f"- Test enjoyment utility R: `{correlation(test_valid['pred_util_enjoyment_own'].to_numpy(), test_valid['util_enjoyment_z'].to_numpy()):.4f}`",
                f"- Test usefulness utility R: `{correlation(test_valid['pred_util_usefulness_own'].to_numpy(), test_valid['util_usefulness_z'].to_numpy()):.4f}`",
                f"- Main plot: `{main_plot.name}`",
                f"- Total utility plot: `{total_plot.name}`",
                "",
            ]
        )

    coef_output = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    coef_output.to_csv(COEF_CSV, index=False)
    SUMMARY_MD.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved {COEF_CSV.name}")
    print(f"Saved {SUMMARY_MD.name}")


if __name__ == "__main__":
    main()
