"""Fit direct linear models on three separate source z-scores.

This avoids pooled mean-z aggregation and avoids review-count shrinkage.
Outputs:
- 2x3 plot: rows=train/test, cols=enjoyment scatter / usefulness scatter /
  cross-applied coefficient filtering curves.
- coefficient summary comparing enjoyment vs usefulness models.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

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
PLOT_PATH = OUTPUT_DIR / "three_source_zscore_linear_train_test_2x3.png"
COEF_CSV = OUTPUT_DIR / "three_source_zscore_linear_coefficients.csv"
SUMMARY_MD = OUTPUT_DIR / "three_source_zscore_linear_summary.md"

Z_COLS = ["goodreads_z", "openlibrary_z", "amazon_z"]


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denom)


def prepare_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_frame()
    train_all, _validation, test_all, _ = split_frame(frame)
    variant = Variant(aggregator="precision_weighted_z", missing_mode="drop", shrink=False)
    train_scored, _ = attach_zscores(train_all, train_all, variant)
    test_scored, _ = attach_zscores(train_all, test_all, variant)
    train_scored = train_scored.copy()
    test_scored = test_scored.copy()
    for frame_part in [train_scored, test_scored]:
        frame_part["all_sources_present"] = frame_part[Z_COLS].notna().all(axis=1)
    return train_scored, test_scored


def fit_models(train_scored: pd.DataFrame) -> tuple[LinearRegression, LinearRegression, pd.DataFrame]:
    fit_frame = train_scored[
        train_scored["all_sources_present"]
        & train_scored["avg_enjoyment"].notna()
        & train_scored["avg_usefulness"].notna()
    ].copy()
    X = fit_frame[Z_COLS].to_numpy()

    enjoy_model = LinearRegression().fit(X, fit_frame["avg_enjoyment"].to_numpy())
    useful_model = LinearRegression().fit(X, fit_frame["avg_usefulness"].to_numpy())

    coef_df = pd.DataFrame(
        {
            "feature": ["intercept", *Z_COLS],
            "enjoyment_coef": [enjoy_model.intercept_, *enjoy_model.coef_.tolist()],
            "usefulness_coef": [useful_model.intercept_, *useful_model.coef_.tolist()],
        }
    )
    return enjoy_model, useful_model, coef_df


def add_predictions(
    frame: pd.DataFrame, enjoy_model: LinearRegression, useful_model: LinearRegression
) -> pd.DataFrame:
    scored = frame.copy()
    mask = scored["all_sources_present"] & scored["avg_enjoyment"].notna() & scored["avg_usefulness"].notna()
    scored["pred_enjoyment_own"] = np.nan
    scored["pred_usefulness_own"] = np.nan
    scored["pred_enjoyment_swapped"] = np.nan
    scored["pred_usefulness_swapped"] = np.nan
    if mask.any():
        X = scored.loc[mask, Z_COLS].to_numpy()
        scored.loc[mask, "pred_enjoyment_own"] = enjoy_model.predict(X)
        scored.loc[mask, "pred_usefulness_own"] = useful_model.predict(X)
        # Swap coefficient vectors across targets.
        scored.loc[mask, "pred_enjoyment_swapped"] = useful_model.predict(X)
        scored.loc[mask, "pred_usefulness_swapped"] = enjoy_model.predict(X)
    return scored


def filtering_curve(
    frame: pd.DataFrame, target_col: str, score_col: str
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
        rows.append(
            {
                "drop_percent": drop_fraction * 100,
                "keep_n": keep_n,
                "mean_actual": float(pd.to_numeric(kept[target_col], errors="coerce").mean()),
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
    axis.scatter(x, y, alpha=0.7, s=34, color="#0B3954")
    axis.plot([1, 5], [1, 5], linestyle="--", color="black", linewidth=1)
    axis.set_xlim(1, 5)
    axis.set_ylim(1, 5)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.grid(alpha=0.15)
    axis.set_title(f"{title}\nR={correlation(x, y):.3f}, n={len(valid)}")


def plot_progress(axis: plt.Axes, frame: pd.DataFrame, split_name: str) -> None:
    line_specs = [
        ("avg_enjoyment", "pred_enjoyment_own", "Enjoyment by enjoy coefs", "#1f77b4"),
        ("avg_enjoyment", "pred_enjoyment_swapped", "Enjoyment by useful coefs", "#6baed6"),
        ("avg_usefulness", "pred_usefulness_own", "Usefulness by useful coefs", "#ff7f0e"),
        ("avg_usefulness", "pred_usefulness_swapped", "Usefulness by enjoy coefs", "#fdae6b"),
    ]
    for target_col, score_col, label, color in line_specs:
        curve = filtering_curve(frame, target_col, score_col)
        if curve.empty:
            continue
        axis.plot(curve["drop_percent"], curve["mean_actual"], marker="o", linewidth=2, label=label, color=color)
    axis.set_xlabel("% dropped")
    axis.set_ylabel("Mean kept actual rating")
    axis.set_title(f"{split_name}: cross-applied coefficients")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)


def plot_2x3(train_scored: pd.DataFrame, test_scored: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    plot_scatter(
        axes[0, 0],
        train_scored,
        "pred_enjoyment_own",
        "avg_enjoyment",
        "Train enjoyment model",
    )
    plot_scatter(
        axes[0, 1],
        train_scored,
        "pred_usefulness_own",
        "avg_usefulness",
        "Train usefulness model",
    )
    plot_progress(axes[0, 2], train_scored, "Train")
    plot_scatter(
        axes[1, 0],
        test_scored,
        "pred_enjoyment_own",
        "avg_enjoyment",
        "Test enjoyment model",
    )
    plot_scatter(
        axes[1, 1],
        test_scored,
        "pred_usefulness_own",
        "avg_usefulness",
        "Test usefulness model",
    )
    plot_progress(axes[1, 2], test_scored, "Test")
    fig.suptitle(
        "Three separate source z-scores -> direct linear predictions (no shrinkage)",
        fontsize=16,
    )
    fig.savefig(PLOT_PATH, dpi=220)
    plt.close(fig)


def summary_markdown(
    coef_df: pd.DataFrame,
    enjoy_model: LinearRegression,
    useful_model: LinearRegression,
    train_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
) -> str:
    enjoy_coef = enjoy_model.coef_
    useful_coef = useful_model.coef_

    train_valid = train_scored[train_scored["pred_enjoyment_own"].notna()].copy()
    test_valid = test_scored[test_scored["pred_enjoyment_own"].notna()].copy()

    lines = [
        "# Three-source z-score linear models",
        "",
        "## Decision rules",
        "- Features are the three separate category-bucket z-scores: `goodreads_z`, `openlibrary_z`, `amazon_z`.",
        "- No shrinkage was used.",
        "- No pooled mean-z score was used.",
        "- Rows are complete-case only for these three sources.",
        "",
        "## Coefficients",
        f"- Enjoyment intercept: `{enjoy_model.intercept_:.4f}`",
        f"- Usefulness intercept: `{useful_model.intercept_:.4f}`",
    ]
    for _, row in coef_df[coef_df["feature"] != "intercept"].iterrows():
        lines.append(
            f"- `{row['feature']}`: enjoyment `{row['enjoyment_coef']:.4f}`, usefulness `{row['usefulness_coef']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Coefficient similarity",
            f"- Cosine similarity: `{cosine_similarity(enjoy_coef, useful_coef):.4f}`",
            f"- Correlation across the three source coefficients: `{correlation(enjoy_coef, useful_coef):.4f}`",
            "",
            "## Fit summary",
            f"- Train enjoyment R: `{correlation(train_valid['pred_enjoyment_own'].to_numpy(), train_valid['avg_enjoyment'].to_numpy()):.4f}`",
            f"- Train usefulness R: `{correlation(train_valid['pred_usefulness_own'].to_numpy(), train_valid['avg_usefulness'].to_numpy()):.4f}`",
            f"- Test enjoyment R: `{correlation(test_valid['pred_enjoyment_own'].to_numpy(), test_valid['avg_enjoyment'].to_numpy()):.4f}`",
            f"- Test usefulness R: `{correlation(test_valid['pred_usefulness_own'].to_numpy(), test_valid['avg_usefulness'].to_numpy()):.4f}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    train_scored, test_scored = prepare_frames()
    enjoy_model, useful_model, coef_df = fit_models(train_scored)
    train_scored = add_predictions(train_scored, enjoy_model, useful_model)
    test_scored = add_predictions(test_scored, enjoy_model, useful_model)

    plot_2x3(train_scored, test_scored)
    coef_df.to_csv(COEF_CSV, index=False)
    SUMMARY_MD.write_text(
        summary_markdown(coef_df, enjoy_model, useful_model, train_scored, test_scored)
    )

    print("Coefficients:")
    print(coef_df.to_string(index=False))
    print(f"Saved {PLOT_PATH.name}")
    print(f"Saved {COEF_CSV.name}")
    print(f"Saved {SUMMARY_MD.name}")


if __name__ == "__main__":
    main()
