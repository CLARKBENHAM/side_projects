"""Compare cleaned per-book reading speed/time with personal book value ratings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_books_tracking.book_wpm_calendar_notes import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    abbreviation_matches_title,
    normalize_title,
    title_match_score,
    titles_match,
)

DATA_DIR = REPO_ROOT / "data"
DEFAULT_SPEED_CSV = DEFAULT_OUTPUT_DIR / "book_speed_analysis.csv"
DEFAULT_MASTER_RATINGS = (
    DATA_DIR / "Books Read and their effects - master_book_metadata_cleaned_final.csv"
)
DEFAULT_NEW_RATINGS = (
    DATA_DIR / "Books Read and their effects - new_books_to_rate 2026.csv"
)
DEFAULT_RATINGS2 = DATA_DIR / "Books Read and their effects - Ratings 2.csv"
VALUE_BASE = 2
MIN_CATEGORY_N = 3


def utility_value(rating: object) -> np.ndarray | float:
    values = np.asarray(rating, dtype=float)
    return VALUE_BASE ** (values - 1) - 1


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mean_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return pd.Series(np.nan, index=frame.index)
    return frame[existing].apply(_to_numeric).mean(axis=1)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(_markdown_table(df), encoding="utf-8")


def load_unified_ratings(
    *,
    master_ratings: Path = DEFAULT_MASTER_RATINGS,
    new_ratings: Path = DEFAULT_NEW_RATINGS,
    ratings2: Path = DEFAULT_RATINGS2,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if master_ratings.exists():
        master = pd.read_csv(master_ratings)
        frames.append(
            pd.DataFrame(
                {
                    "rating_title": master["personal_rating_title"]
                    .fillna(master["title"])
                    .astype(str),
                    "rating_date": pd.to_datetime(
                        master["finished_date"], errors="coerce"
                    ),
                    "avg_enjoyment": _to_numeric(master["avg_enjoyment"]),
                    "avg_usefulness": _to_numeric(master["avg_usefulness"]),
                    "rating_category": master.get("bookshelf"),
                    "rating_source": "master_final",
                    "source_priority": 1,
                }
            )
        )

    if new_ratings.exists():
        new = pd.read_csv(new_ratings)
        frames.append(
            pd.DataFrame(
                {
                    "rating_title": new["title"].astype(str),
                    "rating_date": pd.to_datetime(
                        new["date_finished"], errors="coerce"
                    ),
                    "avg_enjoyment": _mean_existing(
                        new, ["Enjoyment (/5)", "Enjoyment (/5) 2nd"]
                    ),
                    "avg_usefulness": _mean_existing(
                        new, ["Usefulness /5 to Me", "Usefulness /5 to Me.1"]
                    ),
                    "rating_category": new.get("Bookshelf"),
                    "rating_source": "new_books_2026",
                    "source_priority": 0,
                }
            )
        )

    if ratings2.exists():
        old = pd.read_csv(ratings2, usecols=range(8))
        frames.append(
            pd.DataFrame(
                {
                    "rating_title": old["title"].astype(str),
                    "rating_date": pd.to_datetime(
                        old["latest_modified"], errors="coerce"
                    ),
                    "avg_enjoyment": _to_numeric(old["Enjoyment (/5)"]),
                    "avg_usefulness": _to_numeric(old["Usefulness /5 to Me"]),
                    "rating_category": old.get("Bookshelf"),
                    "rating_source": "ratings2",
                    "source_priority": 2,
                }
            )
        )

    ratings = pd.concat(frames, ignore_index=True)
    ratings = ratings[
        ratings["avg_enjoyment"].notna() | ratings["avg_usefulness"].notna()
    ].copy()
    ratings["rating_title_norm"] = ratings["rating_title"].map(normalize_title)
    ratings = ratings.drop_duplicates(
        [
            "rating_title_norm",
            "rating_date",
            "avg_enjoyment",
            "avg_usefulness",
            "rating_source",
        ]
    ).reset_index(drop=True)
    return ratings


def score_rating_title(speed_title: str, rating_title: str) -> float:
    score = title_match_score(speed_title, rating_title)
    if titles_match(speed_title, rating_title):
        score = max(score, 85.0)
    if abbreviation_matches_title(speed_title, rating_title):
        score = max(score, 82.0)
    return float(score)


def match_rating(row: pd.Series, ratings: pd.DataFrame) -> dict[str, object]:
    title = str(row["title"])
    finish_date = pd.to_datetime(row["matched_finish_date"], errors="coerce")
    candidates = ratings.copy()
    candidates["_score"] = candidates["rating_title"].map(
        lambda value: score_rating_title(title, str(value))
    )
    candidates["_date_diff"] = np.nan
    if pd.notna(finish_date):
        candidates["_date_diff"] = (
            (candidates["rating_date"] - finish_date).abs().dt.days
        )
        date_candidates = candidates[
            candidates["_date_diff"].le(2) & candidates["_score"].gt(0)
        ].copy()
    else:
        date_candidates = pd.DataFrame()

    if not date_candidates.empty:
        best = date_candidates.sort_values(
            ["_score", "source_priority", "_date_diff"],
            ascending=[False, True, True],
        ).iloc[0]
        method = "date_title_strong" if best["_score"] >= 70 else "date_title_overlap"
        return rating_match_dict(best, method)

    strong = candidates[candidates["_score"].ge(70)].copy()
    if not strong.empty:
        best = strong.sort_values(
            ["_score", "source_priority", "_date_diff"],
            ascending=[False, True, True],
        ).iloc[0]
        return rating_match_dict(best, "title_strong_any_date")

    return {
        "rating_title": "",
        "rating_date": pd.NaT,
        "rating_source": "",
        "rating_match_method": "unmatched",
        "rating_match_score": np.nan,
        "rating_date_diff_days": np.nan,
        "avg_enjoyment": np.nan,
        "avg_usefulness": np.nan,
        "rating_category": "",
    }


def rating_match_dict(row: pd.Series, method: str) -> dict[str, object]:
    return {
        "rating_title": row["rating_title"],
        "rating_date": row["rating_date"],
        "rating_source": row["rating_source"],
        "rating_match_method": method,
        "rating_match_score": row["_score"],
        "rating_date_diff_days": row.get("_date_diff", np.nan),
        "avg_enjoyment": row["avg_enjoyment"],
        "avg_usefulness": row["avg_usefulness"],
        "rating_category": row.get("rating_category", ""),
    }


def clean_speed_subset(speed: pd.DataFrame) -> pd.DataFrame:
    frame = speed.copy()
    for column in [
        "usable_for_visual_reading_speed",
        "wpm_visual_after_audio_350wpm",
        "primary_first_pass_minutes",
        "non_audio_primary_minutes",
    ]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    clean = frame[
        frame["usable_for_visual_reading_speed"].astype(bool)
        & frame["wpm_visual_after_audio_350wpm"].le(600)
    ].copy()
    return clean


def build_value_speed_dataset(
    *,
    speed_csv: Path = DEFAULT_SPEED_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    master_ratings: Path = DEFAULT_MASTER_RATINGS,
    new_ratings: Path = DEFAULT_NEW_RATINGS,
    ratings2: Path = DEFAULT_RATINGS2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    speed = pd.read_csv(speed_csv)
    ratings = load_unified_ratings(
        master_ratings=master_ratings,
        new_ratings=new_ratings,
        ratings2=ratings2,
    )
    clean = clean_speed_subset(speed)
    matches = pd.DataFrame([match_rating(row, ratings) for _, row in clean.iterrows()])
    joined = pd.concat([clean.reset_index(drop=True), matches], axis=1)
    joined["analysis_category"] = joined["rating_category"].replace("", np.nan)
    joined["analysis_category"] = joined["analysis_category"].fillna(joined["category"])
    joined["analysis_category"] = joined["analysis_category"].fillna("Unknown")
    joined["visual_wpm"] = joined["wpm_visual_after_audio_350wpm"]
    joined["primary_hours"] = joined["primary_first_pass_minutes"] / 60
    joined["non_audio_hours"] = joined["non_audio_primary_minutes"] / 60
    joined["usefulness_utility"] = utility_value(joined["avg_usefulness"])
    joined["enjoyment_utility"] = utility_value(joined["avg_enjoyment"])
    joined["has_value_rating"] = (
        joined["avg_usefulness"].notna() & joined["avg_enjoyment"].notna()
    )
    joined = joined.sort_values("matched_finish_date").reset_index(drop=True)
    joined.to_csv(output_dir / "book_value_speed_joined.csv", index=False)
    _write_markdown_table(
        joined[
            [
                "title",
                "matched_finish_date",
                "analysis_category",
                "visual_wpm",
                "primary_hours",
                "avg_usefulness",
                "avg_enjoyment",
                "rating_match_method",
                "rating_title",
            ]
        ],
        output_dir / "book_value_speed_joined.md",
    )

    summary = summarize_relationships(joined[joined["has_value_rating"]].copy())
    summary.to_csv(
        output_dir / "book_value_speed_relationship_summary.csv", index=False
    )
    _write_markdown_table(
        summary, output_dir / "book_value_speed_relationship_summary.md"
    )

    category = summarize_by_category(joined[joined["has_value_rating"]].copy())
    category.to_csv(output_dir / "book_value_speed_by_category.csv", index=False)
    _write_markdown_table(category, output_dir / "book_value_speed_by_category.md")

    unmatched = joined[~joined["has_value_rating"]].copy()
    unmatched.to_csv(output_dir / "book_value_speed_unmatched_ratings.csv", index=False)

    plot_rating_scatter_matrix(
        joined[joined["has_value_rating"]].copy(),
        output_dir / "book_value_speed_scatter_matrix.png",
    )
    plot_rating_scatter_matrix(
        joined[joined["has_value_rating"]].copy(),
        output_dir / "book_value_speed_utility_scatter_matrix.png",
        utility=True,
    )
    plot_category_relationships(
        category,
        output_dir / "book_value_speed_by_category.png",
    )
    write_report(joined, summary, category, output_dir)
    return joined, summary, category


def summarize_relationships(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = [
        ("visual_wpm", "Adjusted visual WPM", 100.0),
        ("primary_hours", "Primary reading hours", 1.0),
    ]
    outcomes = [
        ("avg_usefulness", "Usefulness rating", "rating_points"),
        ("avg_enjoyment", "Enjoyment rating", "rating_points"),
        ("usefulness_utility", "Usefulness utility value", "utility_points"),
        ("enjoyment_utility", "Enjoyment utility value", "utility_points"),
    ]
    for x_col, x_label, slope_scale in pairs:
        for y_col, y_label, y_units in outcomes:
            subset = frame[[x_col, y_col, "analysis_category"]].dropna().copy()
            if len(subset) < 3:
                continue
            pearson = stats.pearsonr(subset[x_col], subset[y_col])
            spearman = stats.spearmanr(subset[x_col], subset[y_col])
            reg = stats.linregress(subset[x_col], subset[y_col])
            category_adjusted = category_adjusted_corr(subset, x_col, y_col)
            rows.append(
                {
                    "predictor": x_label,
                    "outcome": y_label,
                    "n": len(subset),
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_r": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "slope_per_unit": reg.slope,
                    "slope_scale": slope_scale,
                    "slope_per_scale": reg.slope * slope_scale,
                    "outcome_units": y_units,
                    "category_adjusted_pearson_r": category_adjusted,
                }
            )
    return pd.DataFrame(rows)


def category_adjusted_corr(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    adjusted = frame.copy()
    adjusted["_x_resid"] = adjusted[x_col] - adjusted.groupby("analysis_category")[
        x_col
    ].transform("mean")
    adjusted["_y_resid"] = adjusted[y_col] - adjusted.groupby("analysis_category")[
        y_col
    ].transform("mean")
    adjusted = adjusted.dropna(subset=["_x_resid", "_y_resid"])
    if len(adjusted) < 3 or adjusted["_x_resid"].std() == 0:
        return np.nan
    return float(stats.pearsonr(adjusted["_x_resid"], adjusted["_y_resid"]).statistic)


def summarize_by_category(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("analysis_category")
        .agg(
            n_books=("title", "count"),
            median_visual_wpm=("visual_wpm", "median"),
            median_primary_hours=("primary_hours", "median"),
            mean_usefulness=("avg_usefulness", "mean"),
            mean_enjoyment=("avg_enjoyment", "mean"),
            mean_usefulness_utility=("usefulness_utility", "mean"),
            mean_enjoyment_utility=("enjoyment_utility", "mean"),
        )
        .reset_index()
        .sort_values(["n_books", "mean_usefulness"], ascending=[False, False])
    )
    return grouped


def category_color_map(
    categories: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {category: cmap(index % 10) for index, category in enumerate(categories)}


def plot_rating_scatter_matrix(
    frame: pd.DataFrame, output_path: Path, *, utility: bool = False
) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    if utility:
        outcomes = [
            ("usefulness_utility", "Usefulness utility value"),
            ("enjoyment_utility", "Enjoyment utility value"),
        ]
        title = "Utility Value vs Clean Reading Speed and Time"
    else:
        outcomes = [
            ("avg_usefulness", "Usefulness (/5)"),
            ("avg_enjoyment", "Enjoyment (/5)"),
        ]
        title = "Ratings vs Clean Reading Speed and Time"
    predictors = [
        ("visual_wpm", "Adjusted visual WPM"),
        ("primary_hours", "Primary reading hours"),
    ]
    categories = sorted(plot_frame["analysis_category"].fillna("Unknown").unique())
    colors = category_color_map(categories)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False, sharey=False)
    fig.suptitle(title, fontsize=15)
    for row_index, (y_col, y_label) in enumerate(outcomes):
        for col_index, (x_col, x_label) in enumerate(predictors):
            ax = axes[row_index, col_index]
            for category in categories:
                subset = plot_frame[plot_frame["analysis_category"].eq(category)]
                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    label=category,
                    color=colors[category],
                    s=38,
                    alpha=0.78,
                    edgecolor="white",
                    linewidth=0.4,
                )
            add_trendline(ax, plot_frame, x_col, y_col)
            ax.margins(x=0.1, y=0.08)
            annotate_extremes(ax, plot_frame, x_col, y_col)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.86, 0.5))
    fig.tight_layout(rect=(0, 0, 0.86, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_trendline(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    subset = frame[[x_col, y_col]].dropna()
    if len(subset) < 3:
        return
    slope, intercept, r_value, _, _ = stats.linregress(subset[x_col], subset[y_col])
    x_values = np.linspace(subset[x_col].min(), subset[x_col].max(), 100)
    ax.plot(x_values, intercept + slope * x_values, color="#111827", linewidth=1.5)
    ax.text(
        0.02,
        0.94,
        f"r={r_value:+.2f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
    )


def annotate_extremes(
    ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str, max_labels: int = 3
) -> None:
    subset = frame[[x_col, y_col, "title"]].dropna().copy()
    if len(subset) <= max_labels:
        label_frame = subset
    else:
        label_indexes = set(subset[y_col].nlargest(1).index)
        label_indexes.update(subset[x_col].nlargest(1).index)
        label_indexes.update(subset[x_col].nsmallest(1).index)
        label_frame = subset.loc[sorted(label_indexes)[:max_labels]]
    for _, row in label_frame.iterrows():
        ax.annotate(
            str(row["title"])[:28],
            (row[x_col], row[y_col]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )


def plot_category_relationships(category: pd.DataFrame, output_path: Path) -> None:
    plot_frame = category[category["n_books"].ge(MIN_CATEGORY_N)].copy()
    if plot_frame.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Category-Level Speed/Time vs Ratings (N categories with >= {MIN_CATEGORY_N} books)",
        fontsize=14,
    )
    specs = [
        (
            "median_visual_wpm",
            "mean_usefulness",
            "Median adjusted visual WPM",
            "Mean usefulness",
        ),
        (
            "median_primary_hours",
            "mean_usefulness",
            "Median primary hours",
            "Mean usefulness",
        ),
        (
            "median_visual_wpm",
            "mean_enjoyment",
            "Median adjusted visual WPM",
            "Mean enjoyment",
        ),
        (
            "median_primary_hours",
            "mean_enjoyment",
            "Median primary hours",
            "Mean enjoyment",
        ),
    ]
    for ax, (x_col, y_col, x_label, y_label) in zip(axes.flat, specs):
        sizes = 35 + plot_frame["n_books"] * 16
        ax.scatter(
            plot_frame[x_col],
            plot_frame[y_col],
            s=sizes,
            color="#2563eb",
            alpha=0.72,
        )
        ax.margins(x=0.14, y=0.16)
        for _, row in plot_frame.iterrows():
            ax.annotate(
                str(row["analysis_category"])[:24],
                (row[x_col], row[y_col]),
                fontsize=8,
                xytext=(5, 4),
                textcoords="offset points",
            )
        add_trendline(
            ax, plot_frame.rename(columns={"analysis_category": "title"}), x_col, y_col
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    category: pd.DataFrame,
    output_dir: Path,
) -> None:
    matched = joined[joined["has_value_rating"]].copy()
    unmatched = joined[~joined["has_value_rating"]].copy()
    key = summary[
        summary["outcome"].isin(["Usefulness rating", "Enjoyment rating"])
    ].copy()
    category_display = category[category["n_books"].ge(MIN_CATEGORY_N)].copy()
    lines = [
        "# Book Value vs Speed Analysis",
        "",
        "Clean speed slice: local-file word counts, first reads, primary time over 90 minutes, non-audio-dominant rows, and adjusted visual WPM <= 600.",
        "",
        f"- Clean speed rows: {len(joined)}",
        f"- Rows matched to usefulness/enjoyment ratings: {len(matched)}",
        f"- Unmatched clean rows: {len(unmatched)}",
        "",
        "## Relationship Summary",
        "",
        _markdown_table(
            key[
                [
                    "predictor",
                    "outcome",
                    "n",
                    "pearson_r",
                    "pearson_p",
                    "spearman_r",
                    "spearman_p",
                    "slope_per_scale",
                    "category_adjusted_pearson_r",
                ]
            ]
        ),
        "",
        "## Category Summary",
        "",
        _markdown_table(category_display),
        "",
        "## Plots",
        "",
        "![Ratings vs speed and time](book_value_speed_scatter_matrix.png)",
        "",
        "![Utility value vs speed and time](book_value_speed_utility_scatter_matrix.png)",
        "",
        "![Category relationships](book_value_speed_by_category.png)",
    ]
    (output_dir / "book_value_speed_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed-csv", type=Path, default=DEFAULT_SPEED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--master-ratings", type=Path, default=DEFAULT_MASTER_RATINGS)
    parser.add_argument("--new-ratings", type=Path, default=DEFAULT_NEW_RATINGS)
    parser.add_argument("--ratings2", type=Path, default=DEFAULT_RATINGS2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joined, summary, category = build_value_speed_dataset(
        speed_csv=args.speed_csv,
        output_dir=args.output_dir,
        master_ratings=args.master_ratings,
        new_ratings=args.new_ratings,
        ratings2=args.ratings2,
    )
    print(
        f"Wrote value-speed analysis for {int(joined['has_value_rating'].sum())} "
        f"matched books out of {len(joined)} clean speed rows."
    )
    print(summary.to_string(index=False))
    print(category.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
