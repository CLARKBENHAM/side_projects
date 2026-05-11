"""Plot rating-distribution histograms by year and category.

Uses the baseline personal-ratings file in `data/` and joins external
Goodreads/Open Library/Amazon ratings from the multi-source consolidated file.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent
PROJECT_ROOT = OUTPUT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

BASELINE_PERSONAL_CSV = (
    DATA_DIR / "Books Read and their effects - master_book_metadata_cleaned_final.csv"
)
MULTI_SOURCE_CSV = OUTPUT_DIR / "golden_master_multi_source.csv"

PERSONAL_YEAR_PLOT = OUTPUT_DIR / "personal_ratings_histograms_by_year.png"
PERSONAL_CATEGORY_PLOT = OUTPUT_DIR / "personal_ratings_histograms_by_category.png"
EXTERNAL_YEAR_PLOT = OUTPUT_DIR / "external_ratings_histograms_by_year.png"
EXTERNAL_CATEGORY_PLOT = OUTPUT_DIR / "external_ratings_histograms_by_category.png"
PERSONAL_COUNTS_CSV = OUTPUT_DIR / "personal_rating_histogram_counts.csv"
EXTERNAL_COUNTS_CSV = OUTPUT_DIR / "external_rating_histogram_counts.csv"

PERSONAL_BINS = np.arange(0.875, 5.126, 0.25)
EXTERNAL_BINS = np.arange(2.4, 5.11, 0.2)

PERSONAL_ROWS = [
    ("avg_enjoyment", "Enjoyment", "#1f77b4"),
    ("avg_usefulness", "Usefulness", "#ff7f0e"),
]
EXTERNAL_ROWS = [
    ("goodreads_rating_verified", "Goodreads", "#355070"),
    ("ol_rating_consensus", "Open Library", "#6d597a"),
    ("amazon_rating_consensus", "Amazon", "#b56576"),
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("’", "'")
    text = re.sub(r"\.(pdf|epub|mobi|txt|html)$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wrap_label(value: object, width: int = 18) -> str:
    text = "NA" if pd.isna(value) else str(value)
    return textwrap.fill(text, width=width)


def load_personal_frame() -> pd.DataFrame:
    frame = pd.read_csv(BASELINE_PERSONAL_CSV)
    frame.columns = frame.columns.str.strip()
    frame["finished_dt"] = pd.to_datetime(frame["finished_date"], errors="coerce")
    frame["year"] = frame["finished_dt"].dt.year.astype("Int64")
    frame["year_label"] = frame["year"].astype(str).replace("<NA>", "Unknown")
    frame["category_label"] = frame["bookshelf"].fillna("NA").replace("", "NA")
    frame["filename_key"] = frame["filename"].fillna("").str.strip().str.lower()
    frame["title_key"] = frame["title"].map(normalize_text)
    frame["author_key"] = frame["corrected_author"].map(normalize_text)
    for column, _, _ in PERSONAL_ROWS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_external_frame() -> pd.DataFrame:
    frame = pd.read_csv(MULTI_SOURCE_CSV)
    frame.columns = frame.columns.str.strip()
    frame["filename_key"] = frame["filename"].fillna("").str.strip().str.lower()
    frame["title_key"] = frame["title"].map(normalize_text)
    frame["author_key"] = (
        frame["canonical_author"].fillna(frame["author"]).map(normalize_text)
    )
    keep_columns = [
        "filename_key",
        "title_key",
        "author_key",
        "goodreads_rating_verified",
        "ol_rating_consensus",
        "amazon_rating_consensus",
    ]
    frame = frame[keep_columns].copy()
    for column, _, _ in EXTERNAL_ROWS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def attach_external_ratings(personal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    filename_lookup = (
        external[external["filename_key"].ne("")]
        .drop_duplicates(subset=["filename_key"], keep="first")
        .copy()
    )
    merged = personal.merge(
        filename_lookup[
            ["filename_key", *[column for column, _, _ in EXTERNAL_ROWS]]
        ],
        on="filename_key",
        how="left",
    )

    fallback_lookup = external.drop_duplicates(
        subset=["title_key", "author_key"], keep="first"
    ).copy()
    needs_fallback = merged[[column for column, _, _ in EXTERNAL_ROWS]].isna().all(axis=1)
    if needs_fallback.any():
        fallback_join = merged.loc[needs_fallback, ["title_key", "author_key"]].merge(
            fallback_lookup[
                ["title_key", "author_key", *[column for column, _, _ in EXTERNAL_ROWS]]
            ],
            on=["title_key", "author_key"],
            how="left",
        )
        for column, _, _ in EXTERNAL_ROWS:
            merged.loc[needs_fallback, column] = fallback_join[column].to_numpy()
    return merged


def ordered_facets(frame: pd.DataFrame, facet_col: str) -> list[str]:
    if facet_col == "year_label":
        years = frame["year"].dropna().astype(int).sort_values().unique().tolist()
        labels = [str(year) for year in years]
        if frame["year"].isna().any():
            labels.append("Unknown")
        return labels
    counts = frame[facet_col].fillna("NA").value_counts()
    return counts.index.astype(str).tolist()


def summarize_counts(
    frame: pd.DataFrame,
    facet_col: str,
    rows: list[tuple[str, str, str]],
    scope_label: str,
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    facet_values = ordered_facets(frame, facet_col)
    total_rows = len(frame)
    for facet_value in facet_values:
        if facet_value == "Unknown" and facet_col == "year_label":
            subset = frame[frame["year"].isna()].copy()
        else:
            subset = frame[frame[facet_col].fillna("NA").astype(str) == facet_value].copy()
        for column, label, _ in rows:
            available = int(pd.to_numeric(subset[column], errors="coerce").notna().sum())
            summary_rows.append(
                {
                    "scope": scope_label,
                    "facet_type": facet_col.replace("_label", ""),
                    "facet_value": facet_value,
                    "series": label,
                    "total_rows": int(len(subset)),
                    "n_available": available,
                    "n_missing": int(len(subset) - available),
                    "missing_share": (
                        float((len(subset) - available) / len(subset))
                        if len(subset) > 0
                        else np.nan
                    ),
                    "overall_rows": total_rows,
                }
            )
    return pd.DataFrame(summary_rows)


def facet_figure_size(ncols: int, nrows: int) -> tuple[float, float]:
    return max(12.0, 3.25 * ncols), max(5.5, 2.9 * nrows)


def plot_histogram_grid(
    frame: pd.DataFrame,
    facet_col: str,
    rows: list[tuple[str, str, str]],
    bins: np.ndarray,
    x_label: str,
    title: str,
    output_path: Path,
) -> None:
    facet_values = ordered_facets(frame, facet_col)
    nrows = len(rows)
    ncols = max(1, len(facet_values))
    figsize = facet_figure_size(ncols, nrows)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, (column, row_label, color) in enumerate(rows):
        for col_index, facet_value in enumerate(facet_values):
            axis = axes[row_index, col_index]
            if facet_value == "Unknown" and facet_col == "year_label":
                subset = frame[frame["year"].isna()].copy()
            else:
                subset = frame[
                    frame[facet_col].fillna("NA").astype(str) == facet_value
                ].copy()
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            axis.hist(values, bins=bins, color=color, alpha=0.85, edgecolor="white")
            axis.grid(alpha=0.15, axis="y")
            axis.set_xlim(float(bins[0]), float(bins[-1]))

            if row_index == 0:
                axis.set_title(wrap_label(facet_value), fontsize=10)
            if col_index == 0:
                axis.set_ylabel(f"{row_label}\nCount")
            if row_index == nrows - 1:
                axis.set_xlabel(x_label)

            total = len(subset)
            available = int(values.notna().sum())
            missing = total - available
            axis.text(
                0.03,
                0.97,
                f"n={available}\nNA={missing}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    personal = load_personal_frame()
    external = load_external_frame()
    combined = attach_external_ratings(personal, external)

    personal_counts = pd.concat(
        [
            summarize_counts(personal, "year_label", PERSONAL_ROWS, "personal"),
            summarize_counts(personal, "category_label", PERSONAL_ROWS, "personal"),
        ],
        ignore_index=True,
    )
    external_counts = pd.concat(
        [
            summarize_counts(combined, "year_label", EXTERNAL_ROWS, "external"),
            summarize_counts(combined, "category_label", EXTERNAL_ROWS, "external"),
        ],
        ignore_index=True,
    )

    plot_histogram_grid(
        frame=personal,
        facet_col="year_label",
        rows=PERSONAL_ROWS,
        bins=PERSONAL_BINS,
        x_label="Personal rating",
        title="Personal rating distributions by year",
        output_path=PERSONAL_YEAR_PLOT,
    )
    plot_histogram_grid(
        frame=personal,
        facet_col="category_label",
        rows=PERSONAL_ROWS,
        bins=PERSONAL_BINS,
        x_label="Personal rating",
        title="Personal rating distributions by category",
        output_path=PERSONAL_CATEGORY_PLOT,
    )
    plot_histogram_grid(
        frame=combined,
        facet_col="year_label",
        rows=EXTERNAL_ROWS,
        bins=EXTERNAL_BINS,
        x_label="External site rating",
        title="External rating distributions by year",
        output_path=EXTERNAL_YEAR_PLOT,
    )
    plot_histogram_grid(
        frame=combined,
        facet_col="category_label",
        rows=EXTERNAL_ROWS,
        bins=EXTERNAL_BINS,
        x_label="External site rating",
        title="External rating distributions by category",
        output_path=EXTERNAL_CATEGORY_PLOT,
    )

    personal_counts.to_csv(PERSONAL_COUNTS_CSV, index=False)
    external_counts.to_csv(EXTERNAL_COUNTS_CSV, index=False)

    print(f"Saved {PERSONAL_YEAR_PLOT.name}")
    print(f"Saved {PERSONAL_CATEGORY_PLOT.name}")
    print(f"Saved {EXTERNAL_YEAR_PLOT.name}")
    print(f"Saved {EXTERNAL_CATEGORY_PLOT.name}")
    print(f"Saved {PERSONAL_COUNTS_CSV.name}")
    print(f"Saved {EXTERNAL_COUNTS_CSV.name}")


if __name__ == "__main__":
    main()
