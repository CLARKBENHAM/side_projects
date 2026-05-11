"""Non-linear analysis of external ratings: band cutoffs, plots by category, and validation."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent
ENRICHED_FILE = OUTPUT_DIR / "books_enriched.csv"


def load() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_FILE)
    df.columns = df.columns.str.strip()
    return df


def validate_numbers(df: pd.DataFrame) -> None:
    """Sanity check the Open Library data quality."""
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    has_rating = df["gb_average_rating"].notna()
    has_pages = df["gb_page_count"].notna()
    has_pub = df["pub_year"].notna()

    print(f"\nTotal books: {len(df)}")
    print(f"With OL rating: {has_rating.sum()} ({has_rating.mean():.0%})")
    print(f"With page count: {has_pages.sum()} ({has_pages.mean():.0%})")
    print(f"With pub year: {has_pub.sum()} ({has_pub.mean():.0%})")

    # Check for suspicious matches by comparing titles
    if "ol_title" in df.columns:
        has_ol = df["ol_title"].notna() & (df["ol_title"] != "")
        print(f"With OL title match: {has_ol.sum()}")

        # Show some matches for spot-checking
        print("\nSample title matches (yours -> Open Library):")
        sample = df[has_ol].sample(min(10, has_ol.sum()), random_state=42)
        for _, row in sample.iterrows():
            print(f"  {str(row['title'])[:45]:<45} -> {str(row['ol_title'])[:45]}")

    # Rating distribution
    valid = df[has_rating]
    print("\nOL rating distribution:")
    print(
        f"  mean={valid['gb_average_rating'].mean():.2f}, "
        f"median={valid['gb_average_rating'].median():.2f}, "
        f"sd={valid['gb_average_rating'].std():.2f}"
    )
    print(
        f"  min={valid['gb_average_rating'].min():.2f}, "
        f"max={valid['gb_average_rating'].max():.2f}"
    )

    # Distribution of ratings in bands
    print("\nOL rating band distribution:")
    bins = [0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.01]
    labels = ["<2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]
    valid_copy = valid.copy()
    valid_copy["ol_band"] = pd.cut(
        valid_copy["gb_average_rating"], bins=bins, labels=labels
    )
    band_stats = valid_copy.groupby("ol_band", observed=True).agg(
        n=("Enjoyment (/5)", "count"),
        enjoy_mean=("Enjoyment (/5)", "mean"),
        enjoy_sd=("Enjoyment (/5)", "std"),
        useful_mean=("Usefulness /5 to Me", "mean"),
    )
    print(f"  {'Band':<10} {'N':>4} {'My Enjoy':>10} {'(sd)':>7} {'My Useful':>10}")
    print("  " + "-" * 45)
    for band, row in band_stats.iterrows():
        sd_str = f"({row['enjoy_sd']:.2f})" if pd.notna(row["enjoy_sd"]) else "(N/A)"
        print(
            f"  {band:<10} {row['n']:>4.0f} {row['enjoy_mean']:>10.2f} {sd_str:>7} "
            f"{row['useful_mean']:>10.2f}"
        )


def nonlinear_cutoffs(df: pd.DataFrame) -> None:
    """Test whether extreme OL ratings predict personal enjoyment better than linear."""
    print("\n" + "=" * 70)
    print("NON-LINEAR CUTOFF ANALYSIS")
    print("=" * 70)

    valid = df[df["gb_average_rating"].notna()].copy()

    # Test various cutoffs
    print("\nDoes an OL rating above/below X predict my enjoyment?")
    print(
        f"{'Cutoff':<20} {'N above':>8} {'Enjoy above':>12} {'N below':>8} "
        f"{'Enjoy below':>12} {'diff':>8} {'p':>8}"
    )
    print("-" * 80)

    for threshold in [3.0, 3.5, 3.8, 4.0, 4.2, 4.5]:
        above = valid[valid["gb_average_rating"] >= threshold]
        below = valid[valid["gb_average_rating"] < threshold]
        if len(above) >= 3 and len(below) >= 3:
            diff = above["Enjoyment (/5)"].mean() - below["Enjoyment (/5)"].mean()
            _, p = stats.ttest_ind(above["Enjoyment (/5)"], below["Enjoyment (/5)"])
            print(
                f"  OL >= {threshold:<12.1f} {len(above):>8} {above['Enjoyment (/5)'].mean():>12.2f} "
                f"{len(below):>8} {below['Enjoyment (/5)'].mean():>12.2f} {diff:>+8.2f} {p:>8.3f}"
            )

    # Same for usefulness
    print(
        f"\n{'Cutoff':<20} {'N above':>8} {'Useful above':>12} {'N below':>8} "
        f"{'Useful below':>12} {'diff':>8} {'p':>8}"
    )
    print("-" * 80)

    for threshold in [3.0, 3.5, 3.8, 4.0, 4.2, 4.5]:
        above = valid[valid["gb_average_rating"] >= threshold]
        below = valid[valid["gb_average_rating"] < threshold]
        if len(above) >= 3 and len(below) >= 3:
            diff = (
                above["Usefulness /5 to Me"].mean()
                - below["Usefulness /5 to Me"].mean()
            )
            _, p = stats.ttest_ind(
                above["Usefulness /5 to Me"], below["Usefulness /5 to Me"]
            )
            print(
                f"  OL >= {threshold:<12.1f} {len(above):>8} {above['Usefulness /5 to Me'].mean():>12.2f} "
                f"{len(below):>8} {below['Usefulness /5 to Me'].mean():>12.2f} {diff:>+8.2f} {p:>8.3f}"
            )

    # Bottom/top decile analysis
    print("\nExtreme OL ratings:")
    q10 = valid["gb_average_rating"].quantile(0.1)
    q90 = valid["gb_average_rating"].quantile(0.9)
    bottom = valid[valid["gb_average_rating"] <= q10]
    top = valid[valid["gb_average_rating"] >= q90]
    middle = valid[
        (valid["gb_average_rating"] > q10) & (valid["gb_average_rating"] < q90)
    ]
    print(
        f"  Bottom 10% (OL <= {q10:.2f}, n={len(bottom)}): "
        f"enjoy={bottom['Enjoyment (/5)'].mean():.2f}"
    )
    print(
        f"  Middle 80% (n={len(middle)}): enjoy={middle['Enjoyment (/5)'].mean():.2f}"
    )
    print(
        f"  Top 10% (OL >= {q90:.2f}, n={len(top)}): "
        f"enjoy={top['Enjoyment (/5)'].mean():.2f}"
    )


def plot_ratings_by_category(df: pd.DataFrame) -> None:
    """Plot personal enjoyment vs OL rating, faceted by category."""
    valid = df[df["gb_average_rating"].notna()].copy()

    major_cats = [
        "Business, management",
        "Computer Science",
        "fiction",
        "General Reading",
        "Literature",
    ]
    cat_data = valid[valid["Bookshelf"].isin(major_cats)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # All categories combined
    ax = axes[0]
    ax.scatter(
        valid["gb_average_rating"],
        valid["Enjoyment (/5)"],
        alpha=0.4,
        s=30,
        c="steelblue",
    )
    # Add regression line
    slope, intercept, r, p, se = stats.linregress(
        valid["gb_average_rating"], valid["Enjoyment (/5)"]
    )
    x_line = np.linspace(
        valid["gb_average_rating"].min(), valid["gb_average_rating"].max(), 100
    )
    ax.plot(
        x_line,
        slope * x_line + intercept,
        "r-",
        alpha=0.7,
        label=f"r={r:.2f}, p={p:.3f}",
    )
    ax.set_xlabel("Open Library Rating")
    ax.set_ylabel("My Enjoyment (/5)")
    ax.set_title(f"All Categories (n={len(valid)})")
    ax.legend(fontsize=8)
    ax.set_xlim(1, 5)
    ax.set_ylim(0.5, 5.5)
    ax.grid(alpha=0.3)

    for i, cat in enumerate(major_cats):
        ax = axes[i + 1]
        cat_df = cat_data[cat_data["Bookshelf"] == cat]
        if len(cat_df) < 5:
            ax.set_title(f"{cat} (n={len(cat_df)}, too few)")
            continue

        ax.scatter(
            cat_df["gb_average_rating"],
            cat_df["Enjoyment (/5)"],
            alpha=0.5,
            s=30,
            c="steelblue",
        )
        slope, intercept, r, p, se = stats.linregress(
            cat_df["gb_average_rating"], cat_df["Enjoyment (/5)"]
        )
        x_line = np.linspace(
            cat_df["gb_average_rating"].min(), cat_df["gb_average_rating"].max(), 100
        )
        ax.plot(
            x_line,
            slope * x_line + intercept,
            "r-",
            alpha=0.7,
            label=f"r={r:.2f}, p={p:.3f}",
        )
        ax.set_xlabel("Open Library Rating")
        ax.set_ylabel("My Enjoyment (/5)")
        ax.set_title(f"{cat} (n={len(cat_df)})")
        ax.legend(fontsize=8)
        ax.set_xlim(1, 5)
        ax.set_ylim(0.5, 5.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Personal Enjoyment vs Open Library Rating by Category", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "enjoyment_vs_ol_rating.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUTPUT_DIR / 'enjoyment_vs_ol_rating.png'}")
    plt.close()


def plot_bands(df: pd.DataFrame) -> None:
    """Box plots of personal enjoyment within OL rating bands."""
    valid = df[df["gb_average_rating"].notna()].copy()

    bins = [0, 3.0, 3.5, 4.0, 4.5, 5.01]
    labels = ["<3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"]
    valid["ol_band"] = pd.cut(valid["gb_average_rating"], bins=bins, labels=labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Enjoyment
    band_data_e = [
        valid[valid["ol_band"] == b]["Enjoyment (/5)"].dropna().values for b in labels
    ]
    bp1 = ax1.boxplot(band_data_e, labels=labels, patch_artist=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.5)
    counts = [len(d) for d in band_data_e]
    ax1.set_xlabel("Open Library Rating Band")
    ax1.set_ylabel("My Enjoyment (/5)")
    ax1.set_title("My Enjoyment by OL Rating Band")
    for i, c in enumerate(counts):
        ax1.text(i + 1, ax1.get_ylim()[0] + 0.1, f"n={c}", ha="center", fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    # Usefulness
    band_data_u = [
        valid[valid["ol_band"] == b]["Usefulness /5 to Me"].dropna().values
        for b in labels
    ]
    bp2 = ax2.boxplot(band_data_u, labels=labels, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("coral")
        patch.set_alpha(0.5)
    ax2.set_xlabel("Open Library Rating Band")
    ax2.set_ylabel("My Usefulness (/5)")
    ax2.set_title("My Usefulness by OL Rating Band")
    for i, c in enumerate(counts):
        ax2.text(i + 1, ax2.get_ylim()[0] + 0.1, f"n={c}", ha="center", fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rating_bands_boxplot.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_DIR / 'rating_bands_boxplot.png'}")
    plt.close()


def main() -> None:
    df = load()
    validate_numbers(df)
    nonlinear_cutoffs(df)
    plot_ratings_by_category(df)
    plot_bands(df)


if __name__ == "__main__":
    main()
