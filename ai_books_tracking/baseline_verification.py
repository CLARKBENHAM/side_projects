"""Verify the baseline claims from quit-books.md against the actual data."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PLAY_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
RATINGS_2 = DATA_DIR / "Books Read and their effects - Ratings 2.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df1 = pd.read_csv(PLAY_EXPORT)
    df2 = pd.read_csv(RATINGS_2)
    # Clean column names
    for df in [df1, df2]:
        df.columns = df.columns.str.strip()
    return df1, df2


def verify_distribution_stats(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("1. DISTRIBUTION STATS VERIFICATION")
    print("=" * 70)

    enjoy = df["Enjoyment (/5)"].dropna()
    useful = df["Usefulness /5 to Me"].dropna()

    print(f"\nN books: {len(df)} (blog claims 207, Play Export has {len(df)} rows)")
    print(
        f"\nEnjoyment: mean={enjoy.mean():.2f} (blog: 3.4), sd={enjoy.std():.2f} (blog: 1.0)"
    )
    print(
        f"Usefulness: mean={useful.mean():.2f} (blog: 1.8), sd={useful.std():.2f} (blog: 1.1)"
    )

    print(f"\nEnjoyment range: [{enjoy.min()}, {enjoy.max()}]")
    print(f"Usefulness range: [{useful.min()}, {useful.max()}]")


def verify_category_counts(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("2. CATEGORY BREAKDOWN VERIFICATION")
    print("=" * 70)

    blog_counts = {
        "Business, management": 44,
        "Computer Science": 14,
        "fiction": 51,
        "General Reading": 39,
        "Literature": 50,
        "Machine Learning": 5,
        "Math": 3,
        "Uncategorized Shelf": 1,
    }
    actual_counts = df["Bookshelf"].value_counts()

    print(f"\n{'Category':<25} {'Blog':>6} {'Actual':>6} {'Match':>6}")
    print("-" * 50)
    total_blog = 0
    total_actual = 0
    for cat, blog_n in blog_counts.items():
        actual_n = actual_counts.get(cat, 0)
        match = "OK" if blog_n == actual_n else "DIFF"
        print(f"{cat:<25} {blog_n:>6} {actual_n:>6} {match:>6}")
        total_blog += blog_n
        total_actual += actual_n

    # Show any categories in data but not in blog
    for cat in actual_counts.index:
        if cat not in blog_counts:
            print(f"{cat:<25} {'N/A':>6} {actual_counts[cat]:>6} {'NEW':>6}")
            total_actual += actual_counts[cat]

    print(f"\n{'Total':<25} {total_blog:>6} {len(df):>6}")


def verify_retest_reliability(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("3. RE-TEST RELIABILITY VERIFICATION")
    print("=" * 70)

    # Match books between the two datasets by title
    merged = pd.merge(
        df1[["title", "Enjoyment (/5)", "Usefulness /5 to Me"]],
        df2[["title", "Enjoyment (/5)", "Usefulness /5 to Me"]],
        on="title",
        suffixes=("_1", "_2"),
    )

    print(f"\nMatched {len(merged)} books between Play Export and Ratings 2")

    enjoy1 = merged["Enjoyment (/5)_1"].astype(float)
    enjoy2 = merged["Enjoyment (/5)_2"].astype(float)
    useful1 = merged["Usefulness /5 to Me_1"].astype(float)
    useful2 = merged["Usefulness /5 to Me_2"].astype(float)

    # Drop rows with NaN
    enjoy_mask = enjoy1.notna() & enjoy2.notna()
    useful_mask = useful1.notna() & useful2.notna()

    r_enjoy = np.corrcoef(enjoy1[enjoy_mask], enjoy2[enjoy_mask])[0, 1]
    r_useful = np.corrcoef(useful1[useful_mask], useful2[useful_mask])[0, 1]

    l1_enjoy = np.mean(np.abs(enjoy1[enjoy_mask] - enjoy2[enjoy_mask]))
    l1_useful = np.mean(np.abs(useful1[useful_mask] - useful2[useful_mask]))

    rmse_enjoy = np.sqrt(np.mean((enjoy1[enjoy_mask] - enjoy2[enjoy_mask]) ** 2))
    rmse_useful = np.sqrt(np.mean((useful1[useful_mask] - useful2[useful_mask]) ** 2))

    print(f"\nEnjoyment R:  {r_enjoy:.2f} (blog: 0.77)")
    print(f"Usefulness R: {r_useful:.2f} (blog: 0.85)")
    print(f"\nEnjoyment L1:  {l1_enjoy:.2f} (blog: 0.46)")
    print(f"Usefulness L1: {l1_useful:.2f} (blog: 0.34)")
    print(f"\nEnjoyment RMSE:  {rmse_enjoy:.2f} (blog: 0.66)")
    print(f"Usefulness RMSE: {rmse_useful:.2f} (blog: 0.59)")

    # Show books with largest shifts
    merged["enjoy_diff"] = np.abs(enjoy1 - enjoy2)
    merged["useful_diff"] = np.abs(useful1 - useful2)

    print("\nBooks with largest enjoyment rating shifts (>1.0):")
    big_enjoy = merged[merged["enjoy_diff"] > 1.0].sort_values(
        "enjoy_diff", ascending=False
    )
    for _, row in big_enjoy.iterrows():
        print(
            f"  {row['title'][:50]:<50} "
            f"{row['Enjoyment (/5)_1']:.1f} -> {row['Enjoyment (/5)_2']:.1f} "
            f"(diff={row['enjoy_diff']:.1f})"
        )

    print("\nBooks with largest usefulness rating shifts (>1.5):")
    print("(blog claims 4 books shifted by >1.5)")
    big_useful = merged[merged["useful_diff"] > 1.5].sort_values(
        "useful_diff", ascending=False
    )
    print(f"  Found {len(big_useful)} books with >1.5 usefulness shift")
    for _, row in big_useful.iterrows():
        print(
            f"  {row['title'][:50]:<50} "
            f"{row['Usefulness /5 to Me_1']:.1f} -> {row['Usefulness /5 to Me_2']:.1f} "
            f"(diff={row['useful_diff']:.1f})"
        )


def verify_category_rankings(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("4. CATEGORY MEAN RATINGS")
    print("=" * 70)

    cat_stats = (
        df.groupby("Bookshelf")
        .agg(
            n=("Enjoyment (/5)", "count"),
            enjoy_mean=("Enjoyment (/5)", "mean"),
            enjoy_sd=("Enjoyment (/5)", "std"),
            useful_mean=("Usefulness /5 to Me", "mean"),
            useful_sd=("Usefulness /5 to Me", "std"),
        )
        .sort_values("enjoy_mean", ascending=False)
    )

    print("\nBy Enjoyment (descending):")
    print(
        f"{'Category':<25} {'N':>4} {'Enjoy':>7} {'(sd)':>7} {'Useful':>7} {'(sd)':>7}"
    )
    print("-" * 60)
    for cat, row in cat_stats.iterrows():
        print(
            f"{cat:<25} {row['n']:>4.0f} {row['enjoy_mean']:>7.2f} "
            f"({row['enjoy_sd']:>5.2f}) {row['useful_mean']:>7.2f} ({row['useful_sd']:>5.2f})"
        )

    print("\nBlog claims ranking: CS/Math > General > Business > Literature > Fiction")
    print("Blog claim: Literature > Fiction on average")
    lit = df[df["Bookshelf"] == "Literature"]
    fic = df[df["Bookshelf"] == "fiction"]
    print(
        f"  Literature enjoy={lit['Enjoyment (/5)'].mean():.2f} vs "
        f"Fiction enjoy={fic['Enjoyment (/5)'].mean():.2f}"
    )
    print(
        f"  Literature useful={lit['Usefulness /5 to Me'].mean():.2f} vs "
        f"Fiction useful={fic['Usefulness /5 to Me'].mean():.2f}"
    )


def main() -> None:
    df1, df2 = load_data()
    verify_distribution_stats(df1)
    verify_category_counts(df1)
    verify_retest_reliability(df1, df2)
    verify_category_rankings(df1)


if __name__ == "__main__":
    main()
