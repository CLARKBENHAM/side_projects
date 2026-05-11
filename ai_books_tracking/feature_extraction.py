"""Extract predictive features from the book data: author effects, temporal patterns, reading speed."""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PLAY_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
OUTPUT_DIR = Path(__file__).parent


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(PLAY_EXPORT)
    df.columns = df.columns.str.strip()

    # Parse dates
    for col in ["earliest_modified", "latest_modified"]:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    # Compute reading duration in days
    df["reading_days"] = (df["latest_modified"] - df["earliest_modified"]).dt.days
    # Books read in a single day get 0 days, set to 1 for rate calculations
    df["reading_days_adj"] = df["reading_days"].clip(lower=1)

    # Extract year finished
    df["year_finished"] = df["latest_modified"].dt.year

    # Extract month finished
    df["month_finished"] = df["latest_modified"].dt.month

    return df


def author_effects(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("AUTHOR EFFECTS")
    print("=" * 70)

    # Normalize author names
    df["author_clean"] = df["author"].str.strip().str.lower()

    author_stats = (
        df.groupby("author_clean")
        .agg(
            n=("Enjoyment (/5)", "count"),
            enjoy_mean=("Enjoyment (/5)", "mean"),
            enjoy_sd=("Enjoyment (/5)", "std"),
            useful_mean=("Usefulness /5 to Me", "mean"),
        )
        .sort_values("n", ascending=False)
    )

    repeat_authors = author_stats[author_stats["n"] >= 2]
    print(f"\n{len(repeat_authors)} authors with 2+ books:")
    print(f"{'Author':<35} {'N':>3} {'Enjoy':>7} {'(sd)':>7} {'Useful':>7}")
    print("-" * 65)
    for author, row in repeat_authors.iterrows():
        sd_str = f"({row['enjoy_sd']:.2f})" if pd.notna(row["enjoy_sd"]) else "(N/A)"
        print(
            f"{str(author)[:35]:<35} {row['n']:>3.0f} "
            f"{row['enjoy_mean']:>7.2f} {sd_str:>7} {row['useful_mean']:>7.2f}"
        )

    # Leave-one-out: for each book by a repeat author, does the author's
    # other-book mean predict this book's rating?
    repeat_mask = df["author_clean"].isin(repeat_authors.index)
    repeat_df = df[repeat_mask].copy()

    loo_predictions = []
    for idx, row in repeat_df.iterrows():
        other_books = repeat_df[
            (repeat_df["author_clean"] == row["author_clean"])
            & (repeat_df.index != idx)
        ]
        loo_predictions.append(
            {
                "title": row["title"],
                "actual_enjoy": row["Enjoyment (/5)"],
                "predicted_enjoy": other_books["Enjoyment (/5)"].mean(),
                "actual_useful": row["Usefulness /5 to Me"],
                "predicted_useful": other_books["Usefulness /5 to Me"].mean(),
            }
        )

    loo_df = pd.DataFrame(loo_predictions)
    if len(loo_df) > 1:
        r_enjoy = np.corrcoef(loo_df["actual_enjoy"], loo_df["predicted_enjoy"])[0, 1]
        r_useful = np.corrcoef(loo_df["actual_useful"], loo_df["predicted_useful"])[
            0, 1
        ]
        mae_enjoy = np.mean(np.abs(loo_df["actual_enjoy"] - loo_df["predicted_enjoy"]))
        mae_useful = np.mean(
            np.abs(loo_df["actual_useful"] - loo_df["predicted_useful"])
        )
        print(f"\nLeave-one-out prediction for repeat authors ({len(loo_df)} books):")
        print(f"  Enjoyment:  R={r_enjoy:.3f}, MAE={mae_enjoy:.2f}")
        print(f"  Usefulness: R={r_useful:.3f}, MAE={mae_useful:.2f}")

    return repeat_authors


def temporal_patterns(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TEMPORAL PATTERNS")
    print("=" * 70)

    # Year-over-year trends
    yearly = (
        df.groupby("year_finished")
        .agg(
            n=("Enjoyment (/5)", "count"),
            enjoy_mean=("Enjoyment (/5)", "mean"),
            useful_mean=("Usefulness /5 to Me", "mean"),
        )
        .dropna()
    )

    print("\nYear-over-year ratings:")
    print(f"{'Year':>6} {'N':>4} {'Enjoy':>7} {'Useful':>7}")
    print("-" * 30)
    for year, row in yearly.iterrows():
        print(
            f"{year:>6.0f} {row['n']:>4.0f} {row['enjoy_mean']:>7.2f} {row['useful_mean']:>7.2f}"
        )

    # Trend test
    valid = df.dropna(subset=["year_finished", "Enjoyment (/5)"])
    if len(valid) > 10:
        r_enjoy, p_enjoy = stats.spearmanr(
            valid["year_finished"], valid["Enjoyment (/5)"]
        )
        r_useful, p_useful = stats.spearmanr(
            valid["year_finished"], valid["Usefulness /5 to Me"]
        )
        print("\nSpearman trend with year:")
        print(f"  Enjoyment:  rho={r_enjoy:.3f}, p={p_enjoy:.4f}")
        print(f"  Usefulness: rho={r_useful:.3f}, p={p_useful:.4f}")

    # Seasonality (month)
    monthly = (
        df.groupby("month_finished")
        .agg(
            n=("Enjoyment (/5)", "count"),
            enjoy_mean=("Enjoyment (/5)", "mean"),
        )
        .dropna()
    )
    print("\nMonthly seasonality:")
    print(f"{'Month':>6} {'N':>4} {'Enjoy':>7}")
    print("-" * 20)
    for month, row in monthly.iterrows():
        print(f"{month:>6.0f} {row['n']:>4.0f} {row['enjoy_mean']:>7.2f}")

    # ANOVA for month effect
    month_groups = [
        group["Enjoyment (/5)"].values
        for _, group in df.dropna(subset=["month_finished"]).groupby("month_finished")
    ]
    if len(month_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*month_groups)
        print(f"\n  Month ANOVA: F={f_stat:.2f}, p={p_val:.4f}")


def reading_speed_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("READING SPEED / DURATION")
    print("=" * 70)

    valid = df.dropna(subset=["reading_days", "Enjoyment (/5)"])
    # Exclude books with no date range (reading_days == 0 or NaT)
    has_range = valid[valid["reading_days"] > 0]

    print(f"\n{len(has_range)} books with multi-day reading periods")
    print(
        f"Reading days: median={has_range['reading_days'].median():.0f}, "
        f"mean={has_range['reading_days'].mean():.1f}, "
        f"max={has_range['reading_days'].max():.0f}"
    )

    if len(has_range) > 10:
        r_enjoy, p_enjoy = stats.spearmanr(
            has_range["reading_days"], has_range["Enjoyment (/5)"]
        )
        r_useful, p_useful = stats.spearmanr(
            has_range["reading_days"], has_range["Usefulness /5 to Me"]
        )
        print("\nSpearman correlation with reading duration:")
        print(f"  Enjoyment:  rho={r_enjoy:.3f}, p={p_enjoy:.4f}")
        print(f"  Usefulness: rho={r_useful:.3f}, p={p_useful:.4f}")

    # Single-day vs multi-day reads
    single = valid[valid["reading_days"] == 0]
    multi = valid[valid["reading_days"] > 0]
    if len(single) > 5 and len(multi) > 5:
        t_stat, p_val = stats.ttest_ind(
            single["Enjoyment (/5)"], multi["Enjoyment (/5)"]
        )
        print(
            f"\nSingle-day reads (n={len(single)}): enjoy={single['Enjoyment (/5)'].mean():.2f}"
        )
        print(
            f"Multi-day reads  (n={len(multi)}):  enjoy={multi['Enjoyment (/5)'].mean():.2f}"
        )
        print(f"  t={t_stat:.2f}, p={p_val:.4f}")


def note_length_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("NOTE LENGTH (Long Term Effects)")
    print("=" * 70)

    df["note_length"] = df["Long Term Effects"].fillna("").str.len()
    df["has_note"] = df["note_length"] > 0

    with_notes = df[df["has_note"]]
    without_notes = df[~df["has_note"]]

    print(f"\nBooks with notes: {len(with_notes)}, without: {len(without_notes)}")
    if len(with_notes) > 5 and len(without_notes) > 5:
        print(
            f"With notes:    enjoy={with_notes['Enjoyment (/5)'].mean():.2f}, "
            f"useful={with_notes['Usefulness /5 to Me'].mean():.2f}"
        )
        print(
            f"Without notes: enjoy={without_notes['Enjoyment (/5)'].mean():.2f}, "
            f"useful={without_notes['Usefulness /5 to Me'].mean():.2f}"
        )

    valid = df[df["has_note"]]
    if len(valid) > 10:
        r_enjoy, p_enjoy = stats.spearmanr(
            valid["note_length"], valid["Enjoyment (/5)"]
        )
        r_useful, p_useful = stats.spearmanr(
            valid["note_length"], valid["Usefulness /5 to Me"]
        )
        print("\nCorrelation of note length with rating (among books with notes):")
        print(f"  Enjoyment:  rho={r_enjoy:.3f}, p={p_enjoy:.4f}")
        print(f"  Usefulness: rho={r_useful:.3f}, p={p_useful:.4f}")


def category_year_interaction(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CATEGORY x YEAR INTERACTION")
    print("=" * 70)

    major_cats = [
        "Business, management",
        "Computer Science",
        "fiction",
        "General Reading",
        "Literature",
    ]
    valid = df[df["Bookshelf"].isin(major_cats) & df["year_finished"].notna()].copy()

    print("\nPer-category trend (Spearman rho with year):")
    print(
        f"{'Category':<25} {'N':>4} {'rho_enjoy':>10} {'p':>8} {'rho_useful':>10} {'p':>8}"
    )
    print("-" * 70)
    for cat in major_cats:
        cat_df = valid[valid["Bookshelf"] == cat]
        if len(cat_df) > 5:
            r_e, p_e = stats.spearmanr(
                cat_df["year_finished"], cat_df["Enjoyment (/5)"]
            )
            r_u, p_u = stats.spearmanr(
                cat_df["year_finished"], cat_df["Usefulness /5 to Me"]
            )
            print(
                f"{cat:<25} {len(cat_df):>4} {r_e:>10.3f} {p_e:>8.4f} {r_u:>10.3f} {p_u:>8.4f}"
            )


def save_features(df: pd.DataFrame) -> None:
    """Save the enriched dataframe for use by the prediction model."""
    out_path = OUTPUT_DIR / "books_with_features.csv"
    cols_to_save = [
        "title",
        "author",
        "Bookshelf",
        "earliest_modified",
        "latest_modified",
        "Enjoyment (/5)",
        "Usefulness /5 to Me",
        "Long Term Effects",
        "reading_days",
        "reading_days_adj",
        "year_finished",
        "month_finished",
    ]
    existing_cols = [c for c in cols_to_save if c in df.columns]
    df[existing_cols].to_csv(out_path, index=False)
    print(f"\nSaved feature-enriched data to {out_path}")


def main() -> None:
    df = load_and_clean()
    author_effects(df)
    temporal_patterns(df)
    reading_speed_analysis(df)
    note_length_analysis(df)
    category_year_interaction(df)
    save_features(df)


if __name__ == "__main__":
    main()
