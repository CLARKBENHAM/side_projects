"""Analyze whether reading time and page count bias enjoyment/usefulness ratings.

Uses output from reading_time_extraction.py (book_reading_times.csv).

Key questions:
1. Does how long you spend reading a book affect your rating? (bias check)
2. Which books shift most when modeling usefulness/page vs raw usefulness?
3. How does the model change with per-page or per-minute normalization?
4. Time per category, pages per category, average time per page
5. Rolling average of reading speed over time

Outputs:
- Console analysis with statistics
- page_time_bias_results.csv: per-book data with normalized metrics
- reading_speed_rolling.png: rolling average of minutes/page over time
- category_reading_stats.png: time/pages/speed by category
- model_shift_analysis.csv: books that shift most under normalization
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = Path(__file__).parent
READING_TIMES_CSV = OUTPUT_DIR / "book_reading_times.csv"
GOLDEN_MASTER = OUTPUT_DIR / "golden_master_multi_source.csv"


# Known wrong page counts from Goodreads scraping (wrong edition or wrong book match)
PAGE_COUNT_CORRECTIONS = {
    "THE 48 LAWS OF POWER - Robert Greene.pdf": 452,
    "Knudsen": 336,
    "Harry Potter and the Methods of Rationality": 1967,
    "Surprised by Joy: The shape of my early life": 238,
    "The Good Research Code Handbook.pdf": 60,
}


def load_data():
    rt = pd.read_csv(READING_TIMES_CSV)
    rt["page_count_num"] = pd.to_numeric(rt["page_count"], errors="coerce")
    rt["avg_enjoyment"] = pd.to_numeric(rt["avg_enjoyment"], errors="coerce")
    rt["avg_usefulness"] = pd.to_numeric(rt["avg_usefulness"], errors="coerce")
    rt["goodreads_rating"] = pd.to_numeric(rt["goodreads_rating"], errors="coerce")
    rt["goodreads_rating_count"] = pd.to_numeric(rt["goodreads_rating_count"], errors="coerce")
    rt["total_reading_hours"] = pd.to_numeric(rt["total_reading_hours"], errors="coerce")
    rt["total_reading_minutes"] = pd.to_numeric(rt["total_reading_minutes"], errors="coerce")
    rt["minutes_per_page"] = pd.to_numeric(rt["minutes_per_page"], errors="coerce")
    rt["year_finished"] = pd.to_numeric(rt["year_finished"], errors="coerce")

    # Apply page count corrections
    for title, correct_pages in PAGE_COUNT_CORRECTIONS.items():
        mask = rt["title"] == title
        if mask.any():
            old = rt.loc[mask, "page_count_num"].iloc[0]
            rt.loc[mask, "page_count_num"] = correct_pages
            rt.loc[mask, "page_count"] = correct_pages
            print(f"  PAGE FIX: {title[:50]} {old} -> {correct_pages}")

    # Recompute derived columns after corrections
    has_both = rt["total_reading_hours"].notna() & rt["page_count_num"].notna() & (rt["page_count_num"] > 0)
    rt["total_reading_minutes"] = rt["total_reading_hours"] * 60
    rt.loc[has_both, "minutes_per_page"] = (
        rt.loc[has_both, "total_reading_minutes"] / rt.loc[has_both, "page_count_num"]
    )

    # Normalize category
    cat_col = "category" if "category" in rt.columns else "Bookshelf"
    rt["category_clean"] = rt[cat_col].fillna("Unknown").str.strip()
    cat_map = {
        "Business, management": "Business",
        "Computer Science": "CS",
        "Machine Learning": "ML",
        "General Reading": "General Reading",
        "Literature": "Literature",
        "fiction": "Fiction",
        "Math": "Math",
        "Histories": "Histories",
    }
    rt["category_clean"] = rt["category_clean"].map(lambda x: cat_map.get(x, x))

    return rt


def section_1_category_stats(df):
    """Time per category, pages per category, average time per page."""
    print("\n" + "=" * 70)
    print("SECTION 1: READING STATS BY CATEGORY (finished books with data)")
    print("=" * 70)

    has_time = df["total_reading_hours"].notna()
    has_pages = df["page_count_num"].notna() & (df["page_count_num"] > 0)
    has_both = has_time & has_pages

    # Category stats for books with reading time
    print("\n--- Books with calendar reading time ---")
    cat_time = (
        df[has_time]
        .groupby("category_clean")
        .agg(
            n=("total_reading_hours", "count"),
            mean_hours=("total_reading_hours", "mean"),
            median_hours=("total_reading_hours", "median"),
            total_hours=("total_reading_hours", "sum"),
            mean_sessions=("n_sessions", "mean"),
        )
        .sort_values("mean_hours", ascending=False)
    )
    print(f"\n{'Category':<20} {'N':>4} {'Mean h':>8} {'Med h':>8} {'Tot h':>8} {'Sess':>6}")
    print("-" * 60)
    for cat, row in cat_time.iterrows():
        print(
            f"{cat:<20} {row['n']:>4.0f} {row['mean_hours']:>8.1f} "
            f"{row['median_hours']:>8.1f} {row['total_hours']:>8.1f} {row['mean_sessions']:>6.1f}"
        )

    # Category stats for books with page count
    print("\n--- Books with page count ---")
    cat_pages = (
        df[has_pages]
        .groupby("category_clean")
        .agg(
            n=("page_count_num", "count"),
            mean_pages=("page_count_num", "mean"),
            median_pages=("page_count_num", "median"),
        )
        .sort_values("mean_pages", ascending=False)
    )
    print(f"\n{'Category':<20} {'N':>4} {'Mean pg':>8} {'Med pg':>8}")
    print("-" * 45)
    for cat, row in cat_pages.iterrows():
        print(f"{cat:<20} {row['n']:>4.0f} {row['mean_pages']:>8.0f} {row['median_pages']:>8.0f}")

    # Reading speed by category
    print("\n--- Reading speed (min/page) by category ---")
    cat_speed = (
        df[has_both]
        .groupby("category_clean")
        .agg(
            n=("minutes_per_page", "count"),
            mean_mpp=("minutes_per_page", "mean"),
            median_mpp=("minutes_per_page", "median"),
            std_mpp=("minutes_per_page", "std"),
        )
        .sort_values("mean_mpp", ascending=False)
    )
    print(f"\n{'Category':<20} {'N':>4} {'Mean':>8} {'Median':>8} {'SD':>8} {'Ratio':>8}")
    print("-" * 62)
    overall_median = df.loc[has_both, "minutes_per_page"].median()
    for cat, row in cat_speed.iterrows():
        ratio = row["mean_mpp"] / overall_median if overall_median > 0 else 0
        print(
            f"{cat:<20} {row['n']:>4.0f} {row['mean_mpp']:>8.2f} "
            f"{row['median_mpp']:>8.2f} {row['std_mpp']:>8.2f} {ratio:>8.1f}x"
        )
    print(f"\n  Overall median: {overall_median:.2f} min/page")

    return cat_time, cat_pages, cat_speed


def section_2_bias_analysis(df):
    """Test if ratings correlate with reading time / page count."""
    print("\n" + "=" * 70)
    print("SECTION 2: RATING BIAS ANALYSIS")
    print("=" * 70)
    print("Does how long you spend reading affect your rating?")

    has_time = df["total_reading_hours"].notna() & df["avg_enjoyment"].notna()
    has_pages = (
        df["page_count_num"].notna()
        & (df["page_count_num"] > 0)
        & df["avg_enjoyment"].notna()
    )
    has_both = has_time & has_pages & df["minutes_per_page"].notna()

    # Correlations
    print("\n--- Raw correlations (all books with data) ---")
    pairs = [
        ("total_reading_hours", "avg_enjoyment", has_time),
        ("total_reading_hours", "avg_usefulness", has_time & df["avg_usefulness"].notna()),
        ("page_count_num", "avg_enjoyment", has_pages),
        ("page_count_num", "avg_usefulness", has_pages & df["avg_usefulness"].notna()),
        ("minutes_per_page", "avg_enjoyment", has_both),
        ("minutes_per_page", "avg_usefulness", has_both & df["avg_usefulness"].notna()),
        ("n_sessions", "avg_enjoyment", has_time),
        ("n_sessions", "avg_usefulness", has_time & df["avg_usefulness"].notna()),
    ]

    print(f"\n{'X':<25} {'Y':<18} {'N':>4} {'Spearman':>10} {'p':>10} {'Pearson':>10}")
    print("-" * 82)
    for x, y, mask in pairs:
        sub = df[mask]
        if len(sub) < 5:
            continue
        rho, p = stats.spearmanr(sub[x], sub[y])
        r, _ = stats.pearsonr(sub[x], sub[y])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{x:<25} {y:<18} {len(sub):>4} {rho:>10.3f} {p:>10.4f} {r:>10.3f} {sig}")

    # Partial correlations controlling for category
    print("\n--- Partial correlations (controlling for category) ---")
    for x_col, y_col, mask in [
        ("total_reading_hours", "avg_enjoyment", has_time),
        ("total_reading_hours", "avg_usefulness", has_time & df["avg_usefulness"].notna()),
        ("page_count_num", "avg_enjoyment", has_pages),
        ("page_count_num", "avg_usefulness", has_pages & df["avg_usefulness"].notna()),
        ("minutes_per_page", "avg_enjoyment", has_both),
        ("minutes_per_page", "avg_usefulness", has_both & df["avg_usefulness"].notna()),
    ]:
        sub = df[mask].copy()
        if len(sub) < 10:
            continue
        cat_dummies = pd.get_dummies(sub["category_clean"], drop_first=True, dtype=float)
        # Residualize both x and y on category
        from numpy.linalg import lstsq
        X_cat = cat_dummies.values
        x_resid = sub[x_col].values - X_cat @ lstsq(X_cat, sub[x_col].values, rcond=None)[0]
        y_resid = sub[y_col].values - X_cat @ lstsq(X_cat, sub[y_col].values, rcond=None)[0]
        rho, p = stats.spearmanr(x_resid, y_resid)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {x_col:<25} ~ {y_col:<18} partial rho={rho:>7.3f} p={p:.4f} {sig}")

    # Within-category correlations
    print("\n--- Within-category: reading_hours ~ avg_enjoyment ---")
    for cat in sorted(df["category_clean"].unique()):
        sub = df[has_time & (df["category_clean"] == cat)]
        if len(sub) < 5:
            continue
        rho, p = stats.spearmanr(sub["total_reading_hours"], sub["avg_enjoyment"])
        print(f"  {cat:<20} n={len(sub):>3} rho={rho:>7.3f} p={p:.3f}")


def section_3_normalized_models(df):
    """Model usefulness/page and usefulness/minute vs raw usefulness."""
    print("\n" + "=" * 70)
    print("SECTION 3: NORMALIZED MODELS (usefulness/page, usefulness/minute)")
    print("=" * 70)

    has_all = (
        df["avg_usefulness"].notna()
        & df["goodreads_rating"].notna()
        & df["category_clean"].notna()
    )
    base = df[has_all].copy()

    has_pages = has_all & df["page_count_num"].notna() & (df["page_count_num"] > 0)
    has_time = has_all & df["total_reading_hours"].notna() & (df["total_reading_hours"] > 0)
    has_both = has_pages & has_time

    # Create normalized targets
    df_pages = df[has_pages].copy()
    df_pages["usefulness_per_page"] = df_pages["avg_usefulness"] / df_pages["page_count_num"]
    df_pages["enjoyment_per_page"] = df_pages["avg_enjoyment"] / df_pages["page_count_num"]

    df_time = df[has_time].copy()
    df_time["usefulness_per_minute"] = df_time["avg_usefulness"] / df_time["total_reading_minutes"]
    df_time["enjoyment_per_minute"] = df_time["avg_enjoyment"] / df_time["total_reading_minutes"]

    df_both = df[has_both].copy()
    df_both["usefulness_per_page"] = df_both["avg_usefulness"] / df_both["page_count_num"]
    df_both["usefulness_per_minute"] = df_both["avg_usefulness"] / df_both["total_reading_minutes"]

    # Build features for Ridge model
    def build_features(sub):
        cat_dummies = pd.get_dummies(sub["category_clean"], drop_first=True, dtype=float)
        feats = pd.DataFrame(index=sub.index)
        feats["goodreads_rating"] = sub["goodreads_rating"]
        feats["log_gr_count"] = np.log1p(sub["goodreads_rating_count"].fillna(0))
        feats = pd.concat([feats, cat_dummies], axis=1)
        return feats

    def fit_and_report(sub, target_col, label):
        feats = build_features(sub)
        y = sub[target_col].values
        scaler = StandardScaler()
        X = scaler.fit_transform(feats.values)
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        rho, p = stats.spearmanr(y, preds)
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
        print(f"  {label:<45} n={len(sub):>3} MAE={mae:.4f} rho={rho:.3f} R²={r2:.3f}")

        coef_dict = dict(zip(feats.columns, model.coef_ * scaler.scale_))
        return model, preds, coef_dict

    # Compare models
    print("\n--- Ridge models on common subset (books with pages + GR) ---")
    common_pages = df_pages[df_pages["goodreads_rating"].notna()].copy()
    if len(common_pages) >= 10:
        print(f"\n  Common subset: {len(common_pages)} books")
        _, raw_preds_u, raw_coefs = fit_and_report(common_pages, "avg_usefulness", "Raw usefulness")
        _, norm_preds_u, norm_coefs = fit_and_report(common_pages, "usefulness_per_page", "Usefulness / page")
        _, raw_preds_e, _ = fit_and_report(common_pages, "avg_enjoyment", "Raw enjoyment")
        _, norm_preds_e, _ = fit_and_report(common_pages, "enjoyment_per_page", "Enjoyment / page")

        # Coefficient comparison
        print("\n  Coefficient comparison (raw vs per-page usefulness):")
        print(f"    {'Feature':<30} {'Raw':>10} {'Per-page':>10} {'Diff':>10}")
        print("    " + "-" * 65)
        for feat in raw_coefs:
            raw_c = raw_coefs.get(feat, 0)
            norm_c = norm_coefs.get(feat, 0)
            print(f"    {feat:<30} {raw_c:>10.4f} {norm_c:>10.6f} {norm_c - raw_c:>+10.6f}")

    print("\n--- Ridge models on common subset (books with time + GR) ---")
    common_time = df_time[df_time["goodreads_rating"].notna()].copy()
    if len(common_time) >= 10:
        print(f"\n  Common subset: {len(common_time)} books")
        _, _, _ = fit_and_report(common_time, "avg_usefulness", "Raw usefulness")
        common_time["usefulness_per_minute"] = common_time["avg_usefulness"] / common_time["total_reading_minutes"]
        _, _, _ = fit_and_report(common_time, "usefulness_per_minute", "Usefulness / minute")
        _, _, _ = fit_and_report(common_time, "avg_enjoyment", "Raw enjoyment")
        common_time["enjoyment_per_minute"] = common_time["avg_enjoyment"] / common_time["total_reading_minutes"]
        _, _, _ = fit_and_report(common_time, "enjoyment_per_minute", "Enjoyment / minute")


def section_4_book_shifts(df):
    """Which books shift the most under normalization?"""
    print("\n" + "=" * 70)
    print("SECTION 4: BOOKS THAT SHIFT MOST UNDER NORMALIZATION")
    print("=" * 70)

    has_data = (
        df["avg_usefulness"].notna()
        & df["page_count_num"].notna()
        & (df["page_count_num"] > 0)
        & df["goodreads_rating"].notna()
    )
    sub = df[has_data].copy()
    if len(sub) < 10:
        print("  Not enough data for shift analysis.")
        return None

    # Rank by raw usefulness vs usefulness/page
    sub["usefulness_per_page"] = sub["avg_usefulness"] / sub["page_count_num"]
    sub["raw_rank"] = sub["avg_usefulness"].rank(ascending=False)
    sub["norm_rank"] = sub["usefulness_per_page"].rank(ascending=False)
    sub["rank_shift"] = sub["raw_rank"] - sub["norm_rank"]

    # Also do per-minute if available
    has_time = sub["total_reading_hours"].notna() & (sub["total_reading_hours"] > 0)
    if has_time.sum() >= 10:
        sub_time = sub[has_time].copy()
        sub_time["usefulness_per_minute"] = sub_time["avg_usefulness"] / sub_time["total_reading_minutes"]
        sub_time["raw_rank_t"] = sub_time["avg_usefulness"].rank(ascending=False)
        sub_time["time_norm_rank"] = sub_time["usefulness_per_minute"].rank(ascending=False)
        sub_time["time_rank_shift"] = sub_time["raw_rank_t"] - sub_time["time_norm_rank"]

    print("\n--- Biggest rank GAINS when normalizing by pages (short useful books rise) ---")
    top_gains = sub.nlargest(10, "rank_shift")
    print(f"  {'Title':<45} {'Pages':>5} {'Raw U':>6} {'U/pg':>8} {'Raw#':>5} {'Norm#':>5} {'Shift':>6}")
    print("  " + "-" * 85)
    for _, row in top_gains.iterrows():
        print(
            f"  {str(row['title'])[:45]:<45} {row['page_count_num']:>5.0f} "
            f"{row['avg_usefulness']:>6.2f} {row['usefulness_per_page']:>8.4f} "
            f"{row['raw_rank']:>5.0f} {row['norm_rank']:>5.0f} {row['rank_shift']:>+6.0f}"
        )

    print("\n--- Biggest rank DROPS when normalizing by pages (long useful books fall) ---")
    top_drops = sub.nsmallest(10, "rank_shift")
    print(f"  {'Title':<45} {'Pages':>5} {'Raw U':>6} {'U/pg':>8} {'Raw#':>5} {'Norm#':>5} {'Shift':>6}")
    print("  " + "-" * 85)
    for _, row in top_drops.iterrows():
        print(
            f"  {str(row['title'])[:45]:<45} {row['page_count_num']:>5.0f} "
            f"{row['avg_usefulness']:>6.2f} {row['usefulness_per_page']:>8.4f} "
            f"{row['raw_rank']:>5.0f} {row['norm_rank']:>5.0f} {row['rank_shift']:>+6.0f}"
        )

    if has_time.sum() >= 10:
        print("\n--- Biggest rank GAINS when normalizing by time (fast useful books rise) ---")
        top_time_gains = sub_time.nlargest(10, "time_rank_shift")
        print(f"  {'Title':<45} {'Hours':>5} {'Raw U':>6} {'U/min':>8} {'Shift':>6}")
        print("  " + "-" * 75)
        for _, row in top_time_gains.iterrows():
            print(
                f"  {str(row['title'])[:45]:<45} {row['total_reading_hours']:>5.1f} "
                f"{row['avg_usefulness']:>6.2f} {row['usefulness_per_minute']:>8.5f} {row['time_rank_shift']:>+6.0f}"
            )

        print("\n--- Biggest rank DROPS when normalizing by time (slow useful books fall) ---")
        top_time_drops = sub_time.nsmallest(10, "time_rank_shift")
        print(f"  {'Title':<45} {'Hours':>5} {'Raw U':>6} {'U/min':>8} {'Shift':>6}")
        print("  " + "-" * 75)
        for _, row in top_time_drops.iterrows():
            print(
                f"  {str(row['title'])[:45]:<45} {row['total_reading_hours']:>5.1f} "
                f"{row['avg_usefulness']:>6.2f} {row['usefulness_per_minute']:>8.5f} {row['time_rank_shift']:>+6.0f}"
            )

    # Save shift data
    out_cols = [
        "title", "category_clean", "page_count_num", "total_reading_hours",
        "avg_enjoyment", "avg_usefulness", "usefulness_per_page",
        "raw_rank", "norm_rank", "rank_shift",
    ]
    out_cols = [c for c in out_cols if c in sub.columns]
    sub[out_cols].sort_values("rank_shift", ascending=False).to_csv(
        OUTPUT_DIR / "model_shift_analysis.csv", index=False
    )
    return sub


def section_5_rolling_speed(df):
    """Rolling average of reading speed over time."""
    print("\n" + "=" * 70)
    print("SECTION 5: ROLLING AVERAGE OF READING SPEED")
    print("=" * 70)

    has_both = (
        df["minutes_per_page"].notna()
        & df["last_session"].notna()
        & (df["minutes_per_page"] > 0)
        & (df["minutes_per_page"] < 10)  # exclude extreme outliers
    )
    sub = df[has_both].copy()
    sub["last_session_dt"] = pd.to_datetime(sub["last_session"])
    sub = sub.sort_values("last_session_dt")

    print(f"\n  Books with valid speed data: {len(sub)}")
    print(f"  Date range: {sub['last_session_dt'].min().date()} to {sub['last_session_dt'].max().date()}")
    print(f"  Speed range: {sub['minutes_per_page'].min():.2f} to {sub['minutes_per_page'].max():.2f} min/pg")

    # Check large residuals from rolling median
    if len(sub) >= 10:
        rolling_med = sub["minutes_per_page"].rolling(10, center=True, min_periods=5).median()
        resid = (sub["minutes_per_page"] - rolling_med).abs()
        large_resid = sub[resid > 1.5]
        if len(large_resid) > 0:
            print(f"\n  Books with speed > 1.5 min/pg from rolling median (potential errors):")
            for _, row in large_resid.iterrows():
                print(
                    f"    {str(row['title'])[:45]:<45} "
                    f"{row['minutes_per_page']:.2f} min/pg "
                    f"({row['total_reading_hours']:.1f}h, {row['page_count_num']:.0f}pg) "
                    f"cat={row['category_clean']}"
                )

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.scatter(
        sub["last_session_dt"], sub["minutes_per_page"],
        c=sub["category_clean"].astype("category").cat.codes,
        cmap="tab10", alpha=0.6, s=40, edgecolors="white", linewidth=0.5,
    )

    # Rolling averages
    if len(sub) >= 5:
        roll5 = sub.set_index("last_session_dt")["minutes_per_page"].rolling("180D", min_periods=3).median()
        ax.plot(roll5.index, roll5.values, "r-", linewidth=2, label="6-month rolling median")

    if len(sub) >= 10:
        roll10 = sub.set_index("last_session_dt")["minutes_per_page"].rolling("365D", min_periods=5).median()
        ax.plot(roll10.index, roll10.values, "b-", linewidth=2, label="1-year rolling median")

    # Category legend
    cats = sub["category_clean"].unique()
    cat_codes = sub["category_clean"].astype("category").cat
    cmap = plt.cm.tab10
    for i, cat in enumerate(cat_codes.categories):
        ax.scatter([], [], c=[cmap(i)], label=cat, s=40)

    ax.set_ylabel("Minutes per page")
    ax.set_title("Reading Speed Over Time (minutes per page)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, max(sub["minutes_per_page"].max() * 1.1, 5))
    ax.grid(True, alpha=0.3)

    # Bottom panel: book count per year
    ax2 = axes[1]
    sub["year"] = sub["last_session_dt"].dt.year
    year_counts = sub.groupby("year").size()
    ax2.bar(year_counts.index, year_counts.values, color="steelblue", alpha=0.7)
    ax2.set_ylabel("Books finished")
    ax2.set_xlabel("Year")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "reading_speed_rolling.png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved reading_speed_rolling.png")
    plt.close()


def section_6_category_plots(df):
    """Bar charts of time/pages/speed by category."""
    has_time = df["total_reading_hours"].notna()
    has_pages = df["page_count_num"].notna() & (df["page_count_num"] > 0)
    has_both = has_time & has_pages

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Mean reading hours by category
    cat_hours = (
        df[has_time]
        .groupby("category_clean")["total_reading_hours"]
        .agg(["mean", "count", "std"])
        .sort_values("mean", ascending=True)
    )
    cat_hours = cat_hours[cat_hours["count"] >= 3]
    ax = axes[0]
    bars = ax.barh(cat_hours.index, cat_hours["mean"], xerr=cat_hours["std"] / np.sqrt(cat_hours["count"]), color="steelblue", alpha=0.7)
    for i, (cat, row) in enumerate(cat_hours.iterrows()):
        ax.text(row["mean"] + 0.2, i, f"n={row['count']:.0f}", va="center", fontsize=9)
    ax.set_xlabel("Mean reading hours")
    ax.set_title("Reading Time by Category")

    # Panel 2: Mean pages by category
    cat_pages = (
        df[has_pages]
        .groupby("category_clean")["page_count_num"]
        .agg(["mean", "count", "std"])
        .sort_values("mean", ascending=True)
    )
    cat_pages = cat_pages[cat_pages["count"] >= 3]
    ax = axes[1]
    ax.barh(cat_pages.index, cat_pages["mean"], xerr=cat_pages["std"] / np.sqrt(cat_pages["count"]), color="coral", alpha=0.7)
    for i, (cat, row) in enumerate(cat_pages.iterrows()):
        ax.text(row["mean"] + 5, i, f"n={row['count']:.0f}", va="center", fontsize=9)
    ax.set_xlabel("Mean page count")
    ax.set_title("Page Count by Category")

    # Panel 3: Mean speed by category
    cat_speed = (
        df[has_both]
        .groupby("category_clean")["minutes_per_page"]
        .agg(["mean", "count", "std"])
        .sort_values("mean", ascending=True)
    )
    cat_speed = cat_speed[cat_speed["count"] >= 3]
    ax = axes[2]
    ax.barh(cat_speed.index, cat_speed["mean"], xerr=cat_speed["std"] / np.sqrt(cat_speed["count"]), color="seagreen", alpha=0.7)
    for i, (cat, row) in enumerate(cat_speed.iterrows()):
        ax.text(row["mean"] + 0.02, i, f"n={row['count']:.0f}", va="center", fontsize=9)
    ax.set_xlabel("Mean minutes per page")
    ax.set_title("Reading Speed by Category")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "category_reading_stats.png", dpi=150, bbox_inches="tight")
    print(f"\n  Saved category_reading_stats.png")
    plt.close()


def section_scatter_time_vs_pages(df):
    """Large scatter plot of reading time vs pages, colored by category, with outlier labels."""
    print("\n" + "=" * 70)
    print("SCATTER: READING TIME vs PAGES BY CATEGORY")
    print("=" * 70)

    has_both = (
        df["total_reading_hours"].notna()
        & df["page_count_num"].notna()
        & (df["page_count_num"] > 0)
        & (df["total_reading_hours"] > 0)
    )
    sub = df[has_both].copy()
    print(f"  Books with both time + pages: {len(sub)}")

    if len(sub) < 5:
        print("  Not enough data for scatter plot.")
        return

    # Fit overall regression line
    from numpy.polynomial import polynomial as P
    log_pages = np.log(sub["page_count_num"])
    log_hours = np.log(sub["total_reading_hours"])
    coefs = np.polyfit(log_pages, log_hours, 1)
    pred_log_hours = np.polyval(coefs, log_pages)
    residuals = log_hours - pred_log_hours
    sub["log_residual"] = residuals.values

    # Also fit per-category lines for residual calculation
    cat_residuals = []
    for cat in sub["category_clean"].unique():
        cat_mask = sub["category_clean"] == cat
        if cat_mask.sum() >= 3:
            lp = log_pages[cat_mask]
            lh = log_hours[cat_mask]
            c = np.polyfit(lp, lh, 1)
            r = lh - np.polyval(c, lp)
            cat_residuals.extend(r.values)
        else:
            cat_residuals.extend(residuals[cat_mask].values)
    sub["cat_residual"] = cat_residuals

    # Identify outliers: large residual in either direction
    resid_threshold = 1.0  # in log-space, ~2.7x error
    outliers = sub[sub["log_residual"].abs() > resid_threshold].copy()
    # Also flag books with extreme X or Y values
    high_hours = sub.nlargest(5, "total_reading_hours")
    high_pages = sub.nlargest(5, "page_count_num")
    low_speed = sub.nsmallest(3, "minutes_per_page")
    high_speed = sub.nlargest(5, "minutes_per_page")
    label_set = set(outliers.index) | set(high_hours.index) | set(high_pages.index) | set(high_speed.index) | set(low_speed.index)

    # Create very large plot
    fig, ax = plt.subplots(figsize=(24, 18))

    categories = sorted(sub["category_clean"].unique())
    colors = {
        "Business": "#2196F3",
        "CS": "#FF5722",
        "Fiction": "#9C27B0",
        "General Reading": "#4CAF50",
        "Histories": "#FF9800",
        "Literature": "#E91E63",
        "Math": "#00BCD4",
        "ML": "#795548",
    }

    for cat in categories:
        cat_mask = sub["category_clean"] == cat
        cat_df = sub[cat_mask]
        color = colors.get(cat, "gray")
        ax.scatter(
            cat_df["page_count_num"], cat_df["total_reading_hours"],
            c=color, label=cat, s=80, alpha=0.7, edgecolors="white", linewidth=0.5,
            zorder=3,
        )

        # Per-category regression line
        if cat_mask.sum() >= 3:
            x_range = np.linspace(cat_df["page_count_num"].min(), cat_df["page_count_num"].max(), 50)
            lp = np.log(cat_df["page_count_num"])
            lh = np.log(cat_df["total_reading_hours"])
            c = np.polyfit(lp, lh, 1)
            ax.plot(x_range, np.exp(np.polyval(c, np.log(x_range))),
                    color=color, alpha=0.4, linewidth=1.5, linestyle="--")

    # Overall regression line
    x_range = np.linspace(sub["page_count_num"].min(), sub["page_count_num"].max(), 100)
    ax.plot(x_range, np.exp(np.polyval(coefs, np.log(x_range))),
            color="black", alpha=0.5, linewidth=2, linestyle="-", label="Overall fit")

    # Label outliers and extremes
    for idx in label_set:
        row = sub.loc[idx]
        title = str(row["title"])
        # Truncate title
        if len(title) > 40:
            title = title[:37] + "..."
        speed_str = f"{row['minutes_per_page']:.1f}m/p" if pd.notna(row.get("minutes_per_page")) else ""
        label = f"{title}\n({speed_str})"

        ax.annotate(
            label,
            xy=(row["page_count_num"], row["total_reading_hours"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7, alpha=0.85,
            arrowprops=dict(arrowstyle="-", alpha=0.4, linewidth=0.5),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7),
        )

    ax.set_xlabel("Page Count", fontsize=14)
    ax.set_ylabel("Total Reading Hours (calendar)", fontsize=14)
    ax.set_title("Reading Time vs Page Count by Category\n(labeled: outliers + extremes)", fontsize=16)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, sub["page_count_num"].max() * 1.05)
    ax.set_ylim(0, sub["total_reading_hours"].max() * 1.05)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "time_vs_pages_scatter.png", dpi=150, bbox_inches="tight")
    print(f"  Saved time_vs_pages_scatter.png ({len(label_set)} books labeled)")
    plt.close()

    # Print the labeled outliers
    print(f"\n  Labeled outliers (log-residual > {resid_threshold}):")
    for _, row in outliers.sort_values("log_residual", ascending=False).iterrows():
        direction = "SLOW" if row["log_residual"] > 0 else "FAST"
        print(
            f"    [{direction}] {str(row['title'])[:45]:<45} "
            f"{row['page_count_num']:.0f}pg {row['total_reading_hours']:.1f}h "
            f"{row['minutes_per_page']:.2f}m/p cat={row['category_clean']}"
        )


def section_7_model_comparison(df):
    """Compare raw vs normalized models more formally."""
    print("\n" + "=" * 70)
    print("SECTION 7: FORMAL MODEL COMPARISON (raw vs normalized targets)")
    print("=" * 70)

    # Load full golden master for amazon ratings
    gm = pd.read_csv(GOLDEN_MASTER)
    gm.columns = gm.columns.str.strip()

    # Get amazon ratings
    amazon_cols = [c for c in gm.columns if "amazon_link_rating" in c.lower() or "amazon_nolink_rating" in c.lower()]
    if "amazon_link_rating" in gm.columns:
        gm["amazon_rating"] = pd.to_numeric(gm["amazon_link_rating"], errors="coerce")
    elif "amazon_nolink_rating" in gm.columns:
        gm["amazon_rating"] = pd.to_numeric(gm["amazon_nolink_rating"], errors="coerce")

    # Merge amazon into df
    if "amazon_rating" in gm.columns:
        amazon_map = dict(zip(gm["title"], gm["amazon_rating"]))
        df["amazon_rating"] = df["title"].map(amazon_map)

    has_all = (
        df["avg_usefulness"].notna()
        & df["avg_enjoyment"].notna()
        & df["goodreads_rating"].notna()
    )
    has_pages = has_all & df["page_count_num"].notna() & (df["page_count_num"] > 0)
    has_time = has_all & df["total_reading_hours"].notna() & (df["total_reading_hours"] > 0)

    def loo_ridge(sub, target_col, feature_cols):
        """Leave-one-out Ridge regression."""
        X = sub[feature_cols].values
        y = sub[target_col].values
        preds = np.zeros(len(y))
        for i in range(len(y)):
            mask = np.ones(len(y), dtype=bool)
            mask[i] = False
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[mask])
            X_test = scaler.transform(X[i:i+1])
            model = Ridge(alpha=1.0)
            model.fit(X_train, y[mask])
            preds[i] = model.predict(X_test)[0]
        mae = mean_absolute_error(y, preds)
        rho, p = stats.spearmanr(y, preds)
        return mae, rho, p

    # Build feature columns
    def get_feature_cols(sub):
        cat_dummies = pd.get_dummies(sub["category_clean"], drop_first=True, dtype=float)
        for c in cat_dummies.columns:
            sub[c] = cat_dummies[c]
        feats = ["goodreads_rating", "log_gr_count"] + list(cat_dummies.columns)
        sub["log_gr_count"] = np.log1p(sub["goodreads_rating_count"].fillna(0))
        if "amazon_rating" in sub.columns and sub["amazon_rating"].notna().sum() > len(sub) * 0.5:
            sub["amazon_rating_filled"] = sub["amazon_rating"].fillna(sub["amazon_rating"].median())
            feats.append("amazon_rating_filled")
        return feats

    # LOO comparison on pages subset
    print("\n--- LOO Ridge on books with page count ---")
    sub_p = df[has_pages].copy()
    if len(sub_p) >= 15:
        feat_cols = get_feature_cols(sub_p)
        sub_p["usefulness_per_page"] = sub_p["avg_usefulness"] / sub_p["page_count_num"]
        sub_p["enjoyment_per_page"] = sub_p["avg_enjoyment"] / sub_p["page_count_num"]

        for target, label in [
            ("avg_usefulness", "Raw usefulness"),
            ("usefulness_per_page", "Usefulness/page"),
            ("avg_enjoyment", "Raw enjoyment"),
            ("enjoyment_per_page", "Enjoyment/page"),
        ]:
            mae, rho, p = loo_ridge(sub_p, target, feat_cols)
            print(f"  {label:<25} LOO MAE={mae:.4f} rho={rho:.3f} (p={p:.4f})")

    # LOO comparison on time subset
    print("\n--- LOO Ridge on books with reading time ---")
    sub_t = df[has_time].copy()
    if len(sub_t) >= 15:
        feat_cols = get_feature_cols(sub_t)
        sub_t["usefulness_per_minute"] = sub_t["avg_usefulness"] / sub_t["total_reading_minutes"]
        sub_t["enjoyment_per_minute"] = sub_t["avg_enjoyment"] / sub_t["total_reading_minutes"]

        for target, label in [
            ("avg_usefulness", "Raw usefulness"),
            ("usefulness_per_minute", "Usefulness/minute"),
            ("avg_enjoyment", "Raw enjoyment"),
            ("enjoyment_per_minute", "Enjoyment/minute"),
        ]:
            mae, rho, p = loo_ridge(sub_t, target, feat_cols)
            print(f"  {label:<25} LOO MAE={mae:.4f} rho={rho:.3f} (p={p:.4f})")


def main():
    print("=" * 70)
    print("PAGE/TIME BIAS ANALYSIS FOR BOOK RATINGS")
    print("=" * 70)

    df = load_data()
    print(f"\nLoaded {len(df)} books from {READING_TIMES_CSV}")
    print(f"  With reading time: {df['total_reading_hours'].notna().sum()}")
    print(f"  With page count: {(df['page_count_num'].notna() & (df['page_count_num'] > 0)).sum()}")
    print(f"  With both: {(df['total_reading_hours'].notna() & df['page_count_num'].notna() & (df['page_count_num'] > 0)).sum()}")

    cat_time, cat_pages, cat_speed = section_1_category_stats(df)
    section_2_bias_analysis(df)
    section_3_normalized_models(df)
    shift_df = section_4_book_shifts(df)
    section_5_rolling_speed(df)
    section_6_category_plots(df)
    section_scatter_time_vs_pages(df)
    section_7_model_comparison(df)

    # Save full results
    out_cols = [
        "title", "category_clean", "page_count_num", "total_reading_hours",
        "total_reading_minutes", "minutes_per_page", "n_sessions",
        "avg_enjoyment", "avg_usefulness", "goodreads_rating",
        "first_session", "last_session", "year_finished",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(OUTPUT_DIR / "page_time_bias_results.csv", index=False)
    print(f"\nSaved page_time_bias_results.csv")


if __name__ == "__main__":
    main()
