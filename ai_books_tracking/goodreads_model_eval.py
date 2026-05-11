"""Evaluate prediction models using Goodreads ratings.

Trains on the first 178 books (chronologically) from the goodreads-enriched
208-book dataset, and evaluates on the last 30. Compares models with and
without Goodreads features, plus decision-rule analysis using utility functions.

Utility model: base^(rating - 1) - 1, so a rating of 1 yields 0 utility.
"""

import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path(__file__).parent
ENRICHED_GR = OUTPUT_DIR / "books_enriched_with_goodreads.csv"

HOLDOUT_N = 30
ENJOY_BASE = 1.3
USEFUL_BASE = 1.8

# Shared GR band definitions (used in model, correlation, and plot code)
GR_BAND_BINS = [0, 3.5, 3.8, 4.0, 4.2, 4.5, 5.01]
GR_BAND_LABELS = ["<3.5", "3.5-3.8", "3.8-4.0", "4.0-4.2", "4.2-4.5", "4.5+"]

TARGETS = [
    ("Enjoyment (/5)", "Enjoyment"),
    ("Usefulness /5 to Me", "Usefulness"),
]


def enjoy_utility(r: np.ndarray | float) -> np.ndarray | float:
    return ENJOY_BASE ** (np.asarray(r, dtype=float) - 1) - 1


def useful_utility(r: np.ndarray | float) -> np.ndarray | float:
    return USEFUL_BASE ** (np.asarray(r, dtype=float) - 1) - 1


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_GR)
    df.columns = df.columns.str.strip()
    df["date_sort"] = pd.to_datetime(
        df["latest_modified"], format="mixed", errors="coerce"
    )
    df["gr_rating"] = pd.to_numeric(df["goodreads_rating_raw_best"], errors="coerce")
    df["gr_count"] = pd.to_numeric(
        df["goodreads_rating_count_raw_best"], errors="coerce"
    )
    df["gr_log_count"] = np.log1p(df["gr_count"].fillna(0))
    df["ol_rating"] = pd.to_numeric(df["gb_average_rating"], errors="coerce")
    df["page_count"] = pd.to_numeric(df["gb_page_count"], errors="coerce")
    df["log_pages"] = np.log1p(df["page_count"].fillna(0))
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["note_length"] = df["Long Term Effects"].fillna("").str.len()
    return df


def eval_preds(actual: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, preds)
    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
    if len(actual) > 2 and np.std(preds) > 1e-10:
        r = float(stats.pearsonr(actual, preds)[0])
    else:
        r = float("nan")
    return {"MAE": mae, "RMSE": rmse, "R": r}


def format_r(r: float) -> str:
    return f"{r:>6.3f}" if not np.isnan(r) else "   nan"


def print_table(results: dict[str, dict[str, float]], title: str) -> None:
    print(f"\n  {title}")
    print(f"    {'Model':<50} {'MAE':>6} {'RMSE':>6} {'R':>6}")
    print("    " + "-" * 70)
    for name, m in results.items():
        print(f"    {name:<50} {m['MAE']:>6.3f} {m['RMSE']:>6.3f} {format_r(m['R'])}")


def assign_gr_bands(series: pd.Series) -> pd.Series:
    """Bin a GR rating series into bands. Returns a float-typed categorical."""
    return pd.cut(series, bins=GR_BAND_BINS, labels=GR_BAND_LABELS).astype(str)


# ---- Models ----


def run_all_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "Enjoyment (/5)",
) -> dict[str, dict[str, float]]:
    actual = test[target].values
    results: dict[str, dict[str, float]] = {}

    global_mean = train[target].mean()
    results["Global mean"] = eval_preds(actual, np.full(len(test), global_mean))

    cat_means = train.groupby("Bookshelf")[target].mean().to_dict()
    cat_preds = test["Bookshelf"].map(cat_means).fillna(global_mean).values
    results["Category mean"] = eval_preds(actual, cat_preds)

    # GR linear (global and per-category)
    train_gr = train[train["gr_rating"].notna()].copy()
    if len(train_gr) >= 10:
        slope, intercept, _, _, _ = stats.linregress(
            train_gr["gr_rating"], train_gr[target]
        )
        gr_preds = np.where(
            test["gr_rating"].notna(),
            slope * test["gr_rating"] + intercept,
            global_mean,
        )
        results["Goodreads rating (linear)"] = eval_preds(actual, gr_preds)

        # Per-category linear: use category-specific slope where enough data
        gr_cat_preds = np.full(len(test), global_mean)
        for i, (_, row) in enumerate(test.iterrows()):
            shelf = row["Bookshelf"]
            if pd.notna(row["gr_rating"]):
                cat_train = train_gr[train_gr["Bookshelf"] == shelf]
                if len(cat_train) >= 5:
                    s, ic, _, _, _ = stats.linregress(
                        cat_train["gr_rating"], cat_train[target]
                    )
                    gr_cat_preds[i] = s * row["gr_rating"] + ic
                else:
                    gr_cat_preds[i] = slope * row["gr_rating"] + intercept
            else:
                gr_cat_preds[i] = cat_means.get(shelf, global_mean)
        results["GR rating (per-category linear)"] = eval_preds(actual, gr_cat_preds)

    # GR band predictor
    train_w_gr = train[train["gr_rating"].notna()].copy()
    if len(train_w_gr) >= 10:
        train_w_gr["gr_band"] = assign_gr_bands(train_w_gr["gr_rating"])
        band_means = train_w_gr.groupby("gr_band", observed=True)[target].mean()
        test_bands = assign_gr_bands(test["gr_rating"])
        band_preds = test_bands.map(band_means).astype(float).fillna(global_mean).values
        results["GR band mean"] = eval_preds(actual, band_preds)

    # ML models: build shared feature matrices
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoder.fit(train[["Bookshelf"]])
    X_train_cat = cat_encoder.transform(train[["Bookshelf"]])
    X_test_cat = cat_encoder.transform(test[["Bookshelf"]])
    y_train = train[target].values

    def _add_cols(
        base_tr: np.ndarray, base_te: np.ndarray, cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        extras_tr = train[cols].copy()
        extras_te = test[cols].copy()
        for c in cols:
            med = extras_tr[c].median()
            extras_tr[c] = extras_tr[c].fillna(med)
            extras_te[c] = extras_te[c].fillna(med)
        return (
            np.hstack([base_tr, extras_tr.values]),
            np.hstack([base_te, extras_te.values]),
        )

    # Category only
    ridge_cat = Ridge(alpha=1.0)
    ridge_cat.fit(X_train_cat, y_train)
    results["Ridge (category only)"] = eval_preds(actual, ridge_cat.predict(X_test_cat))

    # Category + GR rating
    X_tr_gr, X_te_gr = _add_cols(X_train_cat, X_test_cat, ["gr_rating"])

    ridge_gr = Ridge(alpha=1.0)
    ridge_gr.fit(X_tr_gr, y_train)
    results["Ridge (cat + GR rating)"] = eval_preds(actual, ridge_gr.predict(X_te_gr))

    # Category + GR rating + log count
    X_tr_gr2, X_te_gr2 = _add_cols(
        X_train_cat, X_test_cat, ["gr_rating", "gr_log_count"]
    )

    ridge_gr2 = Ridge(alpha=1.0)
    ridge_gr2.fit(X_tr_gr2, y_train)
    results["Ridge (cat + GR rating + log count)"] = eval_preds(
        actual, ridge_gr2.predict(X_te_gr2)
    )

    # All numeric features
    num_cols = ["gr_rating", "gr_log_count", "ol_rating", "log_pages", "pub_year"]
    X_tr_all, X_te_all = _add_cols(X_train_cat, X_test_cat, num_cols)

    ridge_all = Ridge(alpha=1.0)
    ridge_all.fit(X_tr_all, y_train)
    results["Ridge (cat + all numeric)"] = eval_preds(
        actual, ridge_all.predict(X_te_all)
    )

    # GBM with all features
    gbm = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    gbm.fit(X_tr_all, y_train)
    results["GBM (cat + all numeric)"] = eval_preds(actual, gbm.predict(X_te_all))

    # GBM category + GR only
    gbm_gr = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    gbm_gr.fit(X_tr_gr, y_train)
    results["GBM (cat + GR rating)"] = eval_preds(actual, gbm_gr.predict(X_te_gr))

    return results


# ---- Correlation analysis ----


def goodreads_correlation(train: pd.DataFrame) -> None:
    """Analyze GR rating correlation with personal ratings in training set."""
    print("=" * 70)
    print("GOODREADS CORRELATION IN TRAINING SET")
    print("=" * 70)

    has_gr = train["gr_rating"].notna()
    print(f"\n  Books with GR rating: {has_gr.sum()}/{len(train)}")

    for target, label in TARGETS:
        valid = train[has_gr & train[target].notna()]
        r, p = stats.pearsonr(valid["gr_rating"], valid[target])
        rho, rho_p = stats.spearmanr(valid["gr_rating"], valid[target])
        print(f"\n  GR rating vs {label}:")
        print(f"    Pearson R = {r:.3f} (p={p:.4f})")
        print(f"    Spearman rho = {rho:.3f} (p={rho_p:.4f})")

    print("\n  GR correlation by category (Enjoyment):")
    print(f"    {'Category':<25} {'N':>4} {'R':>6} {'p':>7}")
    print("    " + "-" * 45)
    for shelf, grp in train[has_gr].groupby("Bookshelf"):
        valid_g = grp[grp["Enjoyment (/5)"].notna()]
        if len(valid_g) >= 5:
            r, p = stats.pearsonr(valid_g["gr_rating"], valid_g["Enjoyment (/5)"])
            print(f"    {str(shelf):<25} {len(valid_g):>4} {r:>6.3f} {p:>7.3f}")

    # GR rating bands
    valid = train[has_gr & train["Enjoyment (/5)"].notna()].copy()
    valid["gr_band"] = assign_gr_bands(valid["gr_rating"])
    print("\n  GR rating bands -> My enjoyment:")
    print(f"    {'Band':<12} {'N':>4} {'My Enjoy':>10} {'My Useful':>10}")
    print("    " + "-" * 40)
    for band in GR_BAND_LABELS:
        grp = valid[valid["gr_band"] == band]
        if len(grp) > 0:
            print(
                f"    {band:<12} {len(grp):>4} "
                f"{grp['Enjoyment (/5)'].mean():>10.2f} "
                f"{grp['Usefulness /5 to Me'].mean():>10.2f}"
            )


# ---- Plots ----


def plot_goodreads(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Plot GR rating vs personal enjoyment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, data, color, label_prefix in [
        (axes[0], train, "steelblue", "Training"),
        (axes[1], test, "coral", "Holdout"),
    ]:
        valid = data[data["gr_rating"].notna() & data["Enjoyment (/5)"].notna()]
        ax.scatter(
            valid["gr_rating"], valid["Enjoyment (/5)"], alpha=0.4, s=30, c=color
        )
        if len(valid) >= 5:
            slope, intercept, r, p, _ = stats.linregress(
                valid["gr_rating"], valid["Enjoyment (/5)"]
            )
            x_line = np.linspace(
                valid["gr_rating"].min(), valid["gr_rating"].max(), 100
            )
            ax.plot(
                x_line,
                slope * x_line + intercept,
                "r-",
                alpha=0.7,
                label=f"R={r:.2f}, p={p:.3f}",
            )
        ax.set_xlabel("Goodreads Rating")
        ax.set_ylabel("My Enjoyment (/5)")
        ax.set_title(f"{label_prefix} Set (n={len(valid)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # GR band boxplot
    ax = axes[2]
    valid_all = train[
        train["gr_rating"].notna() & train["Enjoyment (/5)"].notna()
    ].copy()
    valid_all["gr_band"] = assign_gr_bands(valid_all["gr_rating"])
    band_data = [
        valid_all[valid_all["gr_band"] == b]["Enjoyment (/5)"].dropna().values
        for b in GR_BAND_LABELS
    ]
    bp = ax.boxplot(band_data, tick_labels=GR_BAND_LABELS, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.5)
    for i, d in enumerate(band_data):
        ax.text(i + 1, ax.get_ylim()[0] + 0.1, f"n={len(d)}", ha="center", fontsize=8)
    ax.set_xlabel("Goodreads Rating Band")
    ax.set_ylabel("My Enjoyment (/5)")
    ax.set_title("Enjoyment by GR Band (Training)")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "goodreads_model_eval.png", dpi=150)
    print("\nSaved: goodreads_model_eval.png")
    plt.close()


# ---- Decision rule analysis ----


def decision_rule_analysis(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Analyze how filtering books by GR rating changes expected ratings and utilities."""
    print("\n" + "=" * 70)
    print("DECISION RULE ANALYSIS: filtering by Goodreads rating")
    print("=" * 70)

    thresholds = [0.0, 3.5, 3.8, 4.0, 4.2, 4.5]

    for dataset_name, data in [("Training set", train), ("Holdout set", test)]:
        has_both = (
            data["gr_rating"].notna()
            & data["Enjoyment (/5)"].notna()
            & data["Usefulness /5 to Me"].notna()
        )
        base = data[has_both].copy()

        print(f"\n  {dataset_name} ({len(base)} books with GR rating and ratings):")
        print(
            f"    {'Threshold':<12} {'N':>4} {'Enjoy':>6} {'Useful':>7} "
            f"{'E-Util':>7} {'U-Util':>7} "
            f"{'E-Util tot':>10} {'U-Util tot':>10}"
        )
        print("    " + "-" * 75)

        for thresh in thresholds:
            subset = base[base["gr_rating"] >= thresh] if thresh > 0 else base
            n = len(subset)
            if n == 0:
                continue
            e_vals = subset["Enjoyment (/5)"].values
            u_vals = subset["Usefulness /5 to Me"].values
            label = "All w/ GR" if thresh == 0 else f"GR >= {thresh}"
            print(
                f"    {label:<12} {n:>4} {e_vals.mean():>6.2f} {u_vals.mean():>7.2f} "
                f"{enjoy_utility(e_vals).mean():>7.3f} {useful_utility(u_vals).mean():>7.3f} "
                f"{enjoy_utility(e_vals).sum():>10.2f} {useful_utility(u_vals).sum():>10.2f}"
            )

    # Per-book holdout detail
    has_both = (
        test["gr_rating"].notna()
        & test["Enjoyment (/5)"].notna()
        & test["Usefulness /5 to Me"].notna()
    )
    detail = test[has_both].sort_values("gr_rating", ascending=False).copy()

    print("\n  Holdout books sorted by GR rating (utility view):")
    print(
        f"    {'Title':<40} {'GR':>4} {'Enjoy':>5} {'Useful':>6} "
        f"{'E-Util':>6} {'U-Util':>6}"
    )
    print("    " + "-" * 72)
    for _, row in detail.iterrows():
        e, u = row["Enjoyment (/5)"], row["Usefulness /5 to Me"]
        print(
            f"    {str(row['title'])[:39]:<40} {row['gr_rating']:>4.1f} "
            f"{e:>5.1f} {u:>6.1f} {enjoy_utility(e):>6.3f} {useful_utility(u):>6.3f}"
        )

    # Opportunity cost table
    baseline_e_per = enjoy_utility(detail["Enjoyment (/5)"].values).mean()
    baseline_u_per = useful_utility(detail["Usefulness /5 to Me"].values).mean()

    print("\n  Opportunity cost: per-book utility at different thresholds")
    print(
        f"    {'Strategy':<35} {'E-Util/book':>11} {'U-Util/book':>11} "
        f"{'vs baseline':>12}"
    )
    print("    " + "-" * 72)
    print(
        f"    {'Baseline (all w/ GR)':<35} "
        f"{baseline_e_per:>11.3f} {baseline_u_per:>11.3f} {'--':>12}"
    )
    for thresh in [3.8, 4.0, 4.2, 4.5]:
        above = detail[detail["gr_rating"] >= thresh]
        if len(above) == 0:
            continue
        e_per = enjoy_utility(above["Enjoyment (/5)"].values).mean()
        u_per = useful_utility(above["Usefulness /5 to Me"].values).mean()
        e_gain_pct = (e_per - baseline_e_per) / baseline_e_per * 100
        label = f"Only GR >= {thresh} ({len(above)}/{len(detail)} books)"
        print(f"    {label:<35} {e_per:>11.3f} {u_per:>11.3f} {e_gain_pct:>+11.1f}%")

    # Reference table
    print("\n  Utility function reference (base^(r-1) - 1):")
    print(f"    {'Rating':>6} {'E-Util (1.3)':>12} {'U-Util (1.8)':>12}")
    print("    " + "-" * 32)
    for r in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        print(f"    {r:>6.1f} {enjoy_utility(r):>12.3f} {useful_utility(r):>12.3f}")


# ---- Holdout detail table ----


def print_holdout_details(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("HOLDOUT BOOK DETAILS")
    print("=" * 70)
    cat_means = train.groupby("Bookshelf")["Enjoyment (/5)"].mean().to_dict()
    gm = train["Enjoyment (/5)"].mean()
    print(
        f"\n  {'Title':<45} {'Shelf':<15} {'GR':>4} "
        f"{'Actual':>6} {'CatM':>5} {'Err':>5}"
    )
    print("  " + "-" * 85)
    for _, row in test.iterrows():
        gr = f"{row['gr_rating']:.1f}" if pd.notna(row["gr_rating"]) else "  - "
        actual_e = row["Enjoyment (/5)"]
        cat_pred = cat_means.get(row["Bookshelf"], gm)
        print(
            f"  {str(row['title'])[:44]:<45} {str(row['Bookshelf'])[:14]:<15} "
            f"{gr:>4} {actual_e:>6.1f} {cat_pred:>5.2f} {actual_e - cat_pred:>+5.2f}"
        )


# ---- Main ----


def main() -> None:
    df = load_data()

    # Sort chronologically, split train/holdout
    df = df.sort_values("date_sort", na_position="first").reset_index(drop=True)
    df_valid = df[df["Enjoyment (/5)"].notna()].copy()
    n = len(df_valid)
    train = df_valid.iloc[: n - HOLDOUT_N].copy()
    test = df_valid.iloc[n - HOLDOUT_N :].copy()

    print(f"Total books: {n}")
    print(
        f"Train: {len(train)} (up to {train['date_sort'].max().strftime('%Y-%m-%d')})"
    )
    print(
        f"Holdout: {len(test)} ({test['date_sort'].min().strftime('%Y-%m-%d')} to "
        f"{test['date_sort'].max().strftime('%Y-%m-%d')})"
    )
    print(f"  GR ratings in train: {train['gr_rating'].notna().sum()}/{len(train)}")
    print(f"  GR ratings in holdout: {test['gr_rating'].notna().sum()}/{len(test)}")

    goodreads_correlation(train)

    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON: train on {len(train)}, predict holdout {len(test)}")
    print("=" * 70)

    for target, label in TARGETS:
        results = run_all_models(train, test, target)
        print_table(results, f"{label}:")

    decision_rule_analysis(train, test)
    plot_goodreads(train, test)
    print_holdout_details(train, test)


if __name__ == "__main__":
    main()
