"""Analyze the new book ratings: test-retest reliability, distributions,
category-specific norming, and validation-set model comparison.

Splits new books chronologically: first half = validation, second half = test.
All model iteration uses validation only; test set reported once at the end.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent
NEW_RATINGS = DATA_DIR / "Books Read and their effects - new_books_to_rate 2026.csv"
TRAIN_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
TRAIN_ENRICHED = OUTPUT_DIR / "books_enriched.csv"
TRAIN_FEATURES = OUTPUT_DIR / "books_with_features.csv"

VALUE_BASE = 2


def utility_value(r: np.ndarray | float) -> np.ndarray | float:
    return VALUE_BASE ** (np.asarray(r) - 1) - 1


def load_new() -> pd.DataFrame:
    df = pd.read_csv(NEW_RATINGS)
    df.columns = df.columns.str.strip()
    if "author" in df.columns:
        df = df.drop(columns=["author"])
    rename_map = {
        "Enjoyment (/5)": "enjoy1",
        "Usefulness /5 to Me": "useful1",
        "Enjoyment (/5) 2nd": "enjoy2",
        "Usefulness /5 to Me.1": "useful2",
    }
    df = df.rename(columns=rename_map)
    df["enjoy_avg"] = (df["enjoy1"] + df["enjoy2"]) / 2
    df["useful_avg"] = (df["useful1"] + df["useful2"]) / 2
    return df


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_EXPORT)
    df.columns = df.columns.str.strip()
    return df


def load_train_enriched() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_ENRICHED)
    df.columns = df.columns.str.strip()
    return df


def split_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split new books chronologically: first half = val, second half = test."""
    df = df.sort_values("date_finished").reset_index(drop=True)
    mid = len(df) // 2
    return df.iloc[:mid].copy(), df.iloc[mid:].copy()


# ---- Section 1: Test-Retest Reliability ----


def test_retest_reliability(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("TEST-RETEST RELIABILITY (new books, 2 ratings)")
    print("=" * 70)

    for label, c1, c2 in [
        ("Enjoyment", "enjoy1", "enjoy2"),
        ("Usefulness", "useful1", "useful2"),
    ]:
        v1, v2 = df[c1].values, df[c2].values
        diff = v1 - v2
        r, p = stats.pearsonr(v1, v2)
        rho, rho_p = stats.spearmanr(v1, v2)
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        print(f"\n  {label}:")
        print(f"    Pearson R = {r:.3f} (p={p:.4f})")
        print(f"    Spearman rho = {rho:.3f} (p={rho_p:.4f})")
        print(f"    MAE = {mae:.3f},  RMSE = {rmse:.3f}")
        print(f"    Mean diff (v1-v2) = {np.mean(diff):+.3f},  SD = {np.std(diff):.3f}")

    print("\n  Original 208-book retest for comparison:")
    print("    Enjoyment: R=0.77, MAE=0.46, RMSE=0.66")
    print("    Usefulness: R=0.86, MAE=0.34, RMSE=0.59")


# ---- Section 2: Rating Distributions ----


def rating_distributions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("RATING DISTRIBUTIONS")
    print("=" * 70)

    for label, cols in [
        ("Enjoyment", ["enjoy1", "enjoy2", "enjoy_avg"]),
        ("Usefulness", ["useful1", "useful2", "useful_avg"]),
    ]:
        print(f"\n  {label}:")
        print(
            f"    {'Version':<12} {'Mean':>6} {'Median':>8} "
            f"{'SD':>6} {'Min':>5} {'Max':>5}"
        )
        print("    " + "-" * 50)
        names = ["Rating 1", "Rating 2", "Average"]
        for name, col in zip(names, cols):
            v = df[col]
            print(
                f"    {name:<12} {v.mean():>6.2f} {v.median():>8.2f} "
                f"{v.std():>6.2f} {v.min():>5.1f} {v.max():>5.1f}"
            )

    print("\n  By Bookshelf (enjoy_avg):")
    print(f"    {'Shelf':<25} {'N':>4} {'Mean':>6} {'SD':>6}")
    print("    " + "-" * 45)
    for shelf, grp in df.groupby("Bookshelf"):
        print(
            f"    {str(shelf):<25} {len(grp):>4} "
            f"{grp['enjoy_avg'].mean():>6.2f} {grp['enjoy_avg'].std():>6.2f}"
        )


# ---- Section 3: Category-Specific Norming ----


def category_norming_analysis(df_new: pd.DataFrame, df_train: pd.DataFrame) -> None:
    """Compare global vs category-specific mean-centering."""
    print("\n" + "=" * 70)
    print("CATEGORY-SPECIFIC NORMING")
    print("=" * 70)

    # Training set category stats
    train_cat = df_train.groupby("Bookshelf").agg(
        train_mean=("Enjoyment (/5)", "mean"),
        train_std=("Enjoyment (/5)", "std"),
        train_n=("Enjoyment (/5)", "count"),
    )
    train_global = df_train["Enjoyment (/5)"].mean()

    # New books category stats
    new_cat = df_new.groupby("Bookshelf").agg(
        new_mean=("enjoy_avg", "mean"),
        new_std=("enjoy_avg", "std"),
        new_n=("enjoy_avg", "count"),
    )

    merged = train_cat.join(new_cat, how="outer")

    print("\n  Category means: training vs new books")
    print(
        f"    {'Category':<25} {'Train N':>8} {'Train':>7} "
        f"{'New N':>6} {'New':>7} {'Shift':>7}"
    )
    print("    " + "-" * 65)
    for shelf in sorted(merged.index):
        row = merged.loc[shelf]
        tn = row.get("train_n", 0)
        tm = row.get("train_mean", float("nan"))
        nn = row.get("new_n", 0)
        nm = row.get("new_mean", float("nan"))
        shift = nm - tm if pd.notna(tm) and pd.notna(nm) else float("nan")
        tn_str = f"{int(tn)}" if pd.notna(tn) else "-"
        tm_str = f"{tm:.2f}" if pd.notna(tm) else "-"
        nn_str = f"{int(nn)}" if pd.notna(nn) else "-"
        nm_str = f"{nm:.2f}" if pd.notna(nm) else "-"
        shift_str = f"{shift:+.2f}" if pd.notna(shift) else "-"
        print(
            f"    {str(shelf):<25} {tn_str:>8} {tm_str:>7} "
            f"{nn_str:>6} {nm_str:>7} {shift_str:>7}"
        )
    print(
        f"    {'OVERALL':<25} {len(df_train):>8} {train_global:>7.2f} "
        f"{len(df_new):>6} {df_new['enjoy_avg'].mean():>7.2f} "
        f"{df_new['enjoy_avg'].mean() - train_global:>+7.2f}"
    )

    # Category-normed predictions: use training category mean but shift by
    # the overall train-vs-new offset
    overall_shift = df_new["enjoy_avg"].mean() - train_global
    print(f"\n  Overall mean shift (new - train): {overall_shift:+.3f}")

    # Three prediction approaches:
    # 1) Raw training category means
    # 2) Training category means + global shift
    # 3) Category-specific shifts (where we have enough data)
    raw_preds = df_new["Bookshelf"].map(train_cat["train_mean"]).fillna(train_global)

    shifted_preds = raw_preds + overall_shift

    cat_specific_preds = pd.Series(index=df_new.index, dtype=float)
    for i, row in df_new.iterrows():
        shelf = row["Bookshelf"]
        if shelf in new_cat.index and new_cat.loc[shelf, "new_n"] >= 3:
            # Use category-specific shift
            if shelf in train_cat.index:
                cat_shift = (
                    new_cat.loc[shelf, "new_mean"] - train_cat.loc[shelf, "train_mean"]
                )
                cat_specific_preds[i] = train_cat.loc[shelf, "train_mean"] + cat_shift
            else:
                cat_specific_preds[i] = new_cat.loc[shelf, "new_mean"]
        else:
            cat_specific_preds[i] = raw_preds[i] + overall_shift

    print("\n  Prediction accuracy (enjoy_avg):")
    print(f"    {'Method':<40} {'MAE':>6} {'RMSE':>6} {'R':>6}")
    print("    " + "-" * 60)
    for name, preds in [
        ("Training category mean (raw)", raw_preds),
        ("Training cat mean + global shift", shifted_preds),
        ("Category-specific shift", cat_specific_preds),
    ]:
        mae = mean_absolute_error(df_new["enjoy_avg"], preds)
        rmse = np.sqrt(mean_squared_error(df_new["enjoy_avg"], preds))
        r, _ = stats.pearsonr(df_new["enjoy_avg"], preds)
        print(f"    {name:<40} {mae:>6.3f} {rmse:>6.3f} {r:>6.3f}")

    # Same for usefulness
    train_u_cat = df_train.groupby("Bookshelf")["Usefulness /5 to Me"].mean()
    train_u_global = df_train["Usefulness /5 to Me"].mean()
    u_raw = df_new["Bookshelf"].map(train_u_cat).fillna(train_u_global)
    u_shift = df_new["useful_avg"].mean() - train_u_global
    u_shifted = u_raw + u_shift

    print(f"\n  Usefulness (global shift = {u_shift:+.3f}):")
    for name, preds in [
        ("Training category mean (raw)", u_raw),
        ("Training cat mean + global shift", u_shifted),
    ]:
        mae = mean_absolute_error(df_new["useful_avg"], preds)
        rmse = np.sqrt(mean_squared_error(df_new["useful_avg"], preds))
        r, _ = stats.pearsonr(df_new["useful_avg"], preds)
        print(f"    {name:<40} {mae:>6.3f} {rmse:>6.3f} {r:>6.3f}")


# ---- Section 4: Validation Framework ----


def eval_preds(actual: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, preds)
    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
    if len(actual) > 2 and np.std(preds) > 1e-10:
        r = float(stats.pearsonr(actual, preds)[0])
    else:
        r = float("nan")
    return {"MAE": mae, "RMSE": rmse, "R": r}


def print_results_table(results: dict[str, dict[str, float]], title: str) -> None:
    print(f"\n  {title}")
    print(f"    {'Model':<45} {'MAE':>6} {'RMSE':>6} {'R':>6}")
    print("    " + "-" * 65)
    for name, m in results.items():
        print(f"    {name:<45} {m['MAE']:>6.3f} {m['RMSE']:>6.3f} {m['R']:>6.3f}")


def run_model_comparison(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    target_col: str = "enjoy_avg",
    label: str = "Enjoyment",
) -> dict[str, dict[str, float]]:
    """Run all models and return results dict."""
    actual = df_eval[target_col].values
    results: dict[str, dict[str, float]] = {}

    # --- Baselines ---
    # Global mean
    train_target = "Enjoyment (/5)" if "enjoy" in target_col else "Usefulness /5 to Me"
    global_mean = df_train[train_target].mean()
    results["Global mean"] = eval_preds(actual, np.full(len(actual), global_mean))

    # Category mean (raw)
    cat_means = df_train.groupby("Bookshelf")[train_target].mean().to_dict()
    cat_preds = df_eval["Bookshelf"].map(cat_means).fillna(global_mean).values
    results["Category mean (train)"] = eval_preds(actual, cat_preds)

    # Category mean + global shift
    shift = df_eval[target_col].mean() - global_mean
    results[f"Category mean + shift ({shift:+.2f})"] = eval_preds(
        actual, cat_preds + shift
    )

    # Category-specific norming: LOO within eval set
    loo_cat = np.full(len(df_eval), np.nan)
    for i, (idx, row) in enumerate(df_eval.iterrows()):
        shelf = row["Bookshelf"]
        others = df_eval[(df_eval.index != idx) & (df_eval["Bookshelf"] == shelf)]
        if len(others) >= 2:
            loo_cat[i] = others[target_col].mean()
        else:
            # Fall back to LOO global
            loo_cat[i] = df_eval[df_eval.index != idx][target_col].mean()
    results["LOO category mean (eval only)"] = eval_preds(actual, loo_cat)

    # Combined: train category mean weighted with eval category mean
    # (Bayesian shrinkage toward train prior)
    shrink_preds = np.full(len(df_eval), np.nan)
    for i, (idx, row) in enumerate(df_eval.iterrows()):
        shelf = row["Bookshelf"]
        train_mean = cat_means.get(shelf, global_mean)
        eval_shelf = df_eval[(df_eval.index != idx) & (df_eval["Bookshelf"] == shelf)]
        if len(eval_shelf) >= 2:
            eval_mean = eval_shelf[target_col].mean()
            n_eval = len(eval_shelf)
            # Weight by sample size; prior strength = 5 (like having 5 train obs)
            prior_strength = 5
            w = n_eval / (n_eval + prior_strength)
            shrink_preds[i] = w * eval_mean + (1 - w) * train_mean
        else:
            shrink_preds[i] = train_mean
    results["Shrinkage (train prior + eval)"] = eval_preds(actual, shrink_preds)

    # --- ML Models ---
    # Features available for new books: Bookshelf only (no OL metadata fetched yet)
    # We can construct some features from the data we have

    # Prepare training features
    df_tr = df_train.copy()
    df_tr["note_length"] = df_tr["Long Term Effects"].fillna("").str.len()
    df_tr = df_tr[df_tr[train_target].notna()].copy()

    df_ev = df_eval.copy()
    df_ev["note_length"] = 0  # no notes for new books

    # Simple Ridge with category encoding
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoder.fit(df_tr[["Bookshelf"]])

    X_train_cat = cat_encoder.transform(df_tr[["Bookshelf"]])
    X_eval_cat = cat_encoder.transform(df_ev[["Bookshelf"]])
    y_train = df_tr[train_target].values

    for model_cls, name, kwargs in [
        (Ridge, "Ridge (category only)", {"alpha": 1.0}),
        (Ridge, "Ridge (category, alpha=10)", {"alpha": 10.0}),
        (
            GradientBoostingRegressor,
            "GBM (category only)",
            {"n_estimators": 50, "max_depth": 2, "random_state": 42},
        ),
    ]:
        model = model_cls(**kwargs)
        model.fit(X_train_cat, y_train)
        preds = model.predict(X_eval_cat)
        results[name] = eval_preds(actual, preds)

    # Ridge with category + within-category rank features
    # (position of this book's date within its category in the eval set)
    df_ev_sorted = df_ev.sort_values("date_finished")
    df_ev["cat_rank"] = df_ev_sorted.groupby("Bookshelf").cumcount()
    df_ev["cat_rank_norm"] = df_ev["cat_rank"] / df_ev.groupby("Bookshelf")[
        "cat_rank"
    ].transform("max").clip(lower=1)

    df_tr["cat_rank"] = df_tr.groupby("Bookshelf").cumcount()
    df_tr["cat_rank_norm"] = df_tr["cat_rank"] / df_tr.groupby("Bookshelf")[
        "cat_rank"
    ].transform("max").clip(lower=1)

    X_tr_ext = np.hstack([X_train_cat, df_tr[["cat_rank_norm"]].fillna(0).values])
    X_ev_ext = np.hstack([X_eval_cat, df_ev[["cat_rank_norm"]].fillna(0).values])

    ridge_ext = Ridge(alpha=1.0)
    ridge_ext.fit(X_tr_ext, y_train)
    results["Ridge (cat + temporal rank)"] = eval_preds(
        actual, ridge_ext.predict(X_ev_ext)
    )

    return results


def validation_framework(df_new: pd.DataFrame, df_train: pd.DataFrame) -> None:
    """Split into val/test, iterate models on val, report final on test."""
    print("\n" + "=" * 70)
    print("VALIDATION FRAMEWORK")
    print("=" * 70)

    val, test = split_val_test(df_new)
    print(
        f"\n  Val set: {len(val)} books ({val['date_finished'].min()} to "
        f"{val['date_finished'].max()})"
    )
    print(
        f"  Test set: {len(test)} books ({test['date_finished'].min()} to "
        f"{test['date_finished'].max()})"
    )

    # Val set category breakdown
    print("\n  Val set categories:")
    for shelf, n in val["Bookshelf"].value_counts().items():
        print(f"    {shelf}: {n}")
    print("\n  Test set categories:")
    for shelf, n in test["Bookshelf"].value_counts().items():
        print(f"    {shelf}: {n}")

    # ---- Enjoyment ----
    print("\n" + "-" * 70)
    print("ENJOYMENT PREDICTION")
    print("-" * 70)

    val_results = run_model_comparison(df_train, val, "enjoy_avg", "Enjoyment")
    print_results_table(val_results, "Validation set (first half):")

    # Also try predicting v1 and v2 separately to see noise floor
    for vcol, vlabel in [("enjoy1", "v1"), ("enjoy2", "v2")]:
        actual = val[vcol].values
        cat_means = df_train.groupby("Bookshelf")["Enjoyment (/5)"].mean().to_dict()
        gm = df_train["Enjoyment (/5)"].mean()
        preds = val["Bookshelf"].map(cat_means).fillna(gm).values
        m = eval_preds(actual, preds)
        print(f"    Cat mean -> {vlabel}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}")

    # ---- Usefulness ----
    print("\n" + "-" * 70)
    print("USEFULNESS PREDICTION")
    print("-" * 70)

    val_u_results = run_model_comparison(df_train, val, "useful_avg", "Usefulness")
    print_results_table(val_u_results, "Validation set (first half):")

    # ---- Best model on test set ----
    print("\n" + "=" * 70)
    print("HELD-OUT TEST SET RESULTS (second half of new books)")
    print("=" * 70)

    # Pick best val model for each
    best_e_name = min(val_results, key=lambda k: val_results[k]["MAE"])
    best_u_name = min(val_u_results, key=lambda k: val_u_results[k]["MAE"])
    print(
        f"\n  Best enjoyment model on val: {best_e_name} "
        f"(MAE={val_results[best_e_name]['MAE']:.3f})"
    )
    print(
        f"  Best usefulness model on val: {best_u_name} "
        f"(MAE={val_u_results[best_u_name]['MAE']:.3f})"
    )

    test_results = run_model_comparison(df_train, test, "enjoy_avg", "Enjoyment")
    print_results_table(test_results, "Test set enjoyment:")

    test_u_results = run_model_comparison(df_train, test, "useful_avg", "Usefulness")
    print_results_table(test_u_results, "Test set usefulness:")


# ---- Plots ----


def plot_all(df: pd.DataFrame) -> None:
    """Generate all plots."""
    # 1. Retest scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, c1, c2, label in [
        (ax1, "enjoy1", "enjoy2", "Enjoyment"),
        (ax2, "useful1", "useful2", "Usefulness"),
    ]:
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(df))
        ax.scatter(df[c1] + jitter, df[c2] + jitter, alpha=0.5, s=40, c="steelblue")
        ax.plot([0.5, 5.5], [0.5, 5.5], "k--", alpha=0.3, label="y=x")
        r, _ = stats.pearsonr(df[c1], df[c2])
        ax.set_xlabel(f"{label} Rating 1")
        ax.set_ylabel(f"{label} Rating 2")
        ax.set_title(f"{label}: v1 vs v2 (R={r:.2f})")
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "new_ratings_retest_scatter.png", dpi=150)
    print("\nSaved: new_ratings_retest_scatter.png")
    plt.close()

    # 2. Retest differences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, c1, c2, label, color in [
        (ax1, "enjoy1", "enjoy2", "Enjoyment", "steelblue"),
        (ax2, "useful1", "useful2", "Usefulness", "coral"),
    ]:
        diff = df[c1] - df[c2]
        ax.hist(
            diff,
            bins=np.arange(-2.25, 2.75, 0.5),
            alpha=0.7,
            color=color,
            edgecolor="white",
        )
        ax.axvline(0, color="black", linestyle="-", alpha=0.3)
        ax.axvline(
            diff.mean(),
            color="red",
            linestyle="--",
            label=f"mean={diff.mean():+.2f}",
        )
        rmse = np.sqrt(np.mean(diff**2))
        ax.set_xlabel(f"{label} v1 - v2")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Retest Differences (RMSE={rmse:.2f})")
        ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "new_ratings_retest_diffs.png", dpi=150)
    print("Saved: new_ratings_retest_diffs.png")
    plt.close()

    # 3. Raw + Utility by bookshelf
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    shelves = df["Bookshelf"].value_counts().index
    colors = plt.cm.Set2(np.linspace(0, 1, len(shelves)))
    shelf_color = dict(zip(shelves, colors))

    ax = axes[0]
    for shelf in shelves:
        mask = df["Bookshelf"] == shelf
        ax.scatter(
            df.loc[mask, "enjoy_avg"],
            df.loc[mask, "useful_avg"],
            label=shelf,
            alpha=0.6,
            s=40,
            c=[shelf_color[shelf]],
        )
    ax.set_xlabel("Enjoyment (avg)")
    ax.set_ylabel("Usefulness (avg)")
    ax.set_title("Raw Ratings by Bookshelf")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    enjoy_util = utility_value(df["enjoy_avg"])
    useful_util = utility_value(df["useful_avg"])
    for shelf in shelves:
        mask = df["Bookshelf"] == shelf
        ax.scatter(
            enjoy_util[mask],
            useful_util[mask],
            label=shelf,
            alpha=0.6,
            s=40,
            c=[shelf_color[shelf]],
        )
    ax.set_xlabel("Enjoyment Utility")
    ax.set_ylabel("Usefulness Utility")
    ax.set_title(f"Utility (base={VALUE_BASE}) by Bookshelf")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    r_vals = np.linspace(1, 5, 100)
    ax.plot(r_vals, utility_value(r_vals), "b-", linewidth=2)
    ax.scatter(
        df["enjoy_avg"],
        utility_value(df["enjoy_avg"]),
        alpha=0.3,
        s=20,
        c="red",
        label="actual books",
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Utility")
    ax.set_title(f"Utility curve: {VALUE_BASE}^(r-1) - 1")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "new_ratings_utility.png", dpi=150)
    print("Saved: new_ratings_utility.png")
    plt.close()

    # 4. Distributions (v1, v2, avg) - smaller figures to avoid size limit
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, (col, label) in enumerate(
        [
            ("enjoy1", "Enjoyment v1"),
            ("enjoy2", "Enjoyment v2"),
            ("enjoy_avg", "Enjoyment avg"),
        ]
    ):
        ax = axes[0, i]
        ax.hist(
            df[col],
            bins=np.arange(0.75, 5.75, 0.5),
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        ax.axvline(
            df[col].mean(),
            color="red",
            linestyle="--",
            label=f"mean={df[col].mean():.2f}",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.5)

    for i, (col, label) in enumerate(
        [
            ("useful1", "Usefulness v1"),
            ("useful2", "Usefulness v2"),
            ("useful_avg", "Usefulness avg"),
        ]
    ):
        ax = axes[1, i]
        ax.hist(
            df[col],
            bins=np.arange(0.75, 5.75, 0.5),
            alpha=0.7,
            color="coral",
            edgecolor="white",
        )
        ax.axvline(
            df[col].mean(),
            color="red",
            linestyle="--",
            label=f"mean={df[col].mean():.2f}",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.5)

    plt.suptitle(f"Rating Distributions (n={len(df)} new books)", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "new_ratings_distributions.png", dpi=150)
    print("Saved: new_ratings_distributions.png")
    plt.close()


def main() -> None:
    df = load_new()
    df_train = load_train()

    print(f"Loaded {len(df)} new books with dual ratings")
    print(f"Training set: {len(df_train)} books\n")

    test_retest_reliability(df)
    rating_distributions(df)
    plot_all(df)
    category_norming_analysis(df, df_train)
    validation_framework(df, df_train)


if __name__ == "__main__":
    main()
