"""Normalized rating models with category-first splitting.

Approach:
1. Z-score normalize each rating source (GR, OL, Amazon) using training stats
2. Category-first decision tree (force first split on category)
3. Holdout split: first half 2026 = validation, second half = test
4. Filtering curve: x = percentile dropped, y = avg actual rating of kept books
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text

from book_decision_analysis import load_and_join, impute_missing, find_outliers

AI = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parent.parent / "data"
PLOTS = AI / "plots"
PLOTS.mkdir(exist_ok=True)


def attach_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Get finish dates for holdout ordering."""
    new_books = pd.read_csv(
        DATA / "Books Read and their effects - new_books_to_rate 2026.csv"
    )
    new_books["_norm"] = new_books["title"].apply(
        lambda t: re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(t).lower().strip())
        ).strip()
    )
    date_map: dict[str, str] = {}
    for _, row in new_books.iterrows():
        if pd.notna(row.get("date_finished")):
            date_map[row["_norm"]] = str(row["date_finished"])

    golden = pd.read_csv(AI / "golden_master.csv")
    golden["_norm"] = golden["title"].apply(
        lambda t: re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(t).lower().strip())
        ).strip()
    )
    for _, row in golden.iterrows():
        if pd.notna(row.get("estimated_finish")):
            date_map[row["_norm"]] = str(row["estimated_finish"])

    # new_books dates override golden dates for holdout
    for _, row in new_books.iterrows():
        if pd.notna(row.get("date_finished")):
            date_map[row["_norm"]] = str(row["date_finished"])

    df = df.copy()
    df["_norm"] = df["title"].apply(
        lambda t: re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(t).lower().strip())
        ).strip()
    )
    df["date_finished"] = df["_norm"].map(date_map)
    df["date_finished"] = pd.to_datetime(df["date_finished"], errors="coerce")
    return df


def clean_category(cat: str) -> str:
    m = {
        "fiction": "fiction",
        "Literature": "Literature",
        "Business, management": "Business",
        "General Reading": "General Reading",
        "Computer Science": "CS",
        "Histories": "Histories",
        "Machine Learning": "ML",
        "Math": "Math",
    }
    return m.get(str(cat).strip(), str(cat).strip()) if pd.notna(cat) else "Other"


def normalize_and_build(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize rating sources using training set statistics."""
    df = df.copy()
    df["category_clean"] = df["category"].apply(clean_category)

    train = df[df["dataset"] == "train"]

    # Z-score normalize each source using training set stats
    for col in ["gr_rating", "ol_rating", "amz_rating"]:
        mean = train[col].mean()
        std = train[col].std()
        df[f"{col}_z"] = (df[col] - mean) / std
        print(f"  {col}: mean={mean:.3f}, std={std:.3f}")

    # Also z-score the log counts
    for col in ["log_gr_count", "log_ol_count", "log_amz_count"]:
        mean = train[col].mean()
        std = train[col].std()
        if std > 0:
            df[f"{col}_z"] = (df[col] - mean) / std
        else:
            df[f"{col}_z"] = 0.0

    return df


def split_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """Split holdout into validation (first half by date) and test (second half)."""
    df = attach_dates(df)
    holdout = df[df["dataset"] == "holdout"].copy()
    holdout_sorted = holdout.sort_values("date_finished")

    mid = len(holdout_sorted) // 2
    val_idx = holdout_sorted.index[:mid]
    test_idx = holdout_sorted.index[mid:]

    df["split"] = df["dataset"]
    df.loc[val_idx, "split"] = "validation"
    df.loc[test_idx, "split"] = "test"

    n_val = (df["split"] == "validation").sum()
    n_test = (df["split"] == "test").sum()
    print(f"  Holdout split: {n_val} validation, {n_test} test")

    if n_val > 0:
        val_dates = df.loc[df["split"] == "validation", "date_finished"]
        test_dates = df.loc[df["split"] == "test", "date_finished"]
        print(f"    Validation: {val_dates.min().date()} to {val_dates.max().date()}")
        print(f"    Test: {test_dates.min().date()} to {test_dates.max().date()}")

    return df


def build_feature_matrix(
    df: pd.DataFrame, all_cat_cols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature matrix with z-scored ratings + category dummies."""
    z_cols = [
        "gr_rating_z",
        "ol_rating_z",
        "amz_rating_z",
        "log_gr_count_z",
        "log_ol_count_z",
        "log_amz_count_z",
    ]
    cat_dummies = pd.get_dummies(df["category_clean"], prefix="cat", drop_first=False)
    for c in all_cat_cols:
        if c not in cat_dummies.columns:
            cat_dummies[c] = 0
    cat_dummies = cat_dummies[all_cat_cols]

    X = pd.concat(
        [
            df[z_cols].fillna(0).reset_index(drop=True),
            cat_dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    return X, list(X.columns)


def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train models with z-scored features, evaluate on val and test."""
    cat_dummies = pd.get_dummies(df["category_clean"], prefix="cat", drop_first=False)
    all_cat_cols = sorted(cat_dummies.columns.tolist())

    for target_name, target_col in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        print(f"\n{'=' * 70}")
        print(f"TARGET: {target_name}")
        print(f"{'=' * 70}")

        # Training data
        train_df = df[
            (df["split"] == "train") & df[target_col].notna() & df["gr_rating"].notna()
        ]
        val_df = df[(df["split"] == "validation") & df[target_col].notna()]
        test_df = df[(df["split"] == "test") & df[target_col].notna()]
        full_holdout_df = df[
            (df["split"].isin(["validation", "test"])) & df[target_col].notna()
        ]

        X_train, features = build_feature_matrix(train_df, all_cat_cols)
        X_val, _ = build_feature_matrix(val_df, all_cat_cols)
        X_test, _ = build_feature_matrix(test_df, all_cat_cols)
        X_holdout, _ = build_feature_matrix(full_holdout_df, all_cat_cols)

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        y_holdout = full_holdout_df[target_col].values

        print(f"  Train: {len(y_train)}, Validation: {len(y_val)}, Test: {len(y_test)}")

        # ── Model 1: Ridge with z-scored features ──────────────
        ridge = Ridge(alpha=1.0).fit(X_train.values, y_train)
        pred_val_ridge = ridge.predict(X_val.values)
        pred_test_ridge = ridge.predict(X_test.values)
        pred_holdout_ridge = ridge.predict(X_holdout.values)

        r_val = np.corrcoef(pred_val_ridge, y_val)[0, 1] if len(y_val) > 2 else 0
        r_test = np.corrcoef(pred_test_ridge, y_test)[0, 1] if len(y_test) > 2 else 0
        r_hold = (
            np.corrcoef(pred_holdout_ridge, y_holdout)[0, 1]
            if len(y_holdout) > 2
            else 0
        )

        print(
            f"\n  Ridge (z-scored): Val R={r_val:.3f}, Test R={r_test:.3f}, Full holdout R={r_hold:.3f}"
        )
        coef_df = pd.DataFrame({"feature": features, "coef": ridge.coef_})
        coef_df = coef_df.reindex(
            coef_df["coef"].abs().sort_values(ascending=False).index
        )
        print(f"    Intercept: {ridge.intercept_:.4f}")
        for _, c in coef_df.iterrows():
            if abs(c["coef"]) > 0.01:
                print(f"      {c['feature']:30s}  {c['coef']:+.4f}")

        # ── Model 2: Category-first Decision Tree ─────────────
        # Deep tree that can have different rating slopes per category
        dt_deep = DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=8, random_state=42
        ).fit(X_train.values, y_train)
        pred_val_dt = dt_deep.predict(X_val.values)
        pred_test_dt = dt_deep.predict(X_test.values)
        pred_holdout_dt = dt_deep.predict(X_holdout.values)

        r_val_dt = np.corrcoef(pred_val_dt, y_val)[0, 1] if len(y_val) > 2 else 0
        r_test_dt = np.corrcoef(pred_test_dt, y_test)[0, 1] if len(y_test) > 2 else 0
        r_hold_dt = (
            np.corrcoef(pred_holdout_dt, y_holdout)[0, 1] if len(y_holdout) > 2 else 0
        )

        print(
            f"\n  DT (depth=5, z-scored): Val R={r_val_dt:.3f}, Test R={r_test_dt:.3f}, Full holdout R={r_hold_dt:.3f}"
        )
        tree_text = export_text(dt_deep, feature_names=features, decimals=2)
        print("    Tree rules:")
        for line in tree_text.split("\n"):
            print(f"      {line}")

        # ── Model 3: Shallow tree for interpretability ─────────
        dt_shallow = DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=10, random_state=42
        ).fit(X_train.values, y_train)
        pred_val_dt3 = dt_shallow.predict(X_val.values)
        pred_test_dt3 = dt_shallow.predict(X_test.values)
        pred_holdout_dt3 = dt_shallow.predict(X_holdout.values)

        r_val_dt3 = np.corrcoef(pred_val_dt3, y_val)[0, 1] if len(y_val) > 2 else 0
        r_test_dt3 = np.corrcoef(pred_test_dt3, y_test)[0, 1] if len(y_test) > 2 else 0
        r_hold_dt3 = (
            np.corrcoef(pred_holdout_dt3, y_holdout)[0, 1] if len(y_holdout) > 2 else 0
        )

        print(
            f"\n  DT (depth=3, z-scored): Val R={r_val_dt3:.3f}, Test R={r_test_dt3:.3f}, Full holdout R={r_hold_dt3:.3f}"
        )
        tree_text3 = export_text(dt_shallow, feature_names=features, decimals=2)
        print("    Tree rules:")
        for line in tree_text3.split("\n"):
            print(f"      {line}")

        # ── Model 4: Per-category Ridge ────────────────────────
        z_rating_cols = ["gr_rating_z", "ol_rating_z", "amz_rating_z"]
        z_count_cols = ["log_gr_count_z", "log_ol_count_z", "log_amz_count_z"]
        per_cat_features = z_rating_cols + z_count_cols

        print("\n  Per-category Ridge (ratings + counts only):")
        cat_models: dict[str, Ridge] = {}
        cat_means: dict[str, float] = {}
        for cat in sorted(train_df["category_clean"].unique()):
            cat_train = train_df[train_df["category_clean"] == cat]
            if len(cat_train) < 5:
                continue
            X_cat = cat_train[per_cat_features].fillna(0).values
            y_cat = cat_train[target_col].values
            cat_ridge = Ridge(alpha=1.0).fit(X_cat, y_cat)
            cat_models[cat] = cat_ridge
            cat_means[cat] = y_cat.mean()

            # In-sample R
            pred_cat = cat_ridge.predict(X_cat)
            r_cat = (
                np.corrcoef(pred_cat, y_cat)[0, 1] if len(y_cat) > 2 else float("nan")
            )
            coefs = dict(zip(per_cat_features, cat_ridge.coef_))
            top_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            coef_str = ", ".join(f"{k}={v:+.3f}" for k, v in top_coefs)
            print(
                f"    {cat:20s} n={len(cat_train):3d} train_R={r_cat:.3f} "
                f"intercept={cat_ridge.intercept_:.2f} {coef_str}"
            )

        # Predict holdout using per-category models (fall back to category mean)
        global_mean = y_train.mean()
        pred_holdout_percat = np.zeros(len(full_holdout_df))
        for i, (_, row) in enumerate(full_holdout_df.iterrows()):
            cat = row["category_clean"]
            if cat in cat_models:
                x = np.array([row.get(c, 0) for c in per_cat_features]).reshape(1, -1)
                x = np.nan_to_num(x)
                pred_holdout_percat[i] = cat_models[cat].predict(x)[0]
            elif cat in cat_means:
                pred_holdout_percat[i] = cat_means[cat]
            else:
                pred_holdout_percat[i] = global_mean

        r_hold_percat = (
            np.corrcoef(pred_holdout_percat, y_holdout)[0, 1]
            if len(y_holdout) > 2
            else 0
        )
        # Split for val/test
        pred_val_percat = pred_holdout_percat[: len(y_val)]
        pred_test_percat = pred_holdout_percat[len(y_val) :]
        r_val_pc = np.corrcoef(pred_val_percat, y_val)[0, 1] if len(y_val) > 2 else 0
        r_test_pc = (
            np.corrcoef(pred_test_percat, y_test)[0, 1] if len(y_test) > 2 else 0
        )

        print(
            f"    -> Val R={r_val_pc:.3f}, Test R={r_test_pc:.3f}, Full holdout R={r_hold_percat:.3f}"
        )

        # ── Summary ────────────────────────────────────────────
        print(f"\n  {'Model':<30s} {'Val R':>6s} {'Test R':>7s} {'Holdout R':>10s}")
        print(f"  {'Ridge (z-scored)':<30s} {r_val:6.3f} {r_test:7.3f} {r_hold:10.3f}")
        print(
            f"  {'DT depth=5 (z-scored)':<30s} {r_val_dt:6.3f} {r_test_dt:7.3f} {r_hold_dt:10.3f}"
        )
        print(
            f"  {'DT depth=3 (z-scored)':<30s} {r_val_dt3:6.3f} {r_test_dt3:7.3f} {r_hold_dt3:10.3f}"
        )
        print(
            f"  {'Per-category Ridge':<30s} {r_val_pc:6.3f} {r_test_pc:7.3f} {r_hold_percat:10.3f}"
        )

        # ── Filtering curve plot ───────────────────────────────
        models_for_plot = {
            "Ridge (z-scored)": pred_holdout_ridge,
            "DT depth=5": pred_holdout_dt,
            "DT depth=3": pred_holdout_dt3,
            "Per-cat Ridge": pred_holdout_percat,
        }

        # Use full holdout for the filtering curve
        plot_filtering_curve(
            y_holdout,
            models_for_plot,
            full_holdout_df["title"].values,
            target_name,
        )


def plot_filtering_curve(
    y_actual: np.ndarray,
    model_preds: dict[str, np.ndarray],
    titles: np.ndarray,
    target_name: str,
) -> None:
    """Plot: x = % books dropped, y = avg actual rating of kept books.
    Each point labeled with the predicted rating of the book being dropped."""
    n = len(y_actual)

    fig, axes = plt.subplots(
        1, len(model_preds) + 1, figsize=(6 * (len(model_preds) + 1), 7)
    )

    # Perfect foresight first
    perfect_order = np.argsort(y_actual)  # ascending = worst first to drop
    _plot_single_filter(
        axes[0], y_actual, y_actual, perfect_order, titles, "Perfect Foresight", n
    )

    for i, (name, preds) in enumerate(model_preds.items()):
        drop_order = np.argsort(preds)  # ascending = drop lowest predicted first
        _plot_single_filter(axes[i + 1], y_actual, preds, drop_order, titles, name, n)

    fig.suptitle(
        f"Filtering Curve: {target_name}\n(drop worst-predicted books, track avg of kept)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(PLOTS / f"filter_curve_{target_name.lower()}.png", dpi=150)
    plt.close()
    print(f"  Saved filter_curve_{target_name.lower()}.png")


def _plot_single_filter(
    ax: plt.Axes,
    y_actual: np.ndarray,
    preds: np.ndarray,
    drop_order: np.ndarray,
    titles: np.ndarray,
    model_name: str,
    n: int,
) -> None:
    """Single panel of the filtering curve."""
    pcts_dropped: list[float] = []
    avg_kept: list[float] = []
    labels: list[str] = []

    kept_mask = np.ones(n, dtype=bool)
    pcts_dropped.append(0.0)
    avg_kept.append(y_actual.mean())
    labels.append("")

    for i, idx in enumerate(drop_order):
        kept_mask[idx] = False
        n_dropped = i + 1
        pct = n_dropped / n * 100
        remaining = y_actual[kept_mask]
        if len(remaining) == 0:
            break
        pcts_dropped.append(pct)
        avg_kept.append(remaining.mean())

        actual_r = y_actual[idx]
        pred_r = preds[idx]
        labels.append(f"{actual_r:.1f}/{pred_r:.1f}")

    ax.plot(pcts_dropped, avg_kept, "b-", linewidth=1.5, alpha=0.8)
    ax.scatter(pcts_dropped, avg_kept, s=15, c="blue", zorder=5)

    # Label every ~5th point to avoid clutter, plus first and last few
    step = max(1, n // 12)
    for i in range(len(pcts_dropped)):
        if labels[i] and (i % step == 0 or i <= 2 or i >= len(pcts_dropped) - 3):
            ax.annotate(
                labels[i],
                (pcts_dropped[i], avg_kept[i]),
                fontsize=6,
                rotation=45,
                ha="left",
                va="bottom",
                alpha=0.8,
            )

    ax.set_xlabel("% of Books Dropped")
    ax.set_ylabel(f"Avg Actual {model_name.split('(')[0].strip()}")
    ax.set_title(f"{model_name}")
    ax.grid(alpha=0.2)
    ax.set_xlim(-2, 100)


def main() -> None:
    print("Loading data...")
    df = load_and_join()
    print(f"  {len(df)} books")

    print("Imputing missing ratings...")
    df = impute_missing(df)
    df = find_outliers(df)

    print("\nZ-score normalizing...")
    df = normalize_and_build(df)

    print("\nSplitting holdout into validation/test...")
    df = split_holdout(df)

    train_and_evaluate(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
