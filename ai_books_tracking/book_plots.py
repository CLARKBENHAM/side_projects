"""Plots and filtering analysis for book decision models.

Generates:
1. Predicted vs actual scatter (overall + by category)
2. Ratings over time (actual + predicted)
3. Filtering analysis: what if you dropped bottom N% by prediction?
4. Perfect foresight comparison
5. Utility-space models and gains
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from book_decision_analysis import load_and_join, impute_missing, find_outliers

AI = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parent.parent / "data"
PLOTS = AI / "plots"
PLOTS.mkdir(exist_ok=True)


def _cat_map() -> dict[str, str]:
    return {
        "fiction": "fiction",
        "Literature": "Literature",
        "Business, management": "Business",
        "General Reading": "General Reading",
        "Computer Science": "Computer Science",
        "Histories": "Histories",
        "Machine Learning": "Machine Learning",
        "Math": "Math",
    }


def _clean_category(cat: str) -> str:
    m = _cat_map()
    return m.get(str(cat).strip(), str(cat).strip()) if pd.notna(cat) else "Other"


def _build_features(
    df: pd.DataFrame,
    train_mask: pd.Series,
    holdout_mask: pd.Series,
    target_col: str,
    all_cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    feature_cols = [
        "gr_rating",
        "ol_rating",
        "amz_rating",
        "log_gr_count",
        "log_ol_count",
        "log_amz_count",
    ]
    train_valid = df[
        train_mask & df[target_col].notna() & df["gr_rating"].notna()
    ].copy()
    holdout_valid = df[holdout_mask & df[target_col].notna()].copy()

    train_cats = pd.get_dummies(
        train_valid["category_clean"], prefix="cat", drop_first=False
    )
    holdout_cats = pd.get_dummies(
        holdout_valid["category_clean"], prefix="cat", drop_first=False
    )
    for c in all_cat_cols:
        if c not in train_cats.columns:
            train_cats[c] = 0
        if c not in holdout_cats.columns:
            holdout_cats[c] = 0
    train_cats = train_cats[all_cat_cols]
    holdout_cats = holdout_cats[all_cat_cols]

    X_train = pd.concat([train_valid[feature_cols].fillna(0), train_cats], axis=1)
    X_holdout = pd.concat([holdout_valid[feature_cols].fillna(0), holdout_cats], axis=1)

    return (
        X_train,
        X_holdout,
        train_valid[target_col].values,
        holdout_valid[target_col].values,
        list(X_train.columns),
    )


def _train_models(
    X_train: pd.DataFrame, y_train: np.ndarray, features: list[str]
) -> dict[str, object]:
    ridge = Ridge(alpha=1.0).fit(X_train.values, y_train)
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42,
    ).fit(X_train.values, y_train)
    dt = DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=42).fit(
        X_train.values, y_train
    )
    return {"Ridge": ridge, "GBM": gbm, "DecisionTree": dt}


def load_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach finish dates from golden_master and new_books_to_rate."""
    golden = pd.read_csv(AI / "golden_master.csv")
    golden["_norm"] = golden["title"].apply(
        lambda t: re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(t).lower().strip())
        ).strip()
    )
    golden_dates: dict[str, str] = {}
    for _, row in golden.iterrows():
        if pd.notna(row.get("estimated_finish")):
            golden_dates[row["_norm"]] = str(row["estimated_finish"])

    new_books = pd.read_csv(
        DATA / "Books Read and their effects - new_books_to_rate 2026.csv"
    )
    new_books["_norm"] = new_books["title"].apply(
        lambda t: re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(t).lower().strip())
        ).strip()
    )
    for _, row in new_books.iterrows():
        if pd.notna(row.get("date_finished")):
            golden_dates[row["_norm"]] = str(row["date_finished"])

    df = df.copy()
    df["_norm"] = df["title"].apply(
        lambda t: re.sub(
            r"\s+", " ", re.sub(r"[^a-z0-9]", " ", str(t).lower().strip())
        ).strip()
    )
    df["date_finished"] = df["_norm"].map(golden_dates)
    df["date_finished"] = pd.to_datetime(df["date_finished"], errors="coerce")
    return df


def plot_predicted_vs_actual(df: pd.DataFrame, all_cat_cols: list[str]) -> None:
    """Scatter plots of predicted vs actual enjoyment/usefulness."""
    train_mask = df["dataset"] == "train"
    holdout_mask = df["dataset"] == "holdout"

    for target_name, target_col in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        X_train, X_holdout, y_train, y_holdout, features = _build_features(
            df, train_mask, holdout_mask, target_col, all_cat_cols
        )
        models = _train_models(X_train, y_train, features)

        holdout_valid = df[holdout_mask & df[target_col].notna()].copy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Predicted vs Actual {target_name} (Holdout)", fontsize=14)

        for ax, (name, model) in zip(axes, models.items()):
            preds = model.predict(X_holdout.values)
            r = np.corrcoef(preds, y_holdout)[0, 1] if len(y_holdout) > 2 else 0

            cats = holdout_valid["category_clean"].values
            unique_cats = sorted(set(cats))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
            cat_color = {c: colors[i] for i, c in enumerate(unique_cats)}

            for cat in unique_cats:
                mask = cats == cat
                ax.scatter(
                    y_holdout[mask],
                    preds[mask],
                    c=[cat_color[cat]],
                    label=cat,
                    alpha=0.7,
                    s=40,
                )

            mn, mx = (
                min(y_holdout.min(), preds.min()) - 0.2,
                max(y_holdout.max(), preds.max()) + 0.2,
            )
            ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3, label="y=x")
            ax.set_xlabel(f"Actual {target_name}")
            ax.set_ylabel(f"Predicted {target_name}")
            ax.set_title(f"{name} (R={r:.3f})")
            ax.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        plt.savefig(PLOTS / f"pred_vs_actual_{target_name.lower()}.png", dpi=150)
        plt.close()
        print(f"  Saved pred_vs_actual_{target_name.lower()}.png")


def plot_by_category(df: pd.DataFrame, all_cat_cols: list[str]) -> None:
    """Separate predicted vs actual plot for each category."""
    train_mask = df["dataset"] == "train"
    holdout_mask = df["dataset"] == "holdout"

    for target_name, target_col in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        X_train, X_holdout, y_train, y_holdout, features = _build_features(
            df, train_mask, holdout_mask, target_col, all_cat_cols
        )
        models = _train_models(X_train, y_train, features)
        holdout_valid = df[holdout_mask & df[target_col].notna()].copy()

        # Use best model (Decision Tree had highest R)
        best_model = models["DecisionTree"]
        preds = best_model.predict(X_holdout.values)

        cats = holdout_valid["category_clean"].values
        unique_cats = sorted(set(cats))
        n_cats = len(unique_cats)
        ncols = min(4, n_cats)
        nrows = (n_cats + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )
        fig.suptitle(
            f"Decision Tree: Predicted vs Actual {target_name} by Category", fontsize=14
        )

        for i, cat in enumerate(unique_cats):
            ax = axes[i // ncols][i % ncols]
            mask = cats == cat
            n = mask.sum()
            ax.scatter(y_holdout[mask], preds[mask], alpha=0.7, s=50)
            mn = min(y_holdout[mask].min(), preds[mask].min()) - 0.3
            mx = max(y_holdout[mask].max(), preds[mask].max()) + 0.3
            ax.plot([mn, mx], [mn, mx], "k--", alpha=0.3)
            if n > 2:
                r = np.corrcoef(preds[mask], y_holdout[mask])[0, 1]
                ax.set_title(f"{cat} (n={n}, R={r:.2f})")
            else:
                ax.set_title(f"{cat} (n={n})")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

        # Hide unused axes
        for i in range(n_cats, nrows * ncols):
            axes[i // ncols][i % ncols].set_visible(False)

        plt.tight_layout()
        plt.savefig(PLOTS / f"by_category_{target_name.lower()}.png", dpi=150)
        plt.close()
        print(f"  Saved by_category_{target_name.lower()}.png")


def plot_over_time(df: pd.DataFrame, all_cat_cols: list[str]) -> None:
    """Plot actual and predicted ratings over time."""
    df = load_dates(df)
    has_date = df["date_finished"].notna()
    train_mask = df["dataset"] == "train"
    holdout_mask = df["dataset"] == "holdout"

    for target_name, target_col in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        X_train, X_holdout, y_train, y_holdout, features = _build_features(
            df, train_mask, holdout_mask, target_col, all_cat_cols
        )
        models = _train_models(X_train, y_train, features)

        # Predict on ALL books that have the target
        all_valid = df[df[target_col].notna() & has_date].copy()
        all_cats = pd.get_dummies(
            all_valid["category_clean"], prefix="cat", drop_first=False
        )
        for c in all_cat_cols:
            if c not in all_cats.columns:
                all_cats[c] = 0
        all_cats = all_cats[all_cat_cols]

        feature_cols = [
            "gr_rating",
            "ol_rating",
            "amz_rating",
            "log_gr_count",
            "log_ol_count",
            "log_amz_count",
        ]
        X_all = pd.concat([all_valid[feature_cols].fillna(0), all_cats], axis=1)

        dt_model = models["DecisionTree"]
        all_valid = all_valid.copy()
        all_valid["predicted"] = dt_model.predict(X_all.values)
        all_valid = all_valid.sort_values("date_finished")

        fig, ax = plt.subplots(figsize=(14, 6))
        # Color by dataset
        train_pts = all_valid[all_valid["dataset"] == "train"]
        hold_pts = all_valid[all_valid["dataset"] == "holdout"]

        ax.scatter(
            train_pts["date_finished"],
            train_pts[target_col],
            c="steelblue",
            alpha=0.4,
            s=25,
            label="Actual (train)",
        )
        ax.scatter(
            hold_pts["date_finished"],
            hold_pts[target_col],
            c="darkorange",
            alpha=0.6,
            s=35,
            label="Actual (holdout)",
        )
        ax.scatter(
            hold_pts["date_finished"],
            hold_pts["predicted"],
            c="red",
            marker="x",
            alpha=0.7,
            s=35,
            label="Predicted (holdout)",
        )

        # Rolling average of actuals
        rolling_actual = (
            all_valid.set_index("date_finished")[target_col].rolling("90D").mean()
        )
        ax.plot(
            rolling_actual.index,
            rolling_actual.values,
            color="navy",
            linewidth=2,
            alpha=0.6,
            label="90-day rolling avg",
        )

        ax.set_xlabel("Date Finished")
        ax.set_ylabel(target_name)
        ax.set_title(f"{target_name} Over Time (Actual + Predicted)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(PLOTS / f"over_time_{target_name.lower()}.png", dpi=150)
        plt.close()
        print(f"  Saved over_time_{target_name.lower()}.png")


def filtering_analysis(df: pd.DataFrame, all_cat_cols: list[str]) -> None:
    """What if you dropped books below various prediction thresholds?
    Compare model-based filtering vs perfect foresight."""
    train_mask = df["dataset"] == "train"
    holdout_mask = df["dataset"] == "holdout"

    keep_pcts = [100, 95, 90, 80, 75, 60, 50, 40, 25, 10, 5]

    for target_name, target_col in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        X_train, X_holdout, y_train, y_holdout, features = _build_features(
            df, train_mask, holdout_mask, target_col, all_cat_cols
        )
        models = _train_models(X_train, y_train, features)

        print(f"\n{'=' * 70}")
        print(f"FILTERING ANALYSIS: {target_name}")
        print(f"{'=' * 70}")

        # Enjoyment utility: 1.3^(r-1) - 1, Usefulness utility: 1.8^(r-1) - 1
        base = 1.3 if target_name == "Enjoyment" else 1.8

        def util_fn(r: np.ndarray) -> np.ndarray:
            return base ** (r - 1) - 1

        util_actual = util_fn(y_holdout)
        total_n = len(y_holdout)

        results: dict[str, list[dict]] = {}

        # Perfect foresight
        results["Perfect Foresight"] = []
        sorted_idx = np.argsort(y_holdout)[::-1]
        for pct in keep_pcts:
            n_keep = max(1, int(total_n * pct / 100))
            kept_idx = sorted_idx[:n_keep]
            avg_rating = y_holdout[kept_idx].mean()
            avg_util = util_actual[kept_idx].mean()
            results["Perfect Foresight"].append(
                {
                    "pct": pct,
                    "n_keep": n_keep,
                    "avg_rating": avg_rating,
                    "avg_util": avg_util,
                }
            )

        # Model-based filtering
        for model_name, model in models.items():
            preds = model.predict(X_holdout.values)
            pred_sorted_idx = np.argsort(preds)[::-1]
            results[model_name] = []
            for pct in keep_pcts:
                n_keep = max(1, int(total_n * pct / 100))
                kept_idx = pred_sorted_idx[:n_keep]
                avg_rating = y_holdout[kept_idx].mean()
                avg_util = util_actual[kept_idx].mean()
                results[model_name].append(
                    {
                        "pct": pct,
                        "n_keep": n_keep,
                        "avg_rating": avg_rating,
                        "avg_util": avg_util,
                    }
                )

        # Print table
        all_avg = y_holdout.mean()
        all_util = util_actual.mean()
        header_names = list(results.keys())
        print(
            f"\n  {'Keep%':>5s} {'N':>3s}  "
            + "  ".join(f"{'Avg ' + n:>20s}" for n in header_names)
            + "  "
            + "  ".join(f"{'Util ' + n:>20s}" for n in header_names)
        )

        for i, pct in enumerate(keep_pcts):
            n_keep = results["Perfect Foresight"][i]["n_keep"]
            ratings = "  ".join(
                f"{results[n][i]['avg_rating']:20.3f}" for n in header_names
            )
            utils = "  ".join(
                f"{results[n][i]['avg_util']:20.3f}" for n in header_names
            )
            print(f"  {pct:5d} {n_keep:3d}  {ratings}  {utils}")

        print(f"\n  Baseline (keep all): avg={all_avg:.3f}, utility={all_util:.3f}")

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Filtering Analysis: {target_name}", fontsize=14)

        for name in header_names:
            pcts = [r["pct"] for r in results[name]]
            ratings = [r["avg_rating"] for r in results[name]]
            utils = [r["avg_util"] for r in results[name]]
            ls = "--" if name == "Perfect Foresight" else "-"
            lw = 2 if name == "Perfect Foresight" else 1.5
            ax1.plot(
                pcts, ratings, ls, linewidth=lw, label=name, marker="o", markersize=4
            )
            ax2.plot(
                pcts, utils, ls, linewidth=lw, label=name, marker="o", markersize=4
            )

        ax1.set_xlabel("% of Books Kept (sorted by prediction)")
        ax1.set_ylabel(f"Average Actual {target_name}")
        ax1.set_title(f"Average {target_name} by Filter Threshold")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.2)
        ax1.invert_xaxis()

        util_name = "1.3^(r-1)-1" if target_name == "Enjoyment" else "1.8^(r-1)-1"
        ax2.set_xlabel("% of Books Kept (sorted by prediction)")
        ax2.set_ylabel(f"Average Utility ({util_name})")
        ax2.set_title("Average Utility by Filter Threshold")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)
        ax2.invert_xaxis()

        plt.tight_layout()
        plt.savefig(PLOTS / f"filtering_{target_name.lower()}.png", dpi=150)
        plt.close()
        print(f"  Saved filtering_{target_name.lower()}.png")


def utility_models(df: pd.DataFrame, all_cat_cols: list[str]) -> None:
    """Train models directly on utility-transformed targets."""
    train_mask = df["dataset"] == "train"
    holdout_mask = df["dataset"] == "holdout"

    df = df.copy()
    df["enjoy_utility"] = 1.3 ** (df["avg_enjoyment"] - 1) - 1
    df["useful_utility"] = 1.8 ** (df["avg_usefulness"] - 1) - 1

    print(f"\n{'=' * 70}")
    print("UTILITY-SPACE MODELS")
    print("  enjoy_utility = 1.3^(enjoyment - 1) - 1")
    print("  useful_utility = 1.8^(usefulness - 1) - 1")
    print(f"{'=' * 70}")

    keep_pcts = [100, 95, 90, 80, 75, 60, 50, 40, 25, 10, 5]

    def _enjoy_util(r: np.ndarray) -> np.ndarray:
        return 1.3 ** (r - 1) - 1

    def _useful_util(r: np.ndarray) -> np.ndarray:
        return 1.8 ** (r - 1) - 1

    for target_name, target_col, raw_col, util_fn in [
        ("Enjoyment Utility", "enjoy_utility", "avg_enjoyment", _enjoy_util),
        ("Usefulness Utility", "useful_utility", "avg_usefulness", _useful_util),
    ]:
        X_train, X_holdout, y_train_util, y_holdout_util, features = _build_features(
            df, train_mask, holdout_mask, target_col, all_cat_cols
        )

        models = _train_models(X_train, y_train_util, features)

        print(f"\n{'─' * 60}")
        print(f"  TARGET: {target_name}")
        print(f"{'─' * 60}")

        for name, model in models.items():
            preds_util = model.predict(X_holdout.values)
            r = (
                np.corrcoef(preds_util, y_holdout_util)[0, 1]
                if len(y_holdout_util) > 2
                else 0
            )
            print(f"  {name}: Holdout R={r:.3f} (in utility space)")

        # Compare: training on utility vs training on raw then converting
        # Raw-space models
        _, _, y_train_raw, y_holdout_raw, _ = _build_features(
            df, train_mask, holdout_mask, raw_col, all_cat_cols
        )
        raw_models = _train_models(X_train, y_train_raw, features)

        print("\n  Comparison: utility-trained vs rating-trained then converted")
        print(f"  {'Model':<20s} {'Util-trained R':>15s} {'Rating-trained R':>17s}")

        util_actual = y_holdout_util

        for name in models:
            # Utility-trained
            pred_util = models[name].predict(X_holdout.values)
            r_util = (
                np.corrcoef(pred_util, util_actual)[0, 1] if len(util_actual) > 2 else 0
            )

            # Rating-trained, then converted
            pred_raw = raw_models[name].predict(X_holdout.values)
            pred_converted = util_fn(pred_raw)
            r_conv = (
                np.corrcoef(pred_converted, util_actual)[0, 1]
                if len(util_actual) > 2
                else 0
            )

            print(f"  {name:<20s} {r_util:15.3f} {r_conv:17.3f}")

        # Filtering analysis in utility space
        best_model = models["DecisionTree"]
        preds = best_model.predict(X_holdout.values)
        pred_sorted_idx = np.argsort(preds)[::-1]
        perfect_sorted_idx = np.argsort(y_holdout_util)[::-1]

        print("\n  Filtering gains (DecisionTree, utility-trained):")
        print(
            f"  {'Keep%':>5s} {'N':>3s} {'Model avg util':>15s} {'Perfect avg util':>17s} {'Baseline':>10s}"
        )

        baseline_util = util_actual.mean()
        for pct in keep_pcts:
            n_keep = max(1, int(len(y_holdout_util) * pct / 100))
            model_util = util_actual[pred_sorted_idx[:n_keep]].mean()
            perfect_util = util_actual[perfect_sorted_idx[:n_keep]].mean()
            print(
                f"  {pct:5d} {n_keep:3d} {model_util:15.3f} {perfect_util:17.3f} {baseline_util:10.3f}"
            )


def main() -> None:
    print("Loading data...")
    df = load_and_join()
    print(f"  {len(df)} books loaded")

    print("Imputing missing ratings...")
    df = impute_missing(df)
    df = find_outliers(df)

    df["category_clean"] = df["category"].apply(_clean_category)

    cat_dummies = pd.get_dummies(df["category_clean"], prefix="cat", drop_first=False)
    all_cat_cols = list(cat_dummies.columns)

    print("\n--- Generating plots ---")
    plot_predicted_vs_actual(df, all_cat_cols)
    plot_by_category(df, all_cat_cols)
    plot_over_time(df, all_cat_cols)
    filtering_analysis(df, all_cat_cols)
    utility_models(df, all_cat_cols)

    print(f"\nAll plots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
