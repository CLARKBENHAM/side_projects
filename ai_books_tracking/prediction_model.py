"""Build prediction models for book enjoyment and usefulness.
Uses features from feature_extraction.py and api_enrichment.py."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path(__file__).parent
ENRICHED_FILE = OUTPUT_DIR / "books_enriched.csv"
FEATURES_FILE = OUTPUT_DIR / "books_with_features.csv"


def load_enriched() -> pd.DataFrame:
    if ENRICHED_FILE.exists():
        df = pd.read_csv(ENRICHED_FILE)
    elif FEATURES_FILE.exists():
        df = pd.read_csv(FEATURES_FILE)
    else:
        raise FileNotFoundError("Run feature_extraction.py and api_enrichment.py first")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare feature matrix for modeling."""
    df = df.copy()

    # Parse dates if needed
    for col in ["earliest_modified", "latest_modified"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    # Derived features
    if "reading_days" not in df.columns:
        df["reading_days"] = (df["latest_modified"] - df["earliest_modified"]).dt.days

    if "year_finished" not in df.columns:
        df["year_finished"] = df["latest_modified"].dt.year

    if "month_finished" not in df.columns:
        df["month_finished"] = df["latest_modified"].dt.month

    # Note length
    if "Long Term Effects" in df.columns:
        df["note_length"] = df["Long Term Effects"].fillna("").str.len()
    else:
        df["note_length"] = 0

    # Log page count
    if "gb_page_count" in df.columns:
        df["log_pages"] = np.log1p(df["gb_page_count"].fillna(0))
    else:
        df["log_pages"] = 0

    # Book age (years since publication)
    if "pub_year" in df.columns:
        df["book_age"] = 2025 - df["pub_year"].fillna(2025)
    else:
        df["book_age"] = 0

    # Author mean enjoyment (leave-one-out)
    if "author" in df.columns:
        author_clean = df["author"].str.strip().str.lower()
        author_means = author_clean.map(
            df.groupby(author_clean)["Enjoyment (/5)"].transform("mean")
        )
        author_counts = author_clean.map(author_clean.value_counts())
        # LOO: subtract own contribution
        df["author_mean_enjoy_loo"] = np.where(
            author_counts > 1,
            (author_means * author_counts - df["Enjoyment (/5)"]) / (author_counts - 1),
            np.nan,
        )
        df["author_mean_enjoy_loo"] = df["author_mean_enjoy_loo"].fillna(
            df["Enjoyment (/5)"].mean()
        )
        df["author_book_count"] = author_counts
    else:
        df["author_mean_enjoy_loo"] = df["Enjoyment (/5)"].mean()
        df["author_book_count"] = 1

    numeric_features = [
        "year_finished",
        "reading_days",
        "note_length",
        "log_pages",
        "book_age",
        "author_mean_enjoy_loo",
        "author_book_count",
    ]
    if "gb_average_rating" in df.columns:
        numeric_features.append("gb_average_rating")
    if "gb_ratings_count" in df.columns:
        df["log_ratings_count"] = np.log1p(df["gb_ratings_count"].fillna(0))
        numeric_features.append("log_ratings_count")

    categorical_features = ["Bookshelf"]
    if "inferred_source" in df.columns:
        categorical_features.append("inferred_source")

    # Fill NaNs in numeric features
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill NaNs in categorical features
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df, numeric_features, categorical_features


def build_and_evaluate(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
) -> dict[str, float]:
    """Build models with LOO cross-validation and compare to baselines."""
    print(f"\n{'=' * 70}")
    print(f"PREDICTING: {target}")
    print(f"{'=' * 70}")

    valid_mask = df[target].notna()
    for feat in numeric_features:
        valid_mask &= df[feat].notna()
    df_valid = df[valid_mask].copy()

    y = df_valid[target].values
    n = len(y)

    print(f"\n{n} books with valid data")

    # Baseline 1: Global mean
    global_mean = y.mean()
    baseline_mae = mean_absolute_error(y, np.full(n, global_mean))
    baseline_rmse = np.sqrt(mean_squared_error(y, np.full(n, global_mean)))
    print(f"\nBaseline (global mean={global_mean:.2f}):")
    print(f"  MAE={baseline_mae:.3f}, RMSE={baseline_rmse:.3f}")

    # Baseline 2: Category mean (LOO)
    cat_predictions = np.zeros(n)
    for i in range(n):
        cat = df_valid.iloc[i]["Bookshelf"]
        others = df_valid.drop(df_valid.index[i])
        cat_mean = others[others["Bookshelf"] == cat][target].mean()
        cat_predictions[i] = cat_mean if not np.isnan(cat_mean) else global_mean

    cat_mae = mean_absolute_error(y, cat_predictions)
    cat_rmse = np.sqrt(mean_squared_error(y, cat_predictions))
    print("\nBaseline (category mean, LOO):")
    print(f"  MAE={cat_mae:.3f}, RMSE={cat_rmse:.3f}")

    # Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [f for f in numeric_features if f in df_valid.columns],
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                ),
                [f for f in categorical_features if f in df_valid.columns],
            ),
        ]
    )

    # Models to try
    models = {
        "Ridge (alpha=1)": Ridge(alpha=1.0),
        "Ridge (alpha=10)": Ridge(alpha=10.0),
        "Lasso (alpha=0.1)": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "GBM": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    loo = LeaveOneOut()

    print(f"\n{'Model':<25} {'MAE':>7} {'RMSE':>7} {'vs Cat MAE':>12}")
    print("-" * 55)

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        preds = cross_val_predict(pipe, df_valid, y, cv=loo)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        improvement = (cat_mae - mae) / cat_mae * 100
        print(f"{name:<25} {mae:>7.3f} {rmse:>7.3f} {improvement:>+11.1f}%")
        results[name] = {"mae": mae, "rmse": rmse, "predictions": preds}

    # Feature importance from best model
    best_name = min(results, key=lambda k: results[k]["mae"])
    print(f"\nBest model: {best_name}")

    # Fit on full data for feature importance
    best_model_class = models[best_name]
    pipe = Pipeline([("prep", preprocessor), ("model", best_model_class)])
    pipe.fit(df_valid, y)

    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(pipe.named_steps["model"], "coef_"):
        importances = np.abs(pipe.named_steps["model"].coef_)
    elif hasattr(pipe.named_steps["model"], "feature_importances_"):
        importances = pipe.named_steps["model"].feature_importances_
    else:
        importances = None

    if importances is not None:
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\nFeature importance ({best_name}):")
        for i in sorted_idx[:15]:
            print(f"  {feature_names[i]:<40} {importances[i]:.4f}")

    return results


def ablation_study(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
) -> None:
    """Test which feature groups add predictive value beyond category."""
    print(f"\n{'=' * 70}")
    print(f"ABLATION STUDY: {target}")
    print(f"{'=' * 70}")

    valid_mask = df[target].notna()
    for feat in numeric_features:
        valid_mask &= df[feat].notna()
    df_valid = df[valid_mask].copy()
    y = df_valid[target].values

    loo = LeaveOneOut()

    feature_groups = {
        "Category only": ([], ["Bookshelf"]),
        "+ Author features": (
            ["author_mean_enjoy_loo", "author_book_count"],
            ["Bookshelf"],
        ),
        "+ Temporal": (["year_finished", "reading_days"], ["Bookshelf"]),
        "+ Book metadata": (["log_pages", "book_age"], ["Bookshelf"]),
        "+ Note length": (["note_length"], ["Bookshelf"]),
    }

    if (
        "gb_average_rating" in df_valid.columns
        and df_valid["gb_average_rating"].notna().sum() > 50
    ):
        feature_groups["+ External rating"] = (["gb_average_rating"], ["Bookshelf"])
        if "log_ratings_count" in df_valid.columns:
            feature_groups["+ External rating+count"] = (
                ["gb_average_rating", "log_ratings_count"],
                ["Bookshelf"],
            )

    if "inferred_source" in df_valid.columns:
        feature_groups["+ Rec source"] = ([], ["Bookshelf", "inferred_source"])

    feature_groups["All features"] = (numeric_features, categorical_features)

    print(f"\n{'Feature set':<30} {'MAE':>7} {'RMSE':>7}")
    print("-" * 48)

    for group_name, (num_feats, cat_feats) in feature_groups.items():
        num_available = [f for f in num_feats if f in df_valid.columns]
        cat_available = [f for f in cat_feats if f in df_valid.columns]

        transformers = []
        if num_available:
            transformers.append(("num", StandardScaler(), num_available))
        if cat_available:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="infrequent_if_exist",
                    ),
                    cat_available,
                )
            )

        if not transformers:
            continue

        preprocessor = ColumnTransformer(transformers=transformers)
        pipe = Pipeline([("prep", preprocessor), ("model", Ridge(alpha=10.0))])
        preds = cross_val_predict(pipe, df_valid, y, cv=loo)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"{group_name:<30} {mae:>7.3f} {rmse:>7.3f}")


def retrospective_analysis(df: pd.DataFrame) -> None:
    """For each book, estimate how much time could have been saved
    with optimal stopping given the prediction model."""
    print(f"\n{'=' * 70}")
    print("RETROSPECTIVE: SHOULD I HAVE QUIT?")
    print(f"{'=' * 70}")

    # Use the simulation results from the blog post
    # The blog says to drop 77-94% of books depending on category
    drop_rates = {
        "fiction": 0.942,
        "Literature": 0.926,
        "Business, management": 0.890,
        "General Reading": 0.898,
        "Computer Science": 0.774,
    }

    for cat, drop_rate in drop_rates.items():
        cat_df = df[df["Bookshelf"] == cat].copy()
        if len(cat_df) == 0:
            continue

        cat_df = cat_df.sort_values("Enjoyment (/5)")
        n = len(cat_df)
        n_to_drop = int(n * drop_rate)
        keep_threshold_idx = min(n_to_drop, n - 1)

        threshold = cat_df.iloc[keep_threshold_idx]["Enjoyment (/5)"]
        would_keep = cat_df[cat_df["Enjoyment (/5)"] >= threshold]
        would_drop = cat_df[cat_df["Enjoyment (/5)"] < threshold]

        print(f"\n{cat} (n={n}, optimal drop={drop_rate:.0%}):")
        print(f"  Would keep {len(would_keep)} books (enjoy >= {threshold:.1f})")
        print(f"  Would drop {len(would_drop)} books")
        if len(would_keep) > 0:
            print(
                f"  Kept avg enjoy: {would_keep['Enjoyment (/5)'].mean():.2f} "
                f"vs current: {cat_df['Enjoyment (/5)'].mean():.2f}"
            )
            improvement = (
                would_keep["Enjoyment (/5)"].mean() - cat_df["Enjoyment (/5)"].mean()
            )
            print(f"  Enjoyment improvement: +{improvement:.2f} pts")

        # What books would have been wrongly dropped? (rated low but high on re-read)
        if len(would_drop) > 0:
            high_in_dropped = would_drop[would_drop["Enjoyment (/5)"] >= 3.5]
            if len(high_in_dropped) > 0:
                print(
                    f"  WARNING: {len(high_in_dropped)} books rated >= 3.5 would be dropped:"
                )
                for _, row in high_in_dropped.iterrows():
                    print(
                        f"    - {row['title'][:60]} (enjoy={row['Enjoyment (/5)']:.1f})"
                    )


def main() -> None:
    df = load_enriched()
    df, numeric_features, categorical_features = prepare_features(df)

    # Full model comparison
    for target in ["Enjoyment (/5)", "Usefulness /5 to Me"]:
        build_and_evaluate(df, numeric_features, categorical_features, target)

    # Ablation study
    for target in ["Enjoyment (/5)", "Usefulness /5 to Me"]:
        ablation_study(df, numeric_features, categorical_features, target)

    # Retrospective
    retrospective_analysis(df)


if __name__ == "__main__":
    main()
