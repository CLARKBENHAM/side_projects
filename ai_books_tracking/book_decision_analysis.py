"""Multi-source book rating analysis with decision rules.

1. Load all sources, average OL/Amazon from dual columns
2. Impute missing ratings via regression from Goodreads
3. Identify outlier books (possible data quality issues)
4. Train prediction models (Ridge, GBM) for enjoyment & usefulness
5. Print decision rules with exact coefficients
6. Output holdout predictions CSV
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, export_text

DATA = Path(__file__).resolve().parent.parent / "data"
AI = Path(__file__).resolve().parent


def _norm_title(title: str) -> str:
    t = str(title).lower().strip()
    t = re.sub(r"[^a-z0-9]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _norm_filename(fn: str) -> str:
    fn = str(fn).lower().strip()
    fn = re.sub(r"\(\d+\)", "", fn)
    fn = re.sub(r"\.(pdf|epub|mobi|html|txt)$", "", fn)
    return re.sub(r"\s+", " ", fn).strip()


def _to_num(x: object) -> float | None:
    if pd.isna(x):
        return None
    try:
        v = float(x)
        return v if v > 0 else None
    except (ValueError, TypeError):
        s = str(x).strip()
        if s in ("N/A", "", "nan"):
            return None
        return None


def _parse_combined(x: object) -> tuple[float | None, float | None]:
    """Parse 'rating|count|url' or 'rating|count' format."""
    if pd.isna(x):
        return None, None
    parts = str(x).split("|")
    rating = _to_num(parts[0]) if len(parts) > 0 else None
    count = _to_num(parts[1]) if len(parts) > 1 else None
    return rating, count


# ── Title alias mappings (same as build_golden_master.py) ─────
TITLE_ALIASES: dict[str, str] = {
    "kelly my share of it all": "kelly more than my share of it all",
    "buckley": "buckley the life and the revolution that changed america",
    "now it can be told": "now it can be told the story of the manhattan project",
    "shaping up": "shape up stop running in circles and ship work that matters",
}


def load_and_join() -> pd.DataFrame:
    """Load all sources and build analysis DataFrame."""
    master = pd.read_csv(
        DATA / "Books Read and their effects - master_book_metadata_cleaned.csv"
    )
    play = pd.read_csv(
        DATA / "Books Read and their effects - Play Export.csv",
        usecols=range(9),
    )
    ratings2 = pd.read_csv(
        DATA / "Books Read and their effects - Ratings 2.csv",
        usecols=range(8),
    )
    new_books = pd.read_csv(
        DATA / "Books Read and their effects - new_books_to_rate 2026.csv"
    )

    # ── Extract external ratings from master ──────────────────────
    rows: list[dict] = []
    for _, m in master.iterrows():
        title = str(m["title"])
        author = (
            str(m["corrected_author"]) if pd.notna(m.get("corrected_author")) else ""
        )
        source = str(m["source"])

        # Goodreads
        gr_rating = _to_num(m.get("goodread ratings"))
        gr_count = _to_num(m.get("goodreads number ratings"))
        gr_reviews = _to_num(m.get("goodreads number reviews"))

        # Open Library: two sources
        ol_r1 = _to_num(m.get("open library ratings"))
        ol_r2 = _to_num(m.get("open library rating"))
        ol_n1 = _to_num(m.get("open library number reviews"))
        ol_n2 = _to_num(m.get("open library num reviews"))

        # Amazon: two sources
        amz_combined = m.get("Amazon combined with links")
        amz_r1, amz_n1 = _parse_combined(amz_combined)
        amz_r2 = _to_num(m.get("amazon ratings"))
        amz_n2 = _to_num(m.get("amzon number reviews"))

        # Average where both available
        ol_vals = [v for v in [ol_r1, ol_r2] if v is not None]
        ol_rating = float(np.mean(ol_vals)) if ol_vals else None
        ol_counts = [v for v in [ol_n1, ol_n2] if v is not None]
        ol_count = float(np.max(ol_counts)) if ol_counts else None

        amz_vals = [v for v in [amz_r1, amz_r2] if v is not None]
        amz_rating = float(np.mean(amz_vals)) if amz_vals else None
        amz_counts = [v for v in [amz_n1, amz_n2] if v is not None]
        amz_count = float(np.max(amz_counts)) if amz_counts else None

        rows.append(
            {
                "title": title,
                "author": author,
                "source": source,
                "filename": m.get("filename") if pd.notna(m.get("filename")) else None,
                "gr_rating": gr_rating,
                "gr_count": gr_count,
                "gr_reviews": gr_reviews,
                "ol_rating_raw": ol_rating,
                "ol_count": ol_count,
                "amz_rating_raw": amz_rating,
                "amz_count": amz_count,
                # Keep both sources for outlier detection
                "ol_r1": ol_r1,
                "ol_r2": ol_r2,
                "amz_r1": amz_r1,
                "amz_r2": amz_r2,
            }
        )

    df = pd.DataFrame(rows)
    df["_title_norm"] = df["title"].apply(_norm_title)
    df["_fn_norm"] = df["filename"].apply(
        lambda x: _norm_filename(x) if pd.notna(x) else ""
    )

    # ── Join personal ratings from Play Export + Ratings 2 ────────
    play["_fn_norm"] = play["filename"].apply(
        lambda x: _norm_filename(x) if pd.notna(x) else ""
    )
    play_by_fn: dict[str, pd.Series] = {}
    for _, row in play.iterrows():
        if row["_fn_norm"]:
            play_by_fn[row["_fn_norm"]] = row

    ratings2["_fn_norm"] = ratings2["filename"].apply(
        lambda x: _norm_filename(x) if pd.notna(x) else ""
    )
    r2_by_fn: dict[str, pd.Series] = {}
    for _, row in ratings2.iterrows():
        if row["_fn_norm"]:
            r2_by_fn[row["_fn_norm"]] = row

    new_books["_title_norm"] = new_books["title"].apply(_norm_title)
    nb_by_title: dict[str, pd.Series] = {}
    for _, row in new_books.iterrows():
        nb_by_title[row["_title_norm"]] = row

    enjoyment_1 = []
    usefulness_1 = []
    enjoyment_2 = []
    usefulness_2 = []
    categories = []

    for _, r in df.iterrows():
        e1 = e2 = u1 = u2 = None
        cat = None

        if r["source"] == "Play Export":
            fn = r["_fn_norm"]
            if fn and fn in play_by_fn:
                p = play_by_fn[fn]
                e1 = _to_num(p.get("Enjoyment (/5)"))
                u1 = _to_num(p.get("Usefulness /5 to Me"))
                cat = p.get("Bookshelf") if pd.notna(p.get("Bookshelf")) else None
            if fn and fn in r2_by_fn:
                r2 = r2_by_fn[fn]
                e2 = _to_num(r2.get("Enjoyment (/5)"))
                u2 = _to_num(r2.get("Usefulness /5 to Me"))
        elif r["source"] == "Holdout 2026":
            tn = r["_title_norm"]
            nb = nb_by_title.get(tn)
            if nb is None:
                for short, master_norm in TITLE_ALIASES.items():
                    if tn == master_norm and short in nb_by_title:
                        nb = nb_by_title[short]
                        break
            if nb is not None:
                e1 = _to_num(nb.get("Enjoyment (/5)"))
                u1 = _to_num(nb.get("Usefulness /5 to Me"))
                e2 = _to_num(nb.get("Enjoyment (/5) 2nd"))
                u2 = _to_num(nb.get("Usefulness /5 to Me.1"))
                cat = nb.get("Bookshelf") if pd.notna(nb.get("Bookshelf")) else None

        enjoyment_1.append(e1)
        usefulness_1.append(u1)
        enjoyment_2.append(e2)
        usefulness_2.append(u2)
        categories.append(cat)

    df["enjoyment_1st"] = enjoyment_1
    df["usefulness_1st"] = usefulness_1
    df["enjoyment_2nd"] = enjoyment_2
    df["usefulness_2nd"] = usefulness_2
    df["category"] = categories

    # Average personal ratings
    df["avg_enjoyment"] = df[["enjoyment_1st", "enjoyment_2nd"]].mean(axis=1)
    df["avg_usefulness"] = df[["usefulness_1st", "usefulness_2nd"]].mean(axis=1)

    # Log counts
    df["log_gr_count"] = np.log1p(df["gr_count"].fillna(0))
    df["log_ol_count"] = np.log1p(df["ol_count"].fillna(0))
    df["log_amz_count"] = np.log1p(df["amz_count"].fillna(0))

    # Dataset split
    df["dataset"] = df["source"].map(
        {"Play Export": "train", "Holdout 2026": "holdout"}
    )

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing OL/Amazon ratings via linear regression from GR."""
    df = df.copy()

    for target_col, raw_col in [
        ("ol_rating", "ol_rating_raw"),
        ("amz_rating", "amz_rating_raw"),
    ]:
        has_both = df["gr_rating"].notna() & df[raw_col].notna()
        if has_both.sum() < 10:
            print(
                f"  {target_col}: too few observations ({has_both.sum()}) for regression, using raw"
            )
            df[target_col] = df[raw_col]
            continue

        X_train = df.loc[has_both, "gr_rating"].values.reshape(-1, 1)
        y_train = df.loc[has_both, raw_col].values
        reg = LinearRegression().fit(X_train, y_train)

        r2 = reg.score(X_train, y_train)
        print(
            f"  {target_col} imputation: {raw_col} = {reg.coef_[0]:.3f} * GR + {reg.intercept_:.3f}  (R²={r2:.3f})"
        )
        print(
            f"    available: {has_both.sum()}, imputing: {(df['gr_rating'].notna() & df[raw_col].isna()).sum()}"
        )

        df[target_col] = df[raw_col].copy()
        needs_impute = df[target_col].isna() & df["gr_rating"].notna()
        if needs_impute.any():
            df.loc[needs_impute, target_col] = reg.predict(
                df.loc[needs_impute, "gr_rating"].values.reshape(-1, 1)
            )

    return df


def find_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Find books where cross-source ratings disagree significantly."""
    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK: Cross-source rating disagreements")
    print("=" * 70)

    # 1. GR vs OL disagreements
    both = df[df["gr_rating"].notna() & df["ol_rating_raw"].notna()].copy()
    both["gr_ol_diff"] = both["gr_rating"] - both["ol_rating_raw"]
    outliers_ol = both[abs(both["gr_ol_diff"]) > 0.5].sort_values(
        "gr_ol_diff", key=abs, ascending=False
    )
    print(
        f"\nGR vs OL: {len(both)} books compared, {len(outliers_ol)} with |diff| > 0.5"
    )
    if len(outliers_ol):
        print(
            outliers_ol[
                [
                    "title",
                    "gr_rating",
                    "ol_rating_raw",
                    "gr_ol_diff",
                    "gr_count",
                    "ol_count",
                ]
            ].to_string(index=False)
        )

    # 2. GR vs Amazon disagreements
    both_amz = df[df["gr_rating"].notna() & df["amz_rating_raw"].notna()].copy()
    both_amz["gr_amz_diff"] = both_amz["gr_rating"] - both_amz["amz_rating_raw"]
    outliers_amz = both_amz[abs(both_amz["gr_amz_diff"]) > 0.5].sort_values(
        "gr_amz_diff", key=abs, ascending=False
    )
    print(
        f"\nGR vs Amazon: {len(both_amz)} books compared, {len(outliers_amz)} with |diff| > 0.5"
    )
    if len(outliers_amz):
        print(
            outliers_amz[
                [
                    "title",
                    "gr_rating",
                    "amz_rating_raw",
                    "gr_amz_diff",
                    "gr_count",
                    "amz_count",
                ]
            ].to_string(index=False)
        )

    # 3. OL source 1 vs source 2 disagreements
    both_ol12 = df[df["ol_r1"].notna() & df["ol_r2"].notna()].copy()
    both_ol12["ol_diff"] = both_ol12["ol_r1"] - both_ol12["ol_r2"]
    outliers_ol12 = both_ol12[abs(both_ol12["ol_diff"]) > 0.1].sort_values(
        "ol_diff", key=abs, ascending=False
    )
    print(
        f"\nOL source 1 vs 2: {len(both_ol12)} compared, {len(outliers_ol12)} with |diff| > 0.1"
    )
    if len(outliers_ol12):
        print(
            outliers_ol12[["title", "ol_r1", "ol_r2", "ol_diff"]].to_string(index=False)
        )

    # 4. Amazon source 1 vs source 2 disagreements
    both_amz12 = df[df["amz_r1"].notna() & df["amz_r2"].notna()].copy()
    both_amz12["amz_diff"] = both_amz12["amz_r1"] - both_amz12["amz_r2"]
    outliers_amz12 = both_amz12[abs(both_amz12["amz_diff"]) > 0.1].sort_values(
        "amz_diff", key=abs, ascending=False
    )
    print(
        f"\nAmazon source 1 vs 2: {len(both_amz12)} compared, {len(outliers_amz12)} with |diff| > 0.1"
    )
    if len(outliers_amz12):
        print(
            outliers_amz12[["title", "amz_r1", "amz_r2", "amz_diff"]].to_string(
                index=False
            )
        )

    # 5. OL quality assessment
    print("\n--- Open Library data quality ---")
    ol_with_count = df[df["ol_count"].notna() & df["ol_rating_raw"].notna()]
    low_count_ol = ol_with_count[ol_with_count["ol_count"] <= 3]
    high_count_ol = ol_with_count[ol_with_count["ol_count"] > 3]
    if len(low_count_ol) > 0 and len(high_count_ol) > 0:
        both_low = low_count_ol[low_count_ol["gr_rating"].notna()]
        both_high = high_count_ol[high_count_ol["gr_rating"].notna()]
        if len(both_low) > 0 and len(both_high) > 0:
            rmse_low = np.sqrt(
                ((both_low["ol_rating_raw"] - both_low["gr_rating"]) ** 2).mean()
            )
            rmse_high = np.sqrt(
                ((both_high["ol_rating_raw"] - both_high["gr_rating"]) ** 2).mean()
            )
            print(
                f"  OL with count <= 3: {len(low_count_ol)} books, RMSE vs GR = {rmse_low:.3f}"
            )
            print(
                f"  OL with count > 3:  {len(high_count_ol)} books, RMSE vs GR = {rmse_high:.3f}"
            )
            print(
                f"  RECOMMENDATION: OL entries with <= 3 ratings are {rmse_low/rmse_high:.1f}x noisier."
            )
            if rmse_low > 2 * rmse_high:
                print("  -> Drop OL ratings with count <= 3 (set to NaN)")
                df.loc[
                    df["ol_count"].notna() & (df["ol_count"] <= 3), "ol_rating_raw"
                ] = None

    # Amazon systematic bias
    print("\n--- Amazon vs Goodreads systematic bias ---")
    both_amz_gr = df[df["gr_rating"].notna() & df["amz_rating_raw"].notna()]
    if len(both_amz_gr) > 0:
        mean_diff = (both_amz_gr["amz_rating_raw"] - both_amz_gr["gr_rating"]).mean()
        print(
            f"  Amazon rates {mean_diff:+.2f} higher than GR on average (N={len(both_amz_gr)})"
        )
        print(
            "  This is systematic bias (Amazon has more casual reviewers), not data errors"
        )

    # 6. Books where one source is much higher/lower than all others
    print("\n--- Books where personal rating deviates most from external consensus ---")
    rated = df[df["avg_enjoyment"].notna() & df["gr_rating"].notna()].copy()
    ext_cols = ["gr_rating", "ol_rating_raw", "amz_rating_raw"]
    rated["ext_mean"] = rated[ext_cols].mean(axis=1)
    rated["personal_vs_ext"] = rated["avg_enjoyment"] - rated["ext_mean"]
    extremes = rated.nlargest(5, "personal_vs_ext")[
        [
            "title",
            "avg_enjoyment",
            "gr_rating",
            "ol_rating_raw",
            "amz_rating_raw",
            "personal_vs_ext",
        ]
    ]
    print("\nYou rated MUCH HIGHER than external consensus:")
    print(extremes.to_string(index=False))
    extremes_low = rated.nsmallest(5, "personal_vs_ext")[
        [
            "title",
            "avg_enjoyment",
            "gr_rating",
            "ol_rating_raw",
            "amz_rating_raw",
            "personal_vs_ext",
        ]
    ]
    print("\nYou rated MUCH LOWER than external consensus:")
    print(extremes_low.to_string(index=False))

    return df


def build_models(df: pd.DataFrame) -> None:
    """Train prediction models and print decision rules."""
    print("\n" + "=" * 70)
    print("PREDICTION MODELS")
    print("=" * 70)

    train = df[df["dataset"] == "train"].copy()
    holdout = df[df["dataset"] == "holdout"].copy()

    # Normalize categories
    cat_map: dict[str, str] = {}
    for cat in df["category"].dropna().unique():
        c = str(cat).strip()
        if c.lower() in ("fiction",):
            cat_map[cat] = "fiction"
        elif c.lower() in ("literature",):
            cat_map[cat] = "Literature"
        elif c.lower() in ("business, management",):
            cat_map[cat] = "Business"
        elif c.lower() in ("general reading",):
            cat_map[cat] = "General Reading"
        elif c.lower() in ("computer science",):
            cat_map[cat] = "Computer Science"
        elif c.lower() in ("histories",):
            cat_map[cat] = "Histories"
        elif c.lower() in ("machine learning",):
            cat_map[cat] = "Machine Learning"
        elif c.lower() in ("math",):
            cat_map[cat] = "Math"
        else:
            cat_map[cat] = c
    df["category_clean"] = df["category"].map(cat_map).fillna("Other")
    train["category_clean"] = train["category"].map(cat_map).fillna("Other")
    holdout["category_clean"] = holdout["category"].map(cat_map).fillna("Other")

    # ── Feature matrix ────────────────────────────────────────────
    feature_cols = [
        "gr_rating",
        "ol_rating",
        "amz_rating",
        "log_gr_count",
        "log_ol_count",
        "log_amz_count",
    ]

    # Category dummies
    cat_dummies = pd.get_dummies(df["category_clean"], prefix="cat", drop_first=False)
    all_cat_cols = list(cat_dummies.columns)

    for target_name, target_col in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        print(f"\n{'─' * 60}")
        print(f"  TARGET: {target_name} ({target_col})")
        print(f"{'─' * 60}")

        # Filter to rows with target + at least GR rating
        train_valid = train[
            train[target_col].notna() & train["gr_rating"].notna()
        ].copy()
        holdout_valid = holdout[holdout[target_col].notna()].copy()

        # Build feature matrices
        train_cats = pd.get_dummies(
            train_valid["category_clean"], prefix="cat", drop_first=False
        )
        holdout_cats = pd.get_dummies(
            holdout_valid["category_clean"], prefix="cat", drop_first=False
        )

        # Ensure same columns
        for c in all_cat_cols:
            if c not in train_cats.columns:
                train_cats[c] = 0
            if c not in holdout_cats.columns:
                holdout_cats[c] = 0
        train_cats = train_cats[all_cat_cols]
        holdout_cats = holdout_cats[all_cat_cols]

        X_train_nums = train_valid[feature_cols].fillna(0)
        X_train = pd.concat([X_train_nums, train_cats], axis=1)

        X_holdout_nums = holdout_valid[feature_cols].fillna(0)
        X_holdout = pd.concat([X_holdout_nums, holdout_cats], axis=1)

        y_train = train_valid[target_col].values
        y_holdout = holdout_valid[target_col].values

        all_features = list(X_train.columns)

        # ── Model 1: Simple GR-only linear ────────────────────────
        gr_only = train_valid[["gr_rating"]].values
        gr_holdout = holdout_valid[["gr_rating"]].fillna(0).values
        lr_simple = LinearRegression().fit(gr_only, y_train)
        pred_simple = lr_simple.predict(gr_holdout)
        r_simple = (
            np.corrcoef(pred_simple, y_holdout)[0, 1] if len(y_holdout) > 2 else 0
        )
        rmse_simple = np.sqrt(mean_squared_error(y_holdout, pred_simple))

        print("\n  Model 1: GR-only linear")
        print(
            f"    {target_name} = {lr_simple.coef_[0]:.3f} * GR_rating + {lr_simple.intercept_:.3f}"
        )
        print(
            f"    Train R={np.corrcoef(lr_simple.predict(gr_only), y_train)[0,1]:.3f}, Holdout R={r_simple:.3f}, RMSE={rmse_simple:.3f}"
        )

        # ── Model 2: Ridge with all features ──────────────────────
        ridge = Ridge(alpha=1.0).fit(X_train.values, y_train)
        pred_ridge = ridge.predict(X_holdout.values)
        r_ridge = np.corrcoef(pred_ridge, y_holdout)[0, 1] if len(y_holdout) > 2 else 0
        rmse_ridge = np.sqrt(mean_squared_error(y_holdout, pred_ridge))

        print("\n  Model 2: Ridge (all features)")
        print(f"    Holdout R={r_ridge:.3f}, RMSE={rmse_ridge:.3f}")
        print(f"    Intercept: {ridge.intercept_:.4f}")
        print("    Coefficients:")
        coef_df = pd.DataFrame({"feature": all_features, "coef": ridge.coef_})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False)
        for _, c in coef_df.iterrows():
            if abs(c["coef"]) > 0.01:
                print(f"      {c['feature']:30s}  {c['coef']:+.4f}")

        # Full equation
        print(f"\n    EQUATION: {target_name} = {ridge.intercept_:.4f}", end="")
        for _, c in coef_df.iterrows():
            if abs(c["coef"]) > 0.01:
                print(f"\n      {c['coef']:+.4f} * {c['feature']}", end="")
        print()

        # ── Model 3: GBM ─────────────────────────────────────────
        gbm = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            random_state=42,
        ).fit(X_train.values, y_train)
        pred_gbm = gbm.predict(X_holdout.values)
        r_gbm = np.corrcoef(pred_gbm, y_holdout)[0, 1] if len(y_holdout) > 2 else 0
        rmse_gbm = np.sqrt(mean_squared_error(y_holdout, pred_gbm))

        print("\n  Model 3: GBM")
        print(f"    Holdout R={r_gbm:.3f}, RMSE={rmse_gbm:.3f}")
        print("    Feature importances:")
        imp_df = pd.DataFrame(
            {"feature": all_features, "importance": gbm.feature_importances_}
        )
        imp_df = imp_df.sort_values("importance", ascending=False)
        for _, imp in imp_df.iterrows():
            if imp["importance"] > 0.01:
                print(f"      {imp['feature']:30s}  {imp['importance']:.3f}")

        # ── Model 4: Decision Tree (interpretable) ─────────────
        dt = DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=10,
            random_state=42,
        ).fit(X_train.values, y_train)
        pred_dt = dt.predict(X_holdout.values)
        r_dt = np.corrcoef(pred_dt, y_holdout)[0, 1] if len(y_holdout) > 2 else 0
        rmse_dt = np.sqrt(mean_squared_error(y_holdout, pred_dt))

        print("\n  Model 4: Decision Tree (max_depth=3)")
        print(f"    Holdout R={r_dt:.3f}, RMSE={rmse_dt:.3f}")
        tree_text = export_text(dt, feature_names=all_features, decimals=2)
        print("    Tree rules:")
        for line in tree_text.split("\n"):
            print(f"      {line}")

        # ── Model comparison ──────────────────────────────────────
        print(f"\n  Model comparison (holdout, N={len(y_holdout)}):")
        print(f"    {'Model':<25s} {'R':>6s} {'RMSE':>6s}")
        print(f"    {'GR-only linear':<25s} {r_simple:6.3f} {rmse_simple:6.3f}")
        print(f"    {'Ridge (all features)':<25s} {r_ridge:6.3f} {rmse_ridge:6.3f}")
        print(f"    {'GBM (all features)':<25s} {r_gbm:6.3f} {rmse_gbm:6.3f}")
        print(f"    {'Decision Tree':<25s} {r_dt:6.3f} {rmse_dt:6.3f}")

        # ── Decision rules (GR threshold) ─────────────────────────
        if target_name == "Enjoyment":
            print("\n  DECISION RULES (GR rating thresholds):")
            print("  Utility function: 1.3^(r-1) - 1")
            for thresh in [3.8, 3.9, 4.0, 4.1, 4.2]:
                above = holdout_valid[holdout_valid["gr_rating"] >= thresh]
                below = holdout_valid[holdout_valid["gr_rating"] < thresh]
                if len(above) == 0 or len(below) == 0:
                    continue
                util_above = (1.3 ** (above[target_col].values - 1) - 1).mean()
                util_below = (1.3 ** (below[target_col].values - 1) - 1).mean()
                util_all = (1.3 ** (holdout_valid[target_col].values - 1) - 1).mean()
                gain = (util_above - util_all) / util_all * 100 if util_all > 0 else 0
                print(
                    f"    GR >= {thresh}: keep {len(above)}/{len(holdout_valid)} books, "
                    f"avg enjoyment {above[target_col].mean():.2f} vs {below[target_col].mean():.2f}, "
                    f"utility above {util_above:.3f} vs below {util_below:.3f}, "
                    f"gain vs all {gain:+.1f}%"
                )

        # ── Store predictions on holdout ──────────────────────────
        holdout_valid[f"pred_{target_name.lower()}_simple"] = pred_simple
        holdout_valid[f"pred_{target_name.lower()}_ridge"] = pred_ridge
        holdout_valid[f"pred_{target_name.lower()}_gbm"] = pred_gbm

        if target_name == "Enjoyment":
            holdout_enjoy = holdout_valid
        else:
            holdout_use = holdout_valid

    # ── Category means as additional signal ───────────────────────
    print(f"\n{'─' * 60}")
    print("  CATEGORY MEANS (training set)")
    print(f"{'─' * 60}")
    cat_means = (
        train.groupby("category_clean")
        .agg(
            n=("avg_enjoyment", "size"),
            avg_enjoy=("avg_enjoyment", "mean"),
            avg_useful=("avg_usefulness", "mean"),
            avg_gr=("gr_rating", "mean"),
        )
        .sort_values("avg_enjoy", ascending=False)
    )
    print(cat_means.to_string())

    # ── Combined decision rule ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RECOMMENDED DECISION RULE")
    print(f"{'=' * 70}")

    # Use the best model for predictions
    best_model_name = max(
        [("GR-only", r_simple), ("Ridge", r_ridge), ("GBM", r_gbm)],
        key=lambda x: x[1],
    )[0]
    print(f"\nBest enjoyment model on holdout: {best_model_name}")

    # Print the Ridge equation as the interpretable rule
    print("\nInterpretable rule (Ridge coefficients):")
    print("  Read a book if predicted enjoyment >= 3.0")
    print("  (See EQUATION above for full formula)")

    # ── Output holdout predictions CSV ────────────────────────────
    out_cols = [
        "title",
        "author",
        "category_clean",
        "gr_rating",
        "ol_rating",
        "amz_rating",
        "avg_enjoyment",
        "avg_usefulness",
    ]
    # Merge enjoyment and usefulness predictions
    pred_cols_e = [c for c in holdout_enjoy.columns if c.startswith("pred_enjoyment")]
    pred_cols_u = [c for c in holdout_use.columns if c.startswith("pred_usefulness")]

    out = holdout_enjoy[out_cols + pred_cols_e].copy()
    for c in pred_cols_u:
        out[c] = holdout_use[c].values

    out = out.sort_values("avg_enjoyment", ascending=False)
    out_path = AI / "holdout_predictions_2026.csv"
    out.to_csv(out_path, index=False)
    print(f"\nHoldout predictions saved to {out_path}")
    print(f"  {len(out)} books with predictions vs actuals")

    # Print prediction accuracy summary
    print(f"\n{'─' * 60}")
    print("  HOLDOUT PREDICTIONS vs ACTUALS (sorted by actual enjoyment)")
    print(f"{'─' * 60}")
    display_cols = [
        "title",
        "avg_enjoyment",
        "pred_enjoyment_ridge",
        "avg_usefulness",
        "pred_usefulness_ridge",
        "gr_rating",
    ]
    display_cols = [c for c in display_cols if c in out.columns]
    out_display = out[display_cols].copy()
    for c in display_cols:
        if c.startswith("pred_") or c.startswith("avg_"):
            out_display[c] = out_display[c].round(2)
    print(out_display.to_string(index=False))


def main() -> None:
    print("Loading and joining data...")
    df = load_and_join()
    print(f"  {len(df)} books loaded")
    print(
        f"  Train: {(df['dataset']=='train').sum()}, Holdout: {(df['dataset']=='holdout').sum()}"
    )
    print(f"  Rated: {df['avg_enjoyment'].notna().sum()}")

    print("\nImputing missing ratings...")
    df = impute_missing(df)

    find_outliers(df)
    build_models(df)


if __name__ == "__main__":
    main()
