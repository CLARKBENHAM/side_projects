"""Final analysis: self-consistency ceiling, simple decision rule, and write-up data."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from book_decision_analysis import load_and_join, impute_missing, find_outliers


def main() -> None:
    df = load_and_join()
    df = impute_missing(df)
    # Suppress the outlier printing
    import io
    import sys

    old = sys.stdout
    sys.stdout = io.StringIO()
    df = find_outliers(df)
    sys.stdout = old

    cat_map = {
        "fiction": "fiction",
        "Literature": "Literature",
        "Business, management": "Business",
        "General Reading": "General Reading",
        "Computer Science": "CS",
        "Histories": "Histories",
        "Machine Learning": "ML",
        "Math": "Math",
    }
    df["category_clean"] = df["category"].apply(
        lambda c: (
            cat_map.get(str(c).strip(), str(c).strip()) if pd.notna(c) else "Other"
        )
    )

    print("=" * 70)
    print("1. YOUR RATING SELF-CONSISTENCY (irreducible entropy)")
    print("=" * 70)

    for name, c1, c2 in [
        ("Enjoyment", "enjoyment_1st", "enjoyment_2nd"),
        ("Usefulness", "usefulness_1st", "usefulness_2nd"),
    ]:
        both = df[df[c1].notna() & df[c2].notna()]
        if len(both) < 3:
            print(f"\n{name}: only {len(both)} books with both passes")
            continue
        diff = both[c1] - both[c2]
        rmse = np.sqrt((diff**2).mean())
        r = np.corrcoef(both[c1], both[c2])[0, 1]
        mae = diff.abs().mean()
        print(f"\n{name} (N={len(both)} books rated twice):")
        print(
            f"  1st pass mean={both[c1].mean():.2f}, 2nd pass mean={both[c2].mean():.2f}"
        )
        print(f"  Self-correlation R={r:.3f}")
        print(f"  Self-RMSE={rmse:.3f}, Self-MAE={mae:.3f}")
        for d in [0, 0.5, 1.0, 1.5, 2.0]:
            n_d = (diff.abs() <= d).sum()
            print(
                f"    |diff| <= {d}: {n_d}/{len(both)} ({n_d / len(both) * 100:.0f}%)"
            )

    print("\n" + "=" * 70)
    print("2. MODEL PERFORMANCE vs CEILING")
    print("=" * 70)

    train = df[df["dataset"] == "train"]
    holdout = df[df["dataset"] == "holdout"]
    feature_cols = [
        "gr_rating",
        "ol_rating",
        "amz_rating",
        "log_gr_count",
        "log_ol_count",
        "log_amz_count",
    ]
    cat_dummies = pd.get_dummies(df["category_clean"], prefix="cat", drop_first=False)
    all_cat_cols = list(cat_dummies.columns)

    for tname, tcol in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        tv = train[train[tcol].notna() & train["gr_rating"].notna()]
        hv = holdout[holdout[tcol].notna()]
        tc = pd.get_dummies(tv["category_clean"], prefix="cat", drop_first=False)
        hc = pd.get_dummies(hv["category_clean"], prefix="cat", drop_first=False)
        for c in all_cat_cols:
            if c not in tc.columns:
                tc[c] = 0
            if c not in hc.columns:
                hc[c] = 0
        Xt = pd.concat([tv[feature_cols].fillna(0), tc[all_cat_cols]], axis=1)
        Xh = pd.concat([hv[feature_cols].fillna(0), hc[all_cat_cols]], axis=1)
        yt = tv[tcol].values
        yh = hv[tcol].values

        dt = DecisionTreeRegressor(
            max_depth=3, min_samples_leaf=10, random_state=42
        ).fit(Xt.values, yt)
        pred = dt.predict(Xh.values)
        r_model = np.corrcoef(pred, yh)[0, 1]
        rmse_model = np.sqrt(((pred - yh) ** 2).mean())

        # GR-only baseline
        gr_corr = np.corrcoef(hv["gr_rating"].fillna(0).values, yh)[0, 1]

        # Self-consistency for comparison
        both = (
            df[
                df[f"{tcol.replace('avg_', '')}1st"].notna()
                & df[f"{tcol.replace('avg_', '')}2nd"].notna()
            ]
            if False
            else pd.DataFrame()
        )

        print(f"\n{tname}:")
        print(f"  Holdout std (total variance):  {yh.std():.3f}")
        print(f"  GR-only correlation:           R={gr_corr:.3f}")
        print(
            f"  Best model (DT depth=3):       R={r_model:.3f}, RMSE={rmse_model:.3f}"
        )
        print(f"  Model RMSE / holdout std:      {rmse_model / yh.std():.2f}")
        print(
            f"  R² (variance explained):       {r_model**2:.3f} ({r_model**2 * 100:.1f}%)"
        )

    # Self-consistency R for comparison
    both_e = df[df["enjoyment_1st"].notna() & df["enjoyment_2nd"].notna()]
    both_u = df[df["usefulness_1st"].notna() & df["usefulness_2nd"].notna()]
    if len(both_e) >= 3:
        r_self_e = np.corrcoef(both_e["enjoyment_1st"], both_e["enjoyment_2nd"])[0, 1]
        rmse_self_e = np.sqrt(
            ((both_e["enjoyment_1st"] - both_e["enjoyment_2nd"]) ** 2).mean()
        )
        print(
            f"\n  Self-consistency ceiling (Enjoyment): R={r_self_e:.3f}, RMSE={rmse_self_e:.3f}"
        )
    if len(both_u) >= 3:
        r_self_u = np.corrcoef(both_u["usefulness_1st"], both_u["usefulness_2nd"])[0, 1]
        rmse_self_u = np.sqrt(
            ((both_u["usefulness_1st"] - both_u["usefulness_2nd"]) ** 2).mean()
        )
        print(
            f"  Self-consistency ceiling (Usefulness): R={r_self_u:.3f}, RMSE={rmse_self_u:.3f}"
        )

    print("\n" + "=" * 70)
    print("3. DATA SOURCE INVENTORY")
    print("=" * 70)

    print(f"\nTotal books: {len(df)}")
    print(f"  Play Export (train): {(df['source'] == 'Play Export').sum()}")
    print(f"  Holdout 2026: {(df['source'] == 'Holdout 2026').sum()}")

    print("\nRating sources coverage:")
    for col, label in [
        ("gr_rating", "Goodreads rating"),
        ("ol_rating_raw", "Open Library raw"),
        ("amz_rating_raw", "Amazon raw"),
        ("ol_rating", "OL (after impute)"),
        ("amz_rating", "Amazon (after impute)"),
        ("gr_count", "GR review count"),
        ("ol_count", "OL review count"),
        ("amz_count", "Amazon review count"),
    ]:
        n = df[col].notna().sum()
        print(f"  {label:25s}: {n:3d}/{len(df)} ({n / len(df) * 100:.0f}%)")

    print("\nPersonal ratings:")
    for col, label in [
        ("enjoyment_1st", "Enjoyment 1st pass"),
        ("enjoyment_2nd", "Enjoyment 2nd pass"),
        ("usefulness_1st", "Usefulness 1st pass"),
        ("usefulness_2nd", "Usefulness 2nd pass"),
        ("avg_enjoyment", "Average enjoyment"),
        ("avg_usefulness", "Average usefulness"),
    ]:
        n = df[col].notna().sum()
        print(f"  {label:25s}: {n:3d}/{len(df)} ({n / len(df) * 100:.0f}%)")

    has_cat = (df["category"].notna() & (df["category"] != "")).sum()
    print(
        f"  {'Has category':25s}: {has_cat:3d}/{len(df)} ({has_cat / len(df) * 100:.0f}%)"
    )

    print("\nCategory distribution:")
    for cat, cnt in df["category_clean"].value_counts().items():
        train_n = ((df["dataset"] == "train") & (df["category_clean"] == cat)).sum()
        hold_n = ((df["dataset"] == "holdout") & (df["category_clean"] == cat)).sum()
        print(f"  {cat:20s}: {cnt:3d} (train={train_n}, holdout={hold_n})")

    print("\n" + "=" * 70)
    print("4. SIMPLEST DECISION RULE + HOLDOUT VALIDATION")
    print("=" * 70)

    hv = holdout[holdout["avg_enjoyment"].notna()].copy()
    hv_with_gr = hv[hv["gr_rating"].notna()]
    n_total = len(hv)
    n_with_gr = len(hv_with_gr)

    print(f"\nHoldout set: {n_total} books, {n_with_gr} with GR rating")
    print(f"Baseline avg enjoyment (all): {hv['avg_enjoyment'].mean():.2f}")
    print(f"Baseline avg usefulness (all): {hv['avg_usefulness'].mean():.2f}")

    # Simple GR threshold rules
    print("\nSimple GR threshold rules (holdout):")
    print(
        f"  {'Rule':40s} {'N kept':>6s} {'Avg Enjoy':>10s} {'Avg Useful':>11s} {'Enjoy Gain':>11s}"
    )
    baseline_e = hv["avg_enjoyment"].mean()
    for thresh in [3.5, 3.8, 4.0, 4.1, 4.2, 4.3]:
        above = hv_with_gr[hv_with_gr["gr_rating"] >= thresh]
        if len(above) == 0:
            continue
        ae = above["avg_enjoyment"].mean()
        au = above["avg_usefulness"].mean()
        gain = ae - baseline_e
        print(
            f"  GR >= {thresh} {'':33s} {len(above):6d} {ae:10.2f} {au:11.2f} {gain:+11.2f}"
        )

    # Category + GR combined rules
    print("\nCategory + GR combined rules (holdout):")
    for cat in ["fiction", "Histories", "Business", "General Reading"]:
        cat_books = hv_with_gr[hv_with_gr["category_clean"] == cat]
        if len(cat_books) < 3:
            continue
        avg_e = cat_books["avg_enjoyment"].mean()
        for thresh in [4.0, 4.2]:
            above = cat_books[cat_books["gr_rating"] >= thresh]
            if len(above) >= 2:
                ae = above["avg_enjoyment"].mean()
                print(
                    f"  {cat} & GR >= {thresh}: {len(above)} books, avg enjoy {ae:.2f} (vs {avg_e:.2f} for all {cat})"
                )

    # What would the next year look like?
    print("\n" + "=" * 70)
    print("5. WHAT WOULD NEXT YEAR LOOK LIKE?")
    print("=" * 70)

    print(f"\nYou read {n_total} books in the holdout period.")
    for thresh in [4.0, 4.1, 4.2]:
        above = hv_with_gr[hv_with_gr["gr_rating"] >= thresh]
        n_kept = len(above)
        n_dropped = n_with_gr - n_kept
        ae = above["avg_enjoyment"].mean()
        au = above["avg_usefulness"].mean()
        # Utility calc
        util_kept = (1.3 ** (above["avg_enjoyment"].values - 1) - 1).mean()
        util_all = (1.3 ** (hv_with_gr["avg_enjoyment"].values - 1) - 1).mean()
        util_gain_pct = (util_kept - util_all) / util_all * 100 if util_all > 0 else 0

        dropped = hv_with_gr[hv_with_gr["gr_rating"] < thresh]
        dropped_good = dropped[dropped["avg_enjoyment"] >= 3.5]
        n_false_neg = len(dropped_good)

        print(f"\n  If GR >= {thresh}:")
        print(f"    Keep {n_kept}/{n_with_gr} books ({n_kept/n_with_gr*100:.0f}%)")
        print(f"    Drop {n_dropped} books")
        print(
            f"    Avg enjoyment: {ae:.2f} (vs {hv_with_gr['avg_enjoyment'].mean():.2f} baseline)"
        )
        print(
            f"    Avg usefulness: {au:.2f} (vs {hv_with_gr['avg_usefulness'].mean():.2f} baseline)"
        )
        print(f"    Utility gain: {util_gain_pct:+.1f}%")
        print(f"    FALSE NEGATIVES (dropped but enjoy >= 3.5): {n_false_neg}")
        if n_false_neg > 0:
            for _, row in dropped_good.iterrows():
                print(
                    f"      - {row['title'][:60]:60s} enjoy={row['avg_enjoyment']:.1f} GR={row['gr_rating']:.2f}"
                )

    print("\n" + "=" * 70)
    print("6. DIMINISHING RETURNS ANALYSIS")
    print("=" * 70)

    print("\nHow much variance can models explain?")
    for tname, tcol in [
        ("Enjoyment", "avg_enjoyment"),
        ("Usefulness", "avg_usefulness"),
    ]:
        yh = holdout[holdout[tcol].notna()][tcol].values
        # From our models
        for model_name, model_r in [
            (
                "GR only",
                np.corrcoef(
                    holdout[holdout[tcol].notna()]["gr_rating"].fillna(0).values, yh
                )[0, 1],
            ),
            (
                "DT depth=3",
                r_model if tname == "Usefulness" else np.corrcoef(pred, yh)[0, 1],
            ),
        ]:
            explained = model_r**2 * 100
            print(
                f"  {tname} - {model_name}: R={model_r:.3f}, explains {explained:.1f}% of variance"
            )

    both_e = df[df["enjoyment_1st"].notna() & df["enjoyment_2nd"].notna()]
    if len(both_e) >= 3:
        r_self = np.corrcoef(both_e["enjoyment_1st"], both_e["enjoyment_2nd"])[0, 1]
        print(
            f"\n  Your own rating consistency: R={r_self:.3f}, {r_self**2 * 100:.1f}% shared variance"
        )
        print(
            f"  This means {(1 - r_self**2) * 100:.1f}% of YOUR OWN rating variance is noise/mood"
        )
        print(f"  Best possible model with perfect features: R={r_self:.3f}")
        print(f"  Current best model: R={r_model:.3f}")
        print(
            f"  Gap: {(r_self - r_model):.3f} in R, or {(r_self**2 - r_model**2) * 100:.1f}pp in R²"
        )


if __name__ == "__main__":
    main()
