#!/usr/bin/env python3
"""
Select books optimally for MODEL VALIDATION, not just highest predicted.

Three goals:
  1. Show what each model (RF, GBM, Simple Ridge) recommends from Gen Reading + Business + CS
  2. Design a validation set that TESTS the model, not just uses it
  3. List additional diagnostics to check forecast accuracy

The existing recommendation lists optimize for "pick the best books."
This script optimizes for "learn whether the model is trustworthy."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR.parent / "data"

TARGET_CATEGORIES = {"General Reading", "Business, management", "Computer Science"}
TARGET_CATEGORIES_SHORT = {"General Reading", "Business", "Computer Science"}


def load_scores_with_ridge() -> pd.DataFrame:
    """Load unread book scores and add Simple Ridge predictions."""
    scores = pd.read_csv(OUTPUT_DIR / "ai_actions" / "unread_book_scores.csv")

    master = pd.read_csv(OUTPUT_DIR / "FINAL_CONSOLIDATED_MASTER.csv")
    master["avg_enjoyment"] = pd.to_numeric(master.get("avg_enjoyment", pd.Series(dtype=float)), errors="coerce")
    master["avg_usefulness"] = pd.to_numeric(master.get("avg_usefulness", pd.Series(dtype=float)), errors="coerce")

    if "avg_enjoyment" not in master.columns or master["avg_enjoyment"].notna().sum() == 0:
        all_preds = pd.read_csv(OUTPUT_DIR / "ALL_BOOKS_PREDICTIONS.csv")
        master = all_preds.copy()

    master["gr_rating"] = pd.to_numeric(master["gr_rating"], errors="coerce")
    master["amz_rating"] = pd.to_numeric(master["amz_rating"], errors="coerce")
    master["log_gr_count"] = np.log10(pd.to_numeric(master.get("gr_count", pd.Series(dtype=float)), errors="coerce") + 1)
    master["log_amz_count"] = np.log10(pd.to_numeric(master.get("amz_count", pd.Series(dtype=float)), errors="coerce") + 1)

    cat_col = "category" if "category" in master.columns else "dropbox_category"
    master["category"] = master[cat_col].fillna("Unknown")

    master["gr_rating"] = master["gr_rating"].fillna(master["gr_rating"].mean())
    master["amz_rating"] = master["amz_rating"].fillna(master["gr_rating"].mean())
    master["log_gr_count"] = master["log_gr_count"].fillna(master["log_gr_count"].median())
    master["log_amz_count"] = master["log_amz_count"].fillna(master["log_amz_count"].median())

    numerical_cols = ["gr_rating", "amz_rating", "log_gr_count", "log_amz_count"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    ridge_preds: dict[str, pd.Series] = {}
    for target in ["avg_enjoyment", "avg_usefulness"]:
        train_mask = master[target].notna()
        df_train = master[train_mask].copy()

        cat_train = encoder.fit_transform(df_train[["category"]])
        X_train = np.hstack([df_train[numerical_cols].values, cat_train])
        y_train = df_train[target].values

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        unread_df = scores.copy()
        unread_df["gr_rating_num"] = pd.to_numeric(unread_df["goodreads_rating"], errors="coerce")
        unread_df["amz_rating_num"] = pd.to_numeric(unread_df["amazon_rating"], errors="coerce")
        unread_df["log_gr_count"] = np.log10(pd.to_numeric(unread_df["goodreads_rating_count"], errors="coerce") + 1)
        unread_df["log_amz_count"] = np.log10(pd.to_numeric(unread_df["amazon_count"], errors="coerce") + 1)

        shelf_to_cat = {
            "Business, management": "Business, management",
            "General Reading": "General Reading",
            "Computer Science": "Computer Science",
            "Literature": "Literature",
            "fiction": "fiction",
            "Math": "Math",
            "Machine Learning": "Machine Learning",
        }
        unread_df["category"] = unread_df["Bookshelf"].map(shelf_to_cat).fillna("Unknown")

        gr_mean = master.loc[train_mask, "gr_rating"].mean()
        unread_df["gr_rating_num"] = unread_df["gr_rating_num"].fillna(gr_mean)
        unread_df["amz_rating_num"] = unread_df["amz_rating_num"].fillna(gr_mean)
        unread_df["log_gr_count"] = unread_df["log_gr_count"].fillna(master.loc[train_mask, "log_gr_count"].median())
        unread_df["log_amz_count"] = unread_df["log_amz_count"].fillna(master.loc[train_mask, "log_amz_count"].median())

        cat_unread = encoder.transform(unread_df[["category"]])
        X_unread = np.hstack([
            unread_df[["gr_rating_num", "amz_rating_num", "log_gr_count", "log_amz_count"]].values,
            cat_unread,
        ])
        preds = model.predict(X_unread)
        col_name = f"pred_{'enjoy' if 'enjoy' in target else 'useful'}_ridge"
        ridge_preds[col_name] = preds

        features = numerical_cols + list(encoder.get_feature_names_out(["category"]))
        print(f"\n  Ridge {target} coefficients:")
        for feat, coef in zip(features, model.coef_):
            if abs(coef) > 0.01:
                print(f"    {feat:30s} {coef:+.3f}")
        print(f"    {'intercept':30s} {model.intercept_:+.3f}")

    for col, vals in ridge_preds.items():
        scores[col] = vals

    return scores


def print_model_comparison(df: pd.DataFrame, label: str, include_started: bool = False) -> None:
    """Show top books from each model side by side."""
    cats = df["Bookshelf"].isin(TARGET_CATEGORIES)
    if include_started:
        sub = df[cats].copy()
    else:
        sub = df[cats & ~df["has_been_started"]].copy()

    models = {
        "RF (enjoy)": ("pred_enjoy_rf", True),
        "RF (useful)": ("pred_useful_rf", True),
        "GBM (enjoy)": ("pred_enjoy_gbm", True),
        "GBM (useful)": ("pred_useful_gbm", True),
        "Ridge (enjoy)": ("pred_enjoy_ridge", True),
        "Ridge (useful)": ("pred_useful_ridge", True),
    }

    composite_cols = {
        "RF": ("pred_enjoy_rf", "pred_useful_rf"),
        "GBM": ("pred_enjoy_gbm", "pred_useful_gbm"),
        "Ridge": ("pred_enjoy_ridge", "pred_useful_ridge"),
    }
    for name, (ecol, ucol) in composite_cols.items():
        e_z = (sub[ecol] - sub[ecol].mean()) / sub[ecol].std()
        u_z = (sub[ucol] - sub[ucol].mean()) / sub[ucol].std()
        sub[f"composite_{name}"] = e_z + 2.5 * u_z

    started_label = "ALL (started + unstarted)" if include_started else "UNSTARTED ONLY"
    print(f"\n{'='*120}")
    print(f"  TOP 15 BOOKS BY EACH MODEL — {label} [{started_label}]")
    print(f"  (Gen Reading + Business + CS only, n={len(sub)})")
    print(f"{'='*120}")

    for model_name, (ecol, ucol) in composite_cols.items():
        comp_col = f"composite_{model_name}"
        top = sub.nlargest(15, comp_col)
        print(f"\n  ── {model_name}: Top 15 by composite (2.5*useful + enjoy, z-scored) ──")
        hdr_start = "  " if not include_started else "  "
        started_col = f" {'Started':>7}" if include_started else ""
        started_dash = f" {'-'*7}" if include_started else ""
        print(f"  {'Title':<55} {'Cat':<12} {'GR':>4} {'E':>5} {'U':>5} {'Comp':>6}" + started_col)
        print(f"  {'-'*55} {'-'*12} {'-'*4} {'-'*5} {'-'*5} {'-'*6}" + started_dash)
        for _, r in top.iterrows():
            t = str(r["title"])[:53]
            cat = str(r["Bookshelf"])[:10]
            gr = f"{r['goodreads_rating']:.1f}" if pd.notna(r["goodreads_rating"]) else "N/A"
            started_flag = "  yes" if r.get("has_been_started", False) else "   no"
            line = f"  {t:<55} {cat:<12} {gr:>4} {r[ecol]:5.2f} {r[ucol]:5.2f} {r[comp_col]:+6.2f}"
            if include_started:
                line += f" {started_flag:>7}"
            print(line)

    # Model agreement analysis
    print(f"\n{'='*120}")
    print(f"  MODEL AGREEMENT ANALYSIS — {label} [{started_label}]")
    print(f"{'='*120}")

    for comp_name in composite_cols:
        sub[f"rank_{comp_name}"] = sub[f"composite_{comp_name}"].rank(ascending=False)

    sub["avg_rank"] = sub[[f"rank_{m}" for m in composite_cols]].mean(axis=1)
    sub["rank_spread"] = sub[[f"rank_{m}" for m in composite_cols]].max(axis=1) - sub[[f"rank_{m}" for m in composite_cols]].min(axis=1)

    started_hdr = f" {'Strt':>4}" if include_started else ""
    started_sep = f" {'-'*4}" if include_started else ""

    print("\n  ── ALL 3 MODELS AGREE: Top 15 by average rank ──")
    agree = sub.nsmallest(15, "avg_rank")
    print(f"  {'Title':<50} {'Cat':<12} {'GR':>4} {'RF':>4} {'GBM':>4} {'Rdg':>4} {'Spread':>6}" + started_hdr)
    print(f"  {'-'*50} {'-'*12} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6}" + started_sep)
    for _, r in agree.iterrows():
        t = str(r["title"])[:48]
        cat = str(r["Bookshelf"])[:10]
        gr = f"{r['goodreads_rating']:.1f}" if pd.notna(r["goodreads_rating"]) else "N/A"
        line = f"  {t:<50} {cat:<12} {gr:>4} {r['rank_RF']:4.0f} {r['rank_GBM']:4.0f} {r['rank_Ridge']:4.0f} {r['rank_spread']:6.0f}"
        if include_started:
            line += f" {'  Y' if r.get('has_been_started', False) else '  N':>4}"
        print(line)

    print("\n  ── MODELS DISAGREE MOST: Top 15 by rank spread ──")
    print("  (These are the most informative books for model validation)")
    disagree = sub.nlargest(15, "rank_spread")
    print(f"  {'Title':<50} {'Cat':<12} {'GR':>4} {'RF':>4} {'GBM':>4} {'Rdg':>4} {'Spread':>6}" + started_hdr)
    print(f"  {'-'*50} {'-'*12} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6}" + started_sep)
    for _, r in disagree.iterrows():
        t = str(r["title"])[:48]
        cat = str(r["Bookshelf"])[:10]
        gr = f"{r['goodreads_rating']:.1f}" if pd.notna(r["goodreads_rating"]) else "N/A"
        line = f"  {t:<50} {cat:<12} {gr:>4} {r['rank_RF']:4.0f} {r['rank_GBM']:4.0f} {r['rank_Ridge']:4.0f} {r['rank_spread']:6.0f}"
        if include_started:
            line += f" {'  Y' if r.get('has_been_started', False) else '  N':>4}"
        print(line)

    return sub


def design_validation_set(df: pd.DataFrame, n_target: int = 10, include_started: bool = False) -> pd.DataFrame:
    """
    Design a validation set that TESTS the model, not just uses it.

    A pure "pick top N" strategy has low statistical power for validation because:
    - All books cluster near the same high prediction → small variance → weak test
    - You can't distinguish "model is good" from "these categories are just good"

    Better: stratified sampling across prediction quintiles within target categories,
    with oversampling of model-disagreement books.
    """
    cats = df["Bookshelf"].isin(TARGET_CATEGORIES)
    if include_started:
        sub = df[cats].copy()
    else:
        sub = df[cats & ~df["has_been_started"]].copy()

    composite_cols = {
        "RF": ("pred_enjoy_rf", "pred_useful_rf"),
        "GBM": ("pred_enjoy_gbm", "pred_useful_gbm"),
        "Ridge": ("pred_enjoy_ridge", "pred_useful_ridge"),
    }
    for name, (ecol, ucol) in composite_cols.items():
        e_z = (sub[ecol] - sub[ecol].mean()) / sub[ecol].std()
        u_z = (sub[ucol] - sub[ucol].mean()) / sub[ucol].std()
        sub[f"composite_{name}"] = e_z + 2.5 * u_z
        sub[f"rank_{name}"] = sub[f"composite_{name}"].rank(ascending=False)

    sub["avg_rank"] = sub[[f"rank_{m}" for m in composite_cols]].mean(axis=1)
    sub["rank_spread"] = (
        sub[[f"rank_{m}" for m in composite_cols]].max(axis=1)
        - sub[[f"rank_{m}" for m in composite_cols]].min(axis=1)
    )

    sub["avg_composite"] = sub[[f"composite_{m}" for m in composite_cols]].mean(axis=1)
    sub["pred_quintile"] = pd.qcut(sub["avg_composite"], 5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])

    print(f"\n{'='*120}")
    print(f"  OPTIMAL VALIDATION SET DESIGN")
    print(f"{'='*120}")
    print(f"""
  WHY NOT JUST PICK THE TOP {n_target}?
  If you only read top-ranked books, you learn:
    - "The books the algorithm liked are pretty good" (expected, not very informative)
    - Nothing about false positives (books ranked high that you'd hate)
    - Nothing about false negatives (books ranked low that you'd love)
    - Nothing about calibration (does pred=3.5 really mean 3.5?)

  BETTER: A stratified sample that tests the prediction across its range.
  
  RECOMMENDED ALLOCATION for {n_target} books:
    - 4 from Q5 (top 20%) — confirm the algorithm picks winners
    - 2 from Q4 (60-80%) — test the boundary  
    - 2 from Q3 (40-60%) — calibration check
    - 1 from Q2 (20-40%) — catch false negatives
    - 1 from Q1 (bottom 20%) — expected to be bad; confirms ranking works
    
  PLUS: Prioritize books where models DISAGREE within each stratum.
  Those are the most informative for determining which model to trust.
""")

    allocations = {
        "Q5 (high)": max(1, round(n_target * 0.40)),
        "Q4": max(1, round(n_target * 0.20)),
        "Q3": max(1, round(n_target * 0.20)),
        "Q2": max(1, round(n_target * 0.10)),
        "Q1 (low)": max(1, round(n_target * 0.10)),
    }
    total_alloc = sum(allocations.values())
    if total_alloc < n_target:
        allocations["Q5 (high)"] += n_target - total_alloc
    elif total_alloc > n_target:
        allocations["Q5 (high)"] -= total_alloc - n_target

    selected: list[pd.DataFrame] = []

    for quintile, n_pick in allocations.items():
        q_books = sub[sub["pred_quintile"] == quintile].copy()
        if len(q_books) == 0:
            continue

        q_books["selection_priority"] = (
            0.5 * q_books["rank_spread"].rank(pct=True)
            + 0.3 * (1 - q_books["avg_rank"].rank(pct=True))
            + 0.2 * q_books["goodreads_rating"].fillna(0).rank(pct=True)
        )

        picks = q_books.nlargest(min(n_pick, len(q_books)), "selection_priority")
        picks["validation_quintile"] = quintile
        picks["validation_reason"] = np.where(
            picks["rank_spread"] > picks["rank_spread"].median(),
            "model disagreement",
            "representative",
        )
        selected.append(picks)

    validation = pd.concat(selected, ignore_index=True)

    started_label = "ALL (started + unstarted)" if include_started else "UNSTARTED ONLY"
    print(f"  VALIDATION SET ({len(validation)} books) [{started_label}]:")
    strt_hdr = f" {'S':>1}" if include_started else ""
    strt_sep = f" {'-'*1}" if include_started else ""
    print(f"  {'Quintile':<12} {'Title':<50} {'Cat':<12} {'GR':>4} "
          f"{'E(RF)':>6} {'U(RF)':>6} {'E(Rdg)':>7} {'U(Rdg)':>7} {'Spread':>6} {'Reason':<18}" + strt_hdr)
    print(f"  {'-'*12} {'-'*50} {'-'*12} {'-'*4} "
          f"{'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*18}" + strt_sep)
    for _, r in validation.sort_values("avg_composite", ascending=False).iterrows():
        t = str(r["title"])[:48]
        cat = str(r["Bookshelf"])[:10]
        gr = f"{r['goodreads_rating']:.1f}" if pd.notna(r["goodreads_rating"]) else "N/A"
        line = (f"  {r['validation_quintile']:<12} {t:<50} {cat:<12} {gr:>4} "
              f"{r['pred_enjoy_rf']:6.2f} {r['pred_useful_rf']:6.2f} "
              f"{r['pred_enjoy_ridge']:7.2f} {r['pred_useful_ridge']:7.2f} "
              f"{r['rank_spread']:6.0f} {r['validation_reason']:<18}")
        if include_started:
            line += f" {'Y' if r.get('has_been_started', False) else 'N':>1}"
        print(line)

    return validation


def print_diagnostics_checklist() -> None:
    """Print what else to check for forecast accuracy."""
    print(f"""
{'='*120}
  ADDITIONAL DIAGNOSTICS TO VERIFY FORECAST ACCURACY
{'='*120}

  ── BEFORE READING ──────────────────────────────────────────────────────────

  1. PREDICTION INTERVAL CHECK
     Your models predict point estimates, but the real question is the
     prediction interval. For Ridge with RMSE=0.66 (enjoyment):
       pred=3.5 → 80% PI: [2.66, 4.34]
       pred=4.0 → 80% PI: [3.16, 4.84]
     Most of the 1-5 scale is covered. Compare to the interval you'd get
     from just the category mean (even wider). The model barely narrows it.

  2. CATEGORY LEAKAGE TEST
     The RF and GBM are category + Goodreads. If you only pick from
     Gen Reading + Business, the model is mostly a Goodreads filter within
     those categories. Check: do the predictions vary within a single
     category, or are they just reflecting Goodreads rating?
       → Compute rank correlation of pred_enjoy_rf with goodreads_rating
         WITHIN each category. If rho > 0.9, the model is basically
         just Goodreads with extra steps.

  3. PREDICTION VARIANCE CHECK
     RF predictions on unread books have very low variance (they cluster
     around 3.0-3.5 for enjoyment). This is a sign of the model not
     discriminating much. Compare:
       - Prediction std of RF on unread books vs on training books
       - If unread std < 0.5 * train std, the model is "shrinking to the mean"
         on out-of-distribution data

  4. HOLDOUT CALIBRATION PLOT
     You have 68 holdout books with actual ratings. Plot:
       - predicted vs actual (should follow y=x line)
       - residuals vs predicted (should be flat, no trend)
       - residuals vs Goodreads rating (should be flat)
       - residuals vs category (should be flat)
     Any trend means the model is miscalibrated in that region.

  ── WHILE READING (10-book validation set) ─────────────────────────────────

  5. SEQUENTIAL UPDATING
     After each book, compute running:
       - Mean actual vs mean predicted
       - Cumulative Spearman rho (needs >= 5 books)
       - Bayesian posterior on "true rho" given observations so far
     Stop early if the 95% CI on rho includes 0 after 7+ books.

  6. PAIRED COMPARISON
     For each book, record:
       - Actual enjoyment & usefulness
       - Which model predicted best (smallest |actual - pred|)
       - Whether the book was above/below the category historical mean
     After 10 books: binomial test on "fraction above category mean" vs 0.50

  7. CALIBRATION BIN TEST
     Group your 10 validation books by prediction level:
       - pred >= 3.5: expect actual mean ~3.5
       - pred 3.0-3.5: expect actual mean ~3.25
       - pred < 3.0: expect actual mean < 3.0
     If predictions systematically over- or under-predict, adjust.

  ── AFTER READING ──────────────────────────────────────────────────────────

  8. MODEL HORSE RACE
     With 10 books rated, compute for each model:
       - MAE, RMSE, Spearman rho
       - "Did the model beat category mean baseline?"
     The winner becomes your go-to model for the remaining catalog.

  9. OUT-OF-SAMPLE R² DECOMPOSITION
     How much of your model's predictive power comes from:
       (a) Category selection (Gen Reading/Business > overall)
       (b) Goodreads rating (within-category signal)
       (c) Amazon/OL/log_count (incremental signal)
     Compute partial R² by adding features sequentially.

  10. COMPARE TO TRIVIAL BASELINES
      - "Read the highest Goodreads-rated book" (no model needed)
      - "Read books friends recommended" (Gemini-inferred source)
      - "Read books in the category with highest historical mean"
      If the model doesn't beat these, it's not worth the complexity.

  11. CHECK FOR DATA LEAKAGE IN UNREAD PREDICTIONS
      Some unread books may have been in the training set under a different
      title/edition. Cross-check the unread list against training titles.
      Look for suspiciously high predictions (> 4.0) on books with low GR.
""")


def main() -> None:
    print("Loading scores and fitting Ridge model for unread books...")
    scores = load_scores_with_ridge()

    # ── Unstarted-only analysis (original) ──
    print("\n\n" + "#" * 120)
    print("#  SECTION A: UNSTARTED BOOKS ONLY")
    print("#" * 120)
    sub_unstarted = print_model_comparison(scores, "Gen Reading + Business + CS", include_started=False)
    validation_10 = design_validation_set(sub_unstarted, n_target=10, include_started=False)

    # ── Including started books ──
    print("\n\n" + "#" * 120)
    print("#  SECTION B: INCLUDING STARTED BOOKS")
    print("#" * 120)
    sub_all_books = print_model_comparison(scores, "Gen Reading + Business + CS", include_started=True)
    validation_10_with_started = design_validation_set(sub_all_books, n_target=10, include_started=True)
    validation_15_with_started = design_validation_set(sub_all_books, n_target=15, include_started=True)
    validation_20_with_started = design_validation_set(sub_all_books, n_target=20, include_started=True)

    # Run the category leakage check (diagnostic #2)
    from scipy import stats as sp_stats

    cats = scores["Bookshelf"].isin(TARGET_CATEGORIES)
    sub_all = scores[cats].copy()

    print(f"\n{'='*120}")
    print(f"  DIAGNOSTIC: WITHIN-CATEGORY CORRELATION OF RF PRED VS GOODREADS")
    print(f"{'='*120}")
    for cat in TARGET_CATEGORIES:
        cat_df = sub_all[sub_all["Bookshelf"] == cat]
        has_gr = cat_df["goodreads_rating"].notna()
        if has_gr.sum() > 5:
            rho_e, p_e = sp_stats.spearmanr(
                cat_df.loc[has_gr, "pred_enjoy_rf"],
                cat_df.loc[has_gr, "goodreads_rating"],
            )
            rho_u, p_u = sp_stats.spearmanr(
                cat_df.loc[has_gr, "pred_useful_rf"],
                cat_df.loc[has_gr, "goodreads_rating"],
            )
            print(f"  {cat:<25} (n={has_gr.sum():3d}): "
                  f"enjoy rho={rho_e:.3f} (p={p_e:.3f}), "
                  f"useful rho={rho_u:.3f} (p={p_u:.3f})")

    # Prediction variance check (diagnostic #3)
    print(f"\n{'='*120}")
    print(f"  DIAGNOSTIC: PREDICTION VARIANCE (SHRINKAGE CHECK)")
    print(f"{'='*120}")
    for col in ["pred_enjoy_rf", "pred_useful_rf", "pred_enjoy_ridge", "pred_useful_ridge"]:
        unread_std = sub_all[col].std()
        print(f"  {col:<25} unread std = {unread_std:.3f}")

    print_diagnostics_checklist()

    # Save validation sets
    out_cols = [
        "title", "Bookshelf", "has_been_started", "goodreads_rating",
        "pred_enjoy_rf", "pred_useful_rf",
        "pred_enjoy_gbm", "pred_useful_gbm",
        "pred_enjoy_ridge", "pred_useful_ridge",
        "validation_quintile", "validation_reason",
    ]

    for n, vdf in [(10, validation_10)]:
        existing = [c for c in out_cols if c in vdf.columns]
        vdf[existing].to_csv(
            OUTPUT_DIR / "ai_actions" / f"validation_book_selection_{n}_unstarted.csv",
            index=False,
        )

    for n, vdf in [
        (10, validation_10_with_started),
        (15, validation_15_with_started),
        (20, validation_20_with_started),
    ]:
        existing = [c for c in out_cols if c in vdf.columns]
        vdf[existing].to_csv(
            OUTPUT_DIR / "ai_actions" / f"validation_book_selection_{n}_with_started.csv",
            index=False,
        )
    print(f"\nSaved validation CSVs to ai_actions/validation_book_selection_*")


if __name__ == "__main__":
    main()
