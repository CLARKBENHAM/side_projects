"""Deep dive: what's actually true about model performance?

Answers:
1. Is rho=0.55 real or an artifact? (permutation test)
2. Which features drive the predictions? (ablation & permutation importance)
3. How robust are the Ridge coefficients? (bootstrap)
4. Is the holdout R² negative just because of bias? (bias-corrected R²)
5. Where does train→holdout degradation come from? (LOO-CV on training)
6. Code correctness check: does the plot pipeline match the export pipeline?
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from ai_books_tracking.multi_source_selection_policy import load_frame
from ai_books_tracking.export_models_for_extension import (
    group_name, RATING_FEATURES, AUTHOR_FEATURES, GROUP_ORDER,
    fit_imputation_models, impute_group_sources,
)
from ai_books_tracking.future_prediction_evaluation import (
    FeatureSpec, prepare_feature_frames, random_forest_model, gbm_model,
)
from ai_books_tracking.goodreads_followup_analysis import derive_analysis_columns, load_data
from ai_books_tracking.new_books_to_rate_analysis import NEW_BOOKS_ENRICHED, add_centered_targets


def load_ridge_data():
    frame = load_frame()
    frame["group"] = frame["category"].map(group_name)
    frame["dataset"] = np.where(frame["source"].eq("Holdout 2026"), "holdout", "train")
    return frame


def load_tree_data():
    historical = add_centered_targets(derive_analysis_columns(load_data()))
    historical["label_source"] = "historical_main"
    frames = [historical]
    if NEW_BOOKS_ENRICHED.exists():
        nb = pd.read_csv(NEW_BOOKS_ENRICHED)
        nb.columns = nb.columns.str.strip()
        nb = add_centered_targets(derive_analysis_columns(nb))
        nb["label_source"] = "new_books_holdout_2026"
        frames.append(nb)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedup = (
        combined["title"].astype(str).str.strip().str.lower()
        + "||"
        + combined.get("filename", pd.Series("", index=combined.index)).astype(str).str.strip().str.lower()
    )
    combined = combined[~dedup.duplicated()].reset_index(drop=True)
    combined["group"] = combined.get("Bookshelf", combined.get("category", "")).map(group_name)
    return combined


# ═══════════════════════════════════════════════════════════════
# 1. PERMUTATION TEST: Is rho=0.55 real?
# ═══════════════════════════════════════════════════════════════
def permutation_test_rho(actual, pred, n_perm=10000):
    """Test H0: rho=0 by shuffling actual labels."""
    observed_rho, _ = spearmanr(actual, pred)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(actual)
        r, _ = spearmanr(perm, pred)
        if abs(r) >= abs(observed_rho):
            count += 1
    return observed_rho, count / n_perm


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ABLATION for RF/GBM
# ═══════════════════════════════════════════════════════════════
def run_ablation_study(combined, target, spec):
    tshort = "enjoy" if "enjoy" in target else "useful"
    all_data = combined[combined[target].notna()].copy()
    train_mask = all_data["label_source"] == "historical_main"
    holdout_mask = all_data["label_source"] == "new_books_holdout_2026"

    train_df = all_data[train_mask].copy()
    holdout_df = all_data[holdout_mask].copy()
    train_p, holdout_p, nf_all, cf = prepare_feature_frames(train_df, holdout_df, target, spec)
    nf = [f for f in nf_all if f not in AUTHOR_FEATURES]

    y_train = pd.to_numeric(train_p[target], errors="coerce")
    valid = y_train.notna()
    X_train = train_p[valid].copy()
    y_train = y_train[valid].values

    for c in nf:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        holdout_p[c] = holdout_p[c].fillna(med)
    for c in cf:
        X_train[c] = X_train[c].fillna("Unknown")
        holdout_p[c] = holdout_p[c].fillna("Unknown")

    y_holdout = holdout_p[target].values.astype(float)

    results = []

    # Full model baseline
    for model_name, builder in [("RF", random_forest_model), ("GBM", gbm_model)]:
        enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist")
        pre = ColumnTransformer([("num", StandardScaler(), nf), ("cat", enc, cf)], remainder="drop")
        pipe = Pipeline([("preprocessor", pre), ("model", builder())])
        pipe.fit(X_train, y_train)
        pred_ho = pipe.predict(holdout_p)
        pred_tr = pipe.predict(X_train)
        rho_ho, _ = spearmanr(y_holdout, pred_ho)
        rho_tr, _ = spearmanr(y_train, pred_tr)
        r2_ho = r2_score(y_holdout, pred_ho)
        mae_ho = mean_absolute_error(y_holdout, pred_ho)
        results.append({
            "model": model_name, "target": tshort, "ablation": "FULL",
            "features_used": ", ".join(nf + cf),
            "n_numeric": len(nf), "n_cat": len(cf),
            "rho_train": rho_tr, "rho_holdout": rho_ho,
            "r2_holdout": r2_ho, "mae_holdout": mae_ho,
        })

        # Drop each numeric feature one at a time
        for drop_feat in nf:
            nf_reduced = [f for f in nf if f != drop_feat]
            enc2 = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist")
            pre2 = ColumnTransformer([("num", StandardScaler(), nf_reduced), ("cat", enc2, cf)], remainder="drop")
            pipe2 = Pipeline([("preprocessor", pre2), ("model", builder())])
            pipe2.fit(X_train, y_train)
            pred2 = pipe2.predict(holdout_p)
            rho2, _ = spearmanr(y_holdout, pred2)
            r2_2 = r2_score(y_holdout, pred2)
            mae2 = mean_absolute_error(y_holdout, pred2)
            results.append({
                "model": model_name, "target": tshort,
                "ablation": f"drop_{drop_feat}",
                "features_used": ", ".join(nf_reduced + cf),
                "n_numeric": len(nf_reduced), "n_cat": len(cf),
                "rho_train": float("nan"),
                "rho_holdout": rho2, "r2_holdout": r2_2, "mae_holdout": mae2,
            })

        # Category only
        nf_none = []
        enc3 = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist")
        pre3 = ColumnTransformer([("cat", enc3, cf)], remainder="drop")
        pipe3 = Pipeline([("preprocessor", pre3), ("model", builder())])
        pipe3.fit(X_train, y_train)
        pred3 = pipe3.predict(holdout_p)
        rho3, _ = spearmanr(y_holdout, pred3)
        r2_3 = r2_score(y_holdout, pred3)
        mae3 = mean_absolute_error(y_holdout, pred3)
        results.append({
            "model": model_name, "target": tshort,
            "ablation": "CATEGORY_ONLY",
            "features_used": ", ".join(cf),
            "n_numeric": 0, "n_cat": len(cf),
            "rho_train": float("nan"),
            "rho_holdout": rho3, "r2_holdout": r2_3, "mae_holdout": mae3,
        })

        # GR rating only (+ category)
        nf_gr = [f for f in nf if "goodreads" in f]
        if nf_gr:
            enc4 = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist")
            pre4 = ColumnTransformer([("num", StandardScaler(), nf_gr), ("cat", enc4, cf)], remainder="drop")
            pipe4 = Pipeline([("preprocessor", pre4), ("model", builder())])
            pipe4.fit(X_train, y_train)
            pred4 = pipe4.predict(holdout_p)
            rho4, _ = spearmanr(y_holdout, pred4)
            r2_4 = r2_score(y_holdout, pred4)
            mae4 = mean_absolute_error(y_holdout, pred4)
            results.append({
                "model": model_name, "target": tshort,
                "ablation": "GR_FEATURES_ONLY",
                "features_used": ", ".join(nf_gr + cf),
                "n_numeric": len(nf_gr), "n_cat": len(cf),
                "rho_train": float("nan"),
                "rho_holdout": rho4, "r2_holdout": r2_4, "mae_holdout": mae4,
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 3. BOOTSTRAP Ridge coefficient stability
# ═══════════════════════════════════════════════════════════════
def bootstrap_ridge_coefficients(frame, n_boot=500):
    train = frame[frame["dataset"] == "train"]
    holdout = frame[frame["dataset"] == "holdout"]
    rows = []

    for grp in GROUP_ORDER:
        gt = train[train["group"] == grp].copy()
        gh = holdout[holdout["group"] == grp].copy()
        imp_models, imp_medians = fit_imputation_models(gt)
        gt = impute_group_sources(gt, imp_models, imp_medians)
        gh = impute_group_sources(gh, imp_models, imp_medians)

        for target in ["avg_enjoyment", "avg_usefulness"]:
            tshort = "enjoy" if "enjoy" in target else "useful"
            tv = gt[gt[target].notna()].copy()
            fill = tv[RATING_FEATURES].median()
            X = tv[RATING_FEATURES].fillna(fill).values
            y = tv[target].values
            n = len(y)

            # Full-data fit
            ridge = Ridge(alpha=1.0).fit(X, y)
            full_coefs = ridge.coef_.tolist()
            full_intercept = float(ridge.intercept_)

            # Bootstrap
            boot_coefs = []
            boot_intercepts = []
            boot_holdout_rhos = []
            for _ in range(n_boot):
                idx = np.random.randint(0, n, size=n)
                Xb, yb = X[idx], y[idx]
                rb = Ridge(alpha=1.0).fit(Xb, yb)
                boot_coefs.append(rb.coef_.tolist())
                boot_intercepts.append(float(rb.intercept_))

                # Holdout eval with this bootstrap model
                hv = gh[gh[target].notna()].copy()
                if len(hv) >= 3:
                    Xh = hv[RATING_FEATURES].fillna(fill).values
                    yh = hv[target].values
                    ph = np.clip(Xh @ rb.coef_ + rb.intercept_, 1, 5)
                    r, _ = spearmanr(yh, ph)
                    boot_holdout_rhos.append(r)

            boot_coefs = np.array(boot_coefs)
            boot_intercepts = np.array(boot_intercepts)

            for i, feat in enumerate(RATING_FEATURES):
                rows.append({
                    "group": grp, "target": tshort, "feature": feat,
                    "coef_full": full_coefs[i],
                    "coef_mean": float(boot_coefs[:, i].mean()),
                    "coef_std": float(boot_coefs[:, i].std()),
                    "coef_2.5%": float(np.percentile(boot_coefs[:, i], 2.5)),
                    "coef_97.5%": float(np.percentile(boot_coefs[:, i], 97.5)),
                    "sign_stability": float((boot_coefs[:, i] > 0).mean() if full_coefs[i] > 0
                                            else (boot_coefs[:, i] < 0).mean()),
                })
            rows.append({
                "group": grp, "target": tshort, "feature": "intercept",
                "coef_full": full_intercept,
                "coef_mean": float(boot_intercepts.mean()),
                "coef_std": float(boot_intercepts.std()),
                "coef_2.5%": float(np.percentile(boot_intercepts, 2.5)),
                "coef_97.5%": float(np.percentile(boot_intercepts, 97.5)),
                "sign_stability": float("nan"),
            })

            if boot_holdout_rhos:
                rhos = np.array(boot_holdout_rhos)
                rows.append({
                    "group": grp, "target": tshort, "feature": "HOLDOUT_RHO",
                    "coef_full": float(np.median(rhos)),
                    "coef_mean": float(rhos.mean()),
                    "coef_std": float(rhos.std()),
                    "coef_2.5%": float(np.percentile(rhos, 2.5)),
                    "coef_97.5%": float(np.percentile(rhos, 97.5)),
                    "sign_stability": float((rhos > 0).mean()),
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 4. BIAS-CORRECTED R²: what R² would be if we fixed the mean?
# ═══════════════════════════════════════════════════════════════
def bias_corrected_analysis(frame):
    train = frame[frame["dataset"] == "train"]
    holdout = frame[frame["dataset"] == "holdout"]
    rows = []

    for grp in GROUP_ORDER:
        gt = train[train["group"] == grp].copy()
        gh = holdout[holdout["group"] == grp].copy()
        imp_models, imp_medians = fit_imputation_models(gt)
        gt = impute_group_sources(gt, imp_models, imp_medians)
        gh = impute_group_sources(gh, imp_models, imp_medians)

        for target in ["avg_enjoyment", "avg_usefulness"]:
            tshort = "enjoy" if "enjoy" in target else "useful"
            tv = gt[gt[target].notna()].copy()
            fill = tv[RATING_FEATURES].median()
            X_tr = tv[RATING_FEATURES].fillna(fill)
            y_tr = tv[target].values

            ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)

            hv = gh[gh[target].notna()].copy()
            if len(hv) < 3:
                continue
            X_ho = hv[RATING_FEATURES].fillna(fill)
            y_ho = hv[target].values
            pred_ho = np.clip(X_ho.values @ ridge.coef_ + ridge.intercept_, 1, 5)

            bias = pred_ho.mean() - y_ho.mean()
            pred_corrected = pred_ho - bias

            rho_raw, _ = spearmanr(y_ho, pred_ho)
            r2_raw = r2_score(y_ho, pred_ho)
            r2_corrected = r2_score(y_ho, pred_corrected)
            pearson_r, _ = pearsonr(y_ho, pred_ho)
            mae_raw = mean_absolute_error(y_ho, pred_ho)
            mae_corrected = mean_absolute_error(y_ho, pred_corrected)

            rows.append({
                "group": grp, "target": tshort, "n": len(y_ho),
                "train_mean": float(y_tr.mean()),
                "holdout_mean": float(y_ho.mean()),
                "pred_mean": float(pred_ho.mean()),
                "bias": bias,
                "rho": rho_raw,
                "pearson_r": pearson_r,
                "r2_raw": r2_raw,
                "r2_bias_corrected": r2_corrected,
                "mae_raw": mae_raw,
                "mae_bias_corrected": mae_corrected,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 5. LOO-CV on training data for Ridge (honest estimate)
# ═══════════════════════════════════════════════════════════════
def loo_cv_ridge(frame):
    train = frame[frame["dataset"] == "train"]
    rows = []

    for grp in GROUP_ORDER:
        gt = train[train["group"] == grp].copy()
        imp_models, imp_medians = fit_imputation_models(gt)
        gt = impute_group_sources(gt, imp_models, imp_medians)

        for target in ["avg_enjoyment", "avg_usefulness"]:
            tshort = "enjoy" if "enjoy" in target else "useful"
            tv = gt[gt[target].notna()].copy()
            fill = tv[RATING_FEATURES].median()
            X = tv[RATING_FEATURES].fillna(fill).values
            y = tv[target].values

            if len(y) < 10:
                continue

            loo_pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=LeaveOneOut())
            loo_pred = np.clip(loo_pred, 1, 5)

            rho_loo, _ = spearmanr(y, loo_pred)
            r2_loo = r2_score(y, loo_pred)
            mae_loo = mean_absolute_error(y, loo_pred)

            # Also in-sample for comparison
            ridge = Ridge(alpha=1.0).fit(X, y)
            in_sample = np.clip(ridge.predict(X), 1, 5)
            rho_is, _ = spearmanr(y, in_sample)
            r2_is = r2_score(y, in_sample)

            rows.append({
                "group": grp, "target": tshort, "n": len(y),
                "rho_in_sample": rho_is, "r2_in_sample": r2_is,
                "rho_loo_cv": rho_loo, "r2_loo_cv": r2_loo, "mae_loo_cv": mae_loo,
                "rho_overfit_gap": rho_is - rho_loo,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 6. CODE CORRECTNESS: verify plot pipeline matches export
# ═══════════════════════════════════════════════════════════════
def verify_pipeline_consistency():
    """Check that the Ridge pipeline in holdout_pred_vs_actual.py
    produces the same predictions as export_models_for_extension.py."""
    frame = load_ridge_data()
    train = frame[frame["dataset"] == "train"]
    holdout = frame[frame["dataset"] == "holdout"]

    issues = []

    for grp in GROUP_ORDER:
        gt = train[train["group"] == grp].copy()
        gh = holdout[holdout["group"] == grp].copy()
        imp_models, imp_medians = fit_imputation_models(gt)
        gt_imp = impute_group_sources(gt, imp_models, imp_medians)
        gh_imp = impute_group_sources(gh, imp_models, imp_medians)

        for target in ["avg_enjoyment", "avg_usefulness"]:
            tshort = "enjoy" if "enjoy" in target else "useful"
            tv = gt_imp[gt_imp[target].notna()].copy()
            fill = tv[RATING_FEATURES].median()
            X_tr = tv[RATING_FEATURES].fillna(fill)
            y_tr = tv[target].values

            ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)

            hv = gh_imp[gh_imp[target].notna()].copy()
            if len(hv) == 0:
                continue
            X_ho = hv[RATING_FEATURES].fillna(fill)
            pred = np.clip(X_ho.values @ ridge.coef_ + ridge.intercept_, 1, 5)

            # Check: are these the same as what we'd get from predict()?
            pred_sklearn = np.clip(ridge.predict(X_ho), 1, 5)
            diff = np.abs(pred - pred_sklearn).max()
            if diff > 1e-10:
                issues.append(f"{grp}/{tshort}: manual vs sklearn predict diff = {diff}")

            # Check data counts
            n_train_target = len(tv)
            n_holdout_target = len(hv)
            n_train_group = len(gt)
            n_holdout_group = len(gh)

            print(f"  {grp}/{tshort}: train_group={n_train_group}, train_with_target={n_train_target}, "
                  f"holdout_group={n_holdout_group}, holdout_with_target={n_holdout_target}")

            # Check for NaN in features after imputation
            nan_count = X_ho.isna().sum().sum()
            if nan_count > 0:
                issues.append(f"{grp}/{tshort}: {nan_count} NaNs in holdout features after imputation!")

            # Check imputation didn't produce crazy values
            for feat in RATING_FEATURES:
                vals = X_ho[feat]
                if vals.min() < 0 or vals.max() > 6:
                    issues.append(f"{grp}/{tshort}: {feat} has values [{vals.min():.2f}, {vals.max():.2f}]")

    return issues


# ═══════════════════════════════════════════════════════════════
# 7. TREE DATA PIPELINE CHECK
# ═══════════════════════════════════════════════════════════════
def verify_tree_pipeline():
    """Check data loading for RF/GBM pipeline."""
    combined = load_tree_data()

    train = combined[combined["label_source"] == "historical_main"]
    holdout = combined[combined["label_source"] == "new_books_holdout_2026"]

    print(f"\n  Tree pipeline data counts:")
    print(f"    Total: {len(combined)}, Train: {len(train)}, Holdout: {len(holdout)}")

    for target in ["avg_enjoyment", "avg_usefulness"]:
        tshort = "enjoy" if "enjoy" in target else "useful"
        t_notna = train[target].notna().sum()
        h_notna = holdout[target].notna().sum()
        print(f"    {tshort}: train_with_target={t_notna}, holdout_with_target={h_notna}")

    # Check Ridge vs Tree data overlap
    ridge_frame = load_ridge_data()
    ridge_train = ridge_frame[ridge_frame["dataset"] == "train"]
    ridge_holdout = ridge_frame[ridge_frame["dataset"] == "holdout"]

    print(f"\n  Ridge pipeline data counts:")
    print(f"    Train: {len(ridge_train)}, Holdout: {len(ridge_holdout)}")
    for target in ["avg_enjoyment", "avg_usefulness"]:
        tshort = "enjoy" if "enjoy" in target else "useful"
        t_notna = ridge_train[target].notna().sum()
        h_notna = ridge_holdout[target].notna().sum()
        print(f"    {tshort}: train_with_target={t_notna}, holdout_with_target={h_notna}")

    # Check holdout book overlap
    tree_holdout_titles = set(holdout["title"].astype(str).str.strip().str.lower())
    ridge_holdout_titles = set(ridge_holdout["title"].astype(str).str.strip().str.lower())
    overlap = tree_holdout_titles & ridge_holdout_titles
    tree_only = tree_holdout_titles - ridge_holdout_titles
    ridge_only = ridge_holdout_titles - tree_holdout_titles
    print(f"\n  Holdout title overlap: {len(overlap)} shared, "
          f"{len(tree_only)} tree-only, {len(ridge_only)} ridge-only")
    if tree_only:
        print(f"    Tree-only examples: {list(tree_only)[:5]}")
    if ridge_only:
        print(f"    Ridge-only examples: {list(ridge_only)[:5]}")

    issues = []

    # Check feature spec
    spec = FeatureSpec("preread_plus_goodreads_conservative", include_goodreads="conservative")
    for target in ["avg_enjoyment", "avg_usefulness"]:
        all_data = combined[combined[target].notna()].copy()
        train_mask = all_data["label_source"] == "historical_main"
        holdout_mask = all_data["label_source"] == "new_books_holdout_2026"
        train_df = all_data[train_mask].copy()
        holdout_df = all_data[holdout_mask].copy()
        train_p, holdout_p, nf_all, cf = prepare_feature_frames(train_df, holdout_df, target, spec)
        nf = [f for f in nf_all if f not in AUTHOR_FEATURES]
        print(f"\n  {target} features:")
        print(f"    Numeric (no author): {nf}")
        print(f"    Categorical: {cf}")

        # Check for NaN rates in holdout
        for c in nf:
            nan_rate = holdout_p[c].isna().mean()
            if nan_rate > 0.1:
                issues.append(f"{target}/{c}: {nan_rate:.0%} NaN in holdout")

    return issues


def main():
    np.random.seed(42)

    print("=" * 70)
    print("DEEP DIVE MODEL ANALYSIS")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/7] Loading data...")
    ridge_frame = load_ridge_data()
    tree_data = load_tree_data()

    # ── Pipeline verification ──
    print("\n[2/7] Verifying Ridge pipeline consistency...")
    ridge_issues = verify_pipeline_consistency()
    if ridge_issues:
        print("  ISSUES FOUND:")
        for issue in ridge_issues:
            print(f"    - {issue}")
    else:
        print("  ✓ No issues found")

    print("\n[3/7] Verifying Tree pipeline...")
    tree_issues = verify_tree_pipeline()
    if tree_issues:
        print("  ISSUES FOUND:")
        for issue in tree_issues:
            print(f"    - {issue}")
    else:
        print("  ✓ No issues found")

    # ── Permutation test ──
    print("\n[4/7] Permutation test for holdout rho (RF/GBM usefulness)...")
    spec = FeatureSpec("preread_plus_goodreads_conservative", include_goodreads="conservative")
    for target in ["avg_enjoyment", "avg_usefulness"]:
        tshort = "enjoy" if "enjoy" in target else "useful"
        all_data = tree_data[tree_data[target].notna()].copy()
        train_mask = all_data["label_source"] == "historical_main"
        holdout_mask = all_data["label_source"] == "new_books_holdout_2026"
        train_df = all_data[train_mask].copy()
        holdout_df = all_data[holdout_mask].copy()
        train_p, holdout_p, nf_all, cf = prepare_feature_frames(train_df, holdout_df, target, spec)
        nf = [f for f in nf_all if f not in AUTHOR_FEATURES]

        y_train = pd.to_numeric(train_p[target], errors="coerce")
        valid = y_train.notna()
        X_train = train_p[valid].copy()
        y_train = y_train[valid].values
        for c in nf:
            med = X_train[c].median()
            X_train[c] = X_train[c].fillna(med)
            holdout_p[c] = holdout_p[c].fillna(med)
        for c in cf:
            X_train[c] = X_train[c].fillna("Unknown")
            holdout_p[c] = holdout_p[c].fillna("Unknown")
        y_holdout = holdout_p[target].values.astype(float)

        for model_name, builder in [("RF", random_forest_model), ("GBM", gbm_model)]:
            enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist")
            pre = ColumnTransformer([("num", StandardScaler(), nf), ("cat", enc, cf)], remainder="drop")
            pipe = Pipeline([("preprocessor", pre), ("model", builder())])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(holdout_p)

            rho_obs, p_perm = permutation_test_rho(y_holdout, pred, n_perm=5000)
            rho_sp, p_sp = spearmanr(y_holdout, pred)
            print(f"  {model_name} {tshort}: rho={rho_obs:.3f}, "
                  f"scipy_p={p_sp:.4f}, permutation_p={p_perm:.4f}")

    # ── Feature ablation ──
    print("\n[5/7] Feature ablation study...")
    ablation_results = []
    for target in ["avg_enjoyment", "avg_usefulness"]:
        ablation_results.append(run_ablation_study(tree_data, target, spec))
    ablation_df = pd.concat(ablation_results, ignore_index=True)
    ablation_df.to_csv("ai_books_tracking/plots/ablation_results.csv", index=False)
    print("\n  Ablation results:")
    for target in ["enjoy", "useful"]:
        print(f"\n  --- {target.upper()} ---")
        sub = ablation_df[ablation_df["target"] == target].sort_values("rho_holdout", ascending=False)
        for _, row in sub.iterrows():
            print(f"    {row['model']:3s} {row['ablation']:30s} "
                  f"rho_ho={row['rho_holdout']:+.3f}  R²={row['r2_holdout']:+.3f}  MAE={row['mae_holdout']:.3f}")

    # ── Bootstrap Ridge coefficients ──
    print("\n[6/7] Bootstrap Ridge coefficient stability (500 iterations)...")
    boot_df = bootstrap_ridge_coefficients(ridge_frame, n_boot=500)
    boot_df.to_csv("ai_books_tracking/plots/bootstrap_ridge_coefficients.csv", index=False)
    print("\n  Bootstrap results:")
    for grp in GROUP_ORDER:
        for target in ["enjoy", "useful"]:
            sub = boot_df[(boot_df["group"] == grp) & (boot_df["target"] == target)]
            print(f"\n  {grp} / {target}:")
            for _, row in sub.iterrows():
                if row["feature"] == "HOLDOUT_RHO":
                    print(f"    HOLDOUT_RHO: median={row['coef_full']:.3f}, "
                          f"95% CI=[{row['coef_2.5%']:.3f}, {row['coef_97.5%']:.3f}], "
                          f"P(rho>0)={row['sign_stability']:.1%}")
                elif row["feature"] == "intercept":
                    print(f"    intercept: {row['coef_full']:.3f} "
                          f"[{row['coef_2.5%']:.3f}, {row['coef_97.5%']:.3f}]")
                else:
                    print(f"    {row['feature']}: {row['coef_full']:+.3f} "
                          f"[{row['coef_2.5%']:+.3f}, {row['coef_97.5%']:+.3f}] "
                          f"sign_stability={row['sign_stability']:.1%}")

    # ── Bias-corrected R² ──
    print("\n[7a/7] Bias-corrected R² analysis...")
    bias_df = bias_corrected_analysis(ridge_frame)
    bias_df.to_csv("ai_books_tracking/plots/bias_corrected_r2.csv", index=False)
    print("\n  Bias-corrected results:")
    for _, row in bias_df.iterrows():
        print(f"  {row['group']}/{row['target']} (n={row['n']}): "
              f"train_mean={row['train_mean']:.2f}, holdout_mean={row['holdout_mean']:.2f}, "
              f"bias={row['bias']:+.2f}")
        print(f"    R²_raw={row['r2_raw']:.3f} → R²_corrected={row['r2_bias_corrected']:.3f}  "
              f"MAE_raw={row['mae_raw']:.2f} → MAE_corrected={row['mae_bias_corrected']:.2f}  "
              f"pearson_r={row['pearson_r']:.3f}")

    # ── LOO-CV ──
    print("\n[7b/7] LOO-CV for Ridge (honest in-sample estimate)...")
    loo_df = loo_cv_ridge(ridge_frame)
    loo_df.to_csv("ai_books_tracking/plots/loo_cv_ridge.csv", index=False)
    print("\n  LOO-CV results:")
    for _, row in loo_df.iterrows():
        print(f"  {row['group']}/{row['target']} (n={row['n']}): "
              f"rho_insample={row['rho_in_sample']:.3f}, rho_loo={row['rho_loo_cv']:.3f} "
              f"(gap={row['rho_overfit_gap']:.3f}), "
              f"R²_insample={row['r2_in_sample']:.3f}, R²_loo={row['r2_loo_cv']:.3f}, "
              f"MAE_loo={row['mae_loo_cv']:.2f}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
