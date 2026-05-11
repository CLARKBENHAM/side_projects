"""Export RF, GBM, and Ridge models as JSON for the Chrome extension.

- RF/GBM: from future_prediction_evaluation.py (Bookshelf + GR conservative),
  WITHOUT author features (they're always unknown for new books).
- Ridge: per-group models from practical_group_rules.py pipeline
  (3 raw ratings: GR, OL, AMZ with cross-source imputation per group).

Computes split-conformal prediction interval quantiles from holdout residuals
for each model × target combination, at 50% and 85% coverage levels.
Per-group conformal for Ridge; pooled conformal for RF/GBM.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_books_tracking.future_prediction_evaluation import (
    FeatureSpec,
    prepare_feature_frames,
    random_forest_model,
    gbm_model,
)
from ai_books_tracking.goodreads_followup_analysis import (
    derive_analysis_columns,
    load_data,
)
from ai_books_tracking.new_books_to_rate_analysis import (
    NEW_BOOKS_ENRICHED,
    add_centered_targets,
)
from ai_books_tracking.multi_source_selection_policy import load_frame
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

AUTHOR_FEATURES = {"author_target_mean_hist", "author_book_count_hist"}
RATING_FEATURES = [
    "goodreads_rating_raw",
    "openlibrary_rating_raw",
    "amazon_rating_raw",
]
FULL_RIDGE_NUMERIC_FEATURES = [
    "goodreads_rating_raw",
    "openlibrary_rating_raw",
    "amazon_rating_raw",
    "goodreads_log_count",
    "amazon_log_count",
    "log_pages",
    "book_age",
]
GROUP_ORDER = ["Business_Histories_General", "Fiction_Literature", "Technical_Other"]
COVERAGE_LEVELS = [0.50, 0.85]
PERCENTILE_GRID = list(range(0, 101, 5))


def group_name(cat: object) -> str:
    c = str(cat).strip() if pd.notna(cat) else ""
    if c in ("fiction", "Literature"):
        return "Fiction_Literature"
    if c in ("Computer Science", "Machine Learning", "Math", "Unknown Shelf"):
        return "Technical_Other"
    return "Business_Histories_General"


def predictor_sets_for(target_feature: str) -> list[tuple[str, ...]]:
    others = [f for f in RATING_FEATURES if f != target_feature]
    return [(others[0], others[1]), (others[0],), (others[1],)]


def fit_imputation_models(
    group_train: pd.DataFrame,
) -> tuple[dict[str, dict[tuple[str, ...], Ridge]], pd.Series]:
    numeric = group_train[RATING_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median().fillna(numeric.median().median())
    models: dict[str, dict[tuple[str, ...], Ridge]] = {}
    for target_feature in RATING_FEATURES:
        models[target_feature] = {}
        for predictors in predictor_sets_for(target_feature):
            mask = numeric[target_feature].notna()
            for p in predictors:
                mask &= numeric[p].notna()
            subset = group_train.loc[mask]
            if len(subset) < 8:
                continue
            X = (
                subset[list(predictors)]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy()
            )
            y = pd.to_numeric(subset[target_feature], errors="coerce").to_numpy()
            models[target_feature][predictors] = Ridge(alpha=1.0).fit(X, y)
    return models, medians


def impute_group_sources(
    frame: pd.DataFrame,
    imputation_models: dict[str, dict[tuple[str, ...], Ridge]],
    medians: pd.Series,
) -> pd.DataFrame:
    imputed = frame.copy()
    numeric = imputed[RATING_FEATURES].apply(pd.to_numeric, errors="coerce")
    for target_feature in RATING_FEATURES:
        missing_idx = numeric[numeric[target_feature].isna()].index
        for idx in missing_idx:
            row = numeric.loc[idx]
            value = np.nan
            for predictors in predictor_sets_for(target_feature):
                if any(pd.isna(row[p]) for p in predictors):
                    continue
                model = imputation_models[target_feature].get(predictors)
                if model is None:
                    continue
                X = row[list(predictors)].to_numpy(dtype=float).reshape(1, -1)
                value = float(model.predict(X)[0])
                break
            if not np.isfinite(value):
                value = float(medians[target_feature])
            numeric.loc[idx, target_feature] = value
        imputed[target_feature] = numeric[target_feature]
    return imputed


def export_imputation_cascade(
    imputation_models: dict[str, dict[tuple[str, ...], Ridge]],
    medians: pd.Series,
) -> dict:
    """Serialize imputation models for JS consumption."""
    result: dict[str, object] = {
        "medians": {f: float(medians[f]) for f in RATING_FEATURES}
    }
    for target_feature in RATING_FEATURES:
        steps = []
        for predictors in predictor_sets_for(target_feature):
            model = imputation_models[target_feature].get(predictors)
            if model is None:
                continue
            steps.append(
                {
                    "predictors": list(predictors),
                    "intercept": float(model.intercept_),
                    "coefs": model.coef_.tolist(),
                }
            )
        result[target_feature] = steps
    return result


def compute_conformal_quantiles(residuals: np.ndarray) -> dict[str, float]:
    abs_resid = np.abs(residuals)
    abs_sorted = np.sort(abs_resid)
    n = len(abs_resid)
    quantiles: dict[str, float] = {}
    for level in COVERAGE_LEVELS:
        idx = min(math.ceil((n + 1) * level), n) - 1
        quantiles[f"q{int(level * 100)}"] = float(abs_sorted[idx])
    for level in COVERAGE_LEVELS:
        alpha = 1.0 - level
        quantiles[f"lo{int(level * 100)}"] = float(np.quantile(residuals, alpha / 2))
        quantiles[f"hi{int(level * 100)}"] = float(
            np.quantile(residuals, 1 - alpha / 2)
        )
    return {
        "n_calibration": n,
        "mean_abs_residual": float(np.mean(abs_resid)),
        "residual_std": float(np.std(residuals)),
        **quantiles,
    }


def build_percentile_map(values: pd.Series | np.ndarray) -> dict[str, list[float]]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return {"percentiles": PERCENTILE_GRID, "values": [0.0] * len(PERCENTILE_GRID)}
    return {
        "percentiles": PERCENTILE_GRID,
        "values": np.percentile(numeric.to_numpy(), PERCENTILE_GRID)
        .astype(float)
        .tolist(),
    }


def load_training_frame():
    historical = add_centered_targets(derive_analysis_columns(load_data()))
    historical["label_source"] = "historical_main"
    frames = [historical]
    if NEW_BOOKS_ENRICHED.exists():
        new_holdout = pd.read_csv(NEW_BOOKS_ENRICHED)
        new_holdout.columns = new_holdout.columns.str.strip()
        new_holdout = add_centered_targets(derive_analysis_columns(new_holdout))
        new_holdout["label_source"] = "new_books_holdout_2026"
        frames.append(new_holdout)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_key = (
        combined["title"].astype(str).str.strip().str.lower()
        + "||"
        + combined.get("filename", pd.Series("", index=combined.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    combined = combined.loc[~dedupe_key.duplicated()].reset_index(drop=True)
    return combined


def add_extension_safe_book_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["goodreads_log_count"] = np.log10(
        1 + pd.to_numeric(enriched["goodreads_count_raw"], errors="coerce")
    )
    enriched["amazon_log_count"] = np.log10(
        1 + pd.to_numeric(enriched["amazon_count_raw"], errors="coerce")
    )
    enriched["log_pages"] = pd.to_numeric(enriched["log_pages"], errors="coerce")
    enriched["book_age"] = pd.to_numeric(enriched["book_age"], errors="coerce")
    return enriched


def apply_group_rating_imputation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for group in GROUP_ORDER:
        group_train = train_df[train_df["group"] == group].copy()
        group_test = test_df[test_df["group"] == group].copy()
        if group_train.empty:
            continue
        imputation_models, medians = fit_imputation_models(group_train)
        train_parts.append(
            impute_group_sources(group_train, imputation_models, medians)
        )
        if not group_test.empty:
            test_parts.append(
                impute_group_sources(group_test, imputation_models, medians)
            )
    train_out = pd.concat(train_parts, ignore_index=False).sort_index()
    test_out = (
        pd.concat(test_parts, ignore_index=False).sort_index()
        if test_parts
        else test_df.iloc[0:0].copy()
    )
    return train_out, test_out


def build_pooled_ridge_design(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    fill_values: dict[str, float] = {}
    for column in numeric_cols:
        train_out[column] = pd.to_numeric(train_out[column], errors="coerce")
        test_out[column] = pd.to_numeric(test_out[column], errors="coerce")
        fill_value = (
            float(train_out[column].median())
            if train_out[column].notna().any()
            else 0.0
        )
        fill_values[column] = fill_value
        train_out[column] = train_out[column].fillna(fill_value)
        test_out[column] = test_out[column].fillna(fill_value)

    train_cat = pd.get_dummies(
        train_out["category"], prefix="category", drop_first=True
    )
    test_cat = pd.get_dummies(test_out["category"], prefix="category", drop_first=True)
    test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)
    category_cols = list(train_cat.columns)

    X_train = pd.concat(
        [
            train_out[numeric_cols].reset_index(drop=True),
            train_cat.reset_index(drop=True),
        ],
        axis=1,
    )
    X_test = pd.concat(
        [
            test_out[numeric_cols].reset_index(drop=True),
            test_cat.reset_index(drop=True),
        ],
        axis=1,
    )
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test, fill_values, category_cols


def export_tree(tree, feature_names):
    t = tree.tree_

    def recurse(node_id):
        if t.children_left[node_id] == -1:
            return {"value": float(t.value[node_id].flatten()[0])}
        feat = feature_names[t.feature[node_id]]
        return {
            "feature": feat,
            "threshold": float(t.threshold[node_id]),
            "left": recurse(int(t.children_left[node_id])),
            "right": recurse(int(t.children_right[node_id])),
        }

    return recurse(0)


def main():
    output = {}
    conformal_data: dict[str, dict] = {}
    percentile_maps: dict[str, dict[str, list[float]]] = {}

    # ══════════════════════════════════════════════════════════════
    # RF and GBM: without author features
    # ══════════════════════════════════════════════════════════════
    print("Loading RF/GBM training data...")
    training = load_training_frame()
    spec = FeatureSpec(
        "preread_plus_goodreads_conservative", include_goodreads="conservative"
    )

    for target in ["avg_enjoyment", "avg_usefulness"]:
        target_short = "enjoy" if "enjoy" in target else "useful"
        train_df = training[training[target].notna()].copy()
        dummy_test = train_df.iloc[:1].copy()

        train_prepared, test_prepared, numeric_features_all, categorical_features = (
            prepare_feature_frames(train_df, dummy_test, target, spec)
        )

        # Drop author features
        numeric_features = [f for f in numeric_features_all if f not in AUTHOR_FEATURES]

        for model_name, model_builder in [
            ("rf", random_forest_model),
            ("gbm", gbm_model),
        ]:
            model = model_builder()
            cat_encoder = OneHotEncoder(
                drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"
            )
            preprocessor = ColumnTransformer(
                [
                    ("num", StandardScaler(), numeric_features),
                    ("cat", cat_encoder, categorical_features),
                ],
                remainder="drop",
            )
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            y_train = pd.to_numeric(train_prepared[target], errors="coerce")
            valid = y_train.notna()
            X_train = train_prepared.loc[valid].copy()
            y_train = y_train[valid].values

            for col in numeric_features:
                X_train[col] = X_train[col].fillna(X_train[col].median())
            for col in categorical_features:
                X_train[col] = X_train[col].fillna("Unknown")

            pipeline.fit(X_train, y_train)

            scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
            encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]

            medians = {col: float(X_train[col].median()) for col in numeric_features}
            fitted_model = pipeline.named_steps["model"]
            all_feature_names = numeric_features + list(
                encoder.get_feature_names_out(categorical_features)
            )

            key = f"{model_name}_{target_short}"
            model_data = {
                "type": model_name if model_name == "rf" else "gbm",
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "scaler_means": dict(zip(numeric_features, scaler.mean_.tolist())),
                "scaler_scales": dict(zip(numeric_features, scaler.scale_.tolist())),
                "cat_categories": [list(c) for c in encoder.categories_],
                "cat_drop_idx": (
                    encoder.drop_idx_.tolist()
                    if encoder.drop_idx_ is not None
                    else None
                ),
                "medians": medians,
                "all_feature_names": all_feature_names,
            }
            percentile_maps[key] = build_percentile_map(y_train)

            if model_name == "rf":
                print(
                    f"  Exporting RF {target_short}: {len(fitted_model.estimators_)} trees (no author features)"
                )
                model_data["trees"] = [
                    export_tree(est, all_feature_names)
                    for est in fitted_model.estimators_
                ]
            else:
                print(
                    f"  Exporting GBM {target_short}: {len(fitted_model.estimators_)} stages (no author features)"
                )
                model_data["trees"] = [
                    export_tree(est[0], all_feature_names)
                    for est in fitted_model.estimators_
                ]
                model_data["init_value"] = float(fitted_model.init_.constant_[0][0])
                model_data["learning_rate"] = float(fitted_model.learning_rate)

            output[key] = model_data

    # --- RF/GBM conformal: re-fit on train-only, predict on holdout ---
    print("\nComputing RF/GBM conformal intervals (without author features)...")
    for target in ["avg_enjoyment", "avg_usefulness"]:
        target_short = "enjoy" if "enjoy" in target else "useful"
        all_data = training[training[target].notna()].copy()
        train_mask = all_data["label_source"] == "historical_main"
        holdout_mask = all_data["label_source"] == "new_books_holdout_2026"

        if holdout_mask.sum() < 10:
            print(f"  WARNING: Only {holdout_mask.sum()} holdout for {target_short}")
            continue

        train_df_c = all_data[train_mask].copy()
        holdout_df_c = all_data[holdout_mask].copy()

        train_p_c, holdout_p_c, nf_all, cf_c = prepare_feature_frames(
            train_df_c, holdout_df_c, target, spec
        )
        nf_c = [f for f in nf_all if f not in AUTHOR_FEATURES]
        y_holdout = holdout_p_c[target].values.astype(float)

        for model_name, model_builder in [
            ("rf", random_forest_model),
            ("gbm", gbm_model),
        ]:
            enc_c = OneHotEncoder(
                drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"
            )
            pre_c = ColumnTransformer(
                [("num", StandardScaler(), nf_c), ("cat", enc_c, cf_c)],
                remainder="drop",
            )
            pipe_c = Pipeline([("preprocessor", pre_c), ("model", model_builder())])

            y_train_c = pd.to_numeric(train_p_c[target], errors="coerce")
            valid_c = y_train_c.notna()
            X_train_c = train_p_c.loc[valid_c].copy()
            y_train_c = y_train_c[valid_c].values

            for col in nf_c:
                med = X_train_c[col].median()
                X_train_c[col] = X_train_c[col].fillna(med)
                holdout_p_c[col] = holdout_p_c[col].fillna(med)
            for col in cf_c:
                X_train_c[col] = X_train_c[col].fillna("Unknown")
                holdout_p_c[col] = holdout_p_c[col].fillna("Unknown")

            pipe_c.fit(X_train_c, y_train_c)
            y_pred = pipe_c.predict(holdout_p_c)

            residuals = y_holdout - y_pred
            rho, p = spearmanr(y_holdout, y_pred)
            key = f"{model_name}_{target_short}"
            conformal_data[key] = compute_conformal_quantiles(residuals)
            print(
                f"  {key}: n={len(residuals)}, rho={rho:.3f} (p={p:.4f}), "
                f"50% q={conformal_data[key]['q50']:.3f}, 85% q={conformal_data[key]['q85']:.3f}"
            )

    # ══════════════════════════════════════════════════════════════
    # Ridge: per-group models from golden_master_multi_source.csv
    # 3 features per group: raw GR, OL, AMZ ratings
    # Cross-source imputation per group
    # ══════════════════════════════════════════════════════════════
    print("\nLoading Ridge data (golden_master_multi_source.csv)...")
    frame = load_frame()
    frame["group"] = frame["category"].map(group_name)
    frame["dataset"] = np.where(frame["source"].eq("Holdout 2026"), "holdout", "train")
    frame = add_extension_safe_book_features(frame)

    train_all = frame[frame["dataset"] == "train"].copy()
    holdout_all = frame[frame["dataset"] == "holdout"].copy()

    ridge_groups = {}
    all_ridge_holdout_residuals: dict[str, list[float]] = {"enjoy": [], "useful": []}

    for grp in GROUP_ORDER:
        grp_train = train_all[train_all["group"] == grp].copy()
        grp_holdout = holdout_all[holdout_all["group"] == grp].copy()

        print(f"\n  Group '{grp}': train={len(grp_train)}, holdout={len(grp_holdout)}")

        # Fit cross-source imputation on this group's training data
        imp_models, imp_medians = fit_imputation_models(grp_train)

        # Impute both train and holdout
        grp_train = impute_group_sources(grp_train, imp_models, imp_medians)
        grp_holdout = impute_group_sources(grp_holdout, imp_models, imp_medians)

        group_data = {
            "imputation": export_imputation_cascade(imp_models, imp_medians),
        }

        for target_name in ["avg_enjoyment", "avg_usefulness"]:
            target_short = "enjoy" if "enjoy" in target_name else "useful"

            tv = grp_train[grp_train[target_name].notna()].copy()
            fill_values = tv[RATING_FEATURES].median()
            X_tr = tv[RATING_FEATURES].fillna(fill_values)
            y_tr = tv[target_name].values

            ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)

            coefs = dict(zip(RATING_FEATURES, ridge.coef_.tolist()))
            intercept = float(ridge.intercept_)

            print(
                f"    {target_short}: intercept={intercept:.3f}  "
                + "  ".join(f"{f.split('_')[0]}={c:+.3f}" for f, c in coefs.items())
            )

            model_spec = {
                "intercept": intercept,
                "coefficients": coefs,
                "fill_values": {f: float(fill_values[f]) for f in RATING_FEATURES},
            }

            # Holdout evaluation
            hv = grp_holdout[grp_holdout[target_name].notna()].copy()
            if len(hv) >= 3:
                X_ho = hv[RATING_FEATURES].fillna(fill_values)
                y_ho = hv[target_name].values
                pred_ho = np.clip(X_ho.values @ ridge.coef_ + intercept, 1, 5)
                residuals = y_ho - pred_ho
                rho, p = (
                    spearmanr(y_ho, pred_ho) if len(y_ho) > 3 else (float("nan"), 1.0)
                )
                bias = float(pred_ho.mean() - y_ho.mean())
                print(
                    f"      holdout: n={len(hv)}, rho={rho:.3f} (p={p:.4f}), bias={bias:+.2f}"
                )

                # Per-group conformal if enough holdout data
                if len(hv) >= 10:
                    conf_key = f"ridge_{grp}_{target_short}"
                    conformal_data[conf_key] = compute_conformal_quantiles(residuals)
                    print(
                        f"      conformal: 50% q={conformal_data[conf_key]['q50']:.3f}, "
                        f"85% q={conformal_data[conf_key]['q85']:.3f}"
                    )

                all_ridge_holdout_residuals[target_short].extend(residuals.tolist())

            group_data[target_short] = model_spec

        ridge_groups[grp] = group_data

    # Pooled Ridge conformal (fallback for small groups)
    for target_short in ["enjoy", "useful"]:
        resids = np.array(all_ridge_holdout_residuals[target_short])
        if len(resids) >= 10:
            key = f"ridge_pooled_{target_short}"
            conformal_data[key] = compute_conformal_quantiles(resids)
            rho_str = f"n={len(resids)}"
            print(
                f"\n  {key}: {rho_str}, 50% q={conformal_data[key]['q50']:.3f}, "
                f"85% q={conformal_data[key]['q85']:.3f}"
            )

    output["ridge_groups"] = ridge_groups
    percentile_maps["ridge_enjoy"] = build_percentile_map(train_all["avg_enjoyment"])
    percentile_maps["ridge_useful"] = build_percentile_map(train_all["avg_usefulness"])

    print("\nExporting fuller pooled ridge (counts + book meta + category dummies)...")
    pooled_train, pooled_holdout = apply_group_rating_imputation(train_all, holdout_all)
    for target_name in ["avg_enjoyment", "avg_usefulness"]:
        target_short = "enjoy" if "enjoy" in target_name else "useful"
        tv = pooled_train[pooled_train[target_name].notna()].copy()
        hv = pooled_holdout[pooled_holdout[target_name].notna()].copy()
        X_tr, X_ho, fill_values, category_cols = build_pooled_ridge_design(
            tv,
            hv,
            FULL_RIDGE_NUMERIC_FEATURES,
        )
        y_tr = tv[target_name].values
        pooled_ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)

        coef_map = dict(zip(X_tr.columns, pooled_ridge.coef_.tolist()))
        key = f"ridge_full_{target_short}"
        output[key] = {
            "type": "pooled_ridge",
            "numeric_features": FULL_RIDGE_NUMERIC_FEATURES,
            "category_features": category_cols,
            "fill_values": fill_values,
            "intercept": float(pooled_ridge.intercept_),
            "coefficients": coef_map,
        }
        percentile_maps[key] = build_percentile_map(tv[target_name])

        if len(hv) >= 3:
            y_ho = hv[target_name].values
            pred_ho = np.clip(pooled_ridge.predict(X_ho), 1, 5)
            residuals = y_ho - pred_ho
            rho, p = spearmanr(y_ho, pred_ho) if len(y_ho) > 3 else (float("nan"), 1.0)
            bias = float(pred_ho.mean() - y_ho.mean())
            conformal_data[key] = compute_conformal_quantiles(residuals)
            print(
                f"  {key}: intercept={pooled_ridge.intercept_:.3f}, "
                f"n={len(hv)}, rho={rho:.3f} (p={p:.4f}), bias={bias:+.2f}, "
                f"50% q={conformal_data[key]['q50']:.3f}, 85% q={conformal_data[key]['q85']:.3f}"
            )

    output["conformal"] = conformal_data
    output["percentile_maps"] = percentile_maps

    # ══════════════════════════════════════════════════════════════
    # Write JSON
    # ══════════════════════════════════════════════════════════════
    out_path = Path("ai_books_tracking/chrome_extension/models.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nExported to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
