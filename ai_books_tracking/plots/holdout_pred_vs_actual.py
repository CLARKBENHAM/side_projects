"""Generate predicted vs actual holdout plots for all models."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

GROUP_COLORS = {
    "Business_Histories_General": "#2196F3",
    "Fiction_Literature": "#E91E63",
    "Technical_Other": "#4CAF50",
}
GROUP_LABELS = {
    "Business_Histories_General": "Business/Hist/General",
    "Fiction_Literature": "Fiction/Literature",
    "Technical_Other": "Technical/Other",
}


def get_ridge_preds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train_preds, holdout_preds) DataFrames."""
    frame = load_frame()
    frame["group"] = frame["category"].map(group_name)
    frame["dataset"] = np.where(frame["source"].eq("Holdout 2026"), "holdout", "train")
    train = frame[frame["dataset"] == "train"]
    holdout = frame[frame["dataset"] == "holdout"]

    train_rows, holdout_rows = [], []
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

            # Train predictions
            train_pred = np.clip(X_tr.values @ ridge.coef_ + ridge.intercept_, 1, 5)
            for i, idx in enumerate(tv.index):
                train_rows.append({
                    "model": "Ridge", "target": tshort, "group": grp,
                    "actual": float(tv.loc[idx, target]),
                    "pred": float(train_pred[i]),
                    "title": str(tv.loc[idx, "title"]),
                })

            # Holdout predictions
            hv = gh[gh[target].notna()].copy()
            if len(hv) == 0:
                continue
            X_ho = hv[RATING_FEATURES].fillna(fill)
            pred = np.clip(X_ho.values @ ridge.coef_ + ridge.intercept_, 1, 5)
            for i, idx in enumerate(hv.index):
                holdout_rows.append({
                    "model": "Ridge", "target": tshort, "group": grp,
                    "actual": float(hv.loc[idx, target]),
                    "pred": float(pred[i]),
                    "title": str(hv.loc[idx, "title"]),
                })
    return pd.DataFrame(train_rows), pd.DataFrame(holdout_rows)


def get_tree_preds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train_preds, holdout_preds) DataFrames."""
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

    spec = FeatureSpec("preread_plus_goodreads_conservative", include_goodreads="conservative")
    train_rows, holdout_rows = [], []

    for target in ["avg_enjoyment", "avg_usefulness"]:
        tshort = "enjoy" if "enjoy" in target else "useful"
        all_data = combined[combined[target].notna()].copy()
        train_mask = all_data["label_source"] == "historical_main"
        holdout_mask = all_data["label_source"] == "new_books_holdout_2026"

        train_df = all_data[train_mask].copy()
        holdout_df = all_data[holdout_mask].copy()
        train_p, holdout_p, nf_all, cf = prepare_feature_frames(train_df, holdout_df, target, spec)
        nf = [f for f in nf_all if f not in AUTHOR_FEATURES]

        y_train_raw = pd.to_numeric(train_p[target], errors="coerce")
        valid = y_train_raw.notna()
        X_train = train_p[valid].copy()
        y_train = y_train_raw[valid].values
        train_groups = train_df.loc[valid.values, "group"].values if len(train_df) == len(valid) else train_df["group"].values[:valid.sum()]

        for c in nf:
            med = X_train[c].median()
            X_train[c] = X_train[c].fillna(med)
            holdout_p[c] = holdout_p[c].fillna(med)
        for c in cf:
            X_train[c] = X_train[c].fillna("Unknown")
            holdout_p[c] = holdout_p[c].fillna("Unknown")

        y_holdout = holdout_p[target].values.astype(float)
        holdout_groups = holdout_df["group"].values

        for model_name, builder in [("RF", random_forest_model), ("GBM", gbm_model)]:
            enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist")
            pre = ColumnTransformer([("num", StandardScaler(), nf), ("cat", enc, cf)], remainder="drop")
            pipe = Pipeline([("preprocessor", pre), ("model", builder())])
            pipe.fit(X_train, y_train)

            # Train predictions
            train_pred = pipe.predict(X_train)
            for i in range(len(y_train)):
                train_rows.append({
                    "model": model_name, "target": tshort,
                    "group": train_groups[i],
                    "actual": float(y_train[i]),
                    "pred": float(train_pred[i]),
                    "title": str(train_df.iloc[i].get("title", "") if i < len(train_df) else ""),
                })

            # Holdout predictions
            pred = pipe.predict(holdout_p)
            for i in range(len(y_holdout)):
                holdout_rows.append({
                    "model": model_name, "target": tshort,
                    "group": holdout_groups[i],
                    "actual": float(y_holdout[i]),
                    "pred": float(pred[i]),
                    "title": str(holdout_df.iloc[i].get("title", "")),
                })
    return pd.DataFrame(train_rows), pd.DataFrame(holdout_rows)


def plot_pred_vs_actual(all_df: pd.DataFrame, title_prefix: str, out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    model_order = ["Ridge", "RF", "GBM"]
    target_order = ["enjoy", "useful"]
    target_labels = {"enjoy": "Enjoyment", "useful": "Usefulness"}

    for row, target in enumerate(target_order):
        for col, model in enumerate(model_order):
            ax = axes[row, col]
            subset = all_df[(all_df["model"] == model) & (all_df["target"] == target)]

            for grp in GROUP_ORDER:
                gs = subset[subset["group"] == grp]
                if len(gs) == 0:
                    continue
                ax.scatter(
                    gs["actual"], gs["pred"],
                    c=GROUP_COLORS[grp], label=GROUP_LABELS[grp],
                    s=40, alpha=0.7, edgecolors="white", linewidth=0.5,
                )

            lo = min(subset["actual"].min(), subset["pred"].min(), 1) - 0.2
            hi = max(subset["actual"].max(), subset["pred"].max(), 5) + 0.2
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")

            rho, p = spearmanr(subset["actual"], subset["pred"]) if len(subset) > 3 else (float("nan"), 1)
            r2 = r2_score(subset["actual"], subset["pred"]) if len(subset) > 1 else float("nan")
            mae = mean_absolute_error(subset["actual"], subset["pred"]) if len(subset) > 0 else float("nan")
            ax.set_title(f"{model} {target_labels[target]}\n"
                         f"rho={rho:.3f}  R²={r2:.3f}  MAE={mae:.2f}  n={len(subset)}",
                         fontsize=10)
            ax.set_xlabel("Actual" if row == 1 else "")
            ax.set_ylabel("Predicted" if col == 0 else "")
            ax.grid(True, alpha=0.2)

            group_stats = []
            for grp in GROUP_ORDER:
                gs = subset[subset["group"] == grp]
                if len(gs) >= 2:
                    g_mae = mean_absolute_error(gs["actual"], gs["pred"])
                    g_r2 = r2_score(gs["actual"], gs["pred"]) if len(gs) > 2 else float("nan")
                    group_stats.append(f"{GROUP_LABELS[grp]}: MAE={g_mae:.2f} R²={g_r2:.2f} n={len(gs)}")
            if group_stats:
                ax.text(0.03, 0.97, "\n".join(group_stats), transform=ax.transAxes,
                        fontsize=7, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"{title_prefix} Predicted vs Actual Ratings", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def main():
    print("Computing Ridge predictions...")
    ridge_train, ridge_holdout = get_ridge_preds()
    print("Computing RF/GBM predictions...")
    tree_train, tree_holdout = get_tree_preds()

    holdout_df = pd.concat([ridge_holdout, tree_holdout], ignore_index=True)
    train_df = pd.concat([ridge_train, tree_train], ignore_index=True)

    plot_pred_vs_actual(holdout_df, "Holdout", "ai_books_tracking/plots/holdout_pred_vs_actual.png")
    plot_pred_vs_actual(train_df, "Training", "ai_books_tracking/plots/train_pred_vs_actual.png")


if __name__ == "__main__":
    main()
