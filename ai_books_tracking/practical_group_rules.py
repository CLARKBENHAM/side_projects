"""Build simple per-group practical rules and apply them to full holdout."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ai_books_tracking.multi_source_selection_policy import load_frame

OUTPUT_DIR = Path(__file__).parent
COEF_CSV = OUTPUT_DIR / "practical_group_rule_coefficients.csv"
SUMMARY_MD = OUTPUT_DIR / "practical_group_rules.md"
RANKED_CSV = OUTPUT_DIR / "practical_group_holdout_ranked_books.csv"
AGGREGATE_CSV = OUTPUT_DIR / "practical_group_holdout_aggregate_changes.csv"
DROPPED_CSV = OUTPUT_DIR / "practical_group_holdout_dropped_books_by_level.csv"
KEPT_CSV = OUTPUT_DIR / "practical_group_holdout_kept_books_by_level.csv"

GROUP_ORDER = [
    "Business/Histories/General",
    "Fiction/Literature",
    "Technical/Other",
]
FEATURE_COLS = ["goodreads_rating_raw", "openlibrary_rating_raw", "amazon_rating_raw"]
TARGET_SPECS = [
    ("avg_enjoyment", "Average enjoyment"),
    ("avg_usefulness", "Average usefulness"),
]
KEEP_SHARES = [0.40, 0.30, 0.20]


def group_name(category: object) -> str:
    text = "" if pd.isna(category) else str(category).strip()
    if text in {"fiction", "Literature"}:
        return "Fiction/Literature"
    if text in {"Computer Science", "Machine Learning", "Math", "Unknown Shelf"}:
        return "Technical/Other"
    return "Business/Histories/General"


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def predictor_sets_for(target_feature: str) -> list[tuple[str, ...]]:
    others = [feature for feature in FEATURE_COLS if feature != target_feature]
    return [(others[0], others[1]), (others[0],), (others[1],)]


def fit_imputation_models(group_train: pd.DataFrame) -> tuple[dict[str, dict[tuple[str, ...], Ridge]], pd.Series]:
    numeric = group_train[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median().fillna(numeric.median().median())
    models: dict[str, dict[tuple[str, ...], Ridge]] = {}
    for target_feature in FEATURE_COLS:
        models[target_feature] = {}
        for predictors in predictor_sets_for(target_feature):
            mask = numeric[target_feature].notna()
            for predictor in predictors:
                mask &= numeric[predictor].notna()
            subset = group_train.loc[mask].copy()
            if len(subset) < 8:
                continue
            X = subset[list(predictors)].apply(pd.to_numeric, errors="coerce").to_numpy()
            y = pd.to_numeric(subset[target_feature], errors="coerce").to_numpy()
            models[target_feature][predictors] = Ridge(alpha=1.0).fit(X, y)
    return models, medians


def impute_group_sources(
    frame: pd.DataFrame,
    imputation_models: dict[str, dict[tuple[str, ...], Ridge]],
    medians: pd.Series,
) -> pd.DataFrame:
    imputed = frame.copy()
    numeric = imputed[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    for target_feature in FEATURE_COLS:
        missing_index = numeric[numeric[target_feature].isna()].index
        for idx in missing_index:
            row = numeric.loc[idx]
            predicted_value = np.nan
            for predictors in predictor_sets_for(target_feature):
                if any(pd.isna(row[predictor]) for predictor in predictors):
                    continue
                model = imputation_models[target_feature].get(predictors)
                if model is None:
                    continue
                X = row[list(predictors)].to_numpy(dtype=float).reshape(1, -1)
                predicted_value = float(model.predict(X)[0])
                break
            if not np.isfinite(predicted_value):
                predicted_value = float(medians[target_feature])
            numeric.loc[idx, target_feature] = predicted_value
        imputed[target_feature] = numeric[target_feature]
    return imputed


def load_suspicious_map() -> pd.DataFrame:
    path = OUTPUT_DIR / "multi_source_suspicious_rows.csv"
    if not path.exists():
        return pd.DataFrame(columns=["title", "source", "issues"])
    suspicious = pd.read_csv(path)
    suspicious["title"] = suspicious["title"].fillna("").astype(str)
    suspicious["source"] = suspicious["source"].fillna("").astype(str)
    return suspicious[["title", "source", "issues"]].copy()


def fit_group_rules(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    frame = frame.copy()
    frame["group_name"] = frame["category"].map(group_name)
    frame["dataset"] = np.where(frame["source"].eq("Holdout 2026"), "holdout", "train")

    suspicious = load_suspicious_map()
    frame = frame.merge(suspicious, on=["title", "source"], how="left")

    coefficient_rows: list[dict[str, object]] = []
    summary_lines = ["# Practical Group Rules", ""]
    holdout_scored_parts: list[pd.DataFrame] = []

    for group in GROUP_ORDER:
        group_train = frame[(frame["dataset"] == "train") & frame["group_name"].eq(group)].copy()
        group_holdout = frame[(frame["dataset"] == "holdout") & frame["group_name"].eq(group)].copy()
        imputation_models, fill_values = fit_imputation_models(group_train)
        group_train = impute_group_sources(group_train, imputation_models, fill_values)
        group_holdout = impute_group_sources(group_holdout, imputation_models, fill_values)

        summary_lines.append(f"## {group}")
        summary_lines.append(
            "- Missing source ratings are imputed from the other available source ratings "
            "within the same category family; if no regression is available, fall back to "
            f"family medians `{fill_values.to_dict()}`"
        )

        group_train_features = (
            group_train[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
        )
        group_holdout_features = (
            group_holdout[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
        )

        group_holdout_out = group_holdout.copy()
        for target_col, target_label in TARGET_SPECS:
            fit_train = group_train[group_train[target_col].notna()].copy()
            if len(fit_train) < 8:
                summary_lines.append(f"- {target_label}: not enough train rows.")
                group_holdout_out[f"pred_{target_col}"] = np.nan
                continue

            X_train = (
                fit_train[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(fill_values).to_numpy()
            )
            y_train = pd.to_numeric(fit_train[target_col], errors="coerce").to_numpy()
            model = Ridge(alpha=1.0).fit(X_train, y_train)

            train_pred = model.predict(X_train)
            holdout_eval = group_holdout[group_holdout[target_col].notna()].copy()
            X_holdout_eval = (
                holdout_eval[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(fill_values).to_numpy()
            )
            holdout_pred = model.predict(X_holdout_eval) if len(holdout_eval) else np.array([])

            train_r = correlation(train_pred, y_train)
            holdout_r = correlation(
                holdout_pred,
                pd.to_numeric(holdout_eval[target_col], errors="coerce").to_numpy(),
            )

            full_holdout_pred = model.predict(group_holdout_features.to_numpy()) if len(group_holdout_out) else np.array([])
            group_holdout_out[f"pred_{target_col}"] = full_holdout_pred

            coefficient_rows.append(
                {
                    "group_name": group,
                    "target": target_label,
                    "intercept": float(model.intercept_),
                    "goodreads_rating_raw": float(model.coef_[0]),
                    "openlibrary_rating_raw": float(model.coef_[1]),
                    "amazon_rating_raw": float(model.coef_[2]),
                    "fill_goodreads": float(fill_values["goodreads_rating_raw"]),
                    "fill_openlibrary": float(fill_values["openlibrary_rating_raw"]),
                    "fill_amazon": float(fill_values["amazon_rating_raw"]),
                    "train_n": int(len(fit_train)),
                    "holdout_n": int(len(holdout_eval)),
                    "train_r": train_r,
                    "holdout_r": holdout_r,
                }
            )
            summary_lines.extend(
                [
                    f"- {target_label}: train `R={train_r:.3f}` (`n={len(fit_train)}`), "
                    f"holdout `R={holdout_r:.3f}` (`n={len(holdout_eval)}`)",
                    f"  - Formula: `{model.intercept_:.4f} + "
                    f"{model.coef_[0]:.4f}*GR + {model.coef_[1]:.4f}*OL + {model.coef_[2]:.4f}*AMZ`",
                ]
            )

        if len(group_holdout_out):
            group_holdout_out["pred_balanced"] = (
                group_holdout_out["pred_avg_enjoyment"] + group_holdout_out["pred_avg_usefulness"]
            ) / 2.0
            holdout_scored_parts.append(group_holdout_out)
        summary_lines.append("")

    holdout_ranked = pd.concat(holdout_scored_parts, ignore_index=True) if holdout_scored_parts else pd.DataFrame()
    coef_df = pd.DataFrame(coefficient_rows)
    return frame, holdout_ranked, coef_df, "\n".join(summary_lines) + "\n"


def add_keep_flags(holdout_ranked: pd.DataFrame) -> pd.DataFrame:
    scored = holdout_ranked.sort_values(["pred_balanced", "title"], ascending=[False, True]).reset_index(drop=True)
    n = len(scored)
    for keep_share in KEEP_SHARES:
        keep_n = max(1, int(round(n * keep_share)))
        flag_name = f"keep_top_{int(keep_share * 100)}pct"
        scored[flag_name] = False
        scored.loc[: keep_n - 1, flag_name] = True
        scored[f"dropped_top_{int(keep_share * 100)}pct"] = ~scored[flag_name]
    return scored


def aggregate_changes(holdout_ranked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baseline_enjoy = float(pd.to_numeric(holdout_ranked["avg_enjoyment"], errors="coerce").mean())
    baseline_useful = float(pd.to_numeric(holdout_ranked["avg_usefulness"], errors="coerce").mean())
    baseline_bal = float(
        (
            pd.to_numeric(holdout_ranked["avg_enjoyment"], errors="coerce")
            + pd.to_numeric(holdout_ranked["avg_usefulness"], errors="coerce")
        ).mean()
        / 2.0
    )
    rows.append(
        {
            "keep_share": 1.0,
            "keep_n": len(holdout_ranked),
            "avg_enjoyment": baseline_enjoy,
            "avg_usefulness": baseline_useful,
            "avg_balanced": baseline_bal,
            "enjoyment_gain": 0.0,
            "usefulness_gain": 0.0,
            "balanced_gain": 0.0,
        }
    )
    for keep_share in KEEP_SHARES:
        flag_name = f"keep_top_{int(keep_share * 100)}pct"
        kept = holdout_ranked[holdout_ranked[flag_name]].copy()
        avg_enjoy = float(pd.to_numeric(kept["avg_enjoyment"], errors="coerce").mean())
        avg_useful = float(pd.to_numeric(kept["avg_usefulness"], errors="coerce").mean())
        avg_bal = float(
            (
                pd.to_numeric(kept["avg_enjoyment"], errors="coerce")
                + pd.to_numeric(kept["avg_usefulness"], errors="coerce")
            ).mean()
            / 2.0
        )
        rows.append(
            {
                "keep_share": keep_share,
                "keep_n": len(kept),
                "avg_enjoyment": avg_enjoy,
                "avg_usefulness": avg_useful,
                "avg_balanced": avg_bal,
                "enjoyment_gain": avg_enjoy - baseline_enjoy,
                "usefulness_gain": avg_useful - baseline_useful,
                "balanced_gain": avg_bal - baseline_bal,
            }
        )
    return pd.DataFrame(rows)


def dropped_books_by_level(holdout_ranked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keep_share in KEEP_SHARES:
        keep_pct = int(keep_share * 100)
        flag_name = f"dropped_top_{keep_pct}pct"
        dropped = holdout_ranked[holdout_ranked[flag_name]].copy()
        for _, row in dropped.iterrows():
            rows.append(
                {
                    "keep_share": keep_share,
                    "keep_pct": keep_pct,
                    "group_name": row["group_name"],
                    "title": row["title"],
                    "pred_avg_enjoyment": row["pred_avg_enjoyment"],
                    "pred_avg_usefulness": row["pred_avg_usefulness"],
                    "pred_balanced": row["pred_balanced"],
                    "avg_enjoyment": row["avg_enjoyment"],
                    "avg_usefulness": row["avg_usefulness"],
                    "issues": row.get("issues", ""),
                }
            )
    return pd.DataFrame(rows)


def kept_books_by_level(holdout_ranked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keep_share in KEEP_SHARES:
        keep_pct = int(keep_share * 100)
        flag_name = f"keep_top_{keep_pct}pct"
        kept = holdout_ranked[holdout_ranked[flag_name]].copy()
        for _, row in kept.iterrows():
            rows.append(
                {
                    "keep_share": keep_share,
                    "keep_pct": keep_pct,
                    "group_name": row["group_name"],
                    "title": row["title"],
                    "pred_avg_enjoyment": row["pred_avg_enjoyment"],
                    "pred_avg_usefulness": row["pred_avg_usefulness"],
                    "pred_balanced": row["pred_balanced"],
                    "avg_enjoyment": row["avg_enjoyment"],
                    "avg_usefulness": row["avg_usefulness"],
                    "issues": row.get("issues", ""),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    _frame, holdout_ranked, coef_df, summary_md = fit_group_rules(load_frame())
    holdout_ranked = add_keep_flags(holdout_ranked)
    aggregate_df = aggregate_changes(holdout_ranked)
    dropped_df = dropped_books_by_level(holdout_ranked)
    kept_df = kept_books_by_level(holdout_ranked)

    coef_df.to_csv(COEF_CSV, index=False)
    holdout_ranked.to_csv(RANKED_CSV, index=False)
    aggregate_df.to_csv(AGGREGATE_CSV, index=False)
    dropped_df.to_csv(DROPPED_CSV, index=False)
    kept_df.to_csv(KEPT_CSV, index=False)
    SUMMARY_MD.write_text(summary_md)

    print(f"Saved {COEF_CSV.name}")
    print(f"Saved {RANKED_CSV.name}")
    print(f"Saved {AGGREGATE_CSV.name}")
    print(f"Saved {DROPPED_CSV.name}")
    print(f"Saved {KEPT_CSV.name}")
    print(f"Saved {SUMMARY_MD.name}")


if __name__ == "__main__":
    main()
