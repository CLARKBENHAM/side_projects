"""Evaluate actionable future book-rating forecasts with a 2025 holdout."""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_books_tracking.goodreads_followup_analysis import (
    PRIMARY_TARGET_SPECS,
    derive_analysis_columns,
    load_data,
    spearman_summary,
    transform_target_utility,
    utility_transform_label,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = Path(__file__).parent
RESULTS_CSV = OUTPUT_DIR / "future_prediction_2025_results.csv"
BEST_CSV = OUTPUT_DIR / "future_prediction_2025_best.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "future_prediction_2025_predictions.csv"
ROLLING_PREDICTIONS_CSV = OUTPUT_DIR / "future_prediction_rolling_predictions.csv"
TRAIN_END_YEAR = 2024
TEST_YEAR = 2025
ROLLING_YEARS = (2020, 2021, 2022, 2023, 2024)
BALANCED_KEEP_SHARE_RANGE = (0.25, 0.75)
MIN_POLICY_SIDE_N = 15
MIN_TRAIN_ROWS = 25
PREDICTION_MIN = 1.0
PREDICTION_MAX = 5.0
THRESHOLDS = [round(value, 1) for value in np.arange(1.0, 5.01, 0.1)]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    include_goodreads: str | None = None
    include_postread: bool = False
    include_source: bool = False
    actionable: bool = True


FEATURE_SPECS = (
    FeatureSpec("preread_base"),
    FeatureSpec("preread_plus_goodreads_raw", include_goodreads="raw_best"),
    FeatureSpec(
        "preread_plus_goodreads_conservative", include_goodreads="conservative"
    ),
    FeatureSpec(
        "legacy_postread_plus_source",
        include_goodreads="raw_best",
        include_postread=True,
        include_source=True,
        actionable=False,
    ),
)


def ridge_model():
    return Ridge(alpha=10.0)


def lasso_model():
    return Lasso(alpha=0.05)


def random_forest_model():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=4,
        random_state=42,
    )


def gbm_model():
    return GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )


MODEL_BUILDERS = {
    "Category mean": None,
    "Global mean": None,
    "Ridge": ridge_model,
    "Lasso": lasso_model,
    "Random Forest": random_forest_model,
    "GBM": gbm_model,
}


def add_author_history_features(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    other = other_df.copy()

    target = pd.to_numeric(train[target_col], errors="coerce")
    global_mean = float(target.mean())
    valid_author = train["canonical_author"].ne("")
    valid_target = target.notna()
    valid = valid_author & valid_target

    author_counts = train.loc[valid_author, "canonical_author"].value_counts()
    author_sums = train.loc[valid].groupby("canonical_author")[target_col].sum()
    author_means = author_sums / author_counts.reindex(author_sums.index)

    train["author_target_mean_hist"] = global_mean
    train["author_book_count_hist"] = (
        train["canonical_author"].map(author_counts).fillna(0).astype(float)
    )

    repeat_mask = valid & train["author_book_count_hist"].gt(1)
    if repeat_mask.any():
        repeat_authors = train.loc[repeat_mask, "canonical_author"]
        train.loc[repeat_mask, "author_target_mean_hist"] = (
            author_sums.loc[repeat_authors].to_numpy()
            - pd.to_numeric(
                train.loc[repeat_mask, target_col], errors="coerce"
            ).to_numpy()
        ) / (train.loc[repeat_mask, "author_book_count_hist"].to_numpy() - 1)

    other["author_target_mean_hist"] = (
        other["canonical_author"].map(author_means).fillna(global_mean).astype(float)
    )
    other["author_book_count_hist"] = (
        other["canonical_author"].map(author_counts).fillna(0).astype(float)
    )
    return train, other


def add_goodreads_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rating_col: str,
    count_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    for frame in (train, test):
        frame["goodreads_available"] = frame[rating_col].notna().astype(float)
        frame["goodreads_rating_feature"] = pd.to_numeric(
            frame[rating_col], errors="coerce"
        )
        frame["goodreads_log_count_feature"] = np.log10(
            1 + pd.to_numeric(frame[count_col], errors="coerce")
        )

    return train, test


def prepare_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train, test = add_author_history_features(train_df, test_df, target_col)

    numeric_features = [
        "year_finished",
        "log_pages",
        "book_age",
        "author_target_mean_hist",
        "author_book_count_hist",
    ]
    categorical_features = ["Bookshelf"]

    if spec.include_goodreads == "raw_best":
        train, test = add_goodreads_features(
            train,
            test,
            rating_col="goodreads_rating_raw_best",
            count_col="goodreads_rating_count_raw_best",
        )
        numeric_features.extend(
            [
                "goodreads_available",
                "goodreads_rating_feature",
                "goodreads_log_count_feature",
            ]
        )
    elif spec.include_goodreads == "conservative":
        train, test = add_goodreads_features(
            train,
            test,
            rating_col="goodreads_rating",
            count_col="goodreads_rating_count",
        )
        numeric_features.extend(
            [
                "goodreads_available",
                "goodreads_rating_feature",
                "goodreads_log_count_feature",
            ]
        )

    if spec.include_postread:
        numeric_features.extend(["reading_days", "note_length"])

    if spec.include_source and "inferred_source" in train.columns:
        categorical_features.append("inferred_source")

    for column in numeric_features:
        train[column] = pd.to_numeric(train[column], errors="coerce")
        median = float(train[column].median()) if train[column].notna().any() else 0.0
        train[column] = train[column].fillna(median)
        test[column] = pd.to_numeric(test[column], errors="coerce").fillna(median)

    for column in categorical_features:
        train[column] = train[column].fillna("Unknown")
        test[column] = test[column].fillna("Unknown")

    return train, test, numeric_features, categorical_features


def make_pipeline(
    model,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.clip(predictions, PREDICTION_MIN, PREDICTION_MAX)


def baseline_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    model_name: str,
) -> np.ndarray:
    target = pd.to_numeric(train_df[target_col], errors="coerce")
    global_mean = float(target.mean())
    if model_name == "Global mean":
        return np.full(len(test_df), global_mean)

    grouped = train_df.groupby("Bookshelf")[target_col].mean()
    predictions = test_df["Bookshelf"].map(grouped).fillna(global_mean).to_numpy()
    return clip_predictions(predictions)


def model_predictions_for_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    spec: FeatureSpec,
    model_name: str,
) -> np.ndarray:
    if model_name in {"Global mean", "Category mean"}:
        return baseline_predictions(train_df, test_df, target_col, model_name)

    builder = MODEL_BUILDERS[model_name]
    model = builder()
    train_frame, test_frame, numeric_features, categorical_features = (
        prepare_feature_frames(train_df, test_df, target_col, spec)
    )
    pipeline = make_pipeline(model, numeric_features, categorical_features)
    y_train = pd.to_numeric(train_frame[target_col], errors="coerce").to_numpy()
    pipeline.fit(train_frame, y_train)
    predictions = pipeline.predict(test_frame)
    return clip_predictions(predictions)


def evaluate_year_split(
    df: pd.DataFrame,
    target_col: str,
    spec: FeatureSpec,
    model_name: str,
    test_year: int,
    prediction_split: str,
) -> pd.DataFrame:
    train_df = df[(df["year_finished"] < test_year) & df[target_col].notna()].copy()
    test_df = df[(df["year_finished"] == test_year) & df[target_col].notna()].copy()

    if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
        return pd.DataFrame()

    predictions = model_predictions_for_split(
        train_df, test_df, target_col, spec, model_name
    )
    selected_columns = list(
        dict.fromkeys(
            [
                "title",
                "Bookshelf",
                "year_finished",
                target_col,
                "avg_enjoyment",
                "avg_usefulness",
            ]
        )
    )
    result = test_df[selected_columns].copy()
    result["target"] = target_col
    result["feature_spec"] = spec.name
    result["actionable"] = spec.actionable
    result["model"] = model_name
    result["eval_year"] = test_year
    result["prediction_split"] = prediction_split
    result["prediction"] = predictions
    result["prediction_error"] = predictions - pd.to_numeric(
        result[target_col], errors="coerce"
    )
    return result


def collect_rolling_predictions(
    df: pd.DataFrame,
    target_col: str,
    spec: FeatureSpec,
    model_name: str,
) -> pd.DataFrame:
    parts = [
        evaluate_year_split(
            df,
            target_col,
            spec,
            model_name,
            year,
            prediction_split="rolling_history",
        )
        for year in ROLLING_YEARS
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def summarize_prediction_metrics(
    predictions: pd.DataFrame, target_col: str
) -> dict[str, float]:
    actual = pd.to_numeric(predictions[target_col], errors="coerce")
    predicted = pd.to_numeric(predictions["prediction"], errors="coerce")
    return {
        "n": float(len(predictions)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "spearman_rho": spearman_summary(predicted, actual)[0],
        "spearman_p": spearman_summary(predicted, actual)[1],
    }


def threshold_policy_summary(
    predictions: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions[["prediction", target_col]].dropna().copy()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna()
    overall_mean = float(frame[target_col].mean())
    overall_utility_mean = float(
        transform_target_utility(frame[target_col], target_col).mean()
    )

    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        keep = frame[frame["prediction"] >= threshold].copy()
        skip = frame[frame["prediction"] < threshold].copy()
        if len(keep) < MIN_POLICY_SIDE_N or len(skip) < MIN_POLICY_SIDE_N:
            continue
        keep_utility = transform_target_utility(keep[target_col], target_col)
        skip_utility = transform_target_utility(skip[target_col], target_col)
        rows.append(
            {
                "threshold": threshold,
                "n_total": len(frame),
                "n_keep": len(keep),
                "n_skip": len(skip),
                "keep_share": len(keep) / len(frame),
                "keep_mean": float(keep[target_col].mean()),
                "skip_mean": float(skip[target_col].mean()),
                "delta_keep_vs_all": float(keep[target_col].mean() - overall_mean),
                "delta_keep_vs_skip": float(
                    keep[target_col].mean() - skip[target_col].mean()
                ),
                "utility_transform": utility_transform_label(target_col),
                "keep_utility_mean": float(keep_utility.mean()),
                "skip_utility_mean": float(skip_utility.mean()),
                "delta_keep_utility_vs_all": float(
                    keep_utility.mean() - overall_utility_mean
                ),
                "delta_keep_utility_vs_skip": float(
                    keep_utility.mean() - skip_utility.mean()
                ),
            }
        )

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep, pd.DataFrame()

    best_rows: list[dict[str, object]] = []
    best_rows.append(
        {
            "selection": "best_overall_utility",
            **sweep.sort_values("delta_keep_utility_vs_all", ascending=False)
            .iloc[0]
            .to_dict(),
        }
    )
    balanced = sweep[
        sweep["keep_share"].between(
            BALANCED_KEEP_SHARE_RANGE[0], BALANCED_KEEP_SHARE_RANGE[1]
        )
    ].copy()
    if not balanced.empty:
        best_rows.append(
            {
                "selection": "best_balanced_utility",
                **balanced.sort_values("delta_keep_utility_vs_all", ascending=False)
                .iloc[0]
                .to_dict(),
            }
        )

    return sweep, pd.DataFrame(best_rows)


def apply_policy_to_predictions(
    predictions: pd.DataFrame,
    target_col: str,
    threshold: float,
    selection: str,
) -> dict[str, object]:
    frame = predictions[["prediction", target_col]].dropna().copy()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = frame.dropna()
    keep = frame[frame["prediction"] >= threshold].copy()
    overall_mean = float(frame[target_col].mean())
    overall_utility_mean = float(
        transform_target_utility(frame[target_col], target_col).mean()
    )

    if keep.empty:
        return {
            "selection": selection,
            "policy_threshold": threshold,
            "policy_keep_n": 0,
            "policy_keep_share": 0.0,
            "policy_keep_mean": math.nan,
            "policy_delta_keep_vs_all": math.nan,
            "policy_keep_utility_mean": math.nan,
            "policy_delta_keep_utility_vs_all": math.nan,
        }

    keep_utility = transform_target_utility(keep[target_col], target_col)
    return {
        "selection": selection,
        "policy_threshold": threshold,
        "policy_keep_n": len(keep),
        "policy_keep_share": len(keep) / len(frame),
        "policy_keep_mean": float(keep[target_col].mean()),
        "policy_delta_keep_vs_all": float(keep[target_col].mean() - overall_mean),
        "policy_keep_utility_mean": float(keep_utility.mean()),
        "policy_delta_keep_utility_vs_all": float(
            keep_utility.mean() - overall_utility_mean
        ),
    }


def evaluate_configuration(
    df: pd.DataFrame,
    target_col: str,
    spec: FeatureSpec,
    model_name: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    rolling_predictions = collect_rolling_predictions(df, target_col, spec, model_name)
    if rolling_predictions.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    expected_metrics = summarize_prediction_metrics(rolling_predictions, target_col)
    _, policy_best = threshold_policy_summary(rolling_predictions, target_col)

    test_predictions = evaluate_year_split(
        df,
        target_col,
        spec,
        model_name,
        TEST_YEAR,
        prediction_split="test_2025",
    )
    if test_predictions.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    test_metrics = summarize_prediction_metrics(test_predictions, target_col)

    row: dict[str, object] = {
        "target": target_col,
        "feature_spec": spec.name,
        "actionable": spec.actionable,
        "model": model_name,
        "train_end_year": TRAIN_END_YEAR,
        "test_year": TEST_YEAR,
        "rolling_years": ",".join(str(year) for year in ROLLING_YEARS),
        "expected_n": expected_metrics["n"],
        "expected_mae": expected_metrics["mae"],
        "expected_rmse": expected_metrics["rmse"],
        "expected_spearman_rho": expected_metrics["spearman_rho"],
        "expected_spearman_p": expected_metrics["spearman_p"],
        "test_n": test_metrics["n"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_spearman_rho": test_metrics["spearman_rho"],
        "test_spearman_p": test_metrics["spearman_p"],
        "test_overall_mean": float(pd.to_numeric(test_predictions[target_col]).mean()),
        "test_overall_utility_mean": float(
            transform_target_utility(test_predictions[target_col], target_col).mean()
        ),
    }

    for selection in ["best_overall_utility", "best_balanced_utility"]:
        selected = policy_best[policy_best["selection"] == selection]
        if selected.empty:
            continue
        threshold = float(selected.iloc[0]["threshold"])
        row[f"{selection}_threshold"] = threshold
        row[f"{selection}_expected_keep_share"] = float(selected.iloc[0]["keep_share"])
        row[f"{selection}_expected_uplift"] = float(
            selected.iloc[0]["delta_keep_vs_all"]
        )
        row[f"{selection}_expected_utility_uplift"] = float(
            selected.iloc[0]["delta_keep_utility_vs_all"]
        )
        realized = apply_policy_to_predictions(
            test_predictions,
            target_col=target_col,
            threshold=threshold,
            selection=selection,
        )
        row[f"{selection}_test_keep_n"] = realized["policy_keep_n"]
        row[f"{selection}_test_keep_share"] = realized["policy_keep_share"]
        row[f"{selection}_test_keep_mean"] = realized["policy_keep_mean"]
        row[f"{selection}_test_uplift"] = realized["policy_delta_keep_vs_all"]
        row[f"{selection}_test_keep_utility_mean"] = realized[
            "policy_keep_utility_mean"
        ]
        row[f"{selection}_test_utility_uplift"] = realized[
            "policy_delta_keep_utility_vs_all"
        ]

    for frame in (rolling_predictions, test_predictions):
        frame["feature_spec"] = spec.name
        frame["actionable"] = spec.actionable
        frame["model"] = model_name
        frame["target"] = target_col

    return row, rolling_predictions, test_predictions


def summarize_best(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in results["target"].unique():
        sub = results[results["target"] == target].copy()
        actionable = sub[sub["actionable"]].copy()
        if not actionable.empty:
            rows.append(
                {
                    "target": target,
                    "selection": "best_expected_mae_actionable",
                    **actionable.sort_values("expected_mae").iloc[0].to_dict(),
                }
            )
            rows.append(
                {
                    "target": target,
                    "selection": "best_2025_mae_actionable",
                    **actionable.sort_values("test_mae").iloc[0].to_dict(),
                }
            )
            rows.append(
                {
                    "target": target,
                    "selection": "best_2025_balanced_utility_uplift_actionable",
                    **actionable.sort_values(
                        "best_balanced_utility_test_utility_uplift", ascending=False
                    )
                    .iloc[0]
                    .to_dict(),
                }
            )
        rows.append(
            {
                "target": target,
                "selection": "best_2025_mae_any",
                **sub.sort_values("test_mae").iloc[0].to_dict(),
            }
        )
    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame, best: pd.DataFrame) -> None:
    print("=" * 80)
    print("FUTURE PREDICTION EVALUATION")
    print("=" * 80)

    actionable = results[results["actionable"]].copy()
    for target in actionable["target"].unique():
        sub = actionable[actionable["target"] == target].copy()
        print(f"\nTarget: {target}")
        print(
            sub[
                [
                    "feature_spec",
                    "model",
                    "expected_mae",
                    "test_mae",
                    "expected_spearman_rho",
                    "test_spearman_rho",
                    "best_balanced_utility_threshold",
                    "best_balanced_utility_test_keep_n",
                    "best_balanced_utility_test_uplift",
                    "best_balanced_utility_test_utility_uplift",
                ]
            ]
            .sort_values(["test_mae", "expected_mae"])
            .to_string(index=False)
        )

    if not best.empty:
        print("\nBest rows:")
        print(
            best[
                [
                    "target",
                    "selection",
                    "feature_spec",
                    "model",
                    "expected_mae",
                    "test_mae",
                    "best_balanced_utility_threshold",
                    "best_balanced_utility_test_uplift",
                    "best_balanced_utility_test_utility_uplift",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    df = derive_analysis_columns(load_data())
    df = df[df["year_finished"].between(2018, TEST_YEAR, inclusive="both")].copy()

    result_rows: list[dict[str, object]] = []
    rolling_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for spec in FEATURE_SPECS:
        for model_name in MODEL_BUILDERS:
            if (
                model_name in {"Category mean", "Global mean"}
                and spec != FEATURE_SPECS[0]
            ):
                continue
            for target_spec in PRIMARY_TARGET_SPECS:
                row, rolling_predictions, test_predictions = evaluate_configuration(
                    df,
                    target_col=target_spec.column,
                    spec=spec,
                    model_name=model_name,
                )
                if not row:
                    continue
                result_rows.append(row)
                rolling_parts.append(rolling_predictions)
                test_parts.append(test_predictions)

    results = pd.DataFrame(result_rows).sort_values(
        ["target", "actionable", "test_mae", "expected_mae", "feature_spec", "model"],
        ascending=[True, False, True, True, True, True],
    )
    results.to_csv(RESULTS_CSV, index=False)

    predictions = pd.concat(test_parts + rolling_parts, ignore_index=True)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    rolling = pd.concat(rolling_parts, ignore_index=True)
    rolling.to_csv(ROLLING_PREDICTIONS_CSV, index=False)

    best = summarize_best(results)
    best.to_csv(BEST_CSV, index=False)
    print_summary(results, best)


if __name__ == "__main__":
    main()
