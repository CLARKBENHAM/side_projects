from __future__ import annotations

import html
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "data" / "mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_core.calendar_movie_scores import LOCAL_TZ, REQUEST_HEADERS
from analysis_core.calendar_movie_scores import parse_calendar_directory
from analysis_core.movie_rt_analysis import LIKE_THRESHOLD
from analysis_core.movie_rt_followup_analysis import personal_bucket_5

SUMMARY_DESCRIPTION_PATTERN = re.compile(
    r'<meta\s+property="og:description"\s+content="(?P<body>[^"]+)"',
    re.IGNORECASE,
)
JSON_LD_DESCRIPTION_PATTERN = re.compile(
    r'"description":"(?P<body>.*?)"',
    re.DOTALL,
)
WIKIPEDIA_DISAMBIGUATION_MARKERS = (
    "may refer to",
    "can refer to",
)


@dataclass(frozen=True)
class LifePeriod:
    name: str
    start: str
    end: str | None
    pressure_tier: str


LIFE_PERIODS = [
    LifePeriod("pre_hive", "1900-01-01", "2021-06-06", "low"),
    LifePeriod("early_hive", "2021-06-07", "2022-08-21", "high"),
    LifePeriod("hive_research", "2022-08-22", "2023-07-22", "medium"),
    LifePeriod("summer23_idle", "2023-07-23", "2023-09-17", "low"),
    LifePeriod("ai_safety", "2023-09-18", "2024-05-20", "medium"),
    LifePeriod("transition_2024", "2024-05-21", "2024-06-16", "low"),
    LifePeriod("mats", "2024-06-17", "2024-08-25", "high"),
    LifePeriod("mope_jobs", "2024-08-26", "2025-02-03", "low"),
    LifePeriod("hadrian_vllm", "2025-02-04", "2025-07-27", "medium"),
    LifePeriod("china_transition", "2025-07-28", "2025-08-10", "low"),
    LifePeriod("diesl", "2025-08-11", "2025-12-07", "high"),
    LifePeriod("post_diesl_transition", "2025-12-08", "2026-01-18", "low"),
    LifePeriod("misc_job_analysis", "2026-01-19", None, "low"),
]

CONTEXT_NUMERIC_FEATURES = [
    "rt_critic_on_10",
    "rt_audience_on_10",
    "imdb_score",
    "start_hour",
    "month_index",
]
CONTEXT_BINARY_FEATURES = [
    "saw_in_theater",
    "drink_before_movie",
    "assigned_unfinished_two",
    "likely_with_amelia",
    "is_weekend",
]
CONTEXT_CATEGORICAL_FEATURES = [
    "life_period",
    "pressure_tier",
    "data_source",
]


def parse_bool(value: Any) -> bool:
    lowered = str(value).strip().lower()
    return lowered in {"1", "t", "true", "y", "yes"}


def optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def merge_key(date_value: Any, normalized_title: Any) -> str:
    return f"{date_value}|{normalized_title}"


def assign_life_period(date_value: pd.Timestamp) -> tuple[str, str]:
    current_date = date_value.date()
    for period in LIFE_PERIODS:
        start = pd.Timestamp(period.start).date()
        end = pd.Timestamp(period.end).date() if period.end else None
        if current_date >= start and (end is None or current_date <= end):
            return period.name, period.pressure_tier
    return "unknown", "low"


def describe_same_day_amelia_context(
    datetimes: pd.Series,
    *,
    calendar_dir: Path,
    as_of_local: datetime,
) -> pd.Series:
    events = parse_calendar_directory(calendar_dir, as_of_local=as_of_local)
    amelia_events = [
        event
        for event in events
        if "amelia"
        in f"{event.calendar_name} {event.summary} {event.description} {event.location}".lower()
    ]
    result: list[bool] = []
    for datetime_value in datetimes:
        if pd.isna(datetime_value):
            result.append(False)
            continue
        local_datetime = datetime_value.to_pydatetime()
        has_amelia = any(
            event.start_local.date() == local_datetime.date()
            and abs((event.start_local - local_datetime).total_seconds()) <= 8 * 3600
            for event in amelia_events
        )
        result.append(has_amelia)
    return pd.Series(result, index=datetimes.index, dtype=bool)


def load_legacy_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["date"] = pd.NaT
    df["datetime_local"] = pd.NaT
    df["normalized_title"] = ""
    df["watch_status"] = "finished"
    df["where_seen"] = "unknown"
    df["saw_in_home"] = False
    df["saw_in_theater"] = False
    df["drink_before_movie"] = False
    df["assigned_unfinished_two"] = False
    df["completion_basis"] = "legacy_notes"
    df["completion_confidence"] = "high"
    df["calendar_summary"] = ""
    df["calendar_location"] = ""
    df["imdb_score"] = np.nan
    df["imdb_release_year"] = np.nan
    df["imdb_url"] = ""
    df["likely_with_amelia"] = False
    df["life_period"] = "legacy_notes"
    df["pressure_tier"] = "legacy_notes"
    df["is_weekend"] = False
    df["start_hour"] = np.nan
    df["month_index"] = np.nan
    df["data_source"] = "legacy_notes"
    df["analysis_rating"] = df["my_rating"].astype(float)
    df["liked"] = df["analysis_rating"] >= LIKE_THRESHOLD
    df["rt_average_rating"] = df[["rt_audience_rating", "rt_critic_rating"]].mean(
        axis=1
    )
    df["rt_audience_on_10"] = df["rt_audience_rating"] / 10.0
    df["rt_critic_on_10"] = df["rt_critic_rating"] / 10.0
    df["rt_average_on_10"] = df["rt_average_rating"] / 10.0
    df["external_mean_on_10"] = df["rt_average_on_10"]
    df["analysis_row_key"] = "legacy|" + df["watch_order"].astype(str)
    return df


def load_second_sheet_dataset(
    path: Path,
    *,
    source_name: str,
    include_unfinished_as_two: bool,
    calendar_dir: Path,
    as_of_local: datetime,
) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["datetime_local"] = pd.to_datetime(
        df["datetime_local"], errors="coerce", utc=True
    )
    if df["datetime_local"].notna().any():
        df["datetime_local"] = df["datetime_local"].dt.tz_convert(LOCAL_TZ)
    df["rating_observed"] = pd.to_numeric(df["clark_rating"], errors="coerce")
    if "watch_status" in df.columns:
        df["watch_status"] = df["watch_status"].fillna("finished")
    else:
        df["watch_status"] = "finished"
    df["assigned_unfinished_two"] = False
    if include_unfinished_as_two:
        unfinished_mask = df["rating_observed"].isna() & (
            df["watch_status"] == "unfinished"
        )
        df.loc[unfinished_mask, "rating_observed"] = 2.0
        df.loc[unfinished_mask, "assigned_unfinished_two"] = True
    df = df.loc[df["rating_observed"].notna()].copy()
    df["analysis_rating"] = df["rating_observed"].astype(float)
    df["liked"] = df["analysis_rating"] >= LIKE_THRESHOLD
    for column in (
        "drink_before_movie",
        "saw_in_home",
        "saw_in_theater",
    ):
        if column in df.columns:
            df[column] = df[column].map(parse_bool)
        else:
            df[column] = False
    df["saw_in_theater"] = df.get("saw_in_theater", False) | (
        df["where_seen"].fillna("").str.lower() == "theater"
    )
    df["saw_in_home"] = df.get("saw_in_home", False) | (
        df["where_seen"].fillna("").str.lower() == "home"
    )
    df["rt_audience_rating"] = pd.to_numeric(df["rt_audience_score"], errors="coerce")
    df["rt_critic_rating"] = pd.to_numeric(df["rt_critic_score"], errors="coerce")
    df["rt_average_rating"] = df[["rt_audience_rating", "rt_critic_rating"]].mean(
        axis=1
    )
    df["rt_audience_on_10"] = df["rt_audience_rating"] / 10.0
    df["rt_critic_on_10"] = df["rt_critic_rating"] / 10.0
    df["rt_average_on_10"] = df["rt_average_rating"] / 10.0
    df["imdb_score"] = pd.to_numeric(df["imdb_score"], errors="coerce")
    df["imdb_release_year"] = pd.to_numeric(df["imdb_release_year"], errors="coerce")
    df["external_mean_on_10"] = df[
        ["rt_audience_on_10", "rt_critic_on_10", "imdb_score"]
    ].mean(axis=1)
    df["life_period"] = df["date"].map(
        lambda value: assign_life_period(value)[0] if not pd.isna(value) else "unknown"
    )
    df["pressure_tier"] = df["date"].map(
        lambda value: assign_life_period(value)[1] if not pd.isna(value) else "low"
    )
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["start_hour"] = (
        df["datetime_local"].dt.hour + df["datetime_local"].dt.minute / 60.0
    )
    first_date = df["date"].min()
    df["month_index"] = (df["date"].dt.year - first_date.year) * 12 + (
        df["date"].dt.month - first_date.month
    )
    df["data_source"] = source_name
    df["analysis_row_key"] = [
        merge_key(
            row["date"].date().isoformat() if not pd.isna(row["date"]) else "",
            row["normalized_title"],
        )
        for _, row in df.iterrows()
    ]
    df["likely_with_amelia"] = describe_same_day_amelia_context(
        df["datetime_local"],
        calendar_dir=calendar_dir,
        as_of_local=as_of_local,
    )
    return df


def evaluate_numeric_predictions(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    y_true = actual.to_numpy(dtype=float)
    y_pred = predicted.to_numpy(dtype=float)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid_mask.sum() == 0:
        return {
            "rows": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "correlation": float("nan"),
            "mean_residual": float("nan"),
            "within_half": float("nan"),
            "within_one": float("nan"),
            "rounded_accuracy": float("nan"),
            "liked_accuracy": float("nan"),
            "bucket5_accuracy": float("nan"),
            "bucket5_quadratic_kappa": float("nan"),
        }
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    residuals = y_true - y_pred
    rounded_pred = np.clip(np.rint(y_pred), 1, 10)
    actual_buckets = np.array([personal_bucket_5(value) for value in y_true])
    predicted_buckets = np.array([personal_bucket_5(value) for value in rounded_pred])
    return {
        "rows": float(valid_mask.sum()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "correlation": (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if valid_mask.sum() >= 2
            else float("nan")
        ),
        "mean_residual": float(residuals.mean()),
        "within_half": float((np.abs(residuals) <= 0.5).mean()),
        "within_one": float((np.abs(residuals) <= 1.0).mean()),
        "rounded_accuracy": float((rounded_pred == y_true).mean()),
        "liked_accuracy": float(
            ((y_pred >= LIKE_THRESHOLD) == (y_true >= LIKE_THRESHOLD)).mean()
        ),
        "bucket5_accuracy": float(accuracy_score(actual_buckets, predicted_buckets)),
        "bucket5_quadratic_kappa": float(
            cohen_kappa_score(
                actual_buckets,
                predicted_buckets,
                weights="quadratic",
            )
        ),
    }


def evaluate_probability_predictions(
    actual: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float]:
    probability_values = probabilities.to_numpy(dtype=float)
    actual_values = actual.to_numpy(dtype=float)
    valid_mask = np.isfinite(actual_values) & np.isfinite(probability_values)
    if valid_mask.sum() == 0:
        return {
            "rows": 0.0,
            "accuracy": float("nan"),
            "mean_probability": float("nan"),
        }
    y_true = (actual_values[valid_mask] >= LIKE_THRESHOLD).astype(int)
    y_pred = (probability_values[valid_mask] >= 0.5).astype(int)
    return {
        "rows": float(valid_mask.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mean_probability": float(probability_values[valid_mask].mean()),
    }


def fit_full_legacy_models(legacy_df: pd.DataFrame) -> dict[str, Any]:
    feature_columns = ["rt_critic_on_10", "rt_audience_on_10"]
    x = legacy_df[feature_columns].to_numpy(dtype=float)
    y = legacy_df["analysis_rating"].to_numpy(dtype=float)
    linear_model = LinearRegression().fit(x, y)
    logistic_model = LogisticRegression(random_state=0, max_iter=1000).fit(
        x,
        legacy_df["liked"].astype(int).to_numpy(dtype=int),
    )
    return {
        "linear": linear_model,
        "logistic": logistic_model,
    }


def build_rating_preprocessor(
    *,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "binary",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                binary_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_rating_pipeline(
    *,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
    estimator: Any,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_rating_preprocessor(
                    numeric_features=numeric_features,
                    binary_features=binary_features,
                    categorical_features=categorical_features,
                ),
            ),
            ("estimator", estimator),
        ]
    )


def build_pipeline_feature_frame(
    df: pd.DataFrame,
    *,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    feature_frame = df[numeric_features + binary_features + categorical_features].copy()
    for column in binary_features:
        feature_frame[column] = feature_frame[column].astype(float)
    return feature_frame


def apply_saved_legacy_models(
    df: pd.DataFrame,
    *,
    metrics_payload: dict[str, Any],
    full_legacy_models: dict[str, Any],
    average_threshold: float,
) -> pd.DataFrame:
    working = df.copy()
    linear_coefficients = metrics_payload["linear_coefficients"]
    logistic_coefficients = metrics_payload["logistic_coefficients"]
    feature_columns = ["rt_critic_on_10", "rt_audience_on_10"]
    valid_feature_mask = working[feature_columns].notna().all(axis=1)
    working["legacy_average_prediction"] = working["rt_average_on_10"]
    working["legacy_saved_linear_prediction"] = (
        linear_coefficients["intercept"]
        + linear_coefficients["critic_weight"] * working["rt_critic_on_10"]
        + linear_coefficients["audience_weight"] * working["rt_audience_on_10"]
    )
    linear_model = full_legacy_models["linear"]
    logistic_model = full_legacy_models["logistic"]
    working["legacy_refit_linear_prediction"] = np.nan
    if valid_feature_mask.any():
        feature_frame = working.loc[valid_feature_mask, feature_columns].to_numpy(
            dtype=float
        )
        working.loc[valid_feature_mask, "legacy_refit_linear_prediction"] = (
            linear_model.predict(feature_frame)
        )
    logits = (
        logistic_coefficients["intercept"]
        + logistic_coefficients["critic_weight"] * working["rt_critic_on_10"]
        + logistic_coefficients["audience_weight"] * working["rt_audience_on_10"]
    )
    working["legacy_saved_logistic_probability"] = sigmoid(logits)
    working["legacy_refit_logistic_probability"] = np.nan
    if valid_feature_mask.any():
        feature_frame = working.loc[valid_feature_mask, feature_columns].to_numpy(
            dtype=float
        )
        working.loc[valid_feature_mask, "legacy_refit_logistic_probability"] = (
            logistic_model.predict_proba(feature_frame)[:, 1]
        )
    working["legacy_average_threshold_prediction"] = np.where(
        working["rt_average_on_10"].notna(),
        (working["rt_average_on_10"] >= average_threshold).astype(float),
        np.nan,
    )
    return working


def evaluate_legacy_predictions(
    df: pd.DataFrame,
    *,
    dataset_label: str,
) -> pd.DataFrame:
    regression_specs = {
        "legacy_average_rt": "legacy_average_prediction",
        "legacy_saved_linear": "legacy_saved_linear_prediction",
        "legacy_refit_linear": "legacy_refit_linear_prediction",
    }
    rows: list[dict[str, Any]] = []
    for model_name, column in regression_specs.items():
        metrics = evaluate_numeric_predictions(df["analysis_rating"], df[column])
        rows.append(
            {
                "dataset": dataset_label,
                "model": model_name,
                "task": "rating_regression",
                **metrics,
            }
        )
    for model_name, column in {
        "legacy_saved_logistic": "legacy_saved_logistic_probability",
        "legacy_refit_logistic": "legacy_refit_logistic_probability",
        "legacy_average_threshold": "legacy_average_threshold_prediction",
    }.items():
        if column == "legacy_average_threshold_prediction":
            probabilities = pd.Series(df[column].astype(float), index=df.index)
        else:
            probabilities = df[column]
        metrics = evaluate_probability_predictions(df["analysis_rating"], probabilities)
        rows.append(
            {
                "dataset": dataset_label,
                "model": model_name,
                "task": "liked_classification",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_pipeline_predictions(
    df: pd.DataFrame,
    *,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
    estimator: Any,
) -> np.ndarray:
    pipeline = build_rating_pipeline(
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
        estimator=estimator,
    )
    feature_frame = build_pipeline_feature_frame(
        df,
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )
    loo = LeaveOneOut()
    return cross_val_predict(
        pipeline,
        feature_frame,
        df["analysis_rating"],
        cv=loo,
        n_jobs=None,
    )


def fit_rating_pipeline(
    df: pd.DataFrame,
    *,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
    estimator: Any,
) -> Pipeline:
    pipeline = build_rating_pipeline(
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
        estimator=estimator,
    )
    feature_frame = build_pipeline_feature_frame(
        df,
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )
    pipeline.fit(feature_frame, df["analysis_rating"].astype(float))
    return pipeline


def fit_full_cv_ridge_context_model(
    df: pd.DataFrame,
    *,
    extra_numeric_features: list[str] | None = None,
    extra_binary_features: list[str] | None = None,
    extra_categorical_features: list[str] | None = None,
) -> Pipeline:
    return fit_rating_pipeline(
        df,
        numeric_features=CONTEXT_NUMERIC_FEATURES + (extra_numeric_features or []),
        binary_features=CONTEXT_BINARY_FEATURES + (extra_binary_features or []),
        categorical_features=CONTEXT_CATEGORICAL_FEATURES
        + (extra_categorical_features or []),
        estimator=Ridge(alpha=1.0),
    )


def extract_linear_pipeline_coefficients(
    pipeline: Pipeline,
    *,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    estimator = pipeline.named_steps["estimator"]
    if not hasattr(estimator, "coef_"):
        raise TypeError("Estimator does not expose linear coefficients")
    preprocessor = pipeline.named_steps["preprocessor"]
    coefficients = np.asarray(estimator.coef_, dtype=float)
    rows: list[dict[str, Any]] = []

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    numeric_scaler = numeric_pipeline.named_steps["scaler"]
    numeric_means = numeric_scaler.mean_
    numeric_scales = numeric_scaler.scale_
    raw_intercept = float(estimator.intercept_) - float(
        np.sum(coefficients[: len(numeric_features)] * numeric_means / numeric_scales)
    )
    rows.append(
        {
            "feature_group": "intercept",
            "original_feature": "intercept",
            "level": "",
            "transformed_feature": "intercept",
            "pipeline_coefficient": float(estimator.intercept_),
            "raw_unit_coefficient": raw_intercept,
            "feature_mean": "",
            "feature_scale": "",
        }
    )

    offset = 0
    for index, feature_name in enumerate(numeric_features):
        coefficient = float(coefficients[offset + index])
        scale = float(numeric_scales[index])
        rows.append(
            {
                "feature_group": "numeric",
                "original_feature": feature_name,
                "level": "",
                "transformed_feature": f"numeric__{feature_name}",
                "pipeline_coefficient": coefficient,
                "raw_unit_coefficient": coefficient / scale,
                "feature_mean": float(numeric_means[index]),
                "feature_scale": scale,
            }
        )
    offset += len(numeric_features)

    for index, feature_name in enumerate(binary_features):
        coefficient = float(coefficients[offset + index])
        rows.append(
            {
                "feature_group": "binary",
                "original_feature": feature_name,
                "level": "",
                "transformed_feature": f"binary__{feature_name}",
                "pipeline_coefficient": coefficient,
                "raw_unit_coefficient": coefficient,
                "feature_mean": "",
                "feature_scale": "",
            }
        )
    offset += len(binary_features)

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    categorical_encoder = categorical_pipeline.named_steps["encoder"]
    for feature_name, categories in zip(
        categorical_features,
        categorical_encoder.categories_,
        strict=True,
    ):
        for level in categories:
            coefficient = float(coefficients[offset])
            rows.append(
                {
                    "feature_group": "categorical",
                    "original_feature": feature_name,
                    "level": str(level),
                    "transformed_feature": (f"categorical__{feature_name}_{level}"),
                    "pipeline_coefficient": coefficient,
                    "raw_unit_coefficient": coefficient,
                    "feature_mean": "",
                    "feature_scale": "",
                }
            )
            offset += 1
    return pd.DataFrame(rows)


def build_loocv_model_tables(
    df: pd.DataFrame,
    *,
    dataset_label: str,
    include_context: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = df.copy()
    predictions["cv_average_rt_prediction"] = predictions["rt_average_on_10"]
    predictions["cv_linear_rt_prediction"] = build_pipeline_predictions(
        predictions,
        numeric_features=["rt_critic_on_10", "rt_audience_on_10"],
        binary_features=[],
        categorical_features=[],
        estimator=LinearRegression(),
    )
    predictions["cv_linear_rt_imdb_prediction"] = build_pipeline_predictions(
        predictions,
        numeric_features=["rt_critic_on_10", "rt_audience_on_10", "imdb_score"],
        binary_features=[],
        categorical_features=[],
        estimator=LinearRegression(),
    )
    if include_context:
        predictions["cv_ridge_context_prediction"] = build_pipeline_predictions(
            predictions,
            numeric_features=CONTEXT_NUMERIC_FEATURES,
            binary_features=CONTEXT_BINARY_FEATURES,
            categorical_features=CONTEXT_CATEGORICAL_FEATURES,
            estimator=Ridge(alpha=1.0),
        )
    model_columns = [
        ("cv_average_rt", "cv_average_rt_prediction"),
        ("cv_linear_rt", "cv_linear_rt_prediction"),
        ("cv_linear_rt_imdb", "cv_linear_rt_imdb_prediction"),
    ]
    if include_context:
        model_columns.append(("cv_ridge_context", "cv_ridge_context_prediction"))
    rows: list[dict[str, Any]] = []
    for model_name, column in model_columns:
        metrics = evaluate_numeric_predictions(
            predictions["analysis_rating"],
            predictions[column],
        )
        rows.append(
            {
                "dataset": dataset_label,
                "model": model_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows), predictions


def build_gain_curve(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    model_name: str,
    dataset_label: str,
) -> pd.DataFrame:
    working = df[["analysis_rating", prediction_column]].dropna().copy()
    working = working.sort_values(prediction_column, ascending=False).reset_index(
        drop=True
    )
    rows: list[dict[str, Any]] = []
    total_rows = len(working)
    for keep_fraction in np.linspace(0.2, 1.0, 17):
        keep_count = max(1, math.ceil(total_rows * keep_fraction))
        kept = working.head(keep_count)
        rows.append(
            {
                "dataset": dataset_label,
                "model": model_name,
                "keep_fraction": float(keep_fraction),
                "drop_fraction": float(1.0 - keep_fraction),
                "kept_rows": int(keep_count),
                "mean_rating": float(kept["analysis_rating"].mean()),
                "liked_rate": float((kept["analysis_rating"] >= LIKE_THRESHOLD).mean()),
                "gain_vs_all": float(
                    kept["analysis_rating"].mean() - working["analysis_rating"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_score_cutoff_tradeoff(
    df: pd.DataFrame,
    *,
    score_column: str,
    score_label: str,
) -> pd.DataFrame:
    working = df[["analysis_rating", score_column]].dropna().copy()
    working = working.sort_values(score_column, ascending=False).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    total_rows = len(working)
    for keep_fraction in np.linspace(0.2, 1.0, 17):
        keep_count = max(1, math.ceil(total_rows * keep_fraction))
        kept = working.head(keep_count)
        rows.append(
            {
                "score_label": score_label,
                "score_column": score_column,
                "keep_fraction": float(keep_fraction),
                "drop_fraction": float(1.0 - keep_fraction),
                "kept_rows": int(keep_count),
                "score_cutoff": float(kept[score_column].iloc[-1]),
                "mean_rating": float(kept["analysis_rating"].mean()),
                "liked_rate": float((kept["analysis_rating"] >= LIKE_THRESHOLD).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_time_window_metrics(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    dataset_label: str,
    window_months: int = 6,
) -> pd.DataFrame:
    working = df.loc[
        df["date"].notna(), ["date", "analysis_rating", prediction_column]
    ].copy()
    working["window_index"] = (
        (working["date"].dt.year - working["date"].min().year) * 12
        + (working["date"].dt.month - working["date"].min().month)
    ) // window_months
    rows: list[dict[str, Any]] = []
    for window_index, subset in working.groupby("window_index"):
        metrics = evaluate_numeric_predictions(
            subset["analysis_rating"],
            subset[prediction_column],
        )
        rows.append(
            {
                "dataset": dataset_label,
                "prediction_column": prediction_column,
                "window_index": int(window_index),
                "window_start": subset["date"].min().date().isoformat(),
                "window_end": subset["date"].max().date().isoformat(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def compute_group_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, str]] = [
        ("saw_in_theater", "context_binary"),
        ("drink_before_movie", "context_binary"),
        ("likely_with_amelia", "context_binary"),
        ("assigned_unfinished_two", "context_binary"),
        ("pressure_tier", "context_categorical"),
        ("life_period", "context_categorical"),
    ]
    for column, group_type in group_specs:
        for value, subset in df.groupby(column):
            rows.append(
                {
                    "group_type": group_type,
                    "group_column": column,
                    "group_value": value,
                    "rows": int(len(subset)),
                    "mean_rating": float(subset["analysis_rating"].mean()),
                    "mean_external": float(subset["external_mean_on_10"].mean()),
                    "mean_residual_vs_external": float(
                        (
                            subset["analysis_rating"] - subset["external_mean_on_10"]
                        ).mean()
                    ),
                    "liked_rate": float(
                        (subset["analysis_rating"] >= LIKE_THRESHOLD).mean()
                    ),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["group_column", "group_value"])
        .reset_index(drop=True)
    )


def fit_residual_context_model(df: pd.DataFrame) -> pd.DataFrame:
    working = df.loc[df["external_mean_on_10"].notna()].copy()
    working["target_residual"] = (
        working["analysis_rating"] - working["external_mean_on_10"]
    )
    feature_frame = pd.get_dummies(
        working[
            [
                "saw_in_theater",
                "drink_before_movie",
                "likely_with_amelia",
                "assigned_unfinished_two",
                "is_weekend",
                "pressure_tier",
                "life_period",
            ]
        ],
        drop_first=True,
        dtype=float,
    )
    feature_frame.insert(
        0, "external_mean_on_10", working["external_mean_on_10"].astype(float)
    )
    model = LinearRegression().fit(feature_frame, working["analysis_rating"])
    rows = [
        {
            "feature": "intercept",
            "coefficient": float(model.intercept_),
        }
    ]
    rows.extend(
        {
            "feature": feature_name,
            "coefficient": float(coefficient),
        }
        for feature_name, coefficient in zip(
            feature_frame.columns, model.coef_, strict=True
        )
    )
    return pd.DataFrame(rows)


def compute_repeat_watch_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rated = df.loc[df["normalized_title"].fillna("") != ""].copy()
    rows: list[dict[str, Any]] = []
    for normalized_title, subset in rated.groupby("normalized_title"):
        if len(subset) < 2:
            continue
        ratings = subset["analysis_rating"].to_numpy(dtype=float)
        rows.append(
            {
                "normalized_title": normalized_title,
                "rows": int(len(subset)),
                "mean_rating": float(ratings.mean()),
                "min_rating": float(ratings.min()),
                "max_rating": float(ratings.max()),
                "rating_range": float(ratings.max() - ratings.min()),
                "rating_std": float(ratings.std(ddof=0)),
                "titles": " | ".join(subset["movie_title"].astype(str).tolist()),
                "dates": " | ".join(
                    subset["date"].dt.date.astype(str).tolist()
                    if subset["date"].notna().any()
                    else []
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["rating_range", "rows", "normalized_title"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def compute_bias_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid_external = df.loc[df["external_mean_on_10"].notna()].copy()
    rows = [
        {
            "metric": "rows",
            "value": float(len(df)),
        },
        {
            "metric": "mean_rating",
            "value": float(df["analysis_rating"].mean()),
        },
        {
            "metric": "median_rating",
            "value": float(df["analysis_rating"].median()),
        },
        {
            "metric": "liked_rate",
            "value": float((df["analysis_rating"] >= LIKE_THRESHOLD).mean()),
        },
        {
            "metric": "share_9plus",
            "value": float((df["analysis_rating"] >= 9).mean()),
        },
        {
            "metric": "share_2orless",
            "value": float((df["analysis_rating"] <= 2).mean()),
        },
    ]
    if not valid_external.empty:
        gap = valid_external["analysis_rating"] - valid_external["external_mean_on_10"]
        rows.extend(
            [
                {
                    "metric": "mean_residual_vs_external",
                    "value": float(gap.mean()),
                },
                {
                    "metric": "median_residual_vs_external",
                    "value": float(gap.median()),
                },
                {
                    "metric": "share_above_external",
                    "value": float((gap > 0).mean()),
                },
                {
                    "metric": "share_below_external",
                    "value": float((gap < 0).mean()),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_cleanup_overlap(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    raw_subset = raw_df[
        [
            "analysis_row_key",
            "movie_title",
            "analysis_rating",
            "rt_average_on_10",
            "rt_critic_on_10",
            "rt_audience_on_10",
            "external_mean_on_10",
        ]
    ].rename(
        columns=lambda column: (
            f"raw_{column}" if column != "analysis_row_key" else column
        )
    )
    clean_subset = clean_df[
        [
            "analysis_row_key",
            "movie_title",
            "analysis_rating",
            "rt_average_on_10",
            "rt_critic_on_10",
            "rt_audience_on_10",
            "external_mean_on_10",
        ]
    ].rename(
        columns=lambda column: (
            f"clean_{column}" if column != "analysis_row_key" else column
        )
    )
    return raw_subset.merge(clean_subset, on="analysis_row_key", how="inner")


def build_cleanup_delta_table(
    raw_predictions: pd.DataFrame,
    clean_predictions: pd.DataFrame,
) -> pd.DataFrame:
    joined = (
        raw_predictions[
            [
                "analysis_row_key",
                "movie_title",
                "analysis_rating",
                "legacy_average_prediction",
                "legacy_saved_linear_prediction",
                "legacy_refit_linear_prediction",
            ]
        ]
        .rename(
            columns={
                "movie_title": "raw_movie_title",
                "analysis_rating": "raw_rating",
                "legacy_average_prediction": "raw_avg_prediction",
                "legacy_saved_linear_prediction": "raw_saved_linear_prediction",
                "legacy_refit_linear_prediction": "raw_refit_linear_prediction",
            }
        )
        .merge(
            clean_predictions[
                [
                    "analysis_row_key",
                    "movie_title",
                    "analysis_rating",
                    "legacy_average_prediction",
                    "legacy_saved_linear_prediction",
                    "legacy_refit_linear_prediction",
                ]
            ].rename(
                columns={
                    "movie_title": "clean_movie_title",
                    "analysis_rating": "clean_rating",
                    "legacy_average_prediction": "clean_avg_prediction",
                    "legacy_saved_linear_prediction": "clean_saved_linear_prediction",
                    "legacy_refit_linear_prediction": "clean_refit_linear_prediction",
                }
            ),
            on="analysis_row_key",
            how="inner",
        )
    )
    joined["avg_abs_error_delta"] = (
        joined["raw_rating"] - joined["raw_avg_prediction"]
    ).abs() - (joined["clean_rating"] - joined["clean_avg_prediction"]).abs()
    joined["saved_linear_abs_error_delta"] = (
        joined["raw_rating"] - joined["raw_saved_linear_prediction"]
    ).abs() - (joined["clean_rating"] - joined["clean_saved_linear_prediction"]).abs()
    joined["refit_linear_abs_error_delta"] = (
        joined["raw_rating"] - joined["raw_refit_linear_prediction"]
    ).abs() - (joined["clean_rating"] - joined["clean_refit_linear_prediction"]).abs()
    return joined.sort_values(
        "saved_linear_abs_error_delta",
        ascending=False,
    ).reset_index(drop=True)


def build_residual_inspection_table(
    df: pd.DataFrame,
    *,
    prediction_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for prediction_column in prediction_columns:
            prediction_value = optional_float(row[prediction_column])
            if prediction_value is None:
                continue
            rows.append(
                {
                    "date": (
                        row["date"].date().isoformat()
                        if not pd.isna(row["date"])
                        else ""
                    ),
                    "movie_title": row["movie_title"],
                    "analysis_rating": row["analysis_rating"],
                    "prediction_column": prediction_column,
                    "prediction": prediction_value,
                    "residual": row["analysis_rating"] - prediction_value,
                    "abs_residual": abs(row["analysis_rating"] - prediction_value),
                    "quality_flags": row.get("quality_flags", ""),
                    "watch_status": row["watch_status"],
                    "where_seen": row["where_seen"],
                    "drink_before_movie": row["drink_before_movie"],
                    "likely_with_amelia": row["likely_with_amelia"],
                    "life_period": row["life_period"],
                    "calendar_summary": row.get("calendar_summary", ""),
                    "rt_url": row.get("rt_url", ""),
                    "imdb_url": row.get("imdb_url", ""),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["abs_residual", "movie_title"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def plot_gain_curves(curves: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, metric in zip(axes, ("mean_rating", "liked_rate"), strict=True):
        for (dataset, model), subset in curves.groupby(["dataset", "model"]):
            axis.plot(
                subset["drop_fraction"] * 100.0,
                subset[metric],
                marker="o",
                label=f"{dataset}: {model}",
            )
        axis.set_xlabel("Dropped lowest-predicted movies (%)")
        axis.set_ylabel(
            "Actual mean rating" if metric == "mean_rating" else "Actual liked rate"
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_cutoff_tradeoff(tradeoff: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        tradeoff["score_cutoff"],
        tradeoff["mean_rating"],
        marker="o",
        color="#1f77b4",
    )
    for _, row in tradeoff.iterrows():
        ax.annotate(
            f"{int(round(row['drop_fraction'] * 100.0))}%",
            (row["score_cutoff"], row["mean_rating"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Rotten Tomatoes average cutoff")
    ax.set_ylabel("Actual mean rating after cutoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ridge_coefficients(coefficients: pd.DataFrame, output_path: Path) -> None:
    subset = coefficients.loc[coefficients["feature_group"] != "intercept"].copy()
    subset["abs_effect"] = subset["raw_unit_coefficient"].abs()
    subset = subset.sort_values("abs_effect", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(subset["transformed_feature"], subset["raw_unit_coefficient"])
    ax.axvline(0.0, color="#2d3436", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Coefficient in raw feature units")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_over_time(
    df: pd.DataFrame,
    *,
    prediction_columns: list[str],
    output_path: Path,
) -> None:
    working = df.loc[df["date"].notna()].sort_values("date").copy()
    fig, axes = plt.subplots(
        len(prediction_columns),
        1,
        figsize=(14, 4 * len(prediction_columns)),
        sharex=True,
    )
    if len(prediction_columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, prediction_columns, strict=True):
        residual = working["analysis_rating"] - working[column]
        axis.scatter(working["date"], residual, alpha=0.6, s=25)
        rolling = residual.rolling(20, min_periods=5).mean()
        axis.plot(working["date"], rolling, color="#c0392b", linewidth=2)
        axis.axhline(0.0, color="#2d3436", linestyle="--", linewidth=1.5)
        axis.set_title(column)
        axis.set_ylabel("Residual")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Watch date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cleanup_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    regression = metrics.loc[metrics["task"] == "rating_regression"].copy()
    order = ["legacy_average_rt", "legacy_saved_linear", "legacy_refit_linear"]
    datasets = regression["dataset"].drop_duplicates().tolist()
    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, dataset in enumerate(datasets):
        subset = regression.loc[regression["dataset"] == dataset].set_index("model")
        values = [subset.loc[model, "mae"] for model in order]
        ax.bar(x + index * width, values, width=width, label=dataset)
    ax.set_xticks(x + width / 2, labels=order)
    ax.set_ylabel("MAE on overlapping rows")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_period_residuals(group_effects: pd.DataFrame, output_path: Path) -> None:
    subset = group_effects.loc[group_effects["group_column"] == "life_period"].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(subset["group_value"], subset["mean_residual_vs_external"])
    ax.axhline(0.0, color="#2d3436", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Mean residual vs external score")
    ax.set_xlabel("Life period")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fetch_short_imdb_synopsis(imdb_url: str) -> str:
    if not imdb_url.strip():
        return ""
    request = Request(imdb_url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=10) as response:
        html_text = response.read().decode("utf-8", "ignore")
    match = SUMMARY_DESCRIPTION_PATTERN.search(html_text)
    if match is None:
        match = JSON_LD_DESCRIPTION_PATTERN.search(html_text)
    if match is None:
        return ""
    body = html.unescape(match.group("body"))
    body = body.replace("\\u0027", "'")
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(r"\s*IMDb.*$", "", body)
    return body


def fetch_wikipedia_page_summary(page_title: str) -> str:
    request = Request(
        (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{quote(page_title, safe='')}"
        ),
        headers=REQUEST_HEADERS,
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", "ignore"))
    if payload.get("type") == "disambiguation":
        return ""
    extract = str(payload.get("extract", "")).strip()
    lowered = extract.lower()
    if not extract or any(
        marker in lowered for marker in WIKIPEDIA_DISAMBIGUATION_MARKERS
    ):
        return ""
    return re.sub(r"\s+", " ", extract).strip()


def fetch_wikipedia_summary(movie_title: str, release_year: Any) -> str:
    title = str(movie_title).strip()
    if not title:
        return ""
    year = optional_float(release_year)
    query_variants = [title]
    if year is not None:
        year_int = int(year)
        query_variants = [
            f"{title} {year_int} film",
            f"{title} ({year_int} film)",
            f"{title} film",
            f"{title} {year_int}",
            title,
        ]
    seen_queries: set[str] = set()
    for query in query_variants:
        if query in seen_queries:
            continue
        seen_queries.add(query)
        params = urlencode(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srlimit": 5,
                "srsearch": query,
            }
        )
        request = Request(
            f"https://en.wikipedia.org/w/api.php?{params}",
            headers=REQUEST_HEADERS,
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8", "ignore"))
        search_results = payload.get("query", {}).get("search", [])
        fallback_snippet = ""
        for result in search_results:
            page_title = str(result.get("title", "")).strip()
            if not page_title:
                continue
            if not fallback_snippet:
                snippet = html.unescape(str(result.get("snippet", "")))
                snippet = re.sub(r"<[^>]+>", " ", snippet)
                snippet = re.sub(r"\s+", " ", snippet).strip()
                if snippet:
                    fallback_snippet = snippet
            summary = fetch_wikipedia_page_summary(page_title)
            if summary:
                return summary
        if fallback_snippet:
            return fallback_snippet
    return ""


def fetch_movie_synopsis(
    *,
    imdb_url: str,
    movie_title: str,
    release_year: Any,
) -> str:
    try:
        synopsis = fetch_short_imdb_synopsis(imdb_url)
    except Exception:
        synopsis = ""
    if synopsis:
        return synopsis
    return fetch_wikipedia_summary(movie_title, release_year)


def build_synopsis_dataset(
    df: pd.DataFrame,
    *,
    cache_path: Path,
) -> pd.DataFrame:
    cache: dict[str, str] = (
        json.loads(cache_path.read_text()) if cache_path.exists() else {}
    )
    rows: list[dict[str, Any]] = []
    unique_rows = (
        df.loc[df["movie_title"].fillna("") != ""]
        .drop_duplicates(["movie_title", "imdb_release_year", "imdb_url"])
        .copy()
    )
    for _, row in unique_rows.iterrows():
        imdb_url = str(row["imdb_url"])
        cache_key = (
            imdb_url
            if imdb_url.strip()
            else f"wiki|{row['movie_title']}|{row['imdb_release_year']}"
        )
        synopsis = cache.get(cache_key, "")
        if not synopsis:
            try:
                synopsis = fetch_movie_synopsis(
                    imdb_url=imdb_url,
                    movie_title=str(row["movie_title"]),
                    release_year=row["imdb_release_year"],
                )
            except Exception:
                synopsis = ""
            cache[cache_key] = synopsis
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
        rows.append(
            {
                "movie_title": row["movie_title"],
                "release_year": row["imdb_release_year"],
                "analysis_rating": row["analysis_rating"],
                "watch_status": row["watch_status"],
                "where_seen": row["where_seen"],
                "life_period": row["life_period"],
                "likely_with_amelia": row["likely_with_amelia"],
                "imdb_url": imdb_url,
                "synopsis": synopsis,
            }
        )
    synopsis_df = (
        pd.DataFrame(rows)
        .sort_values(["release_year", "movie_title"])
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(0)
    shuffled_index = rng.permutation(len(synopsis_df))
    holdout_mask = np.zeros(len(synopsis_df), dtype=bool)
    holdout_mask[shuffled_index[::2]] = True
    synopsis_df["random_half_split"] = np.where(
        holdout_mask,
        "holdout_half",
        "train_half",
    )
    return synopsis_df


def summarize_second_pass(
    *,
    rated_row_count: int,
    legacy_metrics: pd.DataFrame,
    second_cv_metrics: pd.DataFrame,
    combined_cv_metrics: pd.DataFrame,
    cleanup_metrics: pd.DataFrame,
    bias_summary: pd.DataFrame,
    source_year_gap_count: int,
) -> str:
    clean_full_legacy = legacy_metrics.loc[
        (legacy_metrics["dataset"] == "clean_full_with_unfinished2")
        & (legacy_metrics["model"] == "legacy_saved_linear")
        & (legacy_metrics["task"] == "rating_regression")
    ].iloc[0]
    second_best = second_cv_metrics.sort_values("mae").iloc[0]
    combined_best = combined_cv_metrics.sort_values("mae").iloc[0]
    cleanup_saved = cleanup_metrics.loc[
        (cleanup_metrics["dataset"] == "clean_overlap")
        & (cleanup_metrics["model"] == "legacy_saved_linear")
        & (cleanup_metrics["task"] == "rating_regression")
    ].iloc[0]
    cleanup_raw = cleanup_metrics.loc[
        (cleanup_metrics["dataset"] == "raw_overlap")
        & (cleanup_metrics["model"] == "legacy_saved_linear")
        & (cleanup_metrics["task"] == "rating_regression")
    ].iloc[0]
    lines = [
        f"Rated rows analyzed after assigning unfinished movies a 2: {rated_row_count}",
        (
            "Legacy saved RT linear model on cleaned second sheet: "
            f"MAE {clean_full_legacy['mae']:.3f}, within-1 {clean_full_legacy['within_one']:.3f} "
            f"on {int(clean_full_legacy['rows'])} rows with full RT scores"
        ),
        (
            "Best CV model on cleaned second sheet: "
            f"{second_best['model']} with MAE {second_best['mae']:.3f}, "
            f"within-1 {second_best['within_one']:.3f}"
        ),
        (
            "Best CV model on combined dataset: "
            f"{combined_best['model']} with MAE {combined_best['mae']:.3f}, "
            f"within-1 {combined_best['within_one']:.3f}"
        ),
        (
            "Cleanup improvement on overlapping rows for legacy saved linear model: "
            f"MAE {cleanup_raw['mae']:.3f} -> {cleanup_saved['mae']:.3f}"
        ),
        ("Remaining source-year-gap rows after cleanup: " f"{source_year_gap_count}"),
        (
            "Overall mean residual vs external scores: "
            f"{bias_summary.loc[bias_summary['metric'] == 'mean_residual_vs_external', 'value'].iloc[0]:.3f}"
        ),
    ]
    return "\n".join(lines)
