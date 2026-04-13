from __future__ import annotations

import io
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "data" / "mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GEMINI_APPROX_PATTERN = re.compile(
    r"### The 25-Movie Dataset \(CSV\)\n\n```\n(?P<body>.*?)```",
    re.DOTALL,
)
GEMINI_CORRECTED_PATTERN = re.compile(
    r"(?P<body>"
    r"Movie Title,User Rating,RT Audience,RT Critic,IMDb \(Actual\),Letterboxd \(Actual\)\n"
    r".*?)"
    r"(?:```|\n## Prompt:)",
    re.DOTALL,
)
GEMINI_TITLE_MAP = {
    "Batman v Superman": "Batman vs Superman: Dawn of Justice",
    "Tropa de Elite": "Tropa de Elite (brazilian police squad, lead to support of Bolsonaro)",
    "2001: A Space Odyssey": "Space Odyssey 2001",
    "Ghost in the Shell (2017)": "Ghost in the Shell (2018)",
    "Facing the Giants": "Facing the Giants (motivational highschool Football)",
    "Everything Everywhere All at Once": "Everything Everywhere all at Once",
    "3:10 to Yuma": "3:10 to Yuma (Cowboy movie with Bale and Crowe)",
    "Trainspotting": "Trainspotters",
}


def _read_embedded_csv(markdown_text: str, pattern: re.Pattern[str]) -> pd.DataFrame:
    match = pattern.search(markdown_text)
    if match is None:
        raise ValueError("Could not find embedded CSV block in Gemini markdown")
    body = match.group("body").strip()
    return pd.read_csv(io.StringIO(body))


def load_gemini_rating_tables(markdown_path: Path) -> dict[str, pd.DataFrame]:
    markdown_text = markdown_path.read_text()
    approximate = _read_embedded_csv(markdown_text, GEMINI_APPROX_PATTERN)
    corrected = _read_embedded_csv(markdown_text, GEMINI_CORRECTED_PATTERN).rename(
        columns={
            "IMDb (Actual)": "IMDb",
            "Letterboxd (Actual)": "Letterboxd",
        }
    )
    return {
        "approximate": approximate,
        "corrected": corrected,
    }


def build_gemini_comparison_table(
    approximate: pd.DataFrame,
    corrected: pd.DataFrame,
    verified: pd.DataFrame,
) -> pd.DataFrame:
    verified_core = verified[
        [
            "movie_title",
            "my_rating",
            "rt_audience_rating",
            "rt_critic_rating",
        ]
    ].rename(
        columns={
            "movie_title": "verified_movie_title",
            "my_rating": "verified_user_rating",
            "rt_audience_rating": "verified_rt_audience",
            "rt_critic_rating": "verified_rt_critic",
        }
    )
    approximate = approximate.copy()
    corrected = corrected.copy()
    approximate["verified_movie_title"] = approximate["Movie Title"].map(
        lambda title: GEMINI_TITLE_MAP.get(title, title)
    )
    corrected["verified_movie_title"] = corrected["Movie Title"].map(
        lambda title: GEMINI_TITLE_MAP.get(title, title)
    )
    merged = approximate.merge(
        corrected.rename(
            columns={
                "IMDb": "IMDb_corrected",
                "Letterboxd": "Letterboxd_corrected",
                "RT Audience": "RT Audience_corrected",
                "RT Critic": "RT Critic_corrected",
                "User Rating": "User Rating_corrected",
            }
        )[
            [
                "verified_movie_title",
                "IMDb_corrected",
                "Letterboxd_corrected",
                "RT Audience_corrected",
                "RT Critic_corrected",
                "User Rating_corrected",
            ]
        ],
        on="verified_movie_title",
        how="left",
    ).merge(
        verified_core,
        on="verified_movie_title",
        how="left",
    )
    merged["rt_audience_abs_error_vs_verified"] = (
        merged["RT Audience"] - merged["verified_rt_audience"]
    ).abs()
    merged["rt_critic_abs_error_vs_verified"] = (
        merged["RT Critic"] - merged["verified_rt_critic"]
    ).abs()
    merged["imdb_abs_error_vs_corrected"] = (
        merged["IMDb"] - merged["IMDb_corrected"]
    ).abs()
    merged["letterboxd_abs_error_vs_corrected"] = (
        merged["Letterboxd"] - merged["Letterboxd_corrected"]
    ).abs()
    return merged


def summarize_gemini_alignment(comparison: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "RT audience",
            "RT Audience",
            "verified_rt_audience",
        ),
        (
            "RT critic",
            "RT Critic",
            "verified_rt_critic",
        ),
        (
            "IMDb",
            "IMDb",
            "IMDb_corrected",
        ),
        (
            "Letterboxd",
            "Letterboxd",
            "Letterboxd_corrected",
        ),
    ]
    rows: list[dict[str, float | str | int]] = []
    for label, left_col, right_col in specs:
        subset = comparison[[left_col, right_col]].dropna().copy()
        diffs = (subset[left_col] - subset[right_col]).abs().to_numpy(dtype=float)
        rows.append(
            {
                "metric": label,
                "n_rows": int(len(subset)),
                "mae": float(diffs.mean()),
                "median_abs_error": float(np.median(diffs)),
                "max_abs_error": float(diffs.max()),
                "share_within_1": float((diffs <= 1).mean()),
                "share_within_2": float((diffs <= 2).mean()),
                "share_within_5": float((diffs <= 5).mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_gemini_rt_model_fits(comparison: pd.DataFrame) -> pd.DataFrame:
    dataset_specs = {
        "gemini_approx": ("RT Audience", "RT Critic"),
        "verified_rt": ("verified_rt_audience", "verified_rt_critic"),
    }
    rows: list[dict[str, float | str]] = []
    for dataset_name, (audience_col, critic_col) in dataset_specs.items():
        subset = comparison[["verified_user_rating", audience_col, critic_col]].dropna()
        y = subset["verified_user_rating"].to_numpy(dtype=float)
        audience = subset[audience_col].to_numpy(dtype=float)
        audience_model = LinearRegression().fit(audience.reshape(-1, 1), y)
        audience_pred = audience_model.predict(audience.reshape(-1, 1))
        rows.append(
            {
                "dataset": dataset_name,
                "model": "audience_only",
                "coef_audience": float(audience_model.coef_[0]),
                "coef_critic": float("nan"),
                "intercept": float(audience_model.intercept_),
                "r2": float(audience_model.score(audience.reshape(-1, 1), y)),
                "mae": float(mean_absolute_error(y, audience_pred)),
            }
        )

        x = subset[[audience_col, critic_col]].to_numpy(dtype=float)
        combo_model = LinearRegression().fit(x, y)
        combo_pred = combo_model.predict(x)
        rows.append(
            {
                "dataset": dataset_name,
                "model": "audience_critic",
                "coef_audience": float(combo_model.coef_[0]),
                "coef_critic": float(combo_model.coef_[1]),
                "intercept": float(combo_model.intercept_),
                "r2": float(combo_model.score(x, y)),
                "mae": float(mean_absolute_error(y, combo_pred)),
            }
        )
    metrics = pd.DataFrame(rows)
    delta_rows: list[dict[str, float | str]] = []
    for model_name in ("audience_only", "audience_critic"):
        approx_row = metrics.loc[
            (metrics["dataset"] == "gemini_approx") & (metrics["model"] == model_name)
        ].iloc[0]
        verified_row = metrics.loc[
            (metrics["dataset"] == "verified_rt") & (metrics["model"] == model_name)
        ].iloc[0]
        delta_rows.append(
            {
                "dataset": "verified_minus_gemini",
                "model": model_name,
                "coef_audience": float(
                    verified_row["coef_audience"] - approx_row["coef_audience"]
                ),
                "coef_critic": (
                    float(verified_row["coef_critic"] - approx_row["coef_critic"])
                    if model_name == "audience_critic"
                    else float("nan")
                ),
                "intercept": float(verified_row["intercept"] - approx_row["intercept"]),
                "r2": float(verified_row["r2"] - approx_row["r2"]),
                "mae": float(verified_row["mae"] - approx_row["mae"]),
            }
        )
    return pd.concat([metrics, pd.DataFrame(delta_rows)], ignore_index=True)


def regression_fit_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predictions = slope * x + intercept
    corr = float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": corr**2,
        "correlation": corr,
        "mae": float(mean_absolute_error(y, x)),
        "rmse": float(math.sqrt(mean_squared_error(y, x))),
        "line_mae": float(mean_absolute_error(y, predictions)),
    }


def leave_one_out_linear_predictions(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    model = LinearRegression()
    return cross_val_predict(
        model,
        df[feature_columns].to_numpy(dtype=float),
        df["my_rating"].to_numpy(dtype=float),
        cv=LeaveOneOut(),
        n_jobs=None,
    )


def personal_bucket_4(rating: float) -> int:
    if rating <= 5:
        return 0
    if rating <= 7:
        return 1
    if rating <= 9:
        return 2
    return 3


def personal_bucket_5(rating: float) -> int:
    if rating <= 4:
        return 1
    if rating <= 6:
        return 2
    if rating == 7:
        return 3
    if rating <= 9:
        return 4
    return 5


def continuous_to_personal_bucket_5(rating: float) -> int:
    if rating < 4.5:
        return 1
    if rating < 6.5:
        return 2
    if rating < 7.5:
        return 3
    if rating < 9.5:
        return 4
    return 5


def personal_bucket_3(rating: float) -> int:
    if rating <= 5:
        return 0
    if rating <= 7:
        return 1
    return 2


def rt_bucket_4(score: float) -> int:
    if score < 60:
        return 0
    if score < 75:
        return 1
    if score < 90:
        return 2
    return 3


def rt_bucket_3(score: float) -> int:
    if score < 60:
        return 0
    if score < 75:
        return 1
    return 2


def evaluate_bucket_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_sets = {
        "critic_only": ["rt_critic_rating"],
        "audience_only": ["rt_audience_rating"],
        "average_only": ["rt_average_rating"],
        "audience_critic": ["rt_audience_rating", "rt_critic_rating"],
    }
    schemes = {
        "3_step": np.array([personal_bucket_3(rating) for rating in df["my_rating"]]),
        "4_step": np.array([personal_bucket_4(rating) for rating in df["my_rating"]]),
        "5_step": np.array([personal_bucket_5(rating) for rating in df["my_rating"]]),
    }
    for scheme_name, y in schemes.items():
        for model_name, feature_columns in feature_sets.items():
            x = df[feature_columns].to_numpy(dtype=float)
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000),
            )
            predictions = cross_val_predict(
                estimator,
                x,
                y,
                cv=LeaveOneOut(),
                method="predict",
                n_jobs=None,
            )
            rows.append(
                {
                    "scheme": scheme_name,
                    "model": model_name,
                    "accuracy": float(accuracy_score(y, predictions)),
                    "macro_f1": float(f1_score(y, predictions, average="macro")),
                    "quadratic_kappa": float(
                        cohen_kappa_score(y, predictions, weights="quadratic")
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["scheme", "accuracy", "macro_f1", "quadratic_kappa"],
        ascending=[True, False, False, False],
    )


def build_direct_bucket_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    personal_3 = np.array([personal_bucket_3(rating) for rating in df["my_rating"]])
    personal_4 = np.array([personal_bucket_4(rating) for rating in df["my_rating"]])
    audience_3 = np.array([rt_bucket_3(score) for score in df["rt_audience_rating"]])
    average_3 = np.array([rt_bucket_3(score) for score in df["rt_average_rating"]])
    audience_4 = np.array([rt_bucket_4(score) for score in df["rt_audience_rating"]])
    average_4 = np.array([rt_bucket_4(score) for score in df["rt_average_rating"]])
    return {
        "audience_3": pd.crosstab(
            pd.Series(audience_3, name="audience_bucket"),
            pd.Series(personal_3, name="my_bucket"),
        ),
        "average_3": pd.crosstab(
            pd.Series(average_3, name="average_bucket"),
            pd.Series(personal_3, name="my_bucket"),
        ),
        "audience_4": pd.crosstab(
            pd.Series(audience_4, name="audience_bucket"),
            pd.Series(personal_4, name="my_bucket"),
        ),
        "average_4": pd.crosstab(
            pd.Series(average_4, name="average_bucket"),
            pd.Series(personal_4, name="my_bucket"),
        ),
    }


def direct_bucket_accuracy(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, table in tables.items():
        values = table.to_numpy(dtype=float)
        rows.append(
            {
                "table": name,
                "accuracy": float(np.trace(values) / values.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("table").reset_index(drop=True)


def leave_one_out_isotonic_predictions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions: list[float] = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(x):
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(x[train_index], y[train_index])
        predictions.append(float(model.predict(x[test_index])[0]))
    return np.array(predictions)


def evaluate_audience_shape_models(df: pd.DataFrame) -> pd.DataFrame:
    x = df["rt_audience_on_10"].to_numpy(dtype=float)
    y = df["my_rating"].to_numpy(dtype=float)
    linear_predictions = cross_val_predict(
        LinearRegression(),
        x.reshape(-1, 1),
        y,
        cv=LeaveOneOut(),
    )
    isotonic_predictions = leave_one_out_isotonic_predictions(x, y)
    rows = []
    for name, predictions in (
        ("linear", linear_predictions),
        ("isotonic", isotonic_predictions),
    ):
        ss_res = float(((y - predictions) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        five_step_truth = np.array([personal_bucket_5(value) for value in y])
        five_step_pred = np.array(
            [
                continuous_to_personal_bucket_5(max(0.0, min(10.0, value)))
                for value in predictions
            ]
        )
        rows.append(
            {
                "model": name,
                "cv_r2": 1.0 - ss_res / ss_tot,
                "mae": float(mean_absolute_error(y, predictions)),
                "rmse": float(math.sqrt(mean_squared_error(y, predictions))),
                "five_step_accuracy": float(
                    accuracy_score(five_step_truth, five_step_pred)
                ),
                "five_step_quadratic_kappa": float(
                    cohen_kappa_score(
                        five_step_truth,
                        five_step_pred,
                        weights="quadratic",
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def build_verified_imdb_dataset(
    corrected: pd.DataFrame,
    verified: pd.DataFrame,
) -> pd.DataFrame:
    corrected = corrected.copy()
    corrected["movie_title"] = corrected["Movie Title"].map(
        lambda title: GEMINI_TITLE_MAP.get(title, title)
    )
    merged = corrected.merge(
        verified[
            [
                "movie_title",
                "my_rating",
                "rt_audience_rating",
                "rt_critic_rating",
            ]
        ],
        on="movie_title",
        how="left",
    )
    if (
        merged[["my_rating", "rt_audience_rating", "rt_critic_rating"]]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError(
            "Could not align every Gemini row with the verified RT dataset"
        )
    for column in ("rt_audience_rating", "rt_critic_rating", "IMDb"):
        merged[f"{column}_z"] = (merged[column] - merged[column].mean()) / merged[
            column
        ].std(ddof=0)
    merged["audience_imdb_zavg"] = (
        merged["rt_audience_rating_z"] + merged["IMDb_z"]
    ) / 2.0
    merged["all3_zavg"] = (
        merged["rt_audience_rating_z"] + merged["rt_critic_rating_z"] + merged["IMDb_z"]
    ) / 3.0
    return merged


def build_provisional_full_imdb_dataset(
    provisional: pd.DataFrame,
    verified: pd.DataFrame,
) -> pd.DataFrame:
    working = provisional.copy()
    working["movie_title"] = working["Movie Title"].map(
        lambda title: GEMINI_TITLE_MAP.get(title, title)
    )
    merged = working.merge(
        verified[
            [
                "movie_title",
                "my_rating",
                "rt_audience_rating",
                "rt_critic_rating",
                "rt_average_rating",
            ]
        ],
        on="movie_title",
        how="left",
    )
    if merged["my_rating"].isnull().any():
        raise ValueError(
            "Could not align every provisional IMDb row with the verified RT dataset"
        )
    merged["imdb_on_10"] = merged["IMDb Rating"] / 10.0
    merged["rt_audience_on_10"] = merged["rt_audience_rating"] / 10.0
    merged["combo_avg_on_10"] = (
        merged["imdb_on_10"] + merged["rt_audience_on_10"]
    ) / 2.0
    for column in ("rt_audience_rating", "rt_critic_rating", "IMDb Rating"):
        merged[f"{column}_z"] = (merged[column] - merged[column].mean()) / merged[
            column
        ].std(ddof=0)
    merged["combo_zavg"] = (
        merged["rt_audience_rating_z"] + merged["IMDb Rating_z"]
    ) / 2.0
    return merged


def evaluate_imdb_rt_continuous_models(df: pd.DataFrame) -> pd.DataFrame:
    feature_sets = {
        "audience_only": ["rt_audience_rating"],
        "imdb_only": ["IMDb"],
        "audience_imdb": ["rt_audience_rating", "IMDb"],
        "audience_critic_imdb": ["rt_audience_rating", "rt_critic_rating", "IMDb"],
        "audience_imdb_zavg": ["audience_imdb_zavg"],
        "all3_zavg": ["all3_zavg"],
    }
    y = df["my_rating"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for model_name, feature_columns in feature_sets.items():
        x = df[feature_columns].to_numpy(dtype=float)
        predictions = cross_val_predict(
            LinearRegression(),
            x,
            y,
            cv=LeaveOneOut(),
        )
        ss_res = float(((y - predictions) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        rows.append(
            {
                "model": model_name,
                "cv_r2": 1.0 - ss_res / ss_tot,
                "mae": float(mean_absolute_error(y, predictions)),
                "rmse": float(math.sqrt(mean_squared_error(y, predictions))),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("cv_r2", ascending=False).reset_index(drop=True)
    )


def evaluate_imdb_rt_bucket_models(
    df: pd.DataFrame,
    *,
    scheme: str = "3_step",
) -> pd.DataFrame:
    feature_sets = {
        "audience_only": ["rt_audience_rating"],
        "imdb_only": ["IMDb"],
        "audience_imdb": ["rt_audience_rating", "IMDb"],
        "audience_critic_imdb": ["rt_audience_rating", "rt_critic_rating", "IMDb"],
        "audience_imdb_zavg": ["audience_imdb_zavg"],
        "all3_zavg": ["all3_zavg"],
    }
    mapping = {
        "3_step": personal_bucket_3,
        "5_step": personal_bucket_5,
    }
    y = np.array([mapping[scheme](rating) for rating in df["my_rating"]])
    rows: list[dict[str, Any]] = []
    for model_name, feature_columns in feature_sets.items():
        x = df[feature_columns].to_numpy(dtype=float)
        predictions = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            x,
            y,
            cv=LeaveOneOut(),
            method="predict",
        )
        rows.append(
            {
                "scheme": scheme,
                "model": model_name,
                "accuracy": float(accuracy_score(y, predictions)),
                "macro_f1": float(f1_score(y, predictions, average="macro")),
                "quadratic_kappa": float(
                    cohen_kappa_score(y, predictions, weights="quadratic")
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["accuracy", "macro_f1", "quadratic_kappa"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def evaluate_full_provisional_imdb_models(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "audience_only": ["rt_audience_rating"],
        "imdb_only": ["IMDb Rating"],
        "combo_avg": ["combo_avg_on_10"],
        "combo_zavg": ["combo_zavg"],
        "audience_imdb": ["rt_audience_rating", "IMDb Rating"],
    }
    y = df["my_rating"].to_numpy(dtype=float)
    continuous_rows: list[dict[str, Any]] = []
    for model_name, feature_columns in feature_sets.items():
        x = df[feature_columns].to_numpy(dtype=float)
        predictions = cross_val_predict(
            LinearRegression(),
            x,
            y,
            cv=LeaveOneOut(),
        )
        ss_res = float(((y - predictions) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        continuous_rows.append(
            {
                "model": model_name,
                "cv_r2": 1.0 - ss_res / ss_tot,
                "mae": float(mean_absolute_error(y, predictions)),
                "rmse": float(math.sqrt(mean_squared_error(y, predictions))),
            }
        )

    bucket_rows: list[dict[str, Any]] = []
    mapping = {
        "3_step": personal_bucket_3,
        "4_step": personal_bucket_4,
        "5_step": personal_bucket_5,
    }
    for scheme_name, bucket_fn in mapping.items():
        yy = np.array([bucket_fn(rating) for rating in df["my_rating"]])
        for model_name, feature_columns in feature_sets.items():
            x = df[feature_columns].to_numpy(dtype=float)
            predictions = cross_val_predict(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                x,
                yy,
                cv=LeaveOneOut(),
                method="predict",
            )
            bucket_rows.append(
                {
                    "scheme": scheme_name,
                    "model": model_name,
                    "accuracy": float(accuracy_score(yy, predictions)),
                    "macro_f1": float(f1_score(yy, predictions, average="macro")),
                    "quadratic_kappa": float(
                        cohen_kappa_score(yy, predictions, weights="quadratic")
                    ),
                }
            )
    return (
        pd.DataFrame(continuous_rows)
        .sort_values("cv_r2", ascending=False)
        .reset_index(drop=True),
        pd.DataFrame(bucket_rows)
        .sort_values(
            ["scheme", "accuracy", "macro_f1", "quadratic_kappa"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True),
    )


def summarize_conjunctive_threshold_rules(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (6.0, 6.5, 7.0, 7.5, 8.0),
) -> pd.DataFrame:
    baseline_mean = float(df["my_rating"].mean())
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        kept = df.loc[
            (df["rt_audience_on_10"] >= threshold) & (df["imdb_on_10"] >= threshold)
        ].copy()
        rows.append(
            {
                "threshold": threshold,
                "keep_rate": float(len(kept) / len(df)),
                "gain_vs_all": (
                    float(kept["my_rating"].mean()) - baseline_mean
                    if len(kept)
                    else float("nan")
                ),
                "liked_rate": (
                    float((kept["my_rating"] >= 7).mean())
                    if len(kept)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _nan_safe_percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def bootstrap_threshold_uncertainty(
    df: pd.DataFrame,
    score_column: str,
    *,
    thresholds: list[float] | None = None,
    iterations: int = 3000,
    random_state: int = 0,
    baseline_mean: float | None = None,
    denominator_n: int | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = list(range(40, 96, 5))
    n_rows = len(df)
    if denominator_n is None:
        denominator_n = n_rows
    rng = np.random.default_rng(random_state)
    point_table = compute_threshold_tradeoff(
        df,
        score_column,
        thresholds=thresholds,
        baseline_mean=baseline_mean,
        denominator_n=denominator_n,
    ).set_index("threshold")
    score_values = df[score_column].to_numpy(dtype=float)
    my_ratings = df["my_rating"].to_numpy(dtype=float)
    liked = (my_ratings >= 7).astype(float)
    bucket5_values = np.array(
        [personal_bucket_5(value) for value in my_ratings],
        dtype=float,
    )

    point_table["liked_rate"] = [
        (
            float((my_ratings[score_values >= threshold] >= 7).mean())
            if (score_values >= threshold).any()
            else float("nan")
        )
        for threshold in thresholds
    ]
    point_table["mean_bucket5"] = [
        (
            float(bucket5_values[score_values >= threshold].mean())
            if (score_values >= threshold).any()
            else float("nan")
        )
        for threshold in thresholds
    ]

    bootstrap_records: list[dict[str, float]] = []
    for _ in range(iterations):
        sample_index = rng.integers(0, n_rows, size=n_rows)
        sample_scores = score_values[sample_index]
        sample_ratings = my_ratings[sample_index]
        sample_liked = liked[sample_index]
        sample_bucket5 = bucket5_values[sample_index]
        sample_baseline = (
            float(sample_ratings.mean()) if baseline_mean is None else baseline_mean
        )
        for threshold in thresholds:
            keep_mask = sample_scores >= threshold
            kept_count = int(keep_mask.sum())
            if kept_count == 0:
                bootstrap_records.append(
                    {
                        "threshold": threshold,
                        "keep_rate": 0.0,
                        "kept_mean_rating": float("nan"),
                        "gain_vs_all": float("nan"),
                        "liked_rate": float("nan"),
                        "mean_bucket5": float("nan"),
                    }
                )
                continue
            kept_ratings = sample_ratings[keep_mask]
            bootstrap_records.append(
                {
                    "threshold": threshold,
                    "keep_rate": kept_count / denominator_n,
                    "kept_mean_rating": float(kept_ratings.mean()),
                    "gain_vs_all": float(kept_ratings.mean() - sample_baseline),
                    "liked_rate": float(sample_liked[keep_mask].mean()),
                    "mean_bucket5": float(sample_bucket5[keep_mask].mean()),
                }
            )

    bootstrap_df = pd.DataFrame(bootstrap_records)
    summary_rows: list[dict[str, float | int]] = []
    metrics = [
        "keep_rate",
        "kept_mean_rating",
        "gain_vs_all",
        "liked_rate",
        "mean_bucket5",
    ]
    for threshold in thresholds:
        row: dict[str, float | int] = {"threshold": threshold}
        for metric in metrics:
            values = bootstrap_df.loc[
                bootstrap_df["threshold"] == threshold, metric
            ].to_numpy(dtype=float)
            row[metric] = float(point_table.loc[threshold, metric])
            row[f"{metric}_p05"] = _nan_safe_percentile(values, 5)
            row[f"{metric}_p25"] = _nan_safe_percentile(values, 25)
            row[f"{metric}_p75"] = _nan_safe_percentile(values, 75)
            row[f"{metric}_p95"] = _nan_safe_percentile(values, 95)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def compute_threshold_tradeoff(
    df: pd.DataFrame,
    score_column: str,
    *,
    thresholds: list[float] | None = None,
    baseline_mean: float | None = None,
    denominator_n: int | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = list(range(40, 96, 5))
    if baseline_mean is None:
        baseline_mean = float(df["my_rating"].mean())
    if denominator_n is None:
        denominator_n = len(df)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        kept = df.loc[df[score_column] >= threshold].copy()
        keep_rate = float(len(kept) / denominator_n)
        rows.append(
            {
                "threshold": threshold,
                "kept_movies": int(len(kept)),
                "keep_rate": keep_rate,
                "kept_mean_rating": (
                    float(kept["my_rating"].mean()) if len(kept) else float("nan")
                ),
                "gain_vs_all": (
                    float(kept["my_rating"].mean()) - baseline_mean
                    if len(kept)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _format_threshold(value: float) -> str:
    rounded = round(value)
    if math.isclose(value, rounded):
        return f"{rounded:d}"
    return f"{value:.1f}"


def _plot_metric_with_bands(
    ax: plt.Axes,
    summary_table: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    threshold_axis_label: str,
    baseline: float | None = None,
    baseline_label: str | None = None,
    annotate_keep_rate: bool = False,
    y_as_percent: bool = False,
    point_color: str = "#f58518",
    band_color: str = "#4c78a8",
) -> None:
    x = summary_table["threshold"].to_numpy(dtype=float)
    point_values = summary_table[metric].to_numpy(dtype=float)
    p05 = summary_table[f"{metric}_p05"].to_numpy(dtype=float)
    p25 = summary_table[f"{metric}_p25"].to_numpy(dtype=float)
    p75 = summary_table[f"{metric}_p75"].to_numpy(dtype=float)
    p95 = summary_table[f"{metric}_p95"].to_numpy(dtype=float)

    band95_mask = np.isfinite(x) & np.isfinite(p05) & np.isfinite(p95)
    band50_mask = np.isfinite(x) & np.isfinite(p25) & np.isfinite(p75)
    point_mask = np.isfinite(x) & np.isfinite(point_values)

    if band95_mask.any():
        ax.fill_between(
            x[band95_mask],
            p05[band95_mask],
            p95[band95_mask],
            color=band_color,
            alpha=0.16,
            label="Bootstrap 5-95%",
        )
    if band50_mask.any():
        ax.fill_between(
            x[band50_mask],
            p25[band50_mask],
            p75[band50_mask],
            color=band_color,
            alpha=0.32,
            label="Bootstrap 25-75%",
        )
    if point_mask.any():
        ax.plot(
            x[point_mask],
            point_values[point_mask],
            color=point_color,
            marker="o",
            linewidth=2.2,
            label="Observed full-data estimate",
        )

    if baseline is not None and math.isfinite(baseline):
        ax.axhline(
            baseline,
            color="#2d3436",
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label=baseline_label or "Overall baseline",
        )

    if annotate_keep_rate:
        for row in summary_table.itertuples(index=False):
            value = getattr(row, metric)
            if not math.isfinite(value):
                continue
            ax.annotate(
                f"{100 * row.keep_rate:.0f}%",
                (row.threshold, value),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                color=point_color,
            )

    if y_as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_title(title)
    ax.set_xlabel(threshold_axis_label)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def plot_threshold_uncertainty_panels(
    summary_table: pd.DataFrame,
    output_path: Path,
    *,
    score_label: str,
    threshold_axis_label: str,
    baseline_mean_rating: float,
    baseline_liked_rate: float,
    baseline_bucket5: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    _plot_metric_with_bands(
        axes[0, 0],
        summary_table,
        "gain_vs_all",
        title=f"{score_label}: rating gain by threshold",
        ylabel="Gain in average personal rating",
        threshold_axis_label=threshold_axis_label,
        baseline=0.0,
        baseline_label="No gain vs overall mean",
        annotate_keep_rate=True,
    )
    _plot_metric_with_bands(
        axes[0, 1],
        summary_table,
        "kept_mean_rating",
        title=f"{score_label}: expected kept-movie rating",
        ylabel="Expected personal rating",
        threshold_axis_label=threshold_axis_label,
        baseline=baseline_mean_rating,
        baseline_label=f"Overall mean ({baseline_mean_rating:.2f})",
    )
    _plot_metric_with_bands(
        axes[1, 0],
        summary_table,
        "liked_rate",
        title=f"{score_label}: liked-share among kept movies",
        ylabel="Liked share (rating >= 7)",
        threshold_axis_label=threshold_axis_label,
        baseline=baseline_liked_rate,
        baseline_label=f"Overall liked share ({baseline_liked_rate:.0%})",
        y_as_percent=True,
    )
    _plot_metric_with_bands(
        axes[1, 1],
        summary_table,
        "mean_bucket5",
        title=f"{score_label}: 5-step enjoyment bucket",
        ylabel="Expected 5-step bucket",
        threshold_axis_label=threshold_axis_label,
        baseline=baseline_bucket5,
        baseline_label=f"Overall bucket mean ({baseline_bucket5:.2f})",
    )

    axes[0, 0].legend(loc="upper left", fontsize=9)
    axes[0, 1].set_ylim(0, 10.2)
    axes[1, 1].set_ylim(1, 5.1)
    for ax in axes.flat:
        ax.set_xlim(
            float(summary_table["threshold"].min()) - 1,
            float(summary_table["threshold"].max()) + 1,
        )
    fig.suptitle(
        f"{score_label} threshold uncertainty analysis",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_practical_tradeoffs(
    summary_table: pd.DataFrame,
    output_path: Path,
    *,
    score_label: str,
    threshold_axis_label: str,
    baseline_liked_rate: float,
    practical_thresholds: tuple[float, ...] = (50, 60, 70, 75, 80, 85, 90),
) -> None:
    practical = summary_table.loc[
        summary_table["threshold"].isin(practical_thresholds)
    ].copy()
    practical = practical.dropna(
        subset=[
            "gain_vs_all",
            "gain_vs_all_p05",
            "gain_vs_all_p95",
            "liked_rate",
            "liked_rate_p05",
            "liked_rate_p95",
        ]
    )
    if practical.empty:
        raise ValueError("No practical thresholds available to plot")

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        practical["gain_vs_all"],
        practical["liked_rate"],
        c=practical["threshold"],
        cmap="viridis",
        s=80 + 320 * practical["keep_rate"],
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        practical["gain_vs_all"],
        practical["liked_rate"],
        color="#7f8c8d",
        linewidth=1.0,
        alpha=0.8,
        zorder=2,
    )
    for row in practical.itertuples(index=False):
        xerr = np.array(
            [
                [row.gain_vs_all - row.gain_vs_all_p05],
                [row.gain_vs_all_p95 - row.gain_vs_all],
            ]
        )
        yerr = np.array(
            [
                [row.liked_rate - row.liked_rate_p05],
                [row.liked_rate_p95 - row.liked_rate],
            ]
        )
        ax.errorbar(
            row.gain_vs_all,
            row.liked_rate,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="#4c78a8",
            alpha=0.45,
            linewidth=1.4,
            capsize=3,
            zorder=1,
        )
        ax.annotate(
            f"{_format_threshold(row.threshold)} / {100 * row.keep_rate:.0f}%",
            (row.gain_vs_all, row.liked_rate),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axvline(0, color="#2d3436", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(
        baseline_liked_rate,
        color="#2d3436",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Gain in average personal rating")
    ax.set_ylabel("Liked share among kept movies")
    ax.set_title(
        f"Practical {score_label} thresholds\n"
        "Labels show threshold / share of movies kept"
    )
    ax.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(threshold_axis_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_gain_detailed(
    summary_table: pd.DataFrame,
    output_path: Path,
    *,
    score_label: str,
    threshold_axis_label: str,
    practical_thresholds: tuple[float, ...] = (60, 70, 75, 80, 85, 90),
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _plot_threshold_gain_axis(
        ax,
        summary_table,
        score_label=score_label,
        threshold_axis_label=threshold_axis_label,
        practical_thresholds=practical_thresholds,
        show_colorbar=True,
        fig=fig,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_gain_axis(
    ax: plt.Axes,
    summary_table: pd.DataFrame,
    *,
    score_label: str,
    threshold_axis_label: str,
    practical_thresholds: tuple[float, ...],
    show_colorbar: bool = False,
    fig: plt.Figure | None = None,
) -> None:
    x = summary_table["threshold"].to_numpy(dtype=float)
    gain = summary_table["gain_vs_all"].to_numpy(dtype=float)
    p05 = summary_table["gain_vs_all_p05"].to_numpy(dtype=float)
    p25 = summary_table["gain_vs_all_p25"].to_numpy(dtype=float)
    p75 = summary_table["gain_vs_all_p75"].to_numpy(dtype=float)
    p95 = summary_table["gain_vs_all_p95"].to_numpy(dtype=float)
    keep_pct = 100 * summary_table["keep_rate"].to_numpy(dtype=float)

    band95_mask = np.isfinite(p05) & np.isfinite(p95)
    band50_mask = np.isfinite(p25) & np.isfinite(p75)
    point_mask = np.isfinite(gain)

    if band95_mask.any():
        ax.fill_between(
            x[band95_mask],
            p05[band95_mask],
            p95[band95_mask],
            color="#4c78a8",
            alpha=0.16,
            label="Bootstrap 5-95%",
        )
    if band50_mask.any():
        ax.fill_between(
            x[band50_mask],
            p25[band50_mask],
            p75[band50_mask],
            color="#4c78a8",
            alpha=0.30,
            label="Bootstrap 25-75%",
        )

    ax.plot(
        x[point_mask],
        gain[point_mask],
        color="#7f8c8d",
        linewidth=1.2,
        alpha=0.9,
        zorder=2,
    )
    scatter = ax.scatter(
        x[point_mask],
        gain[point_mask],
        c=keep_pct[point_mask],
        cmap="viridis",
        s=80,
        alpha=0.9,
        zorder=3,
        label="Observed full-data estimate",
    )

    for row in summary_table.itertuples(index=False):
        if not math.isfinite(row.gain_vs_all):
            continue
        bbox = None
        if any(math.isclose(row.threshold, target) for target in practical_thresholds):
            bbox = dict(boxstyle="round,pad=0.15", facecolor="yellow", alpha=0.55)
        ax.annotate(
            f"{100 * row.keep_rate:.0f}%",
            (row.threshold, row.gain_vs_all),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            bbox=bbox,
        )

    ax.axhline(0, color="#2d3436", linestyle="--", linewidth=1.3, alpha=0.8)
    ax.set_xlabel(threshold_axis_label)
    ax.set_ylabel("Gain in average personal rating")
    ax.set_title(
        f"{score_label}: threshold curve with uncertainty\n"
        "Point labels show share of movies kept"
    )
    ax.grid(alpha=0.2)
    if show_colorbar and fig is not None:
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("% movies kept")
    ax.legend(loc="upper left")


def compare_thresholds_by_keep_rate(
    comparison_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    *,
    comparison_label: str,
    reference_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for row in comparison_summary.itertuples(index=False):
        reference_row = reference_summary.iloc[
            (reference_summary["keep_rate"] - row.keep_rate).abs().argsort()[:1]
        ].iloc[0]
        rows.append(
            {
                f"{comparison_label}_threshold": row.threshold,
                f"{comparison_label}_keep_pct": 100 * row.keep_rate,
                f"{comparison_label}_gain": row.gain_vs_all,
                f"{comparison_label}_liked_pct": 100 * row.liked_rate,
                f"{reference_label}_threshold_nearest_keep": reference_row["threshold"],
                f"{reference_label}_keep_pct": 100 * reference_row["keep_rate"],
                f"{reference_label}_gain": reference_row["gain_vs_all"],
                f"{reference_label}_liked_pct": 100 * reference_row["liked_rate"],
                "keep_pct_gap": abs(
                    100 * row.keep_rate - 100 * reference_row["keep_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_super_comparison(
    rt_summary: pd.DataFrame,
    imdb_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    _plot_threshold_gain_axis(
        axes[0],
        rt_summary,
        score_label="RT audience (25-row subset)",
        threshold_axis_label="Minimum RT audience score (/10)",
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )
    _plot_threshold_gain_axis(
        axes[1],
        imdb_summary,
        score_label="IMDb (25-row subset)",
        threshold_axis_label="Minimum IMDb score (/10)",
        practical_thresholds=(6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0),
    )

    overlay_ax = axes[2]
    for table, label, color in (
        (rt_summary, "RT audience", "#1f77b4"),
        (imdb_summary, "IMDb", "#d62728"),
    ):
        drop_pct = 100 * (1.0 - table["keep_rate"].to_numpy(dtype=float))
        gain = table["gain_vs_all"].to_numpy(dtype=float)
        overlay_ax.plot(
            drop_pct,
            gain,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        for row in table.itertuples(index=False):
            overlay_ax.annotate(
                _format_threshold(row.threshold),
                (100 * (1.0 - row.keep_rate), row.gain_vs_all),
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                color=color,
            )
    overlay_ax.axhline(0, color="#2d3436", linestyle="--", linewidth=1.2, alpha=0.8)
    overlay_ax.set_xlabel("Drop %")
    overlay_ax.set_ylabel("Gain in average personal rating")
    overlay_ax.set_title(
        "RT audience vs IMDb on same subset\n" "Point labels show cutoff values"
    )
    overlay_ax.grid(alpha=0.2)
    overlay_ax.legend(loc="upper left")
    fig.suptitle(
        "RT audience vs IMDb threshold comparison",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gated_combo_threshold_panels(
    gated_tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(gated_tables), figsize=(7 * len(gated_tables), 6))
    if len(gated_tables) == 1:
        axes = [axes]
    for ax, (title, table) in zip(axes, gated_tables.items(), strict=True):
        _plot_threshold_gain_axis(
            ax,
            table,
            score_label=title,
            threshold_axis_label="Minimum average(RT audience, IMDb) (/10)",
            practical_thresholds=tuple(table["threshold"].tolist()),
        )
    fig.suptitle(
        "Combined-score thresholds after requiring both RT and IMDb to clear a gate",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_threshold_uncertainty_summary(
    output_path: Path,
    summary_table: pd.DataFrame,
    *,
    score_label: str,
    practical_thresholds: tuple[float, ...] = (60, 70, 75, 80, 85, 90),
) -> None:
    valid = summary_table.dropna(subset=["gain_vs_all"])
    best_gain_row = valid.sort_values("gain_vs_all", ascending=False).iloc[0]
    best_liked_row = (
        summary_table.dropna(subset=["liked_rate"])
        .sort_values("liked_rate", ascending=False)
        .iloc[0]
    )
    lines = [
        f"{score_label} threshold uncertainty summary:",
        (
            f"  Best average-rating gain: {_format_threshold(best_gain_row['threshold'])} "
            f"-> gain {best_gain_row['gain_vs_all']:.2f} "
            f"(90% band {best_gain_row['gain_vs_all_p05']:.2f} to {best_gain_row['gain_vs_all_p95']:.2f}), "
            f"keep {100 * best_gain_row['keep_rate']:.1f}%"
        ),
        (
            f"  Highest liked-share: {_format_threshold(best_liked_row['threshold'])} "
            f"-> liked {best_liked_row['liked_rate']:.1%} "
            f"(90% band {best_liked_row['liked_rate_p05']:.1%} to {best_liked_row['liked_rate_p95']:.1%}), "
            f"keep {100 * best_liked_row['keep_rate']:.1f}%"
        ),
        "",
        "Practical thresholds:",
    ]
    practical = summary_table.loc[
        summary_table["threshold"].isin(practical_thresholds)
    ].copy()
    for row in practical.itertuples(index=False):
        lines.append(
            f"  {_format_threshold(row.threshold)} -> gain {row.gain_vs_all:.2f} "
            f"[{row.gain_vs_all_p05:.2f}, {row.gain_vs_all_p95:.2f}], "
            f"liked {row.liked_rate:.1%}, keep {100 * row.keep_rate:.1f}%"
        )
    output_path.write_text("\n".join(lines) + "\n")


def plot_gemini_deviation(
    comparison: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    specs = [
        ("RT Audience", "verified_rt_audience", "RT audience"),
        ("RT Critic", "verified_rt_critic", "RT critic"),
        ("IMDb", "IMDb_corrected", "IMDb"),
        ("Letterboxd", "Letterboxd_corrected", "Letterboxd"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics: dict[str, dict[str, float]] = {}
    for ax, (x_column, y_column, label) in zip(axes.flat, specs, strict=True):
        subset = comparison[[x_column, y_column, "Movie Title"]].dropna()
        x = subset[x_column].to_numpy(dtype=float)
        y = subset[y_column].to_numpy(dtype=float)
        metrics[label] = {
            "mae": float(mean_absolute_error(y, x)),
            "rmse": float(math.sqrt(mean_squared_error(y, x))),
            "max_abs_error": float(np.abs(y - x).max()),
            "correlation": float(np.corrcoef(x, y)[0, 1]),
        }
        ax.scatter(x, y, alpha=0.8, color="#1f77b4")
        low = min(x.min(), y.min()) - 2
        high = max(x.max(), y.max()) + 2
        ax.plot([low, high], [low, high], linestyle="--", linewidth=2, color="#2d3436")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(
            f"{label}: Gemini first table vs corrected\n"
            f"MAE {metrics[label]['mae']:.2f}, max err {metrics[label]['max_abs_error']:.0f}"
        )
        ax.set_xlabel("Gemini first table")
        ax.set_ylabel("Corrected / verified")
        ax.grid(alpha=0.2)
    fig.suptitle("Gemini dataset drift against corrected values", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_verified_predictors(
    df: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    working = df.copy()
    working["loo_linear_prediction"] = leave_one_out_linear_predictions(
        working,
        ["rt_critic_on_10", "rt_audience_on_10"],
    )
    predictors = [
        ("rt_audience_on_10", "Audience (/10)"),
        ("rt_average_on_10", "Audience + critic average (/10)"),
        ("loo_linear_prediction", "LOO linear combo prediction (/10)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    metrics: dict[str, dict[str, float]] = {}
    y = working["my_rating"].to_numpy(dtype=float)
    for ax, (column, label) in zip(axes, predictors, strict=True):
        x = working[column].to_numpy(dtype=float)
        stats = regression_fit_metrics(x, y)
        metrics[column] = stats
        ax.scatter(x, y, alpha=0.8, color="#2a6f97")
        slope = stats["slope"]
        intercept = stats["intercept"]
        xs = np.linspace(min(x.min(), y.min()) - 0.2, max(x.max(), y.max()) + 0.2, 200)
        ax.plot(xs, xs, linestyle="--", linewidth=2, color="#7f8c8d")
        ax.plot(xs, slope * xs + intercept, linewidth=2, color="#c0392b")
        ax.set_title(
            f"{label}\n"
            f"slope {stats['slope']:.2f}, R² {stats['r2']:.2f}, MAE {stats['mae']:.2f}"
        )
        ax.set_xlabel(label)
        ax.set_xlim(0, 10.2)
        ax.set_ylim(0, 10.5)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("My rating (/10)")
    fig.suptitle("Verified RT predictors against my rating", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_audience_shape_comparison(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    x = df["rt_audience_on_10"].to_numpy(dtype=float)
    y = df["my_rating"].to_numpy(dtype=float)
    linear_predictions = cross_val_predict(
        LinearRegression(),
        x.reshape(-1, 1),
        y,
        cv=LeaveOneOut(),
    )
    isotonic_predictions = leave_one_out_isotonic_predictions(x, y)
    metrics = evaluate_audience_shape_models(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, predictions, title in (
        (axes[0], linear_predictions, "Audience only linear"),
        (axes[1], isotonic_predictions, "Audience only isotonic"),
    ):
        ax.scatter(predictions, y, alpha=0.8, color="#3a86ff")
        low = min(predictions.min(), y.min()) - 0.2
        high = max(predictions.max(), y.max()) + 0.2
        ax.plot([low, high], [low, high], linestyle="--", color="#2d3436", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Predicted rating")
        ax.set_xlim(0, 10.2)
        ax.set_ylim(0, 10.5)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("My rating")
    fig.suptitle("Does a monotone audience-only model help?", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_bucket_accuracy(bucket_metrics: pd.DataFrame, output_path: Path) -> None:
    schemes = sorted(bucket_metrics["scheme"].unique().tolist())
    labels = {
        "critic_only": "Critic only",
        "audience_only": "Audience only",
        "average_only": "Average only",
        "audience_critic": "Audience + critic",
    }
    fig, axes = plt.subplots(
        1, len(schemes), figsize=(6 * len(schemes), 5), sharey=True
    )
    if len(schemes) == 1:
        axes = [axes]
    for ax, scheme in zip(axes, schemes, strict=True):
        subset = bucket_metrics.loc[bucket_metrics["scheme"] == scheme].copy()
        subset["label"] = subset["model"].map(labels)
        ax.bar(subset["label"], subset["accuracy"], color="#4c956c")
        ax.set_title(f"{scheme.replace('_', '-')} bucket accuracy")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        for row in subset.itertuples(index=False):
            ax.text(
                row.label,
                row.accuracy + 0.01,
                f"{row.accuracy:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Leave-one-out accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_imdb_rt_correlation(df: pd.DataFrame, output_path: Path) -> dict[str, float]:
    x = df["rt_audience_rating"].to_numpy(dtype=float)
    y = df["IMDb"].to_numpy(dtype=float)
    correlation = float(np.corrcoef(x, y)[0, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.8, color="#8338ec")
    low = min(x.min(), y.min()) - 2
    high = max(x.max(), y.max()) + 2
    ax.plot([low, high], [low, high], linestyle="--", linewidth=2, color="#2d3436")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Verified RT audience")
    ax.set_ylabel("IMDb")
    ax.set_title(
        f"RT audience vs IMDb\nr = {correlation:.3f}, R² = {correlation**2:.3f}"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"correlation": correlation, "r2": correlation**2}


def plot_imdb_rt_model_bars(
    continuous_metrics: pd.DataFrame,
    bucket_metrics_3: pd.DataFrame,
    bucket_metrics_5: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = {
        "audience_only": "Audience",
        "imdb_only": "IMDb",
        "audience_imdb": "Audience + IMDb",
        "audience_critic_imdb": "Audience + critic + IMDb",
        "audience_imdb_zavg": "Z avg (aud+IMDb)",
        "all3_zavg": "Z avg (all 3)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cont = continuous_metrics.copy()
    cont["label"] = cont["model"].map(labels)
    axes[0].bar(cont["label"], cont["cv_r2"], color="#577590")
    axes[0].set_title("25-movie continuous CV R²")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].axhline(0, linestyle="--", color="#2d3436")
    axes[0].grid(axis="y", alpha=0.2)

    for ax, table, title in (
        (axes[1], bucket_metrics_3, "25-movie 3-step accuracy"),
        (axes[2], bucket_metrics_5, "25-movie 5-step accuracy"),
    ):
        plot_df = table.copy()
        plot_df["label"] = plot_df["model"].map(labels)
        ax.bar(plot_df["label"], plot_df["accuracy"], color="#43aa8b")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_direct_bucket_accuracy(
    accuracy_table: pd.DataFrame, output_path: Path
) -> None:
    labels = {
        "audience_3": "Audience 3-step",
        "average_3": "Average 3-step",
        "audience_4": "Audience 4-step",
        "average_4": "Average 4-step",
    }
    plot_df = accuracy_table.copy()
    plot_df["label"] = plot_df["table"].map(labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["label"], plot_df["accuracy"], color="#577590")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Direct binned-tier accuracy")
    ax.tick_params(axis="x", rotation=20)
    for row in plot_df.itertuples(index=False):
        ax.text(
            row.label,
            row.accuracy + 0.01,
            f"{row.accuracy:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff_curves(
    threshold_tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    color_map = {
        "Audience": "#1f77b4",
        "Average": "#d62728",
    }
    for label, table in threshold_tables.items():
        ax.plot(
            table["threshold"],
            table["gain_vs_all"],
            marker="o",
            linewidth=2,
            color=color_map[label],
            label=label,
        )
        for row in table.itertuples(index=False):
            if math.isnan(row.gain_vs_all):
                continue
            ax.text(
                row.threshold,
                row.gain_vs_all,
                f"{100 * row.keep_rate:.0f}%",
                fontsize=8,
                ha="center",
                va="bottom",
                color=color_map[label],
            )
    ax.axhline(0, color="#2d3436", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Minimum RT threshold (%)")
    ax.set_ylabel("Gain in average personal rating")
    ax.set_title("Thresholding tradeoff: rating gain vs cutoff")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_followup_summary(
    output_path: Path,
    *,
    gemini_metrics: dict[str, dict[str, float]],
    predictor_metrics: dict[str, dict[str, float]],
    bucket_metrics: pd.DataFrame,
    direct_bucket_metrics: pd.DataFrame,
    threshold_tables: dict[str, pd.DataFrame],
) -> None:
    best_3 = bucket_metrics.loc[bucket_metrics["scheme"] == "3_step"].iloc[0]
    best_4 = bucket_metrics.loc[bucket_metrics["scheme"] == "4_step"].iloc[0]
    best_5 = bucket_metrics.loc[bucket_metrics["scheme"] == "5_step"].iloc[0]
    direct_3 = direct_bucket_metrics.loc[
        direct_bucket_metrics["table"] == "audience_3"
    ].iloc[0]
    direct_4 = direct_bucket_metrics.loc[
        direct_bucket_metrics["table"] == "audience_4"
    ].iloc[0]
    audience_tradeoff = threshold_tables["Audience"].sort_values(
        "gain_vs_all", ascending=False
    )
    average_tradeoff = threshold_tables["Average"].sort_values(
        "gain_vs_all", ascending=False
    )
    lines = [
        "Gemini drift vs corrected values:",
    ]
    for label, stats in gemini_metrics.items():
        lines.append(
            f"  {label}: MAE {stats['mae']:.2f}, RMSE {stats['rmse']:.2f}, "
            f"max abs error {stats['max_abs_error']:.0f}"
        )
    lines.extend(
        [
            "",
            "Verified continuous predictors:",
            (
                "  Audience: "
                f"slope {predictor_metrics['rt_audience_on_10']['slope']:.2f}, "
                f"R² {predictor_metrics['rt_audience_on_10']['r2']:.3f}, "
                f"MAE {predictor_metrics['rt_audience_on_10']['mae']:.3f}"
            ),
            (
                "  Average: "
                f"slope {predictor_metrics['rt_average_on_10']['slope']:.2f}, "
                f"R² {predictor_metrics['rt_average_on_10']['r2']:.3f}, "
                f"MAE {predictor_metrics['rt_average_on_10']['mae']:.3f}"
            ),
            (
                "  LOO linear combo: "
                f"slope {predictor_metrics['loo_linear_prediction']['slope']:.2f}, "
                f"R² {predictor_metrics['loo_linear_prediction']['r2']:.3f}, "
                f"MAE {predictor_metrics['loo_linear_prediction']['mae']:.3f}"
            ),
            "",
            "Bucket prediction accuracy:",
            (
                f"  Best 3-step model: {best_3['model']} "
                f"(accuracy {best_3['accuracy']:.3f}, macro F1 {best_3['macro_f1']:.3f})"
            ),
            (
                f"  Best 4-step model: {best_4['model']} "
                f"(accuracy {best_4['accuracy']:.3f}, macro F1 {best_4['macro_f1']:.3f})"
            ),
            (
                f"  Best 5-step model: {best_5['model']} "
                f"(accuracy {best_5['accuracy']:.3f}, macro F1 {best_5['macro_f1']:.3f}, "
                f"QWK {best_5['quadratic_kappa']:.3f})"
            ),
            f"  Direct audience 3-step tier accuracy: {direct_3['accuracy']:.3f}",
            f"  Direct audience 4-step tier accuracy: {direct_4['accuracy']:.3f}",
            "",
            "Threshold tradeoff peaks:",
            (
                "  Audience: "
                f"threshold {int(audience_tradeoff.iloc[0]['threshold'])}% -> "
                f"gain {audience_tradeoff.iloc[0]['gain_vs_all']:.2f}, "
                f"keep {100 * audience_tradeoff.iloc[0]['keep_rate']:.1f}%"
            ),
            (
                "  Average: "
                f"threshold {int(average_tradeoff.iloc[0]['threshold'])}% -> "
                f"gain {average_tradeoff.iloc[0]['gain_vs_all']:.2f}, "
                f"keep {100 * average_tradeoff.iloc[0]['keep_rate']:.1f}%"
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
