from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from analysis_core.calendar_movie_scores import http_get_text
from analysis_core.movie_rt_second_pass_analysis import (
    CONTEXT_BINARY_FEATURES,
    CONTEXT_CATEGORICAL_FEATURES,
    CONTEXT_NUMERIC_FEATURES,
    build_pipeline_predictions,
)

RT_METADATA_GENRES_PATTERN = re.compile(r"metadataGenres\"?\s*:\s*\[(?P<body>[^\]]*)\]")
RT_CAG_GENRE_PATTERN = re.compile(r'cag\[genre\]:"(?P<body>[^"]+)"')


def evaluate_regression_predictions(
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
            "within_one": float("nan"),
            "rounded_accuracy": float("nan"),
        }
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    return {
        "rows": float(valid_mask.sum()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(((y_true - y_pred) ** 2).mean())),
        "correlation": (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if valid_mask.sum() >= 2
            else float("nan")
        ),
        "within_one": float((np.abs(y_true - y_pred) <= 1.0).mean()),
        "rounded_accuracy": float((np.clip(np.rint(y_pred), 1, 10) == y_true).mean()),
    }


def loo_linear_feature_set_metrics(
    df: pd.DataFrame,
    *,
    subset_name: str,
    feature_sets: dict[str, list[str]],
    target_column: str = "analysis_rating",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for model_name, feature_columns in feature_sets.items():
        working = df[[target_column, *feature_columns]].dropna().copy()
        x = working[feature_columns].to_numpy(dtype=float)
        y = working[target_column].to_numpy(dtype=float)
        predictions = cross_val_predict(LinearRegression(), x, y, cv=LeaveOneOut())
        metrics = evaluate_regression_predictions(
            working[target_column],
            pd.Series(predictions, index=working.index),
        )
        metrics_rows.append(
            {
                "subset": subset_name,
                "model": model_name,
                "feature_columns": "|".join(feature_columns),
                **metrics,
            }
        )
        for fold_index, (train_index, _) in enumerate(LeaveOneOut().split(x)):
            model = LinearRegression().fit(x[train_index], y[train_index])
            coefficient_rows.append(
                {
                    "subset": subset_name,
                    "model": model_name,
                    "fold_index": fold_index,
                    "feature": "intercept",
                    "coefficient": float(model.intercept_),
                }
            )
            for feature_name, coefficient in zip(
                feature_columns,
                model.coef_,
                strict=True,
            ):
                coefficient_rows.append(
                    {
                        "subset": subset_name,
                        "model": model_name,
                        "fold_index": fold_index,
                        "feature": feature_name,
                        "coefficient": float(coefficient),
                    }
                )
    return pd.DataFrame(metrics_rows), pd.DataFrame(coefficient_rows)


def build_unique_movie_prediction_frame(
    second_sheet_predictions: pd.DataFrame,
) -> pd.DataFrame:
    unique = second_sheet_predictions.drop_duplicates(
        ["movie_title", "imdb_release_year", "imdb_url"]
    ).copy()
    return unique.rename(columns={"imdb_release_year": "release_year"})


def build_taste_numeric_holdout_table(
    *,
    unique_predictions: pd.DataFrame,
    synopsis_holdout: pd.DataFrame,
    taste_predictions: pd.DataFrame,
) -> pd.DataFrame:
    merged = synopsis_holdout[["movie_title", "release_year", "analysis_rating"]].merge(
        taste_predictions,
        on=["movie_title", "release_year"],
        suffixes=("_holdout", "_taste"),
    )
    merged = merged.merge(
        unique_predictions[
            [
                "movie_title",
                "release_year",
                "analysis_rating",
                "cv_linear_rt_prediction",
                "cv_linear_rt_imdb_prediction",
                "cv_ridge_context_prediction",
            ]
        ],
        on=["movie_title", "release_year", "analysis_rating"],
        how="inner",
    )
    return merged


def evaluate_holdout_blends(
    holdout_df: pd.DataFrame,
    *,
    numeric_prediction_columns: list[str],
    taste_prediction_column: str = "predicted_rating",
    target_column: str = "analysis_rating",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = holdout_df[target_column].to_numpy(dtype=float)
    taste_prediction = holdout_df[taste_prediction_column].to_numpy(dtype=float)
    metrics_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []

    def append_metric_row(model_name: str, prediction: np.ndarray) -> None:
        metrics = evaluate_regression_predictions(
            pd.Series(y_true),
            pd.Series(prediction),
        )
        metrics_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    append_metric_row("taste_agent", taste_prediction)
    for numeric_column in numeric_prediction_columns:
        numeric_prediction = holdout_df[numeric_column].to_numpy(dtype=float)
        append_metric_row(numeric_column, numeric_prediction)
        append_metric_row(
            f"blend50_{numeric_column}",
            0.5 * numeric_prediction + 0.5 * taste_prediction,
        )
        for alpha in np.linspace(0.0, 1.0, 101):
            blend = alpha * numeric_prediction + (1.0 - alpha) * taste_prediction
            alpha_rows.append(
                {
                    "numeric_model": numeric_column,
                    "alpha_numeric": float(alpha),
                    **evaluate_regression_predictions(
                        pd.Series(y_true),
                        pd.Series(blend),
                    ),
                }
            )
    return pd.DataFrame(metrics_rows), pd.DataFrame(alpha_rows)


def parse_rt_genres_from_html(html_text: str) -> list[str]:
    metadata_match = RT_METADATA_GENRES_PATTERN.search(html_text)
    if metadata_match is not None:
        body = metadata_match.group("body").strip()
        if body:
            return [
                genre.strip() for genre in json.loads(f"[{body}]") if str(genre).strip()
            ]
    cag_match = RT_CAG_GENRE_PATTERN.search(html_text)
    if cag_match is not None:
        return [
            part.strip() for part in cag_match.group("body").split("|") if part.strip()
        ]
    return []


def fetch_rt_genres(rt_url: str) -> list[str]:
    if not str(rt_url).strip():
        return []
    return parse_rt_genres_from_html(http_get_text(rt_url))


def load_json_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def genre_column_name(genre: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", genre.lower()).strip("_")
    return f"rt_genre__{slug}"


def enrich_with_rt_genres(
    df: pd.DataFrame,
    *,
    cache_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cache = load_json_cache(cache_path)
    url_to_genres: dict[str, list[str]] = {}
    unique_urls = sorted(
        {str(url).strip() for url in df["rt_url"].fillna("") if str(url).strip()}
    )
    for rt_url in unique_urls:
        cached = cache.get(rt_url)
        if cached is None:
            genres = fetch_rt_genres(rt_url)
            cache[rt_url] = {"genres": genres}
            save_json_cache(cache_path, cache)
        else:
            genres = list(cached.get("genres", []))
        url_to_genres[rt_url] = genres

    working = df.copy()
    working["rt_genres"] = [
        url_to_genres.get(str(rt_url).strip(), [])
        for rt_url in working["rt_url"].fillna("")
    ]
    working["rt_genres_missing"] = working["rt_genres"].map(lambda values: not values)
    genre_vocab = sorted(
        {genre for genres in working["rt_genres"] for genre in genres if genre}
    )
    genre_columns = [genre_column_name(genre) for genre in genre_vocab]
    for genre, column_name in zip(genre_vocab, genre_columns, strict=True):
        working[column_name] = working["rt_genres"].map(
            lambda genres, current=genre: float(current in genres)
        )

    genre_rows = []
    for rt_url in unique_urls:
        genres = url_to_genres.get(rt_url, [])
        genre_rows.append(
            {
                "rt_url": rt_url,
                "genre_count": len(genres),
                "genres": " | ".join(genres),
            }
        )
    return working, pd.DataFrame(genre_rows), genre_columns


def evaluate_rt_genre_model_lift(
    df: pd.DataFrame,
    *,
    cache_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched, genre_table, genre_columns = enrich_with_rt_genres(
        df,
        cache_path=cache_path,
    )
    base_prediction = build_pipeline_predictions(
        enriched,
        numeric_features=CONTEXT_NUMERIC_FEATURES,
        binary_features=CONTEXT_BINARY_FEATURES,
        categorical_features=CONTEXT_CATEGORICAL_FEATURES,
        estimator=Ridge(alpha=1.0),
    )
    genre_prediction = build_pipeline_predictions(
        enriched,
        numeric_features=CONTEXT_NUMERIC_FEATURES,
        binary_features=CONTEXT_BINARY_FEATURES + ["rt_genres_missing"] + genre_columns,
        categorical_features=CONTEXT_CATEGORICAL_FEATURES,
        estimator=Ridge(alpha=1.0),
    )
    metrics = pd.DataFrame(
        [
            {
                "model": "context_ridge_same_rows",
                **evaluate_regression_predictions(
                    enriched["analysis_rating"],
                    pd.Series(base_prediction, index=enriched.index),
                ),
            },
            {
                "model": "context_ridge_plus_rt_genres",
                **evaluate_regression_predictions(
                    enriched["analysis_rating"],
                    pd.Series(genre_prediction, index=enriched.index),
                ),
            },
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "rows": int(len(enriched)),
                "rows_with_rt_url": int(enriched["rt_url"].fillna("").ne("").sum()),
                "rows_with_genres": int((~enriched["rt_genres_missing"]).sum()),
                "unique_rt_urls": int(genre_table["rt_url"].nunique()),
                "unique_genres": int(len(genre_columns)),
            }
        ]
    )
    return metrics, coverage, genre_table


def build_data_quality_candidate_table(
    df: pd.DataFrame,
    *,
    prediction_column: str,
) -> pd.DataFrame:
    working = df.copy()
    source_cols = ["rt_critic_on_10", "rt_audience_on_10", "imdb_score"]
    working["source_spread"] = working[source_cols].max(axis=1) - working[
        source_cols
    ].min(axis=1)
    working["rt_imdb_gap"] = (working["rt_average_on_10"] - working["imdb_score"]).abs()
    working["model_residual"] = working["analysis_rating"] - working[prediction_column]
    working["external_residual"] = (
        working["analysis_rating"] - working["external_mean_on_10"]
    )
    working["suspicion_score"] = (
        working["quality_flags"].fillna("").ne("").astype(float) * 4.0
        + working["source_spread"].fillna(0.0)
        + working["rt_imdb_gap"].fillna(0.0)
        + working["model_residual"].abs().fillna(0.0) * 0.5
        + working["external_residual"].abs().fillna(0.0) * 0.5
    )
    columns = [
        "date",
        "movie_title",
        "watch_status",
        "analysis_rating",
        "quality_flags",
        "source_spread",
        "rt_imdb_gap",
        "model_residual",
        "external_residual",
        "suspicion_score",
        "rt_matched_title",
        "rt_release_year",
        "rt_url",
        "imdb_matched_title",
        "imdb_release_year",
        "imdb_url",
    ]
    available_columns = [column for column in columns if column in working.columns]
    return (
        working[available_columns]
        .sort_values(
            ["suspicion_score", "movie_title"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
