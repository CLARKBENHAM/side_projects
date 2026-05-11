from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_DROP_FRACTIONS = (0.2, 0.5, 0.8)
MIN_TOTAL_BOOKS = 80
MIN_TRAIN_BOOKS = 60
MIN_HOLDOUT_BOOKS = 20
TAIL_HOLDOUT_SHARE = 0.2
FITTED_MODEL_NAMES = ("goodreads_linear", "goodreads_category_ridge")
MIN_CV_VALIDATION_BOOKS = 15
MIN_UNIQUE_USER_RATINGS = 3


@dataclass(frozen=True)
class HoldoutSplit:
    split_type: str
    holdout_label: str
    n_train: int
    n_holdout: int


def profile_has_sufficient_rating_variation(
    profile_books: pd.DataFrame,
    min_unique_user_ratings: int = MIN_UNIQUE_USER_RATINGS,
) -> bool:
    user_ratings = pd.to_numeric(profile_books["user_rating"], errors="coerce")
    return int(user_ratings.dropna().nunique()) >= min_unique_user_ratings


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return math.nan
    return float(left.corr(right, method="spearman"))


def _safe_pearson(left: pd.Series, right: pd.Series) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return math.nan
    return float(left.corr(right, method="pearson"))


def choose_holdout_split(
    profile_books: pd.DataFrame,
    min_total_books: int = MIN_TOTAL_BOOKS,
    min_train_books: int = MIN_TRAIN_BOOKS,
    min_holdout_books: int = MIN_HOLDOUT_BOOKS,
    tail_holdout_share: float = TAIL_HOLDOUT_SHARE,
) -> tuple[pd.DataFrame, pd.DataFrame, HoldoutSplit] | None:
    frame = profile_books.sort_values(["event_date", "title"]).reset_index(drop=True)
    if len(frame) < min_total_books:
        return None

    yearly_counts = (
        frame["event_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index(ascending=False)
    )
    for year, holdout_count in yearly_counts.items():
        if holdout_count < min_holdout_books:
            continue
        holdout_mask = frame["event_year"].eq(year)
        train_mask = frame["event_year"].lt(year)
        train_count = int(train_mask.sum())
        if train_count < min_train_books:
            continue
        return (
            frame.loc[train_mask].reset_index(drop=True),
            frame.loc[holdout_mask].reset_index(drop=True),
            HoldoutSplit(
                split_type="calendar_year",
                holdout_label=str(year),
                n_train=train_count,
                n_holdout=int(holdout_count),
            ),
        )

    holdout_size = max(
        min_holdout_books, int(math.ceil(len(frame) * tail_holdout_share))
    )
    if len(frame) - holdout_size < min_train_books:
        return None
    train = frame.iloc[:-holdout_size].reset_index(drop=True)
    holdout = frame.iloc[-holdout_size:].reset_index(drop=True)
    start = holdout["event_date"].min()
    end = holdout["event_date"].max()
    label = "tail"
    if pd.notna(start) and pd.notna(end):
        label = f"{start.date()} to {end.date()}"
    return (
        train,
        holdout,
        HoldoutSplit(
            split_type="chronological_tail",
            holdout_label=label,
            n_train=len(train),
            n_holdout=len(holdout),
        ),
    )


def _design_matrix(
    frame: pd.DataFrame,
    include_category: bool,
    reference_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    data = pd.DataFrame(
        {
            "goodreads_avg_rating": pd.to_numeric(
                frame["average_rating"], errors="coerce"
            ),
        },
        index=frame.index,
    )
    median_value = float(data["goodreads_avg_rating"].median())
    if not np.isfinite(median_value):
        median_value = 0.0
    data["goodreads_avg_rating"] = data["goodreads_avg_rating"].fillna(median_value)

    if include_category:
        categories = pd.get_dummies(
            frame["category"].fillna("General Reading"),
            prefix="category",
            dtype=float,
        )
        data = pd.concat([data, categories], axis=1)

    if reference_columns is not None:
        data = data.reindex(columns=reference_columns, fill_value=0.0)
        return data, list(reference_columns)
    return data, data.columns.tolist()


def _fit_ridge(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    include_category: bool,
) -> np.ndarray:
    X_train, columns = _design_matrix(train_df, include_category=include_category)
    X_holdout, _ = _design_matrix(
        holdout_df,
        include_category=include_category,
        reference_columns=columns,
    )
    y_train = pd.to_numeric(train_df["user_rating"], errors="coerce").to_numpy()
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_holdout)
    return np.clip(preds, 1.0, 5.0)


def build_score_map(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    baseline_prediction = np.full(
        len(holdout_df), float(train_df["user_rating"].mean())
    )
    return {
        "baseline_mean": baseline_prediction,
        "goodreads_raw": holdout_df["average_rating"].to_numpy(),
        "goodreads_linear": _fit_ridge(train_df, holdout_df, include_category=False),
        "goodreads_category_ridge": _fit_ridge(
            train_df,
            holdout_df,
            include_category=True,
        ),
    }


def prediction_metrics(
    actual_ratings: pd.Series,
    scores: pd.Series,
) -> dict[str, float]:
    actual = pd.to_numeric(actual_ratings, errors="coerce")
    pred = pd.to_numeric(scores, errors="coerce")
    aligned = pd.DataFrame({"actual": actual, "score": pred}).dropna()
    if aligned.empty:
        return {
            "mae": math.nan,
            "pearson_r": math.nan,
            "spearman_rho": math.nan,
        }
    return {
        "mae": float((aligned["actual"] - aligned["score"]).abs().mean()),
        "pearson_r": _safe_pearson(aligned["score"], aligned["actual"]),
        "spearman_rho": _safe_spearman(aligned["score"], aligned["actual"]),
    }


def cross_validate_fitted_models(
    train_df: pd.DataFrame,
    model_names: tuple[str, ...] = FITTED_MODEL_NAMES,
    min_validation_books: int = MIN_CV_VALIDATION_BOOKS,
) -> pd.DataFrame:
    ordered = train_df.sort_values(["event_date", "title"]).reset_index(drop=True)
    max_splits = min(
        3,
        max(0, int(len(ordered) / min_validation_books) - 1),
    )
    if max_splits < 2:
        return pd.DataFrame()

    splitter = TimeSeriesSplit(n_splits=max_splits)
    fold_rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, validation_idx) in enumerate(splitter.split(ordered), 1):
        fold_train = ordered.iloc[train_idx].reset_index(drop=True)
        fold_validation = ordered.iloc[validation_idx].reset_index(drop=True)
        score_map = build_score_map(fold_train, fold_validation)
        actual = fold_validation["user_rating"]
        for model_name in model_names:
            metrics = prediction_metrics(
                actual,
                pd.Series(score_map[model_name], index=fold_validation.index),
            )
            fold_rows.append(
                {
                    "model_name": model_name,
                    "fold_idx": fold_idx,
                    "n_train": len(fold_train),
                    "n_validation": len(fold_validation),
                    **metrics,
                }
            )

    if not fold_rows:
        return pd.DataFrame()

    folds = pd.DataFrame(fold_rows)
    return (
        folds.groupby("model_name", as_index=False)
        .agg(
            cv_folds=("fold_idx", "nunique"),
            cv_mean_mae=("mae", "mean"),
            cv_mean_pearson_r=("pearson_r", "mean"),
            cv_mean_spearman_rho=("spearman_rho", "mean"),
        )
        .sort_values(
            ["cv_mean_spearman_rho", "cv_mean_mae", "model_name"],
            ascending=[False, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def select_best_fitted_model(
    train_df: pd.DataFrame,
    model_names: tuple[str, ...] = FITTED_MODEL_NAMES,
) -> tuple[str, pd.DataFrame]:
    cv_results = cross_validate_fitted_models(train_df, model_names=model_names)
    if cv_results.empty:
        fallback = model_names[0]
        return fallback, pd.DataFrame(
            [
                {
                    "model_name": fallback,
                    "cv_folds": 0,
                    "cv_mean_mae": math.nan,
                    "cv_mean_pearson_r": math.nan,
                    "cv_mean_spearman_rho": math.nan,
                }
            ]
        )

    best_row = cv_results.iloc[0]
    return str(best_row["model_name"]), cv_results


def drop_curve(
    actual_ratings: pd.Series,
    scores: pd.Series,
    drop_fractions: tuple[float, ...] = DEFAULT_DROP_FRACTIONS,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(actual_ratings, errors="coerce"),
            "score": pd.to_numeric(scores, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame()

    baseline = float(frame["actual"].mean())
    if frame["score"].nunique(dropna=True) < 2:
        rows = []
        for drop_fraction in drop_fractions:
            keep_share = max(0.0, 1.0 - drop_fraction)
            n_kept = int(math.ceil(len(frame) * keep_share))
            rows.append(
                {
                    "drop_fraction": drop_fraction,
                    "keep_share": keep_share,
                    "baseline_mean": baseline,
                    "kept_mean": baseline,
                    "rating_gain": 0.0,
                    "n_books": len(frame),
                    "n_kept": n_kept,
                }
            )
        return pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    ordered = (
        frame.assign(_tie_breaker=rng.random(len(frame)))
        .sort_values(
            ["score", "_tie_breaker"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, float]] = []
    for drop_fraction in drop_fractions:
        drop_count = int(math.floor(len(ordered) * drop_fraction))
        kept = ordered.iloc[drop_count:]
        if kept.empty:
            continue
        keep_share = len(kept) / len(ordered)
        rows.append(
            {
                "drop_fraction": drop_fraction,
                "keep_share": keep_share,
                "baseline_mean": baseline,
                "kept_mean": float(kept["actual"].mean()),
                "rating_gain": float(kept["actual"].mean() - baseline),
                "n_books": len(ordered),
                "n_kept": len(kept),
            }
        )
    return pd.DataFrame(rows)


def expected_gain_from_correlation(
    std_dev: float, correlation: float, keep_share: float
) -> float:
    if not np.isfinite(std_dev) or not np.isfinite(correlation) or keep_share <= 0:
        return math.nan
    if keep_share >= 1:
        return 0.0
    z_score = NormalDist().inv_cdf(1 - keep_share)
    lambda_factor = NormalDist().pdf(z_score) / keep_share
    return float(std_dev * correlation * lambda_factor)


def evaluate_profile(
    profile_books: pd.DataFrame,
    drop_fractions: tuple[float, ...] = DEFAULT_DROP_FRACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]] | None:
    if not profile_has_sufficient_rating_variation(profile_books):
        return None
    split = choose_holdout_split(profile_books)
    if split is None:
        return None
    train_df, holdout_df, split_info = split
    score_map = build_score_map(train_df, holdout_df)

    actual = pd.to_numeric(holdout_df["user_rating"], errors="coerce")
    std_dev = float(actual.std()) if len(actual) > 1 else math.nan
    metric_rows: list[dict[str, object]] = []
    drop_rows: list[pd.DataFrame] = []
    for model_name, predictions in score_map.items():
        pred_series = pd.Series(predictions, index=holdout_df.index, dtype=float)
        metrics = prediction_metrics(actual, pred_series)
        metric_rows.append(
            {
                "profile_slug": str(holdout_df["profile_slug"].iloc[0]),
                "display_name": str(holdout_df["display_name"].iloc[0]),
                "split_type": split_info.split_type,
                "holdout_label": split_info.holdout_label,
                "model_name": model_name,
                "n_train": split_info.n_train,
                "n_holdout": split_info.n_holdout,
                "holdout_rating_mean": float(actual.mean()),
                "holdout_rating_std": std_dev,
                **metrics,
            }
        )
        curve = drop_curve(actual, pred_series, drop_fractions=drop_fractions)
        if curve.empty:
            continue
        curve["profile_slug"] = str(holdout_df["profile_slug"].iloc[0])
        curve["display_name"] = str(holdout_df["display_name"].iloc[0])
        curve["split_type"] = split_info.split_type
        curve["holdout_label"] = split_info.holdout_label
        curve["model_name"] = model_name
        metric_pearson = metric_rows[-1]["pearson_r"]
        curve["expected_gain_gaussian"] = curve["keep_share"].map(
            lambda keep_share: expected_gain_from_correlation(
                std_dev,
                float(metric_pearson) if pd.notna(metric_pearson) else math.nan,
                keep_share,
            )
        )
        drop_rows.append(curve)

    summary = {
        "profile_slug": str(profile_books["profile_slug"].iloc[0]),
        "display_name": str(profile_books["display_name"].iloc[0]),
        "n_rated_books": int(len(profile_books)),
        "n_distinct_years": int(profile_books["event_year"].dropna().nunique()),
        "latest_event_year": (
            int(profile_books["event_year"].dropna().max())
            if profile_books["event_year"].notna().any()
            else None
        ),
        "dominant_year_share": (
            float(
                profile_books["event_year"]
                .dropna()
                .astype(int)
                .value_counts(normalize=True)
                .iloc[0]
            )
            if profile_books["event_year"].notna().any()
            else math.nan
        ),
        "split_type": split_info.split_type,
        "holdout_label": split_info.holdout_label,
        "n_train": split_info.n_train,
        "n_holdout": split_info.n_holdout,
    }
    return pd.DataFrame(metric_rows), pd.concat(drop_rows, ignore_index=True), summary
