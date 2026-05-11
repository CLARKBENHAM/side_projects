from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _normalized_order(frame: pd.DataFrame) -> np.ndarray:
    if len(frame) <= 1:
        return np.zeros(len(frame), dtype=float)
    return np.linspace(0.0, 1.0, len(frame), dtype=float)


def _ols_slope(y_values: pd.Series, x_values: np.ndarray) -> float:
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(x_values)
    if valid.sum() < 2:
        return math.nan
    x = x_values[valid]
    y = y[valid]
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom == 0:
        return math.nan
    return float(np.dot(x_centered, y - y.mean()) / denom)


def _quartile_delta(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    if len(numeric) < 8:
        return math.nan
    quartile_size = max(1, int(math.floor(len(numeric) * 0.25)))
    early = numeric.iloc[:quartile_size]
    late = numeric.iloc[-quartile_size:]
    return float(late.mean() - early.mean())


def compute_temporal_picking_profile_metrics(
    prepared_profile_books: pd.DataFrame,
    minimum_books: int = 100,
    minimum_years: int = 2,
    max_dominant_year_share: float = 0.5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, profile_books in prepared_profile_books.groupby("profile_slug", sort=True):
        ordered = profile_books.sort_values(["event_date", "title"]).reset_index(
            drop=True
        )
        years = ordered["event_year"].dropna()
        if len(ordered) < minimum_books or years.nunique() < minimum_years:
            continue
        dominant_year_share = float(
            years.astype(int).value_counts(normalize=True).iloc[0]
        )
        order = _normalized_order(ordered)
        rows.append(
            {
                "profile_slug": str(ordered["profile_slug"].iloc[0]),
                "display_name": str(ordered["display_name"].iloc[0]),
                "n_books": int(len(ordered)),
                "distinct_years": int(years.nunique()),
                "dominant_year_share": dominant_year_share,
                "passes_temporal_filter": dominant_year_share
                <= max_dominant_year_share,
                "user_rating_slope_full_span": _ols_slope(
                    ordered["user_rating"], order
                ),
                "goodreads_rating_slope_full_span": _ols_slope(
                    ordered["average_rating"], order
                ),
                "user_rating_late_minus_early": _quartile_delta(ordered["user_rating"]),
                "goodreads_rating_late_minus_early": _quartile_delta(
                    ordered["average_rating"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["passes_temporal_filter", "n_books", "display_name"],
        ascending=[False, False, True],
    )


def summarize_temporal_picking(
    profile_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if profile_metrics.empty:
        return pd.DataFrame()

    scope_frames = {
        "all_eligible_profiles": profile_metrics,
        "temporally_credible_profiles": profile_metrics[
            profile_metrics["passes_temporal_filter"]
        ],
    }
    summary_rows: list[dict[str, object]] = []
    for scope_name, scope_frame in scope_frames.items():
        if scope_frame.empty:
            continue
        for metric_name in [
            "user_rating_slope_full_span",
            "goodreads_rating_slope_full_span",
            "user_rating_late_minus_early",
            "goodreads_rating_late_minus_early",
        ]:
            values = scope_frame[metric_name].dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "scope_name": scope_name,
                    "metric_name": metric_name,
                    "n_profiles": int(len(values)),
                    "mean_value": float(values.mean()) if len(values) else math.nan,
                    "median_value": (
                        float(np.median(values)) if len(values) else math.nan
                    ),
                    "p15_value": (
                        float(np.percentile(values, 15)) if len(values) else math.nan
                    ),
                    "p85_value": (
                        float(np.percentile(values, 85)) if len(values) else math.nan
                    ),
                }
            )
    return pd.DataFrame(summary_rows)
