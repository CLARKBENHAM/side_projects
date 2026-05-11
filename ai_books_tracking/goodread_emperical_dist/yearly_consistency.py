from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ai_books_tracking.goodread_emperical_dist.evaluation import (
    drop_curve,
    prediction_metrics,
)
from ai_books_tracking.goodread_emperical_dist.policy_analysis import (
    POLICY_DROP_FRACTIONS,
    build_cutoff_curve,
    curve_metrics_at_threshold,
)


def _choose_best_threshold_with_cap(
    curve,
    max_drop_share: float = 0.8,
) -> dict[str, float]:
    if curve is None or len(curve.thresholds) == 0:
        return {
            "optimal_threshold": math.nan,
            "optimal_drop_share": math.nan,
            "optimal_gain": math.nan,
            "optimal_kept_mean": math.nan,
            "optimal_n_kept": 0,
        }

    eligible = np.where(curve.drop_shares <= max_drop_share)[0]
    if len(eligible) == 0:
        return {
            "optimal_threshold": math.nan,
            "optimal_drop_share": math.nan,
            "optimal_gain": math.nan,
            "optimal_kept_mean": math.nan,
            "optimal_n_kept": 0,
        }

    best = eligible[
        np.lexsort(
            (
                curve.thresholds[eligible],
                curve.drop_shares[eligible],
                -curve.rating_gains[eligible],
            )
        )[0]
    ]
    return {
        "optimal_threshold": float(curve.thresholds[best]),
        "optimal_drop_share": float(curve.drop_shares[best]),
        "optimal_gain": float(curve.rating_gains[best]),
        "optimal_kept_mean": float(curve.kept_means[best]),
        "optimal_n_kept": int(curve.n_kept[best]),
    }


def _build_year_frame(
    prepared_profile_books: pd.DataFrame,
    minimum_books_per_year: int,
) -> pd.DataFrame:
    return (
        prepared_profile_books.dropna(subset=["event_year"])
        .assign(
            event_year=lambda frame: pd.to_numeric(
                frame["event_year"], errors="coerce"
            ).astype("Int64")
        )
        .dropna(subset=["event_year"])
        .groupby(["profile_slug", "display_name", "event_year"])
        .filter(lambda frame: len(frame) >= minimum_books_per_year)
        .sort_values(["profile_slug", "event_year", "event_date", "title"])
        .reset_index(drop=True)
    )


def compute_yearly_consistency_outputs(
    prepared_profile_books: pd.DataFrame,
    minimum_books_per_year: int = 30,
    max_drop_share: float = 0.8,
    require_consecutive_years: bool = True,
) -> dict[str, pd.DataFrame]:
    eligible = _build_year_frame(prepared_profile_books, minimum_books_per_year)
    if eligible.empty:
        empty = pd.DataFrame()
        return {
            "yearly_profile_metrics": empty,
            "yearly_transfer_metrics": empty,
            "yearly_gain_differences": empty,
            "yearly_correlation_pairs": empty,
            "yearly_consistency_summary": empty,
        }

    yearly_rows: list[dict[str, object]] = []
    drop_rows: list[dict[str, object]] = []
    year_curve_lookup: dict[tuple[str, int], object] = {}
    for keys, year_frame in eligible.groupby(
        ["profile_slug", "display_name", "event_year"], sort=True
    ):
        profile_slug, display_name, event_year = keys
        curve = build_cutoff_curve(
            year_frame["user_rating"], year_frame["average_rating"]
        )
        metrics = prediction_metrics(
            pd.to_numeric(year_frame["user_rating"], errors="coerce"),
            pd.to_numeric(year_frame["average_rating"], errors="coerce"),
        )
        optimal = _choose_best_threshold_with_cap(curve, max_drop_share=max_drop_share)
        baseline_mean = float(
            pd.to_numeric(year_frame["user_rating"], errors="coerce").mean()
        )
        goodreads_mean = float(
            pd.to_numeric(year_frame["average_rating"], errors="coerce").mean()
        )
        yearly_rows.append(
            {
                "profile_slug": str(profile_slug),
                "display_name": str(display_name),
                "event_year": int(event_year),
                "n_books": int(len(year_frame)),
                "user_rating_mean": baseline_mean,
                "goodreads_rating_mean": goodreads_mean,
                "pearson_r": metrics["pearson_r"],
                "spearman_rho": metrics["spearman_rho"],
                **optimal,
            }
        )
        year_curve_lookup[(str(profile_slug), int(event_year))] = curve
        curve_rows = drop_curve(
            pd.to_numeric(year_frame["user_rating"], errors="coerce"),
            pd.to_numeric(year_frame["average_rating"], errors="coerce"),
            drop_fractions=POLICY_DROP_FRACTIONS,
        )
        for row in curve_rows.itertuples(index=False):
            drop_rows.append(
                {
                    "profile_slug": str(profile_slug),
                    "display_name": str(display_name),
                    "event_year": int(event_year),
                    "drop_fraction": float(row.drop_fraction),
                    "baseline_mean": float(row.baseline_mean),
                    "kept_mean": float(row.kept_mean),
                    "rating_gain": float(row.rating_gain),
                    "n_books": int(row.n_books),
                    "n_kept": int(row.n_kept),
                }
            )

    yearly_profile_metrics = pd.DataFrame(yearly_rows).sort_values(
        ["profile_slug", "event_year"]
    )
    yearly_drop_metrics = pd.DataFrame(drop_rows).sort_values(
        ["profile_slug", "event_year", "drop_fraction"]
    )

    transfer_rows: list[dict[str, object]] = []
    gain_difference_rows: list[dict[str, object]] = []
    correlation_pair_rows: list[dict[str, object]] = []
    for (profile_slug, display_name), profile_years in yearly_profile_metrics.groupby(
        ["profile_slug", "display_name"], sort=True
    ):
        ordered = profile_years.sort_values("event_year").reset_index(drop=True)
        for left_idx in range(len(ordered) - 1):
            year_one = ordered.iloc[left_idx]
            year_two = ordered.iloc[left_idx + 1]
            if (
                require_consecutive_years
                and int(year_two["event_year"]) != int(year_one["event_year"]) + 1
            ):
                continue

            key_two = (str(profile_slug), int(year_two["event_year"]))
            year_two_curve = year_curve_lookup[key_two]
            transfer_metrics = curve_metrics_at_threshold(
                year_two_curve,
                float(year_one["optimal_threshold"]),
            )
            transfer_rows.append(
                {
                    "profile_slug": str(profile_slug),
                    "display_name": str(display_name),
                    "year_1": int(year_one["event_year"]),
                    "year_2": int(year_two["event_year"]),
                    "n_books_year_1": int(year_one["n_books"]),
                    "n_books_year_2": int(year_two["n_books"]),
                    "optimal_threshold_year_1": float(year_one["optimal_threshold"]),
                    "optimal_drop_share_year_1": float(year_one["optimal_drop_share"]),
                    "optimal_gain_year_1": float(year_one["optimal_gain"]),
                    "optimal_threshold_year_2": float(year_two["optimal_threshold"]),
                    "optimal_drop_share_year_2": float(year_two["optimal_drop_share"]),
                    "optimal_gain_year_2": float(year_two["optimal_gain"]),
                    "year_2_gain_using_year_1_threshold": float(
                        transfer_metrics["rating_gain"]
                    ),
                    "year_2_drop_share_using_year_1_threshold": float(
                        transfer_metrics["actual_drop_share"]
                    ),
                    "year_2_regret_vs_optimal": float(year_two["optimal_gain"])
                    - float(transfer_metrics["rating_gain"]),
                    "year_2_optimal_efficiency": (
                        float(transfer_metrics["rating_gain"])
                        / float(year_two["optimal_gain"])
                        if pd.notna(year_two["optimal_gain"])
                        and float(year_two["optimal_gain"]) > 0
                        else math.nan
                    ),
                }
            )
            correlation_pair_rows.append(
                {
                    "profile_slug": str(profile_slug),
                    "display_name": str(display_name),
                    "year_1": int(year_one["event_year"]),
                    "year_2": int(year_two["event_year"]),
                    "n_books_year_1": int(year_one["n_books"]),
                    "n_books_year_2": int(year_two["n_books"]),
                    "spearman_rho_year_1": float(year_one["spearman_rho"]),
                    "spearman_rho_year_2": float(year_two["spearman_rho"]),
                    "spearman_rho_delta": float(year_two["spearman_rho"])
                    - float(year_one["spearman_rho"]),
                    "pearson_r_year_1": float(year_one["pearson_r"]),
                    "pearson_r_year_2": float(year_two["pearson_r"]),
                    "pearson_r_delta": float(year_two["pearson_r"])
                    - float(year_one["pearson_r"]),
                }
            )

            year_one_drop = yearly_drop_metrics[
                (yearly_drop_metrics["profile_slug"] == profile_slug)
                & (yearly_drop_metrics["event_year"] == int(year_one["event_year"]))
            ]
            year_two_drop = yearly_drop_metrics[
                (yearly_drop_metrics["profile_slug"] == profile_slug)
                & (yearly_drop_metrics["event_year"] == int(year_two["event_year"]))
            ]
            merged = year_one_drop.merge(
                year_two_drop,
                on=["profile_slug", "display_name", "drop_fraction"],
                suffixes=("_year_1", "_year_2"),
            )
            for merged_row in merged.itertuples(index=False):
                gain_difference_rows.append(
                    {
                        "profile_slug": str(profile_slug),
                        "display_name": str(display_name),
                        "year_1": int(year_one["event_year"]),
                        "year_2": int(year_two["event_year"]),
                        "drop_fraction": float(merged_row.drop_fraction),
                        "gain_year_1": float(merged_row.rating_gain_year_1),
                        "gain_year_2": float(merged_row.rating_gain_year_2),
                        "gain_difference_year_2_minus_year_1": float(
                            merged_row.rating_gain_year_2
                            - merged_row.rating_gain_year_1
                        ),
                    }
                )

    yearly_transfer_metrics = pd.DataFrame(transfer_rows)
    if not yearly_transfer_metrics.empty:
        yearly_transfer_metrics = yearly_transfer_metrics.sort_values(
            ["profile_slug", "year_1"]
        ).reset_index(drop=True)
    yearly_gain_differences = pd.DataFrame(gain_difference_rows)
    if not yearly_gain_differences.empty:
        yearly_gain_differences = yearly_gain_differences.sort_values(
            ["profile_slug", "year_1", "drop_fraction"]
        ).reset_index(drop=True)
    yearly_correlation_pairs = pd.DataFrame(correlation_pair_rows)
    if not yearly_correlation_pairs.empty:
        yearly_correlation_pairs = yearly_correlation_pairs.sort_values(
            ["profile_slug", "year_1"]
        ).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    if not yearly_transfer_metrics.empty:
        summary_rows.extend(
            [
                {
                    "metric_name": "year_2_optimal_efficiency",
                    "n_pairs": int(
                        yearly_transfer_metrics["year_2_optimal_efficiency"]
                        .notna()
                        .sum()
                    ),
                    "mean_value": float(
                        yearly_transfer_metrics["year_2_optimal_efficiency"]
                        .dropna()
                        .mean()
                    ),
                    "median_value": float(
                        yearly_transfer_metrics["year_2_optimal_efficiency"]
                        .dropna()
                        .median()
                    ),
                },
                {
                    "metric_name": "year_2_regret_vs_optimal",
                    "n_pairs": int(
                        yearly_transfer_metrics["year_2_regret_vs_optimal"]
                        .notna()
                        .sum()
                    ),
                    "mean_value": float(
                        yearly_transfer_metrics["year_2_regret_vs_optimal"]
                        .dropna()
                        .mean()
                    ),
                    "median_value": float(
                        yearly_transfer_metrics["year_2_regret_vs_optimal"]
                        .dropna()
                        .median()
                    ),
                },
                {
                    "metric_name": "optimal_drop_share_abs_change",
                    "n_pairs": int(
                        (
                            yearly_transfer_metrics["optimal_drop_share_year_1"]
                            - yearly_transfer_metrics["optimal_drop_share_year_2"]
                        )
                        .abs()
                        .notna()
                        .sum()
                    ),
                    "mean_value": float(
                        (
                            yearly_transfer_metrics["optimal_drop_share_year_1"]
                            - yearly_transfer_metrics["optimal_drop_share_year_2"]
                        )
                        .abs()
                        .dropna()
                        .mean()
                    ),
                    "median_value": float(
                        (
                            yearly_transfer_metrics["optimal_drop_share_year_1"]
                            - yearly_transfer_metrics["optimal_drop_share_year_2"]
                        )
                        .abs()
                        .dropna()
                        .median()
                    ),
                },
            ]
        )
    if not yearly_correlation_pairs.empty:
        for left_name, right_name, label in [
            ("spearman_rho_year_1", "spearman_rho_year_2", "spearman_year_to_year"),
            ("pearson_r_year_1", "pearson_r_year_2", "pearson_year_to_year"),
        ]:
            aligned = yearly_correlation_pairs[[left_name, right_name]].dropna()
            summary_rows.append(
                {
                    "metric_name": label,
                    "n_pairs": int(len(aligned)),
                    "mean_value": (
                        float(aligned[left_name].corr(aligned[right_name]))
                        if len(aligned) >= 2
                        else math.nan
                    ),
                    "median_value": (
                        float((aligned[right_name] - aligned[left_name]).median())
                        if not aligned.empty
                        else math.nan
                    ),
                }
            )
    yearly_consistency_summary = pd.DataFrame(summary_rows)

    return {
        "yearly_profile_metrics": yearly_profile_metrics,
        "yearly_transfer_metrics": yearly_transfer_metrics,
        "yearly_gain_differences": yearly_gain_differences,
        "yearly_correlation_pairs": yearly_correlation_pairs,
        "yearly_consistency_summary": yearly_consistency_summary,
    }
