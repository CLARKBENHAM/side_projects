from __future__ import annotations

from collections import defaultdict
import math
from itertools import combinations

import numpy as np
import pandas as pd


def compute_reviewer_volume_metrics(
    prepared_profile_books: pd.DataFrame,
    analysis_manifest: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    if prepared_profile_books.empty:
        empty = pd.DataFrame()
        return {
            "reviewer_volume_summary": empty,
            "reviewer_year_counts": empty,
            "yearly_activity_summary": empty,
        }

    observed_reviewer_volume = (
        prepared_profile_books.groupby(["profile_slug", "display_name"], as_index=False)
        .agg(
            observed_n_books=("title", "size"),
            n_active_years=("event_year", lambda years: years.dropna().nunique()),
        )
        .sort_values(["observed_n_books", "display_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    observed_reviewer_volume = observed_reviewer_volume.rename(
        columns={"display_name": "observed_display_name"}
    )
    reviewer_volume = observed_reviewer_volume.copy()
    reviewer_volume["display_name"] = reviewer_volume["observed_display_name"]
    reviewer_volume["total_rated_books"] = reviewer_volume["observed_n_books"]

    if analysis_manifest is not None and not analysis_manifest.empty:
        manifest_subset = (
            analysis_manifest[
                ["profile_slug", "display_name", "public_ratings", "source_group"]
            ]
            .drop_duplicates(subset=["profile_slug"])
            .rename(
                columns={
                    "display_name": "manifest_display_name",
                    "public_ratings": "public_ratings_total",
                }
            )
        )
        reviewer_volume = manifest_subset.merge(
            observed_reviewer_volume,
            on="profile_slug",
            how="left",
        )
        reviewer_volume["display_name"] = reviewer_volume[
            "manifest_display_name"
        ].fillna(reviewer_volume["observed_display_name"])
        reviewer_volume["observed_n_books"] = reviewer_volume[
            "observed_n_books"
        ].fillna(0)
        reviewer_volume["n_active_years"] = reviewer_volume["n_active_years"].fillna(0)
        reviewer_volume["total_rated_books"] = reviewer_volume[
            "public_ratings_total"
        ].fillna(reviewer_volume["observed_n_books"])

    reviewer_volume["books_per_active_year"] = np.where(
        reviewer_volume["n_active_years"] > 0,
        reviewer_volume["observed_n_books"] / reviewer_volume["n_active_years"],
        math.nan,
    )
    reviewer_volume["n_books"] = reviewer_volume["total_rated_books"]
    reviewer_volume["was_likely_capped_by_max_pages"] = reviewer_volume[
        "observed_n_books"
    ].between(400, 499) & (
        reviewer_volume["total_rated_books"] > reviewer_volume["observed_n_books"]
    )
    reviewer_volume = reviewer_volume.sort_values(
        ["total_rated_books", "display_name"],
        ascending=[False, True],
    ).reset_index(drop=True)

    reviewer_year_counts = (
        prepared_profile_books.dropna(subset=["event_year"])
        .groupby(["profile_slug", "display_name", "event_year"], as_index=False)
        .agg(n_books=("title", "size"))
        .sort_values(["profile_slug", "event_year"], ascending=[True, True])
        .reset_index(drop=True)
    )
    reviewer_year_counts["event_year"] = reviewer_year_counts["event_year"].astype(int)
    yearly_activity_summary = (
        reviewer_year_counts.groupby("event_year", as_index=False)
        .agg(
            active_reviewers=("profile_slug", "nunique"),
            total_reviews=("n_books", "sum"),
        )
        .sort_values("event_year")
        .reset_index(drop=True)
    )

    return {
        "reviewer_volume_summary": reviewer_volume,
        "reviewer_year_counts": reviewer_year_counts,
        "yearly_activity_summary": yearly_activity_summary,
    }


def _pair_corr_from_sums(
    n_obs: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    numerator = n_obs * sum_xy - sum_x * sum_y
    left = n_obs * sum_x2 - sum_x**2
    right = n_obs * sum_y2 - sum_y**2
    denominator = math.sqrt(left * right) if left > 0 and right > 0 else 0.0
    if denominator == 0:
        return math.nan
    return float(numerator / denominator)


def _safe_series_corr(
    left: pd.Series,
    right: pd.Series,
    method: str,
) -> float:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    if (
        len(aligned) < 2
        or aligned["left"].nunique(dropna=True) < 2
        or aligned["right"].nunique(dropna=True) < 2
    ):
        return math.nan
    correlation = aligned["left"].corr(aligned["right"], method=method)
    return float(correlation) if pd.notna(correlation) else math.nan


RATING_COUNT_BOOTSTRAP_METRICS = [
    "row_spearman_rho",
    "row_pearson_r",
    "row_profile_centered_pearson_r",
    "book_mean_spearman_rho",
    "book_mean_pearson_r",
    "mean_individual_absolute_error",
    "mean_book_absolute_error",
    "mean_sample_minus_goodreads",
]


def _rating_count_bin_metrics(
    row_subset: pd.DataFrame,
    book_subset: pd.DataFrame,
) -> dict[str, float]:
    return {
        "row_spearman_rho": _safe_series_corr(
            row_subset["average_rating"],
            row_subset["user_rating"],
            method="spearman",
        ),
        "row_pearson_r": _safe_series_corr(
            row_subset["average_rating"],
            row_subset["user_rating"],
            method="pearson",
        ),
        "row_profile_centered_pearson_r": _safe_series_corr(
            row_subset["average_rating"],
            row_subset["profile_centered_user_rating"],
            method="pearson",
        ),
        "book_mean_spearman_rho": _safe_series_corr(
            book_subset["average_goodreads_rating"],
            book_subset["mean_user_rating"],
            method="spearman",
        ),
        "book_mean_pearson_r": _safe_series_corr(
            book_subset["average_goodreads_rating"],
            book_subset["mean_user_rating"],
            method="pearson",
        ),
        "mean_individual_absolute_error": float(
            row_subset["individual_absolute_error"].mean()
        ),
        "mean_book_absolute_error": float(
            book_subset["book_mean_absolute_error"].mean()
        ),
        "mean_sample_minus_goodreads": float(book_subset["book_mean_error"].mean()),
    }


def _central_interval(
    values: list[float], interval_width: float
) -> tuple[float, float]:
    finite_values = np.asarray([value for value in values if np.isfinite(value)])
    if len(finite_values) == 0:
        return math.nan, math.nan
    lower = (1.0 - interval_width) / 2.0 * 100.0
    upper = (1.0 + interval_width) / 2.0 * 100.0
    return (
        float(np.percentile(finite_values, lower)),
        float(np.percentile(finite_values, upper)),
    )


def _bootstrap_rating_count_bin_metrics(
    row_subset: pd.DataFrame,
    book_subset: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, float]:
    if bootstrap_samples <= 0 or len(book_subset) < 2:
        return {}

    book_ids = book_subset["book_id"].astype(str).to_numpy()
    book_lookup = book_subset.set_index("book_id", drop=False)
    row_lookup = {
        str(book_id): frame
        for book_id, frame in row_subset.groupby("book_id", sort=False)
    }
    bootstrap_values: dict[str, list[float]] = {
        metric: [] for metric in RATING_COUNT_BOOTSTRAP_METRICS
    }
    for _ in range(bootstrap_samples):
        sampled_book_ids = rng.choice(book_ids, size=len(book_ids), replace=True)
        sampled_books = book_lookup.loc[sampled_book_ids].reset_index(drop=True)
        sampled_rows = pd.concat(
            [row_lookup[str(book_id)] for book_id in sampled_book_ids],
            ignore_index=True,
        )
        sample_metrics = _rating_count_bin_metrics(sampled_rows, sampled_books)
        for metric_name, metric_value in sample_metrics.items():
            bootstrap_values[metric_name].append(metric_value)

    interval_columns: dict[str, float] = {}
    for metric_name, values in bootstrap_values.items():
        for interval_name, interval_width in [("p80", 0.80), ("p95", 0.95)]:
            low, high = _central_interval(values, interval_width)
            interval_columns[f"{metric_name}_{interval_name}_low"] = low
            interval_columns[f"{metric_name}_{interval_name}_high"] = high
    return interval_columns


def compute_book_overlap_metrics(
    prepared_profile_books: pd.DataFrame,
    minimum_shared_books_for_pair_corr: int = 5,
) -> dict[str, pd.DataFrame]:
    if prepared_profile_books.empty:
        empty = pd.DataFrame()
        return {
            "reviewer_pair_correlations": empty,
            "overlap_books": empty,
            "overlap_rater_count_summary": empty,
        }

    base = prepared_profile_books[
        [
            "book_id",
            "title",
            "author_name",
            "profile_slug",
            "display_name",
            "user_rating",
            "average_rating",
        ]
    ].copy()
    base["profile_slug"] = base["profile_slug"].astype(str)
    base["book_id"] = base["book_id"].astype(str)
    base["user_rating"] = pd.to_numeric(base["user_rating"], errors="coerce")
    base["average_rating"] = pd.to_numeric(base["average_rating"], errors="coerce")
    base = base.dropna(subset=["book_id", "profile_slug", "user_rating"])
    if base.empty:
        empty = pd.DataFrame()
        return {
            "reviewer_pair_correlations": empty,
            "overlap_books": empty,
            "overlap_rater_count_summary": empty,
        }

    profile_codes, profile_uniques = pd.factorize(base["profile_slug"], sort=True)
    display_lookup = (
        base[["profile_slug", "display_name"]]
        .drop_duplicates()
        .set_index("profile_slug")["display_name"]
        .to_dict()
    )
    base["profile_code"] = profile_codes.astype(int)
    pair_accumulators: dict[tuple[int, int], list[float]] = defaultdict(
        lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    for _, book_frame in base.groupby("book_id", sort=False):
        if len(book_frame) < 2:
            continue
        ordered = book_frame.sort_values("profile_code")
        profile_list = ordered["profile_code"].to_numpy(dtype=int)
        rating_list = ordered["user_rating"].to_numpy(dtype=float)
        for left_idx in range(len(profile_list) - 1):
            left_code = int(profile_list[left_idx])
            left_rating = float(rating_list[left_idx])
            for right_idx in range(left_idx + 1, len(profile_list)):
                right_code = int(profile_list[right_idx])
                right_rating = float(rating_list[right_idx])
                key = (left_code, right_code)
                accumulator = pair_accumulators[key]
                accumulator[0] += 1
                accumulator[1] += left_rating
                accumulator[2] += right_rating
                accumulator[3] += left_rating * left_rating
                accumulator[4] += right_rating * right_rating
                accumulator[5] += left_rating * right_rating

    reviewer_pair_rows: list[dict[str, object]] = []
    pair_corr_lookup: dict[tuple[int, int], float] = {}
    for (left_code, right_code), values in pair_accumulators.items():
        n_shared, sum_x, sum_y, sum_x2, sum_y2, sum_xy = values
        correlation = (
            _pair_corr_from_sums(
                int(n_shared),
                float(sum_x),
                float(sum_y),
                float(sum_x2),
                float(sum_y2),
                float(sum_xy),
            )
            if n_shared >= minimum_shared_books_for_pair_corr
            else math.nan
        )
        left_slug = str(profile_uniques[left_code])
        right_slug = str(profile_uniques[right_code])
        reviewer_pair_rows.append(
            {
                "left_profile_slug": left_slug,
                "right_profile_slug": right_slug,
                "left_display_name": display_lookup[left_slug],
                "right_display_name": display_lookup[right_slug],
                "n_shared_books": int(n_shared),
                "pearson_r": correlation,
            }
        )
        pair_corr_lookup[(left_code, right_code)] = correlation

    reviewer_pair_correlations = pd.DataFrame(reviewer_pair_rows)
    if not reviewer_pair_correlations.empty:
        reviewer_pair_correlations = reviewer_pair_correlations.sort_values(
            ["n_shared_books", "left_profile_slug", "right_profile_slug"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    overlap_rows: list[dict[str, object]] = []
    for book_id, book_frame in base.groupby("book_id", sort=False):
        if len(book_frame) < 2:
            continue
        ordered = book_frame.sort_values("profile_code")
        rating_values = ordered["user_rating"].to_numpy(dtype=float)
        profile_code_list = ordered["profile_code"].to_numpy(dtype=int)
        pairwise_corrs = [
            pair_corr_lookup[(left_code, right_code)]
            for left_code, right_code in combinations(profile_code_list, 2)
            if np.isfinite(pair_corr_lookup[(left_code, right_code)])
        ]
        overlap_rows.append(
            {
                "book_id": str(book_id),
                "title": str(ordered["title"].iloc[0]),
                "author_name": str(ordered["author_name"].iloc[0]),
                "n_raters": int(len(ordered)),
                "mean_user_rating": float(np.mean(rating_values)),
                "rating_sd": (
                    float(np.std(rating_values, ddof=1))
                    if len(rating_values) > 1
                    else math.nan
                ),
                "average_goodreads_rating": float(
                    ordered["average_rating"].dropna().mean()
                ),
                "mean_pairwise_profile_corr": (
                    float(np.mean(pairwise_corrs)) if pairwise_corrs else math.nan
                ),
                "n_pairwise_corrs_used": int(len(pairwise_corrs)),
                "reader_slugs": ";".join(ordered["profile_slug"].astype(str).tolist()),
            }
        )

    overlap_books = pd.DataFrame(overlap_rows)
    if not overlap_books.empty:
        overlap_books = overlap_books.sort_values(
            ["n_raters", "rating_sd", "title"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        overlap_rater_count_summary = (
            overlap_books.groupby("n_raters", as_index=False)
            .agg(
                n_books=("book_id", "size"),
                mean_rating_sd=("rating_sd", "mean"),
                median_rating_sd=("rating_sd", "median"),
                mean_pairwise_profile_corr=("mean_pairwise_profile_corr", "mean"),
                median_pairwise_profile_corr=("mean_pairwise_profile_corr", "median"),
                mean_pairwise_corr_count=("n_pairwise_corrs_used", "mean"),
            )
            .sort_values("n_raters")
            .reset_index(drop=True)
        )
    else:
        overlap_rater_count_summary = pd.DataFrame()

    return {
        "reviewer_pair_correlations": reviewer_pair_correlations,
        "overlap_books": overlap_books,
        "overlap_rater_count_summary": overlap_rater_count_summary,
    }


def compute_goodreads_rating_count_accuracy(
    prepared_profile_books: pd.DataFrame,
    book_rating_counts: pd.DataFrame,
    n_bins: int = 8,
    minimum_books_per_bin: int = 20,
    bootstrap_samples: int = 1000,
    random_seed: int = 314159,
) -> dict[str, pd.DataFrame]:
    if prepared_profile_books.empty or book_rating_counts.empty:
        empty = pd.DataFrame()
        return {
            "rating_count_matched_ratings": empty,
            "rating_count_book_metrics": empty,
            "rating_count_bin_summary": empty,
        }

    count_column = (
        "goodreads_rating_count"
        if "goodreads_rating_count" in book_rating_counts.columns
        else "rating_count"
    )
    if count_column not in book_rating_counts.columns:
        empty = pd.DataFrame()
        return {
            "rating_count_matched_ratings": empty,
            "rating_count_book_metrics": empty,
            "rating_count_bin_summary": empty,
        }

    counts = book_rating_counts[["book_id", count_column]].copy()
    counts = counts.rename(columns={count_column: "goodreads_rating_count"})
    counts["book_id"] = counts["book_id"].astype(str)
    counts["goodreads_rating_count"] = pd.to_numeric(
        counts["goodreads_rating_count"], errors="coerce"
    )
    counts = (
        counts.dropna(subset=["book_id", "goodreads_rating_count"])
        .loc[lambda frame: frame["goodreads_rating_count"].gt(0)]
        .sort_values(["book_id", "goodreads_rating_count"], ascending=[True, False])
        .drop_duplicates(subset=["book_id"], keep="first")
        .reset_index(drop=True)
    )
    if counts.empty:
        empty = pd.DataFrame()
        return {
            "rating_count_matched_ratings": empty,
            "rating_count_book_metrics": empty,
            "rating_count_bin_summary": empty,
        }

    required_columns = [
        "book_id",
        "title",
        "author_name",
        "profile_slug",
        "display_name",
        "user_rating",
        "average_rating",
    ]
    missing_columns = [
        column for column in required_columns if column not in prepared_profile_books
    ]
    if missing_columns:
        raise KeyError(
            "prepared_profile_books is missing required columns: "
            + ", ".join(missing_columns)
        )

    ratings = prepared_profile_books[required_columns].copy()
    ratings["book_id"] = ratings["book_id"].astype(str)
    ratings["profile_slug"] = ratings["profile_slug"].astype(str)
    ratings["user_rating"] = pd.to_numeric(ratings["user_rating"], errors="coerce")
    ratings["average_rating"] = pd.to_numeric(
        ratings["average_rating"], errors="coerce"
    )
    ratings = ratings.dropna(subset=["book_id", "profile_slug", "user_rating"])
    matched = ratings.merge(counts, on="book_id", how="inner")
    matched = matched.dropna(subset=["average_rating", "goodreads_rating_count"])
    if matched.empty:
        empty = pd.DataFrame()
        return {
            "rating_count_matched_ratings": empty,
            "rating_count_book_metrics": empty,
            "rating_count_bin_summary": empty,
        }

    profile_means = matched.groupby("profile_slug")["user_rating"].transform("mean")
    matched["profile_centered_user_rating"] = matched["user_rating"] - profile_means
    matched["individual_error"] = matched["user_rating"] - matched["average_rating"]
    matched["individual_absolute_error"] = matched["individual_error"].abs()

    book_metrics = (
        matched.groupby("book_id", as_index=False)
        .agg(
            title=("title", "first"),
            author_name=("author_name", "first"),
            goodreads_rating_count=("goodreads_rating_count", "first"),
            average_goodreads_rating=("average_rating", "mean"),
            mean_user_rating=("user_rating", "mean"),
            median_user_rating=("user_rating", "median"),
            n_sample_ratings=("user_rating", "size"),
            n_sample_profiles=("profile_slug", "nunique"),
            user_rating_sd=("user_rating", "std"),
            mean_profile_centered_user_rating=(
                "profile_centered_user_rating",
                "mean",
            ),
            mean_individual_absolute_error=("individual_absolute_error", "mean"),
        )
        .sort_values(["goodreads_rating_count", "title"], ascending=[True, True])
        .reset_index(drop=True)
    )
    book_metrics["book_mean_error"] = (
        book_metrics["mean_user_rating"] - book_metrics["average_goodreads_rating"]
    )
    book_metrics["book_mean_absolute_error"] = book_metrics["book_mean_error"].abs()
    book_metrics["log10_goodreads_rating_count"] = np.log10(
        book_metrics["goodreads_rating_count"]
    )

    bin_count = min(
        max(1, int(n_bins)),
        max(1, len(book_metrics) // max(1, int(minimum_books_per_bin))),
    )
    if book_metrics["log10_goodreads_rating_count"].nunique(dropna=True) < 2:
        book_metrics["_rating_count_bin"] = pd.Interval(
            left=float(book_metrics["log10_goodreads_rating_count"].min()),
            right=float(book_metrics["log10_goodreads_rating_count"].max()),
            closed="both",
        )
    else:
        book_metrics["_rating_count_bin"] = pd.qcut(
            book_metrics["log10_goodreads_rating_count"],
            q=bin_count,
            duplicates="drop",
        )
    bin_lookup = book_metrics.set_index("book_id")["_rating_count_bin"].to_dict()
    matched["_rating_count_bin"] = matched["book_id"].map(bin_lookup)

    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for bin_index, (bin_value, book_subset) in enumerate(
        book_metrics.groupby("_rating_count_bin", observed=True),
        start=1,
    ):
        row_subset = matched[matched["_rating_count_bin"].eq(bin_value)]
        if book_subset.empty or row_subset.empty:
            continue
        bin_metrics = _rating_count_bin_metrics(row_subset, book_subset)
        bootstrap_metrics = _bootstrap_rating_count_bin_metrics(
            row_subset,
            book_subset,
            rng=rng,
            bootstrap_samples=bootstrap_samples,
        )
        summary_rows.append(
            {
                "bin_index": bin_index,
                "rating_count_min": float(book_subset["goodreads_rating_count"].min()),
                "rating_count_median": float(
                    book_subset["goodreads_rating_count"].median()
                ),
                "rating_count_max": float(book_subset["goodreads_rating_count"].max()),
                "log10_rating_count_median": float(
                    np.log10(book_subset["goodreads_rating_count"].median())
                ),
                "n_books": int(len(book_subset)),
                "n_sample_ratings": int(len(row_subset)),
                "n_sample_profiles": int(row_subset["profile_slug"].nunique()),
                **bin_metrics,
                **bootstrap_metrics,
            }
        )

    bin_summary = pd.DataFrame(summary_rows)
    if not bin_summary.empty:
        bin_summary = bin_summary.sort_values("rating_count_median").reset_index(
            drop=True
        )

    matched = matched.drop(columns=["_rating_count_bin"]).reset_index(drop=True)
    book_metrics = book_metrics.drop(columns=["_rating_count_bin"]).reset_index(
        drop=True
    )
    return {
        "rating_count_matched_ratings": matched,
        "rating_count_book_metrics": book_metrics,
        "rating_count_bin_summary": bin_summary,
    }
