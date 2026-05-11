from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np
import pandas as pd

from ai_books_tracking.goodread_emperical_dist.evaluation import (
    drop_curve,
    profile_has_sufficient_rating_variation,
)
from ai_books_tracking.goodread_emperical_dist.policy_analysis import (
    DEFAULT_CUTOFF_SWEEP_THRESHOLDS,
    build_cutoff_curve,
    curve_metrics_at_threshold,
)

FIXED_RULE_NAME_ORDER = (
    "original_distribution",
    "goodreads_cutoff_4_0",
    "drop_lower_goodreads_half",
)
FIXED_RULE_LABELS = {
    "original_distribution": "Original distribution",
    "goodreads_cutoff_4_0": "Keep Goodreads >= 4.0",
    "drop_lower_goodreads_half": "Drop lower Goodreads half",
}
GENERIC_SHELF_TOKENS = {
    "",
    "adult",
    "book-club",
    "books",
    "did-not-finish",
    "ebook",
    "e-books",
    "favorites",
    "fiction",
    "kindle",
    "library",
    "maybe",
    "owned",
    "own",
    "physical",
    "read",
    "to-buy",
    "to-read",
    "wishlist",
}
DEFAULT_FIXED_RULE_DROP_FRACTIONS = tuple(np.round(np.arange(0.1, 0.91, 0.1), 2))


def _clean_profile_frame(profile_books: pd.DataFrame) -> pd.DataFrame:
    frame = profile_books.copy()
    frame["user_rating"] = pd.to_numeric(frame["user_rating"], errors="coerce")
    frame["average_rating"] = pd.to_numeric(frame["average_rating"], errors="coerce")
    frame = frame[
        frame["user_rating"].notna()
        & frame["average_rating"].notna()
        & frame["user_rating"].between(1, 5)
    ].copy()
    frame["title_key"] = (
        frame["title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    frame["book_id_key"] = frame["book_id"].fillna("").astype(str)
    return frame.reset_index(drop=True)


def _shares_by_rating(ratings: pd.Series) -> dict[int, float]:
    if ratings.empty:
        return {rating_value: math.nan for rating_value in range(1, 6)}
    counts = ratings.round().astype(int).value_counts(normalize=True)
    return {
        rating_value: float(counts.get(rating_value, 0.0))
        for rating_value in range(1, 6)
    }


def _top_category_string(categories: pd.Series, max_categories: int = 3) -> str:
    clean = categories.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    if clean.empty:
        return ""
    shares = clean.value_counts(normalize=True).head(max_categories)
    return "; ".join(f"{category} ({share:.0%})" for category, share in shares.items())


def _extract_shelf_tokens(user_shelves: pd.Series) -> list[str]:
    tokens: list[str] = []
    for shelf_text in user_shelves.dropna().astype(str):
        for token in shelf_text.split(","):
            clean = token.strip().lower()
            clean = re.sub(r"\s+", "-", clean)
            if clean and clean not in GENERIC_SHELF_TOKENS:
                tokens.append(clean)
    return tokens


def _top_shelf_string(user_shelves: pd.Series, max_tokens: int = 5) -> str:
    counts = Counter(_extract_shelf_tokens(user_shelves))
    if not counts:
        return ""
    total = sum(counts.values())
    top_tokens = counts.most_common(max_tokens)
    return "; ".join(f"{token} ({count / total:.0%})" for token, count in top_tokens)


def _non_ascii_share(values: pd.Series) -> float:
    clean = values.dropna().astype(str).str.strip()
    if clean.empty:
        return math.nan
    return float(clean.map(lambda value: not value.isascii()).mean())


def _profile_context_row(profile_books: pd.DataFrame) -> dict[str, object]:
    goodreads_rho = profile_books["user_rating"].corr(
        profile_books["average_rating"],
        method="spearman",
    )
    goodreads_pearson = profile_books["user_rating"].corr(
        profile_books["average_rating"],
        method="pearson",
    )
    return {
        "profile_slug": str(profile_books["profile_slug"].iloc[0]),
        "display_name": str(profile_books["display_name"].iloc[0]),
        "n_books": int(len(profile_books)),
        "baseline_mean": float(profile_books["user_rating"].mean()),
        "profile_rating_std_all_books": float(profile_books["user_rating"].std(ddof=0)),
        "goodreads_mean": float(profile_books["average_rating"].mean()),
        "goodreads_spearman_rho_all_books": (
            float(goodreads_rho) if pd.notna(goodreads_rho) else math.nan
        ),
        "goodreads_pearson_r_all_books": (
            float(goodreads_pearson) if pd.notna(goodreads_pearson) else math.nan
        ),
        "share_below_4_0": float((profile_books["average_rating"] < 4.0).mean()),
        "median_goodreads_rating": float(profile_books["average_rating"].median()),
        "non_ascii_title_share": _non_ascii_share(profile_books["title"]),
        "non_ascii_author_share": _non_ascii_share(profile_books["author_name"]),
        "top_categories": _top_category_string(profile_books["category"]),
        "top_shelves": _top_shelf_string(profile_books["user_shelves"]),
    }


def _keep_mask_from_bottom_fraction_rule(
    profile_books: pd.DataFrame,
    drop_fraction: float,
) -> pd.Series:
    n_books = len(profile_books)
    n_drop = int(math.floor(n_books * drop_fraction))
    if n_drop <= 0:
        return pd.Series(True, index=profile_books.index)
    if n_drop >= n_books:
        return pd.Series(False, index=profile_books.index)
    rng = np.random.default_rng(0)
    ordered = (
        profile_books.assign(_tie_breaker=rng.random(n_books))
        .sort_values(
            ["average_rating", "_tie_breaker"],
            ascending=[True, True],
            na_position="last",
            kind="mergesort",
        )
        .reset_index(drop=False)
    )
    drop_index = ordered["index"].iloc[:n_drop]
    keep_mask = ~profile_books.index.isin(drop_index)
    return pd.Series(keep_mask, index=profile_books.index)


def _rule_metric_row(
    profile_books: pd.DataFrame,
    keep_mask: pd.Series,
    rule_name: str,
    rule_parameter: float | None,
) -> dict[str, object]:
    kept = profile_books.loc[keep_mask].copy()
    dropped = profile_books.loc[~keep_mask].copy()
    baseline_mean = float(profile_books["user_rating"].mean())
    kept_mean = float(kept["user_rating"].mean()) if not kept.empty else math.nan
    kept_goodreads_mean = (
        float(kept["average_rating"].mean()) if not kept.empty else math.nan
    )
    dropped_mean = (
        float(dropped["user_rating"].mean()) if not dropped.empty else math.nan
    )
    dropped_goodreads_mean = (
        float(dropped["average_rating"].mean()) if not dropped.empty else math.nan
    )
    dropped_five_star_share = (
        float((dropped["user_rating"] == 5).mean()) if not dropped.empty else math.nan
    )
    kept_low_star_share = (
        float((kept["user_rating"] <= 2).mean()) if not kept.empty else math.nan
    )
    kept_shares = _shares_by_rating(kept["user_rating"])
    dropped_shares = _shares_by_rating(dropped["user_rating"])
    return {
        "profile_slug": str(profile_books["profile_slug"].iloc[0]),
        "display_name": str(profile_books["display_name"].iloc[0]),
        "rule_name": rule_name,
        "rule_label": FIXED_RULE_LABELS[rule_name],
        "rule_parameter": rule_parameter,
        "n_books": int(len(profile_books)),
        "n_kept": int(keep_mask.sum()),
        "n_dropped": int((~keep_mask).sum()),
        "actual_drop_share": float((~keep_mask).mean()),
        "baseline_mean": baseline_mean,
        "kept_mean": kept_mean,
        "rating_gain": (
            kept_mean - baseline_mean if not math.isnan(kept_mean) else math.nan
        ),
        "kept_goodreads_mean": kept_goodreads_mean,
        "dropped_user_mean": dropped_mean,
        "dropped_goodreads_mean": dropped_goodreads_mean,
        "dropped_five_star_share": dropped_five_star_share,
        "kept_low_star_share": kept_low_star_share,
        "kept_rating_1_share": kept_shares[1],
        "kept_rating_2_share": kept_shares[2],
        "kept_rating_3_share": kept_shares[3],
        "kept_rating_4_share": kept_shares[4],
        "kept_rating_5_share": kept_shares[5],
        "dropped_rating_1_share": dropped_shares[1],
        "dropped_rating_2_share": dropped_shares[2],
        "dropped_rating_3_share": dropped_shares[3],
        "dropped_rating_4_share": dropped_shares[4],
        "dropped_rating_5_share": dropped_shares[5],
    }


def _rating_share_rows(
    profile_books: pd.DataFrame,
    keep_mask: pd.Series,
    rule_name: str,
) -> list[dict[str, object]]:
    kept = profile_books.loc[keep_mask].copy()
    shares = _shares_by_rating(kept["user_rating"])
    rows: list[dict[str, object]] = []
    for rating_value in range(1, 6):
        rows.append(
            {
                "profile_slug": str(profile_books["profile_slug"].iloc[0]),
                "display_name": str(profile_books["display_name"].iloc[0]),
                "rule_name": rule_name,
                "rule_label": FIXED_RULE_LABELS[rule_name],
                "rating_value": rating_value,
                "share_of_books": shares[rating_value],
                "share_percent": (
                    shares[rating_value] * 100
                    if pd.notna(shares[rating_value])
                    else math.nan
                ),
                "n_kept": int(len(kept)),
                "n_books": int(len(profile_books)),
            }
        )
    return rows


def compute_fixed_rule_outputs(
    prepared_profile_books: pd.DataFrame,
    analysis_manifest: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    profile_metric_rows: list[dict[str, object]] = []
    rating_share_rows: list[dict[str, object]] = []
    profile_context_rows: list[dict[str, object]] = []

    for _, profile_books in prepared_profile_books.groupby("profile_slug", sort=True):
        clean_books = _clean_profile_frame(profile_books)
        if clean_books.empty or not profile_has_sufficient_rating_variation(
            clean_books
        ):
            continue
        profile_context_rows.append(_profile_context_row(clean_books))

        keep_masks = {
            "original_distribution": pd.Series(True, index=clean_books.index),
            "goodreads_cutoff_4_0": clean_books["average_rating"] >= 4.0,
            "drop_lower_goodreads_half": _keep_mask_from_bottom_fraction_rule(
                clean_books,
                drop_fraction=0.5,
            ),
        }
        rule_parameters = {
            "original_distribution": math.nan,
            "goodreads_cutoff_4_0": 4.0,
            "drop_lower_goodreads_half": 0.5,
        }
        for rule_name in FIXED_RULE_NAME_ORDER:
            keep_mask = keep_masks[rule_name]
            profile_metric_rows.append(
                _rule_metric_row(
                    clean_books,
                    keep_mask=keep_mask,
                    rule_name=rule_name,
                    rule_parameter=rule_parameters[rule_name],
                )
            )
            rating_share_rows.extend(
                _rating_share_rows(
                    clean_books,
                    keep_mask=keep_mask,
                    rule_name=rule_name,
                )
            )

    profile_metrics = pd.DataFrame(profile_metric_rows)
    rating_shares = pd.DataFrame(rating_share_rows)
    profile_context = pd.DataFrame(profile_context_rows)
    if profile_metrics.empty:
        return {
            "profile_metrics": profile_metrics,
            "rating_shares": rating_shares,
            "summary": pd.DataFrame(),
            "profile_context": profile_context,
        }

    if analysis_manifest is not None and not analysis_manifest.empty:
        manifest_columns = [
            "profile_slug",
            "goodreads_url",
            "source_group",
            "fit_notes",
            "books_on_goodreads",
            "public_ratings",
            "public_reviews",
        ]
        manifest_subset = analysis_manifest[
            [
                column
                for column in manifest_columns
                if column in analysis_manifest.columns
            ]
        ].drop_duplicates(subset=["profile_slug"])
        profile_context = profile_context.merge(
            manifest_subset,
            on="profile_slug",
            how="left",
        )

    profile_metrics = profile_metrics.merge(
        profile_context.drop(columns=["n_books", "baseline_mean"], errors="ignore"),
        on=["profile_slug", "display_name"],
        how="left",
    )

    summary_rows: list[dict[str, object]] = []
    evaluable_rules = [
        rule_name
        for rule_name in FIXED_RULE_NAME_ORDER
        if rule_name != "original_distribution"
    ]
    metric_subset = profile_metrics[
        profile_metrics["rule_name"].isin(evaluable_rules)
    ].copy()
    for rule_name, rule_frame in metric_subset.groupby("rule_name", sort=False):
        summary_rows.append(
            {
                "rule_name": rule_name,
                "rule_label": FIXED_RULE_LABELS[rule_name],
                "n_profiles": int(len(rule_frame)),
                "mean_gain": float(rule_frame["rating_gain"].mean()),
                "median_gain": float(rule_frame["rating_gain"].median()),
                "mean_drop_share": float(rule_frame["actual_drop_share"].mean()),
                "median_drop_share": float(rule_frame["actual_drop_share"].median()),
                "positive_gain_share": float((rule_frame["rating_gain"] > 0).mean()),
                "negative_gain_share": float((rule_frame["rating_gain"] < 0).mean()),
                "mean_kept_mean": float(rule_frame["kept_mean"].mean()),
                "median_kept_mean": float(rule_frame["kept_mean"].median()),
            }
        )

    comparison = (
        metric_subset.pivot(
            index=["profile_slug", "display_name"],
            columns="rule_name",
            values="rating_gain",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if not comparison.empty:
        rule_a = "goodreads_cutoff_4_0"
        rule_b = "drop_lower_goodreads_half"
        summary_rows.extend(
            [
                {
                    "rule_name": "comparison",
                    "rule_label": "4.0 beats lower-half",
                    "n_profiles": int(len(comparison)),
                    "mean_gain": float(
                        (comparison[rule_a] - comparison[rule_b]).mean()
                    ),
                    "median_gain": float(
                        (comparison[rule_a] - comparison[rule_b]).median()
                    ),
                    "mean_drop_share": float("nan"),
                    "median_drop_share": float("nan"),
                    "positive_gain_share": float(
                        (comparison[rule_a] > comparison[rule_b]).mean()
                    ),
                    "negative_gain_share": float(
                        (comparison[rule_a] < comparison[rule_b]).mean()
                    ),
                    "mean_kept_mean": float("nan"),
                    "median_kept_mean": float("nan"),
                }
            ]
        )

    return {
        "profile_metrics": profile_metrics.sort_values(
            ["rule_name", "rating_gain", "display_name"],
            ascending=[True, False, True],
        ).reset_index(drop=True),
        "rating_shares": rating_shares.sort_values(
            ["rule_name", "rating_value", "display_name"],
            ascending=[True, True, True],
        ).reset_index(drop=True),
        "summary": pd.DataFrame(summary_rows),
        "profile_context": profile_context.sort_values(
            ["display_name", "profile_slug"],
            ascending=[True, True],
        ).reset_index(drop=True),
    }


def compute_fixed_rule_parameter_comparison(
    prepared_profile_books: pd.DataFrame,
    cutoff_thresholds: tuple[float, ...] = DEFAULT_CUTOFF_SWEEP_THRESHOLDS,
    drop_fractions: tuple[float, ...] = DEFAULT_FIXED_RULE_DROP_FRACTIONS,
) -> dict[str, pd.DataFrame]:
    comparison_rows: list[dict[str, object]] = []

    for _, profile_books in prepared_profile_books.groupby("profile_slug", sort=True):
        clean_books = _clean_profile_frame(profile_books)
        if clean_books.empty or not profile_has_sufficient_rating_variation(
            clean_books
        ):
            continue

        actual = pd.to_numeric(clean_books["user_rating"], errors="coerce")
        scores = pd.to_numeric(clean_books["average_rating"], errors="coerce")
        cutoff_curve = build_cutoff_curve(actual, scores)
        percentile_curve = drop_curve(actual, scores, drop_fractions=drop_fractions)
        if cutoff_curve is None or percentile_curve.empty:
            continue

        percentile_lookup = percentile_curve.set_index("drop_fraction")
        for cutoff_threshold in cutoff_thresholds:
            cutoff_metrics = curve_metrics_at_threshold(cutoff_curve, cutoff_threshold)
            cutoff_gain = float(cutoff_metrics["rating_gain"])
            if not np.isfinite(cutoff_gain):
                continue
            cutoff_drop_share = float(cutoff_metrics["actual_drop_share"])

            for target_drop_fraction in drop_fractions:
                if target_drop_fraction not in percentile_lookup.index:
                    continue
                percentile_row = percentile_lookup.loc[target_drop_fraction]
                if isinstance(percentile_row, pd.DataFrame):
                    percentile_row = percentile_row.iloc[0]
                percentile_gain = float(percentile_row["rating_gain"])
                percentile_drop_share = 1.0 - float(percentile_row["keep_share"])
                comparison_rows.append(
                    {
                        "profile_slug": str(clean_books["profile_slug"].iloc[0]),
                        "display_name": str(clean_books["display_name"].iloc[0]),
                        "cutoff_threshold": float(cutoff_threshold),
                        "target_drop_fraction": float(target_drop_fraction),
                        "cutoff_gain": cutoff_gain,
                        "target_drop_gain": percentile_gain,
                        "gain_difference_cutoff_minus_target_drop": (
                            cutoff_gain - percentile_gain
                        ),
                        "cutoff_actual_drop_share": cutoff_drop_share,
                        "target_drop_actual_share": percentile_drop_share,
                    }
                )

    comparison_profiles = pd.DataFrame(comparison_rows)
    if comparison_profiles.empty:
        return {
            "comparison_profiles": comparison_profiles,
            "comparison_summary": pd.DataFrame(),
        }

    grouped = comparison_profiles.groupby(
        ["cutoff_threshold", "target_drop_fraction"],
        as_index=False,
        sort=True,
    )
    comparison_summary = grouped.agg(
        n_profiles=("profile_slug", "nunique"),
        cutoff_mean_gain=("cutoff_gain", "mean"),
        target_drop_mean_gain=("target_drop_gain", "mean"),
        cutoff_median_actual_drop_share=("cutoff_actual_drop_share", "median"),
        target_drop_median_actual_share=("target_drop_actual_share", "median"),
        benefit_share_cutoff_beats_target_drop=(
            "gain_difference_cutoff_minus_target_drop",
            lambda values: float((values > 0).mean()),
        ),
        mean_gain_difference_cutoff_minus_target_drop=(
            "gain_difference_cutoff_minus_target_drop",
            "mean",
        ),
        gain_difference_p25=(
            "gain_difference_cutoff_minus_target_drop",
            lambda values: float(np.percentile(values, 25)),
        ),
        gain_difference_p75=(
            "gain_difference_cutoff_minus_target_drop",
            lambda values: float(np.percentile(values, 75)),
        ),
    )
    comparison_summary["benefit_share_percent"] = (
        comparison_summary["benefit_share_cutoff_beats_target_drop"] * 100.0
    )

    return {
        "comparison_profiles": comparison_profiles.sort_values(
            ["cutoff_threshold", "target_drop_fraction", "display_name"],
            ascending=[True, True, True],
        ).reset_index(drop=True),
        "comparison_summary": comparison_summary.sort_values(
            ["target_drop_fraction", "cutoff_threshold"],
            ascending=[True, True],
        ).reset_index(drop=True),
    }


def build_fixed_rule_profile_frame(profile_metrics: pd.DataFrame) -> pd.DataFrame:
    if profile_metrics.empty:
        return pd.DataFrame()
    profile_columns = [
        "profile_slug",
        "display_name",
        "n_books",
        "baseline_mean",
        "profile_rating_std_all_books",
        "goodreads_mean",
        "goodreads_spearman_rho_all_books",
        "goodreads_pearson_r_all_books",
        "share_below_4_0",
        "median_goodreads_rating",
        "non_ascii_title_share",
        "non_ascii_author_share",
        "top_categories",
        "top_shelves",
        "goodreads_url",
        "source_group",
        "fit_notes",
        "books_on_goodreads",
        "public_ratings",
        "public_reviews",
    ]
    available_columns = [
        column for column in profile_columns if column in profile_metrics.columns
    ]
    return (
        profile_metrics[profile_metrics["rule_name"] == "original_distribution"][
            available_columns
        ]
        .drop_duplicates(subset=["profile_slug"])
        .reset_index(drop=True)
    )


def filter_fixed_rule_profiles_by_percentile_window(
    profile_metrics: pd.DataFrame,
    rating_shares: pd.DataFrame,
    metric_column: str,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, object]:
    profile_frame = build_fixed_rule_profile_frame(profile_metrics)
    if profile_frame.empty or metric_column not in profile_frame.columns:
        return {
            "profile_metrics": pd.DataFrame(),
            "rating_shares": pd.DataFrame(),
            "profile_frame": pd.DataFrame(),
            "summary": {
                "metric_column": metric_column,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_bound": math.nan,
                "upper_bound": math.nan,
                "n_profiles": 0,
            },
        }
    metric_values = profile_frame[metric_column]
    finite_values = metric_values[np.isfinite(metric_values)]
    if finite_values.empty:
        return {
            "profile_metrics": pd.DataFrame(),
            "rating_shares": pd.DataFrame(),
            "profile_frame": pd.DataFrame(),
            "summary": {
                "metric_column": metric_column,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_bound": math.nan,
                "upper_bound": math.nan,
                "n_profiles": 0,
            },
        }
    lower_bound = float(finite_values.quantile(lower_quantile))
    upper_bound = float(finite_values.quantile(upper_quantile))
    selected_profiles = profile_frame[
        profile_frame[metric_column].between(lower_bound, upper_bound, inclusive="both")
    ]["profile_slug"]
    filtered_metrics = profile_metrics[
        profile_metrics["profile_slug"].isin(selected_profiles)
    ].copy()
    filtered_shares = rating_shares[
        rating_shares["profile_slug"].isin(selected_profiles)
    ].copy()
    filtered_profiles = profile_frame[
        profile_frame["profile_slug"].isin(selected_profiles)
    ].copy()
    return {
        "profile_metrics": filtered_metrics.reset_index(drop=True),
        "rating_shares": filtered_shares.reset_index(drop=True),
        "profile_frame": filtered_profiles.reset_index(drop=True),
        "summary": {
            "metric_column": metric_column,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_profiles": int(filtered_profiles["profile_slug"].nunique()),
        },
    }


def assign_fixed_rule_correlation_bands(
    profile_metrics: pd.DataFrame,
    rating_shares: pd.DataFrame,
    correlation_column: str = "goodreads_spearman_rho_all_books",
    quantile_ranges: tuple[tuple[float, float], ...] = (
        (0.10, 0.25),
        (0.25, 0.50),
        (0.50, 0.75),
        (0.75, 0.90),
    ),
) -> dict[str, pd.DataFrame]:
    profile_frame = build_fixed_rule_profile_frame(profile_metrics)
    if profile_frame.empty or correlation_column not in profile_frame.columns:
        return {
            "profile_metrics": pd.DataFrame(),
            "rating_shares": pd.DataFrame(),
            "band_summary": pd.DataFrame(),
        }
    finite_values = profile_frame[correlation_column]
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.empty:
        return {
            "profile_metrics": pd.DataFrame(),
            "rating_shares": pd.DataFrame(),
            "band_summary": pd.DataFrame(),
        }

    band_rows: list[dict[str, object]] = []
    band_lookup: dict[str, str] = {}
    for lower_quantile, upper_quantile in quantile_ranges:
        lower_bound = float(finite_values.quantile(lower_quantile))
        upper_bound = float(finite_values.quantile(upper_quantile))
        band_name = f"p{int(lower_quantile * 100)}_to_p{int(upper_quantile * 100)}"
        band_label = (
            f"{int(lower_quantile * 100)}th-{int(upper_quantile * 100)}th percentile"
        )
        selected_profiles = profile_frame[
            profile_frame[correlation_column].between(
                lower_bound,
                upper_bound,
                inclusive="both",
            )
        ]["profile_slug"]
        for profile_slug in selected_profiles:
            band_lookup[str(profile_slug)] = band_name
        band_rows.append(
            {
                "band_name": band_name,
                "band_label": band_label,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "n_profiles": int(selected_profiles.nunique()),
            }
        )

    filtered_metrics = profile_metrics[
        profile_metrics["profile_slug"].isin(band_lookup)
    ].copy()
    filtered_metrics["correlation_band"] = filtered_metrics["profile_slug"].map(
        band_lookup
    )
    filtered_shares = rating_shares[
        rating_shares["profile_slug"].isin(band_lookup)
    ].copy()
    filtered_shares["correlation_band"] = filtered_shares["profile_slug"].map(
        band_lookup
    )
    band_summary = pd.DataFrame(band_rows)
    return {
        "profile_metrics": filtered_metrics.reset_index(drop=True),
        "rating_shares": filtered_shares.reset_index(drop=True),
        "band_summary": band_summary.reset_index(drop=True),
    }


def _sample_books_for_rule(
    profile_books: pd.DataFrame,
    rule_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_books = _clean_profile_frame(profile_books)
    if clean_books.empty:
        return clean_books, clean_books
    if rule_name == "goodreads_cutoff_4_0":
        keep_mask = clean_books["average_rating"] >= 4.0
    elif rule_name == "drop_lower_goodreads_half":
        keep_mask = _keep_mask_from_bottom_fraction_rule(
            clean_books,
            drop_fraction=0.5,
        )
    else:
        keep_mask = pd.Series(True, index=clean_books.index)
    kept = clean_books.loc[keep_mask].copy()
    dropped = clean_books.loc[~keep_mask].copy()
    return kept, dropped


def _example_string(frame: pd.DataFrame, ascending: bool, max_examples: int = 3) -> str:
    if frame.empty:
        return ""
    ranked = frame.sort_values(
        ["user_rating", "average_rating", "title_key"],
        ascending=[ascending, ascending, True],
    ).head(max_examples)
    return " | ".join(
        f"{row.title} [you {row.user_rating:.0f}, GR {row.average_rating:.2f}]"
        for row in ranked.itertuples(index=False)
    )


def build_fixed_rule_audit_report(
    profile_metrics: pd.DataFrame,
    prepared_profile_books: pd.DataFrame,
    cutoff_sweep_profiles_all_books: pd.DataFrame | None = None,
    other_outlier_count: int = 6,
    per_rule_count: int = 6,
) -> str:
    if profile_metrics.empty:
        return "No fixed-rule metrics available.\n"

    lines = [
        "# Fixed Rule Audit",
        "",
        "This note reviews who benefits most and least from two simple all-books rules:",
        "- keep only books with Goodreads average rating at least 4.0",
        "- drop the lower Goodreads half of each reader's rated books",
        "",
    ]

    for rule_name in ("goodreads_cutoff_4_0", "drop_lower_goodreads_half"):
        rule_frame = profile_metrics[profile_metrics["rule_name"] == rule_name].copy()
        lines.extend(
            [
                f"## {FIXED_RULE_LABELS[rule_name]}",
                "",
                "### Biggest beneficiaries",
                "",
            ]
        )
        top_profiles = rule_frame.sort_values(
            ["rating_gain", "n_books", "display_name"],
            ascending=[False, False, True],
        ).head(per_rule_count)
        bottom_profiles = rule_frame.sort_values(
            ["rating_gain", "n_books", "display_name"],
            ascending=[True, False, True],
        ).head(per_rule_count)
        for section_name, subset in (
            ("benefit", top_profiles),
            ("hurt", bottom_profiles),
        ):
            if section_name == "hurt":
                lines.extend(["", "### Smallest beneficiaries / possible anti-fit", ""])
            for row in subset.itertuples(index=False):
                profile_books = prepared_profile_books[
                    prepared_profile_books["profile_slug"] == row.profile_slug
                ]
                kept, dropped = _sample_books_for_rule(profile_books, rule_name)
                explanation_parts: list[str] = []
                if row.actual_drop_share < 0.15 and row.rating_gain < 0.1:
                    explanation_parts.append(
                        "low exposure: this reader already chooses mostly books the rule would keep"
                    )
                if row.actual_drop_share > 0.55 and row.rating_gain > 0.3:
                    explanation_parts.append(
                        "high exposure plus strong Goodreads alignment"
                    )
                if pd.notna(row.goodreads_spearman_rho_all_books):
                    if row.goodreads_spearman_rho_all_books < 0.1:
                        explanation_parts.append("weak Goodreads-personal alignment")
                    elif row.goodreads_spearman_rho_all_books > 0.45:
                        explanation_parts.append("strong Goodreads-personal alignment")
                if (
                    pd.notna(row.dropped_five_star_share)
                    and row.dropped_five_star_share > 0.2
                ):
                    explanation_parts.append(
                        "the rule drops a meaningful share of books they actually loved"
                    )
                if pd.notna(row.kept_low_star_share) and row.kept_low_star_share > 0.2:
                    explanation_parts.append(
                        "many disappointments remain even after filtering"
                    )
                if (
                    pd.notna(row.non_ascii_title_share)
                    and row.non_ascii_title_share > 0.4
                ):
                    explanation_parts.append(
                        "profile is heavily non-English or translated, so Goodreads consensus may be thinner"
                    )
                explanation = (
                    "; ".join(explanation_parts) or "mixed case; inspect examples"
                )
                lines.extend(
                    [
                        f"- {row.display_name} (`{row.profile_slug}`)",
                        f"  gain `{row.rating_gain:.3f}`, drop `{row.actual_drop_share:.0%}`, Goodreads rho `{row.goodreads_spearman_rho_all_books:.3f}`",
                        f"  reads like: {row.top_categories or 'unclear categories'}",
                        f"  shelves: {row.top_shelves or 'no strong public shelves'}",
                        f"  explanation: {explanation}",
                        f"  dropped examples: {_example_string(dropped, ascending=False)}",
                        f"  kept weak examples: {_example_string(kept, ascending=True)}",
                    ]
                )

    other_sections: list[tuple[str, pd.DataFrame]] = []
    correlation_profiles = profile_metrics[
        profile_metrics["rule_name"] == "goodreads_cutoff_4_0"
    ].copy()
    other_sections.append(
        (
            "Lowest all-books Goodreads correlation",
            correlation_profiles.sort_values(
                ["goodreads_spearman_rho_all_books", "n_books", "display_name"],
                ascending=[True, False, True],
            ).head(other_outlier_count),
        )
    )
    other_sections.append(
        (
            "Highest all-books Goodreads correlation",
            correlation_profiles.sort_values(
                ["goodreads_spearman_rho_all_books", "n_books", "display_name"],
                ascending=[False, False, True],
            ).head(other_outlier_count),
        )
    )
    if (
        cutoff_sweep_profiles_all_books is not None
        and not cutoff_sweep_profiles_all_books.empty
    ):
        high_cutoff_tail = cutoff_sweep_profiles_all_books[
            cutoff_sweep_profiles_all_books["threshold"].eq(4.3)
            & cutoff_sweep_profiles_all_books["n_kept"].ge(25)
        ].sort_values(["kept_mean", "n_kept"], ascending=[True, False])
        other_sections.append(
            (
                "Lowest kept means even after a 4.3 cutoff",
                high_cutoff_tail.head(other_outlier_count).merge(
                    correlation_profiles[
                        [
                            "profile_slug",
                            "display_name",
                            "goodreads_spearman_rho_all_books",
                            "top_categories",
                            "top_shelves",
                            "non_ascii_title_share",
                        ]
                    ],
                    on=["profile_slug", "display_name"],
                    how="left",
                ),
            )
        )

    lines.extend(["", "## Other outliers", ""])
    for section_title, subset in other_sections:
        lines.extend([f"### {section_title}", ""])
        for row in subset.itertuples(index=False):
            descriptor = []
            if hasattr(row, "goodreads_spearman_rho_all_books") and pd.notna(
                row.goodreads_spearman_rho_all_books
            ):
                descriptor.append(f"rho `{row.goodreads_spearman_rho_all_books:.3f}`")
            if hasattr(row, "kept_mean") and pd.notna(row.kept_mean):
                descriptor.append(f"kept mean `{row.kept_mean:.3f}`")
            if hasattr(row, "n_kept") and pd.notna(row.n_kept):
                descriptor.append(f"`n_kept={int(row.n_kept)}`")
            detail = ", ".join(descriptor)
            lines.extend(
                [
                    f"- {row.display_name} (`{row.profile_slug}`) {detail}",
                    f"  categories: {getattr(row, 'top_categories', '') or 'unclear categories'}",
                    f"  shelves: {getattr(row, 'top_shelves', '') or 'no strong public shelves'}",
                ]
            )

    return "\n".join(lines) + "\n"
