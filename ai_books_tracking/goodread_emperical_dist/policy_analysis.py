from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ai_books_tracking.goodread_emperical_dist.evaluation import (
    HoldoutSplit,
    build_score_map,
    choose_holdout_split,
    drop_curve,
    profile_has_sufficient_rating_variation,
    prediction_metrics,
    select_best_fitted_model,
)

POLICY_DROP_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
DEFAULT_BOOTSTRAP_SAMPLES = 300
DEFAULT_SMALL_MULTIPLE_COUNT = 19
MAX_SMALL_MULTIPLE_RECENCY_GAP = 7.0
MAX_SMALL_MULTIPLE_DOMINANT_YEAR_SHARE = 0.5
DEFAULT_CUTOFF_SWEEP_THRESHOLDS = tuple(np.round(np.arange(3.5, 4.51, 0.1), 2))
DEFAULT_GAIN_PERCENTILES = (15, 35, 65, 85)
POLICY_NAME_ORDER = (
    "global_goodreads_cutoff",
    "personal_goodreads_cutoff",
    "shrunk_goodreads_cutoff",
    "training_selected_model",
)
POLICY_LABELS = {
    "global_goodreads_cutoff": "1. Global Goodreads cutoff",
    "personal_goodreads_cutoff": "2. Person-optimal Goodreads cutoff",
    "shrunk_goodreads_cutoff": "2b. Shrunk personal cutoff",
    "training_selected_model": "3. Training-selected fitted model",
}
CORRELATION_SERIES_LABELS = {
    "goodreads_raw": "Goodreads raw",
    "training_selected_model": "Training-selected fitted model",
}


@dataclass(frozen=True)
class ProfileSplitBundle:
    profile_slug: str
    display_name: str
    full_df: pd.DataFrame
    train_df: pd.DataFrame
    holdout_df: pd.DataFrame
    split_info: HoldoutSplit
    n_rated_books: int
    latest_event_year: int | None
    holdout_year: int | None
    recency_gap: float
    dominant_year_share: float


@dataclass(frozen=True)
class CutoffCurve:
    thresholds: np.ndarray
    n_dropped: np.ndarray
    n_kept: np.ndarray
    drop_shares: np.ndarray
    kept_means: np.ndarray
    rating_gains: np.ndarray
    baseline_mean: float
    n_books: int


def _clean_numeric_arrays(
    actual_ratings: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual_ratings, dtype=float)
    pred = np.asarray(scores, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    return actual[valid], pred[valid]


def build_cutoff_curve(
    actual_ratings: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
) -> CutoffCurve | None:
    actual, pred = _clean_numeric_arrays(actual_ratings, scores)
    if len(actual) == 0:
        return None

    order = np.lexsort((actual, pred))
    pred = pred[order]
    actual = actual[order]

    thresholds, first_indices = np.unique(pred, return_index=True)
    n_books = len(actual)
    baseline_mean = float(actual.mean())
    cumulative_actual = np.cumsum(actual)
    dropped_sums = np.zeros(len(first_indices), dtype=float)
    has_dropped = first_indices > 0
    dropped_sums[has_dropped] = cumulative_actual[first_indices[has_dropped] - 1]
    n_dropped = first_indices.astype(int)
    n_kept = n_books - n_dropped
    kept_sums = float(actual.sum()) - dropped_sums
    kept_means = kept_sums / n_kept
    return CutoffCurve(
        thresholds=thresholds.astype(float),
        n_dropped=n_dropped.astype(int),
        n_kept=n_kept.astype(int),
        drop_shares=n_dropped / n_books,
        kept_means=kept_means.astype(float),
        rating_gains=(kept_means - baseline_mean).astype(float),
        baseline_mean=baseline_mean,
        n_books=n_books,
    )


def curve_metrics_at_threshold(
    curve: CutoffCurve | None,
    threshold: float,
) -> dict[str, float]:
    if curve is None or not np.isfinite(threshold):
        return {
            "threshold": float(threshold),
            "baseline_mean": math.nan,
            "kept_mean": math.nan,
            "rating_gain": math.nan,
            "actual_drop_share": math.nan,
            "n_books": 0,
            "n_kept": 0,
        }

    index = int(np.searchsorted(curve.thresholds, threshold, side="left"))
    if index >= len(curve.thresholds):
        return {
            "threshold": float(threshold),
            "baseline_mean": curve.baseline_mean,
            "kept_mean": math.nan,
            "rating_gain": math.nan,
            "actual_drop_share": 1.0,
            "n_books": curve.n_books,
            "n_kept": 0,
        }

    return {
        "threshold": float(threshold),
        "baseline_mean": curve.baseline_mean,
        "kept_mean": float(curve.kept_means[index]),
        "rating_gain": float(curve.rating_gains[index]),
        "actual_drop_share": float(curve.drop_shares[index]),
        "n_books": int(curve.n_books),
        "n_kept": int(curve.n_kept[index]),
    }


def choose_threshold_for_target(
    curve: CutoffCurve | None,
    target_drop_fraction: float,
) -> dict[str, float]:
    if curve is None or len(curve.thresholds) == 0:
        return {
            "threshold": math.nan,
            "train_drop_share": math.nan,
            "train_gain": math.nan,
            "train_drop_error": math.nan,
        }

    drop_errors = np.abs(curve.drop_shares - target_drop_fraction)
    order = np.lexsort(
        (
            curve.thresholds,
            -curve.rating_gains,
            drop_errors,
        )
    )
    best_index = int(order[0])
    return {
        "threshold": float(curve.thresholds[best_index]),
        "train_drop_share": float(curve.drop_shares[best_index]),
        "train_gain": float(curve.rating_gains[best_index]),
        "train_drop_error": float(drop_errors[best_index]),
    }


def bootstrap_thresholds(
    actual_ratings: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    target_drop_fraction: float,
    n_bootstrap_samples: int,
    seed: int,
) -> np.ndarray:
    actual, pred = _clean_numeric_arrays(actual_ratings, scores)
    if len(actual) == 0 or n_bootstrap_samples <= 0:
        return np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    thresholds: list[float] = []
    for _ in range(n_bootstrap_samples):
        sample_index = rng.integers(0, len(actual), size=len(actual))
        curve = build_cutoff_curve(actual[sample_index], pred[sample_index])
        estimate = choose_threshold_for_target(curve, target_drop_fraction)
        if np.isfinite(estimate["threshold"]):
            thresholds.append(float(estimate["threshold"]))
    return np.asarray(thresholds, dtype=float)


def build_profile_splits(
    prepared_profile_books: pd.DataFrame,
) -> list[ProfileSplitBundle]:
    splits: list[ProfileSplitBundle] = []
    for _, profile_books in prepared_profile_books.groupby("profile_slug", sort=True):
        if not profile_has_sufficient_rating_variation(profile_books):
            continue
        evaluation_split = choose_holdout_split(profile_books)
        if evaluation_split is None:
            continue
        train_df, holdout_df, split_info = evaluation_split
        years = profile_books["event_year"].dropna()
        holdout_year = pd.to_numeric(split_info.holdout_label, errors="coerce")
        latest_event_year = int(years.max()) if not years.empty else None
        dominant_year_share = (
            float(years.astype(int).value_counts(normalize=True).iloc[0])
            if not years.empty
            else math.nan
        )
        recency_gap = math.nan
        if latest_event_year is not None and pd.notna(holdout_year):
            recency_gap = float(latest_event_year - int(holdout_year))
        splits.append(
            ProfileSplitBundle(
                profile_slug=str(profile_books["profile_slug"].iloc[0]),
                display_name=str(profile_books["display_name"].iloc[0]),
                full_df=profile_books.reset_index(drop=True),
                train_df=train_df,
                holdout_df=holdout_df,
                split_info=split_info,
                n_rated_books=int(len(profile_books)),
                latest_event_year=latest_event_year,
                holdout_year=int(holdout_year) if pd.notna(holdout_year) else None,
                recency_gap=recency_gap,
                dominant_year_share=dominant_year_share,
            )
        )
    return splits


def _select_global_thresholds(
    split_curves: dict[str, CutoffCurve],
) -> tuple[dict[float, float], pd.DataFrame]:
    candidate_thresholds = np.unique(
        np.concatenate([curve.thresholds for curve in split_curves.values()])
    )
    global_thresholds: dict[float, float] = {}
    summary_rows: list[dict[str, float]] = []

    for target_drop_fraction in POLICY_DROP_FRACTIONS:
        candidate_rows: list[dict[str, float]] = []
        for threshold in candidate_thresholds:
            per_profile_rows = [
                curve_metrics_at_threshold(curve, float(threshold))
                for curve in split_curves.values()
            ]
            candidate_df = pd.DataFrame(per_profile_rows)
            if candidate_df["n_kept"].eq(0).any():
                continue
            candidate_rows.append(
                {
                    "target_drop_fraction": target_drop_fraction,
                    "threshold": float(threshold),
                    "mean_train_drop_share": float(
                        candidate_df["actual_drop_share"].mean()
                    ),
                    "mean_train_gain": float(candidate_df["rating_gain"].mean()),
                    "mean_abs_train_drop_error": float(
                        (candidate_df["actual_drop_share"] - target_drop_fraction)
                        .abs()
                        .mean()
                    ),
                }
            )

        if not candidate_rows:
            continue

        candidates = pd.DataFrame(candidate_rows)
        candidates["mean_train_drop_gap"] = (
            candidates["mean_train_drop_share"] - target_drop_fraction
        ).abs()
        candidates = candidates.sort_values(
            [
                "mean_abs_train_drop_error",
                "mean_train_drop_gap",
                "mean_train_gain",
                "threshold",
            ],
            ascending=[True, True, False, True],
        )
        best = candidates.iloc[0]
        global_thresholds[target_drop_fraction] = float(best["threshold"])
        summary_rows.append(
            {
                "target_drop_fraction": target_drop_fraction,
                "global_threshold": float(best["threshold"]),
                "global_mean_train_drop_share": float(best["mean_train_drop_share"]),
                "global_mean_train_gain": float(best["mean_train_gain"]),
                "global_mean_abs_train_drop_error": float(
                    best["mean_abs_train_drop_error"]
                ),
            }
        )

    return global_thresholds, pd.DataFrame(summary_rows)


def _threshold_policy_rows(
    split: ProfileSplitBundle,
    target_drop_fraction: float,
    policy_name: str,
    threshold: float,
) -> dict[str, object]:
    holdout_curve = build_cutoff_curve(
        split.holdout_df["user_rating"], split.holdout_df["average_rating"]
    )
    metrics = curve_metrics_at_threshold(holdout_curve, threshold)
    return {
        "profile_slug": split.profile_slug,
        "display_name": split.display_name,
        "split_type": split.split_info.split_type,
        "holdout_label": split.split_info.holdout_label,
        "target_drop_fraction": target_drop_fraction,
        "policy_name": policy_name,
        "policy_label": POLICY_LABELS[policy_name],
        "threshold": float(threshold),
        "actual_holdout_drop_share": metrics["actual_drop_share"],
        "holdout_baseline_mean": metrics["baseline_mean"],
        "holdout_kept_mean": metrics["kept_mean"],
        "rating_gain": metrics["rating_gain"],
        "n_holdout_books": metrics["n_books"],
        "n_holdout_kept": metrics["n_kept"],
        "selected_model": "",
    }


def _model_policy_rows(
    split: ProfileSplitBundle,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    score_map = build_score_map(split.train_df, split.holdout_df)
    selected_model, cv_results = select_best_fitted_model(split.train_df)
    selection_rows: list[dict[str, object]] = []
    for row in cv_results.itertuples(index=False):
        selection_rows.append(
            {
                "profile_slug": split.profile_slug,
                "display_name": split.display_name,
                "candidate_model": row.model_name,
                "selected_model": selected_model,
                "is_selected": row.model_name == selected_model,
                "cv_folds": row.cv_folds,
                "cv_mean_mae": row.cv_mean_mae,
                "cv_mean_pearson_r": row.cv_mean_pearson_r,
                "cv_mean_spearman_rho": row.cv_mean_spearman_rho,
            }
        )

    actual_holdout = pd.to_numeric(split.holdout_df["user_rating"], errors="coerce")
    fitted_scores = pd.Series(
        score_map[selected_model], index=split.holdout_df.index, dtype=float
    )
    raw_scores = pd.Series(
        score_map["goodreads_raw"], index=split.holdout_df.index, dtype=float
    )
    selected_metrics = prediction_metrics(actual_holdout, fitted_scores)
    raw_metrics = prediction_metrics(actual_holdout, raw_scores)
    correlation_rows = [
        {
            "profile_slug": split.profile_slug,
            "display_name": split.display_name,
            "series_name": "goodreads_raw",
            "series_label": CORRELATION_SERIES_LABELS["goodreads_raw"],
            "selected_model": selected_model,
            **raw_metrics,
        },
        {
            "profile_slug": split.profile_slug,
            "display_name": split.display_name,
            "series_name": "training_selected_model",
            "series_label": CORRELATION_SERIES_LABELS["training_selected_model"],
            "selected_model": selected_model,
            **selected_metrics,
        },
    ]

    model_curve = drop_curve(
        actual_holdout,
        fitted_scores,
        drop_fractions=POLICY_DROP_FRACTIONS,
    )
    model_rows: list[dict[str, object]] = []
    for row in model_curve.itertuples(index=False):
        model_rows.append(
            {
                "profile_slug": split.profile_slug,
                "display_name": split.display_name,
                "split_type": split.split_info.split_type,
                "holdout_label": split.split_info.holdout_label,
                "target_drop_fraction": row.drop_fraction,
                "policy_name": "training_selected_model",
                "policy_label": POLICY_LABELS["training_selected_model"],
                "threshold": math.nan,
                "actual_holdout_drop_share": row.drop_fraction,
                "holdout_baseline_mean": row.baseline_mean,
                "holdout_kept_mean": row.kept_mean,
                "rating_gain": row.rating_gain,
                "n_holdout_books": row.n_books,
                "n_holdout_kept": row.n_kept,
                "selected_model": selected_model,
            }
        )
    return model_rows, selection_rows, correlation_rows


def select_small_multiples_profiles(
    splits: list[ProfileSplitBundle],
    max_profiles: int = DEFAULT_SMALL_MULTIPLE_COUNT,
    source_group_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    source_group_priority = {
        "user_supplied": 0,
        "expanded_verified": 1,
        "network_discovered": 2,
    }
    rows = [
        {
            "profile_slug": split.profile_slug,
            "display_name": split.display_name,
            "n_rated_books": split.n_rated_books,
            "split_type": split.split_info.split_type,
            "holdout_label": split.split_info.holdout_label,
            "latest_event_year": split.latest_event_year,
            "holdout_year": split.holdout_year,
            "recency_gap": split.recency_gap,
            "dominant_year_share": split.dominant_year_share,
            "source_group": (
                source_group_map.get(split.profile_slug, "network_discovered")
                if source_group_map is not None
                else "network_discovered"
            ),
        }
        for split in splits
    ]
    selection = pd.DataFrame(rows)
    if selection.empty:
        return selection
    selection = selection.sort_values(
        ["recency_gap", "dominant_year_share", "n_rated_books", "display_name"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    selection["passes_temporal_filter"] = (
        selection["recency_gap"].fillna(MAX_SMALL_MULTIPLE_RECENCY_GAP + 1)
        <= MAX_SMALL_MULTIPLE_RECENCY_GAP
    ) & (
        selection["dominant_year_share"].fillna(1.0)
        <= MAX_SMALL_MULTIPLE_DOMINANT_YEAR_SHARE
    )
    selection["source_priority"] = (
        selection["source_group"]
        .map(source_group_priority)
        .fillna(max(source_group_priority.values()) + 1)
    )
    preferred = selection[selection["passes_temporal_filter"]]
    remainder = selection[~selection["passes_temporal_filter"]]
    selection = pd.concat(
        [preferred, remainder],
        ignore_index=True,
    )
    selection = selection.sort_values(
        [
            "source_priority",
            "recency_gap",
            "dominant_year_share",
            "n_rated_books",
            "display_name",
        ],
        ascending=[True, True, True, False, True],
        na_position="last",
    ).head(max_profiles)
    selection["selection_rank"] = np.arange(1, len(selection) + 1)
    return selection


def run_policy_analysis(
    prepared_profile_books: pd.DataFrame,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    small_multiple_count: int = DEFAULT_SMALL_MULTIPLE_COUNT,
    analysis_manifest: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    splits = build_profile_splits(prepared_profile_books)
    if not splits:
        empty = pd.DataFrame()
        return {
            "policy_holdout_results": empty,
            "policy_threshold_summary": empty,
            "training_model_selection": empty,
            "holdout_correlations": empty,
            "small_multiples_selection": empty,
        }

    train_curves = {
        split.profile_slug: build_cutoff_curve(
            split.train_df["user_rating"],
            split.train_df["average_rating"],
        )
        for split in splits
    }
    _, global_summary = _select_global_thresholds(train_curves)

    threshold_rows: list[dict[str, object]] = []
    for split_index, split in enumerate(splits):
        actual_train, score_train = _clean_numeric_arrays(
            split.train_df["user_rating"],
            split.train_df["average_rating"],
        )
        train_curve = train_curves[split.profile_slug]
        for target_drop_fraction in POLICY_DROP_FRACTIONS:
            personal = choose_threshold_for_target(train_curve, target_drop_fraction)
            bootstrap_values = bootstrap_thresholds(
                actual_train,
                score_train,
                target_drop_fraction,
                n_bootstrap_samples=bootstrap_samples,
                seed=(split_index + 1) * 10_000
                + int(round(target_drop_fraction * 100)),
            )
            threshold_rows.append(
                {
                    "profile_slug": split.profile_slug,
                    "display_name": split.display_name,
                    "target_drop_fraction": target_drop_fraction,
                    "personal_threshold": personal["threshold"],
                    "personal_train_drop_share": personal["train_drop_share"],
                    "personal_train_gain": personal["train_gain"],
                    "personal_train_drop_error": personal["train_drop_error"],
                    "bootstrap_threshold_mean": (
                        float(np.mean(bootstrap_values))
                        if len(bootstrap_values)
                        else math.nan
                    ),
                    "bootstrap_threshold_std": (
                        float(np.std(bootstrap_values, ddof=1))
                        if len(bootstrap_values) > 1
                        else 0.0
                    ),
                    "bootstrap_threshold_var": (
                        float(np.var(bootstrap_values, ddof=1))
                        if len(bootstrap_values) > 1
                        else 0.0
                    ),
                    "bootstrap_samples_used": int(len(bootstrap_values)),
                }
            )

    threshold_summary = pd.DataFrame(threshold_rows)
    if threshold_summary.empty:
        threshold_summary = pd.DataFrame()
    else:
        threshold_summary = threshold_summary.merge(
            global_summary,
            on="target_drop_fraction",
            how="left",
        )
        threshold_summary = threshold_summary.rename(
            columns={"global_threshold": "global_goodreads_threshold"}
        )
        threshold_summary["between_person_threshold_var"] = 0.0
        threshold_summary["shrinkage_weight"] = 0.0
        threshold_summary["shrunk_threshold"] = threshold_summary[
            "global_goodreads_threshold"
        ]
        for target_drop_fraction in POLICY_DROP_FRACTIONS:
            mask = threshold_summary["target_drop_fraction"].eq(target_drop_fraction)
            between_var = float(
                threshold_summary.loc[mask, "personal_threshold"].var(ddof=1)
            )
            if not np.isfinite(between_var):
                between_var = 0.0
            boot_var = threshold_summary.loc[mask, "bootstrap_threshold_var"].fillna(
                0.0
            )
            denom = between_var + boot_var
            weight = np.where(denom > 0, between_var / denom, 0.0)
            threshold_summary.loc[mask, "between_person_threshold_var"] = between_var
            threshold_summary.loc[mask, "shrinkage_weight"] = weight
            threshold_summary.loc[mask, "shrunk_threshold"] = threshold_summary.loc[
                mask, "global_goodreads_threshold"
            ] + weight * (
                threshold_summary.loc[mask, "bootstrap_threshold_mean"].fillna(
                    threshold_summary.loc[mask, "personal_threshold"]
                )
                - threshold_summary.loc[mask, "global_goodreads_threshold"]
            )

    threshold_lookup = {
        (row.profile_slug, row.target_drop_fraction): row
        for row in threshold_summary.itertuples(index=False)
    }

    policy_rows: list[dict[str, object]] = []
    model_selection_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for split in splits:
        model_rows, selection_rows, split_correlation_rows = _model_policy_rows(split)
        policy_rows.extend(model_rows)
        model_selection_rows.extend(selection_rows)
        correlation_rows.extend(split_correlation_rows)
        for target_drop_fraction in POLICY_DROP_FRACTIONS:
            lookup_key = (split.profile_slug, target_drop_fraction)
            threshold_row = threshold_lookup[lookup_key]
            policy_rows.append(
                _threshold_policy_rows(
                    split,
                    target_drop_fraction,
                    "global_goodreads_cutoff",
                    float(threshold_row.global_goodreads_threshold),
                )
            )
            policy_rows.append(
                _threshold_policy_rows(
                    split,
                    target_drop_fraction,
                    "personal_goodreads_cutoff",
                    float(threshold_row.personal_threshold),
                )
            )
            policy_rows.append(
                _threshold_policy_rows(
                    split,
                    target_drop_fraction,
                    "shrunk_goodreads_cutoff",
                    float(threshold_row.shrunk_threshold),
                )
            )

    policy_holdout_results = pd.DataFrame(policy_rows)
    if not policy_holdout_results.empty:
        policy_holdout_results = policy_holdout_results.sort_values(
            ["target_drop_fraction", "policy_name", "display_name"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    model_selection = pd.DataFrame(model_selection_rows)
    if not model_selection.empty:
        model_selection = model_selection.sort_values(
            ["display_name", "is_selected", "candidate_model"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    holdout_correlations = pd.DataFrame(correlation_rows)
    if not holdout_correlations.empty:
        holdout_correlations = holdout_correlations.sort_values(
            ["series_name", "display_name"],
            ascending=[True, True],
        ).reset_index(drop=True)

    source_group_map = None
    if analysis_manifest is not None and not analysis_manifest.empty:
        source_group_map = (
            analysis_manifest[["profile_slug", "source_group"]]
            .drop_duplicates(subset=["profile_slug"])
            .set_index("profile_slug")["source_group"]
            .to_dict()
        )
    small_multiples_selection = select_small_multiples_profiles(
        splits,
        max_profiles=small_multiple_count,
        source_group_map=source_group_map,
    )

    return {
        "policy_holdout_results": policy_holdout_results,
        "policy_threshold_summary": threshold_summary,
        "training_model_selection": model_selection,
        "holdout_correlations": holdout_correlations,
        "small_multiples_selection": small_multiples_selection,
    }


def summarize_cutoff_sweep(
    profile_metrics: pd.DataFrame,
    gain_percentiles: tuple[int, ...] = DEFAULT_GAIN_PERCENTILES,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    if profile_metrics.empty:
        return pd.DataFrame(rows)

    for threshold, threshold_frame in profile_metrics.groupby("threshold", sort=True):
        gain_values = threshold_frame["rating_gain"].dropna().to_numpy(dtype=float)
        row: dict[str, float] = {
            "threshold": float(threshold),
            "n_profiles_total": int(threshold_frame["profile_slug"].nunique()),
            "n_profiles_with_kept_books": int(np.isfinite(gain_values).sum()),
            "drop_share_median": float(
                threshold_frame["actual_drop_share"].dropna().median()
            ),
            "drop_share_mean": float(
                threshold_frame["actual_drop_share"].dropna().mean()
            ),
        }
        drop_values = (
            threshold_frame["actual_drop_share"].dropna().to_numpy(dtype=float)
        )
        for percentile in gain_percentiles:
            row[f"gain_p{percentile}"] = (
                float(np.percentile(gain_values, percentile))
                if len(gain_values)
                else math.nan
            )
            row[f"drop_p{percentile}"] = (
                float(np.percentile(drop_values, percentile))
                if len(drop_values)
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _cutoff_sweep_rows_for_frame(
    ratings_frame: pd.DataFrame,
    profile_slug: str,
    display_name: str,
    split_type: str,
    holdout_label: str,
    evaluation_scope: str,
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    curve = build_cutoff_curve(
        ratings_frame["user_rating"], ratings_frame["average_rating"]
    )
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        metrics = curve_metrics_at_threshold(curve, float(threshold))
        rows.append(
            {
                "evaluation_scope": evaluation_scope,
                "profile_slug": profile_slug,
                "display_name": display_name,
                "split_type": split_type,
                "holdout_label": holdout_label,
                "threshold": float(threshold),
                "actual_drop_share": metrics["actual_drop_share"],
                "baseline_mean": metrics["baseline_mean"],
                "kept_mean": metrics["kept_mean"],
                "rating_gain": metrics["rating_gain"],
                "n_books": metrics["n_books"],
                "n_kept": metrics["n_kept"],
                "holdout_baseline_mean": (
                    metrics["baseline_mean"]
                    if evaluation_scope == "holdout"
                    else math.nan
                ),
                "holdout_kept_mean": (
                    metrics["kept_mean"] if evaluation_scope == "holdout" else math.nan
                ),
            }
        )
    return rows


def build_goodreads_cutoff_sweep(
    prepared_profile_books: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_CUTOFF_SWEEP_THRESHOLDS,
    evaluation_scope: str = "holdout",
) -> dict[str, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    if evaluation_scope == "holdout":
        splits = build_profile_splits(prepared_profile_books)
        for split in splits:
            rows.extend(
                _cutoff_sweep_rows_for_frame(
                    ratings_frame=split.holdout_df,
                    profile_slug=split.profile_slug,
                    display_name=split.display_name,
                    split_type=split.split_info.split_type,
                    holdout_label=split.split_info.holdout_label,
                    evaluation_scope=evaluation_scope,
                    thresholds=thresholds,
                )
            )
    elif evaluation_scope == "all_books":
        for _, profile_books in prepared_profile_books.groupby(
            "profile_slug", sort=True
        ):
            if not profile_has_sufficient_rating_variation(profile_books):
                continue
            rows.extend(
                _cutoff_sweep_rows_for_frame(
                    ratings_frame=profile_books,
                    profile_slug=str(profile_books["profile_slug"].iloc[0]),
                    display_name=str(profile_books["display_name"].iloc[0]),
                    split_type="all_books",
                    holdout_label="all_books",
                    evaluation_scope=evaluation_scope,
                    thresholds=thresholds,
                )
            )
    else:
        raise ValueError(f"Unsupported evaluation_scope: {evaluation_scope}")

    profile_metrics = pd.DataFrame(rows)
    percentile_summary = summarize_cutoff_sweep(profile_metrics)
    return {
        "cutoff_sweep_profiles": profile_metrics,
        "cutoff_sweep_percentiles": percentile_summary,
    }
