"""Bootstrap the Goodreads-covered new-books holdout to assess rule stability."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ai_books_tracking.future_prediction_evaluation import THRESHOLDS
from ai_books_tracking.new_books_to_rate_analysis import (
    POLICY_BALANCED_KEEP_SHARE_RANGE,
    POLICY_MIN_SIDE_N,
    POLICY_TARGETS,
    PREDICTIONS_CSV,
    TARGET_LABELS,
    requested_utility,
)

OUTPUT_DIR = Path(__file__).parent
BOOTSTRAP_SUMMARY_CSV = OUTPUT_DIR / "new_books_to_rate_2026_bootstrap_rule_summary.csv"
BOOTSTRAP_BEST_CSV = OUTPUT_DIR / "new_books_to_rate_2026_bootstrap_rule_best.csv"
BOOTSTRAP_TARGET_SUMMARY_CSV = (
    OUTPUT_DIR / "new_books_to_rate_2026_bootstrap_target_summary.csv"
)
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class RuleCandidate:
    target: str
    feature_spec: str
    model: str
    threshold: float
    n_total: int
    n_keep: int
    keep_share: float
    observed_delta_keep_vs_all: float
    observed_delta_keep_utility_vs_all: float
    keep_mask: np.ndarray


def load_prediction_frame(path: Path = PREDICTIONS_CSV) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["target"].isin(POLICY_TARGETS)].copy()
    frame = frame[frame["goodreads_rating"].notna()].copy()
    frame["row_id"] = (
        frame["title"].astype(str).str.strip()
        + "||"
        + frame["date_finished"].astype(str).str.strip()
    )
    return frame


def build_target_base(
    predictions: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    target_frame = predictions[predictions["target"] == target_col].copy()
    base = (
        target_frame[
            ["row_id", "title", "date_finished", target_col, "goodreads_rating"]
        ]
        .drop_duplicates(subset=["row_id"])
        .sort_values(["date_finished", "title"])
        .reset_index(drop=True)
    )
    actual = pd.to_numeric(base[target_col], errors="coerce").to_numpy(dtype=float)
    utility = requested_utility(pd.Series(actual), target_col).to_numpy(dtype=float)
    return base, actual, utility


def evaluate_rule_arrays(
    prediction_values: np.ndarray,
    actual: np.ndarray,
    utility: np.ndarray,
    threshold: float,
    min_side_n: int = POLICY_MIN_SIDE_N,
) -> RuleCandidate | None:
    keep_mask = prediction_values >= threshold
    n_total = int(len(actual))
    n_keep = int(keep_mask.sum())
    n_skip = n_total - n_keep
    if n_keep < min_side_n or n_skip < min_side_n:
        return None

    overall_mean = float(np.mean(actual))
    keep_mean = float(np.mean(actual[keep_mask]))
    overall_utility_mean = float(np.mean(utility))
    keep_utility_mean = float(np.mean(utility[keep_mask]))
    return RuleCandidate(
        target="",
        feature_spec="",
        model="",
        threshold=threshold,
        n_total=n_total,
        n_keep=n_keep,
        keep_share=n_keep / n_total,
        observed_delta_keep_vs_all=keep_mean - overall_mean,
        observed_delta_keep_utility_vs_all=keep_utility_mean - overall_utility_mean,
        keep_mask=keep_mask,
    )


def build_candidates(
    predictions: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[RuleCandidate]]:
    base, actual, utility = build_target_base(predictions, target_col)
    target_frame = predictions[predictions["target"] == target_col].copy()
    candidates: list[RuleCandidate] = []
    for (feature_spec, model), group in target_frame.groupby(["feature_spec", "model"]):
        prediction_map = group.set_index("row_id")["prediction"]
        prediction_values = (
            base["row_id"].map(prediction_map).astype(float).to_numpy(dtype=float)
        )
        for threshold in THRESHOLDS:
            candidate = evaluate_rule_arrays(
                prediction_values=prediction_values,
                actual=actual,
                utility=utility,
                threshold=float(threshold),
            )
            if candidate is None:
                continue
            candidates.append(
                RuleCandidate(
                    target=target_col,
                    feature_spec=feature_spec,
                    model=model,
                    threshold=float(threshold),
                    n_total=candidate.n_total,
                    n_keep=candidate.n_keep,
                    keep_share=candidate.keep_share,
                    observed_delta_keep_vs_all=candidate.observed_delta_keep_vs_all,
                    observed_delta_keep_utility_vs_all=(
                        candidate.observed_delta_keep_utility_vs_all
                    ),
                    keep_mask=candidate.keep_mask,
                )
            )
    return base, actual, utility, candidates


def bootstrap_candidate_matrix(
    actual: np.ndarray,
    utility: np.ndarray,
    candidates: list[RuleCandidate],
    n_bootstraps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(actual), size=(n_bootstraps, len(actual)))
    boot_actual = actual[indices]
    boot_utility = utility[indices]
    rating_matrix = np.full((len(candidates), n_bootstraps), -np.inf, dtype=float)
    utility_matrix = np.full((len(candidates), n_bootstraps), -np.inf, dtype=float)

    overall_mean = boot_actual.mean(axis=1)
    overall_utility_mean = boot_utility.mean(axis=1)

    for rule_index, candidate in enumerate(candidates):
        boot_keep = candidate.keep_mask[indices]
        keep_count = boot_keep.sum(axis=1)
        valid = (keep_count > 0) & (keep_count < len(actual))
        if not np.any(valid):
            continue

        keep_sum = (boot_keep * boot_actual).sum(axis=1)
        keep_utility_sum = (boot_keep * boot_utility).sum(axis=1)
        keep_mean = np.divide(
            keep_sum,
            keep_count,
            out=np.full(n_bootstraps, np.nan, dtype=float),
            where=keep_count > 0,
        )
        keep_utility_mean = np.divide(
            keep_utility_sum,
            keep_count,
            out=np.full(n_bootstraps, np.nan, dtype=float),
            where=keep_count > 0,
        )
        rating_delta = keep_mean - overall_mean
        utility_delta = keep_utility_mean - overall_utility_mean
        rating_matrix[rule_index, valid] = rating_delta[valid]
        utility_matrix[rule_index, valid] = utility_delta[valid]

    return rating_matrix, utility_matrix


def rank_matrix(metric_matrix: np.ndarray) -> np.ndarray:
    order = np.argsort(-metric_matrix, axis=0, kind="stable")
    ranks = np.empty_like(order)
    columns = np.arange(metric_matrix.shape[1])
    ranks[order, columns] = np.arange(1, metric_matrix.shape[0] + 1)[:, None]
    return ranks.astype(float)


def summarize_subset(
    candidates: list[RuleCandidate],
    rating_matrix: np.ndarray,
    utility_matrix: np.ndarray,
    subset_name: str,
    include_mask: np.ndarray,
) -> pd.DataFrame:
    subset_indices = np.flatnonzero(include_mask)
    if len(subset_indices) == 0:
        return pd.DataFrame()

    subset_rating = rating_matrix[subset_indices]
    subset_utility = utility_matrix[subset_indices]
    rating_ranks = rank_matrix(subset_rating)
    utility_ranks = rank_matrix(subset_utility)

    rows: list[dict[str, object]] = []
    for local_index, candidate_index in enumerate(subset_indices):
        candidate = candidates[candidate_index]
        rating_values = subset_rating[local_index]
        utility_values = subset_utility[local_index]
        finite_rating = rating_values[np.isfinite(rating_values)]
        finite_utility = utility_values[np.isfinite(utility_values)]
        rows.append(
            {
                "target": candidate.target,
                "target_label": TARGET_LABELS[candidate.target],
                "subset": subset_name,
                "feature_spec": candidate.feature_spec,
                "model": candidate.model,
                "threshold": candidate.threshold,
                "n_total": candidate.n_total,
                "n_keep": candidate.n_keep,
                "keep_share": candidate.keep_share,
                "observed_delta_keep_vs_all": candidate.observed_delta_keep_vs_all,
                "observed_delta_keep_utility_vs_all": (
                    candidate.observed_delta_keep_utility_vs_all
                ),
                "bootstrap_mean_delta_keep_vs_all": float(np.mean(finite_rating)),
                "bootstrap_p05_delta_keep_vs_all": float(
                    np.quantile(finite_rating, 0.05)
                ),
                "bootstrap_p50_delta_keep_vs_all": float(
                    np.quantile(finite_rating, 0.50)
                ),
                "bootstrap_p95_delta_keep_vs_all": float(
                    np.quantile(finite_rating, 0.95)
                ),
                "bootstrap_prob_positive_delta_keep_vs_all": float(
                    np.mean(rating_values > 0)
                ),
                "bootstrap_mean_delta_keep_utility_vs_all": float(
                    np.mean(finite_utility)
                ),
                "bootstrap_p05_delta_keep_utility_vs_all": float(
                    np.quantile(finite_utility, 0.05)
                ),
                "bootstrap_p50_delta_keep_utility_vs_all": float(
                    np.quantile(finite_utility, 0.50)
                ),
                "bootstrap_p95_delta_keep_utility_vs_all": float(
                    np.quantile(finite_utility, 0.95)
                ),
                "bootstrap_prob_positive_delta_keep_utility_vs_all": float(
                    np.mean(utility_values > 0)
                ),
                "bootstrap_mean_rating_rank": float(np.mean(rating_ranks[local_index])),
                "bootstrap_median_rating_rank": float(
                    np.median(rating_ranks[local_index])
                ),
                "bootstrap_prob_rating_best": float(
                    np.mean(rating_ranks[local_index] == 1)
                ),
                "bootstrap_prob_rating_top3": float(
                    np.mean(rating_ranks[local_index] <= min(3, len(subset_indices)))
                ),
                "bootstrap_mean_utility_rank": float(
                    np.mean(utility_ranks[local_index])
                ),
                "bootstrap_median_utility_rank": float(
                    np.median(utility_ranks[local_index])
                ),
                "bootstrap_prob_utility_best": float(
                    np.mean(utility_ranks[local_index] == 1)
                ),
                "bootstrap_prob_utility_top3": float(
                    np.mean(utility_ranks[local_index] <= min(3, len(subset_indices)))
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap_target(
    predictions: pd.DataFrame,
    target_col: str,
    n_bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, actual, utility, candidates = build_candidates(predictions, target_col)
    rating_matrix, utility_matrix = bootstrap_candidate_matrix(
        actual=actual,
        utility=utility,
        candidates=candidates,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    all_mask = np.ones(len(candidates), dtype=bool)
    balanced_mask = np.array(
        [
            POLICY_BALANCED_KEEP_SHARE_RANGE[0]
            <= candidate.keep_share
            <= POLICY_BALANCED_KEEP_SHARE_RANGE[1]
            for candidate in candidates
        ],
        dtype=bool,
    )
    all_summary = summarize_subset(
        candidates=candidates,
        rating_matrix=rating_matrix,
        utility_matrix=utility_matrix,
        subset_name="all",
        include_mask=all_mask,
    )
    balanced_summary = summarize_subset(
        candidates=candidates,
        rating_matrix=rating_matrix,
        utility_matrix=utility_matrix,
        subset_name="balanced",
        include_mask=balanced_mask,
    )
    combined = pd.concat([all_summary, balanced_summary], ignore_index=True)
    target_summary = pd.DataFrame(
        [
            {
                "target": target_col,
                "target_label": TARGET_LABELS[target_col],
                "n_books": int(len(actual)),
                "n_rules_all": int(all_mask.sum()),
                "n_rules_balanced": int(balanced_mask.sum()),
                "n_bootstraps": int(n_bootstraps),
                "seed": int(seed),
            }
        ]
    )
    return combined, target_summary


def summarize_best_rules(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_col in POLICY_TARGETS:
        target_summary = summary[summary["target"] == target_col].copy()
        for subset_name in target_summary["subset"].unique():
            subset = target_summary[target_summary["subset"] == subset_name].copy()
            if subset.empty:
                continue
            rows.append(
                {
                    **subset.sort_values(
                        "observed_delta_keep_utility_vs_all", ascending=False
                    )
                    .iloc[0]
                    .to_dict(),
                    "target": target_col,
                    "subset": subset_name,
                    "selection": "best_observed_utility",
                }
            )
            rows.append(
                {
                    **subset.sort_values(
                        "bootstrap_mean_delta_keep_utility_vs_all", ascending=False
                    )
                    .iloc[0]
                    .to_dict(),
                    "target": target_col,
                    "subset": subset_name,
                    "selection": "best_bootstrap_mean_utility",
                }
            )
            rows.append(
                {
                    **subset.sort_values("bootstrap_prob_utility_best", ascending=False)
                    .iloc[0]
                    .to_dict(),
                    "target": target_col,
                    "subset": subset_name,
                    "selection": "highest_prob_utility_best",
                }
            )
            rows.append(
                {
                    **subset.sort_values(
                        "bootstrap_mean_delta_keep_vs_all", ascending=False
                    )
                    .iloc[0]
                    .to_dict(),
                    "target": target_col,
                    "subset": subset_name,
                    "selection": "best_bootstrap_mean_rating",
                }
            )
    return pd.DataFrame(rows)


def print_summary(target_summary: pd.DataFrame, best_rules: pd.DataFrame) -> None:
    print("=" * 80)
    print("NEW BOOKS BOOTSTRAP RULE STABILITY")
    print("=" * 80)
    print("\nTarget summary:")
    print(target_summary.to_string(index=False))
    print("\nBest rules:")
    print(
        best_rules[
            [
                "target",
                "subset",
                "selection",
                "feature_spec",
                "model",
                "threshold",
                "observed_delta_keep_vs_all",
                "observed_delta_keep_utility_vs_all",
                "bootstrap_mean_delta_keep_vs_all",
                "bootstrap_mean_delta_keep_utility_vs_all",
                "bootstrap_prob_utility_best",
                "bootstrap_prob_utility_top3",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_prediction_frame(args.predictions)

    summary_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    for target_col in POLICY_TARGETS:
        summary, target_summary = summarize_bootstrap_target(
            predictions=predictions,
            target_col=target_col,
            n_bootstraps=args.bootstraps,
            seed=args.seed,
        )
        summary_parts.append(summary)
        target_parts.append(target_summary)

    summary_df = pd.concat(summary_parts, ignore_index=True).sort_values(
        [
            "target",
            "subset",
            "bootstrap_mean_delta_keep_utility_vs_all",
            "bootstrap_prob_utility_best",
        ],
        ascending=[True, True, False, False],
    )
    summary_df.to_csv(BOOTSTRAP_SUMMARY_CSV, index=False)

    target_summary_df = pd.concat(target_parts, ignore_index=True)
    target_summary_df.to_csv(BOOTSTRAP_TARGET_SUMMARY_CSV, index=False)

    best_rules = summarize_best_rules(summary_df)
    best_rules.to_csv(BOOTSTRAP_BEST_CSV, index=False)
    print_summary(target_summary_df, best_rules)


if __name__ == "__main__":
    main()
