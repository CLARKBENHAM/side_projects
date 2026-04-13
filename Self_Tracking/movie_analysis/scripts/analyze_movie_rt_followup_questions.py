from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "data" / "mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis_core.movie_rt_followup_questions import (  # noqa: E402
    build_data_quality_candidate_table,
    build_taste_numeric_holdout_table,
    build_unique_movie_prediction_frame,
    evaluate_rt_genre_model_lift,
    evaluate_holdout_blends,
    loo_linear_feature_set_metrics,
)


def plot_coefficient_intervals(
    coefficient_df: pd.DataFrame,
    output_path: Path,
) -> None:
    working = coefficient_df.loc[coefficient_df["model"] == "rt_imdb"].copy()
    summary = (
        working.groupby(["subset", "feature"])["coefficient"]
        .agg(
            median="median",
            lower=lambda values: values.quantile(0.05),
            upper=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )
    subset_order = [
        value
        for value in ["complete_all", "finished_no_flags"]
        if value in summary["subset"].unique()
    ]
    feature_order = ["rt_critic_on_10", "rt_audience_on_10", "imdb_score"]
    fig, axes = plt.subplots(
        1,
        len(subset_order),
        figsize=(6 * max(1, len(subset_order)), 5),
        sharey=True,
    )
    if len(subset_order) == 1:
        axes = [axes]
    for axis, subset_name in zip(axes, subset_order, strict=True):
        subset = summary.loc[summary["subset"] == subset_name].set_index("feature")
        x = range(len(feature_order))
        medians = [subset.loc[feature, "median"] for feature in feature_order]
        lowers = [
            medians[i] - subset.loc[feature, "lower"]
            for i, feature in enumerate(feature_order)
        ]
        uppers = [
            subset.loc[feature, "upper"] - medians[i]
            for i, feature in enumerate(feature_order)
        ]
        axis.errorbar(
            x,
            medians,
            yerr=[lowers, uppers],
            fmt="o",
            capsize=5,
            linewidth=2,
        )
        axis.set_xticks(list(x), labels=["critic", "audience", "imdb"])
        axis.set_title(subset_name.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Coefficient")
    fig.suptitle("LOO coefficient medians with 5-95% intervals")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_set_performance(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    subset_order = [
        value
        for value in ["complete_all", "finished_no_flags"]
        if value in metrics_df["subset"].unique()
    ]
    model_order = [
        "critic_only",
        "audience_only",
        "imdb_only",
        "rt_only",
        "rt_imdb",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, metric in zip(axes, ["mae", "within_one"], strict=True):
        width = 0.35
        for index, subset_name in enumerate(subset_order):
            subset = metrics_df.loc[metrics_df["subset"] == subset_name].set_index(
                "model"
            )
            values = [subset.loc[model_name, metric] for model_name in model_order]
            x = [value + index * width for value in range(len(model_order))]
            axis.bar(x, values, width=width, label=subset_name.replace("_", " "))
        axis.set_xticks(
            [value + width / 2 for value in range(len(model_order))],
            labels=model_order,
            rotation=30,
            ha="right",
        )
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(alpha=0.25, axis="y")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_correlations(
    corr_df: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = ["critic", "audience", "imdb"]
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = corr_df.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(image, ax=ax)
    ax.set_title("Predictor correlation on complete cases")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_taste_blend_performance(
    blend_metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    order = [
        "taste_agent",
        "cv_linear_rt_prediction",
        "cv_linear_rt_imdb_prediction",
        "cv_ridge_context_prediction",
        "blend50_cv_linear_rt_imdb_prediction",
        "blend50_cv_ridge_context_prediction",
    ]
    subset = blend_metrics_df.set_index("model").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, metric in zip(axes, ["mae", "within_one"], strict=True):
        axis.bar(subset["model"], subset[metric])
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_blend_alpha_curves(
    alpha_curve_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, metric in zip(axes, ["mae", "within_one"], strict=True):
        for numeric_model, subset in alpha_curve_df.groupby("numeric_model"):
            axis.plot(
                subset["alpha_numeric"],
                subset[metric],
                label=numeric_model,
            )
        axis.set_xlabel("Weight on numeric model")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_genre_lift(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ["mae", "within_one"], strict=True):
        axis.bar(metrics_df["model"], metrics_df[metric], color=["#4e79a7", "#f28e2b"])
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_data_quality_scatter(
    review_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    flagged = review_df["quality_flags"].fillna("").ne("")
    ax.scatter(
        review_df.loc[~flagged, "source_spread"],
        review_df.loc[~flagged, "model_residual"].abs(),
        alpha=0.45,
        label="No existing quality flag",
    )
    ax.scatter(
        review_df.loc[flagged, "source_spread"],
        review_df.loc[flagged, "model_residual"].abs(),
        alpha=0.8,
        color="#c0392b",
        label="Has quality flag",
    )
    top_labels = review_df.head(12)
    for _, row in top_labels.iterrows():
        ax.annotate(
            str(row["movie_title"]),
            (row["source_spread"], abs(row["model_residual"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.set_xlabel("Spread across RT critic / RT audience / IMDb")
    ax.set_ylabel("Absolute ridge-context residual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = ROOT / "data" / "summaries"
    output_dir = data_dir / "movie_rt_analysis" / "second_pass_20260404"

    second_sheet_predictions = pd.read_csv(
        output_dir / "second_sheet_cv_predictions.csv"
    )
    synopsis_holdout = pd.read_csv(output_dir / "synopsis_holdout_half.csv")
    taste_predictions = pd.read_csv(output_dir / "taste_agent_holdout_predictions.csv")

    feature_sets = {
        "critic_only": ["rt_critic_on_10"],
        "audience_only": ["rt_audience_on_10"],
        "imdb_only": ["imdb_score"],
        "rt_only": ["rt_critic_on_10", "rt_audience_on_10"],
        "rt_imdb": ["rt_critic_on_10", "rt_audience_on_10", "imdb_score"],
    }

    complete_all = second_sheet_predictions.loc[
        second_sheet_predictions[
            ["analysis_rating", "rt_critic_on_10", "rt_audience_on_10", "imdb_score"]
        ]
        .notna()
        .all(axis=1)
    ].copy()
    finished_no_flags = complete_all.loc[
        (complete_all["watch_status"] == "finished")
        & (complete_all["quality_flags"].fillna("") == "")
    ].copy()

    complete_metrics, complete_coefficients = loo_linear_feature_set_metrics(
        complete_all,
        subset_name="complete_all",
        feature_sets=feature_sets,
    )
    strict_metrics, strict_coefficients = loo_linear_feature_set_metrics(
        finished_no_flags,
        subset_name="finished_no_flags",
        feature_sets=feature_sets,
    )
    feature_metrics = pd.concat(
        [complete_metrics, strict_metrics],
        ignore_index=True,
    )
    coefficient_distribution = pd.concat(
        [complete_coefficients, strict_coefficients],
        ignore_index=True,
    )

    unique_predictions = build_unique_movie_prediction_frame(second_sheet_predictions)
    holdout_merge = build_taste_numeric_holdout_table(
        unique_predictions=unique_predictions,
        synopsis_holdout=synopsis_holdout,
        taste_predictions=taste_predictions,
    )
    taste_blend_metrics, taste_blend_alpha_curves = evaluate_holdout_blends(
        holdout_merge,
        numeric_prediction_columns=[
            "cv_linear_rt_prediction",
            "cv_linear_rt_imdb_prediction",
            "cv_ridge_context_prediction",
        ],
    )
    genre_metrics, genre_coverage, genre_table = evaluate_rt_genre_model_lift(
        second_sheet_predictions,
        cache_path=output_dir / "rt_genre_cache.json",
    )
    data_quality_candidates = build_data_quality_candidate_table(
        second_sheet_predictions,
        prediction_column="cv_ridge_context_prediction",
    )

    correlation_table = complete_all[
        ["rt_critic_on_10", "rt_audience_on_10", "imdb_score"]
    ].corr()

    feature_metrics.to_csv(output_dir / "followup_feature_set_metrics.csv", index=False)
    coefficient_distribution.to_csv(
        output_dir / "followup_coefficient_distribution.csv",
        index=False,
    )
    correlation_table.to_csv(output_dir / "followup_predictor_correlations.csv")
    holdout_merge.to_csv(output_dir / "followup_taste_numeric_holdout.csv", index=False)
    taste_blend_metrics.to_csv(
        output_dir / "followup_taste_blend_metrics.csv",
        index=False,
    )
    taste_blend_alpha_curves.to_csv(
        output_dir / "followup_taste_blend_alpha_curves.csv",
        index=False,
    )
    genre_metrics.to_csv(output_dir / "followup_rt_genre_metrics.csv", index=False)
    genre_coverage.to_csv(output_dir / "followup_rt_genre_coverage.csv", index=False)
    genre_table.to_csv(output_dir / "followup_rt_genre_table.csv", index=False)
    data_quality_candidates.to_csv(
        output_dir / "followup_data_quality_candidates.csv",
        index=False,
    )

    plot_coefficient_intervals(
        coefficient_distribution,
        output_dir / "followup_coefficient_intervals.png",
    )
    plot_feature_set_performance(
        feature_metrics,
        output_dir / "followup_feature_set_performance.png",
    )
    plot_predictor_correlations(
        correlation_table,
        output_dir / "followup_predictor_correlations.png",
    )
    plot_taste_blend_performance(
        taste_blend_metrics,
        output_dir / "followup_taste_blend_performance.png",
    )
    plot_blend_alpha_curves(
        taste_blend_alpha_curves,
        output_dir / "followup_taste_blend_alpha_curves.png",
    )
    plot_genre_lift(
        genre_metrics,
        output_dir / "followup_rt_genre_lift.png",
    )
    plot_data_quality_scatter(
        data_quality_candidates,
        output_dir / "followup_data_quality_scatter.png",
    )

    best_linear_blend = (
        taste_blend_alpha_curves.loc[
            taste_blend_alpha_curves["numeric_model"] == "cv_linear_rt_imdb_prediction"
        ]
        .sort_values("mae")
        .iloc[0]
    )
    best_ridge_blend = (
        taste_blend_alpha_curves.loc[
            taste_blend_alpha_curves["numeric_model"] == "cv_ridge_context_prediction"
        ]
        .sort_values("mae")
        .iloc[0]
    )

    summary_lines = [
        "Feature-set follow-up on cleaned second-pass data",
        (
            "Complete-case rows: "
            f"{len(complete_all)} | finished/no-flags rows: {len(finished_no_flags)}"
        ),
        (
            "RT-only vs RT+IMDb on complete cases: "
            f"{feature_metrics.loc[(feature_metrics['subset'] == 'complete_all') & (feature_metrics['model'] == 'rt_only'), 'mae'].iloc[0]:.3f} "
            "-> "
            f"{feature_metrics.loc[(feature_metrics['subset'] == 'complete_all') & (feature_metrics['model'] == 'rt_imdb'), 'mae'].iloc[0]:.3f}"
        ),
        (
            "RT-only vs RT+IMDb on finished/no-flags: "
            f"{feature_metrics.loc[(feature_metrics['subset'] == 'finished_no_flags') & (feature_metrics['model'] == 'rt_only'), 'mae'].iloc[0]:.3f} "
            "-> "
            f"{feature_metrics.loc[(feature_metrics['subset'] == 'finished_no_flags') & (feature_metrics['model'] == 'rt_imdb'), 'mae'].iloc[0]:.3f}"
        ),
        (
            "Best 50/50 taste blend with RT+IMDb linear MAE: "
            f"{taste_blend_metrics.loc[taste_blend_metrics['model'] == 'blend50_cv_linear_rt_imdb_prediction', 'mae'].iloc[0]:.3f}"
        ),
        (
            "Best 50/50 taste blend with ridge-context MAE: "
            f"{taste_blend_metrics.loc[taste_blend_metrics['model'] == 'blend50_cv_ridge_context_prediction', 'mae'].iloc[0]:.3f}"
        ),
        (
            "Best oracle ridge-context blend weight on numeric model: "
            f"{best_ridge_blend['alpha_numeric']:.2f} with MAE {best_ridge_blend['mae']:.3f}"
        ),
        (
            "Best oracle RT+IMDb linear blend weight on numeric model: "
            f"{best_linear_blend['alpha_numeric']:.2f} with MAE {best_linear_blend['mae']:.3f}"
        ),
        (
            "Context ridge vs ridge+RT genres MAE: "
            f"{genre_metrics.loc[genre_metrics['model'] == 'context_ridge_same_rows', 'mae'].iloc[0]:.3f} "
            "-> "
            f"{genre_metrics.loc[genre_metrics['model'] == 'context_ridge_plus_rt_genres', 'mae'].iloc[0]:.3f}"
        ),
        (
            "Rows with parsed RT genres: "
            f"{int(genre_coverage.iloc[0]['rows_with_genres'])} / "
            f"{int(genre_coverage.iloc[0]['rows'])}"
        ),
    ]
    (output_dir / "followup_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
