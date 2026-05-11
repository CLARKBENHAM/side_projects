# ruff: noqa: E402

from __future__ import annotations

import os
from pathlib import Path

_MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from ai_books_tracking.goodread_emperical_dist.fixed_rule_analysis import (
    FIXED_RULE_LABELS,
)
from ai_books_tracking.goodread_emperical_dist.policy_analysis import (
    CORRELATION_SERIES_LABELS,
    POLICY_DROP_FRACTIONS,
    POLICY_LABELS,
    POLICY_NAME_ORDER,
    build_profile_splits,
)

POLICY_COLORS = {
    "global_goodreads_cutoff": "#4c78a8",
    "personal_goodreads_cutoff": "#f58518",
    "shrunk_goodreads_cutoff": "#54a24b",
    "training_selected_model": "#e45756",
}
CORRELATION_COLORS = {
    "goodreads_raw": "#4c78a8",
    "training_selected_model": "#e45756",
}


def _figure_bins(values: np.ndarray, n_bins: int = 8) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return np.linspace(-0.5, 0.5, n_bins + 1)
    value_min = float(finite_values.min())
    value_max = float(finite_values.max())
    if value_min == value_max:
        span = 0.1 if value_min == 0 else abs(value_min) * 0.1
        value_min -= span
        value_max += span
    padding = (value_max - value_min) * 0.05
    return np.linspace(value_min - padding, value_max + padding, n_bins + 1)


def _percent_weights(n_values: int) -> np.ndarray:
    if n_values <= 0:
        return np.array([], dtype=float)
    return np.full(n_values, 1.0 / n_values, dtype=float)


def _kernel_smoothed_series(
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    if len(x_values) == 0:
        return np.array([], dtype=float)
    if bandwidth is None:
        bandwidth = max(4.0, float(x_values.max() - x_values.min()) / 12.0)
    smoothed = np.empty(len(x_values), dtype=float)
    for idx, center in enumerate(x_values):
        kernel = np.exp(-0.5 * ((x_values - center) / bandwidth) ** 2)
        total_weight = kernel * weights
        smoothed[idx] = np.sum(total_weight * y_values) / np.sum(total_weight)
    return smoothed


def plot_policy_gain_histograms(
    policy_holdout_results: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    bins = np.linspace(-1.25, 1.25, 33)

    for axis, target_drop_fraction in zip(axes, POLICY_DROP_FRACTIONS, strict=True):
        subset = policy_holdout_results[
            policy_holdout_results["target_drop_fraction"].eq(target_drop_fraction)
        ]
        for policy_name in POLICY_NAME_ORDER:
            policy_subset = subset[subset["policy_name"] == policy_name]
            values = policy_subset["rating_gain"].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            label = (
                f"{POLICY_LABELS[policy_name]} "
                f"(median={np.median(values):.3f}, n={len(values)})"
            )
            axis.hist(
                values,
                bins=bins,
                alpha=0.45,
                label=label,
                color=POLICY_COLORS[policy_name],
                edgecolor="white",
                weights=_percent_weights(len(values)),
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(
            f"Holdout gain when targeting drop {int(target_drop_fraction * 100)}%"
        )
        axis.set_xlabel("Rating gain on holdout")
        axis.set_ylabel("Percent of profiles")
        axis.set_xlim(-1.25, 1.25)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    figure.suptitle("Holdout rating-gain histograms by screening policy", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_holdout_correlation_histograms(
    holdout_correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    metric_specs = [
        ("spearman_rho", "Holdout Spearman correlations"),
        ("pearson_r", "Holdout Pearson correlations"),
    ]

    for axis, (metric_name, title) in zip(axes, metric_specs, strict=True):
        subset = holdout_correlations.dropna(subset=[metric_name])
        bins = _figure_bins(subset[metric_name].to_numpy(dtype=float))
        for series_name in ["goodreads_raw", "training_selected_model"]:
            series_subset = subset[subset["series_name"] == series_name]
            values = series_subset[metric_name].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            below_zero = float((values < 0).mean())
            label = (
                f"{CORRELATION_SERIES_LABELS[series_name]} "
                f"(median={np.median(values):.3f}, n={len(values)}, <0={below_zero:.0%})"
            )
            axis.hist(
                values,
                bins=bins,
                alpha=0.5,
                label=label,
                color=CORRELATION_COLORS[series_name],
                edgecolor="white",
                weights=_percent_weights(len(values)),
            )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Correlation coefficient")
        axis.set_ylabel("Percent of profiles")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_goodreads_small_multiples(
    prepared_profile_books: pd.DataFrame,
    selected_profiles: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_slugs = selected_profiles["profile_slug"].tolist()
    if not selected_slugs:
        return

    split_map = {
        split.profile_slug: split
        for split in build_profile_splits(prepared_profile_books)
    }

    figure, axes = plt.subplots(4, 5, figsize=(18, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    for axis in axes:
        axis.set_xlim(2.5, 4.8)
        axis.set_ylim(0.75, 5.25)

    rng = np.random.default_rng(20260404)
    for axis, profile_slug in zip(axes, selected_slugs, strict=False):
        split = split_map.get(profile_slug)
        if split is None:
            axis.set_visible(False)
            continue

        train_df = split.train_df.copy()
        holdout_df = split.holdout_df.copy()
        train_y = pd.to_numeric(train_df["user_rating"], errors="coerce").to_numpy(
            dtype=float
        )
        holdout_y = pd.to_numeric(holdout_df["user_rating"], errors="coerce").to_numpy(
            dtype=float
        )
        train_x = pd.to_numeric(train_df["average_rating"], errors="coerce").to_numpy(
            dtype=float
        )
        holdout_x = pd.to_numeric(
            holdout_df["average_rating"], errors="coerce"
        ).to_numpy(dtype=float)
        train_y = train_y + rng.normal(0, 0.04, size=len(train_y))
        holdout_y = holdout_y + rng.normal(0, 0.04, size=len(holdout_y))

        axis.scatter(
            train_x,
            train_y,
            s=12,
            alpha=0.30,
            color="#4c78a8",
            label="Train",
        )
        axis.scatter(
            holdout_x,
            holdout_y,
            s=18,
            alpha=0.65,
            color="#e45756",
            label="Holdout",
        )
        fit_df = split.full_df[["average_rating", "user_rating"]].apply(
            pd.to_numeric, errors="coerce"
        )
        fit_df = fit_df.dropna()
        if fit_df["average_rating"].nunique() >= 2:
            slope, intercept = np.polyfit(
                fit_df["average_rating"].to_numpy(dtype=float),
                fit_df["user_rating"].to_numpy(dtype=float),
                deg=1,
            )
            x_line = np.array([2.5, 4.8], dtype=float)
            y_line = intercept + slope * x_line
            axis.plot(
                x_line,
                y_line,
                color="#1f1f1f",
                linewidth=1.2,
                alpha=0.9,
            )

        all_corr = split.full_df["average_rating"].corr(
            split.full_df["user_rating"], method="spearman"
        )
        axis.set_title(
            f"{split.display_name}\n"
            f"n={split.n_rated_books}, rho={all_corr:.2f}, gap={split.recency_gap:.0f}",
            fontsize=9,
        )

    legend_axis = axes[len(selected_slugs)] if len(selected_slugs) < len(axes) else None
    for axis in axes[len(selected_slugs) + (1 if legend_axis is not None else 0) :]:
        axis.clear()
        axis.axis("off")

    if legend_axis is not None:
        legend_axis.clear()
        legend_axis.axis("off")
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#4c78a8",
                markersize=7,
                alpha=0.6,
                label="Training books",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#e45756",
                markersize=7,
                alpha=0.8,
                label="Holdout books",
            ),
            plt.Line2D(
                [0],
                [0],
                color="#1f1f1f",
                linewidth=1.2,
                label="Per-profile fitted line",
            ),
        ]
        legend_axis.legend(handles=handles, loc="center", frameon=False)
        legend_axis.text(
            0.5,
            0.15,
            "Selection prefers user-supplied and verified profiles,\nthen recency gap <= 7 and dominant-year share <= 0.50,\nthen ranks by lower recency gap, lower concentration,\nand larger history size.",
            ha="center",
            va="center",
            fontsize=10,
        )

    figure.supxlabel("Goodreads average rating")
    figure.supylabel("User rating")
    figure.suptitle(
        f"Goodreads vs user ratings for {len(selected_slugs)} most temporally credible profiles"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_goodreads_cutoff_percentile_tradeoff(
    percentile_summary: pd.DataFrame,
    output_path: Path,
    scope_label: str = "all rated books",
) -> None:
    if percentile_summary.empty:
        return

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    drop_axis = axis.twinx()
    percentile_specs = [
        (85, "#1b9e77"),
        (65, "#1f78b4"),
        (35, "#7570b3"),
        (15, "#d95f02"),
    ]

    x_values = percentile_summary["threshold"].to_numpy(dtype=float)
    for percentile, color in percentile_specs:
        y_values = percentile_summary[f"gain_p{percentile}"].to_numpy(dtype=float)
        axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"{percentile}th percentile gain",
        )

    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    drop_axis.plot(
        x_values,
        percentile_summary["drop_share_median"].to_numpy(dtype=float) * 100,
        color="#6c6c6c",
        linestyle=":",
        linewidth=2.0,
        marker="s",
        markersize=4,
        label="Median share dropped",
    )
    axis.set_xlabel("Goodreads average-rating cutoff")
    axis.set_ylabel("Rating gain")
    drop_axis.set_ylabel("Median share dropped")
    drop_axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.set_title(
        f"Across readers: gain distribution from a fixed Goodreads cutoff on {scope_label}\n"
        "Colored lines are gain percentiles; grey dotted line is median share dropped"
    )
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    drop_handles, drop_labels = drop_axis.get_legend_handles_labels()
    axis.legend(handles + drop_handles, labels + drop_labels, frameon=False)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_goodreads_cutoff_percentile_and_violin(
    percentile_summary: pd.DataFrame,
    profile_metrics: pd.DataFrame,
    output_path: Path,
    scope_label: str = "all rated books",
) -> None:
    if percentile_summary.empty or profile_metrics.empty:
        return

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    percentile_axis, violin_axis = axes
    percentile_specs = [
        (85, "#1b9e77"),
        (65, "#1f78b4"),
        (35, "#7570b3"),
        (15, "#d95f02"),
    ]
    x_values = percentile_summary["threshold"].to_numpy(dtype=float)

    for percentile, color in percentile_specs:
        percentile_axis.plot(
            x_values,
            percentile_summary[f"gain_p{percentile}"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=color,
            label=f"{percentile}th percentile gain",
        )

    violin_data: list[np.ndarray] = []
    violin_positions: list[float] = []
    for threshold, threshold_frame in profile_metrics.groupby("threshold", sort=True):
        gains = threshold_frame["rating_gain"].dropna().to_numpy(dtype=float)
        if len(gains) == 0:
            continue
        violin_positions.append(float(threshold))
        violin_data.append(gains)
    if violin_data:
        violin_parts = violin_axis.violinplot(
            violin_data,
            positions=violin_positions,
            widths=0.075,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in violin_parts["bodies"]:
            body.set_facecolor("#4c78a8")
            body.set_edgecolor("#2f4b7c")
            body.set_alpha(0.35)
        violin_parts["cmedians"].set_color("#1f1f1f")
        violin_parts["cmedians"].set_linewidth(1.3)

    for axis in axes:
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.2)
        axis.set_ylabel("Rating gain")
    percentile_axis.set_title("Percentile summary across readers")
    percentile_axis.legend(frameon=False)
    violin_axis.set_title("Per-cutoff gain distribution across readers")
    violin_axis.set_xlabel("Goodreads average-rating cutoff")

    figure.suptitle(
        f"Across readers: gains from a fixed Goodreads cutoff on {scope_label}\n"
        "Percentile lines and violin distributions share the same axes"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _draw_violin_panel(
    axis: plt.Axes,
    profile_metrics: pd.DataFrame,
    value_column: str,
    color: str,
    edge_color: str,
    title: str,
    ylabel: str,
    scale: float = 1.0,
    percent_axis: bool = False,
) -> None:
    violin_data: list[np.ndarray] = []
    violin_positions: list[float] = []
    for threshold, threshold_frame in profile_metrics.groupby("threshold", sort=True):
        values = threshold_frame[value_column].dropna().to_numpy(dtype=float) * scale
        if len(values) == 0:
            continue
        violin_positions.append(float(threshold))
        violin_data.append(values)
    if not violin_data:
        return

    violin_parts = axis.violinplot(
        violin_data,
        positions=violin_positions,
        widths=0.075,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(edge_color)
        body.set_alpha(0.35)
    violin_parts["cmedians"].set_color("#1f1f1f")
    violin_parts["cmedians"].set_linewidth(1.3)
    axis.grid(alpha=0.2)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if percent_axis:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))


def plot_goodreads_cutoff_gain_and_drop_violins(
    profile_metrics: pd.DataFrame,
    output_path: Path,
    scope_label: str = "holdout books",
) -> None:
    if profile_metrics.empty:
        return

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 13),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.1]},
    )
    gain_axis, drop_axis, rating_axis = axes

    _draw_violin_panel(
        gain_axis,
        profile_metrics=profile_metrics,
        value_column="rating_gain",
        color="#4c78a8",
        edge_color="#2f4b7c",
        title=f"Per-cutoff gain distribution across readers on {scope_label}",
        ylabel="Rating gain",
    )
    gain_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    _draw_violin_panel(
        drop_axis,
        profile_metrics=profile_metrics,
        value_column="actual_drop_share",
        color="#f58518",
        edge_color="#b85c00",
        title=f"Per-cutoff share of {scope_label} dropped across readers",
        ylabel="Books dropped",
        scale=100.0,
        percent_axis=True,
    )
    drop_axis.set_xlabel("Goodreads average-rating cutoff")

    _draw_violin_panel(
        rating_axis,
        profile_metrics=profile_metrics,
        value_column="kept_mean",
        color="#e45756",
        edge_color="#a62c2b",
        title=f"Per-cutoff average kept rating across readers on {scope_label}",
        ylabel="Average kept rating",
    )
    rating_axis.set_yticks([1, 2, 3, 4, 5])
    rating_axis.set_ylim(0.9, 5.1)
    rating_axis.set_xlabel("Goodreads average-rating cutoff")

    figure.suptitle(
        f"Across readers: gain, drop-share, and kept-rating distributions from a fixed Goodreads cutoff on {scope_label}"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_fixed_rule_rating_share_violins(
    rating_shares: pd.DataFrame,
    profile_metrics: pd.DataFrame,
    output_path: Path,
    figure_title: str = (
        "Across readers: distribution of personal star shares before and after two simple Goodreads rules"
    ),
    subtitle_suffix: str = "",
) -> None:
    if rating_shares.empty or profile_metrics.empty:
        return

    rule_names = [
        "original_distribution",
        "goodreads_cutoff_4_0",
        "drop_lower_goodreads_half",
    ]
    summary_lookup = (
        profile_metrics.drop_duplicates(subset=["profile_slug", "rule_name"])[
            ["profile_slug", "rule_name", "rating_gain", "actual_drop_share"]
        ]
        .groupby("rule_name", sort=False)
        .agg(
            median_gain=("rating_gain", "median"),
            median_drop_share=("actual_drop_share", "median"),
            n_profiles=("profile_slug", "nunique"),
        )
        .reset_index()
        .set_index("rule_name")
    )
    colors = {
        "original_distribution": ("#9c755f", "#6a4c3d"),
        "goodreads_cutoff_4_0": ("#4c78a8", "#2f4b7c"),
        "drop_lower_goodreads_half": ("#54a24b", "#2f6a2f"),
    }

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5.7),
        constrained_layout=True,
        sharey=True,
    )

    for axis, rule_name in zip(axes, rule_names, strict=True):
        _draw_fixed_rule_rating_share_panel(
            axis=axis,
            rating_shares=rating_shares[rating_shares["rule_name"] == rule_name].copy(),
            rule_name=rule_name,
            colors=colors,
        )
        summary_row = summary_lookup.loc[rule_name]
        if rule_name == "original_distribution":
            subtitle = f"n={int(summary_row['n_profiles'])}"
        else:
            subtitle = (
                f"median gain {summary_row['median_gain']:.3f}, "
                f"median drop {summary_row['median_drop_share']:.0%}"
            )
        if subtitle_suffix:
            subtitle = f"{subtitle}\n{subtitle_suffix}"
        axis.set_title(f"{FIXED_RULE_LABELS[rule_name]}\n{subtitle}")
    axes[0].set_ylabel("% of kept books for each reader")
    figure.suptitle(figure_title)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_fixed_rule_rating_share_grouped_rows(
    row_specs: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not row_specs:
        return

    figure, axes = plt.subplots(
        len(row_specs),
        1,
        figsize=(14, 4.2 * len(row_specs)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if len(row_specs) == 1:
        axes = np.asarray([axes])

    rule_names = [
        "original_distribution",
        "goodreads_cutoff_4_0",
        "drop_lower_goodreads_half",
    ]
    colors = {
        "original_distribution": ("#9c755f", "#6a4c3d"),
        "goodreads_cutoff_4_0": ("#4c78a8", "#2f4b7c"),
        "drop_lower_goodreads_half": ("#54a24b", "#2f6a2f"),
    }
    offsets = {
        "original_distribution": -0.26,
        "goodreads_cutoff_4_0": 0.0,
        "drop_lower_goodreads_half": 0.26,
    }
    summary_lookup_frames: list[pd.DataFrame] = []
    for spec in row_specs:
        summary = (
            spec["profile_metrics"]
            .drop_duplicates(subset=["profile_slug", "rule_name"])[
                ["profile_slug", "rule_name", "rating_gain", "actual_drop_share"]
            ]
            .groupby("rule_name", sort=False)
            .agg(
                median_gain=("rating_gain", "median"),
                median_drop_share=("actual_drop_share", "median"),
                n_profiles=("profile_slug", "nunique"),
            )
            .reset_index()
            .set_index("rule_name")
        )
        summary_lookup_frames.append(summary)

    for axis, spec, summary_lookup in zip(
        axes, row_specs, summary_lookup_frames, strict=True
    ):
        rating_shares = spec["rating_shares"]
        for rule_name in rule_names:
            face_color, edge_color = colors[rule_name]
            violin_data: list[np.ndarray] = []
            positions: list[float] = []
            for rating_value in range(1, 6):
                values = (
                    rating_shares[
                        (rating_shares["rule_name"] == rule_name)
                        & (rating_shares["rating_value"] == rating_value)
                    ]["share_percent"]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if len(values) == 0:
                    continue
                positions.append(float(rating_value) + offsets[rule_name])
                violin_data.append(values)
            if violin_data:
                violin_parts = axis.violinplot(
                    violin_data,
                    positions=positions,
                    widths=0.22,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body in violin_parts["bodies"]:
                    body.set_facecolor(face_color)
                    body.set_edgecolor(edge_color)
                    body.set_alpha(0.35)
                violin_parts["cmedians"].set_color("#1f1f1f")
                violin_parts["cmedians"].set_linewidth(1.1)
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.set_xlim(0.4, 5.6)
        axis.set_ylim(0, 100)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis.grid(alpha=0.2)
        axis.set_xlabel("Your rating")
        rule_descriptions = []
        for rule_name in rule_names[1:]:
            rule_descriptions.append(
                f"{FIXED_RULE_LABELS[rule_name]} gain {summary_lookup.loc[rule_name, 'median_gain']:.3f}"
            )
        axis.set_title(
            f"{spec['row_title']}\n"
            f"n={int(summary_lookup.loc['original_distribution', 'n_profiles'])}; "
            + " | ".join(rule_descriptions),
            fontsize=11,
        )
    axes[0].legend(
        [
            plt.Line2D([0], [0], color=colors[rule_name][0], linewidth=6, alpha=0.6)
            for rule_name in rule_names
        ],
        [FIXED_RULE_LABELS[rule_name] for rule_name in rule_names],
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    axes[0].set_ylabel("% of kept books for each reader")
    for axis in axes[1:]:
        axis.set_ylabel("% of kept books for each reader")
    figure.suptitle(
        "Across readers: grouped rating-share violins before and after simple Goodreads rules\n"
        "Within each 1-5 rating bucket, the three adjacent violins are original books, Goodreads >= 4.0, and drop-lower-half"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_fixed_rule_rating_share_panel_grid(
    row_specs: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not row_specs:
        return

    rule_names = [
        "original_distribution",
        "goodreads_cutoff_4_0",
        "drop_lower_goodreads_half",
    ]
    colors = {
        "original_distribution": ("#9c755f", "#6a4c3d"),
        "goodreads_cutoff_4_0": ("#4c78a8", "#2f4b7c"),
        "drop_lower_goodreads_half": ("#54a24b", "#2f6a2f"),
    }
    figure, axes = plt.subplots(
        len(row_specs),
        len(rule_names),
        figsize=(15, 4.2 * len(row_specs)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if len(row_specs) == 1:
        axes = np.asarray([axes])

    for row_index, spec in enumerate(row_specs):
        summary_lookup = (
            spec["profile_metrics"]
            .drop_duplicates(subset=["profile_slug", "rule_name"])[
                ["profile_slug", "rule_name", "rating_gain", "actual_drop_share"]
            ]
            .groupby("rule_name", sort=False)
            .agg(
                median_gain=("rating_gain", "median"),
                median_drop_share=("actual_drop_share", "median"),
                n_profiles=("profile_slug", "nunique"),
            )
            .reset_index()
            .set_index("rule_name")
        )
        for col_index, rule_name in enumerate(rule_names):
            axis = axes[row_index, col_index]
            _draw_fixed_rule_rating_share_panel(
                axis=axis,
                rating_shares=spec["rating_shares"][
                    spec["rating_shares"]["rule_name"] == rule_name
                ].copy(),
                rule_name=rule_name,
                colors=colors,
            )
            summary_row = summary_lookup.loc[rule_name]
            if rule_name == "original_distribution":
                subtitle = f"n={int(summary_row['n_profiles'])}"
            else:
                subtitle = (
                    f"median gain {summary_row['median_gain']:.3f}, "
                    f"median drop {summary_row['median_drop_share']:.0%}"
                )
            axis.set_title(
                f"{FIXED_RULE_LABELS[rule_name]}\n{subtitle}\n{spec['subtitle_suffix']}",
                fontsize=11,
            )
            if col_index == 0:
                axis.set_ylabel(f"{spec['row_title']}\n% of kept books for each reader")
    figure.suptitle(
        "Across readers: fixed-rule rating-share violins across the full panel and percentile-trimmed panels"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_fixed_rule_parameter_comparison_heatmaps(
    comparison_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if comparison_summary.empty:
        return

    cutoff_order = sorted(comparison_summary["cutoff_threshold"].unique())
    drop_order = sorted(comparison_summary["target_drop_fraction"].unique())
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(21, 5.3),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    metric_specs = [
        (
            "benefit_share_percent",
            "% of readers helped by cutoff",
            "viridis",
            None,
            "Percent of readers",
        ),
        (
            "mean_gain_difference_cutoff_minus_target_drop",
            "Mean gain diff (cutoff - drop-% rule)",
            "coolwarm",
            "diff",
            "Mean gain difference",
        ),
        (
            "gain_difference_p25",
            "25th percentile gain diff",
            "coolwarm",
            "diff",
            "25th-percentile gain difference",
        ),
        (
            "gain_difference_p75",
            "75th percentile gain diff",
            "coolwarm",
            "diff",
            "75th-percentile gain difference",
        ),
    ]

    diff_values = comparison_summary[
        [
            "mean_gain_difference_cutoff_minus_target_drop",
            "gain_difference_p25",
            "gain_difference_p75",
        ]
    ].to_numpy(dtype=float)
    diff_limit = float(np.nanmax(np.abs(diff_values))) if diff_values.size else 0.0
    diff_limit = max(diff_limit, 0.05)
    diff_norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit)

    for axis, (column, title, cmap, norm_kind, formatter) in zip(
        axes, metric_specs, strict=True
    ):
        pivot = (
            comparison_summary.pivot(
                index="target_drop_fraction",
                columns="cutoff_threshold",
                values=column,
            )
            .reindex(index=drop_order, columns=cutoff_order)
            .to_numpy(dtype=float)
        )
        image = axis.imshow(
            pivot,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0.0 if norm_kind is None else None,
            vmax=100.0 if norm_kind is None else None,
            norm=diff_norm if norm_kind == "diff" else None,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(len(cutoff_order)))
        axis.set_xticklabels([f"{cutoff:.1f}" for cutoff in cutoff_order])
        axis.set_yticks(range(len(drop_order)))
        axis.set_yticklabels([f"{drop_fraction:.0%}" for drop_fraction in drop_order])
        axis.set_xlabel("Goodreads cutoff")
        if axis is axes[0]:
            axis.set_ylabel("Bottom share dropped by Goodreads score")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label(formatter)

    figure.suptitle(
        "Across readers: when does a fixed Goodreads cutoff beat dropping the bottom share by Goodreads score?\n"
        "Positive differences mean the fixed cutoff yields a larger average personal-rating gain on all rated books"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _draw_fixed_rule_rating_share_panel(
    axis: plt.Axes,
    rating_shares: pd.DataFrame,
    rule_name: str,
    colors: dict[str, tuple[str, str]],
) -> None:
    violin_data: list[np.ndarray] = []
    positions: list[int] = []
    for rating_value in range(1, 6):
        values = (
            rating_shares[rating_shares["rating_value"] == rating_value][
                "share_percent"
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) == 0:
            continue
        positions.append(rating_value)
        violin_data.append(values)
    if violin_data:
        face_color, edge_color = colors[rule_name]
        violin_parts = axis.violinplot(
            violin_data,
            positions=positions,
            widths=0.7,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in violin_parts["bodies"]:
            body.set_facecolor(face_color)
            body.set_edgecolor(edge_color)
            body.set_alpha(0.35)
        violin_parts["cmedians"].set_color("#1f1f1f")
        violin_parts["cmedians"].set_linewidth(1.3)
    axis.set_xticks([1, 2, 3, 4, 5])
    axis.set_ylim(0, 100)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axis.grid(alpha=0.2)
    axis.set_xlabel("Your rating")


def plot_fixed_rule_rating_share_grid_by_correlation(
    rating_shares: pd.DataFrame,
    profile_metrics: pd.DataFrame,
    band_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if rating_shares.empty or profile_metrics.empty or band_summary.empty:
        return

    rule_names = [
        "original_distribution",
        "goodreads_cutoff_4_0",
        "drop_lower_goodreads_half",
    ]
    colors = {
        "original_distribution": ("#9c755f", "#6a4c3d"),
        "goodreads_cutoff_4_0": ("#4c78a8", "#2f4b7c"),
        "drop_lower_goodreads_half": ("#54a24b", "#2f6a2f"),
    }
    summary_lookup = (
        profile_metrics.drop_duplicates(subset=["profile_slug", "rule_name"])[
            [
                "profile_slug",
                "rule_name",
                "correlation_band",
                "rating_gain",
                "actual_drop_share",
            ]
        ]
        .groupby(["correlation_band", "rule_name"], sort=False)
        .agg(
            median_gain=("rating_gain", "median"),
            median_drop_share=("actual_drop_share", "median"),
            n_profiles=("profile_slug", "nunique"),
        )
        .reset_index()
        .set_index(["correlation_band", "rule_name"])
    )

    band_order = band_summary["band_name"].tolist()
    figure, axes = plt.subplots(
        len(band_order),
        len(rule_names),
        figsize=(15, 4.1 * len(band_order)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if len(band_order) == 1:
        axes = np.asarray([axes])

    for row_index, band_name in enumerate(band_order):
        band_row = band_summary[band_summary["band_name"] == band_name].iloc[0]
        for col_index, rule_name in enumerate(rule_names):
            axis = axes[row_index, col_index]
            subset = rating_shares[
                (rating_shares["correlation_band"] == band_name)
                & (rating_shares["rule_name"] == rule_name)
            ].copy()
            _draw_fixed_rule_rating_share_panel(
                axis=axis,
                rating_shares=subset,
                rule_name=rule_name,
                colors=colors,
            )
            summary_row = summary_lookup.loc[(band_name, rule_name)]
            if rule_name == "original_distribution":
                subtitle = (
                    f"n={int(summary_row['n_profiles'])}, "
                    f"rho {band_row['lower_bound']:.2f} to {band_row['upper_bound']:.2f}"
                )
            else:
                subtitle = (
                    f"gain {summary_row['median_gain']:.3f}, "
                    f"drop {summary_row['median_drop_share']:.0%}"
                )
            axis.set_title(f"{FIXED_RULE_LABELS[rule_name]}\n{subtitle}", fontsize=11)
            if col_index == 0:
                axis.set_ylabel(
                    f"{band_row['band_label']}\n% of kept books for each reader"
                )
    figure.suptitle(
        "Across readers: rating-share violins by Goodreads-correlation band\n"
        "Rows are correlation percentile bands; columns are the original books, Goodreads >= 4.0, and drop-lower-half rules"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_reviewer_volume_distributions(
    reviewer_volume: pd.DataFrame,
    reviewer_year_counts: pd.DataFrame,
    yearly_activity_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if (
        reviewer_volume.empty
        or reviewer_year_counts.empty
        or yearly_activity_summary.empty
    ):
        return

    figure, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    profile_axis, year_axis, activity_axis = axes

    profile_values = reviewer_volume["total_rated_books"].to_numpy(dtype=float)
    profile_display_cap = 5000.0
    clipped_profile_values = np.minimum(profile_values, profile_display_cap)
    profile_bins = np.arange(0, profile_display_cap + 250, 250)
    profile_axis.hist(
        clipped_profile_values,
        bins=profile_bins,
        color="#4c78a8",
        alpha=0.7,
        edgecolor="white",
    )
    profile_axis.axvline(
        reviewer_volume["total_rated_books"].median(),
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    overflow_count = int(
        (reviewer_volume["total_rated_books"] > profile_display_cap).sum()
    )
    if overflow_count > 0:
        profile_axis.text(
            0.98,
            0.93,
            f"{overflow_count} reviewers above {int(profile_display_cap):,}\npooled into last bin",
            transform=profile_axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
        )
    profile_axis.set_title("Total books rated per reviewer")
    profile_axis.set_xlabel("Public Goodreads ratings")
    profile_axis.set_ylabel("Reviewers")
    profile_axis.grid(alpha=0.2)

    year_values = reviewer_year_counts["n_books"].to_numpy(dtype=float)
    reviewer_year_display_cap = 200.0
    clipped_year_values = np.minimum(year_values, reviewer_year_display_cap)
    year_bins = np.arange(0.5, reviewer_year_display_cap + 5.5, 5)
    year_axis.hist(
        clipped_year_values,
        bins=year_bins,
        color="#f58518",
        alpha=0.7,
        edgecolor="white",
    )
    year_axis.axvline(
        reviewer_year_counts["n_books"].median(),
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    reviewer_year_overflow = int(
        (reviewer_year_counts["n_books"] > reviewer_year_display_cap).sum()
    )
    if reviewer_year_overflow > 0:
        year_axis.text(
            0.98,
            0.93,
            f"{reviewer_year_overflow} reviewer-years above {int(reviewer_year_display_cap)}\npooled into last bin",
            transform=year_axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
        )
    year_axis.set_title("Books rated per active reviewer-year")
    year_axis.set_xlabel("Rated books in an active year")
    year_axis.set_ylabel("Reviewer-years")
    year_axis.grid(alpha=0.2)

    years = yearly_activity_summary["event_year"].to_numpy(dtype=int)
    reviews_axis = activity_axis.twinx()
    activity_axis.plot(
        years,
        yearly_activity_summary["active_reviewers"].to_numpy(dtype=float),
        color="#54a24b",
        linewidth=2.0,
        label="Active reviewers",
    )
    reviews_axis.bar(
        years,
        yearly_activity_summary["total_reviews"].to_numpy(dtype=float),
        color="#9ecae9",
        alpha=0.5,
        label="Reviews in year",
    )
    activity_axis.set_title("Panel activity by calendar year")
    activity_axis.set_xlabel("Year")
    activity_axis.set_ylabel("Active reviewers", color="#54a24b")
    reviews_axis.set_ylabel("Reviews in year", color="#4c78a8")
    activity_axis.tick_params(axis="y", colors="#54a24b")
    reviews_axis.tick_params(axis="y", colors="#4c78a8")
    activity_axis.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], color="#54a24b", linewidth=2.0, label="Active reviewers"),
        plt.Rectangle(
            (0, 0), 1, 1, color="#9ecae9", alpha=0.5, label="Reviews in year"
        ),
    ]
    activity_axis.legend(handles=handles, loc="upper left", frameon=False)

    figure.suptitle("Reviewer volume distributions")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_book_overlap_summary(
    overlap_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if overlap_summary.empty:
        return

    figure, axes = plt.subplots(
        2, 1, figsize=(12, 9), constrained_layout=True, sharex=True
    )
    corr_axis, sd_axis = axes
    x_values = overlap_summary["n_raters"].to_numpy(dtype=float)
    book_weights = overlap_summary["n_books"].to_numpy(dtype=float)
    smoothed_corr = _kernel_smoothed_series(
        x_values,
        overlap_summary["mean_pairwise_profile_corr"].to_numpy(dtype=float),
        weights=book_weights,
    )
    smoothed_sd = _kernel_smoothed_series(
        x_values,
        overlap_summary["mean_rating_sd"].to_numpy(dtype=float),
        weights=book_weights,
    )

    corr_axis.scatter(
        x_values,
        overlap_summary["mean_pairwise_profile_corr"].to_numpy(dtype=float),
        color="#4c78a8",
        s=24 + np.sqrt(book_weights),
        alpha=0.4,
        edgecolors="none",
    )
    corr_axis.plot(x_values, smoothed_corr, color="#1f4f99", linewidth=2.4)
    corr_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    corr_axis.set_ylabel("Mean pairwise reviewer correlation")
    corr_axis.set_title(
        "Average pairwise reviewer correlation by book overlap count\n"
        "Points are overlap buckets; dark line is weighted kernel smoothing"
    )
    corr_axis.grid(alpha=0.2)

    sd_axis.scatter(
        x_values,
        overlap_summary["mean_rating_sd"].to_numpy(dtype=float),
        color="#e45756",
        s=24 + np.sqrt(book_weights),
        alpha=0.4,
        edgecolors="none",
    )
    sd_axis.plot(x_values, smoothed_sd, color="#a62c2b", linewidth=2.4)
    sd_axis.set_xlabel("People who rated the same book")
    sd_axis.set_ylabel("Mean within-book rating SD")
    sd_axis.set_title("Within-book rating dispersion by overlap count")
    sd_axis.grid(alpha=0.2)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_goodreads_rating_count_accuracy(
    rating_count_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if rating_count_summary.empty:
        return

    summary = rating_count_summary.sort_values("rating_count_median").copy()
    x_values = summary["rating_count_median"].to_numpy(dtype=float)
    point_sizes = 26 + np.sqrt(summary["n_sample_ratings"].to_numpy(dtype=float))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        constrained_layout=True,
        sharex=True,
    )
    corr_axis, error_axis = axes

    def draw_bootstrap_errorbars(
        axis: plt.Axes,
        metric_column: str,
        color: str,
    ) -> None:
        y_values = summary[metric_column].to_numpy(dtype=float)
        for interval_name, alpha, line_width, cap_size in [
            ("p95", 0.18, 8.0, 0.0),
            ("p80", 0.45, 2.4, 3.0),
        ]:
            low_column = f"{metric_column}_{interval_name}_low"
            high_column = f"{metric_column}_{interval_name}_high"
            if low_column not in summary.columns or high_column not in summary.columns:
                continue
            low_values = summary[low_column].to_numpy(dtype=float)
            high_values = summary[high_column].to_numpy(dtype=float)
            mask = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & np.isfinite(low_values)
                & np.isfinite(high_values)
            )
            if not mask.any():
                continue
            lower_error = np.maximum(y_values[mask] - low_values[mask], 0.0)
            upper_error = np.maximum(high_values[mask] - y_values[mask], 0.0)
            axis.errorbar(
                x_values[mask],
                y_values[mask],
                yerr=np.vstack([lower_error, upper_error]),
                fmt="none",
                ecolor=color,
                elinewidth=line_width,
                capsize=cap_size,
                alpha=alpha,
                zorder=1,
            )

    corr_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    draw_bootstrap_errorbars(corr_axis, "row_spearman_rho", "#4c78a8")
    corr_axis.plot(
        x_values,
        summary["row_spearman_rho"].to_numpy(dtype=float),
        color="#4c78a8",
        linewidth=2.0,
        marker="o",
        label="Individual ratings vs Goodreads average",
    )
    corr_axis.scatter(
        x_values,
        summary["row_spearman_rho"].to_numpy(dtype=float),
        color="#4c78a8",
        s=point_sizes,
        alpha=0.65,
        edgecolors="none",
    )
    draw_bootstrap_errorbars(corr_axis, "book_mean_spearman_rho", "#54a24b")
    corr_axis.plot(
        x_values,
        summary["book_mean_spearman_rho"].to_numpy(dtype=float),
        color="#54a24b",
        linewidth=2.0,
        marker="o",
        label="Book sample mean vs Goodreads average",
    )
    corr_axis.scatter(
        x_values,
        summary["book_mean_spearman_rho"].to_numpy(dtype=float),
        color="#54a24b",
        s=point_sizes,
        alpha=0.65,
        edgecolors="none",
    )
    corr_axis.set_ylabel("Spearman correlation")
    corr_axis.set_title(
        "Goodreads average-rating accuracy by total Goodreads rating count\n"
        "Outer bars are central 95% bootstraps; inner capped bars are central 80%"
    )
    corr_axis.grid(alpha=0.2)
    corr_axis.legend(loc="best", frameon=False)

    draw_bootstrap_errorbars(error_axis, "mean_individual_absolute_error", "#e45756")
    error_axis.plot(
        x_values,
        summary["mean_individual_absolute_error"].to_numpy(dtype=float),
        color="#e45756",
        linewidth=2.0,
        marker="o",
        label="Mean individual absolute error",
    )
    draw_bootstrap_errorbars(error_axis, "mean_book_absolute_error", "#f58518")
    error_axis.plot(
        x_values,
        summary["mean_book_absolute_error"].to_numpy(dtype=float),
        color="#f58518",
        linewidth=2.0,
        marker="o",
        label="Mean book-average absolute error",
    )
    error_axis.set_xscale("log")
    error_axis.set_xlabel(
        "Total Goodreads ratings for the book (bin median, log scale)"
    )
    error_axis.set_ylabel("Absolute error in stars")
    error_axis.grid(alpha=0.2)
    error_axis.legend(loc="best", frameon=False)

    for axis in axes:
        axis.set_xlim(
            left=max(1.0, float(np.nanmin(x_values)) * 0.8),
            right=float(np.nanmax(x_values)) * 1.25,
        )

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_yearly_gain_difference_histograms(
    yearly_gain_differences: pd.DataFrame,
    output_path: Path,
) -> None:
    if yearly_gain_differences.empty:
        return

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes = axes.flatten()
    bins = np.linspace(-0.8, 0.8, 33)
    for axis, drop_fraction in zip(axes, POLICY_DROP_FRACTIONS, strict=True):
        subset = yearly_gain_differences[
            yearly_gain_differences["drop_fraction"].eq(drop_fraction)
        ]
        values = (
            subset["gain_difference_year_2_minus_year_1"].dropna().to_numpy(dtype=float)
        )
        if len(values) == 0:
            axis.set_visible(False)
            continue
        axis.hist(
            values,
            bins=bins,
            color="#4c78a8",
            alpha=0.7,
            edgecolor="white",
            weights=_percent_weights(len(values)),
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(f"Year-2 minus year-1 gain at drop {int(drop_fraction * 100)}%")
        axis.set_xlabel("Gain difference")
        axis.set_ylabel("Percent of year-pairs")
        axis.set_xlim(-0.8, 0.8)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.grid(alpha=0.2)
        axis.text(
            0.98,
            0.96,
            "\n".join(
                [
                    f"mean gain y1={subset['gain_year_1'].mean():.3f}",
                    f"mean gain y2={subset['gain_year_2'].mean():.3f}",
                    f"sd gain y1={subset['gain_year_1'].std():.3f}",
                    f"sd gain y2={subset['gain_year_2'].std():.3f}",
                    f"mean delta={subset['gain_difference_year_2_minus_year_1'].mean():.3f}",
                    f"sd delta={subset['gain_difference_year_2_minus_year_1'].std():.3f}",
                ]
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.82,
            },
        )

    figure.suptitle(
        "Year-to-year gain differences from fixed Goodreads screening rates"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _scatter_with_identity(
    axis: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if len(x_values) == 0:
        axis.set_visible(False)
        return
    axis.scatter(
        x_values,
        y_values,
        color="#4c78a8",
        alpha=0.55,
        s=28,
        edgecolors="none",
    )
    all_values = np.concatenate([x_values, y_values])
    lower = float(np.nanmin(all_values))
    upper = float(np.nanmax(all_values))
    padding = max(0.05, (upper - lower) * 0.06)
    axis.plot(
        [lower - padding, upper + padding],
        [lower - padding, upper + padding],
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)


def plot_yearly_correlation_consistency(
    yearly_correlation_pairs: pd.DataFrame,
    output_path: Path,
) -> None:
    if yearly_correlation_pairs.empty:
        return

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    spearman = yearly_correlation_pairs[
        ["spearman_rho_year_1", "spearman_rho_year_2", "spearman_rho_delta"]
    ].dropna()
    pearson = yearly_correlation_pairs[
        ["pearson_r_year_1", "pearson_r_year_2", "pearson_r_delta"]
    ].dropna()

    _scatter_with_identity(
        axes[0, 0],
        spearman["spearman_rho_year_1"].to_numpy(dtype=float),
        spearman["spearman_rho_year_2"].to_numpy(dtype=float),
        "Spearman correlation consistency",
        "Year 1 Spearman",
        "Year 2 Spearman",
    )
    _scatter_with_identity(
        axes[1, 0],
        pearson["pearson_r_year_1"].to_numpy(dtype=float),
        pearson["pearson_r_year_2"].to_numpy(dtype=float),
        "Pearson correlation consistency",
        "Year 1 Pearson",
        "Year 2 Pearson",
    )

    for axis, values, title in [
        (axes[0, 1], spearman["spearman_rho_delta"], "Spearman year-2 minus year-1"),
        (axes[1, 1], pearson["pearson_r_delta"], "Pearson year-2 minus year-1"),
    ]:
        array = values.to_numpy(dtype=float)
        bins = _figure_bins(array, n_bins=12)
        axis.hist(
            array,
            bins=bins,
            color="#f58518",
            alpha=0.7,
            edgecolor="white",
        )
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Delta")
        axis.set_ylabel("Year pairs")
        axis.grid(alpha=0.2)

    figure.suptitle("Year-to-year consistency of Goodreads vs user-rating correlation")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_yearly_cutoff_transfer_summary(
    yearly_transfer_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    if yearly_transfer_metrics.empty:
        return

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    efficiency_axis, regret_axis, drop_axis = axes

    efficiency = (
        yearly_transfer_metrics["year_2_optimal_efficiency"]
        .dropna()
        .to_numpy(dtype=float)
    )
    efficiency_axis.hist(
        efficiency,
        bins=np.linspace(-0.5, 1.5, 33),
        color="#4c78a8",
        alpha=0.75,
        edgecolor="white",
        weights=_percent_weights(len(efficiency)),
    )
    efficiency_axis.axvline(
        np.median(efficiency), color="black", linestyle="--", linewidth=1.0
    )
    efficiency_axis.set_title("Year-2 gain retained from year-1 cutoff")
    efficiency_axis.set_xlabel("Year-2 gain / year-2 optimal gain")
    efficiency_axis.set_ylabel("Percent of year-pairs")
    efficiency_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    efficiency_axis.grid(alpha=0.2)
    efficiency_axis.text(
        0.98,
        0.96,
        "\n".join(
            [
                f"mean={np.mean(efficiency):.3f}",
                f"median={np.median(efficiency):.3f}",
                f"retains >=50%: {(efficiency >= 0.5).mean():.1%}",
                f"retains >=100%: {(efficiency >= 1.0).mean():.1%}",
                f"negative gains: {(yearly_transfer_metrics['year_2_gain_using_year_1_threshold'] < 0).sum()}",
            ]
        ),
        transform=efficiency_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.82},
    )

    regret = (
        yearly_transfer_metrics["year_2_regret_vs_optimal"]
        .dropna()
        .to_numpy(dtype=float)
    )
    regret_axis.hist(
        regret,
        bins=np.linspace(0.0, max(0.8, np.percentile(regret, 99.5)), 30),
        color="#f58518",
        alpha=0.75,
        edgecolor="white",
        weights=_percent_weights(len(regret)),
    )
    regret_axis.axvline(np.median(regret), color="black", linestyle="--", linewidth=1.0)
    regret_axis.set_title("Year-2 regret vs year-2 optimal cutoff")
    regret_axis.set_xlabel("Optimal gain minus transferred gain")
    regret_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    regret_axis.grid(alpha=0.2)

    abs_drop_change = (
        yearly_transfer_metrics["optimal_drop_share_year_2"]
        - yearly_transfer_metrics["optimal_drop_share_year_1"]
    ).abs()
    drop_axis.hist(
        abs_drop_change.to_numpy(dtype=float) * 100,
        bins=np.linspace(0, 60, 31),
        color="#54a24b",
        alpha=0.75,
        edgecolor="white",
        weights=_percent_weights(len(abs_drop_change)),
    )
    drop_axis.axvline(
        abs_drop_change.median() * 100,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    drop_axis.set_title("Absolute change in optimal drop share")
    drop_axis.set_xlabel("Percentage-point change")
    drop_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    drop_axis.grid(alpha=0.2)

    figure.suptitle(
        "Year-to-year transferability of reviewer-specific Goodreads cutoffs"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
