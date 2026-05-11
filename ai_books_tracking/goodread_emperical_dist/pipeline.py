from __future__ import annotations

from datetime import datetime
import math
import pandas as pd

from ai_books_tracking.goodread_emperical_dist.config import (
    ANALYSIS_PROFILE_MANIFEST_CSV,
    AI_ACTIONS_DIR,
    BOOK_OVERLAP_SUMMARY_PNG,
    CUTOFF_SWEEP_PERCENTILES_ALL_BOOKS_CSV,
    CUTOFF_SWEEP_PERCENTILES_CSV,
    CUTOFF_SWEEP_PROFILE_ALL_BOOKS_CSV,
    CUTOFF_SWEEP_PROFILE_CSV,
    DROP_CURVES_CSV,
    FIXED_RULE_PROFILE_AUDIT_MD,
    FIXED_RULE_PARAMETER_COMPARISON_HEATMAP_PNG,
    FIXED_RULE_PARAMETER_COMPARISON_PROFILE_CSV,
    FIXED_RULE_PARAMETER_COMPARISON_SUMMARY_CSV,
    FIXED_RULE_PROFILE_METRICS_CSV,
    FIXED_RULE_RATING_SHARE_CORR_BANDS_PNG,
    FIXED_RULE_RATING_SHARE_GROUPED_ROWS_PNG,
    FIXED_RULE_RATING_SHARE_PANEL_GRID_PNG,
    FIXED_RULE_RATING_SHARE_VIOLINS_PNG,
    FIXED_RULE_RATING_SHARE_VIOLINS_MEAN_TRIM_PNG,
    FIXED_RULE_RATING_SHARE_VIOLINS_STD_TRIM_PNG,
    FIXED_RULE_RATING_SHARES_CSV,
    FIXED_RULE_SUMMARY_CSV,
    GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_ALL_BOOKS_PNG,
    GOODREADS_BOOK_RATING_COUNTS_CSV,
    GOODREADS_RATING_COUNT_ACCURACY_PNG,
    GOODREADS_RATING_COUNT_BIN_SUMMARY_CSV,
    GOODREADS_RATING_COUNT_BOOK_METRICS_CSV,
    GOODREADS_SMALL_MULTIPLES_PNG,
    GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_PNG,
    GOODREADS_CUTOFF_PERCENTILE_PNG,
    GOODREADS_CUTOFF_PERCENTILE_VIOLIN_PNG,
    HOLDOUT_CORRELATIONS_CSV,
    HOLDOUT_CORRELATIONS_PNG,
    MODEL_METRICS_CSV,
    NETWORK_DISCOVERED_PROFILES_CSV,
    NETWORK_VALIDATED_PROFILES_CSV,
    OVERLAP_BOOKS_CSV,
    OVERLAP_BOOKS_GT2_REVIEWERS_TXT,
    OVERLAP_RATER_COUNT_SUMMARY_CSV,
    POLICY_GAIN_HISTOGRAMS_PNG,
    POLICY_HOLDOUT_RESULTS_CSV,
    POLICY_THRESHOLD_SUMMARY_CSV,
    PREPARED_PROFILE_BOOKS_CSV,
    PROFILE_FEEDS_DIR,
    PROFILE_SUMMARY_CSV,
    RAW_PROFILE_BOOKS_CSV,
    REVIEWER_PAIR_CORRELATIONS_CSV,
    REVIEWER_VOLUME_DISTRIBUTIONS_PNG,
    REVIEWER_VOLUME_SUMMARY_CSV,
    REVIEWER_YEAR_COUNTS_CSV,
    RESULTS_SO_FAR_MD,
    RESULTS_SO_FAR_EVERYTHING_TXT,
    SMALL_MULTIPLES_SELECTION_CSV,
    TEMPORAL_PICKING_PROFILE_CSV,
    TEMPORAL_PICKING_SUMMARY_CSV,
    TRAINING_MODEL_SELECTION_CSV,
    YEARLY_CONSISTENCY_SUMMARY_CSV,
    YEARLY_CUTOFF_TRANSFER_SUMMARY_PNG,
    YEARLY_CORRELATION_CONSISTENCY_PNG,
    YEARLY_CORRELATION_PAIRS_CSV,
    YEARLY_GAIN_DIFFERENCE_HISTOGRAMS_PNG,
    YEARLY_GAIN_DIFFERENCES_CSV,
    YEARLY_ACTIVITY_SUMMARY_CSV,
    YEARLY_PROFILE_METRICS_CSV,
    YEARLY_TRANSFER_METRICS_CSV,
    ensure_directories,
)
from ai_books_tracking.goodread_emperical_dist.cross_profile_analysis import (
    compute_goodreads_rating_count_accuracy,
    compute_book_overlap_metrics,
    compute_reviewer_volume_metrics,
)
from ai_books_tracking.goodread_emperical_dist.evaluation import (
    evaluate_profile,
    profile_has_sufficient_rating_variation,
)
from ai_books_tracking.goodread_emperical_dist.feature_engineering import (
    prepare_profile_books,
)
from ai_books_tracking.goodread_emperical_dist.fixed_rule_analysis import (
    FIXED_RULE_LABELS,
    assign_fixed_rule_correlation_bands,
    compute_fixed_rule_parameter_comparison,
    filter_fixed_rule_profiles_by_percentile_window,
    compute_fixed_rule_outputs,
    build_fixed_rule_audit_report,
)
from ai_books_tracking.goodread_emperical_dist.goodreads_rss import fetch_profile_books
from ai_books_tracking.goodread_emperical_dist.plotting import (
    plot_fixed_rule_rating_share_grid_by_correlation,
    plot_fixed_rule_parameter_comparison_heatmaps,
    plot_fixed_rule_rating_share_grouped_rows,
    plot_fixed_rule_rating_share_panel_grid,
    plot_fixed_rule_rating_share_violins,
    plot_goodreads_cutoff_gain_and_drop_violins,
    plot_goodreads_cutoff_percentile_and_violin,
    plot_goodreads_cutoff_percentile_tradeoff,
    plot_goodreads_rating_count_accuracy,
    plot_goodreads_small_multiples,
    plot_book_overlap_summary,
    plot_holdout_correlation_histograms,
    plot_policy_gain_histograms,
    plot_reviewer_volume_distributions,
    plot_yearly_correlation_consistency,
    plot_yearly_cutoff_transfer_summary,
    plot_yearly_gain_difference_histograms,
)
from ai_books_tracking.goodread_emperical_dist.policy_analysis import (
    POLICY_DROP_FRACTIONS,
    POLICY_LABELS,
    POLICY_NAME_ORDER,
    build_goodreads_cutoff_sweep,
    run_policy_analysis,
)
from ai_books_tracking.goodread_emperical_dist.profile_network import (
    crawl_public_profile_network,
    expand_network_until_target_passes,
    validate_network_candidates,
)
from ai_books_tracking.goodread_emperical_dist.profiles import (
    profiles_from_frame,
    write_analysis_manifest,
)
from ai_books_tracking.goodread_emperical_dist.temporal_analysis import (
    compute_temporal_picking_profile_metrics,
    summarize_temporal_picking,
)
from ai_books_tracking.goodread_emperical_dist.yearly_consistency import (
    compute_yearly_consistency_outputs,
)


def _safe_float(value: object) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.3f}"


def _write_results_everything(
    profile_summary: pd.DataFrame,
    cutoff_sweep_percentiles: pd.DataFrame,
    fixed_rule_summary: pd.DataFrame,
    fixed_rule_parameter_comparison_summary: pd.DataFrame,
    network_validated_profiles: pd.DataFrame,
    temporal_picking_summary: pd.DataFrame,
    overlap_rater_count_summary: pd.DataFrame,
    goodreads_rating_count_bin_summary: pd.DataFrame,
    yearly_consistency_summary: pd.DataFrame,
) -> None:
    pass_count = (
        int(network_validated_profiles["passes_network_filter"].fillna(False).sum())
        if not network_validated_profiles.empty
        else 0
    )
    profile_count = int(profile_summary["split_type"].ne("not_evaluated").sum())
    lines = [
        "Goodreads empirical distribution: plot guide and rerun notes",
        "",
        "This file explains every generated plot in this subfolder, what it does and does not mean, and how to rerun the pipeline on newer data.",
        "",
        "Current analysis pool",
        f"- Evaluated profiles: {profile_count}",
        f"- Passing public-network candidates discovered so far: {pass_count}",
        "",
        f"Plot: {FIXED_RULE_RATING_SHARE_VIOLINS_PNG.name}",
        "- What it shows: three side-by-side violin subplots for the distribution of each reader's 1-to-5 star mix under the original books, a fixed Goodreads >= 4.0 rule, and a personalized 'drop the lower Goodreads half' rule.",
        "- Main interpretation: this is the cleanest way to see how those two fixed rules reshape the kinds of ratings readers would end up with, not just their average gain.",
        "- Watch out for: the y-axis is each reader's within-reader percentage of kept books at that star level, so the figure gives each reader equal weight rather than weighting by book count.",
        "",
        f"Plot: {FIXED_RULE_RATING_SHARE_VIOLINS_MEAN_TRIM_PNG.name}",
        "- What it shows: the same three fixed-rule violin panels, but restricted to readers whose average rating is between the 10th and 90th percentiles.",
        "- Main interpretation: this removes extreme easy-graders and harsh-graders so the central panel shape is easier to read.",
        "- Watch out for: this is a descriptive trim, not a correctness filter; extreme raters may still be perfectly real.",
        "",
        f"Plot: {FIXED_RULE_RATING_SHARE_VIOLINS_STD_TRIM_PNG.name}",
        "- What it shows: the same three fixed-rule violin panels, but restricted to readers whose rating standard deviation is between the 10th and 90th percentiles.",
        "- Main interpretation: this removes both very flat raters and extremely volatile raters to show the middle spread more clearly.",
        "- Watch out for: because it trims on personal-rating variance, it changes which profiles dominate the tails even when Goodreads fit is unchanged.",
        "",
        f"Plot: {FIXED_RULE_RATING_SHARE_CORR_BANDS_PNG.name}",
        "- What it shows: a 4x3 grid of the same fixed-rule rating-share violins, split by Goodreads correlation bands: 10th-25th, 25th-50th, 50th-75th, and 75th-90th percentiles.",
        "- Main interpretation: this shows how the shape of the kept-book rating distribution changes as Goodreads-personal alignment improves.",
        "- Watch out for: the bottom and top 10% of correlation are excluded to reduce dominance by the most pathological and most degenerate cases.",
        "",
        f"Plot: {FIXED_RULE_RATING_SHARE_GROUPED_ROWS_PNG.name}",
        "- What it shows: three stacked grouped-violin rows, for the full panel, the mean-trimmed panel, and the SD-trimmed panel. Within each 1-5 rating bucket, the three adjacent violins are original books, Goodreads >= 4.0, and the lower-half rule.",
        "- Main interpretation: this is the easiest direct shape-comparison view because the three rule distributions sit next to each other inside each rating bucket.",
        "- Watch out for: grouped violins emphasize shape comparison, but the medians for each rule are still easier to read in the panel-grid version.",
        "",
        f"Plot: {FIXED_RULE_RATING_SHARE_PANEL_GRID_PNG.name}",
        "- What it shows: a 3x3 grid combining the full-panel, mean-trimmed, and SD-trimmed fixed-rule violin panels into one figure.",
        "- Main interpretation: this lets you compare how the three existing panel-style figures change after trimming extreme mean raters or extreme-variance raters.",
        "- Watch out for: rows use different profile subsets, so differences across rows can reflect composition changes as well as better readability.",
        "",
        f"Plot: {FIXED_RULE_PARAMETER_COMPARISON_HEATMAP_PNG.name}",
        "- What it shows: four side-by-side heatmaps comparing `keep Goodreads >= cutoff` against `drop the bottom x% by Goodreads score`, using all rated books. The color panels show the share of readers helped by the cutoff, the mean gain difference, and the 25th and 75th percentile gain differences.",
        "- Main interpretation: this is the parameter-sensitivity map for the two simple Goodreads rules. Positive gain differences mean the fixed cutoff rule beats the drop-bottom-x% rule for that cell.",
        "- Watch out for: the x-axis is a literal Goodreads cutoff, while the y-axis is a target bottom-share to drop by Goodreads score, so the two axes are not the same kind of rule.",
        "",
        f"Plot: {POLICY_GAIN_HISTOGRAMS_PNG.name}",
        "- What it shows: four histogram panels for holdout rating gain at target drops of 20%, 40%, 60%, and 80%, split by policy.",
        "- Main interpretation: this compares policy-level gains across readers. Wider spread means the same policy helps some readers much more than others.",
        "- Watch out for: the cutoff policies target those drop rates on training data, so their realized holdout drop share is only approximate.",
        "",
        f"Plot: {HOLDOUT_CORRELATIONS_PNG.name}",
        "- What it shows: the distribution of holdout rank correlation coefficients across readers for raw Goodreads and the training-selected fitted model.",
        "- Main interpretation: higher correlation means the scoring rule orders books more like the reader's own ratings, but high correlation alone does not guarantee large screening gains.",
        "- Watch out for: gains also depend on how many candidate books sit below the screening threshold.",
        "",
        f"Plot: {GOODREADS_SMALL_MULTIPLES_PNG.name}",
        "- What it shows: nineteen same-reader scatterplots of Goodreads average rating versus that reader's own rating.",
        "- Main interpretation: upward tilt suggests Goodreads agreement with the reader; vertical spread at a fixed Goodreads level shows where Goodreads is too coarse for that person.",
        "- Watch out for: each panel is one reader, but the figure as a whole is still a selected subset biased toward temporally credible histories.",
        "",
        f"Plot: {GOODREADS_CUTOFF_PERCENTILE_PNG.name}",
        "- What it shows: across-reader percentile lines for all-books gain as a fixed Goodreads cutoff moves from low to high.",
        "- Main interpretation: this is a distribution summary of what people could expect if they followed a fixed Goodreads cutoff rule.",
        "- Watch out for: the grey dotted line now shows median share dropped, because percentile-matched gain and drop labels were misleading when they came from different readers.",
        "",
        f"Plot: {GOODREADS_CUTOFF_PERCENTILE_VIOLIN_PNG.name}",
        "- What it shows: the same cutoff grid, but with the full per-reader gain distribution shown as violin plots under the percentile-line panel on shared axes.",
        "- Main interpretation: this is the easiest plot for seeing both the tail behavior and the dense middle of the reader distribution at each cutoff.",
        "- Watch out for: the violin width is density, not reader count after weighting.",
        "",
        f"Plot: {GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_PNG.name}",
        "- What it shows: three stacked panels, with holdout rating gain violins on top, dropped-share violins in the middle, and a violin of per-reader average kept holdout rating on the 1 to 5 scale at the bottom.",
        "- Main interpretation: this is the most direct way to see the tradeoff between gain, how much reading volume gets filtered out, and what average rating level readers would end up with after following the cutoff.",
        "- Watch out for: all three panels are built from same-reader holdout evaluation, not full-history in-sample filtering.",
        "- Watch out for: the bottom holdout violin can have longer tails than the all-books version because it summarizes per-reader means from much smaller kept sets, often only a few books at high cutoffs.",
        "",
        f"Plot: {GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_ALL_BOOKS_PNG.name}",
        "- What it shows: the same three stacked violin panels, but using each reader's full rated history instead of only the held-out books.",
        "- Main interpretation: this is the in-sample version of the cutoff tradeoff. It is useful for intuition, but it is less trustworthy than the holdout version for causal or predictive claims.",
        "- Watch out for: there is no training step for the cutoff itself, but this is still in-sample because the same books define and evaluate the rule.",
        "",
        f"Plot: {REVIEWER_VOLUME_DISTRIBUTIONS_PNG.name}",
        "- What it shows: total public Goodreads ratings per reviewer, the distribution of books rated in an active reviewer-year, and a dual-axis timeline of active reviewers and total reviews by year.",
        "- Main interpretation: this separates the lifetime panel shape from the annual activity shape and also shows how much of the panel sits in later years.",
        "- Watch out for: the reviewer-year panel counts only years with dated books in the cleaned data, so the leftmost bin is low-volume active years rather than literal zero-review years.",
        "",
        f"Plot: {BOOK_OVERLAP_SUMMARY_PNG.name}",
        "- What it shows: how within-book rating dispersion and average pairwise reviewer correlation change as more people in the panel rate the same book, with weighted smoothing lines over the bucket means.",
        "- Main interpretation: books with higher overlap let you see both disagreement on the book itself and whether the people who converge on it are generally taste-similar.",
        "- Watch out for: pairwise reviewer correlations are computed over full shared histories and only when a reviewer pair has at least 5 shared books.",
        "",
        f"Plot: {GOODREADS_RATING_COUNT_ACCURACY_PNG.name}",
        "- What it shows: books are binned by total Goodreads rating count, then Goodreads average rating is compared against this panel's individual ratings and book-level average ratings inside each bin.",
        "- Error bars: intervals are bootstrapped by resampling books within each bin. The wider translucent bars are central 95% intervals, and the narrower capped bars are central 80% intervals.",
        "- Main interpretation: this answers a different question from `book_overlap_summary.png`. It asks whether the public Goodreads average is more accurate for this sample as total Goodreads popularity/coverage rises, not whether books with more raters inside this panel attract more similar people.",
        "- Main interpretation: total Goodreads rating count is not showing a robust actionable improvement for personal prediction. At most, it is a weak book-consensus/data-quality flag that needs controls.",
        f"- Watch out for: this plot only includes books with a known total Goodreads rating count in `{GOODREADS_BOOK_RATING_COUNTS_CSV.name}`.",
        "- Watch out for: total Goodreads count can be confounded with panel overlap, genre, age/canon status, and cache/fetch coverage.",
        "",
        f"Plot: {YEARLY_GAIN_DIFFERENCE_HISTOGRAMS_PNG.name}",
        "- What it shows: four histograms of year-2 minus year-1 gain for fixed Goodreads screening rates of 20%, 40%, 60%, and 80%.",
        "- Main interpretation: these show how unstable the gain from a fixed screening intensity can be from one year to the next within the same reviewer.",
        "- Watch out for: only adjacent year pairs with at least 30 books in each year are included.",
        "",
        f"Plot: {YEARLY_CORRELATION_CONSISTENCY_PNG.name}",
        "- What it shows: year-1 versus year-2 Goodreads correlation scatterplots, plus delta histograms, for Spearman and Pearson.",
        "- Main interpretation: this is the direct view of whether Goodreads alignment is stable within a reviewer over time.",
        "- Watch out for: correlation stability and gain stability are related but not identical; exposure to low-Goodreads books still matters.",
        "",
        f"Plot: {YEARLY_CUTOFF_TRANSFER_SUMMARY_PNG.name}",
        "- What it shows: the distribution of year-2 gain retained when reusing year-1's optimal cutoff, plus regret and drop-share-change distributions.",
        "- Main interpretation: this is the direct transferability plot for reviewer-specific Goodreads cutoffs across adjacent years.",
        "- Label note: `retains >=50%` and `retains >=100%` refer to the ratio `year-2 transferred gain / year-2 optimal gain`, not a direct comparison of year-2 gain versus year-1 gain.",
        "- Watch out for: the gain-retention ratio can exceed 1 when year 1's cutoff happens to beat year 2's own in-sample optimum because of discrete ties.",
        "",
        "Temporal interpretation",
        "- The temporal drift tables compare early versus late reading within the same reader history.",
    ]
    if not temporal_picking_summary.empty:
        credible = temporal_picking_summary[
            temporal_picking_summary["scope_name"] == "temporally_credible_profiles"
        ]
        own_delta = credible[credible["metric_name"] == "user_rating_late_minus_early"]
        gr_delta = credible[
            credible["metric_name"] == "goodreads_rating_late_minus_early"
        ]
        if not own_delta.empty and not gr_delta.empty:
            lines.extend(
                [
                    f"- In the current run, later books rate {own_delta['mean_value'].iloc[0]:.3f} stars higher on readers' own ratings on average.",
                    f"- Over the same comparison, the Goodreads average rating of chosen books rises by {gr_delta['mean_value'].iloc[0]:.3f} on average.",
                ]
            )
    if not cutoff_sweep_percentiles.empty:
        lines.extend(
            [
                "",
                "How to read the cutoff plots",
                "- X-axis is the Goodreads average-rating cutoff you would use before deciding to read.",
                "- Y-axis is average gain in the reader's own rating among the books kept after screening.",
                "- These plots now summarize all-books outcomes across readers, not holdout-only outcomes.",
                "- The raw Goodreads cutoff plots do not fit the cutoff on training data; they are descriptive distributions for the fixed Goodreads rule.",
            ]
        )
    if not fixed_rule_summary.empty:
        fixed_rule_rows = fixed_rule_summary[
            fixed_rule_summary["rule_name"].isin(
                ["goodreads_cutoff_4_0", "drop_lower_goodreads_half"]
            )
        ]
        if len(fixed_rule_rows) == 2:
            better_row = fixed_rule_summary[
                fixed_rule_summary["rule_name"] == "comparison"
            ]
            lines.extend(
                [
                    "",
                    "Fixed-rule comparison",
                    *[
                        f"- {row.rule_label}: median gain `{row.median_gain:.3f}`, median drop `{row.median_drop_share:.0%}`, positive-gain share `{row.positive_gain_share:.0%}`."
                        for row in fixed_rule_rows.itertuples(index=False)
                    ],
                ]
            )
            if not better_row.empty:
                comparison = better_row.iloc[0]
                lines.append(
                    f"- Goodreads >= 4.0 beats the lower-half rule for `{comparison['positive_gain_share']:.0%}` of readers; the lower-half rule wins for `{comparison['negative_gain_share']:.0%}`."
                )
    if not overlap_rater_count_summary.empty:
        max_overlap = int(overlap_rater_count_summary["n_raters"].max())
        lines.extend(
            [
                "",
                "Overlap interpretation",
                "- `overlap_books.csv` lists every book rated by at least 2 people in the panel, including overlap count, within-book rating SD, and average pairwise reviewer correlation.",
                f"- `{OVERLAP_BOOKS_GT2_REVIEWERS_TXT.name}` is the plain-text export of books with more than 2 raters, sorted by overlap count.",
                f"- In the current run, the maximum observed overlap is `{max_overlap}` raters on the same book.",
            ]
        )
    if not goodreads_rating_count_bin_summary.empty:
        first_bin = goodreads_rating_count_bin_summary.iloc[0]
        last_bin = goodreads_rating_count_bin_summary.iloc[-1]
        lines.extend(
            [
                "",
                "Goodreads rating-count interpretation",
                f"- Lowest count bin: median `{first_bin['rating_count_median']:.0f}` total Goodreads ratings, row Spearman `{_safe_float(first_bin['row_spearman_rho'])}`, book-mean Spearman `{_safe_float(first_bin['book_mean_spearman_rho'])}`, book-mean absolute error `{_safe_float(first_bin['mean_book_absolute_error'])}` stars.",
                f"- Highest count bin: median `{last_bin['rating_count_median']:.0f}` total Goodreads ratings, row Spearman `{_safe_float(last_bin['row_spearman_rho'])}`, book-mean Spearman `{_safe_float(last_bin['book_mean_spearman_rho'])}`, book-mean absolute error `{_safe_float(last_bin['mean_book_absolute_error'])}` stars.",
                "- The p80/p95 bootstrap bars should be read as uncertainty around each bin's sampled books, not as proof of a smooth monotone trend.",
                "- Limited lesson: total Goodreads rating count does not matter much for personal prediction once a book has more than a few thousand ratings. The observed book-mean gains are not stable enough, and are too confounded, to use as a screening rule.",
                "- Before trusting this as a true count effect, rerun with fixed panel overlap, per-reader holdouts, matched low/high-count books, controlled regressions, stratified random count enrichment, and within-bin count permutations.",
            ]
        )
    if not yearly_consistency_summary.empty:
        efficiency = yearly_consistency_summary[
            yearly_consistency_summary["metric_name"] == "year_2_optimal_efficiency"
        ]
        if not efficiency.empty:
            lines.extend(
                [
                    "",
                    "Year-to-year stability interpretation",
                    f"- Applying a reviewer's year-1 optimal Goodreads cutoff to year 2 retains a median `{efficiency['median_value'].iloc[0]:.3f}` of the year-2 optimal gain across adjacent 30-book years.",
                ]
            )
    lines.extend(
        [
            "",
            "How to rerun on cached data",
            "- Seed-only refresh of plots and summaries:",
            "  python -m ai_books_tracking.goodread_emperical_dist.run_pipeline",
            "- Expanded run using the discovered network candidates in analysis, capped at roughly 100 RSS books per profile:",
            "  python -m ai_books_tracking.goodread_emperical_dist.run_pipeline --run-network-expansion --include-network-profiles-in-analysis --analysis-target-total 200 --network-target-pass-count 200 --network-max-depth 2 --network-max-profiles 200 --network-max-profiles-cap 2000 --network-profile-step-size 200 --max-pages 5 --sleep-seconds 0.1 --network-validation-sleep-seconds 0.2",
            "- Slower scraping run when you want to be gentler with Goodreads rate limits:",
            "  python -m ai_books_tracking.goodread_emperical_dist.run_pipeline --run-network-expansion --include-network-profiles-in-analysis --analysis-target-total 200 --network-target-pass-count 200 --network-max-depth 2 --network-max-profiles 200 --network-max-profiles-cap 2000 --network-profile-step-size 200 --max-pages 5 --sleep-seconds 0.6 --network-validation-sleep-seconds 0.3",
            "",
            "Files to inspect when debugging",
            f"- {ANALYSIS_PROFILE_MANIFEST_CSV.name}: exact set of profiles promoted into the analysis pool",
            f"- {PROFILE_SUMMARY_CSV.name}: split quality and evaluation eligibility by reader",
            f"- {FIXED_RULE_PROFILE_METRICS_CSV.name}: per-reader gains and drop shares for the 4.0 and lower-half rules",
            f"- {FIXED_RULE_SUMMARY_CSV.name}: cross-reader summaries for the fixed-rule comparison",
            f"- {FIXED_RULE_PROFILE_AUDIT_MD.name}: manually reviewable profile-level explanations and examples for winners, losers, and outliers",
            f"- {CUTOFF_SWEEP_PROFILE_ALL_BOOKS_CSV.name}: per-reader all-books gains at each fixed Goodreads cutoff",
            f"- {NETWORK_VALIDATED_PROFILES_CSV.name}: discovered profile candidates and pass/fail fields",
            f"- {TEMPORAL_PICKING_PROFILE_CSV.name}: per-reader temporal drift metrics",
            f"- {OVERLAP_BOOKS_CSV.name}: overlapping books and their overlap counts",
            f"- {GOODREADS_BOOK_RATING_COUNTS_CSV.name}: total Goodreads rating-count cache by `book_id`",
            f"- {GOODREADS_RATING_COUNT_BIN_SUMMARY_CSV.name}: popularity-bin accuracy for Goodreads average ratings",
            f"- {GOODREADS_RATING_COUNT_BOOK_METRICS_CSV.name}: book-level panel means and errors used by the count-bin plot",
            f"- {YEARLY_TRANSFER_METRICS_CSV.name}: adjacent-year cutoff transfer results",
        ]
    )
    RESULTS_SO_FAR_EVERYTHING_TXT.write_text("\n".join(lines) + "\n")


def _results_markdown(
    profile_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    drop_curves: pd.DataFrame,
    policy_holdout_results: pd.DataFrame,
    holdout_correlations: pd.DataFrame,
    small_multiples_selection: pd.DataFrame,
    cutoff_sweep_percentiles: pd.DataFrame,
    fixed_rule_summary: pd.DataFrame,
    fixed_rule_parameter_comparison_summary: pd.DataFrame,
    temporal_picking_summary: pd.DataFrame,
    overlap_rater_count_summary: pd.DataFrame,
    goodreads_rating_count_bin_summary: pd.DataFrame,
    yearly_consistency_summary: pd.DataFrame,
    network_validated_profiles: pd.DataFrame,
) -> str:
    lines = [
        "# Goodreads Empirical Distribution Results",
        "",
        "This folder is the isolated cross-user Goodreads evaluation pipeline.",
        "",
        "Current pass is intentionally phase 1:",
        "- targets are public Goodreads star ratings only",
        "- consensus signal is Goodreads average rating from the RSS feed",
        "- trained models are retrained per profile on that profile's own history",
        "- category features are inferred from Goodreads RSS title/shelves/description only",
        "- total Goodreads rating counts are now partially covered for the shared-book analysis; Amazon/Open Library enrichment is still not included in this empirical pipeline",
        "",
    ]
    if profile_summary.empty:
        lines.extend(
            [
                "No eligible profiles were evaluated yet.",
                "",
                "Run `python -m ai_books_tracking.goodread_emperical_dist.run_pipeline` to populate results.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"Profiles fetched: `{len(profile_summary)}`",
            f"Profiles evaluated: `{int(profile_summary['split_type'].ne('not_evaluated').sum())}`",
            f"Total rated books after de-dupe: `{int(profile_summary['n_rated_books'].sum())}`",
            "",
            "## Median Holdout Rating Gain",
            "",
            "| Model | Drop 20% | Drop 50% | Drop 80% | Median rho | Median MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    degenerate_count = int(
        profile_summary["holdout_label"].eq("degenerate_ratings").sum()
    )
    if degenerate_count > 0:
        lines.extend(
            [
                f"- `{degenerate_count}` profiles with fewer than 3 distinct personal ratings were excluded from gain and correlation summaries as degenerate cases.",
                "",
            ]
        )

    model_order = [
        "baseline_mean",
        "goodreads_raw",
        "goodreads_linear",
        "goodreads_category_ridge",
    ]
    for model_name in model_order:
        metric_subset = model_metrics[model_metrics["model_name"] == model_name]
        curve_subset = drop_curves[drop_curves["model_name"] == model_name]
        gains = {
            fraction: curve_subset[curve_subset["drop_fraction"] == fraction][
                "rating_gain"
            ].median()
            for fraction in [0.2, 0.5, 0.8]
        }
        lines.append(
            "| "
            f"{model_name} | {_safe_float(gains[0.2])} | {_safe_float(gains[0.5])} "
            f"| {_safe_float(gains[0.8])} | {_safe_float(metric_subset['spearman_rho'].median())} "
            f"| {_safe_float(metric_subset['mae'].median())} |"
        )

    holdout_years = pd.to_numeric(profile_summary["holdout_label"], errors="coerce")
    recency_gap = profile_summary["latest_event_year"] - holdout_years
    recent_holdouts = int((recency_gap <= 1).sum()) if recency_gap.notna().any() else 0
    category_rho = model_metrics[
        model_metrics["model_name"] == "goodreads_category_ridge"
    ]["spearman_rho"].median()
    raw_rho = model_metrics[model_metrics["model_name"] == "goodreads_raw"][
        "spearman_rho"
    ].median()
    category_drop50 = drop_curves[
        (drop_curves["model_name"] == "goodreads_category_ridge")
        & (drop_curves["drop_fraction"] == 0.5)
    ]["rating_gain"].median()
    raw_drop50 = drop_curves[
        (drop_curves["model_name"] == "goodreads_raw")
        & (drop_curves["drop_fraction"] == 0.5)
    ]["rating_gain"].median()
    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            f"- Raw Goodreads alone already shows a median holdout gain of `{_safe_float(raw_drop50)}` stars at drop-50% across these profiles.",
            f"- The current RSS-only category model lowers median rank correlation (`{_safe_float(raw_rho)}` -> `{_safe_float(category_rho)}`) and lowers median drop-50 gain (`{_safe_float(raw_drop50)}` -> `{_safe_float(category_drop50)}`), so the category heuristic is not trustworthy yet.",
            f'- Only `{recent_holdouts}` of `{len(profile_summary)}` evaluated profiles had a holdout year within one year of their latest observed rating activity. This phase-1 pass is therefore stronger as "does any real cross-user signal exist?" than as "what would have happened last year?"',
            "",
        ]
    )

    if not policy_holdout_results.empty:
        lines.extend(
            [
                "",
                "## Holdout Screening Policy Comparison",
                "",
                "| Policy | Drop 20% | Drop 40% | Drop 60% | Drop 80% |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for policy_name in POLICY_NAME_ORDER:
            subset = policy_holdout_results[
                policy_holdout_results["policy_name"] == policy_name
            ]
            gains = {
                fraction: subset[subset["target_drop_fraction"] == fraction][
                    "rating_gain"
                ].median()
                for fraction in POLICY_DROP_FRACTIONS
            }
            lines.append(
                "| "
                f"{POLICY_LABELS[policy_name]} | {_safe_float(gains[0.2])} "
                f"| {_safe_float(gains[0.4])} | {_safe_float(gains[0.6])} "
                f"| {_safe_float(gains[0.8])} |"
            )

        raw_rho = holdout_correlations[
            holdout_correlations["series_name"] == "goodreads_raw"
        ]["spearman_rho"].median()
        fitted_rho = holdout_correlations[
            holdout_correlations["series_name"] == "training_selected_model"
        ]["spearman_rho"].median()
        selected_count = len(small_multiples_selection)
        lines.extend(
            [
                "",
                "- Cutoff policies above are trained to target 20/40/60/80% drops on the training set; their actual holdout drop shares vary by profile because a fixed cutoff is being applied out of sample.",
                f"- Median holdout Spearman for raw Goodreads is `{_safe_float(raw_rho)}`; for the training-selected fitted model it is `{_safe_float(fitted_rho)}`.",
                f"- The small-multiples figure uses `{selected_count}` profiles that first prefer user-supplied and expanded-verified readers, then apply the temporal-quality ranking.",
                f"- Plot outputs: `{POLICY_GAIN_HISTOGRAMS_PNG.name}`, `{HOLDOUT_CORRELATIONS_PNG.name}`, `{GOODREADS_SMALL_MULTIPLES_PNG.name}`, `{GOODREADS_CUTOFF_PERCENTILE_VIOLIN_PNG.name}`, `{GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_PNG.name}`, `{GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_ALL_BOOKS_PNG.name}`.",
                "",
            ]
        )

    if not cutoff_sweep_percentiles.empty:
        threshold_40 = cutoff_sweep_percentiles.loc[
            cutoff_sweep_percentiles["drop_share_median"].sub(0.4).abs().idxmin()
        ]
        lines.extend(
            [
                "",
                "## Cross-User Goodreads Cutoff Sweep",
                "",
                f"- `{GOODREADS_CUTOFF_PERCENTILE_PNG.name}` plots the `15th/35th/65th/85th` percentile all-books gains across people as the raw Goodreads cutoff moves.",
                "- The grey dotted line on that plot shows the median share of all rated books dropped at each cutoff.",
                f"- `{GOODREADS_CUTOFF_PERCENTILE_VIOLIN_PNG.name}` combines the percentile lines with violin plots of the full gain distribution at each cutoff on the same axes.",
                f"- `{GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_PNG.name}` shows gain violins, drop-share violins, and a violin of per-reader average kept holdout rating at each cutoff.",
                f"- `{GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_ALL_BOOKS_PNG.name}` is the corresponding all-books in-sample version.",
                f"- Near a median drop share of `{threshold_40['drop_share_median']:.0%}`, the cross-user gain range runs from `{_safe_float(threshold_40['gain_p15'])}` at the 15th percentile to `{_safe_float(threshold_40['gain_p85'])}` at the 85th percentile.",
                "",
            ]
        )

    if not fixed_rule_summary.empty:
        rule_rows = fixed_rule_summary[
            fixed_rule_summary["rule_name"].isin(
                ["goodreads_cutoff_4_0", "drop_lower_goodreads_half"]
            )
        ].copy()
        comparison_row = fixed_rule_summary[
            fixed_rule_summary["rule_name"] == "comparison"
        ]
        if not rule_rows.empty:
            lines.extend(
                [
                    "",
                    "## Two Simple Fixed Rules",
                    "",
                    "| Rule | Mean gain | Median gain | Mean drop | Median drop | Positive gain share |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in rule_rows.itertuples(index=False):
                lines.append(
                    "| "
                    f"{row.rule_label} | {row.mean_gain:.3f} | {row.median_gain:.3f} "
                    f"| {row.mean_drop_share:.0%} | {row.median_drop_share:.0%} "
                    f"| {row.positive_gain_share:.0%} |"
                )
            if not comparison_row.empty:
                comparison = comparison_row.iloc[0]
                lines.extend(
                    [
                        "",
                        f"- `{FIXED_RULE_LABELS['goodreads_cutoff_4_0']}` beats `{FIXED_RULE_LABELS['drop_lower_goodreads_half']}` for `{comparison['positive_gain_share']:.0%}` of readers.",
                        f"- The lower-half rule wins for `{comparison['negative_gain_share']:.0%}` of readers.",
                        f"- Plot outputs: `{FIXED_RULE_RATING_SHARE_VIOLINS_PNG.name}`, `{FIXED_RULE_RATING_SHARE_VIOLINS_MEAN_TRIM_PNG.name}`, `{FIXED_RULE_RATING_SHARE_VIOLINS_STD_TRIM_PNG.name}`, `{FIXED_RULE_RATING_SHARE_CORR_BANDS_PNG.name}`, `{FIXED_RULE_RATING_SHARE_GROUPED_ROWS_PNG.name}`, `{FIXED_RULE_RATING_SHARE_PANEL_GRID_PNG.name}`, and `{FIXED_RULE_PARAMETER_COMPARISON_HEATMAP_PNG.name}`.",
                        f"- Profile-level interpretation and outlier review: `{FIXED_RULE_PROFILE_AUDIT_MD.name}`.",
                        "",
                    ]
                )
    if not fixed_rule_parameter_comparison_summary.empty:
        best_cutoff_cell = fixed_rule_parameter_comparison_summary.sort_values(
            [
                "benefit_share_cutoff_beats_target_drop",
                "mean_gain_difference_cutoff_minus_target_drop",
            ],
            ascending=[False, False],
        ).iloc[0]
        lines.extend(
            [
                f"- In the cutoff-vs-drop-share grid, the strongest region for the fixed cutoff rule is around cutoff `{best_cutoff_cell['cutoff_threshold']:.1f}` versus dropping the bottom `{best_cutoff_cell['target_drop_fraction']:.0%}`, where the cutoff wins for `{best_cutoff_cell['benefit_share_cutoff_beats_target_drop']:.0%}` of readers on average.",
                "",
            ]
        )

    if not temporal_picking_summary.empty:
        credible = temporal_picking_summary[
            temporal_picking_summary["scope_name"] == "temporally_credible_profiles"
        ]
        user_metric = credible[
            credible["metric_name"] == "user_rating_late_minus_early"
        ]
        goodreads_metric = credible[
            credible["metric_name"] == "goodreads_rating_late_minus_early"
        ]
        if not user_metric.empty and not goodreads_metric.empty:
            lines.extend(
                [
                    "",
                    "## Temporal Picking Drift",
                    "",
                    f"- Across temporally credible profiles, later books score `{_safe_float(user_metric['mean_value'].iloc[0])}` stars higher than early books on average, using last-quartile minus first-quartile personal ratings.",
                    f"- Over the same comparison, the Goodreads average rating of chosen books rises by `{_safe_float(goodreads_metric['mean_value'].iloc[0])}` on average.",
                    "",
                ]
            )

    if not overlap_rater_count_summary.empty:
        peak_overlap = overlap_rater_count_summary.sort_values(
            ["n_raters"], ascending=[False]
        ).iloc[0]
        lines.extend(
            [
                "",
                "## Shared-Book Overlap",
                "",
                f"- `{OVERLAP_BOOKS_CSV.name}` lists books rated by multiple people, including overlap count, within-book rating SD, and average pairwise reviewer correlation.",
                f"- `{OVERLAP_BOOKS_GT2_REVIEWERS_TXT.name}` is the plain-text export of books with more than 2 raters, sorted by overlap count.",
                f"- `{BOOK_OVERLAP_SUMMARY_PNG.name}` graphs mean pairwise reviewer correlation and mean within-book rating SD against overlap count.",
                f"- The highest overlap bucket currently observed is `{int(peak_overlap['n_raters'])}` raters on the same book.",
                "",
            ]
        )

    if not goodreads_rating_count_bin_summary.empty:
        first_bin = goodreads_rating_count_bin_summary.iloc[0]
        last_bin = goodreads_rating_count_bin_summary.iloc[-1]
        lines.extend(
            [
                "",
                "## Total Goodreads Rating Count",
                "",
                f"- `{GOODREADS_RATING_COUNT_ACCURACY_PNG.name}` bins books by total Goodreads rating count and compares Goodreads average rating against this panel's ratings inside each bin.",
                "- The plot includes book-resampling bootstrap intervals: wider translucent bars are central 95% intervals, and narrower capped bars are central 80% intervals.",
                f"- Coverage from `{GOODREADS_BOOK_RATING_COUNTS_CSV.name}`: `{int(goodreads_rating_count_bin_summary['n_books'].sum())}` books and `{int(goodreads_rating_count_bin_summary['n_sample_ratings'].sum())}` panel ratings.",
                f"- Lowest-count bin median `{first_bin['rating_count_median']:.0f}` total GR ratings: row Spearman `{_safe_float(first_bin['row_spearman_rho'])}`, book-mean Spearman `{_safe_float(first_bin['book_mean_spearman_rho'])}`, book-mean absolute error `{_safe_float(first_bin['mean_book_absolute_error'])}` stars.",
                f"- Highest-count bin median `{last_bin['rating_count_median']:.0f}` total GR ratings: row Spearman `{_safe_float(last_bin['row_spearman_rho'])}`, book-mean Spearman `{_safe_float(last_bin['book_mean_spearman_rho'])}`, book-mean absolute error `{_safe_float(last_bin['mean_book_absolute_error'])}` stars.",
                "- Error-bar read: after the first few thousand total Goodreads ratings, the intervals mostly overlap or move non-monotonically. Treat the line as basically flat/noisy for personal prediction.",
                "- Practical read: do not use total Goodreads rating count as a meaningful screening feature by itself. It is better treated as a metadata quality flag until fixed-overlap, matched-book, and per-reader holdout checks show otherwise.",
                "",
                "### Rating-Count Confounds And Validation Checks",
                "",
                "- Total Goodreads count can be confounded with within-panel overlap, category/genre, publication era, classroom/canon status, series effects, language/region, and cache/fetch coverage.",
                "- Rating counts are current counts, not counts at the time each panel member read the book. This can leak later popularity into older ratings.",
                "- Run fixed-overlap/downsampled versions of the plot, per-reader holdout binning, matched low/high-count book comparisons, controlled regressions with a Goodreads-average-by-log-count interaction, stratified random count enrichment, and within-bin permutation checks.",
                "",
            ]
        )

    if not yearly_consistency_summary.empty:
        efficiency = yearly_consistency_summary[
            yearly_consistency_summary["metric_name"] == "year_2_optimal_efficiency"
        ]
        regret = yearly_consistency_summary[
            yearly_consistency_summary["metric_name"] == "year_2_regret_vs_optimal"
        ]
        if not efficiency.empty and not regret.empty:
            lines.extend(
                [
                    "",
                    "## Year-To-Year Consistency",
                    "",
                    f"- Across adjacent reviewer-years with at least 30 books each, applying year 1's optimal Goodreads cutoff to year 2 retains a mean `{_safe_float(efficiency['mean_value'].iloc[0])}` of year 2's optimal gain.",
                    f"- The corresponding mean year-2 regret versus the year-2 optimum is `{_safe_float(regret['mean_value'].iloc[0])}` stars.",
                    "- In `yearly_cutoff_transfer_summary.png`, `retains >=50%` and `retains >=100%` refer to that retained-gain ratio, not to year-2 beating year-1 in absolute gain.",
                    f"- Plot outputs: `{YEARLY_GAIN_DIFFERENCE_HISTOGRAMS_PNG.name}`, `{YEARLY_CORRELATION_CONSISTENCY_PNG.name}`, and `{YEARLY_CUTOFF_TRANSFER_SUMMARY_PNG.name}`.",
                    "",
                ]
            )

    if not network_validated_profiles.empty:
        passed = int(network_validated_profiles["passes_network_filter"].sum())
        lines.extend(
            [
                "",
                "## Network Expansion",
                "",
                f"- Public-profile network crawling found `{len(network_validated_profiles)}` discoverable profiles and `{passed}` that pass the exploratory filter of `>=100` public ratings with `>=2` observed RSS years.",
                f"- `{ANALYSIS_PROFILE_MANIFEST_CSV.name}` records the exact subset promoted into the main evaluation pool.",
                f"- Network outputs: `{NETWORK_DISCOVERED_PROFILES_CSV.name}` and `{NETWORK_VALIDATED_PROFILES_CSV.name}`.",
                "",
            ]
        )

    best_drop50 = (
        drop_curves[drop_curves["drop_fraction"] == 0.5]
        .sort_values(["rating_gain", "profile_slug"], ascending=[False, True])
        .head(8)
    )
    if not best_drop50.empty:
        lines.extend(
            [
                "",
                "## Best Profile/Model Holdout Gains At Drop-50%",
                "",
                "| Profile | Model | Gain | Split | Holdout |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for row in best_drop50.itertuples(index=False):
            lines.append(
                "| "
                f"{row.display_name} | {row.model_name} | {row.rating_gain:.3f} "
                f"| {row.split_type} | {row.holdout_label} |"
            )

    lines.extend(
        [
            "",
            "## Split Notes",
            "",
            "- `calendar_year` means the most recent year with enough ratings was held out.",
            "- `chronological_tail` means the last 20% of dated books were held out because no recent-year block was large enough.",
            "- `dominant_year_share` in `profile_summary.csv` is a temporal-quality warning; values near `1.0` often indicate heavy backlog imports.",
            "",
            "## Next Improvements",
            "",
            "1. Expand Goodreads `rating_count` coverage beyond the currently cached/fetched `book_id` set.",
            "2. Add Amazon/Open Library enrichment and rerun the same evaluation harness.",
            "3. Replace the RSS-only category mapper with subject-based classification from external metadata.",
            "4. Add profile-level filtering rules for users with low temporal fidelity.",
        ]
    )
    return "\n".join(lines)


def _write_results(
    profile_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    drop_curves: pd.DataFrame,
    policy_holdout_results: pd.DataFrame,
    holdout_correlations: pd.DataFrame,
    small_multiples_selection: pd.DataFrame,
    cutoff_sweep_percentiles: pd.DataFrame,
    fixed_rule_summary: pd.DataFrame,
    fixed_rule_parameter_comparison_summary: pd.DataFrame,
    temporal_picking_summary: pd.DataFrame,
    overlap_rater_count_summary: pd.DataFrame,
    goodreads_rating_count_bin_summary: pd.DataFrame,
    yearly_consistency_summary: pd.DataFrame,
    network_validated_profiles: pd.DataFrame,
) -> None:
    summary_text = _results_markdown(
        profile_summary,
        model_metrics,
        drop_curves,
        policy_holdout_results,
        holdout_correlations,
        small_multiples_selection,
        cutoff_sweep_percentiles,
        fixed_rule_summary,
        fixed_rule_parameter_comparison_summary,
        temporal_picking_summary,
        overlap_rater_count_summary,
        goodreads_rating_count_bin_summary,
        yearly_consistency_summary,
        network_validated_profiles,
    )
    RESULTS_SO_FAR_MD.write_text(summary_text)
    _write_results_everything(
        profile_summary,
        cutoff_sweep_percentiles,
        fixed_rule_summary,
        fixed_rule_parameter_comparison_summary,
        network_validated_profiles,
        temporal_picking_summary,
        overlap_rater_count_summary,
        goodreads_rating_count_bin_summary,
        yearly_consistency_summary,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    action_path = AI_ACTIONS_DIR / f"{timestamp}_phase1_summary.md"
    action_path.write_text(summary_text)


def run_pipeline(
    minimum_public_ratings: int = 100,
    limit_profiles: int | None = None,
    refresh_feeds: bool = False,
    max_pages: int | None = None,
    sleep_seconds: float = 0.1,
    verbose: bool = False,
    bootstrap_samples: int = 300,
    small_multiple_count: int = 19,
    run_network_expansion: bool = False,
    include_network_profiles_in_analysis: bool = False,
    analysis_target_total: int | None = None,
    network_max_depth: int = 2,
    network_max_profiles: int = 600,
    network_target_pass_count: int | None = None,
    network_profile_step_size: int = 200,
    network_max_profiles_cap: int = 4000,
    network_validation_sleep_seconds: float | None = None,
) -> dict[str, pd.DataFrame]:
    ensure_directories()
    network_discovered_profiles = pd.DataFrame()
    network_validated_profiles = pd.DataFrame()
    validation_sleep_seconds = (
        network_validation_sleep_seconds
        if network_validation_sleep_seconds is not None
        else sleep_seconds
    )
    if run_network_expansion:
        if network_target_pass_count is not None and network_target_pass_count > 0:
            network_discovered_profiles, network_validated_profiles = (
                expand_network_until_target_passes(
                    target_pass_count=network_target_pass_count,
                    start_max_profiles=network_max_profiles,
                    profile_step_size=network_profile_step_size,
                    max_profiles_cap=network_max_profiles_cap,
                    max_depth=network_max_depth,
                    crawl_sleep_seconds=sleep_seconds,
                    validation_sleep_seconds=validation_sleep_seconds,
                    verbose=verbose,
                )
            )
        else:
            network_discovered_profiles = crawl_public_profile_network(
                max_depth=network_max_depth,
                max_profiles=network_max_profiles,
                sleep_seconds=sleep_seconds,
                verbose=verbose,
            )
            network_validated_profiles = validate_network_candidates(
                network_discovered_profiles,
                sleep_seconds=validation_sleep_seconds,
                verbose=verbose,
            )
    elif (
        include_network_profiles_in_analysis and NETWORK_VALIDATED_PROFILES_CSV.exists()
    ):
        if NETWORK_DISCOVERED_PROFILES_CSV.exists():
            network_discovered_profiles = pd.read_csv(NETWORK_DISCOVERED_PROFILES_CSV)
        network_validated_profiles = pd.read_csv(NETWORK_VALIDATED_PROFILES_CSV)

    analysis_manifest = write_analysis_manifest(
        network_validated_frame=network_validated_profiles,
        include_network_profiles=include_network_profiles_in_analysis,
        minimum_public_ratings=minimum_public_ratings,
        limit_profiles=limit_profiles,
        analysis_target_total=analysis_target_total,
        output_path=ANALYSIS_PROFILE_MANIFEST_CSV,
    )
    profiles = profiles_from_frame(analysis_manifest)

    raw_frames: list[pd.DataFrame] = []
    prepared_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []
    drop_frames: list[pd.DataFrame] = []

    for profile in profiles:
        feed_path = profile.feed_path(PROFILE_FEEDS_DIR)
        raw_profile = fetch_profile_books(
            profile=profile,
            output_path=feed_path,
            refresh=refresh_feeds,
            max_pages=max_pages,
            sleep_seconds=sleep_seconds,
            verbose=verbose,
        )
        if raw_profile.empty:
            summary_rows.append(
                {
                    "profile_slug": profile.profile_slug,
                    "display_name": profile.display_name,
                    "fetch_status": "empty",
                    "n_rated_books": 0,
                    "n_distinct_years": 0,
                    "latest_event_year": None,
                    "dominant_year_share": math.nan,
                    "split_type": "not_evaluated",
                    "holdout_label": "",
                    "n_train": 0,
                    "n_holdout": 0,
                }
            )
            continue

        raw_frames.append(raw_profile)
        prepared = prepare_profile_books(raw_profile)
        if prepared.empty:
            summary_rows.append(
                {
                    "profile_slug": profile.profile_slug,
                    "display_name": profile.display_name,
                    "fetch_status": "no_ratings_after_cleaning",
                    "n_rated_books": 0,
                    "n_distinct_years": 0,
                    "latest_event_year": None,
                    "dominant_year_share": math.nan,
                    "split_type": "not_evaluated",
                    "holdout_label": "",
                    "n_train": 0,
                    "n_holdout": 0,
                }
            )
            continue

        prepared_frames.append(prepared)
        if not profile_has_sufficient_rating_variation(prepared):
            summary_rows.append(
                {
                    "profile_slug": profile.profile_slug,
                    "display_name": profile.display_name,
                    "fetch_status": "fetched",
                    "n_rated_books": int(len(prepared)),
                    "n_distinct_years": int(prepared["event_year"].dropna().nunique()),
                    "latest_event_year": (
                        int(prepared["event_year"].dropna().max())
                        if prepared["event_year"].notna().any()
                        else None
                    ),
                    "dominant_year_share": (
                        float(
                            prepared["event_year"]
                            .dropna()
                            .astype(int)
                            .value_counts(normalize=True)
                            .iloc[0]
                        )
                        if prepared["event_year"].notna().any()
                        else math.nan
                    ),
                    "split_type": "not_evaluated",
                    "holdout_label": "degenerate_ratings",
                    "n_train": 0,
                    "n_holdout": 0,
                }
            )
            continue
        evaluation = evaluate_profile(prepared)
        if evaluation is None:
            summary_rows.append(
                {
                    "profile_slug": profile.profile_slug,
                    "display_name": profile.display_name,
                    "fetch_status": "fetched",
                    "n_rated_books": int(len(prepared)),
                    "n_distinct_years": int(prepared["event_year"].dropna().nunique()),
                    "latest_event_year": (
                        int(prepared["event_year"].dropna().max())
                        if prepared["event_year"].notna().any()
                        else None
                    ),
                    "dominant_year_share": (
                        float(
                            prepared["event_year"]
                            .dropna()
                            .astype(int)
                            .value_counts(normalize=True)
                            .iloc[0]
                        )
                        if prepared["event_year"].notna().any()
                        else math.nan
                    ),
                    "split_type": "not_evaluated",
                    "holdout_label": "insufficient_data",
                    "n_train": 0,
                    "n_holdout": 0,
                }
            )
            continue

        metrics, curves, summary = evaluation
        summary["fetch_status"] = "fetched"
        summary_rows.append(summary)
        metric_frames.append(metrics)
        drop_frames.append(curves)

    raw_all = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    prepared_all = (
        pd.concat(prepared_frames, ignore_index=True)
        if prepared_frames
        else pd.DataFrame()
    )
    profile_summary = pd.DataFrame(summary_rows).sort_values(
        ["split_type", "n_rated_books", "profile_slug"],
        ascending=[True, False, True],
        na_position="last",
    )
    model_metrics = (
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    )
    drop_curves = (
        pd.concat(drop_frames, ignore_index=True) if drop_frames else pd.DataFrame()
    )
    policy_outputs = run_policy_analysis(
        prepared_all,
        bootstrap_samples=bootstrap_samples,
        small_multiple_count=small_multiple_count,
        analysis_manifest=analysis_manifest,
    )
    policy_holdout_results = policy_outputs["policy_holdout_results"]
    policy_threshold_summary = policy_outputs["policy_threshold_summary"]
    training_model_selection = policy_outputs["training_model_selection"]
    holdout_correlations = policy_outputs["holdout_correlations"]
    small_multiples_selection = policy_outputs["small_multiples_selection"]
    cutoff_sweep_outputs = build_goodreads_cutoff_sweep(prepared_all)
    cutoff_sweep_profiles = cutoff_sweep_outputs["cutoff_sweep_profiles"]
    cutoff_sweep_percentiles = cutoff_sweep_outputs["cutoff_sweep_percentiles"]
    cutoff_sweep_all_books_outputs = build_goodreads_cutoff_sweep(
        prepared_all,
        evaluation_scope="all_books",
    )
    cutoff_sweep_profiles_all_books = cutoff_sweep_all_books_outputs[
        "cutoff_sweep_profiles"
    ]
    cutoff_sweep_percentiles_all_books = cutoff_sweep_all_books_outputs[
        "cutoff_sweep_percentiles"
    ]
    fixed_rule_outputs = compute_fixed_rule_outputs(
        prepared_all,
        analysis_manifest=analysis_manifest,
    )
    fixed_rule_profile_metrics = fixed_rule_outputs["profile_metrics"]
    fixed_rule_rating_shares = fixed_rule_outputs["rating_shares"]
    fixed_rule_summary = fixed_rule_outputs["summary"]
    fixed_rule_parameter_comparison_outputs = compute_fixed_rule_parameter_comparison(
        prepared_all
    )
    fixed_rule_parameter_comparison_profiles = fixed_rule_parameter_comparison_outputs[
        "comparison_profiles"
    ]
    fixed_rule_parameter_comparison_summary = fixed_rule_parameter_comparison_outputs[
        "comparison_summary"
    ]
    fixed_rule_mean_trim = filter_fixed_rule_profiles_by_percentile_window(
        fixed_rule_profile_metrics,
        fixed_rule_rating_shares,
        metric_column="baseline_mean",
        lower_quantile=0.10,
        upper_quantile=0.90,
    )
    fixed_rule_std_trim = filter_fixed_rule_profiles_by_percentile_window(
        fixed_rule_profile_metrics,
        fixed_rule_rating_shares,
        metric_column="profile_rating_std_all_books",
        lower_quantile=0.10,
        upper_quantile=0.90,
    )
    fixed_rule_corr_bands = assign_fixed_rule_correlation_bands(
        fixed_rule_profile_metrics,
        fixed_rule_rating_shares,
    )
    temporal_picking_profiles = compute_temporal_picking_profile_metrics(prepared_all)
    temporal_picking_summary = summarize_temporal_picking(temporal_picking_profiles)
    reviewer_volume_outputs = compute_reviewer_volume_metrics(
        prepared_all,
        analysis_manifest=analysis_manifest,
    )
    reviewer_volume_summary = reviewer_volume_outputs["reviewer_volume_summary"]
    reviewer_year_counts = reviewer_volume_outputs["reviewer_year_counts"]
    yearly_activity_summary = reviewer_volume_outputs["yearly_activity_summary"]
    overlap_outputs = compute_book_overlap_metrics(prepared_all)
    reviewer_pair_correlations = overlap_outputs["reviewer_pair_correlations"]
    overlap_books = overlap_outputs["overlap_books"]
    overlap_rater_count_summary = overlap_outputs["overlap_rater_count_summary"]
    goodreads_book_rating_counts = (
        pd.read_csv(GOODREADS_BOOK_RATING_COUNTS_CSV)
        if GOODREADS_BOOK_RATING_COUNTS_CSV.exists()
        else pd.DataFrame()
    )
    rating_count_accuracy_outputs = compute_goodreads_rating_count_accuracy(
        prepared_all,
        goodreads_book_rating_counts,
    )
    goodreads_rating_count_book_metrics = rating_count_accuracy_outputs[
        "rating_count_book_metrics"
    ]
    goodreads_rating_count_bin_summary = rating_count_accuracy_outputs[
        "rating_count_bin_summary"
    ]
    yearly_consistency_outputs = compute_yearly_consistency_outputs(prepared_all)
    yearly_profile_metrics = yearly_consistency_outputs["yearly_profile_metrics"]
    yearly_transfer_metrics = yearly_consistency_outputs["yearly_transfer_metrics"]
    yearly_gain_differences = yearly_consistency_outputs["yearly_gain_differences"]
    yearly_correlation_pairs = yearly_consistency_outputs["yearly_correlation_pairs"]
    yearly_consistency_summary = yearly_consistency_outputs[
        "yearly_consistency_summary"
    ]

    raw_all.to_csv(RAW_PROFILE_BOOKS_CSV, index=False)
    prepared_all.to_csv(PREPARED_PROFILE_BOOKS_CSV, index=False)
    analysis_manifest.to_csv(ANALYSIS_PROFILE_MANIFEST_CSV, index=False)
    profile_summary.to_csv(PROFILE_SUMMARY_CSV, index=False)
    model_metrics.to_csv(MODEL_METRICS_CSV, index=False)
    drop_curves.to_csv(DROP_CURVES_CSV, index=False)
    policy_holdout_results.to_csv(POLICY_HOLDOUT_RESULTS_CSV, index=False)
    policy_threshold_summary.to_csv(POLICY_THRESHOLD_SUMMARY_CSV, index=False)
    training_model_selection.to_csv(TRAINING_MODEL_SELECTION_CSV, index=False)
    holdout_correlations.to_csv(HOLDOUT_CORRELATIONS_CSV, index=False)
    small_multiples_selection.to_csv(SMALL_MULTIPLES_SELECTION_CSV, index=False)
    cutoff_sweep_profiles.to_csv(CUTOFF_SWEEP_PROFILE_CSV, index=False)
    cutoff_sweep_percentiles.to_csv(CUTOFF_SWEEP_PERCENTILES_CSV, index=False)
    cutoff_sweep_profiles_all_books.to_csv(
        CUTOFF_SWEEP_PROFILE_ALL_BOOKS_CSV, index=False
    )
    cutoff_sweep_percentiles_all_books.to_csv(
        CUTOFF_SWEEP_PERCENTILES_ALL_BOOKS_CSV, index=False
    )
    fixed_rule_profile_metrics.to_csv(FIXED_RULE_PROFILE_METRICS_CSV, index=False)
    fixed_rule_rating_shares.to_csv(FIXED_RULE_RATING_SHARES_CSV, index=False)
    fixed_rule_summary.to_csv(FIXED_RULE_SUMMARY_CSV, index=False)
    fixed_rule_parameter_comparison_profiles.to_csv(
        FIXED_RULE_PARAMETER_COMPARISON_PROFILE_CSV,
        index=False,
    )
    fixed_rule_parameter_comparison_summary.to_csv(
        FIXED_RULE_PARAMETER_COMPARISON_SUMMARY_CSV,
        index=False,
    )
    temporal_picking_profiles.to_csv(TEMPORAL_PICKING_PROFILE_CSV, index=False)
    temporal_picking_summary.to_csv(TEMPORAL_PICKING_SUMMARY_CSV, index=False)
    reviewer_volume_summary.to_csv(REVIEWER_VOLUME_SUMMARY_CSV, index=False)
    reviewer_year_counts.to_csv(REVIEWER_YEAR_COUNTS_CSV, index=False)
    yearly_activity_summary.to_csv(YEARLY_ACTIVITY_SUMMARY_CSV, index=False)
    reviewer_pair_correlations.to_csv(REVIEWER_PAIR_CORRELATIONS_CSV, index=False)
    overlap_books.to_csv(OVERLAP_BOOKS_CSV, index=False)
    overlap_rater_count_summary.to_csv(OVERLAP_RATER_COUNT_SUMMARY_CSV, index=False)
    goodreads_rating_count_book_metrics.to_csv(
        GOODREADS_RATING_COUNT_BOOK_METRICS_CSV,
        index=False,
    )
    goodreads_rating_count_bin_summary.to_csv(
        GOODREADS_RATING_COUNT_BIN_SUMMARY_CSV,
        index=False,
    )
    overlap_books.loc[overlap_books["n_raters"] > 2].to_string(
        OVERLAP_BOOKS_GT2_REVIEWERS_TXT,
        index=False,
        columns=[
            "title",
            "author_name",
            "n_raters",
            "rating_sd",
            "mean_pairwise_profile_corr",
        ],
    )
    yearly_profile_metrics.to_csv(YEARLY_PROFILE_METRICS_CSV, index=False)
    yearly_transfer_metrics.to_csv(YEARLY_TRANSFER_METRICS_CSV, index=False)
    yearly_gain_differences.to_csv(YEARLY_GAIN_DIFFERENCES_CSV, index=False)
    yearly_correlation_pairs.to_csv(YEARLY_CORRELATION_PAIRS_CSV, index=False)
    yearly_consistency_summary.to_csv(YEARLY_CONSISTENCY_SUMMARY_CSV, index=False)
    FIXED_RULE_PROFILE_AUDIT_MD.write_text(
        build_fixed_rule_audit_report(
            fixed_rule_profile_metrics,
            prepared_all,
            cutoff_sweep_profiles_all_books=cutoff_sweep_profiles_all_books,
        )
    )
    if run_network_expansion or not network_discovered_profiles.empty:
        network_discovered_profiles.to_csv(NETWORK_DISCOVERED_PROFILES_CSV, index=False)
    if run_network_expansion or not network_validated_profiles.empty:
        network_validated_profiles.to_csv(NETWORK_VALIDATED_PROFILES_CSV, index=False)

    if not policy_holdout_results.empty:
        plot_policy_gain_histograms(policy_holdout_results, POLICY_GAIN_HISTOGRAMS_PNG)
    if not holdout_correlations.empty:
        plot_holdout_correlation_histograms(
            holdout_correlations,
            HOLDOUT_CORRELATIONS_PNG,
        )
    if not small_multiples_selection.empty:
        plot_goodreads_small_multiples(
            prepared_all,
            small_multiples_selection,
            GOODREADS_SMALL_MULTIPLES_PNG,
        )
    if not cutoff_sweep_percentiles_all_books.empty:
        plot_goodreads_cutoff_percentile_tradeoff(
            cutoff_sweep_percentiles_all_books,
            GOODREADS_CUTOFF_PERCENTILE_PNG,
            scope_label="all rated books",
        )
    if (
        not cutoff_sweep_percentiles_all_books.empty
        and not cutoff_sweep_profiles_all_books.empty
    ):
        plot_goodreads_cutoff_percentile_and_violin(
            cutoff_sweep_percentiles_all_books,
            cutoff_sweep_profiles_all_books,
            GOODREADS_CUTOFF_PERCENTILE_VIOLIN_PNG,
            scope_label="all rated books",
        )
    if not cutoff_sweep_percentiles.empty and not cutoff_sweep_profiles.empty:
        plot_goodreads_cutoff_gain_and_drop_violins(
            cutoff_sweep_profiles,
            GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_PNG,
            scope_label="holdout books",
        )
    if not cutoff_sweep_profiles_all_books.empty:
        plot_goodreads_cutoff_gain_and_drop_violins(
            cutoff_sweep_profiles_all_books,
            GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_ALL_BOOKS_PNG,
            scope_label="all rated books",
        )
    if not fixed_rule_rating_shares.empty and not fixed_rule_profile_metrics.empty:
        plot_fixed_rule_rating_share_violins(
            fixed_rule_rating_shares,
            fixed_rule_profile_metrics,
            FIXED_RULE_RATING_SHARE_VIOLINS_PNG,
        )
    if (
        not fixed_rule_mean_trim["rating_shares"].empty
        and not fixed_rule_mean_trim["profile_metrics"].empty
    ):
        mean_summary = fixed_rule_mean_trim["summary"]
        plot_fixed_rule_rating_share_violins(
            fixed_rule_mean_trim["rating_shares"],
            fixed_rule_mean_trim["profile_metrics"],
            FIXED_RULE_RATING_SHARE_VIOLINS_MEAN_TRIM_PNG,
            figure_title=(
                "Across readers: fixed-rule rating shares after trimming extreme mean raters\n"
                "Restricted to readers between the 10th and 90th percentiles of average personal rating"
            ),
            subtitle_suffix=(
                f"mean in [{mean_summary['lower_bound']:.2f}, {mean_summary['upper_bound']:.2f}]"
            ),
        )
    if (
        not fixed_rule_std_trim["rating_shares"].empty
        and not fixed_rule_std_trim["profile_metrics"].empty
    ):
        std_summary = fixed_rule_std_trim["summary"]
        plot_fixed_rule_rating_share_violins(
            fixed_rule_std_trim["rating_shares"],
            fixed_rule_std_trim["profile_metrics"],
            FIXED_RULE_RATING_SHARE_VIOLINS_STD_TRIM_PNG,
            figure_title=(
                "Across readers: fixed-rule rating shares after trimming extreme rating-variance profiles\n"
                "Restricted to readers between the 10th and 90th percentiles of personal-rating standard deviation"
            ),
            subtitle_suffix=(
                f"std in [{std_summary['lower_bound']:.2f}, {std_summary['upper_bound']:.2f}]"
            ),
        )
    if (
        not fixed_rule_corr_bands["rating_shares"].empty
        and not fixed_rule_corr_bands["profile_metrics"].empty
        and not fixed_rule_corr_bands["band_summary"].empty
    ):
        plot_fixed_rule_rating_share_grid_by_correlation(
            fixed_rule_corr_bands["rating_shares"],
            fixed_rule_corr_bands["profile_metrics"],
            fixed_rule_corr_bands["band_summary"],
            FIXED_RULE_RATING_SHARE_CORR_BANDS_PNG,
        )
    fixed_rule_row_specs = [
        {
            "row_title": "Full panel",
            "subtitle_suffix": "all usable readers",
            "profile_metrics": fixed_rule_profile_metrics,
            "rating_shares": fixed_rule_rating_shares,
        },
        {
            "row_title": "Mean-trimmed panel",
            "subtitle_suffix": (
                f"mean in [{fixed_rule_mean_trim['summary']['lower_bound']:.2f}, "
                f"{fixed_rule_mean_trim['summary']['upper_bound']:.2f}]"
            ),
            "profile_metrics": fixed_rule_mean_trim["profile_metrics"],
            "rating_shares": fixed_rule_mean_trim["rating_shares"],
        },
        {
            "row_title": "SD-trimmed panel",
            "subtitle_suffix": (
                f"std in [{fixed_rule_std_trim['summary']['lower_bound']:.2f}, "
                f"{fixed_rule_std_trim['summary']['upper_bound']:.2f}]"
            ),
            "profile_metrics": fixed_rule_std_trim["profile_metrics"],
            "rating_shares": fixed_rule_std_trim["rating_shares"],
        },
    ]
    if all(not spec["rating_shares"].empty for spec in fixed_rule_row_specs):
        plot_fixed_rule_rating_share_grouped_rows(
            fixed_rule_row_specs,
            FIXED_RULE_RATING_SHARE_GROUPED_ROWS_PNG,
        )
        plot_fixed_rule_rating_share_panel_grid(
            fixed_rule_row_specs,
            FIXED_RULE_RATING_SHARE_PANEL_GRID_PNG,
        )
    if not fixed_rule_parameter_comparison_summary.empty:
        plot_fixed_rule_parameter_comparison_heatmaps(
            fixed_rule_parameter_comparison_summary,
            FIXED_RULE_PARAMETER_COMPARISON_HEATMAP_PNG,
        )
    if not reviewer_volume_summary.empty and not reviewer_year_counts.empty:
        plot_reviewer_volume_distributions(
            reviewer_volume_summary,
            reviewer_year_counts,
            yearly_activity_summary,
            REVIEWER_VOLUME_DISTRIBUTIONS_PNG,
        )
    if not overlap_rater_count_summary.empty:
        plot_book_overlap_summary(
            overlap_rater_count_summary,
            BOOK_OVERLAP_SUMMARY_PNG,
        )
    if not goodreads_rating_count_bin_summary.empty:
        plot_goodreads_rating_count_accuracy(
            goodreads_rating_count_bin_summary,
            GOODREADS_RATING_COUNT_ACCURACY_PNG,
        )
    if not yearly_gain_differences.empty:
        plot_yearly_gain_difference_histograms(
            yearly_gain_differences,
            YEARLY_GAIN_DIFFERENCE_HISTOGRAMS_PNG,
        )
    if not yearly_correlation_pairs.empty:
        plot_yearly_correlation_consistency(
            yearly_correlation_pairs,
            YEARLY_CORRELATION_CONSISTENCY_PNG,
        )
    if not yearly_transfer_metrics.empty:
        plot_yearly_cutoff_transfer_summary(
            yearly_transfer_metrics,
            YEARLY_CUTOFF_TRANSFER_SUMMARY_PNG,
        )

    _write_results(
        profile_summary,
        model_metrics,
        drop_curves,
        policy_holdout_results,
        holdout_correlations,
        small_multiples_selection,
        cutoff_sweep_percentiles_all_books,
        fixed_rule_summary,
        fixed_rule_parameter_comparison_summary,
        temporal_picking_summary,
        overlap_rater_count_summary,
        goodreads_rating_count_bin_summary,
        yearly_consistency_summary,
        network_validated_profiles,
    )

    return {
        "analysis_manifest": analysis_manifest,
        "raw_profile_books": raw_all,
        "prepared_profile_books": prepared_all,
        "profile_summary": profile_summary,
        "model_metrics": model_metrics,
        "drop_curves": drop_curves,
        "policy_holdout_results": policy_holdout_results,
        "policy_threshold_summary": policy_threshold_summary,
        "training_model_selection": training_model_selection,
        "holdout_correlations": holdout_correlations,
        "small_multiples_selection": small_multiples_selection,
        "cutoff_sweep_profiles": cutoff_sweep_profiles,
        "cutoff_sweep_percentiles": cutoff_sweep_percentiles,
        "cutoff_sweep_profiles_all_books": cutoff_sweep_profiles_all_books,
        "cutoff_sweep_percentiles_all_books": cutoff_sweep_percentiles_all_books,
        "fixed_rule_profile_metrics": fixed_rule_profile_metrics,
        "fixed_rule_rating_shares": fixed_rule_rating_shares,
        "fixed_rule_summary": fixed_rule_summary,
        "fixed_rule_parameter_comparison_profiles": (
            fixed_rule_parameter_comparison_profiles
        ),
        "fixed_rule_parameter_comparison_summary": (
            fixed_rule_parameter_comparison_summary
        ),
        "temporal_picking_profiles": temporal_picking_profiles,
        "temporal_picking_summary": temporal_picking_summary,
        "reviewer_volume_summary": reviewer_volume_summary,
        "reviewer_year_counts": reviewer_year_counts,
        "yearly_activity_summary": yearly_activity_summary,
        "reviewer_pair_correlations": reviewer_pair_correlations,
        "overlap_books": overlap_books,
        "overlap_rater_count_summary": overlap_rater_count_summary,
        "goodreads_rating_count_book_metrics": goodreads_rating_count_book_metrics,
        "goodreads_rating_count_bin_summary": goodreads_rating_count_bin_summary,
        "yearly_profile_metrics": yearly_profile_metrics,
        "yearly_transfer_metrics": yearly_transfer_metrics,
        "yearly_gain_differences": yearly_gain_differences,
        "yearly_correlation_pairs": yearly_correlation_pairs,
        "yearly_consistency_summary": yearly_consistency_summary,
        "network_discovered_profiles": network_discovered_profiles,
        "network_validated_profiles": network_validated_profiles,
    }
