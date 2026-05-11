from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AI_ACTIONS_DIR = BASE_DIR / "ai_actions"
PLOTS_DIR = BASE_DIR / "plots"
PROFILE_FEEDS_DIR = DATA_DIR / "profile_feeds"
NETWORK_PROFILE_PAGE_CACHE_DIR = DATA_DIR / "network_profile_pages"
NETWORK_PROFILE_FEEDS_DIR = DATA_DIR / "network_profile_feeds"
PROFILE_MANIFEST_CSV = DATA_DIR / "verified_goodreads_profiles_extended.csv"
ANALYSIS_PROFILE_MANIFEST_CSV = DATA_DIR / "analysis_profile_manifest.csv"
RAW_PROFILE_BOOKS_CSV = DATA_DIR / "raw_profile_books.csv"
PREPARED_PROFILE_BOOKS_CSV = DATA_DIR / "prepared_profile_books.csv"
PROFILE_SUMMARY_CSV = DATA_DIR / "profile_summary.csv"
MODEL_METRICS_CSV = DATA_DIR / "model_metrics.csv"
DROP_CURVES_CSV = DATA_DIR / "drop_curves.csv"
POLICY_HOLDOUT_RESULTS_CSV = DATA_DIR / "policy_holdout_results.csv"
POLICY_THRESHOLD_SUMMARY_CSV = DATA_DIR / "policy_threshold_summary.csv"
TRAINING_MODEL_SELECTION_CSV = DATA_DIR / "training_model_selection.csv"
HOLDOUT_CORRELATIONS_CSV = DATA_DIR / "holdout_correlations.csv"
SMALL_MULTIPLES_SELECTION_CSV = DATA_DIR / "small_multiples_selection.csv"
CUTOFF_SWEEP_PROFILE_CSV = DATA_DIR / "goodreads_cutoff_sweep_profiles.csv"
CUTOFF_SWEEP_PERCENTILES_CSV = DATA_DIR / "goodreads_cutoff_sweep_percentiles.csv"
CUTOFF_SWEEP_PROFILE_ALL_BOOKS_CSV = (
    DATA_DIR / "goodreads_cutoff_sweep_profiles_all_books.csv"
)
CUTOFF_SWEEP_PERCENTILES_ALL_BOOKS_CSV = (
    DATA_DIR / "goodreads_cutoff_sweep_percentiles_all_books.csv"
)
NETWORK_DISCOVERED_PROFILES_CSV = DATA_DIR / "network_discovered_profiles.csv"
NETWORK_VALIDATED_PROFILES_CSV = DATA_DIR / "network_validated_profiles.csv"
TEMPORAL_PICKING_PROFILE_CSV = DATA_DIR / "temporal_picking_profile_metrics.csv"
TEMPORAL_PICKING_SUMMARY_CSV = DATA_DIR / "temporal_picking_summary.csv"
REVIEWER_VOLUME_SUMMARY_CSV = DATA_DIR / "reviewer_volume_summary.csv"
REVIEWER_YEAR_COUNTS_CSV = DATA_DIR / "reviewer_year_counts.csv"
YEARLY_ACTIVITY_SUMMARY_CSV = DATA_DIR / "yearly_activity_summary.csv"
REVIEWER_PAIR_CORRELATIONS_CSV = DATA_DIR / "reviewer_pair_correlations.csv"
OVERLAP_BOOKS_CSV = DATA_DIR / "overlap_books.csv"
OVERLAP_RATER_COUNT_SUMMARY_CSV = DATA_DIR / "overlap_rater_count_summary.csv"
OVERLAP_BOOKS_GT2_REVIEWERS_TXT = DATA_DIR / "overlap_books_gt2_reviewers.txt"
GOODREADS_BOOK_RATING_COUNTS_CSV = DATA_DIR / "goodreads_book_rating_counts.csv"
GOODREADS_RATING_COUNT_BOOK_METRICS_CSV = (
    DATA_DIR / "goodreads_rating_count_book_metrics.csv"
)
GOODREADS_RATING_COUNT_BIN_SUMMARY_CSV = (
    DATA_DIR / "goodreads_rating_count_bin_summary.csv"
)
YEARLY_PROFILE_METRICS_CSV = DATA_DIR / "yearly_profile_metrics.csv"
YEARLY_TRANSFER_METRICS_CSV = DATA_DIR / "yearly_transfer_metrics.csv"
YEARLY_GAIN_DIFFERENCES_CSV = DATA_DIR / "yearly_gain_differences.csv"
YEARLY_CORRELATION_PAIRS_CSV = DATA_DIR / "yearly_correlation_pairs.csv"
YEARLY_CONSISTENCY_SUMMARY_CSV = DATA_DIR / "yearly_consistency_summary.csv"
FIXED_RULE_PROFILE_METRICS_CSV = DATA_DIR / "fixed_rule_profile_metrics.csv"
FIXED_RULE_RATING_SHARES_CSV = DATA_DIR / "fixed_rule_rating_shares.csv"
FIXED_RULE_SUMMARY_CSV = DATA_DIR / "fixed_rule_summary.csv"
FIXED_RULE_PARAMETER_COMPARISON_PROFILE_CSV = (
    DATA_DIR / "fixed_rule_parameter_comparison_profiles.csv"
)
FIXED_RULE_PARAMETER_COMPARISON_SUMMARY_CSV = (
    DATA_DIR / "fixed_rule_parameter_comparison_summary.csv"
)
POLICY_GAIN_HISTOGRAMS_PNG = PLOTS_DIR / "holdout_gain_policy_histograms.png"
HOLDOUT_CORRELATIONS_PNG = PLOTS_DIR / "holdout_correlation_histograms.png"
GOODREADS_SMALL_MULTIPLES_PNG = PLOTS_DIR / "goodreads_vs_user_small_multiples.png"
GOODREADS_CUTOFF_PERCENTILE_PNG = PLOTS_DIR / "goodreads_cutoff_percentile_tradeoff.png"
GOODREADS_CUTOFF_PERCENTILE_VIOLIN_PNG = (
    PLOTS_DIR / "goodreads_cutoff_percentile_and_violin.png"
)
GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_PNG = (
    PLOTS_DIR / "goodreads_cutoff_gain_and_drop_violins.png"
)
GOODREADS_CUTOFF_GAIN_DROP_VIOLIN_ALL_BOOKS_PNG = (
    PLOTS_DIR / "goodreads_cutoff_gain_and_drop_violins_all_books.png"
)
REVIEWER_VOLUME_DISTRIBUTIONS_PNG = PLOTS_DIR / "reviewer_volume_distributions.png"
BOOK_OVERLAP_SUMMARY_PNG = PLOTS_DIR / "book_overlap_summary.png"
GOODREADS_RATING_COUNT_ACCURACY_PNG = PLOTS_DIR / "goodreads_rating_count_accuracy.png"
YEARLY_GAIN_DIFFERENCE_HISTOGRAMS_PNG = (
    PLOTS_DIR / "yearly_gain_difference_histograms.png"
)
YEARLY_CORRELATION_CONSISTENCY_PNG = PLOTS_DIR / "yearly_correlation_consistency.png"
YEARLY_CUTOFF_TRANSFER_SUMMARY_PNG = PLOTS_DIR / "yearly_cutoff_transfer_summary.png"
FIXED_RULE_RATING_SHARE_VIOLINS_PNG = PLOTS_DIR / "fixed_rule_rating_share_violins.png"
FIXED_RULE_RATING_SHARE_VIOLINS_MEAN_TRIM_PNG = (
    PLOTS_DIR / "fixed_rule_rating_share_violins_mean_p10_p90.png"
)
FIXED_RULE_RATING_SHARE_VIOLINS_STD_TRIM_PNG = (
    PLOTS_DIR / "fixed_rule_rating_share_violins_std_p10_p90.png"
)
FIXED_RULE_RATING_SHARE_CORR_BANDS_PNG = (
    PLOTS_DIR / "fixed_rule_rating_share_violins_corr_bands.png"
)
FIXED_RULE_RATING_SHARE_GROUPED_ROWS_PNG = (
    PLOTS_DIR / "fixed_rule_rating_share_grouped_rows.png"
)
FIXED_RULE_RATING_SHARE_PANEL_GRID_PNG = (
    PLOTS_DIR / "fixed_rule_rating_share_3x3_grid.png"
)
FIXED_RULE_PARAMETER_COMPARISON_HEATMAP_PNG = (
    PLOTS_DIR / "fixed_rule_parameter_comparison_heatmaps.png"
)
RESULTS_SO_FAR_MD = BASE_DIR / "results_so_far.md"
RESULTS_SO_FAR_EVERYTHING_TXT = BASE_DIR / "results_so_far_everything.txt"
FIXED_RULE_PROFILE_AUDIT_MD = BASE_DIR / "fixed_rule_profile_audit.md"


def ensure_directories() -> None:
    for path in [
        DATA_DIR,
        AI_ACTIONS_DIR,
        PLOTS_DIR,
        PROFILE_FEEDS_DIR,
        NETWORK_PROFILE_PAGE_CACHE_DIR,
        NETWORK_PROFILE_FEEDS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
