# Book Speed Analysis

Calendar time starts from `data/calendar_analysis.txt`, the cached final TSV produced by `Self_Tracking/calendar_analysis.py`. The primary minute column starts from that dataframe's overlap-adjusted `duration`, then restores full wall-clock duration for book/audiobook rows whose actual overlapped time is longer than 15 minutes; overlaps of exactly 15 minutes remain split by the calendar analysis. Full wall-clock time from `start_time`/`end_time` is retained separately in `first_pass_wall_clock_minutes` and used for the full-time comparison plots.

WPM cohorts require `word_count_source == local_file_word_count`, so page estimates and online word counts remain in the word-count audit but do not feed the main speed numbers. Local EPUB/PDF counts use the reading/body-text count where section-level filtering is available, with raw extracted words and excluded notes/index/front-back matter retained in the local text audit. Rows above 900 WPM are not excluded. The only duration screen used for the main subset is `primary_first_pass_minutes > 90`, and every short-duration exclusion is listed below and in `book_speed_short_duration_exclusions.csv`.

## Core Files

- Calendar time: `book_calendar_first_pass_time.csv` and `book_calendar_first_pass_events.csv`
- Calendar overlap audit: `book_calendar_overlap_audit.csv`
- Calendar reconciliation: `book_calendar_time_reconciliation.csv` and `book_calendar_unmatched_includable_events.csv`
- Word counts: `book_word_counts.csv`, `book_word_count_source_audit.csv`, `book_word_count_validation_summary.csv`, `book_word_count_local_text_audit.csv`, and `book_word_count_local_section_audit.csv`
- Online-vs-local word-count error: `book_word_count_online_error_rates.csv`, `book_word_count_online_error_outliers.csv`, and `book_word_count_online_error_rates.png`
- Joined analysis: `book_speed_analysis.csv`
- Duration-screened subset: `book_speed_duration_screened_subset.csv`
- Visual-reading subset excluding audio-dominant books: `book_speed_visual_reading_subset.csv`
- Audio-dominant exclusions: `book_speed_audio_dominant_exclusions.csv`
- Short-duration exclusions: `book_speed_short_duration_exclusions.csv`
- Rolling category WPM percentiles: `book_speed_rolling_6mo_category_percentiles.csv` and `book_speed_rolling_6mo_category_percentiles.png`
- High-WPM investigation: `book_speed_high_wpm_investigation.csv` and `book_speed_high_wpm_event_evidence.csv`
- >600 WPM duration-screened investigation: `book_speed_gt600_wpm_investigation.csv` and `book_speed_gt600_wpm_event_evidence.csv`

## Summary

| usable_books_before_duration_screen | duration_screened_books | excluded_lte_90_minutes | gt900_wpm_books_investigated | gt600_wpm_duration_screened_books | audio_dominant_books_excluded_from_visual_wpm | visual_reading_books_after_audio_screen | median_duration_screened_wpm | median_visual_wpm_after_350wpm_audio_adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 131.00 | 117.00 | 14.00 | 6.00 | 3.00 | 5.00 | 112.00 | 314.74 | 307.65 |

## WPM Percentiles

| percentile | wpm_first_pass_primary |
| --- | --- |
| 5.00 | 191.73 |
| 10.00 | 233.50 |
| 25.00 | 261.65 |
| 50.00 | 314.74 |
| 75.00 | 390.32 |
| 90.00 | 456.11 |
| 95.00 | 542.06 |

## Aggregate WPM

Aggregate WPM is `sum(chosen_word_count) / sum(primary_first_pass_minutes)`, not the mean of per-book WPMs. Full wall-clock aggregate WPM is included for comparison.

| cohort | n_finished_read_instances | total_estimated_words | total_primary_minutes | total_primary_hours | total_full_wall_clock_minutes | total_full_wall_clock_hours | aggregate_wpm | aggregate_visual_after_audio_350wpm | aggregate_full_wall_clock_wpm | mean_of_book_wpms | mean_of_book_full_wall_clock_wpms | median_book_wpm | median_book_visual_after_audio_350wpm | median_book_full_wall_clock_wpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_finished_read_instances_with_local_file_word_count | 137 | 18858659.00 | 60144.44 | 1002.41 | 60905.00 | 1015.08 | 313.56 | 311.45 | 309.64 | 397.87 | 392.94 | 323.24 | 322.17 | 319.39 |
| first_reads_with_local_file_word_count | 131 | 18429695.00 | 58934.44 | 982.24 | 59695.00 | 994.92 | 312.72 | 310.75 | 308.73 | 399.13 | 393.98 | 322.17 | 321.95 | 318.55 |
| first_reads_gt90m_with_local_file_word_count | 117 | 17649001.00 | 57921.94 | 965.37 | 58675.00 | 977.92 | 304.70 | 302.24 | 300.79 | 335.71 | 330.24 | 314.74 | 310.42 | 306.80 |
| visual_reading_first_reads_gt90m_excluding_audio_dominant | 112 | 16596993.00 | 54966.94 | 916.12 | 55675.00 | 927.92 | 301.94 | 301.00 | 298.10 | 332.89 | 327.89 | 310.64 | 307.65 | 306.64 |

## WPM By Year

| finish_year | n_books | median_wpm | mean_wpm |
| --- | --- | --- | --- |
| 2021.00 | 1.00 | 457.16 | 457.16 |
| 2022.00 | 16.00 | 266.43 | 282.67 |
| 2023.00 | 11.00 | 276.94 | 283.14 |
| 2024.00 | 28.00 | 342.39 | 371.10 |
| 2025.00 | 41.00 | 321.72 | 356.16 |
| 2026.00 | 20.00 | 299.82 | 309.51 |

## WPM By Category

| category_plot | n_books | median_wpm | mean_wpm |
| --- | --- | --- | --- |
| Business, management | 28 | 303.68 | 334.80 |
| fiction | 23 | 315.22 | 325.63 |
| Unknown | 20 | 337.98 | 351.19 |
| General Reading | 16 | 344.86 | 358.72 |
| Histories | 15 | 263.99 | 311.22 |
| Literature | 11 | 259.41 | 311.77 |
| Machine Learning | 3 | 503.29 | 407.69 |
| Computer Science | 1 | 330.18 | 330.18 |

## Highlight Effect

| n_books | pearson_corr_log_highlight_density_wpm | slope_wpm_per_log1p_highlight_density | p25_highlight_words_per_10k_words | p75_highlight_words_per_10k_words | predicted_wpm_at_p25_highlight_density | predicted_wpm_at_p75_highlight_density | observational_wpm_change_if_p75_to_p25_density |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 114.00 | -0.05 | -1.72 | 0.00 | 848.24 | 341.36 | 329.75 | 11.61 |

## Highlight And Note Models

| model | response_column | predictor | n_books | coefficient_wpm_per_log1p_unit | intercept_or_baseline | r_squared | p25_predictor | p75_predictor | estimated_wpm_change_p75_to_p25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unadjusted | wpm_first_pass_primary | highlight_count_per_page | 114 | -76.80 | 349.35 | 0.02 | 0.00 | 0.45 | 28.34 |
| category_fixed_effects | wpm_first_pass_primary | highlight_count_per_page | 114 | -98.17 | 360.83 | 0.05 | 0.00 | 0.45 | 36.23 |
| unadjusted | wpm_first_pass_primary | my_note_words_per_page | 114 | -27.46 | 341.03 | 0.01 | 0.00 | 0.37 | 8.58 |
| category_fixed_effects | wpm_first_pass_primary | my_note_words_per_page | 114 | -34.14 | 350.36 | 0.03 | 0.00 | 0.37 | 10.67 |

## Confidence Comparison

| confidence_subset | n_books | median_wpm | mean_wpm | aggregate_wpm | aggregate_full_wall_clock_wpm |
| --- | --- | --- | --- | --- | --- |
| high_confidence | 117 | 314.74 | 335.71 | 304.70 | 300.79 |
| medium_confidence | 0 |  |  |  |  |
| local_file_high_confidence | 117 | 314.74 | 335.71 | 304.70 | 300.79 |

## Calendar Time Reconciliation

| metric | events | calendar_analysis_minutes | wall_clock_minutes |
| --- | --- | --- | --- |
| explicit_includable_book_time | 2163.00 | 105812.97 | 111152.50 |
| assigned_to_finished_books | 1521.00 | 84164.03 | 88557.50 |
| unmatched_explicit_book_time | 642.00 | 21648.93 | 22595.00 |
| generic_temporal_not_counted | 119.00 | 2942.50 | 3070.00 |
| bare_finished_marker_not_counted | 0.00 | 0.00 | 0.00 |
| assigned_plus_unmatched_minus_explicit_total |  | 0.00 | 0.00 |

## High-WPM Investigation

| title | matched_finish_date | primary_first_pass_minutes | first_pass_wall_clock_minutes | chosen_word_count | word_count_source | wpm_first_pass_primary | wpm_first_pass_full_wall_clock | duration_screen_exclusion_reason | likely_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SQL | 2022-05-25 | 30.00 | 30.00 | 99534.00 | local_file_word_count | 3317.80 | 3317.80 | calendar_time_lte_90_minutes | calendar_time_lte_90_minutes |
| blueprint reading: construction drawings for the building trade | 2025-07-28 | 45.00 | 45.00 | 92387.00 | local_file_word_count | 2053.04 | 2053.04 | calendar_time_lte_90_minutes | calendar_time_lte_90_minutes |
| what is man and other essays | 2024-08-07 | 75.00 | 75.00 | 101021.00 | local_file_word_count | 1346.95 | 1346.95 | calendar_time_lte_90_minutes | calendar_time_lte_90_minutes |
| Everyman for himself and god against all | 2025-09-01 | 90.00 | 90.00 | 103268.00 | local_file_word_count | 1147.42 | 1147.42 | calendar_time_lte_90_minutes | calendar_time_lte_90_minutes |
| the Oxford book of essays | 2025-03-22 | 285.00 | 300.00 | 283993.00 | local_file_word_count | 996.47 | 946.64 |  | long_book_has_too_little_matched_calendar_time |
| Tremendous Trifles | 2024-03-01 | 60.00 | 60.00 | 59778.00 | local_file_word_count | 996.30 | 996.30 | calendar_time_lte_90_minutes | calendar_time_lte_90_minutes |

## >600 WPM Duration-Screened Investigation

| title | matched_finish_date | primary_first_pass_minutes | first_pass_wall_clock_minutes | chosen_word_count | word_count_source | wpm_first_pass_primary | wpm_first_pass_full_wall_clock | duration_screen_exclusion_reason | likely_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| the Oxford book of essays | 2025-03-22 | 285.00 | 300.00 | 283993.00 | local_file_word_count | 996.47 | 946.64 |  | long_book_has_too_little_matched_calendar_time |
| age of ambition | 2025-08-02 | 202.50 | 210.00 | 151444.00 | local_file_word_count | 747.87 | 721.16 |  | needs_manual_calendar_or_word_count_review |
| selfish reasons to have more kids | 2025-07-12 | 97.50 | 105.00 | 65820.00 | local_file_word_count | 675.08 | 626.86 |  | needs_manual_calendar_or_word_count_review |

## Short-Duration Exclusions

| title | matched_finish_date | primary_first_pass_minutes | first_pass_wall_clock_minutes | chosen_word_count | wpm_first_pass_primary | wpm_first_pass_full_wall_clock | duration_screen_exclusion_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SQL | 2022-05-25 | 30.00 | 30.00 | 99534.00 | 3317.80 | 3317.80 | calendar_time_lte_90_minutes |
| blueprint reading: construction drawings for the building trade | 2025-07-28 | 45.00 | 45.00 | 92387.00 | 2053.04 | 2053.04 | calendar_time_lte_90_minutes |
| what is man and other essays | 2024-08-07 | 75.00 | 75.00 | 101021.00 | 1346.95 | 1346.95 | calendar_time_lte_90_minutes |
| Everyman for himself and god against all | 2025-09-01 | 90.00 | 90.00 | 103268.00 | 1147.42 | 1147.42 | calendar_time_lte_90_minutes |
| Tremendous Trifles | 2024-03-01 | 60.00 | 60.00 | 59778.00 | 996.30 | 996.30 | calendar_time_lte_90_minutes |
| high intensity training the mike mentzer way | 2025-09-22 | 90.00 | 90.00 | 71026.00 | 789.18 | 789.18 | calendar_time_lte_90_minutes |
| The Inner game of tennis | 2021-10-17 | 75.00 | 75.00 | 45281.00 | 603.75 | 603.75 | calendar_time_lte_90_minutes |
| The Art of Readable Code | 2022-11-30 | 75.00 | 75.00 | 42585.00 | 567.80 | 567.80 | calendar_time_lte_90_minutes |
| the last interview | 2024-08-28 | 60.00 | 60.00 | 27863.00 | 464.38 | 464.38 | calendar_time_lte_90_minutes |
| art and fear | 2025-12-15 | 60.00 | 60.00 | 27611.00 | 460.18 | 460.18 | calendar_time_lte_90_minutes |
| self help is like a vaccine | 2024-10-11 | 82.50 | 90.00 | 34342.00 | 416.27 | 381.58 | calendar_time_lte_90_minutes |
| memos from the chairman | 2026-02-25 | 90.00 | 90.00 | 32426.00 | 360.29 | 360.29 | calendar_time_lte_90_minutes |
| theory and practice of gamesmanship | 2025-06-20 | 90.00 | 90.00 | 22115.00 | 245.72 | 245.72 | calendar_time_lte_90_minutes |
| how to succeed at business without really trying | 2025-01-16 | 90.00 | 90.00 | 21457.00 | 238.41 | 238.41 | calendar_time_lte_90_minutes |

## Plots

![WPM histogram](book_speed_histogram.png)

![Full wall-clock WPM histogram](book_speed_histogram_full_wall_clock.png)

![Visual WPM after audiobook adjustment](book_speed_histogram_visual_after_audio_350wpm.png)

![WPM by year](book_speed_violin_by_year.png)

![WPM by category](book_speed_violin_by_category.png)

![Six-month rolling WPM percentiles by category](book_speed_rolling_6mo_category_percentiles.png)

![Time vs pages](book_time_vs_pages.png)

![Full wall-clock time vs pages](book_time_vs_pages_full_wall_clock.png)

![Time vs words](book_time_vs_words.png)

![Full wall-clock time vs words](book_time_vs_words_full_wall_clock.png)

![Online word-count search error vs local extraction](book_word_count_online_error_rates.png)

![WPM vs highlighted words](book_speed_wpm_vs_highlighted_words.png)

![Highlight density vs WPM](book_speed_highlight_density_vs_wpm.png)

![Highlights per page vs WPM](book_speed_highlight_count_per_page_vs_wpm.png)

![My note words per page vs WPM](book_speed_note_words_per_page_vs_wpm.png)

![High-confidence histogram](book_speed_histogram_high_confidence.png)

![Medium-confidence histogram](book_speed_histogram_medium_confidence.png)
