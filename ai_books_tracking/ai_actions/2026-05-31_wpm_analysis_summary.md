# WPM Analysis Summary

Generated 2026-05-31. This note summarizes the book WPM work done from calendar entries, local book files, and Google Play Books notes/highlights.

## Goal

Estimate reading speed from finished-book calendar logs by joining:

- calendar-derived first-pass reading time
- book word counts from local files or metadata fallbacks
- Google Play Books highlight/note counts
- category and finish-year metadata

The main output is the reproducible analysis bundle in `ai_books_tracking/book_wpm_outputs/`, with the joined table in `book_speed_analysis.csv` and the narrative report in `book_speed_report.md`.

## Current Best Cohort

The best current WPM subset is `book_speed_final_best_subset.csv`.

Rules:

- first read only
- `primary_first_pass_minutes > 90`
- usable medium/high confidence word count
- not audio-dominant
- audiobook-adjusted visual WPM available
- `wpm_visual_after_audio_350wpm <= 600`

Current result:

- `N=112`
- aggregate visual WPM: `292.7`
- median visual WPM: `303.2`
- mean visual WPM: `312.5`

Primary plots:

- `book_speed_final_best_histogram.png`
- `book_speed_final_best_violin_by_year.png`
- `book_speed_final_best_violin_by_category.png`

## What Was Built

New reusable scripts:

- `book_wpm_calendar_notes.py`: shared parsing and matching helpers for finished books, abbreviations, notes, and metadata.
- `book_calendar_time.py`: starts from `data/calendar_analysis.txt`, classifies book/audiobook/calendar rows, resolves title abbreviations against finished-book timelines, writes auditable first-pass time tables, and produces calendar reconciliation/audit outputs.
- `book_word_counts.py`: estimates word counts from local book files first, then metadata fallbacks, with source/confidence audit columns.
- `book_speed_analysis.py`: joins calendar time, word counts, highlights, categories, and produces WPM tables, outlier audits, aggregate summaries, plots, and final-best subset outputs.
- `tests/test_book_wpm_calendar_notes.py`: focused regression tests for abbreviation/title parsing behavior.

Important generated outputs:

- `book_calendar_first_pass_time.csv`
- `book_calendar_first_pass_events.csv`
- `book_calendar_overlap_audit.csv`
- `book_calendar_time_reconciliation.csv`
- `book_calendar_unmatched_includable_events.csv`
- `book_word_counts.csv`
- `book_word_count_source_audit.csv`
- `book_speed_analysis.csv`
- `book_speed_noise_sensitivity.csv`
- `book_speed_final_best_subset.csv`
- `book_speed_report.md`

## Calendar Time Handling

The analysis now treats `Self_Tracking/calendar_analysis.py` output as the initial processing source of truth by reading `data/calendar_analysis.txt`.

Calendar time columns are retained separately:

- `first_pass_calendar_minutes`: original overlap-adjusted `duration` from `calendar_analysis.txt`
- `first_pass_wall_clock_minutes`: raw `start_time` to `end_time`
- `first_pass_overlap_policy_minutes`: process-wide restoration policy used as primary reading time
- `primary_first_pass_minutes`: the main analysis time column

Overlap restoration is process-level, not per-book:

- restore full wall-clock duration for minor overlaps of 15 minutes or less
- restore full wall-clock duration for compatible overlaps such as walking, commute, travel, driving, biking, hiking, etc.
- for audiobook rows, additionally allow compatible activity contexts such as exercise, chores, errands, cooking, eating, and similar
- retain adjusted time for incompatible large overlaps, such as work-calendar conflicts

Concerns still tracked:

- Starship Troopers remains `175m` primary and `285m` wall-clock because the process sees a large incompatible overlap; the user believes the first read was `105m`. This is intentionally not hard-overridden.
- Talent remains `285m` in the calendar-derived data; the user screenshot suggests `330m`, but the missing 45-minute event is not present in the cached calendar takeout/analysis source being used.
- The unmatched includable book time is mostly unfinished/study/reference reading rather than dropped finished books, but it should stay auditable through `book_calendar_unmatched_includable_events.csv`.

## Name And Abbreviation Matching

Matching was tightened after specific failures:

- full title entries, `Started Book`, `Start Book`, `Book`, `Finished Book`, and bare finished markers are handled together
- first-letter title abbreviations are resolved against known finished titles over time
- two-letter abbreviations now require literal all-title initials, so `SC` can match `Six Crises`
- one-word titles can match by the word itself, so `bureaucracy` remains valid
- overly broad stopword-dropped two-letter matching was removed so abbreviations like `pp` do not incorrectly match `The Path to Power`
- no per-book time overrides are used

Known checks:

- Bureaucracy: `585m`
- Managing the Professional Services Firm: `420m`
- Six Crises: `450m`
- Out of the Silent Planet: `225m`
- The Path to Power: no spurious 2022 Path to Power row

## Word Counts

Word count selection now prioritizes local files under `data/Books` and records:

- chosen word count
- source
- confidence
- local file path
- local match score
- local extraction error if any

The prior web-search-only estimates were too noisy for several books. Local file counts are now the preferred source where available.

Remaining risks:

- Some PDFs/technical/reference books include diagrams, tables, or non-body text that make text-word counts less comparable to normal prose.
- Some metadata/page-estimate books remain medium confidence.
- Anthologies/reference books can look like very high WPM if only partially read or skimmed.

## Audiobooks

Audiobook time is included in first-pass time and tracked separately.

The speed analysis assumes audiobooks are consumed at `350 WPM` for adjustment:

- `audiobook_primary_minutes`
- `non_audio_primary_minutes`
- `estimated_audio_words_at_350wpm`
- `estimated_visual_words_after_audio_350wpm`
- `wpm_visual_after_audio_350wpm`
- `audio_dominant`

Audio-dominant books are excluded from the best visual-reading-speed cohort.

Current audio-dominant exclusions include:

- The Power Broker
- The History of the English Speaking Peoples Volume 4
- Now It Can Be Told
- The Effective Executive
- The Nvidia Way

Concern:

- The 350 WPM audiobook assumption is useful for cohort filtering, but it is still an assumption. It should not be interpreted as exact word allocation within mixed audio/visual books.

## Aggregate Metrics And Noise

Main first-read, `>90m`, word-count cohort:

- `N=121`
- aggregate primary WPM: `306.8`
- median primary WPM: `308.6`
- mean primary WPM: `337.2`
- aggregate full wall-clock WPM: `300.2`
- aggregate visual-after-audio WPM: `304.7`

Sensitivity checks:

- excluding audio-dominant books: `N=116`, aggregate `302.1`, median `307.4`
- excluding `>900 WPM`: `N=119`, aggregate `301.9`, median `308.0`
- excluding `>600 WPM`: `N=115`, aggregate `291.0`, median `304.6`
- strict visual cohort excluding audio-dominant and `>600 WPM`: `N=112`, aggregate visual `292.7`, median visual `303.2`
- high-confidence word counts only: `N=90`, aggregate `313.6`, median `324.3`

Interpretation:

- central reading speed is roughly `300-310 WPM`
- stricter outlier filtering moves aggregate WPM by roughly `4-5%`
- audio-dominant filtering alone moves aggregate WPM by about `1.5%`
- full wall-clock timing moves aggregate WPM down by about `2%`
- the mean is more sensitive than the median because of skim/reference/high-WPM books

Concern:

- Aggregate WPM, median book WPM, and mean book WPM will not exactly match because they answer different questions. The best headline metric is aggregate WPM for total words over total time, with median book WPM as the robustness check.

## Highlight Analysis

Highlight and note data are joined from Google Play Books notes, with baseline/template text stripped.

Current cleaned visual highlight cohort:

- `N=110`
- correlation between log highlight density and visual WPM: `+0.10`
- predicted change from p75 to p25 highlight density: `-21.5 WPM`

All-finished available highlight cohort:

- `N=136` plotted out of `204` finished read instances
- correlation between log highlight density and raw primary WPM: `-0.11`
- predicted change from p75 to p25 highlight density: `+52.5 WPM`

Interpretation:

- The highlight effect is not stable across cohorts.
- There is not yet strong evidence that highlighting less would reliably increase WPM.
- Category, book difficulty, skim/reference behavior, and missing tablet highlights likely confound this analysis.

## Current File Map

Best visual-speed outputs:

- `book_speed_final_best_subset.csv`
- `book_speed_final_best_by_year.csv`
- `book_speed_final_best_by_category.csv`
- `book_speed_final_best_histogram.png`
- `book_speed_final_best_violin_by_year.png`
- `book_speed_final_best_violin_by_category.png`

Noise/audit outputs:

- `book_speed_noise_sensitivity.csv`
- `book_speed_gt600_wpm_investigation.csv`
- `book_speed_gt600_wpm_event_evidence.csv`
- `book_speed_high_wpm_investigation.csv`
- `book_speed_audio_dominant_exclusions.csv`
- `book_calendar_overlap_audit.csv`

Highlight outputs:

- `book_speed_highlight_cleaned_visual_subset.csv`
- `book_speed_highlight_cleaned_visual_effect_summary.csv`
- `book_speed_highlight_cleaned_visual_note_category_model.csv`
- `book_speed_highlight_all_finished_available_subset.csv`
- `book_speed_highlight_all_finished_effect_summary.csv`

## Verification

Commands run in the latest verified state:

```bash
python3 ai_books_tracking/book_speed_analysis.py
python3 -m black ai_books_tracking/book_wpm_calendar_notes.py ai_books_tracking/book_calendar_time.py ai_books_tracking/book_word_counts.py ai_books_tracking/book_speed_analysis.py tests/test_book_wpm_calendar_notes.py
python3 -m ruff check ai_books_tracking/book_wpm_calendar_notes.py ai_books_tracking/book_calendar_time.py ai_books_tracking/book_word_counts.py ai_books_tracking/book_speed_analysis.py tests/test_book_wpm_calendar_notes.py
python3 -m pytest tests/test_book_wpm_calendar_notes.py
```

Focused test result: `15 passed`.

Full repo tests were not run.
