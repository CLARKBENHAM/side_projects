# Rating export runbook

Run the unified extractor from the `side_projects` repository:

```bash
python3 -m Self_Tracking.rating_exports
```

Its current defaults use:

- Calendar Takeout: `data/takeout_07_21_26/Calendar`
- Play Books Takeout: `data/play_books_export_small_failure_07_23_2026/Google Play Books`
- Legacy movie reviews: `Self_Tracking/movie_analysis/data/summaries/movie_rt_notes.txt`
- Cached legacy RT identities:
  `Self_Tracking/movie_analysis/data/summaries/movie_rt_analysis/movie_rt_scores_detailed.csv`
- Inclusive new-item date: `2026-03-12`
- Calendar cutoff: `2026-07-22`

Use `python3 -m Self_Tracking.rating_exports --help` to override any source,
date, or output location. The optional previous-ledger and previous-queue
arguments affect only `comparison_report.md`; they are not extraction inputs.

## Files to review

Every invocation creates a new `data/rating_runs/YYYYMMDD_HHMMSS/` directory.
Creation fails if that directory already exists, so no earlier run is
overwritten.

- `movies_new_to_rate.csv`: the movie titles that still need one personal
  rating. A title is omitted if the old manual movie CSV has any label for its
  normalized title, including `na`.
- `books_new_to_rate.csv`: newly discovered book completions and rereads to
  rate.
- `movies_all.csv`: one row per normalized finished movie title across the
  prior labels and Calendar data. Rewatches are represented by `watch_count`
  and `watch_dates`, not duplicate rating rows. Legacy-only watched movies have
  a blank date and `completion_sources=legacy_review_notes`.
- `books_all.csv`: the combined completion history. Books retain separate
  completion rows for rereads.
- `movie_title_normalizations.txt`: every combined movie normalization, title
  variant, Calendar summary, watch date, and reused old label.
- `movies_unfinished_qc.csv`: only unfinished candidates in the current date
  window.
- `movie_legacy_reconciliation_qc.csv`: one audit row for every legacy
  `title rating; review` record, including its RT identity/year evidence,
  candidate row, automatic-join status, and quarantined conflicts.
- `movie_identity_conflicts_qc.csv`: external title/year identities assigned
  to more than one movie row. Conflicting ratings are shown side by side; the
  exporter does not merge or change them.
- `books_reconciliation_qc.csv`: Calendar-finished books that Play marked
  unfinished.
- `comparison_report.md`: count and normalized-title comparison with the
  previous extractors.
- `manifest.json`: source paths, file hashes, dates, policies, manual
  corrections, and row counts.

The two `*_new_to_rate.csv` files are the files to label. The `*_all.csv`
files and normalization/QC files are audit material.

## Privacy and version control

Commit the exporter, its tests, and this runbook only. The repository ignores
`data/**`; keep Takeout inputs, local override JSON files, generated manifests,
rating CSVs, reviews, and normalization/QC outputs there. Do not force-add
anything under `data/`.

## Source and de-duplication rules

Movies use one rating per reconciled movie title. Calendar watch dates are
combined, and an old label is reused even when the new watch occurred on
another date. The Calendar-derived manual CSV and the original
`movie_rt_notes.txt` are both first-class rating sources. The legacy parser
preserves multiline reviews from the original `title rating; review` format,
then uses the previously generated RT matched title and release year in
`movie_rt_scores_detailed.csv`; it does not re-scrape.

Legacy ratings auto-join only when cleaned identity-bearing titles match or the
current and legacy title/year identities agree. Broad-normalization-only
matches and title/year conflicts are quarantined unless the exact legacy title,
candidate normalization, canonical title, and year appear in
`LEGACY_MOVIE_ADJUDICATIONS`. That narrow allowlist records user-confirmed
matches without making the general matching rule more permissive. Legacy-rated
titles absent from Calendar history are retained in `movies_all.csv` with
unknown watch dates. Confirmed AMC-generated events count as completed
watches; cancelled revisions and the Best Picture Showcase marathon do not.
Same-day split sessions totaling at least 75 minutes count as completed.

The current user-confirmed legacy adjudications canonicalize these Calendar
variants:

- `American Pickle` → `An American Pickle` (2020)
- `The Heat` → `Heat` (1995)
- `Boondock Saints` → `The Boondock Saints` (1999)
- `don’t let the devil know you’re dead` →
  `Before the Devil Knows You're Dead` (2007)
- `the lover and the gentleman` → `An Officer and a Gentleman` (1982)
- `League of Extraordinary Gentlemen` →
  `The League of Extraordinary Gentlemen` (2003)

Title spelling corrections remain in the code. User-confirmed completion dates
and notes belong in the ignored local file
`data/rating_export_manual_movie_completions.json`, not in version control. It
contains a JSON list with `title`, ISO `date`, and `reason` fields. Entries are
recorded in each ignored run manifest for local auditing.

Books use the existing training export and holdout as prior history.
Calendar `finished book`/`finished audiobook` markers are authoritative. Play
Books supplies title, author, shelf, filename, and additional finished titles;
a Play `unfinished` flag cannot undo a Calendar completion. A completion close
to an existing completion is de-duplicated, while a later completion remains a
reread. Export-specific book aliases and preferred display titles belong in
the ignored local file `data/rating_export_book_title_overrides.json`. Its
three string-to-string maps are `aliases`, `prefix_aliases`, and
`preferred_titles`.

## External public ratings

This command deliberately does not fetch public ratings, so the personal
labeling queues remain blind.

The prior movie workflow attached Rotten Tomatoes and IMDb fields in
`Self_Tracking/movie_analysis/analysis_core/calendar_movie_scores.py`. It used
Rotten Tomatoes search-page parsing plus the IMDb datasets and explicit match
overrides; it was a local Python process, not a required Google Sheets step.

For books, the most complete current reference implementation is:

`drafts/books_and_ratings/09_requested_analysis/scripts/build_byrne_rating_dataset.py`
in the separate analysis repository.

It produced the Byrne request CSV mentioned in the project. It resolves Amazon
links and optionally parses Amazon product rating widgets, queries Goodreads
autocomplete, and uses Google Books/Open Library catalog metadata with a
persistent cache and explicit adjudications. It is a direct HTTP/API Python
pipeline—not an OpenAI API call and not a Google Sheets function. It is
Byrne-input-specific today, so adapting it to these new book rows should be a
separate, reviewable enrichment step after personal ratings are recorded.
