# Structured Book Summaries

This project turns books you already own into structured study notes instead of chapter recaps.

What exists now:

- Google Play Takeout cataloging.
- Text extraction for `epub`, `pdf`, `html`, and `txt`.
- Two-pass summary prompts designed around mental models, legibility, power dynamics, and felt experience.
- Optional Goodreads metadata enrichment by reusing the existing scraper logic in [`/Users/clarkbenham/side_projects/ai_books_tracking/goodreads_ratings.py`](/Users/clarkbenham/side_projects/ai_books_tracking/goodreads_ratings.py).
- Highlight-to-Anki conversion and highlight-vs-summary evaluation prompts.

What is intentionally not implemented:

- LibGen download automation. That would be a piracy workflow. This project is scoped to your local library plus lawful metadata enrichment. If you want another acquisition source later, add a legitimate connector such as Open Library or Project Gutenberg.

## Why the prompts are different

The core failure mode of naive summaries is that they retell events. This pipeline instead asks:

- What mental models does the book install?
- What does it make newly legible about institutions, status, persuasion, or human behavior?
- What does the situation feel like from inside?
- Which examples are actually load-bearing, instead of exhaustive?
- Where should the reader push back or verify claims?

That is why chunk prompts extract conceptual payload first, and synthesis is explicitly forbidden from becoming chapter-by-chapter notes.

## Typical workflow

1. Build a catalog from Google Play Takeout:

```bash
cd /Users/clarkbenham/side_projects/make_structured_summaries
pdm run -s python scripts/build_takeout_catalog.py
```

2. Optionally enrich that catalog with Goodreads metadata:

```bash
pdm run -s python scripts/build_takeout_catalog.py --enrich-goodreads
```

3. Dry-run a book to generate extraction and prompt artifacts:

```bash
pdm run -s python scripts/summarize_book.py \
  --catalog-csv data/takeout_catalog.csv \
  --book-id build-a-large-language-model-from-scratch \
  --dry-run
```

4. Run a real summary with Gemini on chunks and Claude on synthesis:

```bash
pdm run -s python scripts/summarize_book.py \
  --catalog-csv data/takeout_catalog.csv \
  --book-id build-a-large-language-model-from-scratch \
  --chunk-backend gemini \
  --chunk-model gemini-3.1-pro-preview \
  --synthesis-backend claude \
  --synthesis-model sonnet
```

5. For unread books, build a shorter pre-read brief instead of a full summary:

```bash
pdm run -s python scripts/preread_book.py \
  --catalog-csv data/takeout_catalog.csv \
  --book-id build-a-large-language-model-from-scratch \
  --chunk-backend claude \
  --chunk-model sonnet \
  --synthesis-backend claude \
  --synthesis-model sonnet
```

This writes chunk-level pre-read extraction to `data/preread_chunk_notes/` and the
final brief to `data/preread_summaries/`. It is calibrated for red/yellow-style
importance and concise pre-scan reading, not exhaustive notes.

6. Convert highlights into Anki cards:

```bash
pdm run -s python scripts/convert_highlights_to_anki.py \
  --input-file /path/to/highlights.md \
  --title "Seeing Like a State"
```

7. Compare a generated summary against your highlights:

```bash
pdm run -s python scripts/temp/check_summary_against_highlights.py \
  --summary-file data/summaries/seeing-like-a-state.md \
  --highlights-file /path/to/highlights.md
```

## Directory layout

- `structured_summaries/`: reusable code.
- `scripts/`: command entrypoints.
- `scripts/temp/`: user-assisted evaluation helpers.
- `data/`: generated catalogs, extracted text, and summaries.
- `ai_actions/`: notes and reports.

## Notes on backends

- `gemini` is wired using the local CLI pattern you already use elsewhere.
- `claude` is wired through `claude -p`.
- In your current setup, prefer `pdm run -s ...` while the `side_projects` conda env is activated. The `-s` flag exposes conda `site-packages`, including `pytest`, `black`, `pandas`, and `requests`.
- Dry-run mode works without spending tokens and is the fastest way to inspect prompts.

## Best next extensions

- Add a lawful acquisition source.
- Export Kindle/Readwise highlights into the same evaluation path.
- Accumulate a summary-vs-highlights evaluation set before considering any fine-tuning.
