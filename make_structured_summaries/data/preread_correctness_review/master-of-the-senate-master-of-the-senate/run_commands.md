# Run Commands

Dry-run prompts only:

```bash
cd /Users/clarkbenham/side_projects/make_structured_summaries
pdm run python scripts/preread_book.py \
  --book-id master-of-the-senate-master-of-the-senate \
  --dry-run \
  --force
```

Real pre-read run with the default stronger model path:

```bash
cd /Users/clarkbenham/side_projects/make_structured_summaries
pdm run python scripts/preread_book.py \
  --book-id master-of-the-senate-master-of-the-senate \
  --force
```

Build left/right HTML against pre-read outputs:

```bash
cd /Users/clarkbenham/side_projects/make_structured_summaries
pdm run python scripts/build_highlight_review_html.py \
  --chunk-root data/preread_chunk_notes \
  --summary-dir data/preread_summaries \
  --out-dir data/eval_reviews/preread_highlight_review_html
```
