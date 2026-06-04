# Combined Highlight Coverage Metrics

Inputs:
- Drive DOCX root: `/Users/clarkbenham/side_projects/data/Play_Books_Notes-20260531T040817Z`
- Tablet markdown root: `/Users/clarkbenham/side_projects/data/play_books_highlights_full_backup_20260530_235315`
- Curated notes root: `/Users/clarkbenham/side_projects/make_structured_summaries/data/my_highlights`
- Source documents parsed: 285
- Summary files scanned: 14
- Summary files evaluated with matched highlights: 10
- Summary files without matched highlights: 4
- Matched base books: 4

Metric notes:
- `covered` means the best local summary/chunk window crossed the deterministic lexical threshold.
- `partial` means there was material overlap, but below the covered threshold.
- Tablet markdown exports do not carry color/page metadata; tablet-only items are counted as uncolored.

## Summary Coverage By Variant

| variant | summaries | highlights | covered | partial+covered | avg score |
| --- | --- | --- | --- | --- | --- |
| current | 4 | 1889 | 24.8% | 84.3% | 0.315 |
| pre_bluegreen_revision | 2 | 923 | 15.1% | 80.7% | 0.291 |
| pre_consequence_ranking_revision | 1 | 816 | 20.5% | 84.3% | 0.306 |
| pre_details_first_revision | 2 | 923 | 11.8% | 74.0% | 0.273 |
| pre_prompt_revision | 1 | 107 | 37.4% | 88.8% | 0.367 |

## Prompt Variant Comparable Subset

This subset only includes books with more than one saved summary variant.

| variant | summaries | highlights | covered | partial+covered | avg score |
| --- | --- | --- | --- | --- | --- |
| current | 2 | 923 | 20.4% | 82.6% | 0.306 |
| pre_bluegreen_revision | 2 | 923 | 15.1% | 80.7% | 0.291 |
| pre_consequence_ranking_revision | 1 | 816 | 20.5% | 84.3% | 0.306 |
| pre_details_first_revision | 2 | 923 | 11.8% | 74.0% | 0.273 |
| pre_prompt_revision | 1 | 107 | 37.4% | 88.8% | 0.367 |

## Current Summary Coverage By Color

| color | highlights | covered | partial+covered | avg score |
| --- | --- | --- | --- | --- |
| blue | 45 | 20.0% | 80.0% | 0.304 |
| green | 104 | 26.9% | 89.4% | 0.320 |
| red | 121 | 36.4% | 89.3% | 0.352 |
| uncolored | 53 | 18.9% | 60.4% | 0.263 |
| yellow | 1566 | 24.1% | 84.5% | 0.314 |

## Current Summary Coverage By Source

| source | highlights | covered | partial+covered | avg score |
| --- | --- | --- | --- | --- |
| curated_notes | 1729 | 24.7% | 85.0% | 0.315 |
| drive_docx | 1836 | 24.9% | 85.0% | 0.317 |
| tablet_md | 182 | 18.7% | 78.6% | 0.293 |

## Book And Variant Detail

| book | variant | highlights | covered | partial+covered | avg score |
| --- | --- | --- | --- | --- | --- |
| master-of-the-senate-master-of-the-senate | current | 658 | 29.6% | 87.1% | 0.325 |
| the-doomsday-machine-the-doomsday-machine | current | 308 | 27.6% | 83.8% | 0.323 |
| the-goal-a-process-of-ongoing-improvement | current | 107 | 29.0% | 85.0% | 0.347 |
| the-goal-a-process-of-ongoing-improvement | pre_bluegreen_revision | 107 | 27.1% | 86.9% | 0.339 |
| the-goal-a-process-of-ongoing-improvement | pre_details_first_revision | 107 | 32.7% | 88.8% | 0.338 |
| the-goal-a-process-of-ongoing-improvement | pre_prompt_revision | 107 | 37.4% | 88.8% | 0.367 |
| the-path-to-power-the-years-of-lyndon-johnson-i | current | 816 | 19.2% | 82.2% | 0.301 |
| the-path-to-power-the-years-of-lyndon-johnson-i | pre_bluegreen_revision | 816 | 13.5% | 79.9% | 0.285 |
| the-path-to-power-the-years-of-lyndon-johnson-i | pre_consequence_ranking_revision | 816 | 20.5% | 84.3% | 0.306 |
| the-path-to-power-the-years-of-lyndon-johnson-i | pre_details_first_revision | 816 | 9.1% | 72.1% | 0.264 |

## Chunk Extraction Coverage

| book | highlights | covered | partial+covered | avg score |
| --- | --- | --- | --- | --- |
| master-of-the-senate-master-of-the-senate | 658 | 20.5% | 81.6% | 0.303 |
| the-doomsday-machine-the-doomsday-machine | 308 | 23.4% | 78.9% | 0.298 |
| the-goal-a-process-of-ongoing-improvement | 107 | 40.2% | 84.1% | 0.376 |
| the-path-to-power-the-years-of-lyndon-johnson-i | 816 | 11.0% | 75.0% | 0.273 |
