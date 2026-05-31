# Book WPM Calendar Analysis

- Finished-book rows: 225
- Books with matched reading time: 204
- Books with page count estimate: 73
- Books with WPM estimate: 70
- Reliable first-read WPM rows: 52
- Strict high-confidence first-read WPM rows: 30
- Parsed classified calendar rows: 2498
- Parsed Play Books note files with highlights: 122
- WPM assumption: pages * 275 words/page / reading minutes.
- Audiobook sessions consume words at 350 WPM before estimating print-reading WPM.
- `observed_total_wpm` keeps the raw pages/time rate; primary `wpm` is audiobook-adjusted when audio time is present.
- Page counts come from existing book metadata / Google Books fields; Play Books max page is retained as `notes_max_page` audit data, not used as total pages.
- Highlighted words count only highlighted passage text; `my_note_words` counts user-written notes attached to highlights.
- Recurrences are expanded for common DAILY/WEEKLY/MONTHLY RRULEs; exception handling is approximate.

## WPM Distribution

- Median WPM: 387.1
- P5/P95 WPM: 192.7 / 1235.1

## Reliable First Reads

- Median WPM: 326.9
- P5/P95 WPM: 198.3 / 650.1
- Years covered: 2020-2025
- Reliable means medium-or-high word-count confidence and medium-or-high calendar confidence, first-read only.

## Year Trend

- Median-WPM trend slope: 35.5 WPM/year.
- First/last reliable year medians: 305.6 / 478.2.

## Highlighting Relationship

- Books in model: 52
- Correlation between log highlight density and WPM: -0.03
- Observed WPM change going from p75 to p25 highlight density: 9.6
- This is observational correlation, not a causal estimate.

## Time vs Pages Outliers

- The Screwtape Letters: 160 pages, 0.2h, 0.16x expected time (faster_than_page_count_predicts)
- The Checklist Manifesto: 209 pages, 0.5h, 0.23x expected time (faster_than_page_count_predicts)
- what is man and other essays: 375 pages, 1.2h, 0.28x expected time (faster_than_page_count_predicts)
- DFW_TV.pdf: 407 pages, 1.5h, 0.30x expected time (faster_than_page_count_predicts)
- Jude the Obscure: 240 pages, 8.5h, 3.27x expected time (slower_than_page_count_predicts)
- bobby-fischer-teaches-chess.pdf: 334 pages, 1.2h, 0.32x expected time (faster_than_page_count_predicts)
- Designing Machine Learning Systems: 386 pages, 1.5h, 0.32x expected time (faster_than_page_count_predicts)
- Arnold Schwarzenegger: 256 pages, 1.0h, 0.36x expected time (faster_than_page_count_predicts)
- Life and Fate: 487 pages, 17.2h, 2.74x expected time (slower_than_page_count_predicts)
- Jeeves Stories: 263 pages, 7.9h, 2.71x expected time (slower_than_page_count_predicts)

## Top Unmatched Calendar Refs

- ml: 41.9h
- s: 29.9h
- Conrad: 16.5h
- (generic): 14.2h
- sp: 14.1h
- Python: 13.5h
- mlfp: 13.2h
- ladr: 12.2h
- 5ebgo: 11.8h
- hotesp: 10.5h
- sim: 9.2h
- ml papers: 8.8h
- Club: Software Design for Flexibility: 8.0h
- TLP: 6.5h
- sc: 5.8h
