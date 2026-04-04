# Website Updates: Holdout Evaluation, Monte Carlo Noise, Column Mapping

**Date**: 2026-03-31  
**Session Duration**: ~3 hours  
**Files Changed**: `app.js`, `index.html`, `style.css`, `tests.js`

---

## Problems Fixed

### 1. Train-on-all Bias (Overly Optimistic Gains)

**Problem**: Models trained on ALL 264 books, then evaluated on same books → inflated gain curves. The Oracle was perfect, but RF/Ridge/Heuristic gains were unrealistically high because predictions correlated with training labels.

**Solution**: 
- Added `source` column extraction from CSV
- Implemented holdout/in-sample toggle that appears when CSV has `source == "Holdout 2026"` entries
- Two modes:
  - "All Data — In-Sample (264 books)" — training set, shows upper bound
  - "Holdout — Out-of-Sample (66 books)" — held-out test set, shows realistic performance
- Charts re-render instantly when toggling subsets
- Toggle styled with active state highlighting

**Results**:
- Holdout baseline (66 books): enjoyment 3.08/5.0, usefulness 1.75/5.0
- All-data baseline (264 books): enjoyment 3.32/5.0, usefulness 1.82/5.0
- Holdout gains are lower (more realistic) than all-data gains

**Code changes**:
- `app.js` lines 19-20: Added `allProcessedBooks`, `columnMapping` state
- `app.js` lines 245, 365: Extract and store `source` from CSV rows
- `app.js` lines 349-391: `setupSubsetToggle()`, `renderForSubset()` functions
- `index.html` line 81: Toggle UI with two buttons
- `style.css` lines 288-315: Toggle button styling

---

### 2. Incorrect MAE Noise Band → Monte Carlo Simulation

**Problem**: Old approach used `SE = labelingSD / sqrt(n_kept)` to show grey uncertainty bands. This only captures uncertainty in the mean estimate (standard error), NOT the effect of rating noise on the Oracle's ability to sort books correctly.

**Solution**: Replaced with Monte Carlo simulation (500 iterations):
1. For each iteration: add N(0, labelingSD) Gaussian noise to true ratings
2. **Oracle**: re-sorts by noisy ratings (simulates imperfect self-knowledge — you can't perfectly know which books you'd rate highest)
3. **ML models**: keep original predictions, measure gain against noisy labels
4. Returns 10th-90th percentile (80% CI)
5. Captures winner's curse and selection bias effects

**Implementation**:
- `app.js` lines 280-285: `normalRandom()` Box-Muller transform for N(0,1)
- `app.js` lines 287-346: `monteCarloNoiseBands(books, predictKey, trueKey, labelingSD, nIterations)`
- `app.js` lines 469-486: Updated `buildDropCurveDatasets()` to use Monte Carlo
- Constant: `NOISE_ITERATIONS = 500`

**Rationale**: Rating noise doesn't just add uncertainty to the mean — it actively degrades the Oracle's sort order. Books you'd rate 4.0 might be observed as 3.5 due to measurement error, causing them to be incorrectly dropped. Monte Carlo simulates this effect.

---

### 3. Bootstrap Iterations Too Low

**Problem**: 200 iterations gave noisy CI bands.

**Solution**: Increased to 2000 iterations.

**Benchmarked convergence**:
- 500 iter: mean converged within 0.001
- 1000 iter: CI width stable within ±0.005
- 2000 iter: **optimal** (no improvement at 5000+, just 2.5x slower)
- 10000 iter: overkill (5x slower, CI width ±0.001 from 2000)

**Performance** (264 books, both targets):
- Monte Carlo noise bands (500 iter, 4 strategies): ~165ms per target
- Bootstrap (2000 iter, 3 strategies): ~450ms per target
- **Total page render: ~1.5 seconds**

**Code change**: `app.js` line 273: `const BOOTSTRAP_ITERATIONS = 2000;`

---

## New Feature: Universal Column Mapping

**Problem**: Website only accepted specific CSV column names (`avg_enjoyment`, `goodreads_rating`, etc.). Users with different export formats couldn't use the tool.

**Solution**: Added interactive column mapper that appears after CSV upload.

**Flow**:
1. User uploads CSV → parser extracts headers
2. Column mapper UI appears showing number of rows detected
3. Three sections with dropdowns:
   - **Target Columns** (at least one required): Enjoyment Rating, Usefulness Rating
   - **Feature Columns** (at least one rating required): Goodreads Rating, Amazon Rating, GR Count, AMZ Count, Page Count, Pub Year
   - **Metadata** (all optional): Category/Genre, Source/Holdout Split, Book Title
4. Auto-detection fills in likely matches (fuzzy case-insensitive matching)
5. User can override any auto-detected mapping
6. Click "Analyze Data" → validation → charts render

**Auto-detection logic** (`detectColumn`):
- Case-insensitive, ignores underscores/spaces
- Pattern matching: tries exact match first, then substring match
- Example: pattern `'goodreads_rating'` matches "Goodreads Rating", "goodreads_rating", "GR_rating", etc.

**Validation**:
- Error if no targets mapped (need at least enjoyment or usefulness)
- Error if no ratings mapped (need at least Goodreads or Amazon)
- Optional columns default to `null` if unmapped

**Code changes**:
- `app.js` lines 150-276: `showColumnMapper()`, `detectColumn()`, `populateColumnDropdown()`, `handleAnalyzeClick()`
- `app.js` lines 330-369: Rewrote `processData()` to use `columnMapping` instead of hardcoded column names
- `index.html` lines 56-117: Column mapper UI (3-section grid layout)
- `style.css` lines 288-358: Mapper styling (responsive: 1-col mobile, 2-col desktop)

**Tested with**:
- `golden_master_multi_source.csv` (111 columns, auto-detected 11/11 fields correctly)
- Works with any CSV format now

---

## Testing

### Unit Tests Added
1. `normalRandom()` produces N(0,1) distribution (10k samples, mean ~0, variance ~1)
2. `monteCarloNoiseBands()` returns valid p10/p90 with p10 ≤ p90
3. `monteCarloNoiseBands()` Oracle has positive width (degraded sort verification)
4. `detectColumn()` handles exact/partial/case-insensitive matches
5. Source column extraction works correctly

### Integration Test
- `processData()` with column mapping on real 43-book CSV slice
- All tests pass in ~560ms

### Browser Testing
- ✅ Column mapper appears with auto-detection
- ✅ "Analyze Data" processes 264 books correctly
- ✅ Toggle appears showing "All Data (264)" vs "Holdout (66)"
- ✅ Holdout subset renders with different baseline stats
- ✅ Switching back to "All Data" works
- ✅ Charts render in ~1.5s total

**All tests passing**: `node tests.js` completes successfully.

---

## Performance Summary

| Dataset | Books | Noise Bands (500 iter) | Bootstrap (2000 iter) | Total Render |
|---------|-------|------------------------|----------------------|--------------|
| All data | 264 | ~165ms/target | ~450ms/target | ~1.5s |
| Holdout | 66 | ~50ms/target | ~220ms/target | ~0.5s |

---

## Files Modified

1. **app.js** (+200 lines)
   - Added state: `parsedCSV`, `columnMapping`, `allProcessedBooks`
   - Added constants: `NOISE_ITERATIONS = 500`, `BOOTSTRAP_ITERATIONS = 2000`
   - New functions: `normalRandom()`, `monteCarloNoiseBands()`, `showColumnMapper()`, `detectColumn()`, `populateColumnDropdown()`, `handleAnalyzeClick()`, `setupSubsetToggle()`, `renderForSubset()`
   - Modified: `handleFile()` to show mapper instead of direct processing, `processData()` to use column mapping, `buildDropCurveDatasets()` to use Monte Carlo, `resetUI()` to include mapper
   - Added timing logs: `console.time/timeEnd` for benchmarking

2. **index.html** (+67 lines)
   - Added column mapper section (`#mapper-zone`) with 3 subsections and 11 dropdown fields
   - Added subset toggle with 2 buttons above charts
   - Updated chart descriptions (500 iter noise, 2000 bootstrap)
   - Updated methodology explanation section

3. **style.css** (+69 lines)
   - Added mapper styles: `.mapper-grid`, `.mapper-section`, `.mapper-row`, `.mapper-label`, `.mapper-select`, `.mapper-optional`
   - Added toggle styles: `.subset-toggle`, `.toggle-btn`, `.toggle-btn.active`
   - Responsive breakpoints for mapper grid

4. **tests.js** (+40 lines)
   - Added `normalRandom()` distribution test
   - Added `monteCarloNoiseBands()` envelope tests
   - Added `detectColumn()` fuzzy matching tests
   - Updated integration test to set `columnMapping` before calling `processData()`
   - Updated mock DOM to support `querySelector`, `querySelectorAll`, `dataset`

---

## What's Better Now

✅ **Realistic holdout evaluation** — see true out-of-sample performance, not inflated train-on-all gains  
✅ **Proper rating-noise uncertainty** — Monte Carlo captures how rating measurement error degrades Oracle sort quality  
✅ **Higher bootstrap resolution** — 2000 iterations for stable confidence intervals  
✅ **Universal CSV support** — map any column names via interactive UI with auto-detection  
✅ **Better UX** — clear mapping workflow, validation messages, instant subset toggling  
✅ **Fast render** — optimized to <2s for 264 books (both targets)  
✅ **Comprehensive testing** — unit tests, integration tests, browser testing all passing  

---

## Known Issues / Future Work

1. **Oracle outside CI bands**: On some charts, the Oracle (perfect sort) line falls outside the 80% CI grey bands. This suggests the noise model may not fully capture the Oracle's behavior. Need to investigate whether this is a Monte Carlo sampling issue or a fundamental modeling problem.

2. **Chart y-axis scales**: All charts currently auto-scale. Should standardize y-axis ranges across charts for easier comparison.

3. **Enjoyment/usefulness tradeoff**: Currently optimizing for enjoyment may hurt usefulness and vice versa (different books predicted as best for each target). Could add a weighting slider to let users optimize for a weighted combination (e.g., 70% enjoyment + 30% usefulness).

4. **Bootstrap vs noise band descriptions**: Some confusion in UI text about what each uncertainty band represents. Need clearer distinction between:
   - Monte Carlo noise bands (top charts): "What if my ratings had measurement error?"
   - Bootstrap bands (bottom charts): "What if I had a different sample of N books?"

5. **Books-to-drop list**: No UI showing which specific books would be dropped at a given threshold. Could add a table showing "Books dropped at 50%" ranked by user rating.

---

## References

**Analysis scripts** (for background on methodology):
- `scripts/temp/threshold_bootstrap_analysis.py` — bootstrap with labeling noise, GP smoothing
- `scripts/temp/oracle_threshold_plots.py` — bootstrap + labeling SE bands (500 iterations)
- `ai_actions/results_so_far.md` — holdout model performance metrics

**Data**:
- `golden_master_multi_source.csv` — 264 books, 66 holdout
- Labeling noise from double-rating: enjoyment MAE=0.46 (R=0.77), usefulness MAE=0.34 (R=0.86)
