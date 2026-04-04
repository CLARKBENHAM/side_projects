# Website Implementation Tasks

## Context

I'm working on a book analysis website at `/Users/clarkbenham/side_projects/ai_books_tracking/website/`. It lets users upload a CSV of book ratings and shows how average enjoyment/usefulness improves if you drop the bottom X% of books according to various prediction strategies (Oracle, Ridge, RF, Simple Heuristic).
The datascience for this project is done in `/Users/clarkbenham/side_projects/ai_books_tracking/` there's everything_so_far for info (likely not needed).

**Recent changes** (see `ai_actions_log/2026-03-31_holdout_noise_mapper.md`):
- Added holdout/in-sample toggle
- Replaced naive SE bands with Monte Carlo noise simulation (500 iter)
- Increased bootstrap from 200 → 2000 iterations
- Added universal column mapping UI

**Key files**:
- `app.js` — main app logic, chart rendering, drop curve calculations, bootstrap
- `index.html` — page structure
- `style.css` — styling
- `tests.js` — unit + integration tests
- `models.json` — exported ML models (881KB)
- `golden_master_multi_source.csv` — 264 books (66 holdout)

---

## Tasks to Implement

### 1. Standardize Y-Axis Scales Across All Charts

**Problem**: Charts currently auto-scale, making visual comparison difficult.

**Task**:
- Determine appropriate fixed y-axis range by looking at actual data across all 4 charts (2 gain charts + 2 bootstrap charts)
- Set consistent y-axis min/max for all enjoyment charts
- Set consistent y-axis min/max for all usefulness charts
- Ensure Oracle (perfect sort) and all ML strategies fit within the chosen range

**Test**: Load CSV and verify all 4 charts have same y-axis scale within each target type.

---

### 2. Add Enjoyment/Usefulness Weighting Slider

**Problem**: Books best predicted for enjoyment ≠ books best predicted for usefulness. Optimizing one target anti-correlates with the other.

**Task**: Add a weighting slider above the drop-curve charts:
- UI: Horizontal slider with label "Enjoyment ← → Usefulness"
- Range: 0.0 (100% usefulness) to 1.0 (100% enjoyment)
- Default: 0.5 (equal weight)
- Display current weights as percentages: "50% Enjoyment | 50% Usefulness"

**Weighting logic**:
```javascript
// For each book, compute weighted prediction
weightedPred = (enjoymentWeight * pred.ridge_full_enjoy) +
               ((1 - enjoymentWeight) * pred.ridge_full_useful)

// Use this weighted prediction for sorting in BOTH gain charts
// Then measure enjoyment gain on enjoyment chart, usefulness gain on usefulness chart
```

**Apply to all strategies**:
- Ridge: `ridge_full_enjoy` and `ridge_full_useful`
- RF: `rf_enjoy` and `rf_useful`
- Heuristic: `external_sum` (same for both, no weighting needed)
- Oracle: weighted combination of `trueEnjoy` and `trueUseful`

**Interaction**: When slider changes, re-render both gain charts using the new weighted sort order.

**Bootstrap**: Also apply weighting to bootstrap uncertainty charts — use the same weighted prediction for sorting during bootstrap resampling.

**Test**:
- Slider at 1.0 (100% enjoyment) → enjoyment gains should be maximized, usefulness gains lower
- Slider at 0.0 (100% usefulness) → usefulness gains maximized, enjoyment gains lower
- Slider at 0.5 (balanced) → compromise on both

---

### 3. Add "Books That Would Be Dropped" Section

**Problem**: Users can't see which specific books would be cut at a given drop percentage.

**Task**: Add a new section below the charts:
- Title: "Books Dropped at X%"
- UI: Dropdown to select drop percentage (0%, 10%, 20%, ..., 90%)
- Display: Table showing books that would be dropped at that threshold
- Columns: Title, True Enjoyment, True Usefulness, Weighted Prediction, Category
- Sort by: User rating (using the same enjoyment/usefulness weighting from the slider)
- Limit: Show top 20 books that would be dropped (worst predictions)

**Interaction**:
- Dropdown selection triggers table re-render
- Table updates when weighting slider changes (different books may be dropped)

**Styling**: Use same glass-card styling as other sections.

**Test**:
- Select 50% drop → should show books with lowest weighted predictions
- Change weighting slider → books list should update

---

### 4. Fix Chart Descriptions and Methodology Text

**Problem**: Current text is inconsistent about what Monte Carlo vs Bootstrap represents.

**Task**: Update descriptions to be clearer:

**Top charts (Gain by Drop %)**:
> "Expected rating gain if you drop the bottom X% of books according to each strategy. **Grey bands show 80% CI from Monte Carlo simulation (500 iterations)**: we add Gaussian noise (MAE 0.46 for enjoyment, 0.34 for usefulness) to your ratings to simulate measurement error, then measure how much the Oracle's sort order degrades. This shows the uncertainty in gains due to rating noise."

**Bottom charts (Bootstrap Uncertainty)**:
> "**Bootstrap resampling (2,000 iterations)**: we sample your N books with replacement, re-sort by predicted score, and compute gain curves. The 80% CI shows sampling uncertainty: would a different set of N books give similar results?"

**Methodology section**: Update to clarify the distinction:
> **Monte Carlo noise bands** (grey, top charts): Simulates what happens if your 1-5 ratings have measurement error. The Oracle's sort order degrades because you can't perfectly know which books you'd rate highest. ML models are also affected because the "true" labels are noisy.

> **Bootstrap bands** (colored, bottom charts): Simulates sampling uncertainty. If you had a different set of N books from the same distribution, would you see the same gain curve?

**Test**: Read descriptions and verify they're clear and accurate.

---

## Testing Requirements

After implementing:
1. Run `node tests.js` — all tests must pass
2. Load `golden_master_multi_source.csv` in browser
3. Verify weighting slider works:
   - 100% enjoyment → enjoyment gains high, usefulness gains low
   - 100% usefulness → usefulness gains high, enjoyment gains low
   - 50% balanced → compromise on both
4. Verify books-dropped table shows correct books
5. Verify y-axis scales are consistent across charts
6. Verify descriptions are clear and accurate

---

## Code Style Notes

From `~/.claude/CLAUDE.md`:
- Use comments judiciously (only if code doesn't explain itself)
- No emojis unless contextually appropriate
- Don't suppress errors without good reason
- Add types where helpful
- Always run tests after changes: `node tests.js`
- Don't add try-catch wrappers around things that should fail the script

---

## Success Criteria

✅ All y-axes standardized (easier visual comparison)
✅ Weighting slider allows exploring enjoyment/usefulness tradeoffs
✅ Bootstrap uncertainty uses weighted predictions
✅ Users can see which specific books would be dropped
✅ Chart descriptions are clear and accurate
✅ All tests pass
✅ Website renders in <2 seconds with full CSV
