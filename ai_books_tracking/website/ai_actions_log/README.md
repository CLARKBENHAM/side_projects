# Website Development Log

This directory tracks major changes and planned work for the book analysis website.

---

## Session: 2026-03-31 (Completed)

**File**: `2026-03-31_holdout_noise_mapper.md`

**What was done**:
- ✅ Fixed train-on-all bias by adding holdout/in-sample toggle
- ✅ Replaced naive SE bands with Monte Carlo noise simulation (500 iter)
- ✅ Increased bootstrap from 200 → 2000 iterations for stable CIs
- ✅ Added universal column mapping UI with auto-detection
- ✅ All tests passing, browser testing complete
- ✅ Render time: ~1.5s for 264 books

---

## Next Session: Website Implementation

**File**: `NEXT_SESSION_PROMPT_WEBSITE.md`

**Tasks**:
1. Standardize y-axis scales across all charts
2. Add enjoyment/usefulness weighting slider
3. Apply weighting to bootstrap intervals
4. Add "Books That Would Be Dropped" section
5. Fix chart descriptions for clarity

**Priority**: High (concrete tasks, can implement immediately)

---

## Next Session: Stats Discussion

**File**: `NEXT_SESSION_PROMPT_STATS.md`

**Questions to resolve**:
1. Why is Oracle outside the 80% CI? (Bug or feature?)
2. Why are grey bands skinnier than bootstrap bands?
3. Is Monte Carlo noise the right approach? (Alternatives?)
4. What should grey bands represent? (Sort degradation vs measurement uncertainty?)
5. Should we retrain models on bootstrap samples?
6. How to convey "if labels had 50% less noise, gains would be X better"?

**Priority**: High (need decisions before updating methodology)

**Recommended approach**: 
- Start with stats discussion to understand what we're trying to measure
- Then implement website changes with correct methodology
- This avoids implementing features that may need to be redesigned

---

## How to Use These Prompts

**For a new Claude session**:

1. **Start with stats discussion** (recommended first):
   ```
   Read ai_books_tracking/website/ai_actions_log/NEXT_SESSION_PROMPT_STATS.md
   
   Let's discuss these statistical questions about the uncertainty bands.
   I want to make sure we're showing users the right information before 
   implementing the website features.
   ```

2. **Then do website implementation**:
   ```
   Read ai_books_tracking/website/ai_actions_log/NEXT_SESSION_PROMPT_WEBSITE.md
   
   Please implement these website features. The key changes are:
   - Standardize y-axis scales
   - Add enjoyment/usefulness weighting slider  
   - Add books-to-drop section
   - Fix chart descriptions
   
   After implementation, run tests and verify in browser with the full CSV.
   ```

3. **Or combine both** (longer session):
   ```
   Read ai_books_tracking/website/ai_actions_log/NEXT_SESSION_PROMPT_STATS.md
   and ai_books_tracking/website/ai_actions_log/NEXT_SESSION_PROMPT_WEBSITE.md
   
   First, let's discuss the stats questions to decide on methodology.
   After we resolve those, implement the website features with the 
   agreed-upon approach.
   ```

---

## File Structure

```
ai_books_tracking/website/
├── ai_actions_log/
│   ├── README.md                              (this file)
│   ├── 2026-03-31_holdout_noise_mapper.md    (completed work)
│   ├── NEXT_SESSION_PROMPT_WEBSITE.md        (implementation tasks)
│   └── NEXT_SESSION_PROMPT_STATS.md          (methodology discussion)
├── app.js                                     (main logic)
├── index.html                                 (page structure)
├── style.css                                  (styling)
├── tests.js                                   (tests)
├── predict.js                                 (ML inference)
└── models.json                                (exported models)
```

---

## Testing

Always run after changes:
```bash
cd /Users/clarkbenham/side_projects/ai_books_tracking/website
node tests.js
```

**Browser testing**:
```bash
python3 -m http.server 8765
# Open http://localhost:8765/ in Chrome
# Upload golden_master_multi_source.csv
# Verify all features work
```

---

## Key Insights

**From double-rating validation** (208 books rated twice):
- Enjoyment: R=0.77, MAE=0.46 → about 23% of variance is measurement error
- Usefulness: R=0.86, MAE=0.34 → about 14% of variance is measurement error

**Prediction ceiling** (from test-retest reliability):
- Enjoyment: max achievable R = 0.933, current models at 17% of ceiling
- Usefulness: max achievable R = 0.962, current models at 36% of ceiling

**Implication**: Rating noise is a major bottleneck. Reducing label noise by 50% could substantially improve model performance. This motivates answering: "How much better would filtering be with cleaner labels?"

---

## Contact / Feedback

Questions about methodology, implementation choices, or results? 
Update this log with notes, or start a new session referencing these files.
