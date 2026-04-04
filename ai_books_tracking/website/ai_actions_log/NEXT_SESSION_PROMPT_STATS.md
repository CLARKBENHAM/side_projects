# Statistics & Methodology Discussion

## Context

Working on book analysis website showing how gains improve when dropping low-rated books. Recent changes added Monte Carlo noise simulation to show uncertainty from rating measurement error.

**Current approach** (Monte Carlo noise bands):
```python
for iteration in range(500):
    noisy_ratings = true_ratings + Normal(0, labeling_SD)
    # Oracle: sort by noisy ratings (degraded sort)
    # ML models: sort by predictions, measure gain vs noisy labels
    compute_gain()
# Return 10th-90th percentile as 80% CI
```

**Rationale**: Rating noise doesn't just add uncertainty to the mean — it actively degrades the Oracle's ability to sort correctly. Books truly rated 4.0 might be observed as 3.5, causing incorrect drops.

---

## Questions to Resolve

### Q1: Why is Oracle Outside the 80% CI?

**Observation**: On the "Gain by Drop %" charts, the Oracle (perfect sort) line falls **outside** the grey 80% CI noise bands. This seems wrong — the Oracle should be at the center of its own uncertainty band.

**Possible explanations**:
1. **Monte Carlo sampling error**: 500 iterations not enough to converge?
2. **Bias in the simulation**: Adding noise to true ratings and then sorting by noisy ratings may not correctly model the Oracle's behavior
3. **Wrong reference point**: Should we be adding noise to the Oracle's sort order, not the ratings themselves?
4. **CI calculation issue**: Are we computing percentiles correctly?

**Question**: Is this a bug, or does the Oracle genuinely fall outside because of some statistical subtlety I'm missing?

**To investigate**: 
- Check if increasing iterations to 2000 fixes it
- Compare Monte Carlo mean to the observed Oracle gain (should be close)
- Plot all 500 iteration results to see the full distribution

---

### Q2: Why Are Grey Bands Different Widths?

**Observation**: On the gain charts (top), the grey noise bands look skinnier than the colored bootstrap bands (bottom). But both use the same books and similar iteration counts (500 vs 2000).

**Possible explanations**:
1. **Different sources of uncertainty**: Noise in labels affects Oracle differently than bootstrap resampling affects ML models
2. **Sample size**: More books → tighter bands for noise (sqrt scaling), but bootstrap doesn't shrink as fast?
3. **Sorting mechanism**: Noise degrades Oracle sort less than bootstrap samples vary?

**Question**: Is this width difference real and meaningful, or should they be similar?

---

### Q3: Is Monte Carlo Noise the Right Approach?

**Current approach**: Add fake noise to ratings, then measure gains.

**User's concern**: "We don't want to be adding fake noise when we show predictions on real data."

**Alternative approaches**:

#### A. Resampling from actual double-ratings
If we have 208 books with two ratings each:
```python
for iteration in range(500):
    # For each book, randomly pick rating_1 or rating_2
    sampled_ratings = [random.choice([r1, r2]) for r1, r2 in double_ratings]
    compute_gain(sampled_ratings)
```
**Pro**: No fake noise, using real measurement variance  
**Con**: Only works if user has double-ratings (most don't)

#### B. Bootstrap with label noise
```python
for iteration in range(2000):
    resampled_books = sample_with_replacement(books)
    noisy_ratings = true_ratings + Normal(0, labeling_SD)  # Add noise
    compute_gain(resampled_books, noisy_ratings)
```
**Pro**: Combines both sources of uncertainty  
**Con**: Mixes two effects (sampling + measurement error)

#### C. Analytical standard error
Use formula for SE when there's measurement error in labels:
```python
SE_with_noise = sqrt((labeling_SD^2 / n) + (sampling_SD^2 / n))
```
**Pro**: Fast, no simulation needed  
**Con**: Assumes independence, may not capture sort degradation

#### D. Counterfactual: "How much better if less noise?"
Compute: "If your labels had 0.5× the MAE, your predictions would likely be X points better."
```python
# Train model on clean labels, test on noisy labels
clean_labels = true_ratings
noisy_labels = true_ratings + Normal(0, labeling_SD)
model_on_clean = train(features, clean_labels)
gain_with_noise = evaluate(model_on_clean, noisy_labels)

# Compare to model trained on noisy labels
model_on_noisy = train(features, noisy_labels)
gain_baseline = evaluate(model_on_noisy, noisy_labels)

improvement = gain_baseline - gain_with_noise
```

**Question**: Which approach best answers the user's question: "How much does rating noise hurt my ability to filter books?"

---

### Q4: What Should Grey Bands Represent?

**Current interpretation**: "If your ratings had MAE = 0.46, how much would measured gains vary?"

**Alternative interpretations**:
1. **Oracle sort degradation**: "How much worse is the Oracle's sort because ratings are noisy?"
2. **Measurement uncertainty**: "If you re-rated these same books, how much would the gain shift?"
3. **Prediction ceiling**: "What's the best gain you could achieve if you had perfect labels?"
4. **Counterfactual improvement**: "How much better would your filtering be if ratings had 50% less noise?"

**User's goal**: 
> "Interpretation: If both bands are tight, the gain is reliable. If bootstrap bands are wide but rating-noise bands are tight, you need more books. If rating-noise bands dominate, the gain is real but your ratings are noisy."

**Question**: Which interpretation is most useful for decision-making? Do we need different bands for different questions?

---

### Q5: How to Treat Model Predictions as "True" When Bootstrapping?

**Current approach**: Bootstrap resamples books, uses original predictions, measures gain vs original true labels.

**Problem**: Predictions are fixed (trained on all data), but we're simulating "what if different books?" This mismatch may underestimate uncertainty.

**Alternative**: Train a new model on each bootstrap sample
```python
for iteration in range(2000):
    resampled_books = sample_with_replacement(books)
    new_model = train(resampled_books)  # Retrain model!
    predictions = new_model.predict(resampled_books)
    gain = compute_gain(predictions, true_labels)
```
**Pro**: Captures model uncertainty  
**Con**: Can't do in browser (no training), would need pre-computed bootstrap models

**Question**: Is the current bootstrap (fixed predictions) sufficient, or does it underestimate uncertainty?

---

### Q6: How to Convey "If Labels Had Less Noise, Predictions Would Be Better"?

**User's request**: 
> "What about a way to calculate: 'if your labels had 1/2 as much noise your predictions would likely be 0.x rating points better'"

**Possible approaches**:

#### A. Noise attenuation formula
```python
# Disattenuation: observed correlation reduced by sqrt(reliability)
reliability = 1 - (MAE^2 / var(ratings))
true_correlation = observed_correlation / sqrt(reliability)

# Translate to gain improvement
gain_with_half_noise = current_gain * sqrt(reliability_at_half_MAE)
improvement = gain_with_half_noise - current_gain
```

#### B. Simulation
```python
# Baseline: current labels
baseline_gain = evaluate(model, labels_with_MAE_0.46)

# Counterfactual: cleaner labels
clean_labels = labels_with_MAE_0.23  # Half the noise
counterfactual_gain = evaluate(model, clean_labels)

improvement = counterfactual_gain - baseline_gain
```

#### C. Display as a table
| If MAE reduced to | Predicted gain improvement |
|---|---|
| 0.40 (13% reduction) | +0.05 rating points |
| 0.30 (35% reduction) | +0.12 rating points |
| 0.23 (50% reduction) | +0.18 rating points |

**Question**: Which approach is most actionable for users? Should this be a separate analysis, or integrated into the existing charts?

---

## Summary of Decisions Needed

1. **Oracle outside CI**: Bug or feature? How to fix?
2. **Band width difference**: Expected or problem?
3. **Monte Carlo approach**: Keep current method, or use alternatives (resampling actual double-ratings, analytical SE, counterfactual)?
4. **Grey band interpretation**: What question are we answering? (Sort degradation, measurement uncertainty, ceiling, counterfactual?)
5. **Bootstrap + model uncertainty**: Should we retrain models on bootstrap samples (pre-compute), or is current approach sufficient?
6. **Noise reduction benefit**: How to show "if labels had 50% less noise, gains would be X better"? Separate analysis or integrated into charts?

---

## Desired Outcome

After discussion, we should have:
- ✅ Clear definition of what grey noise bands represent
- ✅ Understanding of why Oracle is outside CI (and how to fix if it's a bug)
- ✅ Decision on whether to keep Monte Carlo or use alternative approach
- ✅ Plan for conveying "less noise → better predictions" insight
- ✅ Updated text for chart descriptions that accurately reflects methodology

Then we can update the website implementation accordingly.

---

## References

**Existing analysis scripts**:
- `scripts/temp/threshold_bootstrap_analysis.py` — bootstrap with labeling noise, GP smoothing (n_boot=200, labeling MAE enjoy=0.46, useful=0.34)
- `scripts/temp/oracle_threshold_plots.py` — bootstrap + labeling SE bands (500 iterations)

**Data**:
- `golden_master_multi_source.csv` — 264 books, 66 holdout
- Double-rating validation: 208 books rated twice
  - Enjoyment: Pearson R=0.77, RMSE=0.66, MAE=0.46
  - Usefulness: Pearson R=0.86, RMSE=0.59, MAE=0.34

**Key insight from results_so_far.md**:
> "Test-retest reliability sets the prediction ceiling:
> - Enjoyment: observed R=0.77 → Max achievable R=0.933 (R²=0.870)
> - Usefulness: observed R=0.86 → Max achievable R=0.962 (R²=0.925)
> Current best models achieve 17-36% of ceiling R²"

This suggests rating noise is a major limiting factor — answering "how much better if less noise?" is highly relevant.
