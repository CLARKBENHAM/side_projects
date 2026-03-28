# Book Selection Analysis: Results Summary

Generated 2026-03-23. Based on 268 books (200 historical Jan 2019-Mar 2025 + 68 holdout rated 2026).
Full script outputs: `results_so_far_everything.txt`.
Sections 11–13 added 2026-03-23: model specs, power analysis, validation set design.

---

## 1. Current Best Decision Rules

### Primary: Model-based selection (enjoyment)
- **Model**: `preread_plus_goodreads_conservative + Random Forest`
- **Threshold**: predicted `avg_enjoyment >= 3.9`
- **Holdout performance**: keeps 11/67 books, lifts avg enjoyment by **+0.320** rating points
- **Bootstrap stability**: 34.0% prob of being utility-best, 50.4% prob top-3 (across 2000 resamples)
- **Feature spec**: category + Goodreads rating (conservative matches only, rating_count >= 100)

### Secondary: Model-based selection (usefulness)
- **Model**: `preread_plus_goodreads_conservative + Random Forest`
- **Threshold**: predicted `avg_usefulness >= 2.5`
- **Holdout performance**: keeps 13/67, lifts avg usefulness by **+0.515**
- **Less stable** than enjoyment rules across bootstrap resamples

### Fallback: Simple Goodreads cutoff (no model needed)
- **Goodreads >= 4.2**: keeps 63/194 books (32%), lifts enjoyment +0.360, usefulness +0.519
- **Goodreads >= 4.3**: keeps 34/194 (18%), lifts usefulness +0.697
- Use when you don't have a model prediction available

### Final decision model coefficients (Ridge regression)
- **Enjoyment**: `avg_enjoyment = 0.653*GR + 0.445*AMZ + 0.056*log_GR_count + 0.029*log_AMZ_count + category_offsets`
- **Usefulness**: `avg_usefulness = 0.746*GR + 0.593*AMZ - 0.382*log_GR_count + 0.423*log_AMZ_count + category_offsets`
- Goodreads and Amazon are the only rating sources worth including; OpenLibrary adds nothing

---

## 2. Predictive Model Performance

### Holdout accuracy (70 books, 2026)

#### Complex models (category + GR only, via `future_prediction_evaluation` pipeline)
| Target | Model | MAE | RMSE | Spearman rho |
|--------|-------|-----|------|-------------|
| avg_enjoyment | GBM (GR conservative) | **0.645** | 0.832 | 0.149 |
| avg_enjoyment | Random Forest (GR conservative) | 0.674 | 0.829 | 0.236 |
| avg_usefulness | GBM (GR conservative) | **0.497** | 0.681 | **0.563** |
| avg_usefulness | Random Forest (GR conservative) | 0.576 | 0.739 | 0.441 |

#### Simple models (GR + Amazon + OL + category + log_count, imputed via regression)
| Target | Model | MAE | Spearman rho | Pred std |
|--------|-------|-----|-------------|----------|
| avg_enjoyment | **Simple Ridge** | **0.690** | **0.300** (p=0.01) | 0.314 |
| avg_enjoyment | Simple RF | 0.702 | 0.321 (p=0.007) | 0.406 |
| avg_enjoyment | Simple GBM | 0.726 | **0.393** (p=0.001) | 0.483 |
| avg_usefulness | Simple Ridge | 0.583 | 0.521 (p<0.001) | 0.410 |
| avg_usefulness | Simple RF | 0.611 | 0.517 (p<0.001) | 0.477 |
| avg_usefulness | Simple GBM | 0.614 | 0.424 (p<0.001) | 0.522 |
| avg_enjoyment | Category mean baseline | 0.697 | 0.048 (p=0.69) | — |
| avg_usefulness | Category mean baseline | **0.531** | 0.490 (p<0.001) | — |

Simple models with multi-source ratings **rank better** than complex RF/GBM for enjoyment (rho 0.30-0.39 vs 0.15-0.24). For usefulness, all models cluster around rho 0.42-0.57. Adding Amazon and OL ratings helps the simpler models despite OL being weak standalone.

But once remove count of authors book read so far, all rankings are ~0.45 rho on holdout set. The R^2 is often negative since the means of the periods are changing too much.

Simple heuristics might get more than anything else.

#### Simple Ridge coefficients
- **Enjoyment**: `E = -2.12 + 0.76*GR + 0.37*AMZ + 0.08*OL + 0.10*log_count + category_offsets`
  - Category: Math +0.59, General Reading +0.15, Literature +0.10, Business +0.04, fiction -0.12, ML -0.19, **CS -0.58**
- **Usefulness**: `U = -2.45 + 0.83*GR + 0.17*AMZ + 0.05*OL + 0.07*log_count + category_offsets`
  - Category: **Math +0.86**, ML +0.34, CS +0.32, General +0.03, Business -0.15, **Literature -0.47, fiction -0.93**

### Training LOO cross-validation (194 books)
| Target | Best model | MAE | vs category baseline |
|--------|-----------|-----|---------------------|
| avg_enjoyment | Ridge + Goodreads | 0.706 | +7.6% improvement |
| avg_usefulness | GBM + Goodreads | 0.661 | +7.9% improvement |

### Rolling temporal evaluation (train pre-2024, test 2025)
| Target | Best model | Test MAE | Test rho |
|--------|-----------|----------|---------|
| avg_enjoyment | Random Forest (GR conservative) | **0.547** | 0.325 |
| avg_usefulness | GBM (preread base) | **0.688** | 0.746 |

### Feature importance (Ridge, enjoyment)
1. Recommendation source (inferred): "Classic canon" -0.38, "Friend rec" +0.25
2. Category: CS -0.32, General Reading +0.19, Fiction +0.19
3. Reading days: +0.14
4. Note length: +0.14
5. Goodreads rating: appears in combined models but modest standalone contribution

---

## 3. External Rating Signals

### Goodreads (strongest source)
- Spearman rho with avg_enjoyment: **0.268** (n=194, p=0.0002)
- Spearman rho with avg_usefulness: **0.346** (n=194, p<0.001)
- Coverage: 194/208 historical books matched (93%)
- Data quality: curated matches reduced wrong-book rate from 5.3% to <2%

### Amazon (secondary source)
- Correlation with Goodreads: rho=0.593
- Useful for usefulness prediction; less for enjoyment
- Mean Amazon rating much higher than Goodreads (4.6 vs 4.1 typical)

### Open Library (not useful)
- Coverage: 118/208 (57%), often 1-2 ratings per book
- Correlation with enjoyment: rho=0.05 (not significant)
- Adds nothing when Goodreads is available

### Key data quality finding
- Wrong-book Goodreads matches (19 books, 7.8% of original scraping) significantly damaged model performance
- After fixing: enjoyment GR linear R improved from 0.304 to 0.410 (+0.106)
- Heuristic: `rating_count < 100` catches ~80% of wrong-book matches

---

## 4. Category-Specific Patterns

### Goodreads signal strength by category
| Category | N | Enjoy rho | Useful rho |
|----------|---|-----------|-----------|
| Literature | 52 | **0.325** | 0.206 |
| Business/management | 49 | 0.293 | 0.284 |
| General Reading | 69 | 0.293 | **0.376** |
| Computer Science | 15 | 0.276 | 0.248 |
| Fiction | 46 | 0.111 | 0.208 |
| Histories | 18 | -0.089 | -0.091 |

### Category group regression rules (3-source)
- **Business/Histories/General** (84 books): `enjoy = -0.61 + 0.64*GR + 0.29*OL + 0.07*AMZ` (train R=0.264, holdout R=0.413)
- **Fiction/Literature** (101 books): `enjoy = -3.38 + 0.67*GR - 0.05*OL + 0.93*AMZ` (train R=0.353, **holdout R=-0.379**)
- **Technical** (22 books): `enjoy = 2.28 + 0.84*GR - 0.16*OL - 0.45*AMZ` (train R=0.448, holdout R=0.439)

Fiction/Literature models **anti-predict** on holdout -- Goodreads signal inverts for fiction. External ratings are not useful for predicting fiction enjoyment.

### Category mean ratings
| Category | N | Enjoyment | Usefulness |
|----------|---|-----------|-----------|
| Math | 3 | 4.17 | 3.17 |
| General Reading | 40 | 3.69 | 2.21 |
| Business | 44 | 3.48 | 1.90 |
| Literature | 50 | 3.41 | 1.54 |
| Fiction | 51 | 3.24 | 1.24 |
| ML | 5 | 3.10 | 2.50 |
| CS | 14 | 2.93 | 2.46 |

### Recommendation source means (Gemini-inferred)
| Source | GR rating | Enjoyment | Usefulness |
|--------|-----------|-----------|-----------|
| Friend recommendation | 4.15 | **3.97** | 2.08 |
| Self-discovered | 4.18 | 3.69 | 2.19 |
| Professional need | 4.15 | 3.22 | **2.49** |
| SSC/LessWrong | 4.13 | 3.52 | 1.59 |
| Tyler Cowen | 4.03 | 3.26 | 1.79 |
| Classic canon | 3.80 | **2.98** | 1.24 |

---

## 5. Prediction Ceiling Analysis

### Test-retest reliability (your own rating noise)
| Target | Pearson R | RMSE | MAE |
|--------|-----------|------|-----|
| Enjoyment (original 208) | 0.77 | 0.66 | 0.46 |
| Usefulness (original 208) | 0.86 | 0.59 | 0.34 |
| Enjoyment (new 68 books) | 0.81 | 0.50 | 0.33 |
| Usefulness (new 68 books) | 0.85 | 0.43 | 0.31 |

### Theoretical ceiling (perfect oracle, attenuated by rating noise)
- **Enjoyment**: max achievable R = **0.933** (R^2 = 0.870)
- **Usefulness**: max achievable R = **0.962** (R^2 = 0.925)
- Best models currently achieve:
  - **Enjoyment**: Simple GBM rho=0.393, R²≈0.15 → **17%** of ceiling R². Simple Ridge rho=0.300, R²≈0.09 → **10%** of ceiling.
  - **Usefulness**: Orig GBM rho=0.574, R²≈0.33 → **36%** of ceiling R². Simple Ridge rho=0.521, R²≈0.27 → **29%** of ceiling.

### Disattenuated true correlations (correcting for rating noise)
| Source | Observed r (enjoy) | True rho | Observed r (useful) | True rho |
|--------|-------------------|----------|--------------------|---------|
| Goodreads | 0.290 | 0.330 | 0.340 | 0.367 |
| Amazon | 0.180 | 0.205 | 0.280 | 0.302 |
| OpenLibrary | 0.050 | 0.057 | 0.010 | 0.011 |

### Monte Carlo simulation key results
- 1 signal at rho=0.35: expected test R=0.29, R^2=0.08 (matches empirical Goodreads)
- 5 signals at rho=0.35 (optimistic independent): expected test R=0.37, R^2=0.14
- 5 signals at rho=0.50 (strong): expected test R=0.73, R^2=0.52, top-20% gain +0.86
- **Sample size is not the bottleneck**: n=200 vs n=1000 barely changes CI width
- **Diminishing returns from more rating sources**: GR+OL+AMZ combined test R^2=0.033 (sources are correlated)

---

## 6. Filtering/Drop Gains

### Empirical drop curves on holdout (70 books, actual gains)

| Drop % | Simple Ridge | Simple GBM | Orig RF | GR only | Cat mean |
|--------|-------------|-----------|---------|---------|----------|
| **Enjoyment** | | | | | |
| 20% | +0.023 | +0.108 | +0.027 | +0.072 | +0.023 |
| 50% | **+0.200** | **+0.250** | +0.074 | +0.157 | +0.093 |
| 80% | +0.388 | +0.369 | +0.253 | +0.330 | -0.285 |
| **Usefulness** | | | | | |
| 20% | +0.132 | +0.069 | +0.136 | +0.047 | +0.136 |
| 50% | **+0.298** | +0.259 | +0.107 | +0.155 | +0.248 |
| 80% | +0.470 | +0.499 | **+0.538** | +0.413 | +0.115 |

Simple models match or beat complex RF on drop curves, especially at 50% filtering. Simple Ridge at drop-50% lifts enjoyment +0.200 and usefulness +0.298.

### MC simulation: expected gains vs true signal strength (3000 sims, self-noise=40%)

| True rho | Med obs rho | E[Drop 20%] | E[Drop 50%] | E[Drop 80%] | P(gain>0 @ d50) |
|---------|-------------|-------------|-------------|-------------|-----------------|
| 0.10 | 0.066 | +0.024 | +0.054 | +0.102 | 66% |
| 0.20 | 0.166 | +0.061 | +0.140 | +0.250 | 87% |
| **0.30** | **0.245** | **+0.094** | **+0.216** | **+0.383** | **96%** |
| 0.40 | 0.331 | +0.125 | +0.282 | +0.512 | 99% |
| 0.50 | 0.411 | +0.155 | +0.358 | +0.631 | 100% |

### Matching empirical models to simulation

| Model | Obs rho (enjoy) | ~True rho | E[d50] | E[d80] | p(null) |
|-------|----------------|-----------|--------|--------|---------|
| Simple Ridge | 0.300 | ~0.35 | +0.249 | +0.436 | 0.004 |
| Simple GBM | 0.393 | ~0.50 | +0.359 | +0.639 | <0.001 |
| Orig RF | 0.224 | ~0.30 | +0.216 | +0.383 | 0.033 |
| GR only | 0.282 | ~0.35 | +0.249 | +0.436 | 0.009 |
| Category mean | 0.048 | ~0.10 | +0.054 | +0.102 | 0.346 |

All models except category-mean-alone are statistically significant (p<0.05 under null). The simple Ridge's observed rho=0.30 corresponds to a true signal of ~0.35, consistent with Goodreads' disattenuated correlation.

### Simulation interpretation

These are **simulation averages** — what a signal of a given true rho produces across 3000 simulated test sets (n=70, drawn from the empirical rating distribution, 40% self-noise). They are not direct forecasts of real-world gains from applying these models to unread books, because: (1) the simulation assumes test books are drawn from the same distribution as training, while unread books are a different population; (2) the simulation uses a single linear signal, not multi-feature models with imputation; (3) the true rho matching is approximate. The simulation's value is answering "is this signal real?" and "what order of magnitude of gains does a signal this strong produce?" — not precise point predictions.

**Enjoyment** (Simple Ridge obs rho=0.300, matched true rho ~0.35):
- Drop bottom 20%: sim average +0.108
- Drop bottom 50%: sim average +0.249 (observed on holdout: +0.200, within CI)
- Drop bottom 80%: sim average +0.436
- P(any gain at drop-50%) = 96%
- 80% CI at drop-50%: [+0.06, +0.38] — wide at n=70

**Usefulness** (Simple Ridge obs rho=0.521, matched true rho ~0.70):
- Drop bottom 20%: sim average +0.147
- Drop bottom 50%: sim average +0.470 (observed on holdout: +0.298, below sim expectation but within CI)
- Drop bottom 80%: sim average +1.256
- P(any gain at drop-50%) ≈ 100%

Usefulness signal is ~2x stronger than enjoyment. The gap to the usefulness ceiling (R²=0.925) is personal context (professional needs, timing) that public ratings can't capture. The enjoyment ceiling (R²=0.870) has an even larger gap — subjective taste is what's missing.

---

## 7. Stopping Rules Analysis

### When to quit a book
- Current model: power-law error with exponent 1.8
- At 5% read (15 pages of 300): only 8% of uncertainty resolved
- At 10% (30 pages): 15% resolved
- At 33% (100 pages): 42% resolved
- **Mismatch with intuition**: "can tell within 15 pages" implies >50% info by 5%, but model gives only 8%

### Optimal drop rates by category
| Category | Final drop rate | By 5% read | By 10% read |
|----------|----------------|-----------|------------|
| Fiction | 82.5% | 56.4% | 66.0% |
| Literature | 79.7% | 54.7% | 64.4% |
| Business | 78.1% | 51.9% | 62.1% |
| CS | 75.9% | 50.7% | 59.6% |
| General Reading | 75.2% | 47.1% | 57.5% |

### Error function calibration issue
- Model assumes residual error sigma(1.0) = 0.25 after finishing
- Empirical re-test SD = 0.66 -- model underestimates post-read uncertainty by 2.6x
- Calibrated model (floor=0.66) increases recommended drop rate to 87.8%

---

## 8. Play Books Takeout Scoring

Scored 750 books from Google Play Books catalog:
- 253 matched to Goodreads (34%)
- 96 had enough data for review-count filtering
- **20 labeled "primary read"** by the enjoyment model
- **0 labeled "read high priority"** (threshold too strict for cache-only Goodreads coverage)

---

## 9. Multi-Source Approach Comparison

Three approaches compared on common holdout subset (n=44 books):

| Approach | Keep 30%: enjoy lift | Keep 30%: useful lift | Keep 30%: balanced lift |
|----------|--------------------|--------------------|----------------------|
| Z-score complete-case | **+0.402** | **+0.668** | **+0.535** |
| Raw ridge + impute | +0.229 | +0.418 | +0.324 |
| Sequential missingness | +0.364 | +0.649 | +0.506 |

Z-score complete-case wins on the common holdout subset, but covers fewer books (drops rows with any missing source).

### Z-score utility model coefficients
| Feature | Enjoyment utility coef | Usefulness utility coef |
|---------|----------------------|----------------------|
| Goodreads z | **0.169** | **0.545** |
| OpenLibrary z | 0.089 | 0.004 |
| Amazon z | 0.073 | -0.019 |

Goodreads dominates. OpenLibrary adds slight signal for enjoyment only. Amazon adds nothing for usefulness after accounting for Goodreads.

---

## 10. Actionable Takeaways

1. **Use the simple Goodreads >= 4.2 rule as a default filter.** It lifts enjoyment by +0.36 and usefulness by +0.52, requires no model, and is easy to apply.

2. **The simple Ridge model (GR + Amazon + OL + category + log_count) ranks better than the complex RF for enjoyment** (holdout rho 0.300 vs 0.236) while being simpler and more interpretable. For usefulness, performance is similar across models. Use: `E = -2.12 + 0.76*GR + 0.37*AMZ + 0.08*OL + 0.10*log_count + cat_offsets`.

3. **Amazon ratings add real signal when combined with Goodreads in simple models,** despite being weak alone. OL adds marginal signal. Multi-source imputation (regressing missing ratings on available ones) enables full coverage.

4. **External ratings don't work for fiction.** Fiction/Literature models anti-predict on holdout. For fiction, rely on recommendation source and category mean instead.

5. **Track recommendation sources directly.** Gemini-inferred source is the single most promising untapped feature. Friend recommendations average 0.97 enjoyment points higher than classic canon.

6. **The prediction ceiling is not as far away as R=0.93 sounds.** Current best model captures 32% of achievable enjoyment correlation. The remaining 68% requires features orthogonal to public consensus ratings (your personal taste, which external ratings don't measure).

7. **More data won't help much; better signals will.** Going from n=200 to n=1000 barely changes model performance. The bottleneck is signal quality (true rho with your latent preference), not sample size.

8. **Category-specific models for Business/General/Technical show promising holdout R=0.4-0.5.** Build separate decision rules per category group.

9. **The stopping rule model needs mid-read ratings to calibrate.** Current model disagrees with "15 pages" intuition by 12x. Collecting mid-read ratings (at 5%, 33%, 67%) would resolve this.

10. **Data quality matters more than model complexity.** Fixing 19 wrong-book Goodreads matches improved enjoyment R by +0.106 (more than any model change).

---

## 11. Model Specifications

Three models score unread books. Full details in `results_so_far_everything.txt`.

| Model | Features | Holdout rho (enjoy / useful) | Implementation |
|-------|----------|------------------------------|----------------|
| **RF** | category + GR (conservative, count≥100) | 0.236 / 0.441 | `future_prediction_evaluation.py` → `RandomForestRegressor(n_estimators=200, max_depth=5)` |
| **GBM** | same as RF | 0.149 / **0.563** | same file → `GradientBoostingRegressor(n_estimators=120, lr=0.05, max_depth=3)` |
| **Ridge** | GR + AMZ + log counts + category | **0.300** / 0.521 | `final_decision_model.py` → `Ridge(alpha=1.0)` |

Scoring script: `score_unread_books.py` → `ai_actions/unread_book_scores.csv` (RF/GBM); `validation_book_selection.py` adds Ridge.

### Ridge coefficients (trained on 268 books)
```
Enjoyment:  E = -1.241 + 0.602*GR + 0.350*AMZ + 0.149*log_GR_count - 0.085*log_AMZ_count
  Category offsets: Gen Reading +0.402, Math +0.527, CS -0.684, Lit -0.061

Usefulness: U = -3.446 + 0.624*GR + 0.559*AMZ - 0.268*log_GR_count + 0.315*log_AMZ_count
  Category offsets: Math +0.197, Gen Reading -0.060, Lit -0.065
```

Key model differences: RF/GBM use only GR + category and shrink toward the mean on unread books (pred std=0.21). Ridge adds Amazon/counts and discriminates more (std=0.46). Within each category, RF correlates rho=0.65–0.76 with GR — it's largely a GR proxy. Ridge is better for enjoyment ranking; GBM is better for usefulness ranking.

---

## 12. Power Analysis: How Many Books to Validate

From `power_analysis_simulation.py` (10K sims). Pick top N from ~140 unread Gen Reading + Business books by composite (2.5×useful + enjoy), read at ~5 hrs each, one-sided t-test vs category historical mean.

### Same-quality pool (algorithm helps by filtering)

| N | Hours | Enjoy lift | Enjoy power (α=.05) | Useful lift | Useful power (α=.05) |
|---|-------|-----------|--------------------|-----------|--------------------|
| 5 | 25 | +0.42 | 27% | +0.89 | 57% |
| **10** | **50** | **+0.37** | **36%** | **+0.78** | **81%** |
| 20 | 100 | +0.30 | 43% | +0.65 | 94% |

### Degraded pool (unread books 0.2 points worse)

| N | Hours | Enjoy lift | Enjoy power (α=.05) | Useful lift | Useful power (α=.05) |
|---|-------|-----------|--------------------|-----------|--------------------|
| 10 | 50 | +0.20 | 18% | +0.59 | 61% |
| 20 | 100 | +0.13 | 16% | +0.46 | 72% |

**Bottom line**: Usefulness is detectable at 10 books (81% power). Enjoyment needs >30 books — signal too weak relative to noise. Track usefulness for the fastest answer.

---

## 13. Validation: What to Read and What to Expect

### Consensus top picks (all 3 models agree, Gen Reading + Business + CS)

A Pattern Language (GR 4.4), Wages of Destruction (4.5), Mastery (4.3, started), The Power Law (4.4), The Dream Machine (4.5), Mark Manson - Models (4.3, started), The Strangest Secret (4.4, started), Knuth Vol 2–3 (4.4), Elements of Statistical Learning (4.4).

### Biggest model disagreements (most informative to read)

Intro to Statistical Learning (Ridge rank 4, RF rank 285), ergodicity_economics (Ridge 23, RF 303), Mastering Technical Sales (Ridge 2, RF 278, started). Pattern: Ridge loves high-GR + high-AMZ books that RF/GBM rank low since they only see category + GR.

### 10-book stratified validation set

4 from top quintile (Knuth Vol 2–3, From Third World to First, Hard Landing), 2 from Q4 (Masters of Doom [started], High Output Management), 2 from Q3 (Global Logistics and Strategy, Little Book of Semaphores), 1 from Q2 (Seven Habits), 1 from Q1 (The Essence of Software). Mix of representative picks and model-disagreement books. Full predictions in `ai_actions/validation_book_selection_{10,15,20}_with_started.csv`.

### What to expect

**If it works** (rho ~0.30 enjoy, ~0.52 useful): after 5 books, useful mean should be ~2.9–3.1 vs historical 2.18. After 10, usefulness p < 0.05 (~80% prob), expected lift +0.78. Enjoyment lift +0.37 but likely not significant. Q5 books should average ~0.3–0.5 points above Q1.

**If it doesn't work**: ratings scatter around category mean (enjoy ~3.5, useful ~2.2) with no quintile trend. Spearman rho ≈ 0.

**Sequential decision rules**: after 3+ books, useful mean > 2.5 is encouraging. After 5, compute rho — if > 0.4 for usefulness, model works. After 7, if useful p > 0.20, fall back to GR ≥ 4.2. After 10, horse-race all models on MAE/rho and drop losers.

### Information gain by source

| Source | Effort | Info gain |
|--------|--------|-----------|
| Existing 68-book holdout | 0 hrs | High (rho=0.30–0.56 already measured; single test set limitation) |
| 10-book stratified validation | 50 hrs | Medium (80% power for usefulness; enjoyment noisy) |
| 20-book validation | 100 hrs | Medium-high (diminishing returns; enjoy power 36→43%) |
| Finishing started books | 15–30 hrs | Medium (cheap, but selection bias) |
| GR ≥ 4.2 baseline | 0 hrs | Already have: lifts enjoy +0.36, useful +0.52 |

### Key scripts and outputs

- `power_analysis_simulation.py` → `ai_actions/power_analysis_results.csv`, `power_analysis_curves.png`
- `validation_book_selection.py` → `ai_actions/validation_book_selection_*_with_started.csv`
- `score_unread_books.py` → `ai_actions/unread_book_scores.csv`
- `final_decision_model.py` → `ALL_BOOKS_PREDICTIONS.csv`

---

## 14. Prediction Interval Analysis

Added 2026-03-23. Simulation comparing methods for constructing prediction intervals around point predictions. Script: `prediction_interval_simulation.py`. Full output: `ai_actions/prediction_interval_simulation_results.txt`.

### Method comparison (200 sims, enjoyment, rho=0.35 ~ empirical Ridge)

| Method | 50% coverage | 85% coverage | 85% width | Notes |
|--------|-------------|-------------|-----------|-------|
| **Conformal** (recommended) | 51% | **86%** | 3.0 | Guaranteed coverage; constant-width |
| Empirical residual | 48% | 82% | 2.8 | Slightly undercovers (~3% below nominal) |
| Quantile regression (GBM) | 43% | 77–83% | 2.6–3.4 | Undercovers at 50%; at 85% depends strongly on heteroscedasticity |
| Bootstrap (model only) | **7%** | **14%** | 0.2 | Captures model instability only, NOT a prediction interval |
| Bootstrap + residual | 48% | 82% | 2.8 | Matches empirical residual; more expensive |
| Adaptive conformal | 52% | **73%** | 2.7 | Overfits residual-magnitude model at n=50 cal |

### Key findings

1. **Conformal prediction is the winner.** Mean coverage sits on or slightly above nominal across Ridge/RF/GBM in the sim; construction is k = ceil((n+1)·level) on sorted |calibration residuals|. Assumes exchangeability between calibration and future books.

2. **Intervals are wide because R² is low.** The 85% PI for enjoyment spans ~3.0 rating points (e.g., 1.8–4.8 for a book predicted at 3.3). This is honest — with R²≈0.10, most of the 1–5 scale is genuinely uncertain per book.

3. **Bootstrap alone is NOT a prediction interval.** It only measures "how much does my prediction shift if I resample training data?" — ignores irreducible noise. Results in 7% coverage at 50% nominal. Useful as a stability diagnostic only.

4. **Adaptive conformal fails at our sample size.** Needs >100 calibration points to reliably model heteroscedastic residuals; we have 68. Stick with constant-width conformal.

5. **Heteroscedasticity is not the binding constraint.** Even with strong heteroscedasticity, conformal coverage only drops 1–2%. The irreducible noise (R²=0.10–0.30) dominates.

6. **Calibration set of 68 holdout books is sufficient.** Coverage is stable from n_cal=20 to 100. Variance decreases modestly with more calibration data.

### Practical implementation for Chrome extension

Use split conformal with the 68-book holdout as calibration:
- Compute |residual| = |actual - predicted| for each holdout book per model
- Sort the 68 |residuals|
- 50% PI: 35th sorted |residual| → symmetric band around point pred
- 85% PI: 59th sorted |residual| → symmetric band around point pred
- Example: predicted enjoyment 3.8, q85 = 1.5 → "3.8 [2.3 – 5.0] (85% PI)"

### Model × interval interaction

| Base model | Conformal 85% coverage | Width |
|-----------|----------------------|-------|
| Ridge | 85.5% | 3.28 |
| RF | 86.5% | 3.43 |
| GBM | 86.4% | 3.50 |

All base models achieve target coverage. RF and GBM have slightly wider intervals (larger residuals on calibration set).

---

## 15. Updated Chrome Extension Models (2026-03-24)

### Changes from previous extension models

**Old pipeline (Sections 2, 11):**
- Single pooled Ridge across all categories (category dummies as features)
- 6 features: `gr_rating`, `ol_rating`, `amz_rating`, `log_gr_count`, `log_ol_count`, `log_amz_count`
- Data source: `master_book_metadata_cleaned.csv`, OL/AMZ imputed from GR via linear regression
- RF/GBM included `author_target_mean_hist` and `author_book_count_hist` features
- Bugs found: missing `gr_rating.notna()` filter on Ridge training (5 books with gr_rating=0 corrupted coefficients), "Histories" mapped to "General Reading" instead of keeping as separate category

**New pipeline:**
- **Ridge: 3 group-specific models** from `golden_master_multi_source.csv` (verified GR ratings)
  - Groups: Business/Histories/General, Fiction/Literature, Technical/Other
  - 3 features only: raw GR, OL, AMZ ratings (no log counts, no category dummies)
  - Cross-source imputation per group (OL imputed from GR+AMZ within group)
  - Per-group conformal intervals (wider for Fiction/Literature)
- **RF/GBM: author features removed** (always unknown for new books, added noise)

### Holdout performance comparison (Spearman rho on 68 holdout books)

| Model | Target | Old rho | New rho | Change |
|-------|--------|---------|---------|--------|
| Ridge | Enjoyment | 0.090* | 0.292 | +0.202 |
| Ridge | Usefulness | 0.382* | 0.459 | +0.077 |
| RF | Enjoyment | 0.236 | 0.238 | +0.002 |
| RF | Usefulness | 0.441 | 0.552 | +0.111 |
| GBM | Enjoyment | 0.149 | 0.080 | -0.069 |
| GBM | Usefulness | 0.563 | 0.557 | -0.006 |

*Old Ridge rho was degraded by the gr_rating=0 bug. Original correct Ridge (before export) was ~0.287/0.457.

Key findings:
- **Ridge enjoy**: fixed bug + group-specific models → rho from 0.090 to 0.292
- **RF useful**: removing author features *improved* rho from 0.441 to 0.552
- **GBM enjoy**: rho dropped to 0.080 (p=0.52, not significant) — GBM overfits for enjoyment
- All models overpredict by +0.19 to +0.46 (captured by asymmetric conformal intervals)

### Per-group conformal intervals (asymmetric, from holdout residuals)

| Group | Target | n | 50% [lo, hi] | 85% [lo, hi] |
|-------|--------|---|--------------|---------------|
| Business/Hist/General | Enjoy | 48 | [-0.79, -0.07] | [-1.31, +0.56] |
| Business/Hist/General | Useful | 48 | [-0.77, +0.08] | [-1.07, +0.84] |
| Fiction/Literature | Enjoy | 15 | [-1.17, +0.08] | [-1.40, +1.03] |
| Fiction/Literature | Useful | 15 | [-0.67, -0.29] | [-0.76, -0.15] |
| Pooled fallback | Enjoy | 68 | [-0.90, -0.07] | [-1.41, +0.72] |
| Pooled fallback | Useful | 68 | [-0.78, -0.01] | [-1.08, +0.65] |

Fiction/Literature intervals are wider (less predictable) and Fiction/Lit useful intervals are entirely below the prediction (strong overprediction for that group).

### Overfitting check: per-group vs pooled Ridge

Splitting into 3 groups (4 parameters each) vs 1 pooled model (3 parameters) on same holdout:
- Enjoy rho: 0.279 (pooled) → 0.303 (per-group) — modest improvement
- Useful rho: 0.207 (pooled) → 0.457 (per-group) — large improvement from group-specific intercepts

The usefulness gain is real: fiction books have fundamentally different base rates for usefulness.

### Plot

See `plots/holdout_pred_vs_actual.png` — predicted vs actual for all 6 model×target combinations, colored by group.

### Exact deployed Ridge coefficients (raw 3-source models)

These are the exact `Ridge(alpha=1.0)` coefficients currently exported to the extension JSON and used for the Ridge panels in `plots/holdout_pred_vs_actual.png`. Predictions are clipped to `[1, 5]` after the linear formula.

| Group | Target | Intercept | GR | OL | AMZ |
|-------|--------|-----------|----|----|-----|
| Business/Hist/General | Enjoy | -0.6056 | +0.6358 | +0.2897 | +0.0691 |
| Business/Hist/General | Useful | -2.0925 | +0.6079 | +0.1458 | +0.2436 |
| Fiction/Literature | Enjoy | -3.3826 | +0.6723 | -0.0456 | +0.9320 |
| Fiction/Literature | Useful | -3.2443 | +0.4821 | -0.0538 | +0.6519 |
| Technical/Other | Enjoy | +2.2820 | +0.8362 | -0.1587 | -0.4546 |
| Technical/Other | Useful | +1.2503 | +0.9703 | +0.0115 | -0.5813 |

### Exact imputation rules used before those coefficients

The extension only passes Goodreads and Amazon. Open Library is always missing on-page, so it is first imputed within group using:

| Group | OL imputation used when GR and AMZ both present |
|-------|-----------------------------------------------|
| Business/Hist/General | `OL = 1.8081 + 0.5791*GR - 0.0167*AMZ` |
| Fiction/Literature | `OL = 0.2631 + 0.6155*GR + 0.2699*AMZ` |
| Technical/Other | `OL = 2.3022 + 0.3504*GR + 0.0736*AMZ` |

One-source fallbacks in the extension are:
- Business/Hist/General: `OL = 1.7501 + 0.5743*GR`; `OL = 3.0636 + 0.2254*AMZ`
- Fiction/Literature: `OL = 1.0668 + 0.7215*GR`; `OL = 1.0585 + 0.6363*AMZ`
- Technical/Other: `OL = 2.6395 + 0.3521*GR`; `OL = 3.7500 + 0.0833*AMZ`

If both sources are missing, the target-specific fills are:
- Business/Hist/General: `GR=4.12`, `OL=4.10`, `AMZ=4.6428`
- Fiction/Literature: `GR=4.01`, `OL=4.00`, `AMZ=4.60`
- Technical/Other: `GR=4.245`, `OL=4.1196`, `AMZ=4.6898`

### The actual 2-coefficient rules to do in your head

Because OL is not scraped by the extension, the most relevant deployed rule is after substituting the OL imputation above into the ridge:

| Group | Target | Formula when GR and AMZ are both present |
|-------|--------|-------------------------------------------|
| Business/Hist/General | Enjoy | `pred = clip(-0.0819 + 0.8036*GR + 0.0643*AMZ)` |
| Business/Hist/General | Useful | `pred = clip(-1.8288 + 0.6923*GR + 0.2412*AMZ)` |
| Fiction/Literature | Enjoy | `pred = clip(-3.3946 + 0.6442*GR + 0.9196*AMZ)` |
| Fiction/Literature | Useful | `pred = clip(-3.2585 + 0.4490*GR + 0.6374*AMZ)` |
| Technical/Other | Enjoy | `pred = clip(1.9166 + 0.7805*GR - 0.4663*AMZ)` |
| Technical/Other | Useful | `pred = clip(1.2767 + 0.9743*GR - 0.5804*AMZ)` |

So yes, in the normal extension case you are basically using only **two slopes**: Goodreads and Amazon, plus an intercept, with OL folded in implicitly.

If only one source is available, the exact deployed rules become:

| Group | Target | GR-only | AMZ-only | Neither source |
|-------|--------|---------|----------|----------------|
| Business/Hist/General | Enjoy | `0.1410 + 0.8221*GR` | `1.6603 + 0.4002*AMZ` | `3.5223` |
| Business/Hist/General | Useful | `-0.9920 + 0.7618*GR` | `-0.3278 + 0.5306*AMZ` | `2.1410` |
| Fiction/Literature | Enjoy | `-0.5144 + 0.9684*GR` | `-2.5621 + 1.3032*AMZ` | `3.4179` |
| Fiction/Literature | Useful | `-1.2613 + 0.6734*GR` | `-2.6782 + 0.9047*AMZ` | `1.4726` |
| Technical/Other | Enjoy | `-0.2208 + 0.7701*GR` | `5.1415 - 0.4445*AMZ` | `3.0457` |
| Technical/Other | Useful | `-1.3841 + 0.9612*GR` | `5.3021 - 0.5533*AMZ` | `2.6903` |

### How similar are the six ridge models?

Short answer: **not very**, except that Goodreads is always positive.

- Stable pattern across all 6: Goodreads is always positive and usually the biggest weight.
- Unstable pattern across groups: Amazon is positive for Business/Histories/General and Fiction/Literature, but **negative** for Technical/Other for both targets.
- Open Library direct weight is small or near zero in 4 of the 6 models; most of its effect is indirect through the imputation step.
- Within a given group, enjoy/useful have very similar slope direction:
  - Business cosine similarity on `(GR, AMZ)` = `0.968`
  - Fiction cosine similarity = `1.000`
  - Technical cosine similarity = `1.000`
- Across groups, slope direction is only moderately aligned except for Fiction vs Technical, which is basically orthogonal:
  - Enjoyment: Business vs Fiction `0.637`, Business vs Technical `0.815`, Fiction vs Technical `0.073`
  - Usefulness: Business vs Fiction `0.813`, Business vs Technical `0.643`, Fiction vs Technical `0.076`

Interpretation: there is **not** one universal ridge rule hiding underneath. There are really three group-specific rules, and the biggest disagreement is how much to trust Amazon relative to Goodreads.

### Should the extension add log-count, date, or log-pages features?

For this comparison I kept the current 3-group ridge recipe fixed and only added one extra feature family at a time on `golden_master_multi_source.csv`. For `date` I used `book_age` from `pub_year`, since future finish date is not knowable at scoring time.

| Variant | Enjoy rho | Enjoy MAE | Useful rho | Useful MAE |
|--------|-----------|-----------|------------|------------|
| Current 3-rating group ridge | 0.292 | 0.702 | 0.459 | 0.620 |
| + Goodreads log rating count | 0.332 | 0.672 | 0.506 | 0.613 |
| + Goodreads log review count | 0.332 | 0.666 | 0.513 | 0.612 |
| + book age | 0.305 | 0.710 | 0.468 | 0.612 |
| + log pages | 0.290 | 0.703 | 0.461 | 0.616 |
| + review-count + book-age + log-pages | **0.340** | **0.678** | **0.517** | **0.607** |

Recommendation:
- **Worth adding:** a Goodreads count feature. It gives the clearest consistent gain for both targets.
- **Probably not worth adding alone:** `book_age`. Tiny effect.
- **Not worth adding alone:** `log_pages`. Essentially no gain.
- **If you want one minimal upgrade:** add only `log10(1 + Goodreads count)` and keep the rest of the ridge simple.
- **If you want the best of these simple ridge variants:** add count + `book_age` + `log_pages`, but the incremental gain over count-alone is modest, especially for usefulness.

## Extension-safe numeric model comparison and temporal CV

I compared the exact requested extension-safe model families on the same `golden_master_multi_source.csv` frame using:
- full `Holdout 2026`
- leave-one-year-out CV aggregated over all predictions
- leave-one-half-year-out CV aggregated over all predictions

The compared models were:
- old pooled ridge with category dummies + counts
- current grouped ridge
- grouped ridge + counts
- grouped ridge + counts + book meta
- richer RF extension-safe
- richer GBM extension-safe

I also included a fuller pooled ridge candidate with category dummies + counts + book meta.

### Results summary

#### Enjoyment

| Model | Holdout R² | Holdout MAE | Holdout rho | Year agg R² | Halfyear agg R² |
|------|------------|-------------|-------------|-------------|-----------------|
| Grouped ridge + counts | -0.228 | 0.704 | 0.291 | 0.001 | 0.013 |
| Current grouped ridge | -0.232 | 0.702 | 0.292 | **0.033** | **0.033** |
| Grouped ridge + counts + book meta | -0.242 | 0.711 | 0.294 | -0.038 | -0.035 |
| Old pooled ridge + counts | -0.251 | 0.697 | 0.294 | -0.005 | 0.000 |
| Pooled full ridge | -0.260 | 0.700 | 0.288 | -0.019 | -0.015 |
| RF extension-safe | -0.261 | **0.689** | 0.304 | 0.016 | 0.018 |
| GBM extension-safe | -0.328 | 0.713 | **0.343** | -0.060 | -0.062 |

#### Usefulness

| Model | Holdout R² | Holdout MAE | Holdout rho | Year agg R² | Halfyear agg R² |
|------|------------|-------------|-------------|-------------|-----------------|
| GBM extension-safe | **0.039** | **0.578** | 0.462 | 0.100 | 0.097 |
| Old pooled ridge + counts | 0.009 | 0.594 | 0.447 | 0.176 | 0.175 |
| Pooled full ridge | 0.005 | 0.594 | 0.449 | **0.181** | **0.176** |
| RF extension-safe | -0.015 | 0.601 | **0.477** | 0.136 | 0.143 |
| Grouped ridge + counts | -0.068 | 0.636 | 0.468 | 0.123 | 0.159 |
| Grouped ridge + counts + book meta | -0.071 | 0.637 | 0.464 | 0.109 | 0.130 |
| Current grouped ridge | -0.088 | 0.620 | 0.459 | 0.147 | 0.148 |

### Interpretation

- For **numeric usefulness**, the current grouped ridge is no longer the best choice.
- If you optimize for the single 2026 holdout, **GBM extension-safe** is best (`R² = 0.039`, `MAE = 0.578`).
- If you optimize for more stable time-split generalization, the best linear option is the **pooled full ridge** and the very close simpler option is **old pooled ridge + counts**.
- For **numeric enjoyment**, none of these models are genuinely good. Every candidate has negative holdout `R²`.
- The least-bad enjoyment models depend on what you care about:
  - **MAE:** RF extension-safe
  - **Holdout R²:** grouped ridge + counts
  - **Temporal stability:** current grouped ridge

### Recommendation for the extension

- Keep the existing grouped ridge for continuity and interpretability.
- Add a second **Full Ridge** model for numeric prediction: pooled ridge with category dummies + GR/OL/AMZ + Goodreads/Amazon counts + `log_pages` + `book_age`.
- For usefulness, if you want the best raw holdout numeric predictor, prefer the richer **GBM**.
- For enjoyment, treat all numeric predictions as rough filtering signals rather than calibrated point estimates.

This fuller pooled ridge is now exported in `chrome_extension/models.json` as:
- `ridge_full_enjoy`
- `ridge_full_useful`

## Calibrated probability recommendation for the extension

I ran a calibration audit using:
- train-year out-of-fold predictions on finished-book training data
- fixed calibration applied to the 2026 holdout
- nested leave-one-year-out with calibration re-fit only on the outer-train slice

Key files:
- `ai_actions/probability_calibration_report.md`
- `ai_actions/probability_calibration_summary.csv`
- `ai_actions/probability_calibration_reliability.csv`
- `ai_actions/probability_calibration_band_summary.csv`
- full notes copied into `ai_actions/results_so_far_everything.txt`

### Main recommendation

If the extension needs **one output that maps to a decision**, the best current choice is:
- **Primary output:** `P(avg_usefulness >= 2.0)`
- **Model:** `gbm_extension_safe`

Why:
- **Holdout 2026:** `Brier = 0.201`, `AUC = 0.730`
- **Nested year-LOO:** `Brier = 0.199`, `AUC = 0.719`

This is not perfectly calibrated, but it is clearly better than:
- enjoyment probabilities
- raw score outputs
- percentile remappings pretending to be confidence

### Important caveat

`P(avg_usefulness >= 2.0)` is **not** the same question as the old plots where:
- x-axis = drop the bottom `X%` of books
- y-axis = gain in average usefulness / enjoyment

Those old plots were mostly testing **ranking / screening power**.
This new audit is testing **absolute decision calibration**.

So the fact that the old work often looked best around “drop the bottom 70-80%” does **not** mean the calibrated threshold should be near `0.7` or `0.8`.
It just means the models may still be better at screening a large pool than at saying “this specific book clears an absolute bar.”

### Practical interpretation

Recommended UI:
- `Chance this book will be useful: XX%`

Suggested action mapping:
- `< 20%`: `Probably skip`
- `20% to < 40%`: `Low priority`
- `40% to < 60%`: `Worth considering`
- `>= 60%`: `Promising`

Do **not** show:
- enjoyment probabilities as the main number
- score percentiles as if they were calibrated
- `P(avg_usefulness >= 2.5)` as the main number
- any `80% likely` or `high confidence` badge

Why not the `80%` band:
- on nested year-LOO for `P(avg_usefulness >= 2.0)`, the `~80%` bucket had only `n = 9` and realized at `0.444`
- so it is too sparse / unstable to present as a strong claim

### Plots to look at

Absolute-bar calibration:
- `plots/probability_calibration_holdout_2026_avg_usefulness_ge_2p0.png`
- `plots/probability_calibration_loo_year_nested_avg_usefulness_ge_2p0.png`

Ranking-style calibration:
- `plots/probability_calibration_holdout_2026_avg_usefulness_top_10pct_within_year.png`
- `plots/probability_calibration_holdout_2026_avg_usefulness_top_20pct_within_year.png`
- `plots/probability_calibration_loo_year_nested_avg_usefulness_top_10pct_within_year.png`
- `plots/probability_calibration_loo_year_nested_avg_usefulness_top_20pct_within_year.png`

Enjoyment comparison:
- `plots/probability_calibration_holdout_2026_avg_enjoyment_ge_3p5.png`

### What this means

- If the extension should answer “is this book likely useful in an absolute sense?”, use `P(avg_usefulness >= 2.0)`.
- If the extension should mimic the old “screen out the bottom chunk” workflow, a better **secondary** output is `P(top 10% usefulness within year)` or another ranking-style standout signal.
- The fiction concern is real: `avg_usefulness >= 2.0` is an absolute bar, not category-adjusted, so a fiction-heavy pool may look low even when the model is still ranking fiction books correctly within fiction.
- So the cleanest current setup is:
  - one main calibrated usefulness probability
  - optionally one separate shortlist / standout signal for large-pool ranking

### Predicted-score ridge threshold plots

Added two new ridge-threshold plots with the x-axis as the predicted score cutoff:
- `ai_actions/standardized_threshold_plots_fixed_predicted_all_data.png`
- `ai_actions/standardized_threshold_plots_fixed_predicted_holdout.png`

## Deep dive: what's actually true about model performance (2026-03-26)

Comprehensive audit of all prediction pipelines — permutation tests, feature ablation, bootstrap coefficient stability, bias-corrected R², LOO-CV, and full code verification.

Key files produced:
- `plots/deep_dive_model_analysis.py` — the full analysis script
- `plots/ablation_results.csv`
- `plots/bootstrap_ridge_coefficients.csv`
- `plots/bias_corrected_r2.csv`
- `plots/loo_cv_ridge.csv`
- `plots/holdout_pred_vs_actual.png` (updated with R² and MAE by category)
- `plots/train_pred_vs_actual.png` (new)

### Is rho=0.55 real?

Yes. Permutation test (5000 shuffles) confirms:

| Model | Target | Holdout rho | Permutation p | Scipy p |
|-------|--------|-------------|---------------|---------|
| RF | Usefulness | 0.552 | 0.0000 | 0.0000 |
| GBM | Usefulness | 0.557 | 0.0000 | 0.0000 |
| RF | Enjoyment | 0.238 | 0.0460 | 0.0504 |
| GBM | Enjoyment | 0.080 | 0.5060 | 0.5151 |

Usefulness ranking is highly significant. Enjoyment is borderline for RF, not significant for GBM.

### But most of the signal is just category

Feature ablation on RF usefulness (holdout):

| Ablation | Holdout rho | Delta vs FULL |
|----------|-------------|---------------|
| FULL (all features) | 0.552 | baseline |
| drop goodreads_rating | 0.579 | +0.027 (!) |
| drop goodreads_log_count | 0.549 | -0.003 |
| drop log_pages | 0.552 | +0.000 |
| drop year_finished | 0.549 | -0.003 |
| drop book_age | 0.536 | -0.016 |
| GR features only (+ category) | 0.538 | -0.014 |
| CATEGORY ONLY | 0.463 | -0.089 |

Category alone gives rho=0.463 — that's 84% of the full model's 0.552. The models are primarily learning "technical/business books are more useful than fiction." Dropping GR rating actually *improves* RF usefulness slightly, suggesting it adds noise for this target.

For enjoyment, category alone gives rho=-0.048 (useless). GR features are the entire signal (rho=0.238), but it's barely significant (p=0.05).

### Ridge coefficient robustness (500 bootstraps)

| Group | Feature | Full coef | 95% CI | Sign stability |
|-------|---------|-----------|--------|----------------|
| Business/Hist/General | GR | +0.636 | [+0.30, +0.95] | 99.8% |
| Business/Hist/General | OL | +0.290 | [-0.15, +0.76] | 90.4% |
| Business/Hist/General | AMZ | +0.069 | [-0.39, +0.53] | 58.0% |
| Fiction/Literature | GR | +0.672 | [-0.31, +1.67] | 90.2% |
| Fiction/Literature | AMZ | +0.932 | [+0.16, +1.83] | 98.8% |
| Technical/Other | GR | +0.836 | [+0.18, +1.54] | 98.0% |
| Technical/Other | AMZ | -0.455 | [-1.10, +0.12] | 93.4% |

Goodreads is the only consistently stable positive coefficient across all groups. OpenLibrary is noise (sign stability 52-90%). Amazon is group-dependent — positive for Fiction (99%), negative for Technical (93%), coin-flip for Business (58%).

Critically, **Fiction/Literature Ridge is systematically anti-predictive on holdout**: across 500 bootstrap resamples, P(holdout rho > 0) = 0.0% for enjoyment, 1.4% for usefulness. The fiction Ridge model *never* produces positive holdout correlation.

### The negative R² is mostly distribution shift

After bias-correcting (subtracting mean over/underprediction) for Ridge:

| Group/Target | n | R² raw | R² corrected | Pearson r | Bias |
|---|---|---|---|---|---|
| Business enjoy | 48 | -0.15 | **+0.17** | 0.41 | +0.39 |
| Business useful | 48 | -0.07 | **+0.03** | 0.19 | +0.28 |
| Fiction enjoy | 15 | -0.66 | -0.37 | -0.38 | -0.24 |
| Fiction useful | 15 | -1.11 | -0.42 | -0.22 | -0.51 |
| Technical enjoy | 5 | -0.84 | **+0.17** | 0.44 | +0.47 |
| Technical useful | 5 | -35.2 | -0.11 | 0.50 | +1.11 |

Business/Hist/General has real signal (Pearson r=0.41 for enjoyment). The negative R² comes from the holdout mean being 0.39 points lower than training — pure distribution shift. Technical has promising correlation (r=0.44-0.50) but only n=5 so unreliable. Fiction is genuinely anti-predictive even after bias correction.

### LOO-CV reveals per-group Ridge fragility

| Group/Target | n | In-sample rho | LOO-CV rho | Overfit gap |
|---|---|---|---|---|
| Business enjoy | 114 | 0.252 | 0.116 | 0.136 |
| Business useful | 114 | 0.276 | 0.090 | 0.187 |
| Fiction enjoy | 71 | 0.379 | 0.294 | 0.085 |
| Fiction useful | 71 | 0.214 | 0.112 | 0.102 |
| Technical enjoy | 22 | 0.401 | **-0.026** | **0.428** |
| Technical useful | 22 | 0.428 | **-0.292** | **0.720** |

Technical_Other (n=22) completely overfits — LOO-CV shows zero or negative signal. Even Business Ridge LOO-CV rho is only 0.09-0.12. The honest in-sample Ridge performance with 3 correlated rating features is very weak.

### Training vs holdout overfit comparison

| | Ridge train R² | Ridge holdout R² | RF train R² | RF holdout R² | GBM train R² | GBM holdout R² |
|---|---|---|---|---|---|---|
| Enjoyment | 0.13 | -0.23 | 0.41 | -0.21 | 0.62 | -0.61 |
| Usefulness | 0.23 | -0.09 | 0.53 | 0.01 | 0.73 | 0.13 |

GBM overfits the most (train R²=0.73 → holdout R²=0.13 for usefulness). Ridge overfits the least but is just weak. RF is middle ground — the only model with non-negative holdout R² for usefulness.

### Code verification: no bugs found

- Ridge pipeline: manual predictions match sklearn `predict()`, no NaNs after imputation, no out-of-range values
- Tree pipeline: data counts match (207 train, 68 holdout), feature specs correct
- 2 holdout title format mismatches between Ridge and Tree pipelines (truncated vs full titles): cosmetic only, doesn't affect predictions
- All rho numbers in Section 15 are exactly reproducible
- One real code bug in `new_books_to_rate_analysis.py`: `requested_utility()` uses `base^(r-1)-1` with base=1.3 for enjoyment, while `goodreads_followup_analysis.py` uses `(r-1)^1.3` (power transform). Different functions. This only affects utility-weighted policy evaluation, not any model predictions or rho/R²/MAE numbers reported anywhere in this document.

### Bottom line

- **Usefulness ranking works** because "category predicts usefulness" is a strong, robust signal (rho=0.46 from category alone). External ratings add ~0.09 rho on top.
- **GR rating adds marginal signal** beyond category for usefulness. For enjoyment, GR is the entire signal (~0.25 rho) but it's barely significant.
- **Ridge with 3 correlated ratings** per group is too weak (LOO-CV R²≈0). The features are too correlated to learn separate slopes reliably.
- **Fiction models are anti-predictive** on holdout — external ratings correlate negatively with personal fiction preferences. This is consistent across all bootstrap resamples.
- **The old pooled Ridge** (Section 15 table, rho=0.521 useful) was a different model with more features (log counts + category dummies). The per-group switch traded some usefulness performance for interpretability and group-specific intercepts.
- **Technical_Other is too small** (n=22 train, n=5 holdout) for a separate model — LOO-CV shows it completely overfits. Should use pooled fallback.
- **Negative holdout R² is mostly distribution shift**, not garbage models. After bias correction, Business/Hist/General Ridge has R²=+0.17 for enjoyment (Pearson r=0.41).

Main read:
- enjoyment is stable: all-data `rho = 0.290` vs holdout `rho = 0.292`
- usefulness is somewhat optimistic in-sample: all-data `rho = 0.551` vs holdout `rho = 0.459`
- dropping `50%` to `70%` still looks reasonable on holdout, but `80%` to `90%` dropped for usefulness is too optimistic in-sample

More detail, filenames, and line references are in `ai_actions/results_so_far_everything.txt`.


3 models each pretty simple
/Users/clarkbenham/side_projects/ai_books_tracking/ai_actions/combined_target_threshold_holdout.png
  - Business (n=48): Solid upward trend — dropping the bottom 50-70% gains ~0.2-0.4 rating points on combined, enjoyment, and usefulness. Bootstrap bands stay above zero.
  - Fiction (n=15): Flat or negative. The model can't rank Fiction books on holdout.
  - Technical (n=5): Steep gains but bootstrap bands are massive with n=5.
so varies a lot by type in how predictable it is.

high rho is mostly just categories having different means

 1. GBM BASE_PLUS_AMZN_NO_BOOK_AGE wins at holdout rho=0.524, usefulness rho=0.603. Dropping book_age helps (0.524 vs 0.484).
  2. Adding goodreads_rating consistently hurts GBM holdout (0.471 vs 0.484). Same pattern as the per-group Ridge analysis.
  3. FULL model is worst for GBM holdout (0.416) despite best train rho (0.858). Classic overfit — train rho 0.86 vs CV 0.39.
  4. Ridge GR_AMZ_RATINGS_ONLY has the best CV-5 (0.413) and nearly matches holdout (0.413) — most honest model. But GBM with amazon metadata beats it on holdout because the nonlinear model extracts signal from amazon_log_count and
  year_finished that Ridge can't.

Goodreads actually harms performance on holdout. Coef is still positive but on both codex and claude it reduces performance when it's removed.



# Best I could do
ai_books_tracking/ai_actions/oracle_plots/1st_rating_predicts_2nd_rating_all_data.png
ploting rating 1 against rating 2 gets like a point, 1.25pts per category keeping 30ish of books.
With perfect on all data it's 1.5+, but all the books are there and a question of how many I'd want to keep reading.
With 0.3 it's ~1.5 ai_books_tracking/ai_actions/oracle_plots/oracle_threshold_plots_0.3_all_data.png only dropping 60-70% of books.
ai_books_tracking/ai_actions/oracle_plots/oracle_threshold_plots_0.9_all_data.png looks better than the other plots I have: 0.5 enjoyment gains and 1pt+usefulness
So model holdout is a big problem.

Using uniform noise [-0.9,0.9] ai_books_tracking/ai_actions/human_limits_plots/human_limit_plots_noise_0.9_holdout.png
gives 1pt enjoyment, 1.5 for usefulness.
When I combine them weighted I get 1-1.25pts bettter if drop 75-80%.

ai_books_tracking/ai_actions/human_limits_plots/human_limit_combined_plots_all_data.png
So gains of 0.5 rating point while dropping 60-80% would be quite good.

doubling my own rating error brings limits to 0.5-0.75 except for computer science (+1.5) and general reading (1-1.2).
ai_books_tracking/ai_actions/human_limits_plots/human_limit_combined_doubled_error_plots_all_data.png
Thats because usefulness is more predictable/consistent and those categories have more usefulness variance
ai_books_tracking/ai_actions/human_limits_plots/human_limit_separate_doubled_error_plots_all_data.png

MAE is 0.45-0.5 for enjoyment and 0.33-0.4 for usefulness across categories. Except for fiction which is 0.37 and 0.14.
So I'm actually more consistent on fiction even though it's hard to predict out of sample






