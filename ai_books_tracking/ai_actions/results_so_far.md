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
