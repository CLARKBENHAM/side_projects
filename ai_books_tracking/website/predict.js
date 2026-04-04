/**
 * Pure prediction functions for book rating models
 * No extension APIs - works in popup, content script, Node.js, or browser
 * Loaded as a script tag before popup.js, content.js, or app.js
 */

/**
 * Recursively predict value from a decision tree
 * @param {Object} tree - Tree node with feature, threshold, left, right, or value
 * @param {Object} features - Feature dictionary
 * @returns {number} Predicted value
 */
function predictTree(tree, features) {
  if ("value" in tree) return tree.value;
  const val = features[tree.feature];
  if (val === undefined || val === null) {
    return predictTree(tree.left, features);
  }
  return val <= tree.threshold
    ? predictTree(tree.left, features)
    : predictTree(tree.right, features);
}

/**
 * Predict using Random Forest model
 * @param {Object} modelSpec - RF model specification with trees, features, scaler
 * @param {Object} features - Feature dictionary
 * @returns {number} Predicted value (clamped to [1, 5])
 */
function predictRF(modelSpec, features) {
  const scaled = {};
  for (const feat of modelSpec.numeric_features) {
    let val = features[feat];
    if (val === null || val === undefined || isNaN(val)) {
      val = modelSpec.medians[feat];
    }
    scaled[feat] = (val - modelSpec.scaler_means[feat]) / modelSpec.scaler_scales[feat];
  }
  const cats = modelSpec.cat_categories[0];
  const dropIdx = modelSpec.cat_drop_idx[0];
  for (let i = 0; i < cats.length; i++) {
    if (i === dropIdx) continue;
    scaled[`Bookshelf_${cats[i]}`] = features.Bookshelf === cats[i] ? 1.0 : 0.0;
  }
  let sum = 0;
  for (const tree of modelSpec.trees) {
    sum += predictTree(tree, scaled);
  }
  return Math.max(1.0, Math.min(5.0, sum / modelSpec.trees.length));
}

/**
 * Predict using Gradient Boosting Machine model
 * @param {Object} modelSpec - GBM model specification with trees, learning_rate, init_value
 * @param {Object} features - Feature dictionary
 * @returns {number} Predicted value (clamped to [1, 5])
 */
function predictGBM(modelSpec, features) {
  const scaled = {};
  for (const feat of modelSpec.numeric_features) {
    let val = features[feat];
    if (val === null || val === undefined || isNaN(val)) {
      val = modelSpec.medians[feat];
    }
    scaled[feat] = (val - modelSpec.scaler_means[feat]) / modelSpec.scaler_scales[feat];
  }
  const cats = modelSpec.cat_categories[0];
  const dropIdx = modelSpec.cat_drop_idx[0];
  for (let i = 0; i < cats.length; i++) {
    if (i === dropIdx) continue;
    scaled[`Bookshelf_${cats[i]}`] = features.Bookshelf === cats[i] ? 1.0 : 0.0;
  }
  let pred = modelSpec.init_value;
  for (const tree of modelSpec.trees) {
    pred += modelSpec.learning_rate * predictTree(tree, scaled);
  }
  return Math.max(1.0, Math.min(5.0, pred));
}

function imputeRidgeRatings(groupSpec, grRating, amzRating) {
  const imp = groupSpec.imputation;
  const ratings = {
    goodreads_rating_raw: grRating,
    openlibrary_rating_raw: null,
    amazon_rating_raw: amzRating,
  };
  for (const feat of ["goodreads_rating_raw", "openlibrary_rating_raw", "amazon_rating_raw"]) {
    if (ratings[feat] != null) continue;
    const steps = imp[feat] || [];
    let imputed = null;
    for (const step of steps) {
      const vals = step.predictors.map(p => ratings[p]);
      if (vals.some(v => v == null)) continue;
      imputed = step.intercept;
      for (let i = 0; i < vals.length; i++) {
        imputed += step.coefs[i] * vals[i];
      }
      break;
    }
    ratings[feat] = imputed ?? imp.medians[feat];
  }
  for (const feat of ["goodreads_rating_raw", "openlibrary_rating_raw", "amazon_rating_raw"]) {
    if (ratings[feat] == null) ratings[feat] = imp.medians[feat];
  }
  return ratings;
}

function predictGroupRidge(groupSpec, targetSpec, grRating, amzRating) {
  const ratings = imputeRidgeRatings(groupSpec, grRating, amzRating);
  let pred = targetSpec.intercept;
  for (const feat of ["goodreads_rating_raw", "openlibrary_rating_raw", "amazon_rating_raw"]) {
    pred += targetSpec.coefficients[feat] * ratings[feat];
  }
  return Math.max(1.0, Math.min(5.0, pred));
}

function predictPooledRidge(modelSpec, groupSpec, pageData, displayCategory) {
  if (!modelSpec || !groupSpec) return null;
  const ratings = imputeRidgeRatings(groupSpec, pageData.grRating, pageData.amzRating);
  const pooledCategory = mapToRFCategory(displayCategory);
  const numericValues = {
    goodreads_rating_raw: ratings.goodreads_rating_raw,
    openlibrary_rating_raw: ratings.openlibrary_rating_raw,
    amazon_rating_raw: ratings.amazon_rating_raw,
    goodreads_log_count: pageData.grCount != null ? Math.log10(1 + pageData.grCount) : null,
    amazon_log_count: pageData.amzCount != null ? Math.log10(1 + pageData.amzCount) : null,
    log_pages: pageData.pageCount != null && pageData.pageCount > 0 ? Math.log10(pageData.pageCount) : null,
    book_age: pageData.pubYear != null ? 2026 - pageData.pubYear : null,
  };

  let pred = modelSpec.intercept;
  for (const feat of modelSpec.numeric_features) {
    const val = numericValues[feat];
    pred += modelSpec.coefficients[feat] * (val == null || isNaN(val) ? modelSpec.fill_values[feat] : val);
  }
  for (const feat of modelSpec.category_features || []) {
    const active = feat === `category_${pooledCategory}` ? 1.0 : 0.0;
    pred += (modelSpec.coefficients[feat] || 0.0) * active;
  }
  return Math.max(1.0, Math.min(5.0, pred));
}

function computeIntervals(pointPred, conformalSpec) {
  if (!conformalSpec) return null;
  return {
    q50: conformalSpec.q50,
    q85: conformalSpec.q85,
    sym50_lo: Math.max(1.0, pointPred - conformalSpec.q50),
    sym50_hi: Math.min(5.0, pointPred + conformalSpec.q50),
    sym85_lo: Math.max(1.0, pointPred - conformalSpec.q85),
    sym85_hi: Math.min(5.0, pointPred + conformalSpec.q85),
    asym50_lo: Math.max(1.0, pointPred + conformalSpec.lo50),
    asym50_hi: Math.min(5.0, pointPred + conformalSpec.hi50),
    asym85_lo: Math.max(1.0, pointPred + conformalSpec.lo85),
    asym85_hi: Math.min(5.0, pointPred + conformalSpec.hi85),
  };
}

function mapToGroup(displayCat) {
  const map = {
    "Fiction": "Fiction_Literature", "fiction": "Fiction_Literature",
    "Literature": "Fiction_Literature",
    "Computer Science": "Technical_Other", "CS": "Technical_Other",
    "Machine Learning": "Technical_Other", "ML": "Technical_Other",
    "Math": "Technical_Other", "Unknown Shelf": "Technical_Other",
    "Business": "Business_Histories_General",
    "Business, management": "Business_Histories_General",
    "General Reading": "Business_Histories_General",
    "Histories": "Business_Histories_General",
  };
  return map[displayCat] || "Business_Histories_General";
}

function mapToRFCategory(displayCat) {
  const map = {
    "Business": "Business, management", "Business, management": "Business, management",
    "Computer Science": "Computer Science", "General Reading": "General Reading",
    "Histories": "Histories", "Literature": "Literature",
    "Fiction": "fiction", "fiction": "fiction",
    "Machine Learning": "Machine Learning", "Math": "Math",
    "Advanced Finance": "Business, management", "Energy Trading": "Business, management",
    "ML": "Machine Learning", "CS": "Computer Science",
    "Unknown Shelf": "General Reading",
  };
  return map[displayCat] || "General Reading";
}

function valueToPercentile(value, percentileMapping) {
  if (!percentileMapping || value === null || value === undefined || isNaN(value)) {
    return null;
  }

  const values = percentileMapping.values || [];
  const percentiles = percentileMapping.percentiles || [];
  if (values.length === 0 || percentiles.length === 0) return null;
  if (value <= values[0]) return percentiles[0];

  for (let i = 0; i < values.length - 1; i++) {
    if (value <= values[i + 1]) {
      const span = values[i + 1] - values[i];
      const ratio = span === 0 ? 0 : (value - values[i]) / span;
      return percentiles[i] + ratio * (percentiles[i + 1] - percentiles[i]);
    }
  }
  return percentiles[percentiles.length - 1];
}

function intervalToPercentiles(intervalSpec, percentileMapping) {
  if (!intervalSpec || !percentileMapping) return null;
  return {
    asym50_lo: valueToPercentile(intervalSpec.asym50_lo, percentileMapping),
    asym50_hi: valueToPercentile(intervalSpec.asym50_hi, percentileMapping),
    asym85_lo: valueToPercentile(intervalSpec.asym85_lo, percentileMapping),
    asym85_hi: valueToPercentile(intervalSpec.asym85_hi, percentileMapping),
  };
}

function getPercentileCategory(percentile) {
  if (percentile == null) return "unknown";
  if (percentile < 10) return "very_poor";
  if (percentile < 25) return "poor"; 
  if (percentile < 40) return "below_average";
  if (percentile < 60) return "average";
  if (percentile < 75) return "above_average";
  if (percentile < 90) return "good";
  return "excellent";
}

/**
 * Simple heuristic based on external ratings.
 * When both GR and AMZ are available, use sum thresholds.
 * When only one rating is available, use individual thresholds.
 * @param {Object} pageData - Book features
 * @returns {string} "prefer", "avoid", or "neutral"
 */
function getSimpleHeuristic(pageData) {
  const gr = pageData.grRating;
  const amz = pageData.amzRating;
  const hasGR = gr != null && !isNaN(gr);
  const hasAMZ = amz != null && !isNaN(amz);

  if (hasGR && hasAMZ) {
    const sum = gr + amz;
    if (sum >= 9) return "prefer";
    if (sum < 8) return "avoid";
  }

  // Individual rating thresholds (used when one source is missing,
  // or when sum falls in the 8-9 ambiguous range)
  if (hasGR) {
    if (gr >= 4.1) return "prefer";
    if (gr < 3.7) return "avoid";
  }

  if (hasAMZ) {
    if (amz >= 4.3) return "prefer";
    if (amz < 3.8) return "avoid";
  }

  return "neutral";
}

/**
 * Compute a comparable external score regardless of which ratings are available.
 * When both ratings exist, returns their sum (range ~6-10).
 * When only one exists, scales to the same range (doubles it).
 * @param {Object} pageData - Book features
 * @returns {number} External score for sorting
 */
function computeExternalScore(pageData) {
  const gr = pageData.grRating;
  const amz = pageData.amzRating;
  const hasGR = gr != null && !isNaN(gr);
  const hasAMZ = amz != null && !isNaN(amz);

  if (hasGR && hasAMZ) return gr + amz;
  if (hasGR) return gr * 2;
  if (hasAMZ) return amz * 2;
  return 0;
}

/**
 * Run all prediction models on a book's features
 * @param {Object} models - All model specifications
 * @param {Object} pageData - Book features (grRating, amzRating, pageCount, etc.)
 * @param {string} category - Book category/shelf
 * @returns {Object} Predictions, intervals, percentiles, and heuristic
 */
function runAllPredictions(models, pageData, category) {
  const rfCategory = mapToRFCategory(category);
  const group = mapToGroup(category);

  const rfFeatures = {
    year_finished: 2026,
    log_pages: pageData.pageCount ? Math.log(pageData.pageCount + 1) : null,
    book_age: pageData.pubYear ? 2026 - pageData.pubYear : null,
    goodreads_available: pageData.grRating != null ? 1.0 : 0.0,
    goodreads_rating_feature: pageData.grRating,
    goodreads_log_count_feature: pageData.grCount != null ? Math.log10(1 + pageData.grCount) : null,
    Bookshelf: rfCategory,
  };

  const conf = models.conformal || {};
  const groupSpec = (models.ridge_groups || {})[group];

  const preds = {
    rf_enjoy: predictRF(models.rf_enjoy, rfFeatures),
    rf_useful: predictRF(models.rf_useful, rfFeatures),
    gbm_enjoy: predictGBM(models.gbm_enjoy, rfFeatures),
    gbm_useful: predictGBM(models.gbm_useful, rfFeatures),
    ridge_enjoy: groupSpec ? predictGroupRidge(groupSpec, groupSpec.enjoy, pageData.grRating, pageData.amzRating) : null,
    ridge_useful: groupSpec ? predictGroupRidge(groupSpec, groupSpec.useful, pageData.grRating, pageData.amzRating) : null,
    ridge_full_enjoy: groupSpec ? predictPooledRidge(models.ridge_full_enjoy, groupSpec, pageData, category) : null,
    ridge_full_useful: groupSpec ? predictPooledRidge(models.ridge_full_useful, groupSpec, pageData, category) : null,
  };

  // Convert predictions to percentiles using training target distributions.
  const percentiles = {};
  const categories = {};
  const percentileMaps = models.percentile_maps || {};
  for (const key of Object.keys(preds)) {
    percentiles[key] = valueToPercentile(preds[key], percentileMaps[key]);
    categories[key] = getPercentileCategory(percentiles[key]);
  }

  const intervals = {};
  const intervalPercentiles = {};
  for (const key of ["rf_enjoy", "rf_useful", "gbm_enjoy", "gbm_useful", "ridge_full_enjoy", "ridge_full_useful"]) {
    intervals[key] = computeIntervals(preds[key], conf[key]);
    intervalPercentiles[key] = intervalToPercentiles(intervals[key], percentileMaps[key]);
  }
  for (const target of ["enjoy", "useful"]) {
    const ridgeKey = `ridge_${target}`;
    const confSpec = conf[`ridge_${group}_${target}`] || conf[`ridge_pooled_${target}`] || null;
    intervals[ridgeKey] = preds[ridgeKey] != null ? computeIntervals(preds[ridgeKey], confSpec) : null;
    intervalPercentiles[ridgeKey] = intervalToPercentiles(intervals[ridgeKey], percentileMaps[ridgeKey]);
  }

  // Add simple heuristic
  const heuristic = getSimpleHeuristic(pageData);
  
  return {
    ...preds,
    intervals,
    intervalPercentiles,
    percentiles,
    categories,
    heuristic,
    external_sum: computeExternalScore(pageData)
  };
}
