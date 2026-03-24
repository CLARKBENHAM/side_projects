// Pure prediction functions — no extension APIs, works in popup, content script, or Node.
// Loaded as a script tag before popup.js or content.js.

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

function predictGroupRidge(groupSpec, targetSpec, grRating, amzRating) {
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
    if (ratings[feat] == null) ratings[feat] = targetSpec.fill_values[feat];
  }
  let pred = targetSpec.intercept;
  for (const feat of ["goodreads_rating_raw", "openlibrary_rating_raw", "amazon_rating_raw"]) {
    pred += targetSpec.coefficients[feat] * ratings[feat];
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
  };

  const intervals = {};
  for (const key of ["rf_enjoy", "rf_useful", "gbm_enjoy", "gbm_useful"]) {
    intervals[key] = computeIntervals(preds[key], conf[key]);
  }
  for (const target of ["enjoy", "useful"]) {
    const ridgeKey = `ridge_${target}`;
    const confSpec = conf[`ridge_${group}_${target}`] || conf[`ridge_pooled_${target}`] || null;
    intervals[ridgeKey] = preds[ridgeKey] != null ? computeIntervals(preds[ridgeKey], confSpec) : null;
  }

  return { ...preds, intervals };
}
