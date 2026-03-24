let modelsData = null;

async function loadModels() {
  if (modelsData) return modelsData;
  const url = chrome.runtime.getURL("models.json");
  const resp = await fetch(url);
  modelsData = await resp.json();
  return modelsData;
}

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
    const catFeat = `Bookshelf_${cats[i]}`;
    scaled[catFeat] = features.Bookshelf === cats[i] ? 1.0 : 0.0;
  }

  let sum = 0;
  for (const tree of modelSpec.trees) {
    sum += predictTree(tree, scaled);
  }
  const pred = sum / modelSpec.trees.length;
  return Math.max(1.0, Math.min(5.0, pred));
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
    const catFeat = `Bookshelf_${cats[i]}`;
    scaled[catFeat] = features.Bookshelf === cats[i] ? 1.0 : 0.0;
  }

  let pred = modelSpec.init_value;
  for (const tree of modelSpec.trees) {
    pred += modelSpec.learning_rate * predictTree(tree, scaled);
  }
  return Math.max(1.0, Math.min(5.0, pred));
}

function predictRidge(modelSpec, features, imputeOL, imputeAMZ) {
  let pred = modelSpec.intercept;

  const grRating = features.gr_rating ?? modelSpec.gr_mean_impute;

  // Impute OL from GR if missing
  let olRating = features.ol_rating;
  if (olRating == null && imputeOL) {
    olRating = imputeOL.slope * grRating + imputeOL.intercept;
  }
  olRating = olRating ?? modelSpec.ol_mean_impute;

  // Impute AMZ from GR if missing
  let amzRating = features.amz_rating;
  if (amzRating == null && imputeAMZ) {
    amzRating = imputeAMZ.slope * grRating + imputeAMZ.intercept;
  }
  amzRating = amzRating ?? modelSpec.amz_mean_impute;

  // Natural log counts (log1p)
  const logGrCount = features.gr_count != null ? Math.log1p(features.gr_count) : 0;
  const logOlCount = features.ol_count != null ? Math.log1p(features.ol_count) : 0;
  const logAmzCount = features.amz_count != null ? Math.log1p(features.amz_count) : 0;

  const numVals = {
    gr_rating: grRating,
    ol_rating: olRating,
    amz_rating: amzRating,
    log_gr_count: logGrCount,
    log_ol_count: logOlCount,
    log_amz_count: logAmzCount,
  };

  for (const col of modelSpec.numerical_cols) {
    pred += (modelSpec.coefficients[col] || 0) * (numVals[col] ?? 0);
  }

  // Category dummies (prefixed with cat_)
  for (const catCol of modelSpec.category_cols) {
    const catName = catCol.replace("cat_", "");
    pred += (modelSpec.coefficients[catCol] || 0) * (features.category === catName ? 1.0 : 0.0);
  }

  return Math.max(1.0, Math.min(5.0, pred));
}

function runAllPredictions(models, pageData, category) {
  const ridgeCategory = mapToRidgeCategory(category);
  const rfCategory = mapToRFCategory(category);

  const grRating = pageData.grRating;
  const grCount = pageData.grCount;
  const amzRating = pageData.amzRating;
  const amzCount = pageData.amzCount;

  const ridgeFeatures = {
    gr_rating: grRating,
    ol_rating: null,
    amz_rating: amzRating,
    gr_count: grCount,
    ol_count: null,
    amz_count: amzCount,
    category: ridgeCategory,
  };

  const rfFeatures = {
    year_finished: 2026,
    log_pages: pageData.pageCount ? Math.log(pageData.pageCount + 1) : null,
    book_age: pageData.pubYear ? 2026 - pageData.pubYear : null,
    author_target_mean_hist: null,
    author_book_count_hist: 0,
    goodreads_available: grRating != null ? 1.0 : 0.0,
    goodreads_rating_feature: grRating,
    goodreads_log_count_feature: grCount != null ? Math.log10(1 + grCount) : null,
    Bookshelf: rfCategory,
  };

  const imputeOL = models.impute_ol || null;
  const imputeAMZ = models.impute_amz || null;

  return {
    ridge_enjoy: predictRidge(models.ridge_enjoy, ridgeFeatures, imputeOL, imputeAMZ),
    ridge_useful: predictRidge(models.ridge_useful, ridgeFeatures, imputeOL, imputeAMZ),
    rf_enjoy: predictRF(models.rf_enjoy, rfFeatures),
    rf_useful: predictRF(models.rf_useful, rfFeatures),
    gbm_enjoy: predictGBM(models.gbm_enjoy, rfFeatures),
    gbm_useful: predictGBM(models.gbm_useful, rfFeatures),
    inputs: {
      grRating, grCount, amzRating, amzCount,
      ridgeCategory, rfCategory,
      pageCount: pageData.pageCount,
      pubYear: pageData.pubYear,
    },
  };
}

function mapToRidgeCategory(displayCat) {
  // Ridge uses pd.get_dummies(drop_first=False) with categories:
  // Business, Computer Science, General Reading, Literature,
  // Machine Learning, Math, Other, Unknown Shelf, fiction
  const map = {
    "Business": "Business",
    "Business, management": "Business",
    "Computer Science": "Computer Science",
    "General Reading": "General Reading",
    "Histories": "General Reading",
    "Literature": "Literature",
    "Fiction": "fiction",
    "fiction": "fiction",
    "Machine Learning": "Machine Learning",
    "Math": "Math",
    "ML": "Machine Learning",
    "CS": "Computer Science",
  };
  return map[displayCat] || "Other";
}

function mapToRFCategory(displayCat) {
  const map = {
    "Business": "Business, management",
    "Business, management": "Business, management",
    "Computer Science": "Computer Science",
    "General Reading": "General Reading",
    "Histories": "General Reading",
    "Literature": "Literature",
    "Fiction": "fiction",
    "fiction": "fiction",
    "Machine Learning": "Machine Learning",
    "Math": "Math",
    "Advanced Finance": "Business, management",
    "Energy Trading": "Business, management",
    "ML": "Machine Learning",
    "CS": "Computer Science",
    "Unknown Shelf": "General Reading",
  };
  return map[displayCat] || "General Reading";
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "GET_PREDICTIONS") {
    (async () => {
      try {
        const models = await loadModels();
        const results = runAllPredictions(models, msg.pageData, msg.category);
        sendResponse({ success: true, results });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
    })();
    return true;
  }

  if (msg.type === "SEARCH_GOODREADS") {
    (async () => {
      try {
        const query = encodeURIComponent(msg.title + " " + (msg.author || ""));
        const url = `https://www.goodreads.com/search?q=${query}`;
        const resp = await fetch(url);
        const html = await resp.text();
        sendResponse({ success: true, html, url });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
    })();
    return true;
  }

  if (msg.type === "FETCH_GR_PAGE") {
    (async () => {
      try {
        const resp = await fetch(msg.url);
        const html = await resp.text();
        sendResponse({ success: true, html });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
    })();
    return true;
  }

  if (msg.type === "SEARCH_AMAZON") {
    (async () => {
      try {
        const query = encodeURIComponent(msg.title + " " + (msg.author || ""));
        const url = `https://www.amazon.com/s?k=${query}&i=stripbooks`;
        const resp = await fetch(url);
        const html = await resp.text();
        sendResponse({ success: true, html, url });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
    })();
    return true;
  }

  if (msg.type === "FETCH_AMZ_PAGE") {
    (async () => {
      try {
        const resp = await fetch(msg.url, {
          headers: {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
          },
        });
        const html = await resp.text();
        sendResponse({ success: true, html });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
    })();
    return true;
  }
});
