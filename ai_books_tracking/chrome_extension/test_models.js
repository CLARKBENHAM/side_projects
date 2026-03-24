const fs = require("fs");
const path = require("path");

const models = JSON.parse(
  fs.readFileSync(path.join(__dirname, "models.json"), "utf-8")
);

function predictTree(tree, features) {
  if ("value" in tree) return tree.value;
  const val = features[tree.feature];
  if (val === undefined || val === null) return predictTree(tree.left, features);
  return val <= tree.threshold ? predictTree(tree.left, features) : predictTree(tree.right, features);
}

function predictRF(spec, features) {
  const scaled = {};
  for (const f of spec.numeric_features) {
    let v = features[f]; if (v == null || isNaN(v)) v = spec.medians[f];
    scaled[f] = (v - spec.scaler_means[f]) / spec.scaler_scales[f];
  }
  const cats = spec.cat_categories[0], drop = spec.cat_drop_idx[0];
  for (let i = 0; i < cats.length; i++) {
    if (i === drop) continue;
    scaled[`Bookshelf_${cats[i]}`] = features.Bookshelf === cats[i] ? 1 : 0;
  }
  let sum = 0;
  for (const t of spec.trees) sum += predictTree(t, scaled);
  return Math.max(1, Math.min(5, sum / spec.trees.length));
}

function predictGBM(spec, features) {
  const scaled = {};
  for (const f of spec.numeric_features) {
    let v = features[f]; if (v == null || isNaN(v)) v = spec.medians[f];
    scaled[f] = (v - spec.scaler_means[f]) / spec.scaler_scales[f];
  }
  const cats = spec.cat_categories[0], drop = spec.cat_drop_idx[0];
  for (let i = 0; i < cats.length; i++) {
    if (i === drop) continue;
    scaled[`Bookshelf_${cats[i]}`] = features.Bookshelf === cats[i] ? 1 : 0;
  }
  let pred = spec.init_value;
  for (const t of spec.trees) pred += spec.learning_rate * predictTree(t, scaled);
  return Math.max(1, Math.min(5, pred));
}

function predictRidge(spec, grRating, amzRating, grCount, amzCount, category) {
  const imputeOL = models.impute_ol || null;
  const imputeAMZ = models.impute_amz || null;

  let pred = spec.intercept;
  const gr = grRating ?? spec.gr_mean_impute;

  let ol = null;
  if (imputeOL) ol = imputeOL.slope * gr + imputeOL.intercept;
  ol = ol ?? spec.ol_mean_impute;

  let amz = amzRating;
  if (amz == null && imputeAMZ) amz = imputeAMZ.slope * gr + imputeAMZ.intercept;
  amz = amz ?? spec.amz_mean_impute;

  const logGr = grCount != null ? Math.log1p(grCount) : 0;
  const logOl = 0;
  const logAmz = amzCount != null ? Math.log1p(amzCount) : 0;

  const numVals = { gr_rating: gr, ol_rating: ol, amz_rating: amz, log_gr_count: logGr, log_ol_count: logOl, log_amz_count: logAmz };
  for (const col of spec.numerical_cols) {
    pred += (spec.coefficients[col] || 0) * (numVals[col] ?? 0);
  }
  for (const catCol of spec.category_cols) {
    const catName = catCol.replace("cat_", "");
    pred += (spec.coefficients[catCol] || 0) * (category === catName ? 1 : 0);
  }
  return Math.max(1, Math.min(5, pred));
}

// Test a range of books
const books = [
  { name: "A Pattern Language", gr: 4.42, amz: 4.7, grC: 5479, amzC: 1036, cat: "General Reading", shelf: "General Reading", pages: 1171, year: 1977 },
  { name: "Mastering Tech Sales", gr: 4.8, amz: 4.8, grC: 50, amzC: 100, cat: "Business", shelf: "Business, management", pages: 300, year: 2010 },
  { name: "Low-rated CS book", gr: 3.4, amz: 3.8, grC: 5, amzC: 4, cat: "Computer Science", shelf: "Computer Science", pages: 200, year: 2020 },
  { name: "High-rated Math", gr: 4.5, amz: 4.6, grC: 2000, amzC: 500, cat: "Math", shelf: "Math", pages: 400, year: 2000 },
  { name: "Mediocre fiction", gr: 3.7, amz: 4.2, grC: 50000, amzC: 10000, cat: "fiction", shelf: "fiction", pages: 350, year: 2015 },
  { name: "ML textbook", gr: 4.3, amz: 4.5, grC: 3000, amzC: 800, cat: "Machine Learning", shelf: "Machine Learning", pages: 500, year: 2018 },
];

console.log("=== Book Rating Predictor — Model Validation ===\n");
console.log(`${"Book".padEnd(25)} ${"Ridge E".padStart(8)} ${"Ridge U".padStart(8)} | ${"RF E".padStart(6)} ${"RF U".padStart(6)} | ${"GBM E".padStart(6)} ${"GBM U".padStart(6)}`);
console.log("-".repeat(85));

for (const b of books) {
  const rfF = {
    year_finished: 2026, log_pages: Math.log(b.pages + 1), book_age: 2026 - b.year,
    author_target_mean_hist: null, author_book_count_hist: 0,
    goodreads_available: 1, goodreads_rating_feature: b.gr,
    goodreads_log_count_feature: Math.log10(1 + b.grC), Bookshelf: b.shelf,
  };

  const re = predictRidge(models.ridge_enjoy, b.gr, b.amz, b.grC, b.amzC, b.cat);
  const ru = predictRidge(models.ridge_useful, b.gr, b.amz, b.grC, b.amzC, b.cat);
  const rfe = predictRF(models.rf_enjoy, rfF);
  const rfu = predictRF(models.rf_useful, rfF);
  const ge = predictGBM(models.gbm_enjoy, rfF);
  const gu = predictGBM(models.gbm_useful, rfF);

  console.log(
    `${b.name.padEnd(25)} ${re.toFixed(2).padStart(8)} ${ru.toFixed(2).padStart(8)} | ${rfe.toFixed(2).padStart(6)} ${rfu.toFixed(2).padStart(6)} | ${ge.toFixed(2).padStart(6)} ${gu.toFixed(2).padStart(6)}`
  );
}

console.log("\n=== Expected patterns ===");
console.log("- Math should score highest on enjoyment");
console.log("- fiction should score lowest on usefulness");
console.log("- CS should have low enjoyment offset");
console.log("- Higher GR rating → higher predictions within category");
console.log("- Business should have moderate scores (was baseline in old model, now has own offset)");
