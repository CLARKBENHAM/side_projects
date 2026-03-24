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
      for (let i = 0; i < vals.length; i++) imputed += step.coefs[i] * vals[i];
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
  return Math.max(1, Math.min(5, pred));
}

function mapToGroup(cat) {
  if (["fiction", "Literature"].includes(cat)) return "Fiction_Literature";
  if (["Computer Science", "Machine Learning", "Math", "Unknown Shelf"].includes(cat)) return "Technical_Other";
  return "Business_Histories_General";
}

const books = [
  { name: "Skunk Works", gr: 4.46, amz: 4.7, grC: 17423, amzC: 4738, cat: "Histories", shelf: "Histories", pages: 372, year: 1994 },
  { name: "A Pattern Language", gr: 4.42, amz: 4.7, grC: 5479, amzC: 1036, cat: "General Reading", shelf: "General Reading", pages: 1171, year: 1977 },
  { name: "Mastering Tech Sales", gr: 4.8, amz: 4.8, grC: 50, amzC: 100, cat: "Business", shelf: "Business, management", pages: 300, year: 2010 },
  { name: "Low-rated CS book", gr: 3.4, amz: 3.8, grC: 5, amzC: 4, cat: "Computer Science", shelf: "Computer Science", pages: 200, year: 2020 },
  { name: "High-rated Math", gr: 4.5, amz: 4.6, grC: 2000, amzC: 500, cat: "Math", shelf: "Math", pages: 400, year: 2000 },
  { name: "Mediocre fiction", gr: 3.7, amz: 4.2, grC: 50000, amzC: 10000, cat: "fiction", shelf: "fiction", pages: 350, year: 2015 },
  { name: "ML textbook", gr: 4.3, amz: 4.5, grC: 3000, amzC: 800, cat: "Machine Learning", shelf: "Machine Learning", pages: 500, year: 2018 },
];

console.log("=== Book Rating Predictor — Model Validation ===\n");
console.log("RF/GBM now trained WITHOUT author features.");
console.log("Ridge now uses per-group models (3 groups × 3 raw ratings).\n");

const hdr = `${"Book".padEnd(25)} ${"Ridge E".padStart(8)} ${"Ridge U".padStart(8)} | ${"RF E".padStart(6)} ${"RF U".padStart(6)} | ${"GBM E".padStart(6)} ${"GBM U".padStart(6)} | Group`;
console.log(hdr);
console.log("-".repeat(hdr.length));

const rg = models.ridge_groups;

for (const b of books) {
  const group = mapToGroup(b.cat);
  const gs = rg[group];

  const re = gs ? predictGroupRidge(gs, gs.enjoy, b.gr, b.amz) : NaN;
  const ru = gs ? predictGroupRidge(gs, gs.useful, b.gr, b.amz) : NaN;

  const rfF = {
    year_finished: 2026, log_pages: Math.log(b.pages + 1), book_age: 2026 - b.year,
    goodreads_available: 1, goodreads_rating_feature: b.gr,
    goodreads_log_count_feature: Math.log10(1 + b.grC), Bookshelf: b.shelf,
  };
  const rfe = predictRF(models.rf_enjoy, rfF);
  const rfu = predictRF(models.rf_useful, rfF);
  const ge = predictGBM(models.gbm_enjoy, rfF);
  const gu = predictGBM(models.gbm_useful, rfF);

  console.log(
    `${b.name.padEnd(25)} ${re.toFixed(2).padStart(8)} ${ru.toFixed(2).padStart(8)} | ${rfe.toFixed(2).padStart(6)} ${rfu.toFixed(2).padStart(6)} | ${ge.toFixed(2).padStart(6)} ${gu.toFixed(2).padStart(6)} | ${group}`
  );
}

// Conformal intervals
const conf = models.conformal;
if (conf) {
  console.log("\n=== Conformal Prediction Intervals ===");
  for (const key of Object.keys(conf).sort()) {
    const c = conf[key];
    const asym50 = `[${c.lo50.toFixed(2)}, ${c.hi50.toFixed(2)}]`;
    const asym85 = `[${c.lo85.toFixed(2)}, ${c.hi85.toFixed(2)}]`;
    console.log(
      `${key.padEnd(35)} n=${c.n_calibration.toString().padStart(3)}  50%q=${c.q50.toFixed(3)}  85%q=${c.q85.toFixed(3)}  asym50=${asym50.padStart(16)}  asym85=${asym85.padStart(16)}`
    );
  }
}
