const fs = require('fs');
const assert = require('assert');

// 1. Mock browser APIs
const mockElement = () => ({
    classList: { add: ()=>{}, remove: ()=>{}, toggle: ()=>{} },
    style: {},
    textContent: '',
    getContext: () => ({}),
    querySelector: () => mockElement(),
    querySelectorAll: () => [],
    addEventListener: () => {},
    setAttribute: () => {},
    removeAttribute: () => {},
    dataset: {},
});
global.document = {
    addEventListener: () => {},
    getElementById: () => mockElement(),
    querySelectorAll: () => [],
};
global.window = {};
global.fetch = async () => ({ ok: true, json: async () => ({}) });
global.Papa = { parse: () => {} };
global.Chart = class { constructor() {} destroy() {} };

// 2. Load Predict code
const predictCode = fs.readFileSync(__dirname + '/predict.js', 'utf8');
eval(predictCode);

// 3. Load App code
let appCode = fs.readFileSync(__dirname + '/app.js', 'utf8');
// Replace the top-level let modelsData = null; with actual loaded data so processData can see it
appCode = appCode.replace('let modelsData = null;', 'var ObjectConfig = fs.readFileSync(__dirname + "/models.json", "utf8"); var modelsData = JSON.parse(ObjectConfig);');
eval(appCode);

// 4. Wait, Models are now loaded in the eval scope. We can load them separately for our custom tests.
const modelsDataTest = JSON.parse(fs.readFileSync(__dirname + '/models.json', 'utf8'));

console.log("--- Running Unit Tests ---");

// Test: getVal
assert.strictEqual(getVal(''), null, "empty string should be null");
assert.strictEqual(getVal(null), null, "null should be null");
assert.strictEqual(getVal('4.5'), 4.5, "4.5 string should be 4.5 float");
assert.strictEqual(getVal(0), 0, "0 should be 0");
assert.strictEqual(getVal('0'), 0, "0 string should be 0 float");
console.log("✅ getVal tests passed");

// Test: getSimpleHeuristic
assert.strictEqual(getSimpleHeuristic({ grRating: 4.5, amzRating: 4.6 }), "prefer");  // sum=9.1 >= 9
assert.strictEqual(getSimpleHeuristic({ grRating: 3.5, amzRating: 3.6 }), "avoid");   // sum=7.1 < 8
assert.strictEqual(getSimpleHeuristic({ grRating: 4.0, amzRating: null }), "neutral"); // GR only: 3.7 <= 4.0 < 4.1
assert.strictEqual(getSimpleHeuristic({ grRating: 4.2, amzRating: null }), "prefer");  // GR only: 4.2 >= 4.1
assert.strictEqual(getSimpleHeuristic({ grRating: 3.5, amzRating: null }), "avoid");   // GR only: 3.5 < 3.7
assert.strictEqual(getSimpleHeuristic({ grRating: null, amzRating: 4.5 }), "prefer");  // AMZ only: 4.5 >= 4.3

// Test: computeExternalScore
assert.strictEqual(computeExternalScore({ grRating: 4.0, amzRating: 4.5 }), 8.5);    // both: sum
assert.strictEqual(computeExternalScore({ grRating: 4.0, amzRating: null }), 8.0);    // GR only: doubled
assert.strictEqual(computeExternalScore({ grRating: null, amzRating: 4.5 }), 9.0);    // AMZ only: doubled
assert.strictEqual(computeExternalScore({ grRating: null, amzRating: null }), 0);      // neither: 0
console.log("✅ getSimpleHeuristic & computeExternalScore tests passed");

// Test: mapToGroup
assert.strictEqual(mapToGroup('Fiction'), 'Fiction_Literature');
assert.strictEqual(mapToGroup('Machine Learning'), 'Technical_Other');
assert.strictEqual(mapToGroup('Business'), 'Business_Histories_General');
console.log("✅ mapToGroup tests passed");

// Test: mapToRFCategory
assert.strictEqual(mapToRFCategory('Business'), 'Business, management');
assert.strictEqual(mapToRFCategory('ML'), 'Machine Learning');
assert.strictEqual(mapToRFCategory('Literature'), 'Literature');
console.log("✅ mapToRFCategory tests passed");

// Test: runAllPredictions
const mockPageData = {
    grRating: 4.25,
    amzRating: 4.5,
    grCount: 1000,
    amzCount: 500,
    pageCount: 300,
    pubYear: 2015
};
const preds = runAllPredictions(modelsDataTest, mockPageData, 'Machine Learning');
assert.ok(preds.rf_enjoy >= 1 && preds.rf_enjoy <= 5, "rf_enjoy out of bounds: " + preds.rf_enjoy);
assert.ok(preds.gbm_enjoy >= 1 && preds.gbm_enjoy <= 5, "gbm_enjoy out of bounds");
assert.strictEqual(preds.external_sum, 8.75);
console.log("✅ runAllPredictions tests passed");

// Test: calculateDropCurve
const mockBooks = [
    { trueEnjoy: 3, trueUseful: 3, preds: { rf_enjoy: 2.5, rf_useful: 2.5 } }, 
    { trueEnjoy: 5, trueUseful: 5, preds: { rf_enjoy: 4.8, rf_useful: 4.8 } },
    { trueEnjoy: 4, trueUseful: 4, preds: { rf_enjoy: 3.5, rf_useful: 3.5 } },
    { trueEnjoy: 2, trueUseful: 2, preds: { rf_enjoy: 1.0, rf_useful: 1.0 } }  
];
const curve = calculateDropCurve(mockBooks, 'rf', 'trueEnjoy');

// Baseline is 14/4 = 3.5
// at 0%, drop 0 books -> kept = 4, avg = 3.5, gain = 0
const gain0 = curve.find(p => p.x === 0).y;
assert.strictEqual(gain0, 0);

// at 25%, drop 1 book -> (model:1.0). Kept = [3,4,5], avg = 4.0, gain = 0.5
const gain25 = curve.find(p => p.x === 25).y;
assert.strictEqual(gain25, 0.5);

// at 50%, drop 2 books -> (model:1.0, 2.5). Kept = [4,5], avg = 4.5, gain = 1.0
const gain50 = curve.find(p => p.x === 50).y;
assert.strictEqual(gain50, 1.0);

console.log("✅ calculateDropCurve tests passed");

// Test: normalRandom produces reasonable distribution
{
    const samples = [];
    for (let i = 0; i < 10000; i++) samples.push(normalRandom());
    const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
    const variance = samples.reduce((s, v) => s + (v - mean) ** 2, 0) / samples.length;
    assert.ok(Math.abs(mean) < 0.05, `normalRandom mean should be ~0, got ${mean}`);
    assert.ok(Math.abs(variance - 1) < 0.1, `normalRandom variance should be ~1, got ${variance}`);
    console.log("✅ normalRandom distribution tests passed");
}

// Test: monteCarloNoiseBands returns valid envelopes
{
    const mcBooks = [
        { trueEnjoy: 2, trueUseful: 2, preds: { rf_enjoy: 1.5, rf_useful: 1.5 } },
        { trueEnjoy: 3, trueUseful: 3, preds: { rf_enjoy: 2.5, rf_useful: 2.5 } },
        { trueEnjoy: 4, trueUseful: 4, preds: { rf_enjoy: 3.5, rf_useful: 3.5 } },
        { trueEnjoy: 5, trueUseful: 5, preds: { rf_enjoy: 4.5, rf_useful: 4.5 } },
    ];
    const { p10, p90 } = monteCarloNoiseBands(mcBooks, 'rf', 'trueEnjoy', 0.5, 200);
    assert.ok(p10.length > 0, "p10 should have entries");
    assert.ok(p90.length > 0, "p90 should have entries");
    assert.strictEqual(p10.length, p90.length, "p10 and p90 should have same length");

    // At 0% drop, gain should be near 0 (noise centered on 0)
    const p10at0 = p10.find(pt => pt.x === 0);
    const p90at0 = p90.find(pt => pt.x === 0);
    assert.ok(Math.abs(p10at0.y) < 0.3, `p10 at 0% should be near 0, got ${p10at0.y}`);
    assert.ok(Math.abs(p90at0.y) < 0.3, `p90 at 0% should be near 0, got ${p90at0.y}`);

    // p10 <= p90 at every drop %
    for (let i = 0; i < p10.length; i++) {
        assert.ok(p10[i].y <= p90[i].y + 0.001,
            `p10 should <= p90 at ${p10[i].x}%: ${p10[i].y} vs ${p90[i].y}`);
    }

    // Oracle: noise should widen the band (degraded sort)
    const { p10: op10, p90: op90 } = monteCarloNoiseBands(mcBooks, 'oracle', 'trueEnjoy', 0.5, 200);
    const oracleSpread50 = (op90.find(pt => pt.x === 50)?.y ?? 0) - (op10.find(pt => pt.x === 50)?.y ?? 0);
    assert.ok(oracleSpread50 > 0, "Oracle noise band should have positive width at 50% drop");

    console.log("✅ monteCarloNoiseBands tests passed");
}

// Test: source column extraction
{
    const sourceRow = {
        title: 'Test Book',
        goodreads_rating: '4.0',
        amazon_rating_consensus: '4.5',
        avg_enjoyment: '3.5',
        avg_usefulness: '3.0',
        Bookshelf: 'Business',
        source: 'Holdout 2026'
    };
    // Simulate the source extraction logic from processData
    const extractedSource = sourceRow.source || sourceRow.Source || null;
    assert.strictEqual(extractedSource, 'Holdout 2026', "Should extract source column");

    const noSourceRow = { title: 'No Source', goodreads_rating: '4.0', avg_enjoyment: '3.0' };
    const noSource = noSourceRow.source || noSourceRow.Source || null;
    assert.strictEqual(noSource, null, "Missing source should be null");
    console.log("✅ source column extraction tests passed");
}

// Test: detectColumn
{
    const headers = ['Title', 'avg_enjoyment', 'Goodreads Rating', 'page_count', 'source'];

    assert.strictEqual(detectColumn(headers, ['enjoyment', 'avg_enjoyment']), 'avg_enjoyment', "Exact match");
    assert.strictEqual(detectColumn(headers, ['goodreads_rating', 'gr_rating']), 'Goodreads Rating', "Partial match");
    assert.strictEqual(detectColumn(headers, ['pages', 'page_count']), 'page_count', "Match second pattern");
    assert.strictEqual(detectColumn(headers, ['missing', 'not_there']), null, "No match returns null");

    console.log("✅ detectColumn tests passed");
}

console.log("--- Running Integration Tests ---");
const csvContent = fs.readFileSync('/Users/clarkbenham/side_projects/ai_books_tracking/golden_master_multi_source.csv', 'utf8');
const lines = csvContent.split('\n').filter(l => l.trim().length > 0);
const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));

const parsedRows = [];
const re = /,(?=(?:(?:[^"]*"){2})*[^"]*$)/;

for (let i = 1; i < 50; i++) {
    if(!lines[i]) continue;
    const values = lines[i].split(re).map(v => v.trim().replace(/^"|"$/g, ''));
    if (values.length !== headers.length) continue;

    let row = {};
    for (let j=0; j<headers.length; j++) {
        row[headers[j]] = values[j];
    }
    parsedRows.push(row);
}

try {
    // Set up column mapping for the golden_master CSV
    columnMapping = {
        enjoyment: 'avg_enjoyment',
        usefulness: 'avg_usefulness',
        grRating: 'goodreads_rating',
        amzRating: 'amazon_rating_consensus',
        grCount: 'goodreads_rating_count',
        amzCount: 'amazon_review_count_consensus',
        pageCount: 'page_count',
        pubYear: 'pub_year',
        category: 'Bookshelf',
        source: 'source',
        title: 'title',
    };

    // Debug: check first row
    const firstRow = parsedRows[0];
    const hasEnjoyment = firstRow[columnMapping.enjoyment] !== undefined && firstRow[columnMapping.enjoyment] !== '';
    const hasGR = firstRow[columnMapping.grRating] !== undefined && firstRow[columnMapping.grRating] !== '';
    if (!hasEnjoyment || !hasGR) {
        console.error('Debug: First row missing data. Enjoyment:', firstRow[columnMapping.enjoyment], 'GR:', firstRow[columnMapping.grRating]);
        console.error('Available keys:', Object.keys(firstRow).slice(0, 20));
    }

    processData(parsedRows);
    console.log("✅ processData integration test passed (Did not throw on real CSV slice)");
} catch(e) {
    console.error("❌ processData integration test failed", e);
    process.exit(1);
}

console.log("\nAll Tests Verify Successfully!");
