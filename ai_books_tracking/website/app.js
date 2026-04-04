/**
 * Book Analysis Web Application
 * Analyzes uploaded CSV data to show utility gains from dropping low-rated books.
 */

// Constants
const DROP_PERCENTAGES = { MIN: 0, MAX: 95, STEP: 5 };
const REQUIRED_CSV_COLUMNS = ['avg_enjoyment', 'goodreads_rating', 'amazon_rating'];
const MIN_BOOKS_FOR_ANALYSIS = 5;

const CHART_COLORS = {
    perfect: { bc: 'rgba(255, 255, 255, 0.4)', bg: 'rgba(255, 255, 255, 0.05)', dash: [5, 5] },
    ridge: { bc: '#818cf8', bg: 'rgba(129, 140, 248, 0.2)', dash: [] },
    rf: { bc: '#2dd4bf', bg: 'rgba(45, 212, 191, 0.2)', dash: [] },
    heuristic: { bc: '#fbbf24', bg: 'rgba(251, 191, 36, 0.2)', dash: [] }
};

// State
let modelsData = null;
let allProcessedBooks = null;
let parsedCSV = null;
let columnMapping = null;
const chartInstances = {};
let currentWeight = 0.5;
let currentDropPct = 50;
let currentTableModel = 'oracle';
let currentSortCol = 'wtPred';
let currentSortDir = 'asc';

/**
 * Initialize application - load models and setup file uploaders
 */
document.addEventListener('DOMContentLoaded', async () => {
    const errorEl = document.getElementById('file-error');

    try {
        const response = await fetch('models.json');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: Failed to load models.json`);
        }

        modelsData = await response.json();

        if (!validateModelsData(modelsData)) {
            throw new Error('Invalid models.json structure');
        }

        console.log('Successfully loaded model definitions.', Object.keys(modelsData));
        setupUploaders();
        setupControls();
    } catch (e) {
        console.error('Model loading error:', e);
        errorEl.textContent = 'Error: Could not load models.json. Ensure it is in the same directory and hosted via a server.';
        errorEl.classList.remove('hidden');
    }
});

/**
 * Validate that models.json has required structure
 * @param {Object} models - The loaded models data
 * @returns {boolean} True if valid
 */
function validateModelsData(models) {
    const requiredKeys = ['rf_enjoy', 'rf_useful', 'ridge_groups'];
    return requiredKeys.every(key => key in models);
}

/**
 * Setup drag-and-drop and file input handlers
 */
function setupUploaders() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('csv-file');

    if (!dropArea || !fileInput) {
        console.error('Required DOM elements not found');
        return;
    }

    const preventDefaults = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
            dropArea.setAttribute('aria-dropeffect', 'copy');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
            dropArea.removeAttribute('aria-dropeffect');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    }, false);

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });
}

/**
 * Setup UI controls like slider and dropdown
 */
function setupControls() {
    const slider = document.getElementById('weight-slider');
    const display = document.getElementById('weight-display');
    const dropSelect = document.getElementById('drop-percent-select');

    if (slider && display) {
        slider.addEventListener('input', (e) => {
            currentWeight = parseFloat(e.target.value);
            const enjoyPct = Math.round(currentWeight * 100);
            const usefulPct = Math.round((1 - currentWeight) * 100);
            display.textContent = `${enjoyPct}% Enjoyment | ${usefulPct}% Usefulness`;
        });
        
        slider.addEventListener('change', () => {
             if (allProcessedBooks) {
                 const activeBtn = document.querySelector('.toggle-btn.active');
                 const subset = activeBtn && activeBtn.dataset ? activeBtn.dataset.subset : 'all';
                 renderForSubset(subset);
             }
        });
    }

    if (dropSelect) {
        dropSelect.addEventListener('change', (e) => {
            currentDropPct = parseInt(e.target.value, 10);
            updateTableData();
        });
    }

    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            currentTableModel = e.target.value;
            updateTableData();
        });
    }

    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (currentSortCol === col) {
                currentSortDir = currentSortDir === 'asc' ? 'desc' : 'asc';
            } else {
                currentSortCol = col;
                currentSortDir = 'desc'; // default to desc for most numerical sorts
            }
            updateTableData();
        });
    });
}

function updateTableData() {
    if (allProcessedBooks) {
        const activeBtn = document.querySelector('.toggle-btn.active');
        const subset = activeBtn && activeBtn.dataset ? activeBtn.dataset.subset : 'all';
        const books = subset === 'holdout' 
            ? allProcessedBooks.filter(b => b.source === 'Holdout 2026') 
            : allProcessedBooks;
        renderDroppedBooksTable(books);
    }
}

/**
 * Handle uploaded CSV file
 * @param {File} file - The uploaded file
 */
function handleFile(file) {
    if (!file) {
        showError('No file selected.');
        return;
    }

    if (!file.name.endsWith('.csv')) {
        showError('Please select a valid CSV file.');
        return;
    }

    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        showError('File too large. Maximum size is 50MB.');
        return;
    }

    hideError();
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('upload-zone').style.display = 'none';

    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors && results.errors.length > 0) {
                console.warn('CSV parsing warnings:', results.errors);
            }
            parsedCSV = results.data;
            showColumnMapper(results.data);
        },
        error: function(err) {
            showError('Error parsing CSV: ' + err.message);
            resetUI();
        }
    });
}

/**
 * Show column mapping interface for user to map CSV columns to expected fields
 * @param {Array<Object>} data - Parsed CSV data
 */
function showColumnMapper(data) {
    if (!data || data.length === 0) {
        showError('CSV file is empty.');
        resetUI();
        return;
    }

    const headers = Object.keys(data[0]);

    document.getElementById('loading').classList.add('hidden');
    document.getElementById('mapper-zone').classList.remove('hidden');

    // Populate dropdowns
    populateColumnDropdown('map-enjoyment', headers, detectColumn(headers, ['enjoyment', 'avg_enjoyment', 'enjoy']));
    populateColumnDropdown('map-usefulness', headers, detectColumn(headers, ['usefulness', 'avg_usefulness', 'useful', 'utility']));
    populateColumnDropdown('map-gr-rating', headers, detectColumn(headers, ['goodreads_rating', 'gr_rating', 'goodreads', 'gr']));
    populateColumnDropdown('map-amz-rating', headers, detectColumn(headers, ['amazon_rating', 'amazon_rating_consensus', 'amz_rating', 'amazon', 'amz']));
    populateColumnDropdown('map-gr-count', headers, detectColumn(headers, ['goodreads_rating_count', 'gr_count', 'goodreads_count']), true);
    populateColumnDropdown('map-amz-count', headers, detectColumn(headers, ['amazon_review_count', 'amazon_review_count_consensus', 'amz_count']), true);
    populateColumnDropdown('map-page-count', headers, detectColumn(headers, ['page_count', 'pages', 'num_pages']), true);
    populateColumnDropdown('map-pub-year', headers, detectColumn(headers, ['pub_year', 'year', 'publication_year', 'original_publication_year']), true);
    populateColumnDropdown('map-category', headers, detectColumn(headers, ['bookshelf', 'category', 'genre', 'shelf']), true);
    populateColumnDropdown('map-source', headers, detectColumn(headers, ['source', 'dataset', 'split']), true);
    populateColumnDropdown('map-title', headers, detectColumn(headers, ['title', 'book', 'name']), true);

    document.getElementById('total-rows').textContent = data.length;
}

/**
 * Detect which column name matches expected patterns (case-insensitive, ignores underscores/spaces)
 * @param {Array<string>} headers - Available column names
 * @param {Array<string>} patterns - Patterns to match against
 * @returns {string|null} - Matched column name or null
 */
function detectColumn(headers, patterns) {
    const normalize = (s) => s.toLowerCase().replace(/[_\s]/g, '');
    const normalizedHeaders = headers.map(h => normalize(h));

    for (const pattern of patterns) {
        const normalizedPattern = normalize(pattern);

        // Exact match (ignoring underscores/spaces)
        const exactIdx = normalizedHeaders.indexOf(normalizedPattern);
        if (exactIdx !== -1) return headers[exactIdx];

        // Partial match: pattern appears as substring in header
        const partialIdx = normalizedHeaders.findIndex(h => h.includes(normalizedPattern));
        if (partialIdx !== -1) return headers[partialIdx];
    }
    return null;
}

/**
 * Populate a column mapping dropdown
 * @param {string} selectId - ID of the select element
 * @param {Array<string>} headers - Available column names
 * @param {string|null} defaultVal - Default selected value
 * @param {boolean} optional - Whether this column is optional
 */
function populateColumnDropdown(selectId, headers, defaultVal, optional = false) {
    const select = document.getElementById(selectId);
    if (!select) return;

    select.innerHTML = '';

    if (optional) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '(none)';
        select.appendChild(opt);
    }

    for (const header of headers) {
        const opt = document.createElement('option');
        opt.value = header;
        opt.textContent = header;
        if (header === defaultVal) opt.selected = true;
        select.appendChild(opt);
    }
}

/**
 * Handle "Analyze Data" button click - validate mapping and process data
 */
function handleAnalyzeClick() {
    const mapping = {
        enjoyment: document.getElementById('map-enjoyment').value,
        usefulness: document.getElementById('map-usefulness').value,
        grRating: document.getElementById('map-gr-rating').value,
        amzRating: document.getElementById('map-amz-rating').value,
        grCount: document.getElementById('map-gr-count').value || null,
        amzCount: document.getElementById('map-amz-count').value || null,
        pageCount: document.getElementById('map-page-count').value || null,
        pubYear: document.getElementById('map-pub-year').value || null,
        category: document.getElementById('map-category').value || null,
        source: document.getElementById('map-source').value || null,
        title: document.getElementById('map-title').value || null,
    };

    // Validation: at least one target and one rating required
    if (!mapping.enjoyment && !mapping.usefulness) {
        showError('Please map at least one target column (Enjoyment or Usefulness).');
        return;
    }

    if (!mapping.grRating && !mapping.amzRating) {
        showError('Please map at least one rating column (Goodreads or Amazon).');
        return;
    }

    hideError();
    columnMapping = mapping;

    document.getElementById('mapper-zone').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');

    setTimeout(() => processData(parsedCSV), 100);
}

/**
 * Display error message to user
 * @param {string} msg - Error message to display
 */
function showError(msg) {
    const errorEl = document.getElementById('file-error');
    if (errorEl) {
        errorEl.textContent = msg;
        errorEl.classList.remove('hidden');
        errorEl.setAttribute('role', 'alert');
    }
}

/**
 * Hide error message
 */
function hideError() {
    const errorEl = document.getElementById('file-error');
    if (errorEl) {
        errorEl.classList.add('hidden');
        errorEl.removeAttribute('role');
    }
}

/**
 * Reset UI to initial upload state
 */
function resetUI() {
    const loadingEl = document.getElementById('loading');
    const uploadEl = document.getElementById('upload-zone');
    const mapperEl = document.getElementById('mapper-zone');

    if (loadingEl) loadingEl.classList.add('hidden');
    if (mapperEl) mapperEl.classList.add('hidden');
    if (uploadEl) uploadEl.style.display = 'block';
}

/**
 * Parse and validate a numeric value from CSV
 * @param {*} val - Value to parse
 * @returns {number|null} Parsed float or null
 */
function getVal(val) {
    if (val === undefined || val === null || val === '') return null;
    const p = parseFloat(val);
    return isNaN(p) ? null : p;
}

/**
 * Process parsed CSV data into predictions
 * @param {Array<Object>} rows - Parsed CSV rows
 */
function processData(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
        showError('CSV file is empty or invalid.');
        resetUI();
        return;
    }

    const processed = [];
    const skipped = { noRatings: 0, noTargets: 0 };

    rows.forEach((row, index) => {
        // Extract features using column mapping
        const pageData = {
             grRating: columnMapping?.grRating ? getVal(row[columnMapping.grRating]) : null,
             amzRating: columnMapping?.amzRating ? getVal(row[columnMapping.amzRating]) : null,
             grCount: columnMapping?.grCount ? getVal(row[columnMapping.grCount]) : null,
             amzCount: columnMapping?.amzCount ? getVal(row[columnMapping.amzCount]) : null,
             pageCount: columnMapping?.pageCount ? getVal(row[columnMapping.pageCount]) : null,
             pubYear: columnMapping?.pubYear ? getVal(row[columnMapping.pubYear]) : null,
        };

        const category = (columnMapping?.category ? row[columnMapping.category] : null) || 'General Reading';

        // Extract true targets to evaluate against
        const trueEnjoy = columnMapping?.enjoyment ? getVal(row[columnMapping.enjoyment]) : null;
        const trueUseful = columnMapping?.usefulness ? getVal(row[columnMapping.usefulness]) : null;

        // Ensure row has at least the basic features
        if (pageData.grRating === null && pageData.amzRating === null) {
            skipped.noRatings++;
            return;
        }

        if (trueEnjoy === null && trueUseful === null) {
            skipped.noTargets++;
            return;
        }

        try {
            // Call inference functions from predict.js
            const preds = runAllPredictions(modelsData, pageData, category);

            processed.push({
                title: (columnMapping?.title ? row[columnMapping.title] : null) || 'Unknown',
                category: category,
                source: columnMapping?.source ? row[columnMapping.source] : null,
                pageData: pageData,
                trueEnjoy: trueEnjoy,
                trueUseful: trueUseful,
                preds: preds
            });
        } catch (error) {
            console.error(`Error processing row ${index}:`, error);
        }
    });

    console.log(`Processed ${processed.length} books. Skipped: ${skipped.noRatings} missing ratings, ${skipped.noTargets} missing targets.`);

    if (processed.length < MIN_BOOKS_FOR_ANALYSIS) {
        showError(`Insufficient valid data. Found ${processed.length} books, need at least ${MIN_BOOKS_FOR_ANALYSIS}. ` +
                  `Make sure headers match expected format (avg_enjoyment, goodreads_rating, etc).`);
        resetUI();
        return;
    }

    allProcessedBooks = processed;
    setupSubsetToggle(processed);
    renderForSubset('all');
}

// Labeling noise MAE (from double-rating validation)
const LABELING_MAE = { enjoy: 0.46, useful: 0.34 };
// Convert MAE to approximate SD: SD ≈ MAE / 0.8
const LABELING_SD = { enjoy: LABELING_MAE.enjoy / 0.8, useful: LABELING_MAE.useful / 0.8 };
const BOOTSTRAP_ITERATIONS = 2000;
const NOISE_ITERATIONS = 500;

/**
 * Generate a standard normal random number using Box-Muller transform
 * @returns {number} A sample from N(0, 1)
 */
function normalRandom() {
    let u1, u2;
    do { u1 = Math.random(); } while (u1 === 0);
    u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Monte Carlo simulation of rating-noise uncertainty bands.
 * For each iteration: add Gaussian noise to true ratings, re-sort (Oracle uses noisy
 * ratings; ML uses original predictions), compute gain against noisy labels.
 * Returns 10th and 90th percentile envelopes (80% CI).
 * @param {Array<Object>} books - Book data
 * @param {string} predictKey - Key for sorting (trueKey = Oracle, else model pred)
 * @param {string} trueKey - Key for true ratings on book object (e.g. 'trueEnjoy')
 * @param {number} labelingSD - Standard deviation of rating noise
 * @param {number} nIterations - Number of Monte Carlo iterations
 * @returns {{ p10: Array<{x,y}>, p90: Array<{x,y}> }}
 */
function monteCarloNoiseBands(books, strategy, trueKey, labelingSD, nIterations) {
    const N = books.length;
    const dropPcts = [];
    for (let d = DROP_PERCENTAGES.MIN; d <= DROP_PERCENTAGES.MAX; d += DROP_PERCENTAGES.STEP) {
        dropPcts.push(d);
    }

    const samples = {};
    for (const d of dropPcts) samples[d] = [];

    for (let iter = 0; iter < nIterations; iter++) {
        // Pre-compute noisy ratings for this iteration
        const iterNoisyEnjoy = books.map(b => b.trueEnjoy !== null ? b.trueEnjoy + normalRandom() * LABELING_SD.enjoy : 0);
        const iterNoisyUseful = books.map(b => b.trueUseful !== null ? b.trueUseful + normalRandom() * LABELING_SD.useful : 0);
        
        const noisyTrueMeasure = trueKey === 'trueEnjoy' ? iterNoisyEnjoy : iterNoisyUseful;

        const indices = Array.from({length: N}, (_, i) => i);
        indices.sort((a, b) => {
            if (strategy === 'oracle') {
                 const valA = (currentWeight * iterNoisyEnjoy[a]) + ((1 - currentWeight) * iterNoisyUseful[a]);
                 const valB = (currentWeight * iterNoisyEnjoy[b]) + ((1 - currentWeight) * iterNoisyUseful[b]);
                 return valA - valB;
            } else {
                 return getWeightedSortValue(books[a], strategy) - getWeightedSortValue(books[b], strategy);
            }
        });

        const baselineNoisy = noisyTrueMeasure.reduce((sum, v) => sum + v, 0) / N;

        for (const d of dropPcts) {
            const dropCount = Math.floor(N * (d / 100));
            const keptIndices = indices.slice(dropCount);
            if (keptIndices.length === 0) {
                samples[d].push(0);
                continue;
            }
            const keptAvg = keptIndices.reduce((sum, i) => sum + noisyTrueMeasure[i], 0) / keptIndices.length;
            samples[d].push(keptAvg - baselineNoisy);
        }
    }

    const p10 = [], p90 = [];
    for (const d of dropPcts) {
        const vals = samples[d].sort((a, b) => a - b);
        const n = vals.length;
        p10.push({ x: d, y: vals[Math.floor(n * 0.10)] });
        p90.push({ x: d, y: vals[Math.floor(n * 0.90)] });
    }
    return { p10, p90 };
}

/**
 * Setup holdout/in-sample toggle buttons. Hidden when CSV lacks a source column.
 * @param {Array<Object>} books - All processed books
 */
function setupSubsetToggle(books) {
    const toggleEl = document.getElementById('subset-toggle');
    if (!toggleEl) return;

    const holdoutCount = books.filter(b => b.source === 'Holdout 2026').length;
    if (holdoutCount === 0) {
        toggleEl.classList.add('hidden');
        return;
    }

    toggleEl.classList.remove('hidden');

    const allBtn = toggleEl.querySelector('[data-subset="all"]');
    const holdoutBtn = toggleEl.querySelector('[data-subset="holdout"]');
    if (allBtn) allBtn.textContent = `All Data \u2014 In-Sample (${books.length} books)`;
    if (holdoutBtn) holdoutBtn.textContent = `Holdout \u2014 Out-of-Sample (${holdoutCount} books)`;

    toggleEl.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => renderForSubset(btn.dataset.subset));
    });
}

/**
 * Render charts for the selected data subset
 * @param {'all'|'holdout'} subsetKey
 */
function renderForSubset(subsetKey) {
    if (!allProcessedBooks) return;

    const books = subsetKey === 'holdout'
        ? allProcessedBooks.filter(b => b.source === 'Holdout 2026')
        : allProcessedBooks;

    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.subset === subsetKey);
    });

    renderResults(books);
}

/**
 * Calculate the weighted sorting score for a book based on the current weighting slider
 */
function getWeightedSortValue(book, strategy) {
    if (strategy === 'oracle') {
        const enjoy = book.trueEnjoy !== null ? book.trueEnjoy : 0;
        const useful = book.trueUseful !== null ? book.trueUseful : 0;
        return (currentWeight * enjoy) + ((1 - currentWeight) * useful);
    }
    if (strategy === 'ridge') {
        const enjoy = book.preds?.ridge_full_enjoy ?? 0;
        const useful = book.preds?.ridge_full_useful ?? 0;
        return (currentWeight * enjoy) + ((1 - currentWeight) * useful);
    }
    if (strategy === 'rf') {
        const enjoy = book.preds?.rf_enjoy ?? 0;
        const useful = book.preds?.rf_useful ?? 0;
        return (currentWeight * enjoy) + ((1 - currentWeight) * useful);
    }
    if (strategy === 'heuristic') {
        return book.preds?.external_sum ?? 0; 
    }
    return 0;
}

/**
 * Render results dashboard and charts
 * @param {Array<Object>} books - Processed book data
 */
function renderResults(books) {
    document.getElementById('loading').classList.add('hidden');
    
    const resultsEl = document.getElementById('results');
    resultsEl.classList.remove('hidden');
    if (!resultsEl.classList.contains('fade-in-up')) {
        resultsEl.classList.add('fade-in-up');
    }

    const enjoyBooks = books.filter(b => b.trueEnjoy !== null);
    const usefulBooks = books.filter(b => b.trueUseful !== null);

    const baseEnjoy = enjoyBooks.length > 0
        ? enjoyBooks.reduce((sum, b) => sum + b.trueEnjoy, 0) / enjoyBooks.length
        : 0;
    const baseUseful = usefulBooks.length > 0
        ? usefulBooks.reduce((sum, b) => sum + b.trueUseful, 0) / usefulBooks.length
        : 0;

    // Set Dashboard Stats
    document.getElementById('total-books').textContent = books.length;
    document.getElementById('base-enjoy').textContent = baseEnjoy.toFixed(2);
    document.getElementById('base-useful').textContent = baseUseful.toFixed(2);

    // --- Main drop-curve charts with MAE variance band ---
    const enjoyDatasets = buildDropCurveDatasets(enjoyBooks, 'trueEnjoy', [
        { key: 'oracle',    label: 'Oracle (Perfect Sort)', color: CHART_COLORS.perfect },
        { key: 'ridge',     label: 'Ridge Regression',      color: CHART_COLORS.ridge },
        { key: 'rf',        label: 'Random Forest',         color: CHART_COLORS.rf },
        { key: 'heuristic', label: 'Simple Heuristic',      color: CHART_COLORS.heuristic },
    ], LABELING_SD.enjoy);

    const usefulDatasets = buildDropCurveDatasets(usefulBooks, 'trueUseful', [
        { key: 'oracle',    label: 'Oracle (Perfect Sort)', color: CHART_COLORS.perfect },
        { key: 'ridge',     label: 'Ridge Regression',      color: CHART_COLORS.ridge },
        { key: 'rf',        label: 'Random Forest',         color: CHART_COLORS.rf },
        { key: 'heuristic', label: 'Simple Heuristic',      color: CHART_COLORS.heuristic },
    ], LABELING_SD.useful);

    // --- Bootstrap uncertainty charts ---
    const bootstrapEnjoyDatasets = buildBootstrapDatasets(enjoyBooks, 'trueEnjoy', [
        { key: 'ridge',     label: 'Ridge Regression', color: CHART_COLORS.ridge },
        { key: 'rf',        label: 'Random Forest',    color: CHART_COLORS.rf },
        { key: 'heuristic', label: 'Simple Heuristic', color: CHART_COLORS.heuristic },
    ]);

    const bootstrapUsefulDatasets = buildBootstrapDatasets(usefulBooks, 'trueUseful', [
        { key: 'ridge',     label: 'Ridge Regression', color: CHART_COLORS.ridge },
        { key: 'rf',        label: 'Random Forest',    color: CHART_COLORS.rf },
        { key: 'heuristic', label: 'Simple Heuristic', color: CHART_COLORS.heuristic },
    ]);

    // Compute standardized Y-axis bounds
    const findBounds = (datasetsList) => {
        let min = 0, max = 0;
        datasetsList.forEach(ds => {
            ds.forEach(dset => {
                if (dset.data && Array.isArray(dset.data)) {
                    dset.data.forEach(pt => {
                        const y = typeof pt === 'object' ? pt.y : pt;
                        if (y < min) min = y;
                        if (y > max) max = y;
                    });
                }
            });
        });
        return { min: min - 0.1, max: max + 0.1 };
    };

    const enjoyBounds = findBounds([enjoyDatasets, bootstrapEnjoyDatasets]);
    const usefulBounds = findBounds([usefulDatasets, bootstrapUsefulDatasets]);

    renderChart('enjoyChart', enjoyDatasets, enjoyBounds);
    renderChart('usefulChart', usefulDatasets, usefulBounds);
    renderChart('enjoyBootstrapChart', bootstrapEnjoyDatasets, enjoyBounds);
    renderChart('usefulBootstrapChart', bootstrapUsefulDatasets, usefulBounds);
    
    renderDroppedBooksTable(books);
}

/**
 * Build datasets for drop-curve charts with Monte Carlo rating-noise bands.
 * Grey bands show the 80% CI (10th-90th percentile) from simulating noisy ratings:
 * each iteration adds N(0, labelingSD) noise to true ratings, re-sorts (Oracle uses
 * noisy ratings, ML uses original predictions), and measures gain against noisy labels.
 */
function buildDropCurveDatasets(books, trueKey, strategies, labelingSD) {
    console.time('buildDropCurveDatasets (' + trueKey + ')');
    const datasets = [];
    for (const s of strategies) {
        const curve = calculateDropCurve(books, s.key, trueKey);
        datasets.push({ label: s.label, data: curve, ...applyStyle(s.color) });

        // Monte Carlo rating-noise 80% CI band
        const { p10, p90 } = monteCarloNoiseBands(books, s.key, trueKey, labelingSD, NOISE_ITERATIONS);
        datasets.push({
            label: s.label + ' (rating noise 80% CI)',
            data: p90,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            pointRadius: 0,
            fill: false,
            showLine: true,
            borderWidth: 0,
        });
        datasets.push({
            label: '_hide',
            data: p10,
            borderColor: 'transparent',
            backgroundColor: 'rgba(148, 163, 184, 0.12)',
            pointRadius: 0,
            fill: '-1',
            showLine: true,
            borderWidth: 0,
        });
    }
    console.timeEnd('buildDropCurveDatasets (' + trueKey + ')');
    return datasets;
}

/**
 * Build bootstrap uncertainty datasets.
 * Resamples books with replacement, re-sorts by predicted value, computes gain curves.
 * Returns line (bootstrap mean) + shaded band (10th-90th percentile).
 */
function buildBootstrapDatasets(books, trueKey, strategies) {
    if (books.length < 10) return [];
    console.time('buildBootstrapDatasets (' + trueKey + ')');
    const datasets = [];

    for (const s of strategies) {
        const { mean, p10, p90 } = bootstrapDropCurve(books, s.key, trueKey, BOOTSTRAP_ITERATIONS);

        // Bootstrap mean line
        datasets.push({
            label: s.label + ' (bootstrap mean)',
            data: mean,
            borderColor: s.color.bc,
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 5,
            borderDash: [],
        });

        // 80% CI band (p10 to p90)
        datasets.push({
            label: s.label + ' (80% CI upper)',
            data: p90,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            pointRadius: 0,
            fill: false,
            showLine: true,
            borderWidth: 0,
        });
        datasets.push({
            label: '_hide',
            data: p10,
            borderColor: 'transparent',
            backgroundColor: s.color.bg.replace('0.2', '0.15'),
            pointRadius: 0,
            fill: '-1',
            showLine: true,
            borderWidth: 0,
        });
    }
    console.timeEnd('buildBootstrapDatasets (' + trueKey + ')');
    return datasets;
}

/**
 * Run bootstrap resampling on the drop curve.
 * For each iteration: resample books with replacement, re-sort by prediction, compute gains.
 * @returns {{ mean: Array, p10: Array, p90: Array }}
 */
function bootstrapDropCurve(books, strategy, trueKey, nIterations) {
    const N = books.length;
    const dropPcts = [];
    for (let d = DROP_PERCENTAGES.MIN; d <= DROP_PERCENTAGES.MAX; d += DROP_PERCENTAGES.STEP) {
        dropPcts.push(d);
    }

    // Collect gain samples: dropPct -> array of gains
    const samples = {};
    for (const d of dropPcts) samples[d] = [];

    for (let iter = 0; iter < nIterations; iter++) {
        // Resample with replacement
        const resampled = [];
        for (let i = 0; i < N; i++) {
            resampled.push(books[Math.floor(Math.random() * N)]);
        }

        // Sort ascending by predicted value
        resampled.sort((a, b) => {
            return getWeightedSortValue(a, strategy) - getWeightedSortValue(b, strategy);
        });

        const baseline = resampled.reduce((sum, b) => sum + b[trueKey], 0) / N;

        for (const d of dropPcts) {
            const dropCount = Math.floor(N * (d / 100));
            const kept = resampled.slice(dropCount);
            if (kept.length === 0) {
                samples[d].push(0);
                continue;
            }
            const keptAvg = kept.reduce((sum, b) => sum + b[trueKey], 0) / kept.length;
            samples[d].push(keptAvg - baseline);
        }
    }

    // Compute statistics
    const mean = [];
    const p10 = [];
    const p90 = [];
    for (const d of dropPcts) {
        const vals = samples[d].sort((a, b) => a - b);
        const n = vals.length;
        const avg = vals.reduce((s, v) => s + v, 0) / n;
        mean.push({ x: d, y: avg });
        p10.push({ x: d, y: vals[Math.floor(n * 0.10)] });
        p90.push({ x: d, y: vals[Math.floor(n * 0.90)] });
    }
    return { mean, p10, p90 };
}

/**
 * Calculate utility curve showing gain when dropping bottom X% of books
 * @param {Array<Object>} books - Array of book objects
 * @param {string} predictKey - Key for predicted value to sort by
 * @param {string} trueKey - Key for true outcome value
 * @returns {Array<{x: number, y: number}>} Array of {x: dropPct, y: gain} points
 */
function calculateDropCurve(books, strategy, trueKey) {
    if (!books || books.length === 0) {
        return [];
    }

    // Sort books ascending by predicted value (worst first)
    const sorted = [...books].sort((a, b) => {
        return getWeightedSortValue(a, strategy) - getWeightedSortValue(b, strategy);
    });

    const baselineTruth = books.reduce((sum, b) => sum + b[trueKey], 0) / books.length;

    const curve = [];
    const N = sorted.length;

    for (let dropPct = DROP_PERCENTAGES.MIN; dropPct <= DROP_PERCENTAGES.MAX; dropPct += DROP_PERCENTAGES.STEP) {
        const dropCount = Math.floor(N * (dropPct / 100));
        const keptBooks = sorted.slice(dropCount);

        if (keptBooks.length === 0) break;

        const keptAvg = keptBooks.reduce((sum, b) => sum + b[trueKey], 0) / keptBooks.length;
        curve.push({
            x: dropPct,
            y: keptAvg - baselineTruth
        });
    }

    return curve;
}

/**
 * Apply consistent styling to a chart dataset
 * @param {Object} styleObj - Style configuration with bc, bg, dash properties
 * @returns {Object} Chart.js dataset style object
 */
function applyStyle(styleObj) {
    return {
        borderColor: styleObj.bc,
        backgroundColor: styleObj.bg,
        borderDash: styleObj.dash,
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6
    };
}

/**
 * Render a Chart.js line chart
 * @param {string} canvasId - Canvas element ID
 * @param {Array<Object>} datasets - Chart datasets
 */
function renderChart(canvasId, datasets, bounds) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element ${canvasId} not found`);
        return;
    }

    if (datasets.length === 0) return;

    const ctx = canvas.getContext('2d');

    // Update existing chart instance dynamically for smooth animations
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].data.datasets = datasets;
        chartInstances[canvasId].options.scales.y.min = bounds && typeof bounds.min === 'number' ? Number(bounds.min.toFixed(2)) : undefined;
        chartInstances[canvasId].options.scales.y.max = bounds && typeof bounds.max === 'number' ? Number(bounds.max.toFixed(2)) : undefined;
        chartInstances[canvasId].update('active');
        return;
    }

    const instance = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            color: '#94a3b8',
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: '% of Books Dropped', color: '#94a3b8', font: { family: 'Inter' } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' },
                    min: 0,
                    max: 95
                },
                y: {
                    min: bounds && typeof bounds.min === 'number' ? Number(bounds.min.toFixed(2)) : undefined,
                    max: bounds && typeof bounds.max === 'number' ? Number(bounds.max.toFixed(2)) : undefined,
                    title: { display: true, text: 'Rating Gain vs Baseline', color: '#94a3b8', font: { family: 'Inter' } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#f8fafc',
                        font: { family: 'Inter', size: 12 },
                        usePointStyle: true,
                        filter: function(legendItem) {
                            // Hide fill-helper datasets and noise bands from legend
                            return legendItem.text !== '_hide' &&
                                   !legendItem.text.includes('rating noise') &&
                                   !legendItem.text.includes('CI upper');
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 10,
                    filter: function(tooltipItem) {
                        // Hide fill-helper datasets from tooltip
                        const label = tooltipItem.dataset.label || '';
                        return label !== '_hide' && !label.includes('rating noise') && !label.includes('CI upper');
                    },
                    callbacks: {
                        label: function(context) {
                            const sign = context.parsed.y >= 0 ? '+' : '';
                            return `${context.dataset.label}: ${sign}${context.parsed.y.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });

    chartInstances[canvasId] = instance;
}

/**
 * Helper to generate a styled pill element for 1-5 ratings
 * @param {number|null} ratingValue 
 * @returns {HTMLElement} span element with pill styles
 */
function createRatingPill(ratingValue) {
    const pill = document.createElement('span');
    pill.className = 'rating-pill';
    
    if (ratingValue === null || isNaN(ratingValue)) {
        pill.textContent = '-';
        pill.style.background = 'transparent';
        pill.style.border = '1px solid var(--border-glass)';
        pill.style.color = 'var(--text-muted)';
        return pill;
    }
    
    pill.textContent = ratingValue.toFixed(1);
    
    if (ratingValue >= 4.0) pill.classList.add('rating-high');
    else if (ratingValue >= 3.0) pill.classList.add('rating-med');
    else pill.classList.add('rating-low');
    
    return pill;
}

/**
 * Render the table of books that would be dropped at the selected threshold
 */
/**
 * Helper to retrieve specific model predictions for the table
 */
function getPredValues(book, strategy) {
    let enjoy = 0, useful = 0;
    if (strategy === 'oracle') {
        enjoy = book.trueEnjoy !== null ? book.trueEnjoy : 0;
        useful = book.trueUseful !== null ? book.trueUseful : 0;
    } else if (strategy === 'ridge') {
        enjoy = book.preds?.ridge_full_enjoy ?? 0;
        useful = book.preds?.ridge_full_useful ?? 0;
    } else if (strategy === 'rf') {
        enjoy = book.preds?.rf_enjoy ?? 0;
        useful = book.preds?.rf_useful ?? 0;
    } else if (strategy === 'heuristic') {
        enjoy = book.preds?.external_sum ?? 0;
        useful = book.preds?.external_sum ?? 0;
    }
    const wtPred = (currentWeight * enjoy) + ((1 - currentWeight) * useful);
    return { enjoy, useful, wtPred };
}

/**
 * Render the full table of books, highlighting the dropped ones
 */
function renderDroppedBooksTable(books) {
    const tbody = document.getElementById('dropped-books-tbody');
    if (!tbody || !books || books.length === 0) return;

    // 1. Identify dropped boundary according to currentTableModel
    const booksWithScores = books.map(b => {
        const preds = getPredValues(b, currentTableModel);
        return { 
            book: b, 
            ...preds,
            title: b.title || '',
            category: b.category || '',
            trueEnjoy: b.trueEnjoy !== null ? b.trueEnjoy : -99,
            trueUseful: b.trueUseful !== null ? b.trueUseful : -99
        };
    });

    // Sort to find threshold
    const sortedForCutoff = [...booksWithScores].sort((a, b) => a.wtPred - b.wtPred);
    const dropCount = Math.floor(sortedForCutoff.length * (currentDropPct / 100));
    const droppedSet = new Set(sortedForCutoff.slice(0, dropCount).map(i => i.book));

    // 2. Sort according to User's column preference
    const displayList = [...booksWithScores].sort((a, b) => {
        let valA = a[currentSortCol];
        let valB = b[currentSortCol];
        
        // String sort
        if (typeof valA === 'string' && typeof valB === 'string') {
            return currentSortDir === 'asc' 
                ? valA.localeCompare(valB)
                : valB.localeCompare(valA);
        }
        
        // Numeric sort
        return currentSortDir === 'asc' ? valA - valB : valB - valA;
    });

    // 3. Update table header UI classes
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        th.querySelector('.sort-icon').textContent = '↕';
        if (th.dataset.sort === currentSortCol) {
            th.classList.add(`sort-${currentSortDir}`);
            th.querySelector('.sort-icon').textContent = currentSortDir === 'asc' ? '▲' : '▼';
        }
    });

    // 4. Render
    tbody.innerHTML = '';
    
    // Use fragment for performance
    const frag = document.createDocumentFragment();
    
    displayList.forEach(item => {
        const b = item.book;
        const isDropped = droppedSet.has(b);
        
        const tr = document.createElement('tr');
        if (isDropped) tr.classList.add('row-dropped');
        
        // Title & Category
        const titleTd = document.createElement('td');
        titleTd.className = 'col-title';
        titleTd.textContent = item.title;
        tr.appendChild(titleTd);
        
        const catTd = document.createElement('td');
        catTd.textContent = item.category;
        tr.appendChild(catTd);
        
        // True Enjoyment & Usefulness
        const trueEnjoyTd = document.createElement('td');
        trueEnjoyTd.className = 'num-col';
        trueEnjoyTd.appendChild(createRatingPill(b.trueEnjoy));
        tr.appendChild(trueEnjoyTd);
        
        const trueUsefulTd = document.createElement('td');
        trueUsefulTd.className = 'num-col';
        trueUsefulTd.appendChild(createRatingPill(b.trueUseful));
        tr.appendChild(trueUsefulTd);
        
        // Model Predictions
        const predEnjoyTd = document.createElement('td');
        predEnjoyTd.className = 'num-col';
        predEnjoyTd.appendChild(createRatingPill(item.enjoy));
        tr.appendChild(predEnjoyTd);
        
        const predUsefulTd = document.createElement('td');
        predUsefulTd.className = 'num-col';
        predUsefulTd.appendChild(createRatingPill(item.useful));
        tr.appendChild(predUsefulTd);
        
        // Weighted Prediction
        const predTd = document.createElement('td');
        predTd.className = 'num-col';
        
        // Render inline progress bar for final score
        const span = document.createElement('span');
        span.style.cssText = 'min-width: 45px; display: inline-block; font-weight: 600; font-family: Outfit, monospace;';
        span.textContent = item.wtPred.toFixed(3);
        
        const pct = Math.max(0, Math.min(100, ((item.wtPred - 1) / 4) * 100));
        
        const barContainer = document.createElement('div');
        barContainer.className = 'pred-bar-container';
        barContainer.innerHTML = `
            ${span.outerHTML}
            <div class="pred-bar" title="${item.wtPred.toFixed(3)}/5.0">
                <div class="pred-bar-fill" style="width: ${pct}%;"></div>
            </div>
        `;
        predTd.appendChild(barContainer);
        tr.appendChild(predTd);
        
        frag.appendChild(tr);
    });
    
    tbody.appendChild(frag);
}
