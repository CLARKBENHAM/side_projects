const fs = require('fs');

global.document = { addEventListener: () => {}, getElementById: () => ({ classList: { add: ()=>{}, remove: ()=>{} }, style: {}, textContent: '', getContext: () => ({}) }) };
global.window = {}; global.fetch = async () => ({ ok: true, json: async () => ({}) }); global.Papa = { parse: () => {} }; global.Chart = class { constructor() {} destroy() {} };

const predictCode = fs.readFileSync(__dirname + '/predict.js', 'utf8');
eval(predictCode);

let appCode = fs.readFileSync(__dirname + '/app.js', 'utf8');
appCode = appCode.replace('let modelsData = null;', 'var ObjectConfig = fs.readFileSync(__dirname + "/models.json", "utf8"); var modelsData = JSON.parse(ObjectConfig);');
appCode = appCode.replace(/function renderResults\(books\) \{[\s\S]*?function calculateDropCurve/m, 'function renderResults(books){ global.outBooks = books; }\nfunction calculateDropCurve');
eval(appCode);

const csvContent = fs.readFileSync(__dirname + '/gwern_books_rich.csv', 'utf8');
const lines = csvContent.split('\n').filter(l => l.trim().length > 0);
const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));

const rows = [];
const re2 = /,(?=(?:(?:[^"]*"){2})*[^"]*$)/;

for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(re2).map(v => v.trim().replace(/^"|"$/g, ''));
    if (values.length !== headers.length) continue;
    let row = {};
    for (let j=0; j<headers.length; j++) {
        row[headers[j]] = values[j];
    }
    rows.push(row);
}

processData(rows);
const books = global.outBooks;

const outHeaders = ['title', 'category', 'read_year', 'link', 'trueEnjoy', 'grRating', 'ridge_full_enjoy', 'rf_enjoy', 'gbm_enjoy', 'heuristic_sum'];
let outCsv = outHeaders.join(',') + '\n';

for (const b of books) {
    // Find the original row for link and year
    const orig = rows.find(r => r.title === b.title);
    const link = orig ? orig.link : '';
    const read_year = orig ? orig.read_year : '';
    
    // Some titles might have commas, quote them
    const t = '"' + b.title.replace(/"/g, '""') + '"';
    outCsv += `${t},${b.category},${read_year},${link},${b.trueEnjoy},${b.pageData.grRating},${b.preds.ridge_full_enjoy},${b.preds.rf_enjoy},${b.preds.gbm_enjoy},${b.preds.external_sum}\n`;
}

fs.writeFileSync(__dirname + '/gwern_scored.csv', outCsv);
console.log("Exported scored books to gwern_scored.csv");
