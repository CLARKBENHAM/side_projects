// Content script: auto-injects a floating prediction widget on Amazon/Goodreads book pages.
// Communicates with background.js for cross-site search and predictions.

(function () {
  // Avoid double-injection
  if (document.getElementById("book-predictor-widget")) return;

  const CATEGORIES = [
    "Business", "CS", "General", "Literature",
    "Fiction", "Math", "ML", "Histories",
  ];
  const CATEGORY_MAP = {
    "Business": "Business",
    "CS": "Computer Science",
    "General": "General Reading",
    "Literature": "Literature",
    "Fiction": "Fiction",
    "Math": "Math",
    "ML": "Machine Learning",
    "Histories": "Histories",
  };

  let pageData = {};
  let selectedCategory = null;
  let modelsCache = null;

  function cleanSearchTitle(title) {
    return title
      .replace(/,?\s*\b(\d+)(st|nd|rd|th)\s+edition\b/gi, "")
      .replace(/,?\s*\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+edition\b/gi, "")
      .replace(/,?\s*\bedition\b/gi, "")
      .replace(/\s*\([^)]*(?:press|publisher|publishing|edition)[^)]*\)/gi, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  async function getModels() {
    if (modelsCache) return modelsCache;
    const url = chrome.runtime.getURL("models.json");
    const resp = await fetch(url);
    modelsCache = await resp.json();
    return modelsCache;
  }

  // Start loading models immediately
  getModels();

  // --- Detect site and extract data ---

  function isBookPage() {
    const url = location.href;
    if (url.includes("goodreads.com/book/show")) return "goodreads";
    if (url.includes("amazon.com") && (url.includes("/dp/") || url.includes("/gp/product/"))) return "amazon";
    if (url.includes("amazon.co.uk") && (url.includes("/dp/") || url.includes("/gp/product/"))) return "amazon";
    // Amazon product pages without /dp/ but with productTitle
    if ((url.includes("amazon.com") || url.includes("amazon.co.uk")) && document.getElementById("productTitle")) return "amazon";
    return null;
  }

  function extractGoodreads() {
    const data = { source: "goodreads", title: null, author: null, grRating: null, grCount: null, pageCount: null, pubYear: null };
    const titleEl = document.querySelector('h1[data-testid="bookTitle"], h1.Text__title1, #bookTitle, h1');
    if (titleEl) data.title = titleEl.textContent.trim();
    const authorEl = document.querySelector('span.ContributorLink__name, a.authorName span[itemprop="name"]');
    if (authorEl) data.author = authorEl.textContent.trim();
    const ratingEl = document.querySelector('div.RatingStatistics__rating, span[itemprop="ratingValue"]');
    if (ratingEl) { const v = parseFloat(ratingEl.textContent.trim()); if (!isNaN(v)) data.grRating = v; }
    const countEl = document.querySelector('span[data-testid="ratingsCount"]');
    if (countEl) { const v = parseInt(countEl.textContent.replace(/[^0-9]/g, ""), 10); if (!isNaN(v)) data.grCount = v; }
    if (!data.grCount) { const m = document.querySelector('meta[itemprop="ratingCount"]'); if (m) { const v = parseInt(m.getAttribute("content"), 10); if (!isNaN(v)) data.grCount = v; } }
    const pageEl = document.querySelector('p[data-testid="pagesFormat"]');
    if (pageEl) { const m = pageEl.textContent.match(/(\d+)\s*pages/i); if (m) data.pageCount = parseInt(m[1], 10); }
    if (!data.pageCount) { const s = document.querySelector('span[itemprop="numberOfPages"]'); if (s) { const m = s.textContent.match(/(\d+)/); if (m) data.pageCount = parseInt(m[1], 10); } }
    const pubEl = document.querySelector('p[data-testid="publicationInfo"]');
    if (pubEl) { const m = pubEl.textContent.match(/(\d{4})/); if (m) data.pubYear = parseInt(m[1], 10); }
    return data;
  }

  function extractAmazon() {
    const data = { source: "amazon", title: null, author: null, amzRating: null, amzCount: null, pageCount: null, pubYear: null };
    const titleEl = document.querySelector("#productTitle, #ebooksProductTitle");
    if (titleEl) data.title = titleEl.textContent.trim();
    const authorEl = document.querySelector(".author a, .contributorNameID, #bylineInfo a.a-link-normal");
    if (authorEl) data.author = authorEl.textContent.trim();
    for (const sel of ['#acrPopover span.a-size-base.a-color-base', '#acrPopover .a-icon-alt', 'i.a-icon-star span.a-icon-alt']) {
      const el = document.querySelector(sel);
      if (el) { const m = el.textContent.match(/([\d.]+)/); if (m) { data.amzRating = parseFloat(m[1]); break; } }
    }
    for (const sel of ['#acrCustomerReviewCount', '#acrCustomerReviewText']) {
      const el = document.querySelector(sel);
      if (el) { const v = parseInt(el.textContent.replace(/[^0-9]/g, ""), 10); if (!isNaN(v) && v > 0) { data.amzCount = v; break; } }
    }
    const pm = document.body.innerText.match(/(\d+)\s*pages/i);
    if (pm) data.pageCount = parseInt(pm[1], 10);
    for (const item of document.querySelectorAll('#detailBullets_feature_div li, .detail-bullet-list .a-list-item, #productDetailsTable td')) {
      const t = item.textContent;
      if (t.toLowerCase().includes("publisher") || t.toLowerCase().includes("publication")) {
        const dm = t.match(/(\d{4})/);
        if (dm) { data.pubYear = parseInt(dm[1], 10); break; }
      }
    }
    return data;
  }

  // --- Build the floating widget ---

  function createWidget() {
    const host = document.createElement("div");
    host.id = "book-predictor-widget";
    const shadow = host.attachShadow({ mode: "closed" });

    shadow.innerHTML = `
    <style>
      :host { all: initial; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
      .widget {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999999;
        background: #fff;
        border: 1px solid #ccc;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        width: 320px;
        font-size: 13px;
        color: #1a1a1a;
        overflow: hidden;
        transition: height 0.2s;
      }
      .widget.collapsed .body { display: none; }
      .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        background: #1a73e8;
        color: #fff;
        cursor: move;
        font-size: 12px;
        font-weight: 600;
        user-select: none;
      }
      .header-bar .toggle-btn {
        background: none;
        border: none;
        color: #fff;
        cursor: pointer;
        font-size: 16px;
        line-height: 1;
        padding: 0 4px;
      }
      .body { padding: 8px 10px; }
      .title-line {
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .meta-line { font-size: 11px; color: #666; margin-bottom: 6px; }
      .stats-row {
        display: flex;
        gap: 6px;
        margin-bottom: 6px;
      }
      .stat { text-align: center; flex: 1; }
      .stat .sl { font-size: 9px; color: #999; text-transform: uppercase; }
      .stat .sv { font-size: 13px; font-weight: 700; }
      .stat .sv.missing { color: #c00; font-style: italic; font-size: 10px; }
      .cat-row {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        margin-bottom: 6px;
      }
      .cat-btn {
        padding: 3px 6px;
        border: 1px solid #ccc;
        border-radius: 3px;
        background: #f5f5f5;
        cursor: pointer;
        font-size: 10px;
      }
      .cat-btn:hover { background: #e0e0e0; }
      .cat-btn.sel { background: #1a73e8; color: #fff; border-color: #1a73e8; }
      .results { margin-top: 4px; }
      .model-row {
        display: grid;
        grid-template-columns: 36px 1fr 1fr;
        gap: 2px 8px;
        align-items: start;
        padding: 3px 0;
        border-top: 1px solid #f0f0f0;
      }
      .model-row:first-child { border-top: none; }
      .mn { font-weight: 600; font-size: 11px; color: #555; padding-top: 2px; }
      .pred-cell { text-align: right; }
      .pred-cell .pv { font-weight: 700; font-size: 15px; font-variant-numeric: tabular-nums; }
      .pred-cell .pv.excellent { color: #1a7f37; }
      .pred-cell .pv.good { color: #2d8f40; }
      .pred-cell .pv.above_average { color: #b35900; }
      .pred-cell .pv.average { color: #666; }
      .pred-cell .pv.below_average { color: #d73a49; }
      .pred-cell .pv.poor { color: #c00; }
      .pred-cell .pv.very_poor { color: #8b0000; }
      .pred-cell .pi { font-size: 9px; color: #999; font-variant-numeric: tabular-nums; line-height: 1.4; }
      .pred-cell .pi.b85 { color: #bbb; }
      .pred-hdr { font-weight: 700; font-size: 10px; color: #888; text-align: right; padding-bottom: 2px; }
      .status-line { font-size: 10px; color: #888; margin-top: 4px; }
      .status-line.err { color: #c00; }
      .status-line.srch { color: #1a73e8; }
    </style>
    <div class="widget" id="w">
      <div class="header-bar" id="hdr">
        <span>Book Predictor</span>
        <button class="toggle-btn" id="toggle-btn">−</button>
      </div>
      <div class="body" id="body">
        <div class="title-line" id="w-title">—</div>
        <div class="meta-line" id="w-author">—</div>
        <div class="stats-row">
          <div class="stat"><div class="sl">GR</div><div class="sv" id="w-gr">—</div></div>
          <div class="stat"><div class="sl">GR #</div><div class="sv" id="w-grc">—</div></div>
          <div class="stat"><div class="sl">AMZ</div><div class="sv" id="w-amz">—</div></div>
          <div class="stat"><div class="sl">AMZ #</div><div class="sv" id="w-amzc">—</div></div>
          <div class="stat"><div class="sl">Pages</div><div class="sv" id="w-pg">—</div></div>
        </div>
        <div class="cat-row" id="w-cats"></div>
        <div class="results" id="w-results" style="display:none;">
          <div class="model-row">
            <span></span><span class="pred-hdr">Enjoyment</span><span class="pred-hdr">Usefulness</span>
          </div>
          <div class="model-row">
            <span class="mn">Group Ridge</span>
            <div class="pred-cell" id="wr-re"></div>
            <div class="pred-cell" id="wr-ru"></div>
          </div>
          <div class="model-row">
            <span class="mn">Full Ridge</span>
            <div class="pred-cell" id="wr-fre"></div>
            <div class="pred-cell" id="wr-fru"></div>
          </div>
          <div class="model-row">
            <span class="mn">RF</span>
            <div class="pred-cell" id="wr-rfe"></div>
            <div class="pred-cell" id="wr-rfu"></div>
          </div>
          <div class="model-row">
            <span class="mn">GBM</span>
            <div class="pred-cell" id="wr-ge"></div>
            <div class="pred-cell" id="wr-gu"></div>
          </div>
        </div>
        <div class="status-line" id="w-status"></div>
      </div>
    </div>
    `;

    document.body.appendChild(host);
    return shadow;
  }

  function percentileClass(percentile) {
    if (percentile == null) return "";
    if (percentile >= 90) return "excellent";
    if (percentile >= 75) return "good";
    if (percentile >= 60) return "above_average";
    if (percentile >= 40) return "average";
    if (percentile >= 25) return "below_average";
    if (percentile >= 10) return "poor";
    return "very_poor";
  }

  function formatPercentile(percentile) {
    if (percentile == null) return "—";
    return percentile.toFixed(0) + "%";
  }

  function renderPredCell(cell, val, percentile, iv) {
    const f = v => `${Math.round(v)}%`;
    const pctClass = percentileClass(percentile);
    const pctText = formatPercentile(percentile);
    
    let html = `<div class="pv ${pctClass}">${pctText}</div>`;
    if (iv) {
      html += `<div class="pi">50%: ${f(iv.asym50_lo)}–${f(iv.asym50_hi)}</div>`;
      html += `<div class="pi b85">85%: ${f(iv.asym85_lo)}–${f(iv.asym85_hi)}</div>`;
    }
    cell.innerHTML = html;
  }

  // --- Main logic ---

  const site = isBookPage();
  if (!site) return;

  const shadow = createWidget();
  const q = sel => shadow.querySelector(sel);

  // Toggle collapse
  q("#toggle-btn").addEventListener("click", () => {
    const w = q("#w");
    const btn = q("#toggle-btn");
    w.classList.toggle("collapsed");
    btn.textContent = w.classList.contains("collapsed") ? "+" : "−";
  });

  // Make draggable
  let isDragging = false, dragX = 0, dragY = 0;
  const widget = q("#w");
  q("#hdr").addEventListener("mousedown", e => {
    isDragging = true;
    dragX = e.clientX - widget.getBoundingClientRect().left;
    dragY = e.clientY - widget.getBoundingClientRect().top;
    e.preventDefault();
  });
  document.addEventListener("mousemove", e => {
    if (!isDragging) return;
    widget.style.left = (e.clientX - dragX) + "px";
    widget.style.top = (e.clientY - dragY) + "px";
    widget.style.right = "auto";
  });
  document.addEventListener("mouseup", () => { isDragging = false; });

  // Extract page data
  if (site === "goodreads") {
    pageData = extractGoodreads();
  } else {
    pageData = extractAmazon();
  }

  // Populate widget
  q("#w-title").textContent = pageData.title || "Unknown";
  q("#w-author").textContent = pageData.author || "Unknown author";
  q("#w-gr").textContent = pageData.grRating ?? "—";
  q("#w-grc").textContent = pageData.grCount?.toLocaleString() ?? "—";
  q("#w-amz").textContent = pageData.amzRating ?? "—";
  q("#w-amzc").textContent = pageData.amzCount?.toLocaleString() ?? "—";
  q("#w-pg").textContent = pageData.pageCount ?? "—";

  // Category buttons
  const catRow = q("#w-cats");
  for (const cat of CATEGORIES) {
    const btn = document.createElement("span");
    btn.className = "cat-btn";
    btn.textContent = cat;
    btn.addEventListener("click", () => {
      catRow.querySelectorAll(".cat-btn").forEach(b => b.classList.remove("sel"));
      btn.classList.add("sel");
      selectedCategory = CATEGORY_MAP[cat];
      predict();
    });
    catRow.appendChild(btn);
  }

  // Cross-site search
  const statusEl = q("#w-status");
  if (site === "goodreads" && pageData.title) {
    statusEl.textContent = "Searching Amazon...";
    statusEl.className = "status-line srch";
    chrome.runtime.sendMessage({ type: "SEARCH_AMAZON", title: cleanSearchTitle(pageData.title), author: pageData.author || "" }, resp => {
      if (!resp || !resp.success) { statusEl.textContent = "AMZ search failed"; statusEl.className = "status-line err"; return; }
      const parser = new DOMParser();
      const doc = parser.parseFromString(resp.html, "text/html");
      for (const item of doc.querySelectorAll('[data-component-type="s-search-result"]')) {
        const rEl = item.querySelector(".a-icon-alt");
        const cEl = item.querySelector("a .a-size-base, span.a-size-base.s-underline-text");
        if (rEl) {
          const m = rEl.textContent.match(/([\d.]+)/);
          if (m) { pageData.amzRating = parseFloat(m[1]); q("#w-amz").textContent = pageData.amzRating; }
        }
        if (cEl) {
          const v = parseInt(cEl.textContent.replace(/[^0-9]/g, ""), 10);
          if (!isNaN(v) && v > 0) { pageData.amzCount = v; q("#w-amzc").textContent = v.toLocaleString(); }
        }
        if (pageData.amzRating) break;
      }
      statusEl.textContent = pageData.amzRating ? `AMZ: ${pageData.amzRating} ★` : "No AMZ match";
      statusEl.className = "status-line";
      if (selectedCategory) predict();
    });
  } else if (site === "amazon" && pageData.title) {
    statusEl.textContent = "Searching Goodreads...";
    statusEl.className = "status-line srch";
    chrome.runtime.sendMessage({ type: "SEARCH_GOODREADS", title: cleanSearchTitle(pageData.title), author: pageData.author || "" }, resp => {
      if (!resp || !resp.success) { statusEl.textContent = "GR search failed"; statusEl.className = "status-line err"; return; }
      const parser = new DOMParser();
      const doc = parser.parseFromString(resp.html, "text/html");
      for (const row of doc.querySelectorAll("tr[itemscope]")) {
        const rs = row.querySelector("span.minirating");
        if (rs) {
          const m = rs.textContent.match(/([\d.]+)\s*avg.*?([\d,]+)\s*rating/);
          if (m) {
            const rating = parseFloat(m[1]);
            const count = parseInt(m[2].replace(/,/g, ""), 10);
            if (count >= 100) {
              pageData.grRating = rating;
              pageData.grCount = count;
              q("#w-gr").textContent = rating;
              q("#w-grc").textContent = count.toLocaleString();
              break;
            }
          }
        }
      }
      statusEl.textContent = pageData.grRating ? `GR: ${pageData.grRating} ★` : "No GR match";
      statusEl.className = "status-line";
      if (selectedCategory) predict();
    });
  }

  async function predict() {
    if (!selectedCategory) return;
    try {
      const models = await getModels();
      const r = runAllPredictions(models, pageData, selectedCategory);
      q("#w-results").style.display = "block";

      // Use percentiles for display
      renderPredCell(q("#wr-re"), r.ridge_enjoy, r.percentiles?.ridge_enjoy, r.intervalPercentiles?.ridge_enjoy);
      renderPredCell(q("#wr-ru"), r.ridge_useful, r.percentiles?.ridge_useful, r.intervalPercentiles?.ridge_useful);
      renderPredCell(q("#wr-fre"), r.ridge_full_enjoy, r.percentiles?.ridge_full_enjoy, r.intervalPercentiles?.ridge_full_enjoy);
      renderPredCell(q("#wr-fru"), r.ridge_full_useful, r.percentiles?.ridge_full_useful, r.intervalPercentiles?.ridge_full_useful);
      renderPredCell(q("#wr-rfe"), r.rf_enjoy, r.percentiles?.rf_enjoy, r.intervalPercentiles?.rf_enjoy);
      renderPredCell(q("#wr-rfu"), r.rf_useful, r.percentiles?.rf_useful, r.intervalPercentiles?.rf_useful);
      renderPredCell(q("#wr-ge"), r.gbm_enjoy, r.percentiles?.gbm_enjoy, r.intervalPercentiles?.gbm_enjoy);
      renderPredCell(q("#wr-gu"), r.gbm_useful, r.percentiles?.gbm_useful, r.intervalPercentiles?.gbm_useful);

      statusEl.textContent = "";
    } catch (err) {
      statusEl.textContent = "Prediction error: " + err.message;
      statusEl.className = "status-line err";
    }
  }

})();
