const CATEGORIES = [
  "Business", "Computer Science", "General Reading", "Literature",
  "Fiction", "Math", "Machine Learning", "Histories",
];

let pageData = {};
let selectedCategory = null;

function $(id) { return document.getElementById(id); }

function scoreClass(val, type) {
  if (type === "enjoy") {
    if (val >= 3.9) return "good";
    if (val >= 3.3) return "mid";
    return "low";
  }
  if (val >= 2.5) return "good";
  if (val >= 1.8) return "mid";
  return "low";
}

function formatScore(val) {
  return val != null ? val.toFixed(2) : "—";
}

function setField(id, val, missing) {
  const el = $(id);
  if (val != null && val !== undefined) {
    el.textContent = String(val);
    el.classList.remove("missing");
  } else {
    el.textContent = missing || "not found";
    el.classList.add("missing");
  }
}

function initCategoryButtons() {
  const grid = $("category-grid");
  for (const cat of CATEGORIES) {
    const btn = document.createElement("div");
    btn.className = "cat-btn";
    btn.textContent = cat;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".cat-btn").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedCategory = cat;
      $("predict-btn").disabled = false;
      $("predict-btn").textContent = `Predict for "${cat}"`;
    });
    grid.appendChild(btn);
  }
}

// Extraction functions that run inside the page via executeScript
function extractGoodreadsData() {
  const data = { source: "goodreads", title: null, author: null, grRating: null, grCount: null, pageCount: null, pubYear: null };

  const titleEl = document.querySelector('h1[data-testid="bookTitle"], h1.Text__title1');
  if (titleEl) data.title = titleEl.textContent.trim();
  if (!data.title) {
    const h1 = document.querySelector("#bookTitle, h1");
    if (h1) data.title = h1.textContent.trim();
  }

  const authorEl = document.querySelector('span.ContributorLink__name, a.authorName span[itemprop="name"]');
  if (authorEl) data.author = authorEl.textContent.trim();

  const ratingEl = document.querySelector('div.RatingStatistics__rating, span[itemprop="ratingValue"]');
  if (ratingEl) {
    const val = parseFloat(ratingEl.textContent.trim());
    if (!isNaN(val)) data.grRating = val;
  }

  const countEl = document.querySelector('span[data-testid="ratingsCount"]');
  if (countEl) {
    const text = countEl.textContent.replace(/[^0-9]/g, "");
    const val = parseInt(text, 10);
    if (!isNaN(val)) data.grCount = val;
  }
  if (!data.grCount) {
    const metaCount = document.querySelector('meta[itemprop="ratingCount"]');
    if (metaCount) {
      const val = parseInt(metaCount.getAttribute("content"), 10);
      if (!isNaN(val)) data.grCount = val;
    }
  }

  const pageEl = document.querySelector('p[data-testid="pagesFormat"]');
  if (pageEl) {
    const m = pageEl.textContent.match(/(\d+)\s*pages/i);
    if (m) data.pageCount = parseInt(m[1], 10);
  }
  if (!data.pageCount) {
    const pageSpan = document.querySelector('span[itemprop="numberOfPages"]');
    if (pageSpan) {
      const m = pageSpan.textContent.match(/(\d+)/);
      if (m) data.pageCount = parseInt(m[1], 10);
    }
  }

  const pubEl = document.querySelector('p[data-testid="publicationInfo"]');
  if (pubEl) {
    const m = pubEl.textContent.match(/(\d{4})/);
    if (m) data.pubYear = parseInt(m[1], 10);
  }

  return data;
}

function extractAmazonData() {
  const data = { source: "amazon", title: null, author: null, amzRating: null, amzCount: null, pageCount: null, pubYear: null };

  const titleEl = document.querySelector("#productTitle, #ebooksProductTitle");
  if (titleEl) data.title = titleEl.textContent.trim();

  const authorEl = document.querySelector(".author a, .contributorNameID, #bylineInfo a.a-link-normal");
  if (authorEl) data.author = authorEl.textContent.trim();

  // Rating - try multiple selectors
  const ratingSelectors = [
    '#acrPopover span.a-size-base.a-color-base',
    '#acrPopover .a-icon-alt',
    'span[data-action="acrStars-popover"] span.a-size-base',
    '#averageCustomerReviews span.a-icon-alt',
    'i.a-icon-star span.a-icon-alt',
  ];
  for (const sel of ratingSelectors) {
    const el = document.querySelector(sel);
    if (el) {
      const m = el.textContent.match(/([\d.]+)/);
      if (m) { data.amzRating = parseFloat(m[1]); break; }
    }
  }

  // Rating count
  const countSelectors = ['#acrCustomerReviewCount', '#acrCustomerReviewText'];
  for (const sel of countSelectors) {
    const el = document.querySelector(sel);
    if (el) {
      const text = el.textContent.replace(/[^0-9]/g, "");
      const val = parseInt(text, 10);
      if (!isNaN(val) && val > 0) { data.amzCount = val; break; }
    }
  }

  // Page count - scan all detail sections
  const allText = document.body.innerText;
  const pageMatch = allText.match(/(\d+)\s*pages/i);
  if (pageMatch) data.pageCount = parseInt(pageMatch[1], 10);

  // Publication year from detail bullets
  const detailItems = document.querySelectorAll('#detailBullets_feature_div li, .detail-bullet-list .a-list-item, #productDetailsTable td, table.a-keyvalue td');
  for (const item of detailItems) {
    const text = item.textContent;
    if (text.toLowerCase().includes("publisher") || text.toLowerCase().includes("publication")) {
      const dm = text.match(/(\d{4})/);
      if (dm) { data.pubYear = parseInt(dm[1], 10); break; }
    }
  }

  return data;
}

async function extractCurrentPage() {
  let tab;
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    tab = tabs[0];
  } catch (err) {
    $("global-status").textContent = "Cannot query tabs: " + err.message;
    $("global-status").className = "status error";
    return null;
  }

  if (!tab) {
    $("global-status").textContent = "No active tab found";
    $("global-status").className = "status error";
    return null;
  }

  const url = tab.url || "";
  $("global-status").textContent = "Tab URL: " + url.substring(0, 60) + "...";

  let site = null;
  let func = null;

  if (url.includes("goodreads.com")) {
    site = "goodreads";
    func = extractGoodreadsData;
  } else if (url.includes("amazon.com") || url.includes("amazon.co.uk")) {
    site = "amazon";
    func = extractAmazonData;
  } else {
    $("global-status").textContent = "Not a Goodreads or Amazon page: " + url.substring(0, 60);
    $("global-status").className = "status error";
    return null;
  }

  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: func,
    });

    if (results && results[0] && results[0].result) {
      return { site, data: results[0].result };
    } else {
      $("global-status").textContent = "executeScript returned empty: " + JSON.stringify(results);
      $("global-status").className = "status error";
      return null;
    }
  } catch (err) {
    $("global-status").textContent = "executeScript error: " + err.message;
    $("global-status").className = "status error";
    return null;
  }
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str || "";
  return d.innerHTML;
}

async function searchGoodreads(title, author) {
  $("cross-search").style.display = "block";
  $("search-label").textContent = "Searching Goodreads...";
  $("search-status").textContent = "Looking up ratings on Goodreads...";
  $("search-status").className = "status searching";

  try {
    const resp = await chrome.runtime.sendMessage({
      type: "SEARCH_GOODREADS", title, author,
    });

    if (!resp || !resp.success) {
      $("search-status").textContent = "Goodreads search failed: " + (resp?.error || "no response");
      $("search-status").className = "status error";
      return;
    }

    // Parse the search results HTML
    const parser = new DOMParser();
    const doc = parser.parseFromString(resp.html, "text/html");
    const results = [];
    const rows = doc.querySelectorAll("tr[itemscope]");
    for (const row of rows) {
      const titleA = row.querySelector("a.bookTitle");
      const authorA = row.querySelector("a.authorName");
      const ratingSpan = row.querySelector("span.minirating");
      if (titleA) {
        const t = titleA.textContent.trim();
        const href = titleA.getAttribute("href");
        const a = authorA ? authorA.textContent.trim() : "";
        let rating = null, count = null;
        if (ratingSpan) {
          const m = ratingSpan.textContent.match(/([\d.]+)\s*avg.*?([\d,]+)\s*rating/);
          if (m) { rating = parseFloat(m[1]); count = parseInt(m[2].replace(/,/g, ""), 10); }
        }
        results.push({ title: t, author: a, href: `https://www.goodreads.com${href}`, rating, count });
      }
    }

    if (results.length === 0) {
      $("search-status").textContent = "No Goodreads results found";
      $("search-status").className = "status error";
      return;
    }

    const best = results.find(r => r.count >= 100) || results[0];
    if (best.count >= 100) {
      await fetchGRDetails(best);
      return;
    }

    $("search-status").textContent = `Found ${results.length} results — click to select:`;
    $("search-status").className = "status";
    const container = $("search-results");
    container.innerHTML = "";
    for (const r of results.slice(0, 5)) {
      const div = document.createElement("div");
      div.className = "search-result";
      div.innerHTML = `<div class="sr-title">${escapeHtml(r.title)}</div><div class="sr-meta">${escapeHtml(r.author)} · ${r.rating || "?"} ★ · ${(r.count || 0).toLocaleString()} ratings</div>`;
      div.addEventListener("click", () => fetchGRDetails(r));
      container.appendChild(div);
    }
  } catch (err) {
    $("search-status").textContent = "Search error: " + err.message;
    $("search-status").className = "status error";
  }
}

async function fetchGRDetails(result) {
  $("search-status").textContent = `Loading: ${result.title}...`;
  $("search-status").className = "status searching";
  $("search-results").innerHTML = "";

  // Use search result data directly (fetching the page from background often gets blocked)
  pageData.grRating = result.rating;
  pageData.grCount = result.count;
  setField("gr-rating", result.rating);
  setField("gr-count", result.count?.toLocaleString());

  // Try to fetch the full page for extra details
  try {
    const resp = await chrome.runtime.sendMessage({ type: "FETCH_GR_PAGE", url: result.href });
    if (resp && resp.success) {
      const parser = new DOMParser();
      const doc = parser.parseFromString(resp.html, "text/html");

      const ratingEl = doc.querySelector('div.RatingStatistics__rating');
      if (ratingEl) {
        const val = parseFloat(ratingEl.textContent.trim());
        if (!isNaN(val)) { pageData.grRating = val; setField("gr-rating", val); }
      }

      const countEl = doc.querySelector('span[data-testid="ratingsCount"]');
      if (countEl) {
        const val = parseInt(countEl.textContent.replace(/[^0-9]/g, ""), 10);
        if (!isNaN(val)) { pageData.grCount = val; setField("gr-count", val.toLocaleString()); }
      }

      const pageEl = doc.querySelector('p[data-testid="pagesFormat"]');
      if (pageEl) {
        const m = pageEl.textContent.match(/(\d+)\s*pages/i);
        if (m && !pageData.pageCount) { pageData.pageCount = parseInt(m[1], 10); setField("pages", pageData.pageCount); }
      }

      const pubEl = doc.querySelector('p[data-testid="publicationInfo"]');
      if (pubEl) {
        const m = pubEl.textContent.match(/(\d{4})/);
        if (m) pageData.pubYear = parseInt(m[1], 10);
      }
    }
  } catch (e) {
    // Fine, we already have search result data
  }

  $("search-status").textContent = `GR: ${result.title} (${pageData.grRating} ★, ${(pageData.grCount || 0).toLocaleString()} ratings)`;
  $("search-status").className = "status";
  $("search-label").textContent = "Goodreads Match";
}

async function searchAmazon(title, author) {
  $("cross-search").style.display = "block";
  $("search-label").textContent = "Searching Amazon...";
  $("search-status").textContent = "Looking up ratings on Amazon...";
  $("search-status").className = "status searching";

  try {
    const resp = await chrome.runtime.sendMessage({
      type: "SEARCH_AMAZON", title, author,
    });

    if (!resp || !resp.success) {
      $("search-status").textContent = "Amazon search failed: " + (resp?.error || "no response");
      $("search-status").className = "status error";
      return;
    }

    const parser = new DOMParser();
    const doc = parser.parseFromString(resp.html, "text/html");
    const results = [];
    const items = doc.querySelectorAll('[data-component-type="s-search-result"]');
    for (const item of items) {
      const titleEl = item.querySelector("h2 a span, h2 span");
      const linkEl = item.querySelector("h2 a");
      const ratingEl = item.querySelector(".a-icon-alt");
      const countEl = item.querySelector("a .a-size-base, span.a-size-base.s-underline-text");
      if (titleEl) {
        const t = titleEl.textContent.trim();
        let href = linkEl ? linkEl.getAttribute("href") : "";
        if (href && !href.startsWith("http")) href = `https://www.amazon.com${href}`;
        let rating = null, count = null;
        if (ratingEl) { const m = ratingEl.textContent.match(/([\d.]+)/); if (m) rating = parseFloat(m[1]); }
        if (countEl) { const text = countEl.textContent.replace(/[^0-9]/g, ""); count = parseInt(text, 10) || null; }
        results.push({ title: t, href, rating, count });
      }
    }

    if (results.length === 0) {
      $("search-status").textContent = "No Amazon results found (may be blocked by CAPTCHA)";
      $("search-status").className = "status error";
      return;
    }

    const best = results.find(r => r.rating != null) || results[0];
    pageData.amzRating = best.rating;
    pageData.amzCount = best.count;
    setField("amz-rating", best.rating);
    setField("amz-count", best.count?.toLocaleString());
    $("search-status").textContent = `AMZ: "${best.title.substring(0, 50)}" (${best.rating} ★, ${(best.count || 0).toLocaleString()} ratings)`;
    $("search-status").className = "status";
    $("search-label").textContent = "Cross-Site Match";
  } catch (err) {
    $("search-status").textContent = "Amazon search error: " + err.message;
    $("search-status").className = "status error";
  }
}

async function runPrediction() {
  $("predict-btn").disabled = true;
  $("predict-btn").textContent = "Predicting...";
  $("global-status").textContent = "";

  try {
    const resp = await chrome.runtime.sendMessage({
      type: "GET_PREDICTIONS",
      pageData,
      category: selectedCategory,
    });

    if (!resp || !resp.success) {
      $("global-status").textContent = "Prediction error: " + (resp?.error || "no response");
      $("global-status").className = "status error";
      $("predict-btn").disabled = false;
      $("predict-btn").textContent = `Predict for "${selectedCategory}"`;
      return;
    }

    const r = resp.results;
    $("results").style.display = "block";

    for (const [id, val, type] of [
      ["ridge-enjoy", r.ridge_enjoy, "enjoy"],
      ["ridge-useful", r.ridge_useful, "useful"],
      ["rf-enjoy", r.rf_enjoy, "enjoy"],
      ["rf-useful", r.rf_useful, "useful"],
      ["gbm-enjoy", r.gbm_enjoy, "enjoy"],
      ["gbm-useful", r.gbm_useful, "useful"],
    ]) {
      const el = $(id);
      el.textContent = formatScore(val);
      el.className = `score ${scoreClass(val, type)}`;
    }

    $("predict-btn").disabled = false;
    $("predict-btn").textContent = `Predict for "${selectedCategory}"`;
  } catch (err) {
    $("global-status").textContent = "Error: " + err.message;
    $("global-status").className = "status error";
    $("predict-btn").disabled = false;
    $("predict-btn").textContent = `Predict for "${selectedCategory}"`;
  }
}

async function init() {
  initCategoryButtons();
  $("predict-btn").addEventListener("click", runPrediction);

  $("global-status").textContent = "Extracting page data...";
  $("global-status").className = "status searching";

  const extracted = await extractCurrentPage();
  if (!extracted || !extracted.data) {
    $("global-status").textContent = "Could not extract data. Make sure you're on a Goodreads or Amazon book page, then reopen this popup.";
    $("global-status").className = "status error";
    return;
  }

  const { site, data } = extracted;
  pageData = data || {};

  $("global-status").textContent = `Detected: ${site}`;
  $("global-status").className = "status";

  setField("title", data.title);
  setField("author", data.author);

  if (site === "goodreads") {
    setField("gr-rating", data.grRating);
    setField("gr-count", data.grCount?.toLocaleString());
    setField("amz-rating", null, "searching...");
    setField("amz-count", null, "searching...");
    setField("pages", data.pageCount);

    if (data.title) {
      searchAmazon(data.title, data.author);
    }
  } else if (site === "amazon") {
    setField("gr-rating", null, "searching...");
    setField("gr-count", null, "searching...");
    setField("amz-rating", data.amzRating);
    setField("amz-count", data.amzCount?.toLocaleString());
    setField("pages", data.pageCount);

    if (data.title) {
      searchGoodreads(data.title, data.author);
    }
  }
}

document.addEventListener("DOMContentLoaded", init);
