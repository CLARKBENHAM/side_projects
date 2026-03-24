// Background service worker — handles cross-site searches only.
// Predictions run locally in popup.js/content.js via predict.js.

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
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
