"""Build a single golden master sheet joining all book rating sources.

Sources (priority order for conflicts):
  1. master_book_metadata_cleaned.csv — corrected authors, verified GR ratings
  2. Play Export.csv — 1st-pass personal ratings, timestamps, categories, notes
  3. Ratings 2.csv — 2nd-pass personal ratings (same books as Play Export)
  4. new_books_to_rate 2026.csv — holdout ratings (1st+2nd), categories, notes
  5. finished_books_2025_03_16.csv — Google Takeout timestamps (more precise)
  6. dropbox_old_reading_list.txt — categories + dates for older books
  7. books_enriched_with_goodreads.csv / new_books_to_rate_2026_enriched.csv — GR URLs

Output: ai_books_tracking/golden_master.csv

Usage:
  python build_golden_master.py              # build the golden master
  python build_golden_master.py --propagate  # push golden edits to enriched CSVs
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
AI = Path(__file__).resolve().parent

# Manual title mappings: new_books short title → master canonical title
TITLE_ALIASES: dict[str, str] = {
    "kelly my share of it all": "kelly more than my share of it all",
    "buckley": "buckley the life and the revolution that changed america",
    "now it can be told": "now it can be told the story of the manhattan project",
    "shaping up": "shape up stop running in circles and ship work that matters",
}

# Manual filename→master title for books without proper filenames in master
FILENAME_TO_MASTER_TITLE: dict[str, str] = {
    "colin-bennent-trading-volatility-all sorts of info": (
        "Trading Volatility: Trading Volatility, Correlation, Term Structure and Skew"
    ),
}

# Dropbox filenames that map to golden master titles by word similarity
# (for cases where automatic matching fails)
DROPBOX_TO_GOLDEN: dict[str, str] = {
    "enemies of promise": "Enemies-Of-Promise by CYRIL CONNOLLY.pdf",
    "colin bennent trading volatility all sorts of info": (
        "Trading Volatility: Trading Volatility, Correlation, Term Structure and Skew"
    ),
}

# Dropbox category → normalized bookshelf mapping
DROPBOX_CAT_MAP: dict[str, str] = {
    "Advanced Finance": "Advanced Finance",
    "Business, management": "Business, management",
    "Computer Science": "Computer Science",
    "Energy Trading": "Energy Trading",
    "General Reading": "General Reading",
    "Literature": "Literature",
    "Math": "Math",
}


def _norm_title(title: str) -> str:
    """Lowercase, replace all non-alnum with spaces, collapse whitespace."""
    t = str(title).lower().strip()
    t = re.sub(r"[^a-z0-9]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _norm_filename(fn: str) -> str:
    """Normalize filename: lowercase, strip (N) suffixes and extensions."""
    fn = str(fn).lower().strip()
    fn = re.sub(r"\(\d+\)", "", fn)
    fn = re.sub(r"\.(pdf|epub|mobi|html|txt)$", "", fn)
    return re.sub(r"\s+", " ", fn).strip()


def _parse_date(d: object) -> pd.Timestamp | None:
    """Try to parse a date from various formats."""
    if pd.isna(d):
        return None
    s = str(d).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, format="mixed", dayfirst=False)
    except Exception:
        return None


def _parse_dropbox(path: Path) -> pd.DataFrame:
    """Parse dropbox_old_reading_list.txt → DataFrame with columns:
    dropbox_filename, dropbox_filename_norm, dropbox_date, dropbox_category
    """
    records: list[dict[str, str | None]] = []
    current_category: str | None = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," not in line or not re.search(r"\d{4}-\d{2}-\d{2}", line):
                current_category = line
            else:
                parts = line.rsplit(",", 1)
                filename = parts[0].strip()
                date = parts[1].strip() if len(parts) > 1 else None
                records.append(
                    {
                        "dropbox_filename": filename,
                        "dropbox_filename_norm": _norm_filename(filename),
                        "dropbox_date": date,
                        "dropbox_category": DROPBOX_CAT_MAP.get(
                            current_category or "", current_category
                        ),
                    }
                )
    return pd.DataFrame(records)


def _pick_best_date(
    *candidates: object, prefer_latest: bool = False
) -> pd.Timestamp | None:
    """Return earliest (or latest if prefer_latest) non-null date."""
    parsed = [_parse_date(d) for d in candidates]
    valid = [d for d in parsed if d is not None]
    if not valid:
        return None
    return max(valid) if prefer_latest else min(valid)


def build() -> pd.DataFrame:
    """Build the golden master from all sources."""
    # ── Load sources ──────────────────────────────────────────────
    master = pd.read_csv(
        DATA / "Books Read and their effects - master_book_metadata_cleaned.csv"
    )
    play = pd.read_csv(
        DATA / "Books Read and their effects - Play Export.csv",
        usecols=range(9),
    )
    ratings2 = pd.read_csv(
        DATA / "Books Read and their effects - Ratings 2.csv",
        usecols=range(8),
    )
    new_books = pd.read_csv(
        DATA / "Books Read and their effects - new_books_to_rate 2026.csv"
    )
    finished = pd.read_csv(DATA / "finished_books_2025_03_16.csv")
    dropbox = _parse_dropbox(DATA / "dropbox_old_reading_list.txt")

    # Enriched CSVs for GR URLs + metadata
    enr_hist = pd.read_csv(
        AI / "books_enriched_with_goodreads.csv",
        usecols=[
            "title",
            "goodreads_url",
            "goodreads_url_raw_best",
            "gb_page_count",
            "pub_year",
        ],
    )
    enr_hold = pd.read_csv(
        AI / "new_books_to_rate_2026_enriched.csv",
        usecols=[
            "title",
            "goodreads_url",
            "goodreads_url_raw_best",
            "gb_page_count",
            "pub_year",
        ],
    )
    enr = pd.concat([enr_hist, enr_hold], ignore_index=True)
    enr["_enr_title_norm"] = enr["title"].apply(_norm_title)
    # Prefer goodreads_url_raw_best, fallback to goodreads_url
    enr["_best_url"] = enr["goodreads_url_raw_best"].fillna(enr["goodreads_url"])

    # ── Build lookup indices ──────────────────────────────────────
    # Play Export: index by normalized filename
    play["_fn_norm"] = play["filename"].apply(
        lambda x: _norm_filename(x) if pd.notna(x) else ""
    )
    play_by_fn: dict[str, pd.Series] = {}
    for _, row in play.iterrows():
        if row["_fn_norm"]:
            play_by_fn[row["_fn_norm"]] = row

    # Also index play by normalized title for fallback
    play["_title_norm"] = play["title"].apply(_norm_title)
    play_by_title: dict[str, pd.Series] = {}
    for _, row in play.iterrows():
        play_by_title[row["_title_norm"]] = row

    # Ratings 2: index by normalized filename
    ratings2["_fn_norm"] = ratings2["filename"].apply(
        lambda x: _norm_filename(x) if pd.notna(x) else ""
    )
    r2_by_fn: dict[str, pd.Series] = {}
    for _, row in ratings2.iterrows():
        if row["_fn_norm"]:
            r2_by_fn[row["_fn_norm"]] = row

    # new_books: index by normalized title
    new_books["_title_norm"] = new_books["title"].apply(_norm_title)
    nb_by_title: dict[str, pd.Series] = {}
    for _, row in new_books.iterrows():
        nb_by_title[row["_title_norm"]] = row

    # finished_books: index by normalized filename
    finished["_fn_norm"] = finished["filename"].apply(
        lambda x: _norm_filename(x) if pd.notna(x) else ""
    )
    fin_by_fn: dict[str, pd.Series] = {}
    for _, row in finished.iterrows():
        if row["_fn_norm"]:
            fin_by_fn[row["_fn_norm"]] = row

    # dropbox: index by normalized filename
    db_by_fn: dict[str, pd.Series] = {}
    for _, row in dropbox.iterrows():
        db_by_fn[row["dropbox_filename_norm"]] = row

    # enriched: index by normalized title
    enr_by_title: dict[str, pd.Series] = {}
    for _, row in enr.iterrows():
        enr_by_title[row["_enr_title_norm"]] = row

    # Track which Play Export / new_books rows we've matched
    matched_play_fns: set[str] = set()
    matched_nb_titles: set[str] = set()

    # ── Build golden rows ─────────────────────────────────────────
    golden_rows: list[dict] = []

    for _, m in master.iterrows():
        title = str(m["title"])
        author = (
            str(m["corrected_author"])
            if pd.notna(m["corrected_author"])
            else str(m.get("author(old and wrong)", ""))
        )
        source = str(m["source"])
        filename = str(m["filename"]) if pd.notna(m["filename"]) else ""
        fn_norm = _norm_filename(filename) if filename else ""
        title_norm = _norm_title(title)

        gr_rating = m.get("goodread ratings", m.get("ratings"))
        gr_rating = gr_rating if pd.notna(gr_rating) else None
        gr_count = m.get("goodreads number ratings", m.get("number ratings"))
        gr_count = gr_count if pd.notna(gr_count) else None
        gr_reviews = m.get("goodreads number reviews", m.get("number reviews"))
        gr_reviews = gr_reviews if pd.notna(gr_reviews) else None

        row: dict[str, object] = {
            "title": title,
            "author": author,
            "category": None,
            "estimated_start": None,
            "estimated_finish": None,
            "enjoyment_1st": None,
            "usefulness_1st": None,
            "enjoyment_2nd": None,
            "usefulness_2nd": None,
            "goodreads_rating": gr_rating,
            "goodreads_rating_count": gr_count,
            "goodreads_review_count": gr_reviews,
            "goodreads_url": None,
            "page_count": None,
            "pub_year": None,
            "long_term_effects": None,
            "source": source,
            "filename": filename if filename else None,
            "needs_review": "",
        }

        # ── Match to Play Export ──────────────────────────────────
        play_row = None
        if source == "Play Export":
            # Try filename match first
            if fn_norm and fn_norm in play_by_fn:
                play_row = play_by_fn[fn_norm]
                matched_play_fns.add(fn_norm)
            else:
                # Try FILENAME_TO_MASTER_TITLE reverse lookup
                for db_fn, m_title in FILENAME_TO_MASTER_TITLE.items():
                    if m_title == title and db_fn in play_by_fn:
                        play_row = play_by_fn[db_fn]
                        matched_play_fns.add(db_fn)
                        break
                # Fallback: title match
                if play_row is None and title_norm in play_by_title:
                    play_row = play_by_title[title_norm]
                    matched_play_fns.add(play_row["_fn_norm"])

            if play_row is not None:
                row["category"] = (
                    play_row["Bookshelf"]
                    if pd.notna(play_row.get("Bookshelf"))
                    else None
                )
                row["enjoyment_1st"] = (
                    play_row["Enjoyment (/5)"]
                    if pd.notna(play_row.get("Enjoyment (/5)"))
                    else None
                )
                row["usefulness_1st"] = (
                    play_row["Usefulness /5 to Me"]
                    if pd.notna(play_row.get("Usefulness /5 to Me"))
                    else None
                )
                row["long_term_effects"] = (
                    str(play_row["Long Term Effects"])
                    if pd.notna(play_row.get("Long Term Effects"))
                    else None
                )
                row["estimated_start"] = _parse_date(play_row.get("earliest_modified"))
                row["estimated_finish"] = _parse_date(play_row.get("latest_modified"))
                if not filename:
                    row["filename"] = (
                        play_row["filename"]
                        if pd.notna(play_row.get("filename"))
                        else None
                    )

                # ── Match Ratings 2 for 2nd pass ──────────────────
                play_fn = play_row["_fn_norm"]
                if play_fn and play_fn in r2_by_fn:
                    r2_row = r2_by_fn[play_fn]
                    row["enjoyment_2nd"] = (
                        r2_row["Enjoyment (/5)"]
                        if pd.notna(r2_row.get("Enjoyment (/5)"))
                        else None
                    )
                    row["usefulness_2nd"] = (
                        r2_row["Usefulness /5 to Me"]
                        if pd.notna(r2_row.get("Usefulness /5 to Me"))
                        else None
                    )
            else:
                row["needs_review"] = "Play Export book not matched to ratings sheet"

        # ── Match to new_books_2026 ───────────────────────────────
        elif source == "Holdout 2026":
            nb_row = None
            # Try direct title match
            if title_norm in nb_by_title:
                nb_row = nb_by_title[title_norm]
                matched_nb_titles.add(title_norm)
            else:
                # Try alias mapping (master title → new_books title)
                for nb_short, master_norm in TITLE_ALIASES.items():
                    if title_norm == master_norm and nb_short in nb_by_title:
                        nb_row = nb_by_title[nb_short]
                        matched_nb_titles.add(nb_short)
                        break

            if nb_row is not None:
                row["category"] = (
                    nb_row["Bookshelf"] if pd.notna(nb_row.get("Bookshelf")) else None
                )
                row["enjoyment_1st"] = (
                    nb_row["Enjoyment (/5)"]
                    if pd.notna(nb_row.get("Enjoyment (/5)"))
                    else None
                )
                row["usefulness_1st"] = (
                    nb_row["Usefulness /5 to Me"]
                    if pd.notna(nb_row.get("Usefulness /5 to Me"))
                    else None
                )
                row["enjoyment_2nd"] = (
                    nb_row["Enjoyment (/5) 2nd"]
                    if pd.notna(nb_row.get("Enjoyment (/5) 2nd"))
                    else None
                )
                row["usefulness_2nd"] = (
                    nb_row["Usefulness /5 to Me.1"]
                    if pd.notna(nb_row.get("Usefulness /5 to Me.1"))
                    else None
                )
                row["long_term_effects"] = (
                    str(nb_row["Long Term Effects"])
                    if pd.notna(nb_row.get("Long Term Effects"))
                    else None
                )
                finish_date = _parse_date(nb_row.get("date_finished"))
                row["estimated_start"] = finish_date
                row["estimated_finish"] = finish_date
            else:
                row["needs_review"] = "Holdout book not matched to new_books_2026"

        # ── Better timestamps from finished_books ─────────────────
        fn_for_lookup = _norm_filename(str(row["filename"])) if row["filename"] else ""
        if fn_for_lookup and fn_for_lookup in fin_by_fn:
            fin_row = fin_by_fn[fn_for_lookup]
            fin_start = _parse_date(fin_row.get("earliest_modified"))
            fin_end = _parse_date(fin_row.get("latest_modified"))
            row["estimated_start"] = _pick_best_date(row["estimated_start"], fin_start)
            row["estimated_finish"] = _pick_best_date(
                row["estimated_finish"], fin_end, prefer_latest=True
            )

        # ── Dropbox dates and categories ──────────────────────────
        if fn_for_lookup and fn_for_lookup in db_by_fn:
            db_row = db_by_fn[fn_for_lookup]
            db_date = _parse_date(db_row["dropbox_date"])
            if db_date:
                row["estimated_start"] = _pick_best_date(
                    row["estimated_start"], db_date
                )
                row["estimated_finish"] = _pick_best_date(
                    row["estimated_finish"], db_date, prefer_latest=True
                )
            if not row["category"] and pd.notna(db_row.get("dropbox_category")):
                row["category"] = db_row["dropbox_category"]

        # ── Enriched CSV metadata ─────────────────────────────────
        enr_row = enr_by_title.get(title_norm)
        if enr_row is not None:
            if pd.notna(enr_row.get("_best_url")):
                row["goodreads_url"] = enr_row["_best_url"]
            if pd.notna(enr_row.get("gb_page_count")):
                row["page_count"] = int(enr_row["gb_page_count"])
            if pd.notna(enr_row.get("pub_year")):
                row["pub_year"] = int(enr_row["pub_year"])

        # ── Compute averages ──────────────────────────────────────
        e1 = row["enjoyment_1st"]
        e2 = row["enjoyment_2nd"]
        u1 = row["usefulness_1st"]
        u2 = row["usefulness_2nd"]
        row["avg_enjoyment"] = (
            np.mean([x for x in [e1, e2] if x is not None])
            if any(x is not None for x in [e1, e2])
            else None
        )
        row["avg_usefulness"] = (
            np.mean([x for x in [u1, u2] if x is not None])
            if any(x is not None for x in [u1, u2])
            else None
        )

        golden_rows.append(row)

    # ── Add unmatched Play Export books ────────────────────────────
    for _, p in play.iterrows():
        if p["_fn_norm"] and p["_fn_norm"] not in matched_play_fns:
            fn_norm = p["_fn_norm"]
            title = str(p["title"])
            title_norm = _norm_title(title)

            # Check if this is also in new_books (duplicate like "Kelly")
            is_dup = title_norm in nb_by_title or any(
                alias == title_norm or alias.startswith(title_norm + " ")
                for alias in TITLE_ALIASES
            )

            row = {
                "title": title,
                "author": str(p["author"]) if pd.notna(p.get("author")) else "",
                "category": (p["Bookshelf"] if pd.notna(p.get("Bookshelf")) else None),
                "estimated_start": _parse_date(p.get("earliest_modified")),
                "estimated_finish": _parse_date(p.get("latest_modified")),
                "enjoyment_1st": (
                    p["Enjoyment (/5)"] if pd.notna(p.get("Enjoyment (/5)")) else None
                ),
                "usefulness_1st": (
                    p["Usefulness /5 to Me"]
                    if pd.notna(p.get("Usefulness /5 to Me"))
                    else None
                ),
                "enjoyment_2nd": None,
                "usefulness_2nd": None,
                "goodreads_rating": None,
                "goodreads_rating_count": None,
                "goodreads_review_count": None,
                "goodreads_url": None,
                "page_count": None,
                "pub_year": None,
                "long_term_effects": (
                    str(p["Long Term Effects"])
                    if pd.notna(p.get("Long Term Effects"))
                    else None
                ),
                "source": "Play Export (unverified)",
                "filename": p["filename"] if pd.notna(p.get("filename")) else None,
                "needs_review": (
                    "Duplicate with Holdout 2026"
                    if is_dup
                    else "Not in master_book_metadata_cleaned — no verified GR data"
                ),
            }

            # Ratings 2 for 2nd pass
            if fn_norm in r2_by_fn:
                r2_row = r2_by_fn[fn_norm]
                row["enjoyment_2nd"] = (
                    r2_row["Enjoyment (/5)"]
                    if pd.notna(r2_row.get("Enjoyment (/5)"))
                    else None
                )
                row["usefulness_2nd"] = (
                    r2_row["Usefulness /5 to Me"]
                    if pd.notna(r2_row.get("Usefulness /5 to Me"))
                    else None
                )

            # Finished books timestamps
            if fn_norm in fin_by_fn:
                fin_row = fin_by_fn[fn_norm]
                row["estimated_start"] = _pick_best_date(
                    row["estimated_start"],
                    fin_row.get("earliest_modified"),
                )
                row["estimated_finish"] = _pick_best_date(
                    row["estimated_finish"],
                    fin_row.get("latest_modified"),
                    prefer_latest=True,
                )

            # Dropbox
            if fn_norm in db_by_fn:
                db_row = db_by_fn[fn_norm]
                db_date = _parse_date(db_row["dropbox_date"])
                if db_date:
                    row["estimated_start"] = _pick_best_date(
                        row["estimated_start"], db_date
                    )
                    row["estimated_finish"] = _pick_best_date(
                        row["estimated_finish"], db_date, prefer_latest=True
                    )
                if not row["category"]:
                    row["category"] = db_row.get("dropbox_category")

            # Enriched metadata
            enr_row = enr_by_title.get(title_norm)
            if enr_row is not None:
                if pd.notna(enr_row.get("_best_url")):
                    row["goodreads_url"] = enr_row["_best_url"]

            e1 = row["enjoyment_1st"]
            e2 = row["enjoyment_2nd"]
            u1 = row["usefulness_1st"]
            u2 = row["usefulness_2nd"]
            row["avg_enjoyment"] = (
                np.mean([x for x in [e1, e2] if x is not None])
                if any(x is not None for x in [e1, e2])
                else None
            )
            row["avg_usefulness"] = (
                np.mean([x for x in [u1, u2] if x is not None])
                if any(x is not None for x in [u1, u2])
                else None
            )

            golden_rows.append(row)

    # ── Add unmatched new_books entries ────────────────────────────
    for _, nb in new_books.iterrows():
        if nb["_title_norm"] not in matched_nb_titles:
            title = str(nb["title"])
            row = {
                "title": title,
                "author": str(nb["author"]) if pd.notna(nb.get("author")) else "",
                "category": (
                    nb["Bookshelf"] if pd.notna(nb.get("Bookshelf")) else None
                ),
                "estimated_start": _parse_date(nb.get("date_finished")),
                "estimated_finish": _parse_date(nb.get("date_finished")),
                "enjoyment_1st": (
                    nb["Enjoyment (/5)"] if pd.notna(nb.get("Enjoyment (/5)")) else None
                ),
                "usefulness_1st": (
                    nb["Usefulness /5 to Me"]
                    if pd.notna(nb.get("Usefulness /5 to Me"))
                    else None
                ),
                "enjoyment_2nd": (
                    nb["Enjoyment (/5) 2nd"]
                    if pd.notna(nb.get("Enjoyment (/5) 2nd"))
                    else None
                ),
                "usefulness_2nd": (
                    nb["Usefulness /5 to Me.1"]
                    if pd.notna(nb.get("Usefulness /5 to Me.1"))
                    else None
                ),
                "goodreads_rating": None,
                "goodreads_rating_count": None,
                "goodreads_review_count": None,
                "goodreads_url": None,
                "page_count": None,
                "pub_year": None,
                "long_term_effects": (
                    str(nb["Long Term Effects"])
                    if pd.notna(nb.get("Long Term Effects"))
                    else None
                ),
                "source": "Holdout 2026 (unverified)",
                "filename": None,
                "needs_review": "Not in master_book_metadata_cleaned — no verified GR data",
            }
            e1, e2 = row["enjoyment_1st"], row["enjoyment_2nd"]
            u1, u2 = row["usefulness_1st"], row["usefulness_2nd"]
            row["avg_enjoyment"] = (
                np.mean([x for x in [e1, e2] if x is not None])
                if any(x is not None for x in [e1, e2])
                else None
            )
            row["avg_usefulness"] = (
                np.mean([x for x in [u1, u2] if x is not None])
                if any(x is not None for x in [u1, u2])
                else None
            )
            golden_rows.append(row)

    # ── Match dropbox entries to existing golden rows ────────────
    # Build indices from golden_rows for matching
    golden_by_fn: dict[str, int] = {}
    golden_by_title: dict[str, int] = {}
    for idx, gr in enumerate(golden_rows):
        if gr.get("filename"):
            fn = _norm_filename(str(gr["filename"]))
            if fn:
                golden_by_fn[fn] = idx
        tn = _norm_title(str(gr["title"]))
        golden_by_title[tn] = idx
    # Also include matched play filenames (master may have different filename)
    for fn in matched_play_fns:
        if fn and fn not in golden_by_fn:
            # Find the golden row this play filename maps to
            play_row = play_by_fn.get(fn)
            if play_row is not None:
                play_title_norm = _norm_title(str(play_row["title"]))
                if play_title_norm in golden_by_title:
                    golden_by_fn[fn] = golden_by_title[play_title_norm]

    # Words to strip from titles (file extensions, markers)
    _JUNK_WORDS = {"pdf", "epub", "mobi", "html", "txt", "keep"}

    def _title_words(t: str) -> set[str]:
        """Extract significant words (>2 chars, not junk) from normalized title."""
        return {
            w for w in _norm_title(t).split() if len(w) > 2 and w not in _JUNK_WORDS
        }

    def _find_golden_match(db_fn_norm: str, db_filename: str) -> int | None:
        """Find the golden_rows index matching a dropbox entry."""
        # 0. Manual mapping
        db_fn_title_stripped = _norm_title(
            re.sub(r"\.(pdf|epub|mobi|html|txt)$", "", db_filename, flags=re.I)
        )
        for db_key, golden_title in DROPBOX_TO_GOLDEN.items():
            if db_fn_title_stripped == db_key:
                g_norm = _norm_title(golden_title)
                if g_norm in golden_by_title:
                    return golden_by_title[g_norm]
        # 1. Exact normalized filename
        if db_fn_norm in golden_by_fn:
            return golden_by_fn[db_fn_norm]
        # 2. Check if any golden title is contained in the dropbox filename
        #    (min 6 chars to avoid false positives)
        db_fn_title_norm = _norm_title(db_filename)
        for g_title, g_idx in golden_by_title.items():
            if len(g_title) >= 6 and g_title in db_fn_title_norm:
                return g_idx
        # 3. Check if dropbox is "Title by Author" format
        if " by " in db_filename.lower():
            title_part = db_filename.lower().split(" by ")[0].strip()
            title_part_norm = _norm_title(title_part)
            if len(title_part_norm) >= 4:
                for g_title, g_idx in golden_by_title.items():
                    if (
                        g_title.startswith(title_part_norm)
                        or title_part_norm in g_title
                    ):
                        return g_idx
        # 4. Check first 3+ significant words overlap
        db_words = [w for w in db_fn_title_norm.split() if len(w) > 2][:4]
        if len(db_words) >= 3:
            db_prefix = " ".join(db_words)
            for g_title, g_idx in golden_by_title.items():
                if g_title.startswith(db_prefix):
                    return g_idx
        # 5. Word-set overlap: if >60% of the smaller word set overlaps
        db_ws = _title_words(db_filename)
        if len(db_ws) >= 3:
            best_overlap = 0.0
            best_idx: int | None = None
            for g_title, g_idx in golden_by_title.items():
                g_ws = _title_words(g_title)
                if not g_ws:
                    continue
                overlap = len(db_ws & g_ws)
                smaller = min(len(db_ws), len(g_ws))
                ratio = overlap / smaller if smaller else 0
                if ratio > best_overlap and ratio >= 0.6 and overlap >= 3:
                    best_overlap = ratio
                    best_idx = g_idx
            if best_idx is not None:
                return best_idx
        return None

    for _, db in dropbox.iterrows():
        db_fn_norm = db["dropbox_filename_norm"]
        db_filename = str(db["dropbox_filename"])
        match_idx = _find_golden_match(db_fn_norm, db_filename)

        if match_idx is not None:
            # Enhance existing golden row with dropbox date/category
            gr = golden_rows[match_idx]
            db_date = _parse_date(db["dropbox_date"])
            if db_date:
                gr["estimated_start"] = _pick_best_date(gr["estimated_start"], db_date)
                gr["estimated_finish"] = _pick_best_date(
                    gr["estimated_finish"], db_date, prefer_latest=True
                )
            if not gr["category"] and pd.notna(db.get("dropbox_category")):
                gr["category"] = db["dropbox_category"]
        else:
            golden_rows.append(
                {
                    "title": db_filename,
                    "author": "",
                    "category": db.get("dropbox_category"),
                    "estimated_start": _parse_date(db.get("dropbox_date")),
                    "estimated_finish": _parse_date(db.get("dropbox_date")),
                    "enjoyment_1st": None,
                    "usefulness_1st": None,
                    "enjoyment_2nd": None,
                    "usefulness_2nd": None,
                    "avg_enjoyment": None,
                    "avg_usefulness": None,
                    "goodreads_rating": None,
                    "goodreads_rating_count": None,
                    "goodreads_review_count": None,
                    "goodreads_url": None,
                    "page_count": None,
                    "pub_year": None,
                    "long_term_effects": None,
                    "source": "Dropbox only",
                    "filename": db_filename,
                    "needs_review": "Dropbox only — no ratings available. Add ratings or remove.",
                }
            )

    # ── Build DataFrame and sort ──────────────────────────────────
    df = pd.DataFrame(golden_rows)

    # Format dates
    for col in ["estimated_start", "estimated_finish"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].dt.strftime("%Y-%m-%d")
        df[col] = df[col].replace("NaT", "")

    # Sort: needs_review first, then by estimated_finish descending
    df["_sort_review"] = df["needs_review"].apply(lambda x: 0 if x else 1)
    df["_sort_date"] = pd.to_datetime(df["estimated_finish"], errors="coerce")
    df = df.sort_values(
        ["_sort_review", "_sort_date"], ascending=[True, False]
    ).reset_index(drop=True)
    df = df.drop(columns=["_sort_review", "_sort_date"])

    # Column order
    col_order = [
        "title",
        "author",
        "category",
        "estimated_start",
        "estimated_finish",
        "enjoyment_1st",
        "usefulness_1st",
        "enjoyment_2nd",
        "usefulness_2nd",
        "avg_enjoyment",
        "avg_usefulness",
        "goodreads_rating",
        "goodreads_rating_count",
        "goodreads_review_count",
        "goodreads_url",
        "page_count",
        "pub_year",
        "long_term_effects",
        "source",
        "filename",
        "needs_review",
    ]
    df = df[col_order]
    df["needs_review"] = df["needs_review"].fillna("")

    return df


def propagate(golden_path: Path | None = None) -> None:
    """Push edits from golden master back to enriched CSVs.

    Propagates: author, goodreads_rating, goodreads_rating_count, category.
    Matches by normalized title.
    """
    golden_path = golden_path or (AI / "golden_master.csv")
    golden = pd.read_csv(golden_path)
    golden["_title_norm"] = golden["title"].apply(_norm_title)

    targets = [
        (
            AI / "books_enriched_with_goodreads.csv",
            {
                "author": "author",
                "goodreads_rating": "goodreads_rating_raw_best",
                "goodreads_rating_count": "goodreads_rating_count_raw_best",
                "category": "Bookshelf",
            },
        ),
        (
            AI / "new_books_to_rate_2026_enriched.csv",
            {
                "author": "author",
                "goodreads_rating": "goodreads_rating_raw_best",
                "goodreads_rating_count": "goodreads_rating_count_raw_best",
                "category": "Bookshelf",
            },
        ),
    ]

    for target_path, col_map in targets:
        if not target_path.exists():
            print(f"  Skipping {target_path.name}: not found")
            continue
        df = pd.read_csv(target_path)
        df["_title_norm"] = df["title"].apply(_norm_title)
        changes = 0
        for _, g in golden.iterrows():
            mask = df["_title_norm"] == g["_title_norm"]
            if not mask.any():
                continue
            for g_col, t_col in col_map.items():
                if t_col not in df.columns:
                    continue
                new_val = g.get(g_col)
                if pd.isna(new_val):
                    continue
                old_vals = df.loc[mask, t_col]
                if not old_vals.empty and old_vals.iloc[0] != new_val:
                    df.loc[mask, t_col] = new_val
                    changes += 1
        df = df.drop(columns=["_title_norm"])
        df.to_csv(target_path, index=False)
        print(f"  {target_path.name}: {changes} field(s) updated")


def main() -> None:
    if "--propagate" in sys.argv:
        print("Propagating golden master edits to enriched CSVs...")
        propagate()
        return

    print("Building golden master...")
    df = build()
    out = AI / "golden_master.csv"
    df.to_csv(out, index=False)

    n_review = (df["needs_review"] != "").sum()
    n_rated = df["enjoyment_1st"].notna().sum()
    n_gr = df["goodreads_rating"].notna().sum()
    n_dates = (df["estimated_finish"].notna() & (df["estimated_finish"] != "")).sum()

    print(f"  {len(df)} total books")
    print(f"  {n_rated} with personal ratings")
    print(f"  {n_gr} with Goodreads ratings")
    print(f"  {n_dates} with finish dates")
    print(f"  {n_review} flagged for review")
    if n_review:
        print("\nBooks needing review:")
        review = df[df["needs_review"] != ""][
            ["title", "author", "source", "needs_review"]
        ]
        for _, r in review.iterrows():
            print(f"  [{r['source']}] {r['title']}")
            if r["author"]:
                print(f"    author: {r['author']}")
            print(f"    reason: {r['needs_review']}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
