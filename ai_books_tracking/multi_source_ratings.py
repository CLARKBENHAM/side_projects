"""Fetch book ratings from multiple sources and compare predictive power.

Sources:
1. Google Books API (free, no auth for basic search)
2. Open Library API (free, no auth)
3. Goodreads (already cached from goodreads_ratings.py)
4. Gemini Flash grounded search for Amazon ratings

All results are cached to avoid repeated API calls.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR.parent / "data"

# Input files
ENRICHED_208 = OUTPUT_DIR / "books_enriched_with_goodreads.csv"
NEW_BOOKS_ENRICHED = OUTPUT_DIR / "new_books_to_rate_2026_enriched.csv"

# Cache files
GOOGLE_BOOKS_CACHE = OUTPUT_DIR / "google_books_cache.json"
OPENLIBRARY_CACHE = OUTPUT_DIR / "openlibrary_cache.json"
AMAZON_GEMINI_CACHE = OUTPUT_DIR / "amazon_ratings_gemini_cache.json"

# Output
MULTI_SOURCE_CSV = OUTPUT_DIR / "multi_source_ratings.csv"

GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_API = "https://openlibrary.org/search.json"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

# Rating source name -> DataFrame column with the numeric rating
SOURCES: dict[str, str] = {
    "Goodreads": "goodreads_rating_raw_best",
    "Google Books": "google_books_rating",
    "Open Library": "ol_rating",
    "Amazon": "amazon_rating",
}

TARGETS = [("Enjoyment (/5)", "Enjoyment"), ("Usefulness /5 to Me", "Usefulness")]

HOLDOUT_N = 30


# ---- Helpers ----


def load_cache(path: Path) -> dict[str, object]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, object], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def clean_title(title: str) -> str:
    """Strip file extensions, libgen artifacts, and underscores for API search."""
    title = re.sub(r"\.(pdf|epub|html|txt)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"\s*-?\s*libgen\.li\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"#\w+#", "", title)
    title = title.replace("_", " ")
    # Remove series/edition tags like "[Hyperion 1]" or "(Discworld 29)"
    title = re.sub(r"\[.*?\]", "", title)
    title = re.sub(r"\(.*?\)", "", title)
    return title.strip()


def clean_author(author: str) -> str:
    """Get first author name for search queries."""
    if not author or pd.isna(author):
        return ""
    return str(author).split(",")[0].strip()


def _parse_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_int(val: object) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from text, handling nested braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _format_r(r: float) -> str:
    return f"{r:>6.3f}" if not np.isnan(r) else "   nan"


def _cache_key(title: str, author: str) -> str:
    return f"{title}|||{author}"


# ---- Google Books API ----


def _gb_get(params: dict) -> dict:
    """GET with retry on 429."""
    resp = None
    for attempt in range(3):
        resp = requests.get(GOOGLE_BOOKS_API, params=params, timeout=15)
        if resp.status_code == 429:
            time.sleep(2 ** (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    assert resp is not None
    resp.raise_for_status()
    return {}


def fetch_google_books(
    title: str, author: str, cache: dict[str, object]
) -> dict[str, object] | None:
    """Fetch rating from Google Books API (free, no key needed)."""
    key = _cache_key(title, author)
    if key in cache:
        return cache[key]  # type: ignore[return-value]

    clean = clean_title(title)
    if len(clean) < 3:
        cache[key] = None
        return None

    first_author = clean_author(author)
    q_parts = [f"intitle:{clean}"]
    if first_author:
        q_parts.append(f"inauthor:{first_author}")
    params = {"q": "+".join(q_parts), "maxResults": 3, "printType": "books"}

    try:
        data = _gb_get(params)

        # Retry without author filter if no results
        if data.get("totalItems", 0) == 0 and first_author:
            time.sleep(0.5)
            params["q"] = f"intitle:{clean}"
            data = _gb_get(params)

        if data.get("totalItems", 0) == 0:
            cache[key] = None
            save_cache(cache, GOOGLE_BOOKS_CACHE)
            return None

        # Pick first result with a rating, or just the first result
        best: dict[str, object] | None = None
        for item in data.get("items", [])[:3]:
            vi = item.get("volumeInfo", {})
            result: dict[str, object] = {
                "gb_title": vi.get("title", ""),
                "gb_authors": vi.get("authors", []),
                "gb_rating": vi.get("averageRating"),
                "gb_rating_count": vi.get("ratingsCount"),
                "gb_page_count": vi.get("pageCount"),
                "gb_pub_date": vi.get("publishedDate", ""),
                "gb_categories": vi.get("categories", []),
            }
            if best is None:
                best = result
            if result["gb_rating"] is not None:
                best = result
                break

        cache[key] = best
        save_cache(cache, GOOGLE_BOOKS_CACHE)
        return best

    except requests.RequestException as e:
        print(f"  Google Books error for '{clean[:40]}': {e}")
        return None


# ---- Open Library API ----


def fetch_openlibrary(
    title: str, author: str, cache: dict[str, object]
) -> dict[str, object] | None:
    """Fetch rating from Open Library API."""
    key = _cache_key(title, author)
    if key in cache:
        return cache[key]  # type: ignore[return-value]

    clean = clean_title(title)
    if len(clean) < 3:
        cache[key] = None
        return None

    first_author = clean_author(author)
    q = f"{clean} {first_author}" if first_author else clean
    params = {
        "q": q,
        "limit": 1,
        "fields": "title,author_name,first_publish_year,number_of_pages_median,"
        "ratings_average,ratings_count,subject",
    }

    try:
        resp = requests.get(OPEN_LIBRARY_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Retry without author if no results
        if data.get("numFound", 0) == 0 and first_author:
            time.sleep(0.2)
            params["q"] = clean
            resp = requests.get(OPEN_LIBRARY_API, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        if data.get("numFound", 0) == 0:
            cache[key] = None
            save_cache(cache, OPENLIBRARY_CACHE)
            return None

        doc = data["docs"][0]
        result: dict[str, object] = {
            "ol_title": doc.get("title", ""),
            "ol_authors": doc.get("author_name", []),
            "ol_rating": doc.get("ratings_average"),
            "ol_rating_count": doc.get("ratings_count"),
            "ol_page_count": doc.get("number_of_pages_median"),
            "ol_pub_year": doc.get("first_publish_year"),
        }
        cache[key] = result
        save_cache(cache, OPENLIBRARY_CACHE)
        return result

    except requests.RequestException as e:
        print(f"  Open Library error for '{clean[:40]}': {e}")
        return None


# ---- Gemini grounded search for Amazon ratings ----


def fetch_amazon_via_gemini(
    title: str, author: str, cache: dict[str, object]
) -> dict[str, object] | None:
    """Use Gemini Flash with Google Search grounding to find Amazon book rating."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None

    key = _cache_key(title, author)
    if key in cache:
        return cache[key]  # type: ignore[return-value]

    clean = clean_title(title)
    first_author = clean_author(author)
    book_desc = f'"{clean}"'
    if first_author:
        book_desc += f" by {first_author}"

    prompt = (
        f"Find the Amazon.com customer rating for the book {book_desc}. "
        f"Return ONLY a JSON object with these fields:\n"
        f'- "amazon_rating": the average star rating (number like 4.3), or null if not found\n'
        f'- "amazon_rating_count": number of ratings (integer), or null\n'
        f'- "amazon_title": the exact title as listed on Amazon\n'
        f'- "amazon_author": the author as listed on Amazon\n'
        f"Return ONLY the JSON, no other text."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.0},
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        text = ""
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                text += part.get("text", "")

        obj = _extract_json(text)
        if obj is None:
            cache[key] = None
            save_cache(cache, AMAZON_GEMINI_CACHE)
            return None

        parsed: dict[str, object] = {
            "amazon_rating": _parse_float(obj.get("amazon_rating")),
            "amazon_rating_count": _parse_int(obj.get("amazon_rating_count")),
            "amazon_title": obj.get("amazon_title", ""),
            "amazon_author": obj.get("amazon_author", ""),
        }
        cache[key] = parsed
        save_cache(cache, AMAZON_GEMINI_CACHE)
        return parsed

    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  Gemini/Amazon error for '{clean[:40]}': {e}")
        return None


# ---- Build combined dataset ----


def build_book_list() -> pd.DataFrame:
    """Load all books (208 training + 68 new) into a single frame, sorted chronologically."""
    train = pd.read_csv(ENRICHED_208)
    train.columns = train.columns.str.strip()

    new = pd.read_csv(NEW_BOOKS_ENRICHED)
    new.columns = new.columns.str.strip()

    train["dataset"] = "train"
    new["dataset"] = "new"

    for df in [train, new]:
        df["date_sort"] = pd.to_datetime(
            df["latest_modified"], format="mixed", errors="coerce"
        )

    # Best available author for search queries
    for df in [train, new]:
        if "canonical_author" in df.columns:
            df["search_author"] = df["canonical_author"].fillna(
                df.get("goodreads_author", pd.Series(dtype=str))
            )
        elif "goodreads_author" in df.columns:
            df["search_author"] = df["goodreads_author"]
        else:
            df["search_author"] = ""

    common_cols = [
        "title",
        "Bookshelf",
        "Enjoyment (/5)",
        "Usefulness /5 to Me",
        "search_author",
        "dataset",
        "date_sort",
    ]
    for col in ["goodreads_rating_raw_best", "goodreads_rating_count_raw_best"]:
        if (col in train.columns or col in new.columns) and col not in common_cols:
            common_cols.append(col)

    train_sub = train[[c for c in common_cols if c in train.columns]].copy()
    new_sub = new[[c for c in common_cols if c in new.columns]].copy()

    return pd.concat([train_sub, new_sub], ignore_index=True)


def _unpack_gb(result: dict[str, object] | None) -> dict[str, object]:
    """Extract Google Books fields into DataFrame column names."""
    if not result:
        return {
            "google_books_rating": None,
            "google_books_count": None,
            "google_books_title": "",
        }
    return {
        "google_books_rating": _parse_float(result.get("gb_rating")),
        "google_books_count": _parse_int(result.get("gb_rating_count")),
        "google_books_title": str(result.get("gb_title", "")),
    }


def _unpack_ol(result: dict[str, object] | None) -> dict[str, object]:
    """Extract Open Library fields into DataFrame column names."""
    if not result:
        return {"ol_rating": None, "ol_rating_count": None, "ol_title": ""}
    return {
        "ol_rating": _parse_float(result.get("ol_rating")),
        "ol_rating_count": _parse_int(result.get("ol_rating_count")),
        "ol_title": str(result.get("ol_title", "")),
    }


def _unpack_amz(result: dict[str, object] | None) -> dict[str, object]:
    """Extract Amazon/Gemini fields into DataFrame column names."""
    if not result:
        return {"amazon_rating": None, "amazon_rating_count": None, "amazon_title": ""}
    return {
        "amazon_rating": _parse_float(result.get("amazon_rating")),
        "amazon_rating_count": _parse_int(result.get("amazon_rating_count")),
        "amazon_title": str(result.get("amazon_title", "")),
    }


def fetch_all_ratings(
    books: pd.DataFrame, skip_google_books: bool = False
) -> pd.DataFrame:
    """Fetch ratings from all sources for all books."""
    gb_cache = load_cache(GOOGLE_BOOKS_CACHE)
    ol_cache = load_cache(OPENLIBRARY_CACHE)
    amz_cache = load_cache(AMAZON_GEMINI_CACHE)

    gb_initial, ol_initial, amz_initial = len(gb_cache), len(ol_cache), len(amz_cache)
    has_gemini = bool(os.environ.get("GEMINI_API_KEY", ""))

    rows: list[dict[str, object]] = []
    n = len(books)

    for i, (_, row) in enumerate(books.iterrows()):
        title = str(row["title"])
        author = str(row.get("search_author", ""))

        # Google Books
        if skip_google_books:
            gb = gb_cache.get(_cache_key(title, author))
        else:
            gb = fetch_google_books(title, author, gb_cache)
            time.sleep(0.5)

        # Open Library
        ol = fetch_openlibrary(title, author, ol_cache)
        time.sleep(0.15)

        # Amazon via Gemini
        if has_gemini:
            amz = fetch_amazon_via_gemini(title, author, amz_cache)
            time.sleep(0.3)
        else:
            amz = None

        book_row: dict[str, object] = {}
        book_row.update(_unpack_gb(gb))
        book_row.update(_unpack_ol(ol))
        book_row.update(_unpack_amz(amz))
        rows.append(book_row)

        if (i + 1) % 20 == 0:
            print(f"  Fetched {i + 1}/{n} books")

    books = books.copy()
    result_df = pd.DataFrame(rows)
    for col in result_df.columns:
        books[col] = result_df[col].values

    print("\nCache growth:")
    print(f"  Google Books: {gb_initial} -> {len(gb_cache)}")
    print(f"  Open Library: {ol_initial} -> {len(ol_cache)}")
    print(f"  Amazon/Gemini: {amz_initial} -> {len(amz_cache)}")

    return books


# ---- Analysis ----


def eval_preds(actual: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    mask = ~np.isnan(preds) & ~np.isnan(actual)
    if mask.sum() < 3:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R": float("nan"), "N": 0}
    a, p = actual[mask], preds[mask]
    return {
        "MAE": mean_absolute_error(a, p),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "R": float(stats.pearsonr(a, p)[0]) if np.std(p) > 1e-10 else float("nan"),
        "N": int(mask.sum()),
    }


def _print_model_row(label: str, m: dict[str, float], n: int) -> None:
    print(
        f"    {label:<45} {m['MAE']:>6.3f} {m['RMSE']:>6.3f} "
        f"{_format_r(m['R'])} {n:>4}"
    )


def source_comparison(df: pd.DataFrame) -> None:
    """Compare predictive power of each rating source."""
    print("\n" + "=" * 70)
    print("RATING SOURCE COMPARISON")
    print("=" * 70)

    for target, target_label in TARGETS:
        print(f"\n  {target_label}:")
        print(
            f"    {'Source':<20} {'N':>5} {'Coverage':>9} "
            f"{'Pearson R':>10} {'Spearman':>10} {'p-val':>8}"
        )
        print("    " + "-" * 65)

        valid_target = df[target].notna()
        for name, col in SOURCES.items():
            if col not in df.columns:
                continue
            valid = valid_target & pd.to_numeric(df[col], errors="coerce").notna()
            n = valid.sum()
            if n < 5:
                print(
                    f"    {name:<20} {n:>5} {n / valid_target.sum() * 100:>8.1f}%   (too few)"
                )
                continue
            ratings = pd.to_numeric(df.loc[valid, col]).values
            personal = df.loc[valid, target].values
            r, p = stats.pearsonr(ratings, personal)
            rho, _ = stats.spearmanr(ratings, personal)
            pct = n / valid_target.sum() * 100
            print(
                f"    {name:<20} {n:>5} {pct:>8.1f}% "
                f"{r:>10.3f} {rho:>10.3f} {p:>8.4f}"
            )

    # Cross-source correlations
    print("\n  Cross-source correlations (Pearson R):")
    avail = [
        (name, col)
        for name, col in SOURCES.items()
        if col in df.columns
        and pd.to_numeric(df[col], errors="coerce").notna().sum() >= 5
    ]

    header = "    " + " " * 20
    for name, _ in avail:
        header += f"{name:>14}"
    print(header)
    for name_i, col_i in avail:
        row_str = f"    {name_i:<20}"
        vals_i = pd.to_numeric(df[col_i], errors="coerce")
        for _, col_j in avail:
            vals_j = pd.to_numeric(df[col_j], errors="coerce")
            valid = vals_i.notna() & vals_j.notna()
            if valid.sum() < 5:
                row_str += f"{'--':>14}"
            else:
                r, _ = stats.pearsonr(vals_i[valid].values, vals_j[valid].values)
                row_str += f"{r:>14.3f}"
        print(row_str)

    # Rating distribution by source
    print("\n  Rating distributions:")
    print(f"    {'Source':<20} {'N':>5} {'Mean':>6} {'Std':>6} {'Min':>5} {'Max':>5}")
    print("    " + "-" * 50)
    for name, col in SOURCES.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) < 1:
            continue
        print(
            f"    {name:<20} {len(vals):>5} {vals.mean():>6.2f} "
            f"{vals.std():>6.2f} {vals.min():>5.2f} {vals.max():>5.2f}"
        )


def holdout_model_comparison(df: pd.DataFrame) -> None:
    """Train models on first 178 of 208 training books, evaluate on last 30."""
    print("\n" + "=" * 70)
    print(f"HOLDOUT MODEL COMPARISON (train on 208-{HOLDOUT_N}, test {HOLDOUT_N})")
    print("=" * 70)

    train_df = df[df["dataset"] == "train"].copy()
    train_df = train_df.sort_values("date_sort", na_position="first").reset_index(
        drop=True
    )

    valid = train_df[train_df["Enjoyment (/5)"].notna()].copy()
    n = len(valid)
    train = valid.iloc[: n - HOLDOUT_N].copy()
    test = valid.iloc[n - HOLDOUT_N :].copy()

    avail_sources = [
        col
        for _, col in SOURCES.items()
        if col in train.columns
        and pd.to_numeric(train[col], errors="coerce").notna().sum() >= 10
    ]

    for target, label in TARGETS:
        print(f"\n  {label}:")
        print(f"    {'Model':<45} {'MAE':>6} {'RMSE':>6} {'R':>6} {'N':>4}")
        print("    " + "-" * 70)

        actual = test[target].values
        gm = train[target].mean()

        # Global mean baseline
        m = eval_preds(actual, np.full(len(test), gm))
        _print_model_row("Global mean", m, len(test))

        # Category mean baseline
        cat_means = train.groupby("Bookshelf")[target].mean().to_dict()
        cat_preds = test["Bookshelf"].map(cat_means).fillna(gm).values
        m = eval_preds(actual, cat_preds)
        _print_model_row("Category mean", m, len(test))

        # Each source as linear predictor
        for name, col in SOURCES.items():
            if col not in train.columns:
                continue
            tr_valid = train[pd.to_numeric(train[col], errors="coerce").notna()].copy()
            tr_vals = pd.to_numeric(tr_valid[col]).values
            if len(tr_valid) < 10:
                continue
            slope, intercept, _, _, _ = stats.linregress(
                tr_vals, tr_valid[target].values
            )
            te_vals = pd.to_numeric(test[col], errors="coerce")
            preds = np.where(te_vals.notna(), slope * te_vals + intercept, gm)
            m = eval_preds(actual, preds)
            _print_model_row(f"{name} (linear)", m, int(te_vals.notna().sum()))

        # Combined: average of available source ratings as linear predictor
        if len(avail_sources) >= 2:
            train_avg = (
                train[avail_sources].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            )
            test_avg = (
                test[avail_sources].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            )
            tr_valid_avg = train_avg.notna()
            if tr_valid_avg.sum() >= 10:
                slope, intercept, _, _, _ = stats.linregress(
                    train_avg[tr_valid_avg].values,
                    train.loc[tr_valid_avg, target].values,
                )
                preds = np.where(test_avg.notna(), slope * test_avg + intercept, gm)
                m = eval_preds(actual, preds)
                _print_model_row(
                    "Avg all sources (linear)", m, int(test_avg.notna().sum())
                )

        # Ridge: category + all source ratings
        if avail_sources:
            cat_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            cat_enc.fit(train[["Bookshelf"]])
            X_train_cat = cat_enc.transform(train[["Bookshelf"]])
            X_test_cat = cat_enc.transform(test[["Bookshelf"]])

            train_medians = (
                train[avail_sources].apply(pd.to_numeric, errors="coerce").median()
            )
            source_train = (
                train[avail_sources]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(train_medians)
                .values
            )
            source_test = (
                test[avail_sources]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(train_medians)
                .values
            )

            X_tr = np.hstack([X_train_cat, source_train])
            X_te = np.hstack([X_test_cat, source_test])

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr, train[target].values)
            preds = ridge.predict(X_te)
            m = eval_preds(actual, preds)
            src_names = "+".join([n for n, c in SOURCES.items() if c in avail_sources])
            _print_model_row(f"Ridge (cat + {src_names})", m, len(test))


def match_quality_report(df: pd.DataFrame) -> None:
    """Show matched titles across sources for verification."""
    print("\n" + "=" * 70)
    print("MATCH QUALITY SAMPLE (first 20 books)")
    print("=" * 70)
    print(f"  {'Our title':<35} {'GB match':<30} {'OL match':<30} {'AMZ match':<30}")
    print("  " + "-" * 125)

    def _fmt(row: pd.Series, title_col: str, rating_col: str) -> str:
        t = str(row.get(title_col, ""))[:24]
        r = row.get(rating_col)
        if pd.notna(r):
            return f"{t} ({r})"
        return t if t else "—"

    for _, row in df.head(20).iterrows():
        title = str(row["title"])[:34]
        gb = _fmt(row, "google_books_title", "google_books_rating")
        ol = _fmt(row, "ol_title", "ol_rating")
        amz = _fmt(row, "amazon_title", "amazon_rating")
        print(f"  {title:<35} {gb:<30} {ol:<30} {amz:<30}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-google-books",
        action="store_true",
        help="Skip Google Books API calls (use cache only)",
    )
    args = parser.parse_args()

    print("Loading books...")
    books = build_book_list()
    print(
        f"  Total: {len(books)} books ({(books['dataset'] == 'train').sum()} train, "
        f"{(books['dataset'] == 'new').sum()} new)"
    )

    print("\nFetching ratings from all sources...")
    books = fetch_all_ratings(books, skip_google_books=args.skip_google_books)

    # Coverage summary
    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)
    for name, col in SOURCES.items():
        if col in books.columns:
            n = pd.to_numeric(books[col], errors="coerce").notna().sum()
            print(f"  {name:<20}: {n}/{len(books)} ({n / len(books) * 100:.0f}%)")

    books.to_csv(MULTI_SOURCE_CSV, index=False)
    print(f"\nSaved: {MULTI_SOURCE_CSV.name}")

    source_comparison(books)
    holdout_model_comparison(books)
    match_quality_report(books)


if __name__ == "__main__":
    main()
