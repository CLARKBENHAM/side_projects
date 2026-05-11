"""Resolve books to Goodreads pages and extract aggregate rating metadata.

This script prefers direct Goodreads lookup before any model-assisted web search:

1. Build a canonical book table from both book-rating CSVs.
2. Query Goodreads' public autocomplete endpoint for direct candidates.
3. Fetch Goodreads book pages and parse JSON-LD rating metadata.
4. Score candidates by title and author agreement.
5. Fall back to Gemini-grounded web search only for low-confidence or missing matches.
6. Save an audit-friendly CSV with the selected match and the top candidates.

Environment:
    GEMINI_API_KEY:
        Optional. Used only for grounded web-search fallback when Goodreads'
        own endpoints do not produce a confident match.

Examples:
    python ai_books_tracking/goodreads_ratings.py
    python ai_books_tracking/goodreads_ratings.py --limit 20 --verbose
    python ai_books_tracking/goodreads_ratings.py --title "Exhalation"
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PLAY_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
RATINGS_2 = DATA_DIR / "Books Read and their effects - Ratings 2.csv"
OUTPUT_DIR = Path(__file__).parent
SEARCH_CACHE_FILE = OUTPUT_DIR / "goodreads_search_cache.json"
PAGE_CACHE_FILE = OUTPUT_DIR / "goodreads_page_cache.json"
MATCH_CACHE_FILE = OUTPUT_DIR / "goodreads_match_cache.json"
GEMINI_CACHE_FILE = OUTPUT_DIR / "goodreads_gemini_cache.json"
OUTPUT_CSV = OUTPUT_DIR / "books_goodreads.csv"

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36"
)

INVALID_AUTHOR_VALUES = {"", "unknown", "by", "nan", "none"}
KNOWN_GOODREADS_DOMAINS = {"goodreads.com", "www.goodreads.com"}
KNOWN_SOURCE_COLUMNS = [
    "title",
    "author",
    "Bookshelf",
    "earliest_modified",
    "latest_modified",
    "filename",
    "Enjoyment (/5)",
    "Usefulness /5 to Me",
]


@dataclass
class GoodreadsCandidate:
    url: str
    query: str
    search_result_title: str
    search_result_snippet: str
    page_title: str
    page_author: str
    rating_value: float | None
    rating_count: int | None
    title_similarity: float
    author_similarity: float
    query_title_similarity: float
    score: float


def load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        with open(cache_path) as handle:
            return json.load(handle)
    return {}


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    with open(cache_path, "w") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: Any, b: Any) -> float:
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0
    seq = difflib.SequenceMatcher(None, left, right).ratio()
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return seq
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    contains_bonus = 0.08 if left in right or right in left else 0.0
    return min(1.0, 0.65 * seq + 0.35 * jaccard + contains_bonus)


def normalize_missing(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def is_valid_author(author: Any) -> bool:
    value = normalize_missing(author).lower()
    if not value or value in INVALID_AUTHOR_VALUES:
        return False
    return not bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def looks_like_generated_title(title: str) -> bool:
    normalized = normalize_text(title)
    if not normalized:
        return True
    if " " not in normalized and re.fullmatch(r"[a-z0-9]{16,}", normalized):
        return True
    return len(normalized.split()) == 1 and len(normalized) >= 24


def strip_file_extension(text: str) -> str:
    return re.sub(r"\.(pdf|epub|html|txt|mobi|azw3)$", "", text, flags=re.IGNORECASE)


def infer_author_from_filename(filename: str) -> str:
    if not filename:
        return ""
    stem = strip_file_extension(Path(filename).name)
    parts = [part.strip() for part in re.split(r"\s+-\s+", stem) if part.strip()]
    candidates = []
    if parts:
        candidates.extend([parts[0], parts[-1]])
    if "-" in stem:
        candidates.append(stem.rsplit("-", maxsplit=1)[-1].strip())

    for candidate in candidates:
        normalized = normalize_text(candidate)
        if len(normalized.split()) < 2 or len(normalized.split()) > 4:
            continue
        if any(token.isdigit() for token in normalized.split()):
            continue
        if {"press", "books", "publishing", "house"} & set(normalized.split()):
            continue
        return candidate
    return ""


def clean_title_for_search(title: Any, author: str = "") -> str:
    text = normalize_missing(title)
    if not text:
        return ""
    text = strip_file_extension(Path(text).name)
    text = text.replace("_", " ")
    text = re.sub(r"\(\d{4}\)", "", text)
    text = re.sub(r"\(.*?edition.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"project gutenberg ebook", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*-?\s*libgen\.li\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+by\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\d\)$", "", text)
    text = re.sub(r"#\w+#", " ", text)

    author_norm = normalize_text(author)
    parts = [
        part.strip(" -") for part in re.split(r"\s+-\s+", text) if part.strip(" -")
    ]
    if len(parts) >= 2 and author_norm and similarity(parts[0], author) >= 0.9:
        text = " - ".join(parts[1:])

    text = re.sub(
        r"\s*-\s*[A-Z][A-Za-z&.' ]{1,40}\s+\(\d{4}\).*$",
        "",
        text,
    )
    text = re.sub(
        r"\s*-\s*[A-Z][A-Za-z&.' ]{1,40}$",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip(" -:")
    return text


def choose_best_value(candidates: Iterable[tuple[str, Any]]) -> str:
    best_value = ""
    best_score = -1.0
    for source_name, raw_value in candidates:
        value = normalize_missing(raw_value)
        if not value:
            continue
        source_score = {
            "author_ratings2": 1.0,
            "author_play": 0.8,
            "filename_author_ratings2": 0.6,
            "filename_author_play": 0.5,
            "title_ratings2": 1.0,
            "title_play": 0.9,
            "filename_ratings2": 0.7,
            "filename_play": 0.6,
            "Bookshelf_ratings2": 0.8,
            "Bookshelf_play": 0.7,
        }.get(source_name, 0.0)
        score = source_score + min(len(normalize_text(value).split()), 6) * 0.02
        if score > best_score:
            best_score = score
            best_value = value
    return best_value


def extract_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in KNOWN_SOURCE_COLUMNS if column in df.columns]
    return df[available].copy()


def load_source_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = extract_source_columns(df)
    df = df.dropna(how="all")
    if "title" in df.columns:
        df = df[df["title"].notna()]
    return df


def build_canonical_books(
    play_export: Path = PLAY_EXPORT, ratings_2: Path = RATINGS_2
) -> pd.DataFrame:
    play_df = load_source_frame(play_export)
    ratings_df = load_source_frame(ratings_2)

    merged = play_df.merge(
        ratings_df, on="title", how="outer", suffixes=("_play", "_ratings2")
    )
    rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        raw_title = choose_best_value(
            [
                ("title_ratings2", row.get("title")),
                ("title_play", row.get("title")),
                ("filename_ratings2", row.get("filename_ratings2")),
                ("filename_play", row.get("filename_play")),
            ]
        )
        if not raw_title:
            continue

        author = choose_best_value(
            [
                (
                    "author_ratings2",
                    (
                        row.get("author_ratings2")
                        if is_valid_author(row.get("author_ratings2"))
                        else ""
                    ),
                ),
                (
                    "author_play",
                    (
                        row.get("author_play")
                        if is_valid_author(row.get("author_play"))
                        else ""
                    ),
                ),
                (
                    "filename_author_ratings2",
                    infer_author_from_filename(
                        normalize_missing(row.get("filename_ratings2"))
                    ),
                ),
                (
                    "filename_author_play",
                    infer_author_from_filename(
                        normalize_missing(row.get("filename_play"))
                    ),
                ),
            ]
        )

        search_candidates = [
            ("title_ratings2", clean_title_for_search(row.get("title"), author)),
            ("title_play", clean_title_for_search(row.get("title"), author)),
            (
                "filename_ratings2",
                clean_title_for_search(row.get("filename_ratings2"), author),
            ),
            ("filename_play", clean_title_for_search(row.get("filename_play"), author)),
        ]
        search_title = choose_best_value(
            [
                (source_name, candidate)
                for source_name, candidate in search_candidates
                if candidate and not looks_like_generated_title(candidate)
            ]
        )
        if not search_title:
            search_title = clean_title_for_search(raw_title, author)

        bookshelf = choose_best_value(
            [
                ("Bookshelf_ratings2", row.get("Bookshelf_ratings2")),
                ("Bookshelf_play", row.get("Bookshelf_play")),
            ]
        )

        canonical_key = normalize_text(f"{search_title}::{author or raw_title}")
        rows.append(
            {
                "canonical_key": canonical_key,
                "title": raw_title,
                "author": author,
                "Bookshelf": bookshelf,
                "search_title": search_title,
                "search_author": author,
                "title_play": normalize_missing(row.get("title")),
                "title_ratings2": normalize_missing(row.get("title")),
                "author_play": normalize_missing(row.get("author_play")),
                "author_ratings2": normalize_missing(row.get("author_ratings2")),
                "filename_play": normalize_missing(row.get("filename_play")),
                "filename_ratings2": normalize_missing(row.get("filename_ratings2")),
                "earliest_modified_play": normalize_missing(
                    row.get("earliest_modified_play")
                ),
                "latest_modified_play": normalize_missing(
                    row.get("latest_modified_play")
                ),
                "earliest_modified_ratings2": normalize_missing(
                    row.get("earliest_modified_ratings2")
                ),
                "latest_modified_ratings2": normalize_missing(
                    row.get("latest_modified_ratings2")
                ),
                "enjoyment_play": row.get("Enjoyment (/5)_play"),
                "usefulness_play": row.get("Usefulness /5 to Me_play"),
                "enjoyment_ratings2": row.get("Enjoyment (/5)_ratings2"),
                "usefulness_ratings2": row.get("Usefulness /5 to Me_ratings2"),
            }
        )

    books = pd.DataFrame(rows)
    books = books.drop_duplicates(subset=["canonical_key"], keep="first").reset_index(
        drop=True
    )
    return books


def get_gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "")


def build_title_query_variants(book: pd.Series) -> list[str]:
    raw_candidates = [
        normalize_missing(book.get("search_title")),
        clean_title_for_search(
            book.get("title"), normalize_missing(book.get("author"))
        ),
        clean_title_for_search(
            book.get("filename_ratings2"), normalize_missing(book.get("author"))
        ),
        clean_title_for_search(
            book.get("filename_play"), normalize_missing(book.get("author"))
        ),
    ]
    variants: list[str] = []
    for raw in raw_candidates:
        if not raw or looks_like_generated_title(raw):
            continue
        splits = [raw]
        for separator in [" - ", ": "]:
            if separator in raw:
                head = raw.split(separator, maxsplit=1)[0].strip()
                if len(normalize_text(head).split()) >= 2:
                    splits.append(head)
        if "-" in raw:
            head, tail = raw.rsplit("-", maxsplit=1)
            tail_normalized = normalize_text(tail)
            if 2 <= len(tail_normalized.split()) <= 4 and not any(
                token.isdigit() for token in tail_normalized.split()
            ):
                splits.append(head.strip())
        for candidate in splits:
            cleaned = re.sub(r"\s+", " ", candidate).strip(" -:")
            if cleaned and cleaned not in variants:
                variants.append(cleaned)
        if len(variants) >= 4:
            break
    return variants


def build_goodreads_lookup_queries(book: pd.Series) -> list[str]:
    author = normalize_missing(book["search_author"])
    queries: list[str] = []
    for title in build_title_query_variants(book):
        if author:
            queries.append(f"{title} {author}")
        queries.append(title)
    return list(dict.fromkeys(query for query in queries if query.strip()))


def run_goodreads_autocomplete(
    query: str,
    session: requests.Session,
    search_cache: dict[str, Any],
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    if query in search_cache:
        return search_cache[query]

    params = {"format": "json", "q": query}
    response = session.get(
        "https://www.goodreads.com/book/auto_complete",
        params=params,
        timeout=30,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    response.raise_for_status()
    items = response.json()
    search_cache[query] = items
    time.sleep(sleep_seconds)
    return items


def canonical_goodreads_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in KNOWN_GOODREADS_DOMAINS:
        return ""
    path = parsed.path
    path = re.sub(r"^/en/", "/", path)
    path = path.rstrip("/")
    return urlunparse(("https", "www.goodreads.com", path, "", "", ""))


def is_goodreads_book_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in KNOWN_GOODREADS_DOMAINS:
        return False
    return "/book/show/" in parsed.path or "/book/title" in parsed.path


def iter_json_nodes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from iter_json_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_nodes(item)


def parse_author_field(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return normalize_missing(value.get("name"))
    if isinstance(value, list):
        names = [parse_author_field(item) for item in value]
        return ", ".join(name for name in names if name)
    return ""


def to_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except ValueError:
        return None


def extract_json_ld_blocks(html_text: str) -> list[str]:
    blocks = re.findall(
        r"<script[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [unescape(block).strip() for block in blocks if block.strip()]


def extract_goodreads_page_data(html_text: str, url: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_score = -1

    for block in extract_json_ld_blocks(html_text):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in iter_json_nodes(parsed):
            node_type = node.get("@type", [])
            if isinstance(node_type, str):
                node_types = {node_type}
            else:
                node_types = {str(item) for item in node_type}
            aggregate = node.get("aggregateRating") or {}
            if "Book" not in node_types and not aggregate:
                continue
            candidate = {
                "url": url,
                "page_title": normalize_missing(
                    node.get("name") or node.get("headline")
                ),
                "page_author": parse_author_field(node.get("author")),
                "rating_value": to_float(aggregate.get("ratingValue")),
                "rating_count": to_int(
                    aggregate.get("ratingCount") or aggregate.get("reviewCount")
                ),
            }
            score = 0
            if "Book" in node_types:
                score += 2
            if candidate["page_title"]:
                score += 1
            if candidate["rating_value"] is not None:
                score += 2
            if candidate["rating_count"] is not None:
                score += 1
            if score > best_score:
                best_score = score
                best = candidate

    if best:
        return best

    title_match = re.search(
        r"<title>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL
    )
    rating_value = re.search(r'"ratingValue"\s*:\s*"?(?P<value>[0-9.]+)"?', html_text)
    rating_count = re.search(r'"ratingCount"\s*:\s*"?(?P<count>[0-9,]+)"?', html_text)
    author_match = re.search(
        r'"author"\s*:\s*\{[^{}]*"name"\s*:\s*"(?P<author>[^"]+)"',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {
        "url": url,
        "page_title": (
            unescape(title_match.group(1)).replace("| Goodreads", "").strip()
            if title_match
            else ""
        ),
        "page_author": author_match.group("author").strip() if author_match else "",
        "rating_value": to_float(rating_value.group("value")) if rating_value else None,
        "rating_count": to_int(rating_count.group("count")) if rating_count else None,
    }


def fetch_goodreads_page(
    url: str,
    session: requests.Session,
    page_cache: dict[str, Any],
    sleep_seconds: float,
) -> dict[str, Any]:
    canonical_url = canonical_goodreads_url(url)
    if not canonical_url:
        return {}
    if canonical_url in page_cache:
        return page_cache[canonical_url]

    response = session.get(
        canonical_url,
        timeout=30,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        response.raise_for_status()
        parsed = extract_goodreads_page_data(response.text, canonical_url)
    except requests.RequestException:
        parsed = {"url": canonical_url, "page_title": "", "page_author": ""}
    page_cache[canonical_url] = parsed
    time.sleep(sleep_seconds)
    return parsed


def autocomplete_item_to_scored_parts(
    item: dict[str, Any],
    page_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    author_data = item.get("author") or {}
    book_url = normalize_missing(item.get("bookUrl"))
    full_url = f"https://www.goodreads.com{book_url}" if book_url else ""
    raw_title = (
        normalize_missing(item.get("bookTitleBare"))
        or normalize_missing(item.get("title"))
        or normalize_missing(page_data.get("page_title"))
    )
    raw_author = normalize_missing(author_data.get("name")) or normalize_missing(
        page_data.get("page_author")
    )
    search_like_item = {
        "title": raw_title,
        "snippet": raw_author,
        "link": full_url,
    }
    page_like_data = {
        "page_title": normalize_missing(page_data.get("page_title")) or raw_title,
        "page_author": normalize_missing(page_data.get("page_author")) or raw_author,
        "rating_value": (
            page_data.get("rating_value")
            if page_data.get("rating_value") is not None
            else to_float(item.get("avgRating"))
        ),
        "rating_count": (
            page_data.get("rating_count")
            if page_data.get("rating_count") is not None
            else to_int(item.get("ratingsCount"))
        ),
    }
    return search_like_item, page_like_data


def score_candidate(
    book: pd.Series, query: str, item: dict[str, Any], page_data: dict[str, Any]
) -> GoodreadsCandidate:
    search_result_title = normalize_missing(item.get("title"))
    snippet = normalize_missing(item.get("snippet"))
    page_title = normalize_missing(page_data.get("page_title"))
    page_author = normalize_missing(page_data.get("page_author"))
    rating_value = to_float(page_data.get("rating_value"))
    rating_count = to_int(page_data.get("rating_count"))

    title_similarity = similarity(
        book["search_title"], page_title or search_result_title
    )
    author_similarity = (
        similarity(book["search_author"], page_author)
        if normalize_missing(book["search_author"])
        else 0.65
    )
    query_title_similarity = similarity(book["search_title"], search_result_title)

    score = (
        0.65 * title_similarity
        + 0.20 * author_similarity
        + 0.10 * query_title_similarity
    )
    if rating_value is not None:
        score += 0.03
    if rating_count:
        score += min(math.log10(rating_count + 1) / 10, 0.05)

    return GoodreadsCandidate(
        url=canonical_goodreads_url(item.get("link", "")),
        query=query,
        search_result_title=search_result_title,
        search_result_snippet=snippet,
        page_title=page_title,
        page_author=page_author,
        rating_value=rating_value,
        rating_count=rating_count,
        title_similarity=round(title_similarity, 4),
        author_similarity=round(author_similarity, 4),
        query_title_similarity=round(query_title_similarity, 4),
        score=round(score, 4),
    )


def parse_model_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def run_gemini_grounded_match(
    book: pd.Series,
    gemini_cache: dict[str, Any],
    session: requests.Session,
    sleep_seconds: float,
) -> dict[str, Any] | None:
    api_key = get_gemini_api_key()
    if not api_key:
        return None

    cache_key = book["canonical_key"]
    if cache_key in gemini_cache:
        return gemini_cache[cache_key]

    prompt = f"""Find the Goodreads page for this exact book using web search.

Return ONLY JSON with these keys:
{{
  "goodreads_url": string|null,
  "goodreads_title": string|null,
  "goodreads_author": string|null,
  "goodreads_rating": number|null,
  "goodreads_rating_count": integer|null,
  "confidence": "high"|"medium"|"low",
  "reason": string
}}

Book metadata:
{json.dumps(
    {
        "title": book["title"],
        "search_title": book["search_title"],
        "author": book["author"],
        "bookshelf": book["Bookshelf"],
        "filename_play": book["filename_play"],
        "filename_ratings2": book["filename_ratings2"],
    },
    ensure_ascii=False,
    indent=2,
)}

Only return a Goodreads URL if the match is for the same work, not just a similarly named book."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    response = session.post(
        f"{GEMINI_GENERATE_URL}?key={api_key}",
        json=payload,
        timeout=60,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    data = response.json()
    candidate = ((data.get("candidates") or [{}])[0]) if data else {}
    parts = ((candidate.get("content") or {}).get("parts")) or [{}]
    parsed = parse_model_json(normalize_missing(parts[0].get("text")))
    if not parsed:
        return None

    grounding = candidate.get("groundingMetadata") or {}
    parsed["goodreads_url"] = canonical_goodreads_url(
        normalize_missing(parsed.get("goodreads_url"))
    )
    parsed["grounding_queries"] = grounding.get("webSearchQueries", []) or []
    parsed["grounding_sources"] = grounding.get("groundingChunks", []) or []
    gemini_cache[cache_key] = parsed
    time.sleep(sleep_seconds)
    return parsed


def resolve_goodreads_match(
    book: pd.Series,
    session: requests.Session,
    search_cache: dict[str, Any],
    page_cache: dict[str, Any],
    gemini_cache: dict[str, Any],
    sleep_seconds: float,
    verbose: bool = False,
) -> dict[str, Any]:
    candidates_by_url: dict[str, GoodreadsCandidate] = {}

    for query in build_goodreads_lookup_queries(book):
        items = run_goodreads_autocomplete(
            query=query,
            session=session,
            search_cache=search_cache,
            sleep_seconds=sleep_seconds,
        )
        for item in items:
            book_url = normalize_missing(item.get("bookUrl"))
            full_url = f"https://www.goodreads.com{book_url}" if book_url else ""
            if not is_goodreads_book_url(full_url):
                continue
            page_data = fetch_goodreads_page(
                full_url,
                session=session,
                page_cache=page_cache,
                sleep_seconds=sleep_seconds,
            )
            search_like_item, page_like_data = autocomplete_item_to_scored_parts(
                item, page_data
            )
            candidate = score_candidate(book, query, search_like_item, page_like_data)
            if not candidate.url:
                continue
            previous = candidates_by_url.get(candidate.url)
            if previous is None or candidate.score > previous.score:
                candidates_by_url[candidate.url] = candidate

    ranked = sorted(
        candidates_by_url.values(), key=lambda item: item.score, reverse=True
    )
    chosen = ranked[0] if ranked else None
    match_method = "goodreads_autocomplete"

    need_fallback = not ranked or chosen.score < 0.78
    if need_fallback:
        grounded = run_gemini_grounded_match(
            book=book,
            gemini_cache=gemini_cache,
            session=session,
            sleep_seconds=sleep_seconds,
        )
        grounded_url = (
            canonical_goodreads_url(normalize_missing(grounded.get("goodreads_url")))
            if grounded
            else ""
        )
        if grounded_url:
            page_data = fetch_goodreads_page(
                grounded_url,
                session=session,
                page_cache=page_cache,
                sleep_seconds=sleep_seconds,
            )
            grounded_item = {
                "title": normalize_missing(grounded.get("goodreads_title")),
                "snippet": normalize_missing(grounded.get("goodreads_author")),
                "link": grounded_url,
            }
            grounded_page = {
                "page_title": normalize_missing(page_data.get("page_title"))
                or normalize_missing(grounded.get("goodreads_title")),
                "page_author": normalize_missing(page_data.get("page_author"))
                or normalize_missing(grounded.get("goodreads_author")),
                "rating_value": (
                    page_data.get("rating_value")
                    if page_data.get("rating_value") is not None
                    else to_float(grounded.get("goodreads_rating"))
                ),
                "rating_count": (
                    page_data.get("rating_count")
                    if page_data.get("rating_count") is not None
                    else to_int(grounded.get("goodreads_rating_count"))
                ),
            }
            grounded_candidate = score_candidate(
                book=book,
                query="gemini_grounded_search",
                item=grounded_item,
                page_data=grounded_page,
            )
            if chosen is None or grounded_candidate.score >= chosen.score:
                chosen = grounded_candidate
                match_method = "gemini_grounded_search"
                ranked = [grounded_candidate] + ranked

    if chosen is None:
        return {
            "goodreads_status": "unmatched",
            "goodreads_match_method": "none",
            "goodreads_candidates_json": "[]",
        }

    status = "matched"
    if chosen.score < 0.70:
        status = "review"

    if verbose:
        print(
            f"{book['search_title'][:45]:<45} -> {chosen.page_title[:45]:<45} "
            f"score={chosen.score:.3f} method={match_method}"
        )

    return {
        "goodreads_status": status,
        "goodreads_match_method": match_method,
        "goodreads_url": chosen.url,
        "goodreads_title": chosen.page_title,
        "goodreads_author": chosen.page_author,
        "goodreads_rating": chosen.rating_value,
        "goodreads_rating_count": chosen.rating_count,
        "goodreads_match_score": chosen.score,
        "goodreads_title_similarity": chosen.title_similarity,
        "goodreads_author_similarity": chosen.author_similarity,
        "goodreads_query": chosen.query,
        "goodreads_candidates_json": json.dumps(
            [asdict(candidate) for candidate in ranked[:5]], ensure_ascii=False
        ),
    }


def enrich_books_with_goodreads(
    books: pd.DataFrame,
    output_path: Path = OUTPUT_CSV,
    limit: int | None = None,
    title_filter: str | None = None,
    force_refresh: bool = False,
    sleep_seconds: float = 0.2,
    verbose: bool = False,
) -> pd.DataFrame:
    books = books.copy()
    if title_filter:
        title_filter_norm = normalize_text(title_filter)
        books = books[
            books["search_title"]
            .map(normalize_text)
            .str.contains(title_filter_norm, regex=False)
        ]
    if limit is not None:
        books = books.head(limit)

    search_cache = load_cache(SEARCH_CACHE_FILE)
    page_cache = load_cache(PAGE_CACHE_FILE)
    match_cache = load_cache(MATCH_CACHE_FILE)
    gemini_cache = load_cache(GEMINI_CACHE_FILE)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: list[dict[str, Any]] = []
    for index, (_, book) in enumerate(books.iterrows(), start=1):
        cache_key = book["canonical_key"]
        if cache_key in match_cache and not force_refresh:
            match = match_cache[cache_key]
        else:
            match = resolve_goodreads_match(
                book=book,
                session=session,
                search_cache=search_cache,
                page_cache=page_cache,
                gemini_cache=gemini_cache,
                sleep_seconds=sleep_seconds,
                verbose=verbose,
            )
            match_cache[cache_key] = match

        if verbose and index % 20 == 0:
            print(f"Processed {index}/{len(books)} books")

        merged = {**book.to_dict(), **match}
        results.append(merged)

        if index % 10 == 0:
            save_cache(search_cache, SEARCH_CACHE_FILE)
            save_cache(page_cache, PAGE_CACHE_FILE)
            save_cache(match_cache, MATCH_CACHE_FILE)
            save_cache(gemini_cache, GEMINI_CACHE_FILE)

    save_cache(search_cache, SEARCH_CACHE_FILE)
    save_cache(page_cache, PAGE_CACHE_FILE)
    save_cache(match_cache, MATCH_CACHE_FILE)
    save_cache(gemini_cache, GEMINI_CACHE_FILE)

    output = pd.DataFrame(results)
    output.to_csv(output_path, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play-export", type=Path, default=PLAY_EXPORT)
    parser.add_argument("--ratings-2", type=Path, default=RATINGS_2)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--title", type=str)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    books = build_canonical_books(args.play_export, args.ratings_2)
    enriched = enrich_books_with_goodreads(
        books=books,
        output_path=args.output,
        limit=args.limit,
        title_filter=args.title,
        force_refresh=args.force_refresh,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose,
    )

    status_counts = enriched["goodreads_status"].value_counts(dropna=False).to_dict()
    print(f"Saved Goodreads enrichment to {args.output}")
    print(f"Status counts: {status_counts}")
    if "review" in status_counts:
        review_cols = [
            "search_title",
            "author",
            "goodreads_title",
            "goodreads_author",
            "goodreads_match_score",
            "goodreads_url",
        ]
        print("\nManual-review rows:")
        print(
            enriched[enriched["goodreads_status"] == "review"][review_cols]
            .head(15)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
