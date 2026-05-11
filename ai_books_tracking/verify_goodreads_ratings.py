"""Verify Goodreads ratings using Gemini API with grounded search.

Compares Gemini-verified ratings against the web-scraping pipeline's matches
to find and fix wrong matches (wrong book, summaries, study guides, etc.).

The scraping pipeline picks by title similarity, which systematically selects
obscure/wrong books when titles are generic (e.g. "Shoe Dog" -> children's book
instead of Phil Knight's memoir). This script uses Gemini + Google Search to find
the canonical Goodreads page and flag discrepancies.

Usage:
    python verify_goodreads_ratings.py              # verify only low-count matches
    python verify_goodreads_ratings.py --all        # verify all books
    python verify_goodreads_ratings.py --fix        # verify and fix enriched CSVs
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

OUTPUT_DIR = Path(__file__).parent

ENRICHED_208 = OUTPUT_DIR / "books_enriched_with_goodreads.csv"
NEW_BOOKS_ENRICHED = OUTPUT_DIR / "new_books_to_rate_2026_enriched.csv"
VERIFY_CACHE = OUTPUT_DIR / "goodreads_verify_cache.json"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

LOW_COUNT_THRESHOLD = 100
DISAGREE_THRESHOLD = 0.3


# ---- Helpers (duplicated from multi_source_ratings.py to keep standalone) ----


def _load_cache(path: Path) -> dict[str, object]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, object], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


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


def _clean_title(title: str) -> str:
    """Strip file extensions, libgen artifacts for search."""
    title = re.sub(r"\.(pdf|epub|html|txt)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"\s*-?\s*libgen\.li\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"#\w+#", "", title)
    title = title.replace("_", " ")
    title = re.sub(r"\[.*?\]", "", title)
    title = re.sub(r"\(.*?\)", "", title)
    return title.strip()


def _get_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")


# ---- Gemini verification ----


def fetch_goodreads_via_gemini(
    title: str,
    bookshelf: str,
    cache: dict[str, object],
) -> dict[str, object] | None:
    """Use Gemini Flash with grounded search to find the correct Goodreads rating.

    Deliberately does NOT take the pipeline's author as input, since for
    wrong matches the author is contaminated. Uses bookshelf for disambiguation.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    # Cache by title only (author might be wrong)
    cache_key = title.strip()
    if cache_key in cache:
        return cache[cache_key]  # type: ignore[return-value]

    clean = _clean_title(title)
    if len(clean) < 3:
        cache[cache_key] = None
        return None

    shelf_hint = ""
    if bookshelf and pd.notna(bookshelf):
        shelf_hint = (
            f"\nContext: The reader categorized this as '{bookshelf}'. "
            f"Use this to disambiguate if there are multiple books with this title."
        )

    prompt = (
        f'Find the Goodreads page for the book "{clean}".{shelf_hint}\n'
        f"I need the most popular/canonical edition — NOT summaries, study guides, "
        f"abridged versions, or children's adaptations.\n"
        f"Return ONLY a JSON object with these fields:\n"
        f'- "goodreads_url": the canonical Goodreads URL\n'
        f'- "goodreads_title": the exact title on Goodreads\n'
        f'- "goodreads_author": the primary author on Goodreads\n'
        f'- "goodreads_rating": the current average rating (e.g. 4.23), or null\n'
        f'- "goodreads_rating_count": number of ratings (integer), or null\n'
        f'- "confidence": 0.0 to 1.0 confidence this is the correct book\n'
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
            cache[cache_key] = None
            _save_cache(cache, VERIFY_CACHE)
            return None

        parsed: dict[str, object] = {
            "goodreads_url": obj.get("goodreads_url", ""),
            "goodreads_title": obj.get("goodreads_title", ""),
            "goodreads_author": obj.get("goodreads_author", ""),
            "goodreads_rating": _parse_float(obj.get("goodreads_rating")),
            "goodreads_rating_count": _parse_int(obj.get("goodreads_rating_count")),
            "confidence": _parse_float(obj.get("confidence")),
        }
        cache[cache_key] = parsed
        _save_cache(cache, VERIFY_CACHE)
        return parsed

    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  Gemini error for '{clean[:40]}': {e}")
        return None


# ---- Verification logic ----


def verify_dataset(
    df: pd.DataFrame,
    dataset_label: str,
    cache: dict[str, object],
    only_low_count: bool = True,
) -> list[dict]:
    """Verify GR ratings for a dataset and return discrepancies."""
    discrepancies: list[dict] = []
    verified = 0
    skipped = 0
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        title = str(row["title"])

        pipeline_rating = _parse_float(row.get("goodreads_rating_raw_best"))
        pipeline_count = _parse_int(row.get("goodreads_rating_count_raw_best"))

        if pipeline_rating is None:
            continue

        if (
            only_low_count
            and pipeline_count is not None
            and pipeline_count >= LOW_COUNT_THRESHOLD
        ):
            skipped += 1
            continue

        bookshelf = str(row.get("Bookshelf", "")) if "Bookshelf" in df.columns else ""

        result = fetch_goodreads_via_gemini(title, bookshelf, cache)
        verified += 1

        if result and result.get("goodreads_rating") is not None:
            gemini_rating = float(result["goodreads_rating"])
            gemini_count = result.get("goodreads_rating_count")
            diff = gemini_rating - pipeline_rating

            if abs(diff) > DISAGREE_THRESHOLD:
                entry = {
                    "title": title,
                    "dataset": dataset_label,
                    "pipeline_rating": pipeline_rating,
                    "pipeline_count": pipeline_count,
                    "gemini_rating": gemini_rating,
                    "gemini_count": gemini_count,
                    "gemini_title": result.get("goodreads_title", ""),
                    "gemini_author": result.get("goodreads_author", ""),
                    "gemini_url": result.get("goodreads_url", ""),
                    "confidence": result.get("confidence"),
                    "diff": diff,
                }
                discrepancies.append(entry)
                print(
                    f"  MISMATCH: {title[:40]:<42} "
                    f"pipeline={pipeline_rating:.2f} ({pipeline_count or '?':>6})  "
                    f"gemini={gemini_rating:.2f} ({gemini_count or '?':>6})  "
                    f"diff={diff:+.2f}"
                )

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{n} books")

        time.sleep(0.3)

    print(
        f"  Verified: {verified}, skipped (high count): {skipped}, "
        f"discrepancies: {len(discrepancies)}"
    )
    return discrepancies


def _should_fix(d: dict) -> bool:
    """Decide whether a Gemini result should override the pipeline.

    Fix when Gemini is clearly more reliable:
    - Pipeline count < 100 (likely wrong match) and Gemini count >= 50
    - Gemini count >= 5x pipeline count (much more popular edition found)
    Skip when Gemini matched something obscure (count < pipeline count).
    """
    p_count = d["pipeline_count"] or 0
    g_count = d["gemini_count"] or 0

    if g_count < p_count:
        return False
    if p_count < LOW_COUNT_THRESHOLD and g_count >= 50:
        return True
    if g_count >= 5 * max(p_count, 1):
        return True
    return False


def apply_fixes(csv_path: Path, discrepancies: list[dict]) -> int:
    """Update enriched CSV with Gemini-verified ratings where they disagree."""
    df = pd.read_csv(csv_path)
    fixes = 0

    for d in discrepancies:
        if not _should_fix(d):
            print(f"    SKIP (Gemini not clearly better): {d['title'][:50]}")
            continue

        mask = df["title"] == d["title"]
        if not mask.any():
            continue

        idx = df.index[mask][0]

        # Update rating columns
        for col in ["goodreads_rating_raw_best", "goodreads_rating"]:
            if col in df.columns:
                df.loc[idx, col] = d["gemini_rating"]

        if d["gemini_count"] is not None:
            for col in ["goodreads_rating_count_raw_best", "goodreads_rating_count"]:
                if col in df.columns:
                    df.loc[idx, col] = d["gemini_count"]

        # Update author
        if d["gemini_author"]:
            for col in [
                "canonical_author",
                "goodreads_author",
                "goodreads_author_raw_best",
            ]:
                if col in df.columns:
                    df.loc[idx, col] = d["gemini_author"]

        # Update title
        if d["gemini_title"]:
            for col in ["goodreads_title", "goodreads_title_raw_best"]:
                if col in df.columns:
                    df.loc[idx, col] = d["gemini_title"]

        # Update URL
        if d["gemini_url"]:
            for col in ["goodreads_url", "goodreads_url_raw_best"]:
                if col in df.columns:
                    df.loc[idx, col] = d["gemini_url"]

        # Mark verification source
        if "goodreads_raw_best_source" in df.columns:
            df.loc[idx, "goodreads_raw_best_source"] = "gemini_verified"

        fixes += 1

    df.to_csv(csv_path, index=False)
    return fixes


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify Goodreads ratings via Gemini")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all books, not just low-count matches",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes to enriched CSVs for significant discrepancies",
    )
    args = parser.parse_args()

    if not _get_api_key():
        print("Error: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        return

    cache = _load_cache(VERIFY_CACHE)
    cache_initial = len(cache)
    only_low_count = not args.all

    all_discrepancies: list[dict] = []

    # Training set
    print("=" * 70)
    print("VERIFYING TRAINING SET (208 books)")
    print("=" * 70)
    train = pd.read_csv(ENRICHED_208)
    train.columns = train.columns.str.strip()
    has_rating = train["goodreads_rating_raw_best"].notna().sum()
    low_count = (
        train["goodreads_rating_count_raw_best"].fillna(0) < LOW_COUNT_THRESHOLD
    ).sum()
    mode = "low-count only" if only_low_count else "all"
    print(
        f"  {has_rating} books with pipeline ratings, {low_count} with count < {LOW_COUNT_THRESHOLD}"
    )
    print(f"  Mode: {mode}")
    train_disc = verify_dataset(train, "train", cache, only_low_count)
    all_discrepancies.extend(train_disc)

    # Holdout set
    print(f"\n{'=' * 70}")
    print("VERIFYING HOLDOUT SET (68 books)")
    print("=" * 70)
    holdout = pd.read_csv(NEW_BOOKS_ENRICHED)
    holdout.columns = holdout.columns.str.strip()
    has_rating = holdout["goodreads_rating_raw_best"].notna().sum()
    low_count = (
        holdout["goodreads_rating_count_raw_best"].fillna(0) < LOW_COUNT_THRESHOLD
    ).sum()
    print(
        f"  {has_rating} books with pipeline ratings, {low_count} with count < {LOW_COUNT_THRESHOLD}"
    )
    print(f"  Mode: {mode}")
    holdout_disc = verify_dataset(holdout, "holdout", cache, only_low_count)
    all_discrepancies.extend(holdout_disc)

    # Summary
    print(f"\n{'=' * 70}")
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Cache: {cache_initial} -> {len(cache)} entries")
    print(f"  Discrepancies (|diff| > {DISAGREE_THRESHOLD}): {len(all_discrepancies)}")

    if all_discrepancies:
        all_discrepancies.sort(key=lambda x: -abs(x["diff"]))
        print(
            f"\n  {'Title':<40} {'Set':<6} {'Pipe':>6} {'Cnt':>7} "
            f"{'Gemini':>6} {'Cnt':>7} {'Diff':>6} {'Fix?':<4}  Gemini matched"
        )
        print("  " + "-" * 130)
        for d in all_discrepancies:
            p_cnt = f"{d['pipeline_count']:>,}" if d["pipeline_count"] else "?"
            g_cnt = f"{d['gemini_count']:>,}" if d["gemini_count"] else "?"
            fix = "YES" if _should_fix(d) else "no"
            print(
                f"  {d['title'][:39]:<40} {d['dataset']:<6} "
                f"{d['pipeline_rating']:>6.2f} {p_cnt:>7} "
                f"{d['gemini_rating']:>6.2f} {g_cnt:>7} "
                f"{d['diff']:>+6.2f} {fix:<4}  "
                f"{d['gemini_title'][:30]} by {d['gemini_author'][:20]}"
            )

        avg_diff = np.mean([d["diff"] for d in all_discrepancies])
        avg_abs_diff = np.mean([abs(d["diff"]) for d in all_discrepancies])
        print(f"\n  Mean diff: {avg_diff:+.3f}  Mean |diff|: {avg_abs_diff:.3f}")

    if args.fix and all_discrepancies:
        print(f"\n{'=' * 70}")
        print("APPLYING FIXES")
        print("=" * 70)

        train_fixes = [d for d in all_discrepancies if d["dataset"] == "train"]
        holdout_fixes = [d for d in all_discrepancies if d["dataset"] == "holdout"]

        if train_fixes:
            n = apply_fixes(ENRICHED_208, train_fixes)
            print(f"  Fixed {n} ratings in {ENRICHED_208.name}")

        if holdout_fixes:
            n = apply_fixes(NEW_BOOKS_ENRICHED, holdout_fixes)
            print(f"  Fixed {n} ratings in {NEW_BOOKS_ENRICHED.name}")

        print("\n  Next steps:")
        print("    1. python add_authors_to_csv.py   (fix contaminated authors)")
        print("    2. python goodreads_model_eval.py  (re-evaluate models)")


if __name__ == "__main__":
    main()
