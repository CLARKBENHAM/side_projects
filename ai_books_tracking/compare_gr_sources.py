"""Compare Goodreads ratings from multiple sources against manually verified ground truth."""

import csv
import json
import math
import re
from pathlib import Path
from typing import Optional


# --- File paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_BOOKS_DIR = PROJECT_ROOT / "ai_books_tracking"
DATA_DIR = PROJECT_ROOT / "data"

GT_CSV = DATA_DIR / "Books Read and their effects - master_book_metadata_cleaned.csv"
PIPELINE_TRAINING_CSV = AI_BOOKS_DIR / "books_enriched_with_goodreads.csv"
PIPELINE_HOLDOUT_CSV = AI_BOOKS_DIR / "new_books_to_rate_2026_enriched.csv"
GEMINI_CACHE_JSON = AI_BOOKS_DIR / "goodreads_verify_cache.json"
GEMINI_CLI_CSV = AI_BOOKS_DIR / "verified_holdout.csv"


def clean_title(title: str) -> str:
    """Normalize a title for matching: lowercase, strip whitespace, remove file extensions
    and common suffixes like author names appended with ' - '."""
    t = title.strip().lower()
    # Remove file extensions
    t = re.sub(r"\.(pdf|epub|mobi|txt|html|azw3|doc|docx)$", "", t)
    # Remove trailing parenthesized years like (2006)
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
    # Remove trailing " - Author Name" patterns (common in filename-style titles)
    # But be careful not to strip meaningful subtitles
    # Strip leading/trailing whitespace again
    t = t.strip()
    return t


def further_clean(title: str) -> str:
    """Even more aggressive cleaning for fuzzy matching."""
    t = clean_title(title)
    # Remove anything after " - " that looks like an author/publisher
    t = re.sub(r"\s*-\s*[A-Z][a-z].*$", "", t)
    # Remove underscores
    t = t.replace("_", " ")
    # Remove "by" at end
    t = re.sub(r"\s+by\s*$", "", t)
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t)
    # Remove leading brackets like [Hyperion 1]
    t = re.sub(r"^\[.*?\]\s*", "", t)
    # Remove author names that appear after the title with various separators
    # e.g., "lightning rods helen dewitt by Helen Dewitt"
    t = t.strip()
    return t


def make_key_variants(title: str) -> list[str]:
    """Return a list of progressively cleaned title keys for matching."""
    variants = []
    v1 = clean_title(title)
    variants.append(v1)
    v2 = further_clean(title)
    if v2 != v1:
        variants.append(v2)
    # Also try stripping everything after common separators
    for sep in ["_ ", " - ", "- "]:
        if sep in v1:
            prefix = v1.split(sep)[0].strip()
            if prefix and prefix not in variants:
                variants.append(prefix)
    return variants


def safe_float(val: object) -> Optional[float]:
    """Convert a value to float, returning None if not possible."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("n/a", "nan", "none", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_ground_truth() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(GT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating = safe_float(row["ratings"])
            count = safe_float(row["number ratings"])
            reviews = safe_float(row["number reviews"])
            rows.append(
                {
                    "title": row["title"],
                    "author": row.get("corrected_author", ""),
                    "source": row["source"],
                    "gt_rating": rating,
                    "gt_count": count,
                    "gt_reviews": reviews,
                }
            )
    return rows


def load_pipeline() -> dict[str, dict[str, Optional[float]]]:
    """Load both pipeline CSVs (training + holdout) keyed by cleaned title."""
    result: dict[str, dict[str, Optional[float]]] = {}
    for csv_path in [PIPELINE_TRAINING_CSV, PIPELINE_HOLDOUT_CSV]:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row["title"]
                rating = safe_float(row.get("goodreads_rating_raw_best"))
                count = safe_float(row.get("goodreads_rating_count_raw_best"))
                entry = {"rating": rating, "count": count, "raw_title": title}
                # Store under multiple key variants
                for key in make_key_variants(title):
                    if key not in result:
                        result[key] = entry
    return result


def load_gemini_cache() -> dict[str, dict[str, Optional[float]]]:
    """Load Gemini verification cache keyed by cleaned title."""
    with open(GEMINI_CACHE_JSON, encoding="utf-8") as f:
        cache = json.load(f)
    result: dict[str, dict[str, Optional[float]]] = {}
    for raw_key, val in cache.items():
        if val is None:
            entry: dict[str, Optional[float]] = {
                "rating": None,
                "count": None,
                "raw_title": raw_key,
            }
        else:
            entry = {
                "rating": safe_float(val.get("goodreads_rating")),
                "count": safe_float(val.get("goodreads_rating_count")),
                "raw_title": raw_key,
            }
        for key in make_key_variants(raw_key):
            if key not in result:
                result[key] = entry
    return result


def load_gemini_cli() -> dict[str, dict[str, Optional[float]]]:
    """Load Gemini CLI verified holdout keyed by cleaned title."""
    result: dict[str, dict[str, Optional[float]]] = {}
    with open(GEMINI_CLI_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("original_title", "")
            rating = safe_float(row.get("goodreads_rating"))
            count = safe_float(row.get("goodreads_rating_count"))
            entry = {"rating": rating, "count": count, "raw_title": title}
            for key in make_key_variants(title):
                if key not in result:
                    result[key] = entry
    return result


def lookup(
    source_dict: dict[str, dict[str, Optional[float]]], gt_title: str
) -> Optional[dict[str, Optional[float]]]:
    """Try to find a match for gt_title in source_dict using multiple key variants."""
    for key in make_key_variants(gt_title):
        if key in source_dict:
            return source_dict[key]
    return None


def categorize_error(
    gt_title: str,
    gt_author: str,
    gt_rating: float,
    src_rating: float,
    gt_count: Optional[float],
    src_count: Optional[float],
) -> str:
    """Attempt to categorize the type of mismatch."""
    diff = abs(src_rating - gt_rating)
    if diff <= 0.05:
        return "rating_drift"

    # Check if count is wildly different (suggesting wrong book)
    if gt_count is not None and src_count is not None:
        if gt_count > 0 and src_count > 0:
            count_ratio = max(gt_count, src_count) / min(gt_count, src_count)
            if count_ratio > 50:
                return "wrong_book_entirely"
            if count_ratio > 5:
                return "wrong_edition_or_book"

    if diff <= 0.15:
        return "rating_drift"
    if diff <= 0.3:
        return "moderate_discrepancy"
    return "significant_mismatch"


def main() -> None:
    gt_rows = load_ground_truth()
    pipeline = load_pipeline()
    gemini_cache = load_gemini_cache()
    gemini_cli = load_gemini_cli()

    # Build comparison records
    records: list[dict[str, object]] = []
    for gt in gt_rows:
        title = str(gt["title"])
        gt_rating = gt["gt_rating"]
        gt_count = gt["gt_count"]
        source_cat = str(gt["source"])

        pipe_match = lookup(pipeline, title)
        gem_match = lookup(gemini_cache, title)
        cli_match = lookup(gemini_cli, title)

        pipe_rating = pipe_match["rating"] if pipe_match else None
        pipe_count = pipe_match["count"] if pipe_match else None
        gem_rating = gem_match["rating"] if gem_match else None
        gem_count = gem_match["count"] if gem_match else None
        cli_rating = cli_match["rating"] if cli_match else None
        cli_count = cli_match["count"] if cli_match else None

        pipe_diff: Optional[float] = None
        gem_diff: Optional[float] = None
        cli_diff: Optional[float] = None

        if gt_rating is not None and pipe_rating is not None:
            pipe_diff = pipe_rating - gt_rating
        if gt_rating is not None and gem_rating is not None:
            gem_diff = gem_rating - gt_rating
        if gt_rating is not None and cli_rating is not None:
            cli_diff = cli_rating - gt_rating

        records.append(
            {
                "title": title,
                "author": gt.get("author", ""),
                "source_cat": source_cat,
                "gt_rating": gt_rating,
                "gt_count": gt_count,
                "pipe_rating": pipe_rating,
                "pipe_count": pipe_count,
                "pipe_diff": pipe_diff,
                "gem_rating": gem_rating,
                "gem_count": gem_count,
                "gem_diff": gem_diff,
                "cli_rating": cli_rating,
                "cli_count": cli_count,
                "cli_diff": cli_diff,
            }
        )

    # ===== SECTION 1: Full table =====
    print("=" * 160)
    print("FULL COMPARISON TABLE: Ground Truth vs All Sources")
    print("=" * 160)
    header = (
        f"{'Title':<45} {'Src':^6} {'GT Rtg':>6} {'GT Cnt':>9} "
        f"{'Pipe Rtg':>8} {'P Diff':>7} "
        f"{'Gem Rtg':>8} {'G Diff':>7} "
        f"{'CLI Rtg':>8} {'C Diff':>7}"
    )
    print(header)
    print("-" * 160)

    for r in records:
        title_short = str(r["title"])[:44]
        src = str(r["source_cat"])[:6]
        gt_r = f"{r['gt_rating']:.2f}" if r["gt_rating"] is not None else "  N/A"
        gt_c = f"{r['gt_count']:.0f}" if r["gt_count"] is not None else "    N/A"

        pr = f"{r['pipe_rating']:.2f}" if r["pipe_rating"] is not None else "   N/A"
        pd = f"{r['pipe_diff']:+.2f}" if r["pipe_diff"] is not None else "   N/A"

        gr = f"{r['gem_rating']:.2f}" if r["gem_rating"] is not None else "   N/A"
        gd = f"{r['gem_diff']:+.2f}" if r["gem_diff"] is not None else "   N/A"

        cr = f"{r['cli_rating']:.2f}" if r["cli_rating"] is not None else "   N/A"
        cd = f"{r['cli_diff']:+.2f}" if r["cli_diff"] is not None else "   N/A"

        print(
            f"{title_short:<45} {src:^6} {gt_r:>6} {gt_c:>9} "
            f"{pr:>8} {pd:>7} "
            f"{gr:>8} {gd:>7} "
            f"{cr:>8} {cd:>7}"
        )

    # ===== SECTION 2: Summary statistics =====
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS BY SOURCE")
    print("=" * 100)

    source_names = [
        ("Pipeline (web-scraping)", "pipe_diff", "pipe_rating"),
        ("Gemini Verify (cache)", "gem_diff", "gem_rating"),
        ("Gemini CLI (5 books)", "cli_diff", "cli_rating"),
    ]

    for label, diff_key, rating_key in source_names:
        diffs = [float(r[diff_key]) for r in records if r[diff_key] is not None]
        n_with_gt = sum(1 for r in records if r["gt_rating"] is not None)
        n_with_rating = sum(1 for r in records if r[rating_key] is not None)
        n_matched = len(diffs)

        if n_matched == 0:
            print(f"\n--- {label} ---")
            print("  No matched books with both GT and source ratings.")
            continue

        mean_diff = sum(diffs) / n_matched
        mean_abs = sum(abs(d) for d in diffs) / n_matched
        rmse = math.sqrt(sum(d**2 for d in diffs) / n_matched)
        max_abs = max(abs(d) for d in diffs)
        wrong_count = sum(1 for d in diffs if abs(d) > 0.3)

        print(f"\n--- {label} ---")
        print(f"  Books with GT rating:    {n_with_gt}")
        print(f"  Books with source rating:{n_with_rating}")
        print(f"  Matched (both have val): {n_matched}")
        print(f"  Mean diff (src - GT):    {mean_diff:+.4f}")
        print(f"  Mean |diff|:             {mean_abs:.4f}")
        print(f"  RMSE:                    {rmse:.4f}")
        print(f"  Max |diff|:              {max_abs:.4f}")
        print(
            f"  Wrong matches (|d|>0.3): {wrong_count} ({wrong_count/n_matched*100:.1f}%)"
        )

    # ===== SECTION 2b: Breakdown by source category (Play Export vs Holdout) =====
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS BY SOURCE x DATASET SPLIT")
    print("=" * 100)

    for label, diff_key, rating_key in source_names:
        for split in ["Play Export", "Holdout 2026"]:
            diffs = [
                float(r[diff_key])
                for r in records
                if r[diff_key] is not None and r["source_cat"] == split
            ]
            n_matched = len(diffs)
            if n_matched == 0:
                continue
            mean_diff = sum(diffs) / n_matched
            mean_abs = sum(abs(d) for d in diffs) / n_matched
            rmse = math.sqrt(sum(d**2 for d in diffs) / n_matched)
            max_abs = max(abs(d) for d in diffs)
            wrong_count = sum(1 for d in diffs if abs(d) > 0.3)

            print(f"\n  {label} | {split}")
            print(f"    N matched:       {n_matched}")
            print(f"    Mean diff:       {mean_diff:+.4f}")
            print(f"    Mean |diff|:     {mean_abs:.4f}")
            print(f"    RMSE:            {rmse:.4f}")
            print(f"    Max |diff|:      {max_abs:.4f}")
            print(f"    Wrong (|d|>0.3): {wrong_count}")

    # ===== SECTION 3: Worst mismatches =====
    print("\n" + "=" * 100)
    print("WORST MISMATCHES (|diff| > 0.3) BY SOURCE")
    print("=" * 100)

    for label, diff_key, rating_key in source_names:
        count_key = rating_key.replace("_rating", "_count")
        bad = [
            r
            for r in records
            if r[diff_key] is not None and abs(float(r[diff_key])) > 0.3
        ]
        bad.sort(key=lambda r: abs(float(r[diff_key])), reverse=True)

        print(f"\n--- {label}: {len(bad)} mismatches ---")
        if not bad:
            print("  (none)")
            continue

        for r in bad:
            diff_val = float(r[diff_key])
            src_r = float(r[rating_key]) if r[rating_key] is not None else None
            src_c = r[count_key]
            gt_c = r["gt_count"]

            error_cat = "unknown"
            if r["gt_rating"] is not None and src_r is not None:
                error_cat = categorize_error(
                    str(r["title"]),
                    str(r.get("author", "")),
                    float(r["gt_rating"]),
                    src_r,
                    float(gt_c) if gt_c is not None else None,
                    float(src_c) if src_c is not None else None,
                )

            print(
                f"  {str(r['title'])[:55]:<55} "
                f"GT={r['gt_rating']:.2f}(n={gt_c})  "
                f"Src={src_r:.2f}(n={src_c})  "
                f"Diff={diff_val:+.2f}  "
                f"[{error_cat}]"
            )

    # ===== SECTION 4: Error categorization summary =====
    print("\n" + "=" * 100)
    print("ERROR CATEGORIZATION SUMMARY")
    print("=" * 100)

    for label, diff_key, rating_key in source_names:
        count_key = rating_key.replace("_rating", "_count")
        categories: dict[str, int] = {}
        n_compared = 0
        for r in records:
            if r[diff_key] is None or r["gt_rating"] is None or r[rating_key] is None:
                continue
            n_compared += 1
            cat = categorize_error(
                str(r["title"]),
                str(r.get("author", "")),
                float(r["gt_rating"]),
                float(r[rating_key]),
                float(r["gt_count"]) if r["gt_count"] is not None else None,
                float(r[count_key]) if r[count_key] is not None else None,
            )
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\n--- {label} (N={n_compared}) ---")
        for cat in sorted(categories.keys()):
            cnt = categories[cat]
            print(f"  {cat:<30} {cnt:>4} ({cnt/n_compared*100:5.1f}%)")

    # ===== SECTION 5: Unmatched books =====
    print("\n" + "=" * 100)
    print("UNMATCHED BOOKS (GT has rating but source could not be found)")
    print("=" * 100)

    for label, diff_key, rating_key in source_names:
        unmatched = [
            r for r in records if r["gt_rating"] is not None and r[rating_key] is None
        ]
        print(f"\n--- {label}: {len(unmatched)} unmatched ---")
        for r in unmatched:
            print(f"  [{r['source_cat']}] {str(r['title'])[:70]}")

    # ===== SECTION 6: Ground truth rating distribution =====
    print("\n" + "=" * 100)
    print("GROUND TRUTH RATING DISTRIBUTION")
    print("=" * 100)

    gt_ratings = [float(r["gt_rating"]) for r in records if r["gt_rating"] is not None]
    gt_counts = [float(r["gt_count"]) for r in records if r["gt_count"] is not None]

    if gt_ratings:
        print(f"\n  N books with rating: {len(gt_ratings)}")
        print(f"  Mean rating:         {sum(gt_ratings)/len(gt_ratings):.3f}")
        print(f"  Min rating:          {min(gt_ratings):.2f}")
        print(f"  Max rating:          {max(gt_ratings):.2f}")
        print(f"  Median rating:       {sorted(gt_ratings)[len(gt_ratings)//2]:.2f}")

        # Histogram
        print("\n  Rating histogram:")
        bins = [
            (2.5, 3.0),
            (3.0, 3.25),
            (3.25, 3.5),
            (3.5, 3.75),
            (3.75, 4.0),
            (4.0, 4.25),
            (4.25, 4.5),
            (4.5, 5.0),
        ]
        for lo, hi in bins:
            n = sum(1 for r in gt_ratings if lo <= r < hi)
            bar = "#" * n
            print(f"    [{lo:.2f}-{hi:.2f}): {n:3d} {bar}")

    if gt_counts:
        print(f"\n  N books with count:  {len(gt_counts)}")
        print(f"  Mean count:          {sum(gt_counts)/len(gt_counts):.0f}")
        print(f"  Median count:        {sorted(gt_counts)[len(gt_counts)//2]:.0f}")
        print(f"  Min count:           {min(gt_counts):.0f}")
        print(f"  Max count:           {max(gt_counts):.0f}")

        # Count distribution
        print("\n  Rating count histogram:")
        count_bins = [
            (0, 10),
            (10, 100),
            (100, 1000),
            (1000, 10000),
            (10000, 100000),
            (100000, 1000000),
        ]
        for lo, hi in count_bins:
            n = sum(1 for c in gt_counts if lo <= c < hi)
            bar = "#" * n
            print(f"    [{lo:>7d}-{hi:>7d}): {n:3d} {bar}")

    # ===== SECTION 7: Per-split distribution =====
    print("\n  By split:")
    for split in ["Play Export", "Holdout 2026"]:
        split_ratings = [
            float(r["gt_rating"])
            for r in records
            if r["gt_rating"] is not None and r["source_cat"] == split
        ]
        if split_ratings:
            print(
                f"    {split}: N={len(split_ratings)}, "
                f"mean={sum(split_ratings)/len(split_ratings):.3f}, "
                f"min={min(split_ratings):.2f}, max={max(split_ratings):.2f}"
            )


if __name__ == "__main__":
    main()
