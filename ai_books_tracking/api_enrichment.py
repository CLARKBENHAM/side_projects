"""Fetch external metadata for books using Open Library API (free, no auth, generous limits)
and Gemini for recommendation source inference."""

import json
import os
import time
import re
import pandas as pd
import numpy as np
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PLAY_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
OUTPUT_DIR = Path(__file__).parent
CACHE_FILE = OUTPUT_DIR / "openlibrary_cache.json"
GEMINI_CACHE_FILE = OUTPUT_DIR / "gemini_recommendations_cache.json"

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PLAY_EXPORT)
    df.columns = df.columns.str.strip()
    return df


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_path: Path) -> None:
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def clean_title_for_search(title: str) -> str:
    """Strip file extensions and clean up title for API search."""
    if not isinstance(title, str):
        return ""
    title = re.sub(r"\.(pdf|epub|html|txt)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\(\d{4}\)", "", title)
    # Remove publisher info patterns
    title = re.sub(r"\s*-\s*[A-Z][a-z]+\s+(Press|Books|Publishing|House).*$", "", title)
    title = re.sub(r"\s+by\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"#\w+#", "", title)
    title = title.replace("_", " ")
    # Remove "libgen.li" and similar
    title = re.sub(r"\s*-?\s*libgen\.li\s*", "", title, flags=re.IGNORECASE)
    # Remove "(1)" style suffixes from filenames
    title = re.sub(r"\(\d\)$", "", title)
    return title.strip()


def fetch_openlibrary(title: str, author: str, cache: dict) -> dict | None:
    """Fetch book metadata from Open Library Search API."""
    cache_key = f"{title}|||{author}"
    if cache_key in cache:
        return cache[cache_key]

    clean = clean_title_for_search(title)
    if len(clean) < 3:
        cache[cache_key] = None
        return None

    author_is_valid = (
        author
        and str(author) != "by"
        and str(author) != "Unknown"
        and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(author))
    )

    params: dict = {
        "q": clean,
        "limit": 1,
        "fields": "title,author_name,first_publish_year,number_of_pages_median,ratings_average,ratings_count,subject,cover_i",
    }
    if author_is_valid:
        # Use combined title + author search
        params["q"] = f"{clean} {author}"

    try:
        resp = requests.get(OPEN_LIBRARY_SEARCH, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("numFound", 0) == 0 and author_is_valid:
            # Retry without author
            time.sleep(0.2)
            params["q"] = clean
            resp = requests.get(OPEN_LIBRARY_SEARCH, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

        if data.get("numFound", 0) == 0:
            cache[cache_key] = None
            return None

        doc = data["docs"][0]
        result = {
            "ol_title": doc.get("title", ""),
            "ol_authors": doc.get("author_name", []),
            "ol_first_publish_year": doc.get("first_publish_year"),
            "ol_page_count": doc.get("number_of_pages_median"),
            "ol_rating": doc.get("ratings_average"),
            "ol_ratings_count": doc.get("ratings_count"),
            "ol_subjects": (doc.get("subject", []) or [])[:10],
        }
        cache[cache_key] = result
        return result

    except requests.RequestException as e:
        print(f"  API error for '{clean[:40]}': {e}")
        return None


def enrich_with_openlibrary(df: pd.DataFrame) -> pd.DataFrame:
    """Add Open Library metadata to dataframe."""
    cache = load_cache(CACHE_FILE)
    initial_cache_size = len(cache)

    print("Fetching Open Library metadata...")
    results = []
    for i, row in df.iterrows():
        result = fetch_openlibrary(row["title"], row.get("author", ""), cache)
        results.append(result or {})
        time.sleep(0.15)  # ~6 req/s, well within OL limits
        if (i + 1) % 40 == 0:
            print(f"  Processed {i + 1}/{len(df)} books")
            save_cache(cache, CACHE_FILE)

    save_cache(cache, CACHE_FILE)
    n_found = sum(1 for r in results if r)
    print(
        f"  Cache: {initial_cache_size} -> {len(cache)} entries ({n_found}/{len(df)} found)"
    )

    gb_df = pd.DataFrame(results)
    for col in [
        "ol_page_count",
        "ol_rating",
        "ol_ratings_count",
        "ol_first_publish_year",
        "ol_title",
    ]:
        if col not in gb_df.columns:
            gb_df[col] = np.nan

    df = pd.concat([df.reset_index(drop=True), gb_df.reset_index(drop=True)], axis=1)

    # Rename for consistency with prediction model
    df["gb_page_count"] = df["ol_page_count"]
    df["gb_average_rating"] = df["ol_rating"]
    df["gb_ratings_count"] = df["ol_ratings_count"]
    df["pub_year"] = pd.to_numeric(df["ol_first_publish_year"], errors="coerce")

    return df


def analyze_api_features(df: pd.DataFrame) -> None:
    """Analyze correlations between external metadata and personal ratings."""
    from scipy import stats

    print("\n" + "=" * 70)
    print("OPEN LIBRARY API RESULTS")
    print("=" * 70)

    has_rating = df["gb_average_rating"].notna()
    has_pages = df["gb_page_count"].notna()
    has_pub = df["pub_year"].notna()

    print(
        f"\nCoverage: {has_rating.sum()} with OL rating, "
        f"{has_pages.sum()} with page count, {has_pub.sum()} with pub year"
    )

    if has_rating.sum() > 10:
        valid = df[has_rating]
        r_enjoy, p_enjoy = stats.spearmanr(
            valid["gb_average_rating"], valid["Enjoyment (/5)"]
        )
        r_useful, p_useful = stats.spearmanr(
            valid["gb_average_rating"], valid["Usefulness /5 to Me"]
        )
        print(f"\nOpen Library avg rating vs personal rating (n={has_rating.sum()}):")
        print(f"  vs Enjoyment:  rho={r_enjoy:.3f}, p={p_enjoy:.4f}")
        print(f"  vs Usefulness: rho={r_useful:.3f}, p={p_useful:.4f}")
        print(
            f"  OL rating range: [{valid['gb_average_rating'].min():.2f}, "
            f"{valid['gb_average_rating'].max():.2f}], "
            f"mean={valid['gb_average_rating'].mean():.2f}"
        )

    has_count = df["gb_ratings_count"].notna()
    if has_count.sum() > 10:
        valid = df[has_count]
        r_enjoy, p_enjoy = stats.spearmanr(
            valid["gb_ratings_count"], valid["Enjoyment (/5)"]
        )
        print(f"\nOL ratings count (popularity) vs Enjoyment (n={has_count.sum()}):")
        print(f"  rho={r_enjoy:.3f}, p={p_enjoy:.4f}")

    if has_pages.sum() > 10:
        valid = df[has_pages]
        r_enjoy, p_enjoy = stats.spearmanr(
            valid["gb_page_count"], valid["Enjoyment (/5)"]
        )
        r_useful, p_useful = stats.spearmanr(
            valid["gb_page_count"], valid["Usefulness /5 to Me"]
        )
        print(f"\nPage count vs personal rating (n={has_pages.sum()}):")
        print(f"  vs Enjoyment:  rho={r_enjoy:.3f}, p={p_enjoy:.4f}")
        print(f"  vs Usefulness: rho={r_useful:.3f}, p={p_useful:.4f}")
        print(
            f"  Page count: median={valid['gb_page_count'].median():.0f}, "
            f"mean={valid['gb_page_count'].mean():.0f}"
        )

    if has_pub.sum() > 10:
        valid = df[has_pub]
        r_enjoy, p_enjoy = stats.spearmanr(valid["pub_year"], valid["Enjoyment (/5)"])
        r_useful, p_useful = stats.spearmanr(
            valid["pub_year"], valid["Usefulness /5 to Me"]
        )
        print(f"\nPublication year vs personal rating (n={has_pub.sum()}):")
        print(f"  vs Enjoyment:  rho={r_enjoy:.3f}, p={p_enjoy:.4f}")
        print(f"  vs Usefulness: rho={r_useful:.3f}, p={p_useful:.4f}")

        median_year = valid["pub_year"].median()
        old = valid[valid["pub_year"] < median_year]
        new = valid[valid["pub_year"] >= median_year]
        print(
            f"\n  Books published before {median_year:.0f} (n={len(old)}): "
            f"enjoy={old['Enjoyment (/5)'].mean():.2f}"
        )
        print(
            f"  Books published {median_year:.0f}+ (n={len(new)}): "
            f"enjoy={new['Enjoyment (/5)'].mean():.2f}"
        )

    if has_pages.sum() > 10:
        print("\nPage count by category:")
        for cat in df["Bookshelf"].unique():
            cat_df = df[(df["Bookshelf"] == cat) & has_pages]
            if len(cat_df) >= 3:
                print(
                    f"  {cat:<25}: median={cat_df['gb_page_count'].median():.0f} pages "
                    f"(n={len(cat_df)})"
                )


def infer_recommendation_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Use Gemini to infer likely recommendation sources for books."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\nNo GEMINI_API_KEY found, skipping recommendation source inference")
        return df

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    cache = load_cache(GEMINI_CACHE_FILE)

    known_sources = [
        "Tyler Cowen / Marginal Revolution",
        "Tanner Greer / Scholar's Stage",
        "The Last Psychiatrist / Alone",
        "Scott Alexander / SSC / ACX / LessWrong / Rationalist community",
        "Gwern",
        "Twitter / X general",
        "College / school assignment",
        "Self-discovered / browsing",
        "Classic canon / Great Books list",
        "Professional / work-related need",
        "Friend recommendation",
        "Author's other work (already read another by same author)",
        "Unknown / can't determine",
    ]

    books_to_query = []
    for _, row in df.iterrows():
        title = row["title"]
        if title not in cache:
            books_to_query.append(row)

    print(f"\n{len(cache)} books already cached, {len(books_to_query)} to query Gemini")

    batch_size = 30
    for batch_start in range(0, len(books_to_query), batch_size):
        batch = books_to_query[batch_start : batch_start + batch_size]

        book_list = "\n".join(
            f"- \"{row['title']}\" by {row.get('author', 'unknown')} "
            f"(category: {row.get('Bookshelf', 'unknown')}, "
            f"notes: {str(row.get('Long Term Effects', ''))[:200]})"
            for _, row in pd.DataFrame(batch).iterrows()
        )

        prompt = f"""For each book below, infer the most likely recommendation source from this list:
{chr(10).join(f'  {s}' for s in known_sources)}

The reader is a software engineer in the rationalist / EA / tech community who reads
Tyler Cowen, Tanner Greer, Scott Alexander, Gwern, and tech Twitter. They read CS/ML
textbooks for work, literature for self-improvement (influenced by Cowen/Greer), and
fiction mostly from rationalist community recommendations.

Books:
{book_list}

Respond with ONLY a JSON object mapping each book title (use the EXACT title given) to
the most likely source. Use the exact source names from the list above. Example:
{{"Book Title Here": "Tyler Cowen / Marginal Revolution"}}"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            cache.update(result)
            save_cache(cache, GEMINI_CACHE_FILE)
            print(f"  Processed batch {batch_start // batch_size + 1}")
            time.sleep(2)  # rate limiting for free tier
        except Exception as e:
            print(f"  Gemini error on batch {batch_start // batch_size + 1}: {e}")

    df["inferred_source"] = df["title"].map(cache)

    print("\n" + "=" * 70)
    print("INFERRED RECOMMENDATION SOURCES")
    print("=" * 70)

    matched = df["inferred_source"].notna().sum()
    print(f"\nMatched {matched}/{len(df)} books to recommendation sources")

    source_stats = (
        df[df["inferred_source"].notna()]
        .groupby("inferred_source")
        .agg(
            n=("Enjoyment (/5)", "count"),
            enjoy_mean=("Enjoyment (/5)", "mean"),
            useful_mean=("Usefulness /5 to Me", "mean"),
        )
        .sort_values("enjoy_mean", ascending=False)
    )

    print(f"\n{'Source':<50} {'N':>3} {'Enjoy':>7} {'Useful':>7}")
    print("-" * 72)
    for source, row in source_stats.iterrows():
        print(
            f"{str(source)[:50]:<50} {row['n']:>3.0f} "
            f"{row['enjoy_mean']:>7.2f} {row['useful_mean']:>7.2f}"
        )

    return df


def save_enriched(df: pd.DataFrame) -> None:
    out_path = OUTPUT_DIR / "books_enriched.csv"
    cols = [
        "title",
        "author",
        "Bookshelf",
        "earliest_modified",
        "latest_modified",
        "Enjoyment (/5)",
        "Usefulness /5 to Me",
        "Long Term Effects",
        "ol_title",
        "gb_page_count",
        "gb_average_rating",
        "gb_ratings_count",
        "pub_year",
        "ol_subjects",
    ]
    if "inferred_source" in df.columns:
        cols.append("inferred_source")
    existing = [c for c in cols if c in df.columns]
    df[existing].to_csv(out_path, index=False)
    print(f"\nSaved enriched data to {out_path}")


def main() -> None:
    df = load_data()
    df = enrich_with_openlibrary(df)
    analyze_api_features(df)
    df = infer_recommendation_sources(df)
    save_enriched(df)


if __name__ == "__main__":
    main()
