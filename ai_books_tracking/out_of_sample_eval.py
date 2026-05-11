"""Evaluate all prediction methods on new books read after the training set.

Usage:
    1. Create a CSV of new books with (at minimum) columns:
       title, author, Bookshelf, Enjoyment (/5), Usefulness /5 to Me
       Optional: earliest_modified, latest_modified, Long Term Effects

    2. Run: python out_of_sample_eval.py path/to/new_books.csv

    This will:
    - Train all models on the original 208-book dataset
    - Fetch Open Library metadata for new books
    - Infer recommendation sources via Gemini
    - Run Gemini LOO predictions for new books
    - Evaluate each method against the actual new ratings
    - Print a comparison table
"""

import json
import os
import sys
import time
import re
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_ENRICHED = OUTPUT_DIR / "books_enriched.csv"
TRAIN_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
UNFINISHED_CSV = DATA_DIR / "unfinished_books_2025_03_16.csv"
OL_CACHE = OUTPUT_DIR / "openlibrary_cache.json"
GEMINI_REC_CACHE = OUTPUT_DIR / "gemini_recommendations_cache.json"
GEMINI_OOS_CACHE = OUTPUT_DIR / "gemini_oos_predictions_cache.json"

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"


# ---- Shared helpers (duplicated from other scripts to keep this self-contained) ----


def clean_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    title = re.sub(r"\.(pdf|epub|html|txt)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"\s*-\s*[A-Z][a-z]+\s+(Press|Books|Publishing|House).*$", "", title)
    title = re.sub(r"\s+by\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"#\w+#", "", title)
    title = title.replace("_", " ")
    title = re.sub(r"\s*-?\s*libgen\.li\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\(\d\)$", "", title)
    return title.strip()


def load_json_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json_cache(cache: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ---- Data loading ----


def load_train() -> pd.DataFrame:
    if TRAIN_ENRICHED.exists():
        df = pd.read_csv(TRAIN_ENRICHED)
    else:
        df = pd.read_csv(TRAIN_EXPORT)
    df.columns = df.columns.str.strip()
    return df


def load_new_books(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = ["title", "Bookshelf", "Enjoyment (/5)", "Usefulness /5 to Me"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"New books CSV missing required columns: {missing}")
    return df


# ---- Survivorship bias correction ----


def load_quit_books(unfinished_path: Path = UNFINISHED_CSV) -> pd.DataFrame:
    """Load started-but-unfinished books (those with modification dates)."""
    if not unfinished_path.exists():
        raise FileNotFoundError(
            f"Unfinished books CSV not found at {unfinished_path}. "
            "Run: python Self_Tracking/play_books_to_csv.py"
        )
    df = pd.read_csv(unfinished_path)
    df.columns = df.columns.str.strip()
    # Rename 'bookshelf' -> 'Bookshelf' to match training data convention
    if "bookshelf" in df.columns and "Bookshelf" not in df.columns:
        df = df.rename(columns={"bookshelf": "Bookshelf"})
    has_dates = df["latest_modified"].notna() & (df["latest_modified"] != "")
    started = df[has_dates].copy()
    return started


def compute_finish_rates(
    train_df: pd.DataFrame, quit_df: pd.DataFrame
) -> dict[str, float]:
    """Compute P(finish|category) from finished + started-unfinished counts."""
    finished_counts = train_df["Bookshelf"].value_counts()
    quit_counts = quit_df["Bookshelf"].value_counts()

    all_cats = set(finished_counts.index) | set(quit_counts.index)
    rates: dict[str, float] = {}
    for cat in all_cats:
        n_fin = finished_counts.get(cat, 0)
        n_quit = quit_counts.get(cat, 0)
        total = n_fin + n_quit
        rates[cat] = n_fin / total if total > 0 else 1.0
    return rates


def compute_ipw_weights(
    train_df: pd.DataFrame, finish_rates: dict[str, float]
) -> np.ndarray:
    """Inverse probability weights: 1/P(finish|category) for each training book."""
    weights = (
        train_df["Bookshelf"].map(lambda cat: 1.0 / finish_rates.get(cat, 1.0)).values
    )
    return weights


def show_ipw_impact(train_df: pd.DataFrame, finish_rates: dict[str, float]) -> None:
    """Show how IPW shifts category means."""
    print("\n" + "=" * 70)
    print("SURVIVORSHIP BIAS: IPW CORRECTION")
    print("=" * 70)

    # Synthetic quit-book rating used for the unobserved counterfactual
    QUIT_ENJOY = 1.4

    print(
        f"\n{'Category':<25} {'P(fin)':>7} {'N_fin':>6} {'N_quit':>7} "
        f"{'Mean':>6} {'IPW Mean':>9} {'Pop Est':>8}"
    )
    print("-" * 75)

    for cat in sorted(finish_rates.keys()):
        cat_mask = train_df["Bookshelf"] == cat
        if cat_mask.sum() == 0:
            continue
        p_fin = finish_rates[cat]
        n_fin = cat_mask.sum()
        n_quit = round(n_fin * (1 / p_fin - 1))
        raw_mean = train_df.loc[cat_mask, "Enjoyment (/5)"].mean()
        # IPW mean of finished books (re-weights within finished, doesn't change
        # the mean itself since all books in same category get same weight)
        ipw_mean = raw_mean
        # Population estimate: P(finish)*E[enjoy|finish] + P(quit)*E[enjoy|quit]
        pop_est = p_fin * raw_mean + (1 - p_fin) * QUIT_ENJOY
        print(
            f"  {cat:<23} {p_fin:>7.0%} {n_fin:>6} {n_quit:>7} "
            f"{raw_mean:>6.2f} {ipw_mean:>9.2f} {pop_est:>8.2f}"
        )

    overall_raw = train_df["Enjoyment (/5)"].mean()
    # Weighted population estimate across categories present in training data
    train_cats = [cat for cat in finish_rates if (train_df["Bookshelf"] == cat).any()]
    fin_counts = train_df["Bookshelf"].value_counts()
    total_started = sum(
        fin_counts.get(cat, 0) / finish_rates[cat] for cat in train_cats
    )
    overall_pop = (
        sum(
            (fin_counts.get(cat, 0) / finish_rates[cat])
            * (
                finish_rates[cat]
                * train_df.loc[train_df["Bookshelf"] == cat, "Enjoyment (/5)"].mean()
                + (1 - finish_rates[cat]) * QUIT_ENJOY
            )
            for cat in train_cats
        )
        / total_started
    )

    print(f"\n  Overall finished mean: {overall_raw:.2f}")
    print(
        f"  Overall population estimate (incl. quits at {QUIT_ENJOY}): {overall_pop:.2f}"
    )
    print(f"  Bias magnitude: {overall_raw - overall_pop:+.2f}")
    print(
        "\n  Note: IPW within finished books doesn't change category means (all books"
    )
    print("  in a category get the same weight). The real correction is the population")
    print(f"  estimate which accounts for unobserved quit books rated ~{QUIT_ENJOY}.")
    print("  Categories with low finish rates (CS, Math, ML) are most inflated.")


def evaluate_quit_discrimination(
    train_df: pd.DataFrame,
    quit_df: pd.DataFrame,
) -> None:
    """Test whether models predict lower enjoyment for quit books vs finished.

    A good model should rank quit books below finished books. We measure this
    via AUC: probability that a random finished book gets a higher predicted
    enjoyment than a random quit book.
    """
    print("\n" + "=" * 70)
    print("QUIT-BOOK DISCRIMINATION")
    print("=" * 70)

    # Only evaluate categories present in both
    common_cats = set(train_df["Bookshelf"].unique()) & set(
        quit_df["Bookshelf"].unique()
    )
    quit_eval = quit_df[quit_df["Bookshelf"].isin(common_cats)].copy()
    fin_eval = train_df[train_df["Bookshelf"].isin(common_cats)].copy()

    print(f"\n  Finished books in shared categories: {len(fin_eval)}")
    print(f"  Quit books in shared categories: {len(quit_eval)}")

    # Enrich quit books with OL data
    quit_eval = enrich_new_books(quit_eval)

    # Prepare features
    quit_eval, num_feats, cat_feats = prepare_features(quit_eval, train_df)

    # Category mean predictions
    train_cat_means = train_df.groupby("Bookshelf")["Enjoyment (/5)"].mean()
    train_mean = train_df["Enjoyment (/5)"].mean()

    quit_cat_preds = (
        quit_eval["Bookshelf"].map(train_cat_means).fillna(train_mean).values
    )
    fin_cat_preds = fin_eval["Bookshelf"].map(train_cat_means).fillna(train_mean).values

    # ML model predictions for quit books
    try:
        ml_preds = train_and_predict(
            train_df, quit_eval, num_feats, cat_feats, "Enjoyment (/5)"
        )
    except Exception as e:
        print(f"  ML model error: {e}")
        ml_preds = {}

    # Prepare finished book ML predictions (train on all, predict on self — biased
    # but we're comparing distributions, not evaluating accuracy)
    fin_prepped, _, _ = prepare_features(fin_eval, train_df)
    try:
        fin_ml_preds = train_and_predict(
            train_df, fin_prepped, num_feats, cat_feats, "Enjoyment (/5)"
        )
    except Exception as e:
        print(f"  ML model error on finished: {e}")
        fin_ml_preds = {}

    from scipy.stats import mannwhitneyu

    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Category mean": (fin_cat_preds, quit_cat_preds),
    }
    for name in ml_preds:
        if name in fin_ml_preds:
            methods[name] = (fin_ml_preds[name], ml_preds[name])

    print(
        f"\n  {'Method':<25} {'Fin pred':>9} {'Quit pred':>10} {'Diff':>7} "
        f"{'AUC':>6} {'p':>8}"
    )
    print("  " + "-" * 70)

    for name, (fin_p, quit_p) in methods.items():
        fin_mean = np.nanmean(fin_p)
        quit_mean = np.nanmean(quit_p)
        diff = fin_mean - quit_mean

        # AUC via Mann-Whitney U
        valid_fin = fin_p[~np.isnan(fin_p)]
        valid_quit = quit_p[~np.isnan(quit_p)]
        if len(valid_fin) > 0 and len(valid_quit) > 0:
            u_stat, p_val = mannwhitneyu(valid_fin, valid_quit, alternative="greater")
            auc = u_stat / (len(valid_fin) * len(valid_quit))
        else:
            auc, p_val = 0.5, 1.0

        print(
            f"  {name:<25} {fin_mean:>9.2f} {quit_mean:>10.2f} {diff:>+7.2f} "
            f"{auc:>6.3f} {p_val:>8.4f}"
        )

    # Per-category breakdown
    print("\n  Per-category quit discrimination (Category mean):")
    print(
        f"  {'Category':<25} {'N_fin':>6} {'N_quit':>7} {'Fin mean':>9} {'Quit mean':>10}"
    )
    print("  " + "-" * 60)
    for cat in sorted(common_cats):
        n_fin = (fin_eval["Bookshelf"] == cat).sum()
        n_quit = (quit_eval["Bookshelf"] == cat).sum()
        fin_enjoy = fin_eval.loc[fin_eval["Bookshelf"] == cat, "Enjoyment (/5)"].mean()
        # Category mean prediction is the same for both finished and quit in same category
        cat_pred = train_cat_means.get(cat, train_mean)
        print(f"  {cat:<25} {n_fin:>6} {n_quit:>7} {fin_enjoy:>9.2f} {cat_pred:>10.2f}")

    print("\n  Interpretation: AUC > 0.5 means the model tends to predict higher")
    print("  enjoyment for finished books than quit books. AUC=1.0 would be perfect")
    print("  discrimination. Category mean gives all books in a category the same")
    print("  prediction, so its AUC reflects only cross-category differences in quit")
    print("  rates — categories with more quits (CS, ML, Math) have higher predicted")
    print("  enjoyment, which is the opposite of what we want, showing the bias.")


# ---- Feature engineering (same logic as prediction_model.py) ----


def prepare_features(
    df: pd.DataFrame, train_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare features for new books, using training data for author means etc."""
    df = df.copy()

    for col in ["earliest_modified", "latest_modified"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    if "reading_days" not in df.columns:
        if "earliest_modified" in df.columns and "latest_modified" in df.columns:
            df["reading_days"] = (
                df["latest_modified"] - df["earliest_modified"]
            ).dt.days
        else:
            df["reading_days"] = np.nan

    if "year_finished" not in df.columns:
        if "latest_modified" in df.columns:
            df["year_finished"] = df["latest_modified"].dt.year
        else:
            df["year_finished"] = 2025.0

    if "month_finished" not in df.columns:
        if "latest_modified" in df.columns:
            df["month_finished"] = df["latest_modified"].dt.month
        else:
            df["month_finished"] = 6.0

    if "Long Term Effects" in df.columns:
        df["note_length"] = df["Long Term Effects"].fillna("").str.len()
    else:
        df["note_length"] = 0

    if "gb_page_count" in df.columns:
        df["log_pages"] = np.log1p(df["gb_page_count"].fillna(0))
    else:
        df["log_pages"] = 0.0

    if "pub_year" in df.columns:
        df["book_age"] = 2025 - df["pub_year"].fillna(2025)
    else:
        df["book_age"] = 0.0

    # Author mean from TRAINING data only (no leakage)
    train_author_means = (
        train_df.assign(
            author_clean=train_df["author"].astype(str).str.strip().str.lower()
        )
        .groupby("author_clean")["Enjoyment (/5)"]
        .mean()
    )
    train_global_mean = train_df["Enjoyment (/5)"].mean()

    if "author" in df.columns:
        author_clean = df["author"].astype(str).str.strip().str.lower()
        df["author_mean_enjoy_loo"] = author_clean.map(train_author_means).fillna(
            train_global_mean
        )
        train_author_counts = train_df.assign(
            author_clean=train_df["author"].astype(str).str.strip().str.lower()
        )["author_clean"].value_counts()
        df["author_book_count"] = author_clean.map(train_author_counts).fillna(0)
    else:
        df["author_mean_enjoy_loo"] = train_global_mean
        df["author_book_count"] = 0

    numeric_features = [
        "year_finished",
        "reading_days",
        "note_length",
        "log_pages",
        "book_age",
        "author_mean_enjoy_loo",
        "author_book_count",
    ]
    if "gb_average_rating" in df.columns:
        numeric_features.append("gb_average_rating")
    if "gb_ratings_count" in df.columns:
        df["log_ratings_count"] = np.log1p(df["gb_ratings_count"].fillna(0))
        numeric_features.append("log_ratings_count")

    categorical_features = ["Bookshelf"]
    if "inferred_source" in df.columns:
        categorical_features.append("inferred_source")

    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df, numeric_features, categorical_features


# ---- Open Library enrichment ----


def fetch_ol(title: str, author: str, cache: dict) -> dict | None:
    cache_key = f"{title}|||{author}"
    if cache_key in cache:
        return cache[cache_key]

    clean = clean_title(title)
    if len(clean) < 3:
        cache[cache_key] = None
        return None

    author_valid = (
        author
        and str(author) != "by"
        and str(author) != "Unknown"
        and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(author))
    )

    q = f"{clean} {author}" if author_valid else clean
    params = {
        "q": q,
        "limit": 1,
        "fields": "title,author_name,first_publish_year,number_of_pages_median,"
        "ratings_average,ratings_count,subject",
    }

    try:
        resp = requests.get(OPEN_LIBRARY_SEARCH, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("numFound", 0) == 0 and author_valid:
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
            "ol_page_count": doc.get("number_of_pages_median"),
            "ol_rating": doc.get("ratings_average"),
            "ol_ratings_count": doc.get("ratings_count"),
            "ol_first_publish_year": doc.get("first_publish_year"),
        }
        cache[cache_key] = result
        return result
    except requests.RequestException as e:
        print(f"  OL error for '{clean[:40]}': {e}")
        return None


def enrich_new_books(df: pd.DataFrame) -> pd.DataFrame:
    cache = load_json_cache(OL_CACHE)
    print("Fetching Open Library metadata for new books...")
    results = []
    for _, row in df.iterrows():
        result = fetch_ol(row["title"], row.get("author", ""), cache)
        results.append(result or {})
        time.sleep(0.15)
    save_json_cache(cache, OL_CACHE)

    ol_df = pd.DataFrame(results)
    for col in [
        "ol_page_count",
        "ol_rating",
        "ol_ratings_count",
        "ol_first_publish_year",
    ]:
        if col not in ol_df.columns:
            ol_df[col] = np.nan

    df = pd.concat([df.reset_index(drop=True), ol_df.reset_index(drop=True)], axis=1)
    df["gb_page_count"] = df.get("ol_page_count")
    df["gb_average_rating"] = df.get("ol_rating")
    df["gb_ratings_count"] = df.get("ol_ratings_count")
    df["pub_year"] = pd.to_numeric(df.get("ol_first_publish_year"), errors="coerce")

    n_found = sum(1 for r in results if r)
    print(f"  {n_found}/{len(df)} books found in Open Library")
    return df


# ---- Gemini recommendation source ----


def infer_sources_new(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import google.generativeai as genai
    except ImportError:
        print("  google-generativeai not available, skipping source inference")
        return df

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  No GEMINI_API_KEY, skipping source inference")
        return df

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    cache = load_json_cache(GEMINI_REC_CACHE)

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

    to_query = [row for _, row in df.iterrows() if row["title"] not in cache]
    if to_query:
        print(f"  Querying Gemini for {len(to_query)} book sources...")
        book_list = "\n".join(
            f"- \"{row['title']}\" by {row.get('author', 'unknown')} "
            f"(category: {row.get('Bookshelf', 'unknown')})"
            for row in to_query
        )
        prompt = f"""For each book below, infer the most likely recommendation source from this list:
{chr(10).join(f'  {s}' for s in known_sources)}

The reader is a software engineer in the rationalist / EA / tech community.

Books:
{book_list}

Respond with ONLY a JSON object mapping each book title (EXACT) to the source name."""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            cache.update(result)
            save_json_cache(cache, GEMINI_REC_CACHE)
        except Exception as e:
            print(f"  Gemini source error: {e}")

    df["inferred_source"] = df["title"].map(cache).fillna("Unknown")
    return df


# ---- Gemini LOO predictor for new books ----


def gemini_predict_new(
    new_df: pd.DataFrame, train_df: pd.DataFrame
) -> dict[str, float]:
    """Ask Gemini to predict ratings for each new book given the full training profile."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("  google-generativeai not available")
        return {}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  No GEMINI_API_KEY")
        return {}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    cache = load_json_cache(GEMINI_OOS_CACHE)

    # Build taste profile from ALL training data
    cat_stats = (
        train_df.groupby("Bookshelf")
        .agg(
            n=("Enjoyment (/5)", "count"),
            enjoy_mean=("Enjoyment (/5)", "mean"),
            useful_mean=("Usefulness /5 to Me", "mean"),
        )
        .sort_values("enjoy_mean", ascending=False)
    )
    cat_summary = "\n".join(
        f"  {cat}: {row['n']:.0f} books, avg enjoyment {row['enjoy_mean']:.1f}/5, "
        f"avg usefulness {row['useful_mean']:.1f}/5"
        for cat, row in cat_stats.iterrows()
    )
    top5 = train_df.nlargest(5, "Enjoyment (/5)")
    top_str = "\n".join(
        f"  - {row['title'][:60]} (enjoy={row['Enjoyment (/5)']:.1f}, "
        f"useful={row['Usefulness /5 to Me']:.1f})"
        for _, row in top5.iterrows()
    )
    bottom5 = train_df.nsmallest(5, "Enjoyment (/5)")
    bottom_str = "\n".join(
        f"  - {row['title'][:60]} (enjoy={row['Enjoyment (/5)']:.1f})"
        for _, row in bottom5.iterrows()
    )

    profile = f"""Reader profile: Software engineer, rationalist community.
{len(train_df)} books read, avg enjoyment {train_df['Enjoyment (/5)'].mean():.2f}/5, avg usefulness {train_df['Usefulness /5 to Me'].mean():.2f}/5

Category breakdown:
{cat_summary}

Most enjoyed:
{top_str}

Least enjoyed:
{bottom_str}"""

    predictions = {}
    for _, row in new_df.iterrows():
        title = row["title"]
        cache_key = f"oos_{title}"
        if cache_key in cache:
            predictions[title] = cache[cache_key]
            continue

        author = row.get("author", "unknown")
        cat = row.get("Bookshelf", "unknown")
        prompt = f"""{profile}

Predict this reader's enjoyment and usefulness rating (1-5, 0.5 increments):
  Title: {title}
  Author: {author}
  Category: {cat}

Respond with ONLY JSON: {{"enjoyment": X, "usefulness": Y}}"""

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            pred = json.loads(text)
            cache[cache_key] = pred
            predictions[title] = pred
            save_json_cache(cache, GEMINI_OOS_CACHE)
            time.sleep(1.5)
        except Exception as e:
            print(f"  Gemini predict error for '{title[:40]}': {e}")

    return predictions


# ---- Train models on original data, predict new ----


def train_and_predict(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
) -> dict[str, np.ndarray]:
    """Train each model on training set, predict on new books."""
    # Prepare training features using the same logic (train_df as its own reference)
    train_prepped, train_num, train_cat = prepare_features(train_df, train_df)

    # Use intersection of features available in both
    num_feats = [
        f
        for f in numeric_features
        if f in train_prepped.columns and f in new_df.columns
    ]
    cat_feats = [
        f
        for f in categorical_features
        if f in train_prepped.columns and f in new_df.columns
    ]

    valid_train = train_prepped[train_prepped[target].notna()].copy()
    y_train = valid_train[target].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                ),
                cat_feats,
            ),
        ]
    )

    models = {
        "Ridge": Ridge(alpha=10.0),
        "GBM": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    predictions = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(valid_train, y_train)
        preds = pipe.predict(new_df)
        predictions[name] = preds

    return predictions


# ---- Main evaluation ----


def evaluate(new_path: str | None = None) -> None:
    print("=" * 70)
    print("OUT-OF-SAMPLE EVALUATION")
    print("=" * 70)

    train_df = load_train()

    # --- Survivorship bias analysis ---
    try:
        quit_df = load_quit_books()
        finish_rates = compute_finish_rates(train_df, quit_df)
        show_ipw_impact(train_df, finish_rates)
        evaluate_quit_discrimination(train_df, quit_df)
    except FileNotFoundError as e:
        print(f"\n  Skipping survivorship bias analysis: {e}")
        quit_df = None

    if new_path is None:
        print("\nNo new books CSV provided, skipping OOS prediction evaluation.")
        return

    new_df = load_new_books(new_path)
    print(f"\nTraining set: {len(train_df)} books")
    print(f"New books to evaluate: {len(new_df)} books")

    # Enrich new books with Open Library data
    new_df = enrich_new_books(new_df)

    # Infer recommendation sources
    new_df = infer_sources_new(new_df)

    # Prepare features
    new_df, num_feats, cat_feats = prepare_features(new_df, train_df)

    # Collect all predictions
    all_predictions: dict[str, dict[str, np.ndarray]] = {}

    for target in ["Enjoyment (/5)", "Usefulness /5 to Me"]:
        actuals = new_df[target].values
        n = len(actuals)
        preds_collection: dict[str, np.ndarray] = {}

        # 1. Global mean baseline
        train_mean = train_df[target].mean()
        preds_collection["Global mean"] = np.full(n, train_mean)

        # 2. Category mean baseline
        train_cat_means = train_df.groupby("Bookshelf")[target].mean()
        cat_preds = new_df["Bookshelf"].map(train_cat_means).fillna(train_mean).values
        preds_collection["Category mean"] = cat_preds

        # 3. OL rating >= 4.0 adjusted category mean
        if "gb_average_rating" in new_df.columns:
            ol_adjusted = cat_preds.copy()
            high_ol = new_df["gb_average_rating"].fillna(0) >= 4.0
            ol_adjusted[high_ol] = (
                ol_adjusted[high_ol] + 0.39
            )  # observed diff from training
            preds_collection["Cat + OL>=4.0 bump"] = ol_adjusted

        # 4. ML models
        try:
            ml_preds = train_and_predict(train_df, new_df, num_feats, cat_feats, target)
            preds_collection.update(ml_preds)
        except Exception as e:
            print(f"  ML model error for {target}: {e}")

        # 5. Gemini predictions
        if target == "Enjoyment (/5)":
            gemini_preds = gemini_predict_new(new_df, train_df)
            if gemini_preds:
                gemini_enjoy = []
                gemini_useful = []
                for _, row in new_df.iterrows():
                    pred = gemini_preds.get(row["title"], {})
                    gemini_enjoy.append(pred.get("enjoyment", np.nan))
                    gemini_useful.append(pred.get("usefulness", np.nan))
                preds_collection["Gemini"] = np.array(gemini_enjoy)
                all_predictions["Gemini_usefulness"] = np.array(gemini_useful)

        if target == "Usefulness /5 to Me" and "Gemini_usefulness" in all_predictions:
            preds_collection["Gemini"] = all_predictions["Gemini_usefulness"]

        all_predictions[target] = preds_collection

    # Print results
    for target in ["Enjoyment (/5)", "Usefulness /5 to Me"]:
        print(f"\n{'=' * 70}")
        print(f"RESULTS: {target}")
        print(f"{'=' * 70}")

        actuals = new_df[target].values
        preds_collection = all_predictions[target]

        print(f"\n{'Method':<30} {'MAE':>7} {'RMSE':>7} {'R':>7} {'vs Cat':>10}")
        print("-" * 65)

        cat_mae = None
        for name, preds in preds_collection.items():
            valid = ~np.isnan(preds) & ~np.isnan(actuals)
            if valid.sum() < 3:
                print(f"{name:<30} (too few valid predictions: {valid.sum()})")
                continue

            mae = mean_absolute_error(actuals[valid], preds[valid])
            rmse = np.sqrt(mean_squared_error(actuals[valid], preds[valid]))
            r = (
                np.corrcoef(actuals[valid], preds[valid])[0, 1]
                if valid.sum() > 2
                else 0
            )

            if name == "Category mean":
                cat_mae = mae
                vs_cat = "baseline"
            elif cat_mae:
                improvement = (cat_mae - mae) / cat_mae * 100
                vs_cat = f"{improvement:+.1f}%"
            else:
                vs_cat = ""

            print(f"{name:<30} {mae:>7.3f} {rmse:>7.3f} {r:>7.3f} {vs_cat:>10}")

    # Per-book detail
    print(f"\n{'=' * 70}")
    print("PER-BOOK DETAIL")
    print(f"{'=' * 70}")

    cat_means = train_df.groupby("Bookshelf")["Enjoyment (/5)"].mean()
    gemini_cache = load_json_cache(GEMINI_OOS_CACHE)

    print(
        f"\n{'Title':<45} {'Cat':>5} {'Actual':>7} {'CatMn':>6} {'Gemini':>7} {'Ridge':>6}"
    )
    print("-" * 80)

    ridge_preds = all_predictions.get("Enjoyment (/5)", {}).get("Ridge")
    for i, (_, row) in enumerate(new_df.iterrows()):
        title = str(row["title"])[:44]
        cat_pred = cat_means.get(row["Bookshelf"], train_df["Enjoyment (/5)"].mean())
        gem_pred = gemini_cache.get(f"oos_{row['title']}", {}).get("enjoyment", "")
        gem_str = f"{gem_pred:.1f}" if isinstance(gem_pred, (int, float)) else "N/A"
        ridge_str = f"{ridge_preds[i]:.1f}" if ridge_preds is not None else "N/A"
        print(
            f"{title:<45} {row['Bookshelf'][:5]:>5} "
            f"{row['Enjoyment (/5)']:>7.1f} {cat_pred:>6.1f} {gem_str:>7} {ridge_str:>6}"
        )

    # Save results
    out_path = OUTPUT_DIR / "oos_results.csv"
    result_rows = []
    for i, (_, row) in enumerate(new_df.iterrows()):
        r = {
            "title": row["title"],
            "category": row["Bookshelf"],
            "actual_enjoy": row["Enjoyment (/5)"],
            "actual_useful": row["Usefulness /5 to Me"],
            "pred_cat_mean_enjoy": cat_means.get(
                row["Bookshelf"], train_df["Enjoyment (/5)"].mean()
            ),
        }
        gem = gemini_cache.get(f"oos_{row['title']}", {})
        r["pred_gemini_enjoy"] = gem.get("enjoyment")
        r["pred_gemini_useful"] = gem.get("usefulness")
        if ridge_preds is not None:
            r["pred_ridge_enjoy"] = ridge_preds[i]
        result_rows.append(r)

    pd.DataFrame(result_rows).to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python out_of_sample_eval.py [path_to_new_books.csv]")
        print()
        print("  With no arguments: runs survivorship bias analysis only")
        print("  With CSV path: also runs OOS prediction evaluation")
        print()
        print("Expected CSV format:")
        print("  title,author,Bookshelf,Enjoyment (/5),Usefulness /5 to Me")
        print("  The Great Gatsby,F. Scott Fitzgerald,Literature,4.0,2.0")
        print()
        print("Optional columns: earliest_modified, latest_modified, Long Term Effects")
        evaluate(None)
    else:
        evaluate(sys.argv[1])


if __name__ == "__main__":
    main()
