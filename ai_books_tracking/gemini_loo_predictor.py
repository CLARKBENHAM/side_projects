"""1. Investigate how Gemini classified 'friend recommendation' books.
2. Build an LLM-based LOO predictor: summarize tastes, have Gemini predict ratings."""

import json
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
PLAY_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
ENRICHED_FILE = OUTPUT_DIR / "books_enriched.csv"
GEMINI_CACHE_FILE = OUTPUT_DIR / "gemini_recommendations_cache.json"
LOO_CACHE_FILE = OUTPUT_DIR / "gemini_loo_cache.json"


def investigate_friend_recommendations() -> None:
    """Show what info Gemini had when classifying books as 'Friend recommendation'."""
    print("=" * 70)
    print("INVESTIGATING 'FRIEND RECOMMENDATION' CLASSIFICATIONS")
    print("=" * 70)

    df = pd.read_csv(PLAY_EXPORT)
    df.columns = df.columns.str.strip()
    cache = json.load(open(GEMINI_CACHE_FILE))

    friend_titles = [k for k, v in cache.items() if v == "Friend recommendation"]

    print(f"\n{len(friend_titles)} books classified as 'Friend recommendation':")
    print("\nFor each, here is the info Gemini had:\n")

    for title in friend_titles:
        row = df[df["title"] == title]
        if row.empty:
            print(f"  '{title}' - NOT FOUND in data")
            continue
        row = row.iloc[0]
        author = row.get("author", "unknown")
        cat = row.get("Bookshelf", "unknown")
        notes = str(row.get("Long Term Effects", ""))[:200]
        enjoy = row.get("Enjoyment (/5)", "?")
        useful = row.get("Usefulness /5 to Me", "?")
        print(f"  Title:  {title}")
        print(f"  Author: {author}, Category: {cat}")
        print(f"  Rating: enjoy={enjoy}, useful={useful}")
        if notes and notes != "nan":
            print(f"  Notes:  {notes[:150]}")
        print()

    print("ANALYSIS: Gemini was inferring 'friend recommendation' based on the book's")
    print("characteristics and the reader profile, NOT from actual data about who")
    print(
        "recommended what. These are books that LOOK like friend recs (popular fiction,"
    )
    print("specific genre picks, well-known accessible books) rather than books that")
    print(
        "appeared on known recommendation lists or fit the rationalist community profile."
    )
    print(
        "\nThis classification is speculative and should be taken with a grain of salt."
    )


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def build_taste_profile(df: pd.DataFrame, exclude_idx: int) -> str:
    """Build a summary of reading tastes excluding one book."""
    other = df.drop(index=exclude_idx)

    # Category stats
    cat_stats = (
        other.groupby("Bookshelf")
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

    # Top and bottom books
    top5 = other.nlargest(5, "Enjoyment (/5)")
    bottom5 = other.nsmallest(5, "Enjoyment (/5)")

    top_str = "\n".join(
        f"  - {row['title'][:60]} (enjoy={row['Enjoyment (/5)']:.1f}, "
        f"useful={row['Usefulness /5 to Me']:.1f}) "
        f"Notes: {str(row.get('Long Term Effects', ''))[:100]}"
        for _, row in top5.iterrows()
    )
    bottom_str = "\n".join(
        f"  - {row['title'][:60]} (enjoy={row['Enjoyment (/5)']:.1f}, "
        f"useful={row['Usefulness /5 to Me']:.1f}) "
        f"Notes: {str(row.get('Long Term Effects', ''))[:100]}"
        for _, row in bottom5.iterrows()
    )

    # Category-specific examples (top 2 per category)
    cat_examples = []
    for cat in cat_stats.index:
        cat_df = other[other["Bookshelf"] == cat].nlargest(2, "Enjoyment (/5)")
        for _, row in cat_df.iterrows():
            cat_examples.append(
                f"  [{cat}] {row['title'][:50]} "
                f"(enjoy={row['Enjoyment (/5)']:.1f}, useful={row['Usefulness /5 to Me']:.1f})"
            )

    return f"""Reader profile: Software engineer, rationalist community member.
Overall stats: {len(other)} books, avg enjoyment {other['Enjoyment (/5)'].mean():.2f}/5, avg usefulness {other['Usefulness /5 to Me'].mean():.2f}/5

Category breakdown:
{cat_summary}

Most enjoyed books:
{top_str}

Least enjoyed books:
{bottom_str}

Category highlights:
{chr(10).join(cat_examples[:14])}"""


def gemini_loo_predict(df: pd.DataFrame, sample_size: int = 40) -> None:
    """Leave-one-out prediction using Gemini: for each held-out book,
    summarize tastes from remaining books and ask Gemini to predict the rating."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\nNo GEMINI_API_KEY found, skipping LLM LOO prediction")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    cache = load_cache(LOO_CACHE_FILE)

    print("\n" + "=" * 70)
    print("GEMINI LOO PREDICTION")
    print("=" * 70)

    # Sample books stratified by category for efficiency
    if sample_size < len(df):
        sampled = df.groupby("Bookshelf", group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(2, int(sample_size * len(x) / len(df)))),
                random_state=42,
            )
        )
        print(
            f"\nSampled {len(sampled)} books (stratified by category) for LOO prediction"
        )
    else:
        sampled = df

    results = []
    for idx, row in sampled.iterrows():
        title = row["title"]
        cache_key = f"loo_{title}"
        if cache_key in cache:
            pred = cache[cache_key]
        else:
            profile = build_taste_profile(df, idx)
            author = row.get("author", "unknown")
            cat = row.get("Bookshelf", "unknown")

            prompt = f"""{profile}

Based on this reader's taste profile, predict how much they would enjoy and find useful the following book. Give ratings on a 1-5 scale (can use 0.5 increments).

Book to predict:
  Title: {title}
  Author: {author}
  Category: {cat}

Respond with ONLY a JSON object like: {{"enjoyment": 3.5, "usefulness": 2.0, "reasoning": "brief reason"}}"""

            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                pred = json.loads(text)
                cache[cache_key] = pred
                save_cache(cache, LOO_CACHE_FILE)
                time.sleep(1.5)
            except Exception as e:
                print(f"  Error for '{title[:40]}': {e}")
                continue

        if pred and "enjoyment" in pred:
            results.append(
                {
                    "title": title,
                    "category": row["Bookshelf"],
                    "actual_enjoy": row["Enjoyment (/5)"],
                    "predicted_enjoy": pred["enjoyment"],
                    "actual_useful": row["Usefulness /5 to Me"],
                    "predicted_useful": pred.get("usefulness", np.nan),
                    "reasoning": pred.get("reasoning", ""),
                }
            )

    if not results:
        print("No predictions generated")
        return

    res_df = pd.DataFrame(results)
    save_cache(cache, LOO_CACHE_FILE)

    # Evaluate
    n = len(res_df)
    print(f"\n{n} predictions generated")

    mae_enjoy = mean_absolute_error(res_df["actual_enjoy"], res_df["predicted_enjoy"])
    r_enjoy = np.corrcoef(res_df["actual_enjoy"], res_df["predicted_enjoy"])[0, 1]

    # Baselines
    global_mean = df["Enjoyment (/5)"].mean()
    baseline_mae = mean_absolute_error(res_df["actual_enjoy"], np.full(n, global_mean))

    cat_preds = []
    for _, row in res_df.iterrows():
        cat_mean = df[df["Bookshelf"] == row["category"]]["Enjoyment (/5)"].mean()
        cat_preds.append(cat_mean)
    cat_mae = mean_absolute_error(res_df["actual_enjoy"], cat_preds)

    print("\nEnjoyment prediction:")
    print(f"  Global mean baseline MAE: {baseline_mae:.3f}")
    print(f"  Category mean baseline MAE: {cat_mae:.3f}")
    print(f"  Gemini LOO MAE: {mae_enjoy:.3f}")
    print(f"  Gemini LOO R: {r_enjoy:.3f}")
    print(f"  Improvement vs category: {(cat_mae - mae_enjoy) / cat_mae * 100:+.1f}%")

    if res_df["predicted_useful"].notna().sum() > 5:
        valid_u = res_df[res_df["predicted_useful"].notna()]
        mae_useful = mean_absolute_error(
            valid_u["actual_useful"], valid_u["predicted_useful"]
        )
        r_useful = np.corrcoef(valid_u["actual_useful"], valid_u["predicted_useful"])[
            0, 1
        ]
        baseline_useful = mean_absolute_error(
            valid_u["actual_useful"],
            np.full(len(valid_u), df["Usefulness /5 to Me"].mean()),
        )
        print("\nUsefulness prediction:")
        print(f"  Global mean baseline MAE: {baseline_useful:.3f}")
        print(f"  Gemini LOO MAE: {mae_useful:.3f}")
        print(f"  Gemini LOO R: {r_useful:.3f}")

    # Show biggest misses
    res_df["error"] = res_df["predicted_enjoy"] - res_df["actual_enjoy"]
    res_df["abs_error"] = res_df["error"].abs()

    print("\nBiggest over-predictions (Gemini thought you'd like it more):")
    for _, row in res_df.nlargest(5, "error").iterrows():
        print(
            f"  {row['title'][:50]:<50} pred={row['predicted_enjoy']:.1f} "
            f"actual={row['actual_enjoy']:.1f} ({row['error']:+.1f})"
        )

    print("\nBiggest under-predictions (Gemini thought you'd like it less):")
    for _, row in res_df.nsmallest(5, "error").iterrows():
        print(
            f"  {row['title'][:50]:<50} pred={row['predicted_enjoy']:.1f} "
            f"actual={row['actual_enjoy']:.1f} ({row['error']:+.1f})"
        )

    # Per-category performance
    print("\nPer-category Gemini LOO MAE:")
    for cat in res_df["category"].unique():
        cat_res = res_df[res_df["category"] == cat]
        if len(cat_res) >= 3:
            cat_mae_g = mean_absolute_error(
                cat_res["actual_enjoy"], cat_res["predicted_enjoy"]
            )
            cat_mae_b = mean_absolute_error(
                cat_res["actual_enjoy"],
                np.full(
                    len(cat_res), df[df["Bookshelf"] == cat]["Enjoyment (/5)"].mean()
                ),
            )
            print(
                f"  {cat:<25} Gemini={cat_mae_g:.2f}, Cat baseline={cat_mae_b:.2f} "
                f"({(cat_mae_b - cat_mae_g) / cat_mae_b * 100:+.1f}%)"
            )


def main() -> None:
    investigate_friend_recommendations()

    df = pd.read_csv(ENRICHED_FILE)
    df.columns = df.columns.str.strip()
    gemini_loo_predict(df, sample_size=50)


if __name__ == "__main__":
    main()
