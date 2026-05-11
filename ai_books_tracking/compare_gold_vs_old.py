
import pandas as pd
from pathlib import Path

def main():
    gold_path = Path('data/Books Read and their effects - master_book_metadata_cleaned.csv')
    old_path = Path('ai_books_tracking/books_enriched_with_goodreads.csv')
    
    if not gold_path.exists() or not old_path.exists():
        print("Required files missing for comparison.")
        return

    # 1. Internal Audit of Golden File
    df_gold_raw = pd.read_csv(gold_path)
    
    # User note: "ratings" is the truth, "ratings good reads gsheets" might be wrong.
    df_gold_raw['ratings'] = pd.to_numeric(df_gold_raw['ratings'], errors='coerce')
    df_gold_raw['ratings good reads gsheets'] = pd.to_numeric(df_gold_raw['ratings good reads gsheets'], errors='coerce')
    
    internal_diffs = df_gold_raw[
        (df_gold_raw['ratings'] - df_gold_raw['ratings good reads gsheets']).abs() > 0.01
    ].copy()
    
    print("=== INTERNAL AUDIT: 'ratings' vs 'gsheets' in Golden File ===")
    if not internal_diffs.empty:
        print(f"Found {len(internal_diffs)} internal discrepancies. Using 'ratings' as truth.")
        print(internal_diffs[['title', 'ratings', 'ratings good reads gsheets']].head(10).to_string(index=False))
    else:
        print("No differences between 'ratings' and 'gsheets' columns found.")

    # 2. Comparison with Old AI Enriched Data
    df_gold = df_gold_raw[['title', 'corrected_author', 'ratings']]
    df_old = pd.read_csv(old_path)[['title', 'author', 'goodreads_rating']]
    
    df_gold['title_norm'] = df_gold['title'].str.lower().str.strip()
    df_old['title_norm'] = df_old['title'].str.lower().str.strip()
    
    # Ensure numeric for comparison
    df_old['goodreads_rating'] = pd.to_numeric(df_old['goodreads_rating'], errors='coerce')
    
    merged = pd.merge(df_gold, df_old, on='title_norm', how='inner')
    
    # Filter for significant differences
    merged['rating_diff'] = (merged['ratings'].fillna(0) - merged['goodreads_rating'].fillna(0)).abs()
    diffs = merged[merged['rating_diff'] > 0.05].copy()
    
    print(f"\n=== EXTERNAL COMPARISON: Golden 'ratings' vs Old AI Ratings ===")
    print(f"Total books compared: {len(merged)}")
    print(f"Significant differences found: {len(diffs)}")
    
    # Save all external differences
    diffs_out = Path('ai_books_tracking/GR_RATING_EXTERNAL_DIFFS.csv')
    diffs.to_csv(diffs_out, index=False)
    
    # Print all external differences
    print("\nALL EXTERNAL DIFFERENCES:")
    print(diffs[['title_x', 'corrected_author', 'ratings', 'goodreads_rating']].to_string(index=False))
    
    print(f"\nFull list of external differences saved to: {diffs_out}")

if __name__ == "__main__":
    main()
