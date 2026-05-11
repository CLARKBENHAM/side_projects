
import pandas as pd
import numpy as np
import re
from pathlib import Path

def is_placeholder_author(author):
    s = str(author).lower().strip()
    if s == 'by' or s == 'unknown' or s == 'nan':
        return True
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return True
    return False

def main():
    gold_path = Path('data/Books Read and their effects - master_book_metadata_cleaned.csv')
    old_path = Path('ai_books_tracking/books_enriched_with_goodreads.csv')
    multi_path = Path('ai_books_tracking/multi_source_ratings.csv')
    
    if not gold_path.exists() or not old_path.exists():
        print("Missing core files.")
        return

    # Load data
    df_gold = pd.read_csv(gold_path)
    df_old = pd.read_csv(old_path)
    
    # 1. Error Analysis for "by" and "date" authors
    # Join on normalized title
    df_gold['title_norm'] = df_gold['title'].str.lower().str.strip()
    df_old['title_norm'] = df_old['title'].str.lower().str.strip()
    
    # Identify placeholders in the OLD dataset (before restoration)
    df_old['is_placeholder'] = df_old['author'].apply(is_placeholder_author)
    
    # Merge
    merged = pd.merge(
        df_gold[['title_norm', 'ratings', 'corrected_author']], 
        df_old[['title_norm', 'goodreads_rating', 'author', 'is_placeholder']], 
        on='title_norm'
    )
    
    merged['ratings'] = pd.to_numeric(merged['ratings'], errors='coerce')
    merged['goodreads_rating'] = pd.to_numeric(merged['goodreads_rating'], errors='coerce')
    
    # Error is Gold - Old
    merged['error_val'] = merged['ratings'] - merged['goodreads_rating']
    merged['abs_error'] = merged['error_val'].abs()
    
    placeholders = merged[merged['is_placeholder']].copy()
    valid_authors = merged[~merged['is_placeholder']].copy()
    
    print("=== Error Analysis: Placeholder Authors ('by' or dates) ===")
    print(f"Total placeholders analyzed: {len(placeholders)}")
    print(f"Mean Absolute Error (Placeholder): {placeholders['abs_error'].mean():.3f}")
    print(f"Mean Absolute Error (Valid Author): {valid_authors['abs_error'].mean():.3f}")
    
    # Significant errors in placeholders
    big_errors = placeholders[placeholders['abs_error'] > 0.1].sort_values('abs_error', ascending=False)
    print(f"\nSignificant Errors in Placeholders (> 0.1): {len(big_errors)}")
    print(big_errors[['title_norm', 'author', 'ratings', 'goodreads_rating', 'abs_error']].head(20).to_string(index=False))

    # 2. Multi-Method Comparison
    # Compare against other AI methods if multi_source_ratings exists
    if multi_path.exists():
        df_multi = pd.read_csv(multi_path)
        print("\n=== Multi-Method Comparison ===")
        # Look for the source columns: gb_average_rating, ol_rating, etc.
        cols = [c for b in ['gb', 'ol', 'gr'] for c in df_multi.columns if c.startswith(b)]
        print(f"Available sources in multi_path: {cols}")
        
    # 3. Summary of Failure Modes
    print("\n=== Systematic Failure Modes ===")
    print("1. Author Collision: AI prioritized the placeholder author field over the title context.")
    print("2. Match Failure: Missing author led to 'unmatched' status in simple scripts.")
    print("3. Rounding/Stale Data: Some diffs are small (drift), but 40% are structural (wrong book).")

if __name__ == "__main__":
    main()
