
import pandas as pd
from pathlib import Path

def main():
    # Load all sources
    train_path = Path('ai_books_tracking/books_enriched.csv')
    holdout_path = Path('data/Books Read and their effects - new_books_to_rate 2026.csv')
    verified_train_path = Path('ai_books_tracking/verified_training.csv')
    verified_holdout_path = Path('ai_books_tracking/verified_holdout.csv')

    # 1. Training Set
    df_train = pd.read_csv(train_path)[['title', 'author']]
    df_train['dataset'] = 'Training'
    
    # 2. Holdout Set
    df_holdout = pd.read_csv(holdout_path)[['title', 'author']]
    df_holdout['dataset'] = 'Holdout'
    
    combined_original = pd.concat([df_train, df_holdout], ignore_index=True)
    
    # 3. Verified Results
    verified_results = []
    if verified_train_path.exists():
        v_train = pd.read_csv(verified_train_path)
        v_train['dataset'] = 'Training'
        verified_results.append(v_train)
    if verified_holdout_path.exists():
        v_holdout = pd.read_csv(verified_holdout_path)
        v_holdout['dataset'] = 'Holdout'
        verified_results.append(v_holdout)
        
    if verified_results:
        combined_verified = pd.concat(verified_results, ignore_index=True)
        # Merge on original title and original author to show the pairing
        final = pd.merge(
            combined_original, 
            combined_verified[['original_title', 'original_author', 'goodreads_title', 'goodreads_author', 'goodreads_rating', 'goodreads_url', 'confidence']],
            left_on=['title', 'author'],
            right_on=['original_title', 'original_author'],
            how='left'
        )
    else:
        final = combined_original
        print("No verified metadata found yet.")

    # Export for review
    out_path = Path('ai_books_tracking/FINAL_METADATA_REVIEW.csv')
    final.to_csv(out_path, index=False)
    
    # Summary Table for the CLI
    print(f"\nConsolidated {len(final)} books for review.")
    print(f"Verified metadata available for {final['goodreads_url'].notna().sum()} books.")
    print(f"\nReview the full list at: {out_path}")
    
    print("\nTop Matches (Sample):")
    print(final[final['goodreads_url'].notna()][['title', 'goodreads_title', 'goodreads_author', 'goodreads_rating']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
