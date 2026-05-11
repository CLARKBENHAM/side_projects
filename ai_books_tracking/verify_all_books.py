
import pandas as pd
import subprocess
import json
import os
import time
from pathlib import Path

def get_metadata(title, author, model="flash"):
    cmd = ["./ai_books_tracking/verify_goodreads.sh", title, author, model]
    try:
        # We use --accept-raw-output-risk inside the script if needed, or here
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        
        # Basic JSON extraction
        start = output.find('{')
        end = output.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(output[start:end])
    except Exception as e:
        print(f"Error for {title}: {e}")
    return None

def verify_dataset(input_path, output_path, title_col='title', author_col='author', rating_col='gb_average_rating'):
    df = pd.read_csv(input_path)
    
    # Check if we have previous verified results to resume
    if os.path.exists(output_path):
        results_df = pd.read_csv(output_path)
        # Find which books are already done by checking 'original_title' and 'original_author'
        done_keys = set(zip(results_df['original_title'], results_df['original_author'].astype(str)))
        results = results_df.to_dict('records')
        print(f"Resuming {input_path}, {len(done_keys)} already verified.")
    else:
        results = []
        done_keys = set()

    books = df.to_dict('records')
    count = 0
    for book in books:
        title = str(book[title_col])
        # Truncate very long author lists (common in holdout)
        author = str(book[author_col])
        if len(author) > 100:
            author = author[:100] + "..."
            
        if (title, author) in done_keys:
            continue
            
        print(f"[{count}/{len(books)}] Verifying: {title} by {author}...")
        
        data = get_metadata(title, author)
        if data:
            data['original_title'] = title
            data['original_author'] = author
            data['original_rating'] = book.get(rating_col, None)
            results.append(data)
            
            # Save progress every book (to be safer)
            pd.DataFrame(results).to_csv(output_path, index=False)
        
        count += 1
        # Simple rate limit
        time.sleep(0.5)

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    return final_df

def main():
    # Verify Holdout (new books)
    print("--- Verifying Holdout Dataset (first 30) ---")
    holdout_input = Path('data/Books Read and their effects - new_books_to_rate 2026.csv')
    holdout_output = Path('ai_books_tracking/verified_holdout.csv')
    # verify_dataset(holdout_input, holdout_output, rating_col='Enjoyment (/5)') # Use enjoyment as placeholder if no rating
    df_holdout = pd.read_csv(holdout_input).head(30)
    # create a temporary input file to use the existing function
    temp_holdout = Path('data/temp_holdout_subset.csv')
    df_holdout.to_csv(temp_holdout, index=False)
    verify_dataset(temp_holdout, holdout_output, rating_col='Enjoyment (/5)')

    # Verify Training (existing books)
    print("\n--- Verifying Training Dataset (first 30) ---")
    train_input = Path('ai_books_tracking/books_enriched.csv')
    train_output = Path('ai_books_tracking/verified_training.csv')
    df_train = pd.read_csv(train_input).head(30)
    temp_train = Path('data/temp_train_subset.csv')
    df_train.to_csv(temp_train, index=False)
    verify_dataset(temp_train, train_output, rating_col='gb_average_rating')

    print("\nAll verifications complete.")

if __name__ == "__main__":
    main()
