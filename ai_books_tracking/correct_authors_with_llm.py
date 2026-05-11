
import pandas as pd
import subprocess
import json
import os
import time
from pathlib import Path

def get_corrected_author(title, author, filename, model="flash"):
    prompt = f"""You are a book metadata expert. I have a list of books with potentially incorrect or placeholder authors.
Your task is to provide the CORRECT primary author for each book.

Title: {title}
Original Author: {author}
Filename: {filename}

Instructions:
1. If the author is obviously a date (like 2021-01-19) or just "by", "Unknown", or a filename fragment, ignore it.
2. Look at the filename for clues (e.g. "Title-Author.epub").
3. Use your knowledge of famous books to correct common mistakes (e.g. "You Just Don't Understand" is Deborah Tannen).
4. If you are 100% sure of the author, provide it.
5. If you are not sure, but the filename has a name, use that.
6. If you really don't know, return "Unknown".

Return ONLY a JSON object:
{{
  "corrected_author": "Author Name"
}}
"""
    cmd = ["gemini", "-m", model, "-p", prompt, "--raw-output", "--accept-raw-output-risk"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        start = output.find('{')
        end = output.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(output[start:end])
    except Exception as e:
        print(f"Error for {title}: {e}")
    return None

def main():
    input_path = Path('ai_books_tracking/master_book_metadata_raw.csv')
    output_path = Path('ai_books_tracking/master_book_metadata_cleaned.csv')
    
    df = pd.read_csv(input_path)
    
    if output_path.exists():
        results_df = pd.read_csv(output_path)
        done_keys = set(zip(results_df['title'].astype(str), results_df['author'].astype(str)))
        results = results_df.to_dict('records')
        print(f"Resuming: {len(done_keys)} already cleaned.")
    else:
        results = []
        done_keys = set()

    books = df.to_dict('records')
    
    # Process in batches of 10 to be efficient
    batch_size = 10
    for i in range(0, len(books), batch_size):
        batch = books[i:i+batch_size]
        
        # Check if batch is already done
        if all((str(b['title']), str(b['author'])) in done_keys for b in batch):
            continue
            
        print(f"Processing batch {i//batch_size + 1}...")
        
        # We'll do them sequentially for simplicity in this specific cleanup script
        for book in batch:
            if (str(book['title']), str(book['author'])) in done_keys:
                continue
                
            print(f"  Cleaning: {book['title']}...")
            data = get_corrected_author(book['title'], book['author'], book['filename'])
            if data:
                book['corrected_author'] = data['corrected_author']
                results.append(book)
                done_keys.add((str(book['title']), str(book['author'])))
            
            time.sleep(0.5)
            
        # Save progress
        pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nCleanup complete. Results in {output_path}")

if __name__ == "__main__":
    main()
