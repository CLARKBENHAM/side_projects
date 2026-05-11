import json
import subprocess
import time
from pathlib import Path

import pandas as pd


def get_batch_metadata(books, model="flash"):
    # Format the batch for the prompt
    # We now explicitly include more context to help the LLM avoid common traps
    books_str = "\n".join(
        [
            f"- Title: {b['title']} | Author: {b['author']} | Filename: {b.get('filename', 'N/A')}"
            for b in books
        ]
    )

    prompt = f"""You are a book metadata expert. Find the EXACT Goodreads information for this list of books.
Some of these might be PDFs or generic titles; use the Author and Filename to ensure you find the right version.

BOOKS TO VERIFY:
{books_str}

For EACH book, return a JSON object in a list with these keys:
- original_title: The title I provided
- original_author: The author I provided
- goodreads_url: The canonical URL for the book (null if not on Goodreads)
- goodreads_title: The exact title on Goodreads
- goodreads_author: The primary author on Goodreads
- goodreads_rating: The current average rating (numeric, null if not found)
- goodreads_rating_count: The number of ratings (numeric, null if not found)
- confidence: A score from 0 to 1 on how sure you are this is the correct book

Return ONLY a JSON list of objects.
"""

    # Call gemini CLI
    cmd = ["gemini", "-m", model, "-p", prompt, "--raw-output", "--accept-raw-output-risk"]

    # Fast retry logic (up to 3 times)
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(
                    f"  Attempt {attempt+1} failed with return code {result.returncode}."
                    " Retrying..."
                )
                continue

            output = result.stdout
            # Find JSON array
            start = output.find("[")
            end = output.rfind("]") + 1
            if start != -1 and end != 0:
                return json.loads(output[start:end])
            else:
                print(f"  Attempt {attempt+1} - No JSON found in output. Retrying...")
        except subprocess.TimeoutExpired:
            print(f"  Attempt {attempt+1} - Timeout. Retrying...")
        except Exception as e:
            print(f"  Attempt {attempt+1} - Error: {e}. Retrying...")

        time.sleep(2)  # Wait before retry

    return None


def main():
    # INPUT: The user-cleaned golden master
    golden_path = Path("data/Books Read and their effects - master_book_metadata_cleaned.csv")
    output_path = Path("ai_books_tracking/FINAL_VERIFIED_METADATA.csv")

    if not golden_path.exists():
        print(f"Error: {golden_path} not found.")
        return

    df_gold = pd.read_csv(golden_path)
    # Mapping to standard names
    df_gold = df_gold.rename(columns={"corrected_author": "author"})

    all_books = df_gold[["title", "author", "filename"]].to_dict("records")

    # Load existing to resume
    if output_path.exists():
        results_df = pd.read_csv(output_path)
        results = results_df.to_dict("records")
        done_keys = set(
            zip(results_df["original_title"].astype(str), results_df["original_author"].astype(str))
        )
        print(f"Resuming: {len(done_keys)} books already verified.")
    else:
        results = []
        done_keys = set()

    # Filter for pending
    pending = [b for b in all_books if (str(b["title"]), str(b["author"])) not in done_keys]
    print(f"Total pending books: {len(pending)}")

    # Process in batches of 5 for stability
    batch_size = 5
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(pending)-1)//batch_size + 1}...")

        batch_data = get_batch_metadata(batch)
        if batch_data:
            results.extend(batch_data)
            # Save every batch
            pd.DataFrame(results).to_csv(output_path, index=False)
        else:
            print("  Failed to process batch after retries. Moving to next.")

        time.sleep(1)

    print(f"\nVerification complete. Results in {output_path}")


if __name__ == "__main__":
    main()
