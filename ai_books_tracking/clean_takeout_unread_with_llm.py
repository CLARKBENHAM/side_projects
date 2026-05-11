import pandas as pd
import subprocess
import json
import time
from pathlib import Path


def get_corrected_metadata(title, author, filename, model="flash"):
    prompt = f"""You are a book metadata expert. I have a list of books with potentially noisy titles and missing or incorrect authors.
Your task is to provide the CORRECT primary author and a CLEAN searchable title for each book.

Title: {title}
Original Author: {author}
Filename: {filename}

Instructions:
1. If the author is obviously missing, "by", "Unknown", reversed, or a filename fragment, ignore it and infer the real author.
2. Look at the filename for clues (e.g. "Title-Author.epub").
3. Clean the title by removing filename junk, extensions, OCR artifacts, ISBN fragments, and redundant author/publisher text.
4. Keep the actual book identity in the title; keep meaningful series info only when it helps identify the book.
5. If you are 100% sure of the title/author, provide them.
6. If you are somewhat unsure, return the safest searchable title and best guess author from the filename/title.
7. If you really don't know the author, return "Unknown".

Return ONLY a JSON object:
{{
  "cleaned_title": "Book Title",
  "cleaned_author": "Author Name"
}}
"""
    cmd = ["gemini", "-m", model, "-p", prompt, "--raw-output", "--accept-raw-output-risk"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout
        start = output.find("{")
        end = output.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(output[start:end])
    except Exception as e:
        print(f"Error for {title}: {e}")
    return None


def main():
    input_path = Path("ai_books_tracking/takeout_play_books_03_16_25_unread_for_external_search.csv")
    output_path = Path(
        "ai_books_tracking/takeout_play_books_03_16_25_unread_for_external_search_cleaned.csv"
    )

    df = pd.read_csv(input_path)

    if output_path.exists():
        results_df = pd.read_csv(output_path)
        done_keys = set(
            zip(
                results_df["raw_title"].astype(str),
                results_df["raw_author"].astype(str),
                results_df["filename"].astype(str),
            )
        )
        results = results_df.to_dict("records")
        print(f"Resuming: {len(done_keys)} already cleaned.")
    else:
        results = []
        done_keys = set()

    books = df.to_dict("records")

    batch_size = 10
    for i in range(0, len(books), batch_size):
        batch = books[i : i + batch_size]

        if all(
            (
                str(b.get("title", "")),
                str(b.get("author", "")),
                str(b.get("filename", "")),
            )
            in done_keys
            for b in batch
        ):
            continue

        print(f"Processing batch {i // batch_size + 1}...")

        for book in batch:
            key = (
                str(book.get("title", "")),
                str(book.get("author", "")),
                str(book.get("filename", "")),
            )
            if key in done_keys:
                continue

            print(f"  Cleaning: {book.get('title', '')}...")
            data = get_corrected_metadata(
                book.get("title", ""),
                book.get("author", ""),
                book.get("filename", ""),
            )

            cleaned_title = str(book.get("title", "")).strip()
            cleaned_author = str(book.get("author", "")).strip()
            if data:
                cleaned_title = str(data.get("cleaned_title", cleaned_title)).strip()
                cleaned_author = str(data.get("cleaned_author", cleaned_author)).strip()

            out_book = dict(book)
            out_book["raw_title"] = book.get("title", "")
            out_book["raw_author"] = book.get("author", "")
            out_book["cleaned_title"] = cleaned_title
            out_book["cleaned_author"] = cleaned_author
            results.append(out_book)
            done_keys.add(key)

            time.sleep(0.5)

        pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nCleanup complete. Results in {output_path}")


if __name__ == "__main__":
    main()
