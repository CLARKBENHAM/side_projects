#!/usr/bin/env python3
"""
Match local books in ~/Documents/Books with categorizations from CSV
and copy them to eleven_upload folders.
"""
import argparse
import csv
import os
import re
import shutil
from difflib import SequenceMatcher


def normalize_text(text):
    """Normalize text for comparison by removing punctuation and lowercasing."""
    if not text:
        return ""
    # Remove common punctuation and extra spaces
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_title_author_from_filename(filename):
    """
    Extract likely title and author from filename.
    Common patterns:
    - Title - Author - (Year, Publisher).ext
    - Title by Author.ext
    - Title_ Author - (Year, Publisher).ext
    """
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Try pattern: "Title - Author - (Year, Publisher)"
    match = re.match(r"^(.+?)\s*-\s*(.+?)\s*-\s*\(", name)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Try pattern: "Title by Author"
    match = re.match(r"^(.+?)\s+by\s+(.+?)(?:\s*\(|$)", name)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Try pattern: "Title - Author"
    match = re.match(r"^(.+?)\s*-\s*(.+?)(?:\s*\(|$)", name)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Try pattern: "Title_ Author"
    match = re.match(r"^(.+?)_\s*(.+?)(?:\s*-\s*\(|$)", name)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # If no pattern matches, return the whole name as title
    return name.strip(), ""


def similarity_score(str1, str2):
    """Calculate similarity between two strings (0-1)."""
    norm1 = normalize_text(str1)
    norm2 = normalize_text(str2)

    if not norm1 or not norm2:
        return 0.0

    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, norm1, norm2).ratio()


def find_best_match(local_title, local_author, csv_books):
    """
    Find the best matching book from CSV based on title and author.
    Returns (best_match_dict, score) or (None, 0) if no good match.
    """
    best_match = None
    best_score = 0.0

    for book in csv_books:
        csv_title = book["title"]
        csv_author = book["author"]

        # Calculate title similarity
        title_sim = similarity_score(local_title, csv_title)

        # Calculate author similarity
        author_sim = similarity_score(local_author, csv_author) if local_author else 0.5

        # Combined score: weight title more heavily (70% title, 30% author)
        combined_score = 0.7 * title_sim + 0.3 * author_sim

        if combined_score > best_score:
            best_score = combined_score
            best_match = book

    return best_match, best_score


def sanitize_folder_name(name):
    """Sanitize folder name by removing invalid characters."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
    sanitized = sanitized.strip(". ")
    return sanitized if sanitized else "Uncategorized"


def load_csv_books(csv_path):
    """Load books from CSV file."""
    books = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            books.append(row)
    return books


def organize_local_books(books_dir, csv_path, dest_dir, min_score=0.5, dry_run=True):
    """
    Match local books with CSV categorizations and copy to eleven_upload.

    Args:
        books_dir: Directory containing local book files
        csv_path: Path to CSV with book categorizations
        dest_dir: Destination directory for organized books
        min_score: Minimum similarity score to accept a match (0-1)
        dry_run: If True, only print what would be done
    """
    # Load CSV books
    csv_books = load_csv_books(csv_path)
    print(f"Loaded {len(csv_books)} books from CSV")

    # Find all book files
    book_files = []
    for fname in os.listdir(books_dir):
        if fname.startswith("."):
            continue
        fpath = os.path.join(books_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".epub", ".pdf")):
            book_files.append(fpath)

    print(f"Found {len(book_files)} book files in {books_dir}\n")

    # Match and categorize
    matched = []
    unmatched = []

    for book_path in sorted(book_files):
        filename = os.path.basename(book_path)
        local_title, local_author = extract_title_author_from_filename(filename)

        # Find best match
        best_match, score = find_best_match(local_title, local_author, csv_books)

        if score >= min_score and best_match:
            matched.append(
                {
                    "local_path": book_path,
                    "local_title": local_title,
                    "local_author": local_author,
                    "csv_title": best_match["title"],
                    "csv_author": best_match["author"],
                    "category": best_match["bookshelf"],
                    "score": score,
                }
            )
        else:
            unmatched.append(
                {
                    "local_path": book_path,
                    "local_title": local_title,
                    "local_author": local_author,
                    "best_guess": best_match["title"] if best_match else "None",
                    "score": score,
                }
            )

    # Print results
    print(f"Matched: {len(matched)} books")
    print(f"Unmatched: {len(unmatched)} books (score < {min_score})\n")

    # Group by category
    by_category = {}
    for book in matched:
        category = book["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(book)

    # Copy books
    total_copied = 0
    for category in sorted(by_category.keys()):
        books = by_category[category]
        safe_category = sanitize_folder_name(category)
        dest_category_dir = os.path.join(dest_dir, safe_category)

        print(f"\nCategory: {category} ({len(books)} books)")

        if not dry_run:
            os.makedirs(dest_category_dir, exist_ok=True)

        for book in books:
            src_file = book["local_path"]
            dest_file = os.path.join(dest_category_dir, os.path.basename(src_file))

            if dry_run:
                print(f"  Would copy: {book['local_title']}")
                print(f"    Matched with: {book['csv_title']} (score: {book['score']:.2f})")
                print(f"    From: {src_file}")
                print(f"    To: {dest_file}")
            else:
                try:
                    shutil.copy2(src_file, dest_file)
                    print(
                        f"  Copied: {book['local_title']} -> {category} (score: {book['score']:.2f})"
                    )
                    total_copied += 1
                except Exception as e:
                    print(f"  Error copying {book['local_title']}: {e}")

    # Report unmatched
    if unmatched:
        print(f"\n{'='*60}")
        print(f"UNMATCHED BOOKS ({len(unmatched)}):")
        print(f"{'='*60}")
        for book in unmatched[:20]:  # Show first 20
            print(f"\n  Local: {book['local_title']}")
            if book["local_author"]:
                print(f"  Author: {book['local_author']}")
            print(f"  Best guess: {book['best_guess']} (score: {book['score']:.2f})")
        if len(unmatched) > 20:
            print(f"\n  ... and {len(unmatched) - 20} more")

    if not dry_run:
        print(f"\nTotal books copied: {total_copied}")
    else:
        print(f"\nDry run complete. Would copy {len(matched)} books.")
        print("Run with --copy to actually copy the files.")


def main():
    parser = argparse.ArgumentParser(
        description="Match local books with CSV categorizations and copy to eleven_upload"
    )
    parser.add_argument(
        "--copy", action="store_true", help="Actually copy the files (default is dry run)"
    )
    parser.add_argument(
        "--books-dir",
        default="~/Documents/Books",
        help="Directory containing local book files",
    )
    parser.add_argument(
        "--csv",
        default="~/Downloads/unfinished_books.csv",
        help="CSV file with book categorizations",
    )
    parser.add_argument(
        "--dest",
        default="~/Documents/Books/eleven_upload",
        help="Destination directory for organized books",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum similarity score to accept a match (0-1, default: 0.5)",
    )
    args = parser.parse_args()

    books_dir = os.path.expanduser(args.books_dir)
    csv_path = os.path.expanduser(args.csv)
    dest_dir = os.path.expanduser(args.dest)

    if not os.path.exists(books_dir):
        print(f"Error: Books directory not found: {books_dir}")
        return

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    if not args.copy:
        print("=== Dry Run ===")
        organize_local_books(books_dir, csv_path, dest_dir, args.min_score, dry_run=True)
        print("\nDry run complete. Use --copy to actually copy the files.")
    else:
        print("=== Copying Books ===")
        organize_local_books(books_dir, csv_path, dest_dir, args.min_score, dry_run=False)
        print("\nDone!")


if __name__ == "__main__":
    main()
