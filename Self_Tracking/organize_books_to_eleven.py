#!/usr/bin/env python3
"""
Organize unfinished books from Google Play Books Takeout into categorized folders
for the Eleven Reader.
"""
import argparse
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

from bs4 import BeautifulSoup


def extract_book_info(html_file):
    """Extract book information from Google Play Books HTML export."""
    with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    finished = True
    finished_text = soup.select_one(".meta-entry")
    if not finished_text or "finished this book" not in finished_text.text:
        finished = False

    # Extract title
    title_elem = soup.select_one("h1")
    title = title_elem.text.strip() if title_elem else "Unknown Title"

    # Extract author
    author_elem = soup.select_one(".author")
    author = (
        author_elem.text.strip().replace("by\n", "").strip() if author_elem else "Unknown Author"
    )

    # Extract bookshelf
    multi_shelves = ("Great Books", "Bad Books")  # don't want this as single category
    bookshelf = "Unknown Shelf"
    target_h2 = soup.find("h2", string="Custom shelves with this book")
    if target_h2:
        current_element = target_h2.next_sibling
        while current_element:
            if current_element.name == "div":
                if current_element.get_text(strip=True) not in multi_shelves:
                    bookshelf = current_element.get_text(strip=True)
                    break
            elif current_element.name is not None and current_element.name != "div":
                break
            current_element = current_element.next_sibling

    # Extract modification timestamps
    timestamps = []
    for date_elem in soup.select(".last-modified-date"):
        date_text = date_elem.text
        match = re.search(r"Last modified on\s+(.*?)\s+Pacific Time", date_text)
        if match:
            date_str = match.group(1)
            try:
                date_obj = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
                timestamps.append(date_obj)
            except ValueError:
                continue

    latest_timestamp = max(timestamps) if timestamps else None

    return {
        "title": title,
        "author": author,
        "bookshelf": bookshelf,
        "latest_modified": latest_timestamp,
        "finished": finished,
        "html_file": html_file,
    }


def find_book_file(book_dir, min_size_kb=300):
    """Find the actual book file (epub or pdf) in the directory."""
    for fname in os.listdir(book_dir):
        fpath = os.path.join(book_dir, fname)
        if not os.path.isfile(fpath):
            continue

        if fname.lower().endswith((".epub", ".pdf")):
            size_kb = os.path.getsize(fpath) / 1024
            if fname.lower().endswith(".epub") or size_kb >= min_size_kb:
                return fpath
    return None


def normalize_name(name):
    """Normalize book title or author for comparison."""
    # Remove non-alphanumeric, convert to lowercase
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized


def is_similar(name1, name2, threshold=0.8):
    """Check if two names are similar based on string overlap."""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)

    if not norm1 or not norm2:
        return False

    # Simple substring check
    if norm1 in norm2 or norm2 in norm1:
        return True

    # Check character overlap
    set1 = set(norm1)
    set2 = set(norm2)
    overlap = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return (overlap / union) >= threshold if union > 0 else False


def sanitize_folder_name(name):
    """Sanitize folder name by removing invalid characters."""
    # Remove or replace invalid characters for folder names
    sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
    sanitized = sanitized.strip(". ")
    return sanitized if sanitized else "Uncategorized"


def organize_books(source_dir, dest_base_dir, dry_run=True):
    """
    Organize unfinished books from Google Play Books Takeout into categorized folders.

    Args:
        source_dir: Path to 'Google Play Books' directory in takeout
        dest_base_dir: Path to '~/Documents/Books/eleven_upload'
        dry_run: If True, only print what would be done without copying
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    # Collect all unfinished books with their metadata
    unfinished_books = []
    book_dirs = [
        d
        for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith(".")
    ]

    print(f"Scanning {len(book_dirs)} book directories...")

    for book_dir_name in book_dirs:
        book_dir = os.path.join(source_dir, book_dir_name)

        # Find HTML file
        html_files = [f for f in os.listdir(book_dir) if f.endswith(".html")]
        if not html_files:
            continue

        html_file = os.path.join(book_dir, html_files[0])

        # Extract metadata
        try:
            info = extract_book_info(html_file)
        except Exception as e:
            print(f"Warning: Failed to extract info from {html_file}: {e}")
            continue

        # Skip finished books
        if info["finished"]:
            continue

        # Find the actual book file
        book_file = find_book_file(book_dir)
        if not book_file:
            print(f"Warning: No book file found in {book_dir}")
            continue

        info["book_file"] = book_file
        unfinished_books.append(info)

    print(f"Found {len(unfinished_books)} unfinished books")

    # Group by normalized title+author to detect duplicates
    book_groups = defaultdict(list)
    for book in unfinished_books:
        key = (normalize_name(book["title"]), normalize_name(book["author"]))
        book_groups[key].append(book)

    # For each group, take the latest one
    books_to_copy = []
    for key, books in book_groups.items():
        if len(books) > 1:
            # Sort by latest modified timestamp
            books_sorted = sorted(
                books,
                key=lambda x: x["latest_modified"] if x["latest_modified"] else datetime.min,
                reverse=True,
            )
            selected = books_sorted[0]
            print(f"Duplicate found: '{selected['title']}' - selecting latest version")
            books_to_copy.append(selected)
        else:
            books_to_copy.append(books[0])

    print(f"After deduplication: {len(books_to_copy)} books to organize")

    # Group by category
    by_category = defaultdict(list)
    for book in books_to_copy:
        category = book["bookshelf"]
        by_category[category].append(book)

    # Copy books to categorized folders
    total_copied = 0
    for category, books in sorted(by_category.items()):
        safe_category = sanitize_folder_name(category)
        dest_dir = os.path.join(dest_base_dir, safe_category)

        print(f"\nCategory: {category} ({len(books)} books)")

        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)

        for book in books:
            src_file = book["book_file"]
            dest_file = os.path.join(dest_dir, os.path.basename(src_file))

            if dry_run:
                print(f"  Would copy: {book['title']}")
                print(f"    From: {src_file}")
                print(f"    To: {dest_file}")
            else:
                try:
                    shutil.copy2(src_file, dest_file)
                    print(f"  Copied: {book['title']}")
                    total_copied += 1
                except Exception as e:
                    print(f"  Error copying {book['title']}: {e}")

    if not dry_run:
        print(f"\nTotal books copied: {total_copied}")
    else:
        print(f"\nDry run complete. Would copy {len(books_to_copy)} books.")
        print("Run with dry_run=False to actually copy the files.")


def main():
    parser = argparse.ArgumentParser(
        description="Organize unfinished books from Google Play Books Takeout"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Actually copy the files (default is dry run)",
    )
    parser.add_argument(
        "--source",
        default="/Users/clarkbenham/Downloads/Play Books Takeout 5/Google Play Books",
        help="Source directory containing Google Play Books",
    )
    parser.add_argument(
        "--dest",
        default="~/Documents/Books/eleven_upload",
        help="Destination directory for organized books",
    )
    args = parser.parse_args()

    source_dir = args.source
    dest_dir = os.path.expanduser(args.dest)

    if not args.copy:
        print("=== Dry Run ===")
        organize_books(source_dir, dest_dir, dry_run=True)
        print("\nDry run complete. Use --copy to actually copy the files.")
    else:
        print("=== Copying Books ===")
        organize_books(source_dir, dest_dir, dry_run=False)
        print("\nDone!")


if __name__ == "__main__":
    main()
