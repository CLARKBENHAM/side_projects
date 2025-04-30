# %%
import os
import re
import csv
import glob
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path


def extract_book_info(html_file):
    """Extract book information from Google Play Books HTML export."""
    with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    # Check if book is finished
    finished_text = soup.select_one(".meta-entry")
    if not finished_text or "finished this book" not in finished_text.text:
        return None  # Skip books that are not finished

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
        # Start from the h2 element and look for following siblings
        current_element = target_h2.next_sibling
        while current_element:
            # Check if it's a div
            if current_element.name == "div":
                # Add the text content of this div
                if current_element.get_text(strip=True) not in multi_shelves:
                    bookshelf = current_element.get_text(strip=True)
                    break
            elif current_element.name is not None and current_element.name != "div":
                # If we encounter any non-div element (apart from navigable strings), break
                break

            # Move to the next element
            current_element = current_element.next_sibling

    # Extract modification timestamps
    timestamps = []
    for date_elem in soup.select(".last-modified-date"):
        date_text = date_elem.text
        match = re.search(r"Last modified on\s+(.*?)\s+Pacific Time", date_text)
        if match:
            date_str = match.group(1)
            try:
                # Parse the date string
                date_obj = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
                timestamps.append(date_obj)
            except ValueError as e:
                print("Skipping. ", e)
                continue
        else:
            print("didn't finished", html_file)

    earliest_timestamp = min(timestamps) if timestamps else None
    latest_timestamp = max(timestamps) if timestamps else None
    return {
        "title": title,
        "author": author,
        "bookshelf": bookshelf,
        "earliest_modified": (
            earliest_timestamp.strftime("%Y-%m-%d %H:%M:%S") if earliest_timestamp else ""
        ),
        "latest_modified": (
            latest_timestamp.strftime("%Y-%m-%d %H:%M:%S") if latest_timestamp else ""
        ),
        "filename": os.path.basename(html_file),
    }


def process_google_books_exports(directory_pattern):
    """Process all Google Play Books HTML exports (and mis-labeled small PDFs) recursively."""
    results = []

    matching_dirs = glob.glob(directory_pattern)
    if not matching_dirs:
        print(f"No directories matching '{directory_pattern}' found")
        return results

    for directory in matching_dirs:
        for root, dirs, files in os.walk(directory):
            # collect .html exports plus any .pdf < 500 KB
            # google had bug where those files were improperly labeled .pdf not .html
            # in cases where there's no annotations
            exports = []
            for fname in files:
                path = os.path.join(root, fname)
                if fname.lower().endswith(".html"):
                    exports.append(path)
                elif fname.lower().endswith(".pdf") and os.path.getsize(path) < 500 * 1024:
                    exports.append(path)

            for export_file in exports:
                book_info = extract_book_info(export_file)
                if book_info:  # Only include finished books
                    results.append(book_info)
    return results


def save_to_csv(
    books, output_file=os.path.join(os.path.expanduser("~"), "Downloads", "finished_books.csv")
):
    """Save the extracted book information to a CSV file."""
    if not books:
        print("No finished books found.")
        return

    fieldnames = [
        "title",
        "author",
        "bookshelf",
        "earliest_modified",
        "latest_modified",
        "filename",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: re.sub("\s+", " ", v) for k, v in d.items()} for d in books])

    print(f"Found {len(books)} finished books. Information saved to {output_file}")


def main():
    # Use the home directory and expand the pattern
    home_dir = os.path.expanduser("~")
    pattern = os.path.join(home_dir, "Downloads", "Play Books Takeout*")

    books = process_google_books_exports(pattern)
    save_to_csv(books)


if __name__ == "__main__":
    main()


def f():
    """Extract info from google inspect page directly; but note it mixes up titles and authors:

    (() => {
      const csv = prompt('Paste your CSV (including header) here');
      if (!csv) return;
      const lines = csv.trim().split('\n').filter(Boolean);
      const header = lines[0].match(/(".*?"|[^",]+)(?=\s*,|\s*$)/g);
      const titleIdx = header.indexOf('title');
      const csvTitles = new Set(
        lines.slice(1).map(line => {
          const cols = line.match(/(".*?"|[^",]+)(?=\s*,|\s*$)/g);
          return cols[titleIdx].replace(/^"|"$/g, '').trim();
        })
      );
      const uiBooks = Array.from(document.querySelectorAll('a.title')).map((t, i) => {
        const title = t.getAttribute('title').trim();
        const authorEl = document.querySelectorAll('a.author.ng-star-inserted')[i];
        const author = authorEl?.getAttribute('title').trim() || '';
        return { title, author };
      });
      const missing = uiBooks.filter(b => !csvTitles.has(b.title));
      console.table(missing);
      copy(JSON.stringify(missing));
      console.log(`Copied ${missing.length} missing books as JSON`);
    })();
    """
    pass
