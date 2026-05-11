"""Add author names to the new_books_to_rate 2026 CSV using already-resolved Goodreads data.

Reads canonical_author from the enriched CSV and writes it back to the source
CSV the user labels, without touching any other columns.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent

SOURCE_CSV = DATA_DIR / "Books Read and their effects - new_books_to_rate 2026.csv"
ENRICHED_CSV = OUTPUT_DIR / "new_books_to_rate_2026_enriched.csv"

# Books without Goodreads matches that need manual author assignment
MANUAL_AUTHORS: dict[str, str] = {
    "the history of the english speaking peoples abriged": "Winston S. Churchill",
}

# Columns to check for author, in priority order
AUTHOR_COLUMNS = ["canonical_author", "goodreads_author", "author"]


def _title_case_author(name: str) -> str:
    """Title-case an author name, preserving particles like 'de', 'von', etc."""
    parts = name.split()
    result = []
    for i, part in enumerate(parts):
        # Preserve already-capitalized internal particles
        if i > 0 and part.lower() in ("de", "von", "van", "di", "le", "la", "du"):
            result.append(part.lower())
        elif "'" in part:
            # Handle O'Brien, McDonald-style names: capitalize after apostrophe
            sub = part.split("'")
            result.append("'".join(s.capitalize() for s in sub))
        elif part.startswith("Mc") and len(part) > 2:
            result.append("Mc" + part[2:].capitalize())
        else:
            result.append(part.capitalize())
    return " ".join(result)


def build_author_map(enriched: pd.DataFrame) -> dict[str, str]:
    """Build lowercase-title -> display-author mapping from enriched data."""
    author_map: dict[str, str] = {}
    for _, row in enriched.iterrows():
        title_key = str(row["title"]).strip().lower()
        for col in AUTHOR_COLUMNS:
            if col in enriched.columns and pd.notna(row.get(col)):
                val = str(row[col]).strip()
                if val:
                    author_map[title_key] = _title_case_author(val)
                    break
    # Layer manual overrides on top (setdefault: only if not already found)
    for title_key, author in MANUAL_AUTHORS.items():
        author_map.setdefault(title_key, author)
    return author_map


def main() -> None:
    source = pd.read_csv(SOURCE_CSV)
    enriched = pd.read_csv(ENRICHED_CSV)
    author_map = build_author_map(enriched)

    authors: list[str] = []
    for _, row in source.iterrows():
        key = str(row["title"]).strip().lower()
        authors.append(author_map.get(key, ""))

    matched = sum(1 for a in authors if a)

    if "author" in source.columns:
        source["author"] = authors
    else:
        source.insert(1, "author", authors)

    source.to_csv(SOURCE_CSV, index=False)
    print(f"Added authors to {SOURCE_CSV.name}: {matched}/{len(source)} books matched")

    missing = [row["title"] for (_, row), a in zip(source.iterrows(), authors) if not a]
    if missing:
        print(f"\nMissing authors ({len(missing)}):")
        for t in missing:
            print(f"  {t}")


if __name__ == "__main__":
    main()
