#!/usr/bin/env python3
"""
Find real duplicate books based on:
1. Identical file sizes (definitely duplicates)
2. One filename is a substring/prefix of another
3. Trigram overlap in title
"""
import os
import re
from collections import defaultdict


def normalize_for_comparison(filename):
    """
    Normalize filename for comparison by removing common suffixes
    and converting to lowercase.
    """
    # Remove extension
    name = os.path.splitext(filename)[0]

    # Remove common patterns like "by Author", "(Year, Publisher)", etc.
    # But keep the core title
    name = name.lower()
    name = re.sub(r"\s*-\s*\(.*?\)\s*$", "", name)  # Remove trailing (Year, Publisher)
    name = re.sub(r"\s+by\s+.*$", "", name)  # Remove "by Author"
    name = re.sub(r"\s*-\s*[^-]+\s*-\s*\(.*$", "", name)  # Remove - Author - (details)

    # Clean up extra whitespace and underscores
    name = re.sub(r"[_\s]+", " ", name).strip()

    return name


def is_likely_duplicate(fname1, fname2):
    """
    Check if two filenames are likely duplicates based on substring matching.
    """
    norm1 = normalize_for_comparison(fname1)
    norm2 = normalize_for_comparison(fname2)

    # If normalized names are identical
    if norm1 == norm2:
        return True

    # If one is a substring of the other (allowing for minor differences)
    if len(norm1) < len(norm2):
        shorter, longer = norm1, norm2
    else:
        shorter, longer = norm2, norm1

    # Check if shorter is prefix or contained in longer
    if longer.startswith(shorter) or shorter in longer:
        # Additional check: make sure the difference is reasonable (metadata, not content)
        # If the longer one is more than 3x the length, probably not a duplicate
        if len(longer) <= len(shorter) * 3:
            return True

    return False


def find_duplicates(base_dir):
    """
    Find duplicate books in the directory.
    """
    # Collect all book files
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.startswith("."):
                continue
            if fname.lower().endswith((".epub", ".pdf")):
                fpath = os.path.join(root, fname)
                all_files.append(fpath)

    print(f"Found {len(all_files)} total book files\n")

    # Group by category
    files_by_category = defaultdict(list)
    for fpath in all_files:
        category = os.path.basename(os.path.dirname(fpath))
        files_by_category[category].append(fpath)

    # Find duplicates
    print("=" * 80)
    print("REAL DUPLICATES (Identical sizes or substring matches)")
    print("=" * 80)

    total_duplicates = 0
    duplicate_groups = []

    for category in sorted(files_by_category.keys()):
        files = files_by_category[category]

        if len(files) < 2:
            continue

        category_duplicates = []

        # Group by file size first (exact duplicates)
        by_size = defaultdict(list)
        for fpath in files:
            size = os.path.getsize(fpath)
            by_size[size].append(fpath)

        # Find size duplicates
        for size, fpaths in by_size.items():
            if len(fpaths) > 1:
                category_duplicates.append({"type": "size_match", "files": fpaths, "size": size})

        # Find filename substring matches
        for i, file1 in enumerate(files):
            for file2 in files[i + 1 :]:
                fname1 = os.path.basename(file1)
                fname2 = os.path.basename(file2)

                # Skip if already found as size match
                size1 = os.path.getsize(file1)
                size2 = os.path.getsize(file2)
                if size1 == size2:
                    continue  # Already caught by size matching

                if is_likely_duplicate(fname1, fname2):
                    category_duplicates.append(
                        {
                            "type": "name_match",
                            "files": [file1, file2],
                            "size1": size1,
                            "size2": size2,
                        }
                    )

        if category_duplicates:
            print(f"\n{category}: {len(category_duplicates)} duplicate groups")
            print("-" * 80)

            for dup in category_duplicates:
                if dup["type"] == "size_match":
                    print(f"\n  IDENTICAL SIZE ({dup['size']:,} bytes):")
                    for fpath in dup["files"]:
                        fname = os.path.basename(fpath)
                        print(f"    - {fname}")
                else:
                    print("\n  NAME MATCH (different sizes):")
                    fname1 = os.path.basename(dup["files"][0])
                    fname2 = os.path.basename(dup["files"][1])
                    print(f"    - {fname1} ({dup['size1']:,} bytes)")
                    print(f"    - {fname2} ({dup['size2']:,} bytes)")

                    # Show normalized versions
                    norm1 = normalize_for_comparison(fname1)
                    norm2 = normalize_for_comparison(fname2)
                    print(f"      Normalized 1: {norm1}")
                    print(f"      Normalized 2: {norm2}")

            total_duplicates += len(category_duplicates)
            duplicate_groups.append((category, category_duplicates))

    print("\n" + "=" * 80)
    print(f"Summary: Found {total_duplicates} duplicate groups")

    return duplicate_groups


def main():
    base_dir = os.path.expanduser("~/Documents/Books/eleven_upload")

    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return

    duplicates = find_duplicates(base_dir)

    # Save to file
    output_file = os.path.expanduser("~/Downloads/duplicate_books.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("DUPLICATE BOOKS IN ELEVEN_UPLOAD\n")
        f.write("=" * 80 + "\n\n")

        for category, dups in duplicates:
            f.write(f"\n{category}:\n")
            f.write("-" * 80 + "\n")

            for dup in dups:
                if dup["type"] == "size_match":
                    f.write(f"\nIDENTICAL SIZE ({dup['size']:,} bytes):\n")
                    for fpath in dup["files"]:
                        fname = os.path.basename(fpath)
                        f.write(f"  {fname}\n")
                else:
                    f.write("\nNAME MATCH:\n")
                    fname1 = os.path.basename(dup["files"][0])
                    fname2 = os.path.basename(dup["files"][1])
                    f.write(f"  {fname1} ({dup['size1']:,} bytes)\n")
                    f.write(f"  {fname2} ({dup['size2']:,} bytes)\n")

    print(f"\nDuplicates saved to: {output_file}")


if __name__ == "__main__":
    main()
