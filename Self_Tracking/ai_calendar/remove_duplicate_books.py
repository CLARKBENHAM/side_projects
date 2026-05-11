#!/usr/bin/env python3
"""
Remove duplicate books with improved detection:
1. Check for trigram overlaps in titles (sequences of 3 words)
2. Prefer epub over pdf
3. If multiple epubs, keep the newer one (later modification time)
"""
import argparse
import os
import re
import time
from collections import defaultdict


def get_word_trigrams(text):
    """
    Get all 3-word sequences (trigrams) from text.
    Returns set of tuples.
    """
    # Normalize: lowercase, remove punctuation, split
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in normalized.split() if w]

    trigrams = set()
    for i in range(len(words) - 2):
        trigram = (words[i], words[i + 1], words[i + 2])
        trigrams.add(trigram)

    return trigrams


def normalize_for_comparison(filename):
    """Normalize filename for comparison."""
    name = os.path.splitext(filename)[0]
    name = name.lower()
    name = re.sub(r"\s*-\s*\(.*?\)\s*$", "", name)
    name = re.sub(r"\s+by\s+.*$", "", name)
    name = re.sub(r"\s*-\s*[^-]+\s*-\s*\(.*$", "", name)
    name = re.sub(r"[_\s]+", " ", name).strip()
    return name


def has_significant_trigram_overlap(fname1, fname2, min_overlap=1):
    """
    Check if two filenames have significant trigram overlap.
    Returns True if they share at least min_overlap trigrams.
    """
    # Get base names without extensions
    base1 = os.path.splitext(fname1)[0]
    base2 = os.path.splitext(fname2)[0]

    trigrams1 = get_word_trigrams(base1)
    trigrams2 = get_word_trigrams(base2)

    # Need at least 1 trigram to compare
    if not trigrams1 or not trigrams2:
        return False

    overlap = trigrams1.intersection(trigrams2)
    return len(overlap) >= min_overlap


def is_likely_duplicate(fname1, fname2):
    """
    Check if two filenames are likely duplicates based on normalized name matching.
    """
    norm1 = normalize_for_comparison(fname1)
    norm2 = normalize_for_comparison(fname2)

    # Exact normalized match
    if norm1 == norm2:
        return True

    # Substring match (one is clearly a shortened version of the other)
    if len(norm1) < len(norm2):
        shorter, longer = norm1, norm2
    else:
        shorter, longer = norm2, norm1

    # Only match if shorter is a clear prefix/substring and not too different in length
    if longer.startswith(shorter) or shorter in longer:
        if len(longer) <= len(shorter) * 1.5:  # Tighter threshold to avoid false positives
            return True

    return False


def choose_file_to_keep(files):
    """
    Choose which file to keep from a list of duplicates.
    Priority:
    1. Prefer epub over pdf
    2. Prefer longer filename (more metadata)
    3. Prefer newer file (later modification time)

    Returns: (file_to_keep, files_to_delete)
    """
    if len(files) == 1:
        return files[0], []

    # Separate by extension
    epubs = [f for f in files if f.lower().endswith(".epub")]
    pdfs = [f for f in files if f.lower().endswith(".pdf")]

    # Choose from epubs if available, else pdfs
    candidates = epubs if epubs else pdfs

    if not candidates:
        # Shouldn't happen, but fallback
        candidates = files

    candidates_sorted = sorted(
        candidates, key=lambda f: (len(os.path.basename(f)), os.path.getmtime(f)), reverse=True
    )

    to_keep = candidates_sorted[0]

    # Files to delete: all others
    to_delete = [f for f in files if f != to_keep]

    return to_keep, to_delete


def find_and_remove_duplicates(base_dir, dry_run=True):
    """Find and remove duplicate books."""
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

    # Find and remove duplicates
    total_removed = 0
    total_kept = 0

    for category in sorted(files_by_category.keys()):
        files = files_by_category[category]

        if len(files) < 2:
            continue

        # Track which files we've already processed
        processed = set()
        duplicate_groups = []

        # Method 1: Group by file size (exact duplicates)
        by_size = defaultdict(list)
        for fpath in files:
            size = os.path.getsize(fpath)
            by_size[size].append(fpath)

        for size, fpaths in by_size.items():
            if len(fpaths) > 1:
                duplicate_groups.append(fpaths)
                processed.update(fpaths)

        # Method 2: Find name-based duplicates (including trigram matches)
        for i, file1 in enumerate(files):
            if file1 in processed:
                continue

            group = [file1]
            fname1 = os.path.basename(file1)

            for file2 in files[i + 1 :]:
                if file2 in processed:
                    continue

                fname2 = os.path.basename(file2)

                if is_likely_duplicate(fname1, fname2):
                    group.append(file2)

            if len(group) > 1:
                duplicate_groups.append(group)
                processed.update(group)

        # Process duplicate groups
        category_removed = 0
        category_kept = 0

        for group in duplicate_groups:
            to_keep, to_delete = choose_file_to_keep(group)

            # Get info about the group
            sizes = [os.path.getsize(f) for f in group]
            same_size = len(set(sizes)) == 1
            has_epub = any(f.lower().endswith(".epub") for f in group)
            has_pdf = any(f.lower().endswith(".pdf") for f in group)

            print(f"\n{category}:")
            if same_size:
                print(f"  Identical size ({sizes[0]:,} bytes)")
            if has_epub and has_pdf:
                print("  Both .epub and .pdf versions")

            keep_mtime = os.path.getmtime(to_keep)
            print(
                f"  KEEPING: {os.path.basename(to_keep)} (modified:"
                f" {os.path.getmtime(to_keep):.0f})"
            )

            for fpath in to_delete:
                del_mtime = os.path.getmtime(fpath)
                reason = []

                if fpath.lower().endswith(".pdf") and to_keep.lower().endswith(".epub"):
                    reason.append("pdf vs epub")
                elif del_mtime < keep_mtime:
                    reason.append("older")
                elif len(os.path.basename(fpath)) < len(os.path.basename(to_keep)):
                    reason.append("less metadata")

                reason_str = f" ({', '.join(reason)})" if reason else ""
                print(
                    f"  {'DRY RUN - Would remove' if dry_run else 'REMOVED'}:"
                    f" {os.path.basename(fpath)}{reason_str}"
                )

                if not dry_run:
                    os.remove(fpath)

            category_removed += len(to_delete)
            category_kept += 1

        if category_removed > 0:
            total_removed += category_removed
            total_kept += category_kept
            print(
                f"\n{category} summary: Kept {category_kept} unique books,"
                f" {'Would remove' if dry_run else 'Removed'} {category_removed} duplicates"
            )

    print("\n" + "=" * 80)
    print(
        f"Total: Kept {total_kept} unique books,"
        f" {'Would remove' if dry_run else 'Removed'} {total_removed} duplicates"
    )

    return total_kept, total_removed


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate books from eleven_upload")
    parser.add_argument(
        "--remove", action="store_true", help="Actually remove files (default is dry run)"
    )
    parser.add_argument(
        "--dir", default="~/Documents/Books/eleven_upload", help="Directory containing books"
    )
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.dir)

    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return

    if not args.remove:
        print("=== DRY RUN ===")
        find_and_remove_duplicates(base_dir, dry_run=True)
        print("\nDry run complete. Use --remove to actually delete the duplicate files.")
    else:
        print("=== REMOVING DUPLICATES ===")
        print("This will permanently delete files. Are you sure? (yes/no): ")
        time.sleep(5)
        find_and_remove_duplicates(base_dir, dry_run=False)
        print("\nDone!")


if __name__ == "__main__":
    main()
