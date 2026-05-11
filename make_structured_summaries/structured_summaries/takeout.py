"""Google Play Takeout cataloging."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .models import BookRecord, make_book_id
from .utils import (
    clean_title_for_search,
    infer_author_from_filename,
    remove_duplicate_suffix,
)

SUPPORTED_EXTENSIONS = {
    ".epub": "epub",
    ".pdf": "pdf",
    ".txt": "txt",
    ".html": "html",
    ".htm": "html",
}
FORMAT_PRIORITY = {"epub": 0, "pdf": 1, "txt": 2, "html": 3}


def discover_candidate_files(book_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in book_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def display_title_from_dir(book_dir: Path) -> str:
    title = book_dir.name.replace("_", " ").strip()
    return " ".join(title.split())


def _file_rank(path: Path) -> tuple[int, int, int, str]:
    file_format = SUPPORTED_EXTENSIONS[path.suffix.lower()]
    is_duplicate = 1 if remove_duplicate_suffix(path.name) != path.name else 0
    return (
        FORMAT_PRIORITY[file_format],
        is_duplicate,
        len(remove_duplicate_suffix(path.stem)),
        path.name.lower(),
    )


def choose_primary_file(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("choose_primary_file requires at least one path")
    return sorted(paths, key=_file_rank)[0]


def build_record_from_book_dir(book_dir: Path) -> BookRecord:
    paths = discover_candidate_files(book_dir)
    if not paths:
        raise ValueError(f"No supported book files found in {book_dir}")

    primary_path = choose_primary_file(paths)
    primary_format = SUPPORTED_EXTENSIONS[primary_path.suffix.lower()]
    html_candidates = [
        path
        for path in paths
        if SUPPORTED_EXTENSIONS[path.suffix.lower()] == "html" and path != primary_path
    ]
    companion_html_path = html_candidates[0] if html_candidates else None

    title = display_title_from_dir(book_dir)
    author = ""
    for candidate in [primary_path, *paths]:
        author = infer_author_from_filename(candidate.name)
        if author:
            break

    return BookRecord(
        book_id=make_book_id(title, author),
        title=title,
        author=author,
        book_dir=book_dir,
        primary_path=primary_path,
        primary_format=primary_format,
        companion_html_path=companion_html_path,
        all_paths=tuple(paths),
        duplicate_count=max(0, len(paths) - 1),
        search_title=clean_title_for_search(title, author),
    )


def build_record_from_path(path: Path) -> BookRecord:
    primary_path = path.resolve()
    primary_format = SUPPORTED_EXTENSIONS[primary_path.suffix.lower()]
    book_dir = primary_path.parent
    title = display_title_from_dir(book_dir if book_dir.name else primary_path.parent)
    author = infer_author_from_filename(primary_path.name)
    html_path = None
    for sibling in discover_candidate_files(book_dir):
        if sibling.suffix.lower() in {".html", ".htm"} and sibling != primary_path:
            html_path = sibling
            break
    return BookRecord(
        book_id=make_book_id(title, author),
        title=title,
        author=author,
        book_dir=book_dir,
        primary_path=primary_path,
        primary_format=primary_format,
        companion_html_path=html_path,
        all_paths=(primary_path,),
        search_title=clean_title_for_search(title, author),
    )


def _ensure_unique_ids(records: list[BookRecord]) -> list[BookRecord]:
    counts: Counter[str] = Counter()
    unique_records: list[BookRecord] = []
    for record in records:
        counts[record.book_id] += 1
        if counts[record.book_id] == 1:
            unique_records.append(record)
            continue
        unique_records.append(
            BookRecord(
                book_id=f"{record.book_id}-{counts[record.book_id]}",
                title=record.title,
                author=record.author,
                book_dir=record.book_dir,
                primary_path=record.primary_path,
                primary_format=record.primary_format,
                companion_html_path=record.companion_html_path,
                all_paths=record.all_paths,
                duplicate_count=record.duplicate_count,
                source_name=record.source_name,
                search_title=record.search_title,
            )
        )
    return unique_records


def scan_takeout_root(root: Path) -> list[BookRecord]:
    root = root.expanduser().resolve()
    records: list[BookRecord] = []
    for book_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            records.append(build_record_from_book_dir(book_dir))
        except ValueError:
            continue
    return _ensure_unique_ids(records)


def write_catalog(records: list[BookRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(records[0].to_row().keys())
        if records
        else [
            "book_id",
            "title",
            "author",
            "book_dir",
            "primary_path",
            "primary_format",
            "companion_html_path",
            "all_paths",
            "duplicate_count",
            "source_name",
            "search_title",
        ]
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return output_path


def load_catalog(path: Path) -> list[BookRecord]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [BookRecord.from_row(row) for row in csv.DictReader(handle)]


def catalog_statistics(records: list[BookRecord]) -> dict[str, int]:
    stats: Counter[str] = Counter()
    stats["books"] = len(records)
    for record in records:
        stats[f"format_{record.primary_format}"] += 1
        if record.author:
            stats["with_inferred_author"] += 1
        if record.companion_html_path:
            stats["with_html_companion"] += 1
    return dict(stats)
