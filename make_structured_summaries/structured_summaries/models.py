"""Project data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import clean_title_for_search, normalize_text, slugify


@dataclass(frozen=True)
class BookRecord:
    book_id: str
    title: str
    author: str
    book_dir: Path
    primary_path: Path
    primary_format: str
    companion_html_path: Path | None
    all_paths: tuple[Path, ...]
    duplicate_count: int = 0
    source_name: str = "play_books_takeout"
    search_title: str = ""

    def with_search_title(self) -> "BookRecord":
        if self.search_title:
            return self
        return BookRecord(
            book_id=self.book_id,
            title=self.title,
            author=self.author,
            book_dir=self.book_dir,
            primary_path=self.primary_path,
            primary_format=self.primary_format,
            companion_html_path=self.companion_html_path,
            all_paths=self.all_paths,
            duplicate_count=self.duplicate_count,
            source_name=self.source_name,
            search_title=clean_title_for_search(self.title, self.author),
        )

    def to_row(self) -> dict[str, str]:
        enriched = self.with_search_title()
        return {
            "book_id": enriched.book_id,
            "title": enriched.title,
            "author": enriched.author,
            "book_dir": str(enriched.book_dir),
            "primary_path": str(enriched.primary_path),
            "primary_format": enriched.primary_format,
            "companion_html_path": str(enriched.companion_html_path or ""),
            "all_paths": "|".join(str(path) for path in enriched.all_paths),
            "duplicate_count": str(enriched.duplicate_count),
            "source_name": enriched.source_name,
            "search_title": enriched.search_title,
        }

    def to_goodreads_row(self) -> dict[str, str]:
        enriched = self.with_search_title()
        return {
            "canonical_key": normalize_text(
                f"{enriched.search_title}::{enriched.author or enriched.title}"
            ),
            "title": enriched.title,
            "author": enriched.author,
            "Bookshelf": "Play Books Takeout",
            "search_title": enriched.search_title,
            "search_author": enriched.author,
            "filename_play": enriched.primary_path.name,
            "filename_ratings2": "",
        }

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "BookRecord":
        all_paths_raw = row.get("all_paths", "")
        all_paths = tuple(
            Path(value) for value in all_paths_raw.split("|") if value.strip()
        )
        primary_path = Path(row["primary_path"])
        return cls(
            book_id=row["book_id"],
            title=row["title"],
            author=row.get("author", ""),
            book_dir=Path(row["book_dir"]),
            primary_path=primary_path,
            primary_format=row["primary_format"],
            companion_html_path=(
                Path(row["companion_html_path"])
                if row.get("companion_html_path")
                else None
            ),
            all_paths=all_paths or (primary_path,),
            duplicate_count=int(row.get("duplicate_count", "0") or 0),
            source_name=row.get("source_name", "play_books_takeout"),
            search_title=row.get("search_title", ""),
        )


def make_book_id(title: str, author: str = "") -> str:
    if author:
        return slugify(f"{title}-{author}")
    return slugify(title)
