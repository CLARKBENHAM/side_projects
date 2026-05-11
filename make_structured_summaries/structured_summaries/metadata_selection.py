"""Join cleaned metadata to the local takeout catalog and build review queues."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import re

from .models import BookRecord
from .utils import (
    clean_title_for_search,
    infer_author_from_filename,
    normalize_missing,
    normalize_text,
    remove_duplicate_suffix,
    strip_file_extension,
)

DEFAULT_FICTION_SHELVES = ("fiction", "literature")
TITLE_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "for",
    "to",
    "by",
    "in",
    "on",
    "with",
    "from",
    "more",
    "than",
    "based",
    "revised",
    "updated",
    "edition",
    "editions",
    "library",
    "barnes",
    "noble",
    "program",
    "guide",
    "volume",
    "vol",
    "book",
    "series",
    "saga",
}
EXACT_MATCH_METHODS = {"filename_key", "title_author_key", "title_key"}
MATCH_METHOD_PRIORITY = {
    "filename_key": 4,
    "title_author_key": 3,
    "title_key": 2,
    "fuzzy_title_author": 1,
    "fuzzy_title": 0,
}


@dataclass(frozen=True)
class CatalogMetadataLink:
    book: BookRecord
    metadata_row: dict[str, str] | None
    match_method: str
    override_reason: str = ""

    @property
    def is_matched(self) -> bool:
        return self.metadata_row is not None

    @property
    def is_read(self) -> bool:
        if not self.metadata_row:
            return False
        return bool(normalize_missing(self.metadata_row.get("finished_date")))

    @property
    def bookshelf(self) -> str:
        if not self.metadata_row:
            return ""
        return normalize_missing(self.metadata_row.get("bookshelf"))

    @property
    def avg_enjoyment(self) -> float:
        if not self.metadata_row:
            return 0.0
        return _parse_float(self.metadata_row.get("avg_enjoyment"))

    @property
    def avg_usefulness(self) -> float:
        if not self.metadata_row:
            return 0.0
        return _parse_float(self.metadata_row.get("avg_usefulness"))

    @property
    def signal_score(self) -> float:
        return max(self.avg_enjoyment, self.avg_usefulness)

    def is_fictionish(
        self,
        *,
        fiction_shelves: tuple[str, ...] = DEFAULT_FICTION_SHELVES,
    ) -> bool:
        shelf = normalize_text(self.bookshelf)
        return shelf in {normalize_text(value) for value in fiction_shelves}

    def to_row(self) -> dict[str, str]:
        row = {
            **self.book.to_row(),
            "match_method": self.match_method,
            "matched_metadata": str(self.is_matched),
            "metadata_title": "",
            "metadata_author": "",
            "metadata_filename": "",
            "metadata_source": "",
            "metadata_bookshelf": "",
            "metadata_finished_date": "",
            "avg_enjoyment": "",
            "avg_usefulness": "",
            "signal_score": "",
            "is_read": str(self.is_read),
            "is_fictionish": "False",
            "queue_reason": "",
            "override_reason": self.override_reason,
        }
        if not self.metadata_row:
            row["queue_reason"] = "no_metadata_match"
            return row

        row.update(
            {
                "metadata_title": _metadata_title(self.metadata_row),
                "metadata_author": _metadata_author(self.metadata_row),
                "metadata_filename": normalize_missing(
                    self.metadata_row.get("filename")
                ),
                "metadata_source": normalize_missing(self.metadata_row.get("source")),
                "metadata_bookshelf": self.bookshelf,
                "metadata_finished_date": normalize_missing(
                    self.metadata_row.get("finished_date")
                ),
                "avg_enjoyment": str(self.avg_enjoyment or ""),
                "avg_usefulness": str(self.avg_usefulness or ""),
                "signal_score": str(self.signal_score or ""),
                "is_fictionish": str(self.is_fictionish()),
                "queue_reason": (
                    "manual_read_override"
                    if self.override_reason
                    else (
                        "explicit_unread_metadata"
                        if not self.is_read
                        else "read_metadata_match"
                    )
                ),
                "override_reason": self.override_reason,
            }
        )
        return row


def _parse_float(value: str | None) -> float:
    try:
        return float(normalize_missing(value))
    except ValueError:
        return 0.0


def _metadata_title(row: dict[str, str]) -> str:
    return normalize_missing(row.get("personal_rating_title")) or normalize_missing(
        row.get("title")
    )


def _metadata_author(row: dict[str, str]) -> str:
    return normalize_missing(row.get("corrected_author")) or normalize_missing(
        row.get("author(old and wrong)")
    )


def _filename_key(filename: str) -> str:
    stem = strip_file_extension(remove_duplicate_suffix(Path(filename).name))
    stem = stem.replace("_", " ")
    stem = stem.replace(".", " ")
    return normalize_text(stem)


def _title_key(title: str, author: str = "") -> str:
    return normalize_text(clean_title_for_search(title, author))


def _title_author_key(title: str, author: str) -> str:
    title_key = _title_key(title, author)
    author_key = normalize_text(author)
    if title_key and author_key:
        return f"{title_key}::{author_key}"
    return ""


def _normalize_title_text(value: str) -> str:
    text = normalize_missing(value)
    if not text:
        return ""
    text = strip_file_extension(remove_duplicate_suffix(Path(text).name))
    text = text.replace("_", " ")
    text = text.replace(".", " ")
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"project gutenberg ebook", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"libgen\.[a-z]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"#\w+#", " ", text)
    text = re.sub(r"\b\d{4}\b", " ", text)
    return normalize_text(text)


def _title_variants(title: str, author: str = "") -> set[str]:
    candidates = {
        _normalize_title_text(title),
        _normalize_title_text(clean_title_for_search(title, author)),
    }
    author_key = normalize_text(author)
    variants = {candidate for candidate in candidates if candidate}
    for candidate in list(variants):
        if author_key and candidate.startswith(f"{author_key} "):
            stripped = candidate.removeprefix(f"{author_key} ").strip()
            if stripped:
                variants.add(stripped)
        if candidate.startswith("the "):
            variants.add(candidate.removeprefix("the ").strip())
    return {variant for variant in variants if variant}


def _significant_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.split()
        if len(token) >= 3
        and token not in TITLE_STOPWORDS
        and not (token.isdigit() and len(token) >= 4)
    }


def _author_candidates(book: BookRecord) -> set[str]:
    candidates = {normalize_text(book.author)}
    inferred = infer_author_from_filename(book.primary_path.name)
    candidates.add(normalize_text(inferred))
    return {candidate for candidate in candidates if candidate}


def _author_match_strength(book: BookRecord, metadata_author: str) -> float:
    metadata_author_key = normalize_text(metadata_author)
    if not metadata_author_key:
        return 0.0
    for candidate in _author_candidates(book):
        if candidate == metadata_author_key:
            return 0.08
        candidate_tokens = set(candidate.split())
        metadata_tokens = set(metadata_author_key.split())
        if candidate_tokens and metadata_tokens and candidate_tokens <= metadata_tokens:
            return 0.04
    return 0.0


def _title_similarity(left: str, right: str) -> tuple[float, int]:
    if not left or not right:
        return 0.0, 0
    if left == right:
        return 1.0, len(_significant_tokens(left))

    left_tokens = _significant_tokens(left)
    right_tokens = _significant_tokens(right)
    shared_tokens = len(left_tokens & right_tokens)
    min_tokens = min(len(left_tokens), len(right_tokens))
    prefix = _is_prefixish(left, right)
    seq = SequenceMatcher(None, left, right).ratio()

    if min_tokens and shared_tokens == min_tokens:
        if min_tokens >= 3:
            return 0.93, shared_tokens
        if min_tokens == 1 and prefix:
            return 0.9, shared_tokens

    containment = shared_tokens / max(1, min_tokens)
    jaccard = shared_tokens / max(1, len(left_tokens | right_tokens))
    prefix_bonus = 0.08 if prefix and shared_tokens >= 2 else 0.0
    score = max(seq, 0.68 * containment + 0.22 * jaccard + prefix_bonus)
    return score, shared_tokens


def _is_prefixish(left: str, right: str) -> bool:
    if left.startswith(right) or right.startswith(left):
        return True

    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return False

    shorter_tokens, longer_tokens = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    if len(shorter_tokens) >= len(longer_tokens):
        return False

    for index, token in enumerate(shorter_tokens):
        longer_token = longer_tokens[index]
        if token == longer_token:
            continue
        is_last_shorter_token = index == len(shorter_tokens) - 1
        if is_last_shorter_token and len(token) >= 3 and longer_token.startswith(token):
            continue
        return False
    return True


def _book_title_candidates(book: BookRecord) -> set[str]:
    candidates: set[str] = set()
    for value in (
        book.title,
        book.search_title,
        book.book_dir.name,
        book.primary_path.name,
        *(path.name for path in book.all_paths),
    ):
        candidates |= _title_variants(value, book.author)
    return candidates


def _metadata_title_candidates(row: dict[str, str]) -> set[str]:
    author = _metadata_author(row)
    candidates: set[str] = set()
    for value in (_metadata_title(row), row.get("title", ""), row.get("filename", "")):
        candidates |= _title_variants(value, author)
    return candidates


def _score_book_to_metadata(
    book: BookRecord,
    metadata_row: dict[str, str],
) -> tuple[float, str]:
    filename_key = _filename_key(normalize_missing(metadata_row.get("filename")))
    if filename_key and filename_key in _catalog_filename_keys(book):
        return 1.0, "filename_key"

    metadata_titles = _metadata_title_candidates(metadata_row)
    book_titles = _book_title_candidates(book)
    author_strength = _author_match_strength(book, _metadata_author(metadata_row))

    for book_title in book_titles:
        for metadata_title in metadata_titles:
            if book_title == metadata_title:
                if author_strength:
                    return min(0.99, 0.98 + author_strength), "title_author_key"
                return 0.95, "title_key"

    best_score = 0.0
    best_shared_tokens = 0
    for book_title in book_titles:
        for metadata_title in metadata_titles:
            score, shared_tokens = _title_similarity(book_title, metadata_title)
            if score > best_score or (
                score == best_score and shared_tokens > best_shared_tokens
            ):
                best_score = score
                best_shared_tokens = shared_tokens

    adjusted_score = min(0.99, best_score + author_strength)
    method = "fuzzy_title_author" if author_strength else "fuzzy_title"
    if best_shared_tokens >= 3 and adjusted_score >= 0.78:
        return adjusted_score, method
    if best_shared_tokens >= 2 and adjusted_score >= 0.84:
        return adjusted_score, method
    if best_shared_tokens == 1 and adjusted_score >= 0.9:
        return adjusted_score, method
    return 0.0, ""


def _is_clear_candidate(
    *,
    best_score: float,
    second_score: float,
    match_method: str,
) -> bool:
    if match_method in EXACT_MATCH_METHODS:
        return True
    margin = best_score - second_score
    if best_score >= 0.92 and margin >= 0.02:
        return True
    if best_score >= 0.84 and margin >= 0.08:
        return True
    if best_score >= 0.8 and match_method == "fuzzy_title_author" and margin >= 0.08:
        return True
    return False


def _manual_override_row(book: BookRecord, reason: str) -> dict[str, str]:
    return {
        "personal_rating_title": book.title,
        "title": book.title,
        "corrected_author": book.author,
        "author(old and wrong)": book.author,
        "filename": book.primary_path.name,
        "source": "manual_read_override",
        "bookshelf": "",
        "finished_date": "manual_read_override",
        "avg_enjoyment": "",
        "avg_usefulness": "",
        "override_reason": reason,
    }


def load_read_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            normalize_missing(row.get("book_id")): normalize_missing(row.get("reason"))
            for row in csv.DictReader(handle)
            if normalize_missing(row.get("book_id"))
        }


def load_master_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _catalog_filename_keys(book: BookRecord) -> set[str]:
    return {_filename_key(path.name) for path in book.all_paths}


def build_catalog_metadata_links(
    records: list[BookRecord],
    metadata_rows: list[dict[str, str]],
    *,
    read_overrides: dict[str, str] | None = None,
) -> list[CatalogMetadataLink]:
    read_overrides = read_overrides or {}
    book_candidates: dict[str, list[tuple[float, int, str]]] = {}
    for record in records:
        candidates: list[tuple[float, int, str]] = []
        for metadata_index, metadata_row in enumerate(metadata_rows):
            score, method = _score_book_to_metadata(record, metadata_row)
            if not method:
                continue
            candidates.append((score, metadata_index, method))
        book_candidates[record.book_id] = sorted(
            candidates,
            key=lambda item: (
                MATCH_METHOD_PRIORITY[item[2]],
                item[0],
            ),
            reverse=True,
        )

    candidate_pairs: list[tuple[float, str, str, int]] = []
    for record in records:
        candidates = book_candidates[record.book_id]
        if not candidates:
            continue
        best_score, metadata_index, match_method = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        if _is_clear_candidate(
            best_score=best_score,
            second_score=second_score,
            match_method=match_method,
        ):
            candidate_pairs.append(
                (best_score, match_method, record.book_id, metadata_index)
            )

    matched_rows_by_book_id: dict[str, dict[str, str]] = {}
    match_methods_by_book_id: dict[str, str] = {}
    matched_book_ids: set[str] = set()
    matched_metadata_indices: set[int] = set()
    for score, match_method, book_id, metadata_index in sorted(
        candidate_pairs,
        key=lambda item: (
            MATCH_METHOD_PRIORITY[item[1]],
            item[0],
        ),
        reverse=True,
    ):
        if book_id in matched_book_ids or metadata_index in matched_metadata_indices:
            continue
        matched_book_ids.add(book_id)
        matched_metadata_indices.add(metadata_index)
        matched_rows_by_book_id[book_id] = metadata_rows[metadata_index]
        match_methods_by_book_id[book_id] = match_method

    links: list[CatalogMetadataLink] = []
    for record in records:
        metadata_row = matched_rows_by_book_id.get(record.book_id)
        override_reason = ""
        match_method = match_methods_by_book_id.get(record.book_id, "")
        if record.book_id in read_overrides:
            override_reason = read_overrides[record.book_id]
            metadata_row = _manual_override_row(record, override_reason)
            match_method = "manual_override"
        links.append(
            CatalogMetadataLink(
                book=record,
                metadata_row=metadata_row,
                match_method=match_method,
                override_reason=override_reason,
            )
        )
    return links


def select_high_signal_read_links(
    links: list[CatalogMetadataLink],
    *,
    min_signal_score: float = 4.0,
    fiction_shelves: tuple[str, ...] = DEFAULT_FICTION_SHELVES,
) -> list[CatalogMetadataLink]:
    return sorted(
        [
            link
            for link in links
            if link.is_read
            and not link.is_fictionish(fiction_shelves=fiction_shelves)
            and link.signal_score >= min_signal_score
        ],
        key=lambda link: (
            link.signal_score,
            link.avg_usefulness,
            link.avg_enjoyment,
            link.book.title,
        ),
        reverse=True,
    )


def select_unread_local_links(
    links: list[CatalogMetadataLink],
) -> list[CatalogMetadataLink]:
    return sorted(
        [
            link
            for link in links
            if not link.is_matched or (link.is_matched and not link.is_read)
        ],
        key=lambda link: link.book.title.lower(),
    )


def write_link_rows(links: list[CatalogMetadataLink], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [link.to_row() for link in links]
    fieldnames = (
        list(rows[0].keys())
        if rows
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
            "match_method",
            "matched_metadata",
            "metadata_title",
            "metadata_author",
            "metadata_filename",
            "metadata_source",
            "metadata_bookshelf",
            "metadata_finished_date",
            "avg_enjoyment",
            "avg_usefulness",
            "signal_score",
            "is_read",
            "is_fictionish",
            "queue_reason",
            "override_reason",
        ]
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
