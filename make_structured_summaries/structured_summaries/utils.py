"""Shared normalization helpers."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

INVALID_AUTHOR_VALUES = {"", "unknown", "by", "nan", "none"}


def normalize_missing(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_file_extension(text: str) -> str:
    return re.sub(r"\.(pdf|epub|html|txt|mobi|azw3)$", "", text, flags=re.IGNORECASE)


def remove_duplicate_suffix(filename: str) -> str:
    return re.sub(r"\(\d+\)(?=\.[^.]+$)", "", filename)


def is_valid_author(author: Any) -> bool:
    value = normalize_missing(author).lower()
    if not value or value in INVALID_AUTHOR_VALUES:
        return False
    return not bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def infer_author_from_filename(filename: str) -> str:
    if not filename:
        return ""
    stem = strip_file_extension(remove_duplicate_suffix(Path(filename).name))
    parts = [part.strip() for part in re.split(r"\s+-\s+", stem) if part.strip()]
    candidates: list[str] = []
    if len(parts) >= 2:
        candidates.extend([parts[0], parts[-1]])
    if re.search(r"\s+-\s+", stem):
        candidates.append(parts[-1].strip())

    for candidate in candidates:
        normalized = normalize_text(candidate)
        if len(normalized.split()) < 2 or len(normalized.split()) > 4:
            continue
        if any(token.isdigit() for token in normalized.split()):
            continue
        if {"press", "books", "publishing", "house"} & set(normalized.split()):
            continue
        return candidate
    return ""


def clean_title_for_search(title: Any, author: str = "") -> str:
    text = normalize_missing(title)
    if not text:
        return ""
    text = strip_file_extension(remove_duplicate_suffix(Path(text).name))
    text = text.replace("_", " ")
    text = text.replace(".", " ")
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"\(\d{4}\)", "", text)
    text = re.sub(r"\(.*?edition.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"project gutenberg ebook", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*-?\s*libgen\.[a-z]+\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+by\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\d\)$", "", text)
    text = re.sub(r"#\w+#", " ", text)

    author_norm = normalize_text(author)
    parts = [
        part.strip(" -") for part in re.split(r"\s+-\s+", text) if part.strip(" -")
    ]
    if len(parts) >= 2 and author_norm and normalize_text(parts[0]) == author_norm:
        text = " - ".join(parts[1:])

    text = re.sub(
        r"\s+-\s+[A-Z][A-Za-z&.' ]{1,40}\s+\(\d{4}\).*$",
        "",
        text,
    )
    text = re.sub(
        r"\s+-\s+[A-Z][A-Za-z&.' ]{1,40}$",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip(" -:")


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "book"


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
    return cleaned.strip()


def parse_json_maybe(text: str) -> Any | None:
    cleaned = strip_code_fence(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
