"""Export Play Books takeout titles that are not already labeled as read."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

from ai_books_tracking.goodreads_ratings import clean_title_for_search, normalize_text
from ai_books_tracking.score_play_books_takeout import (
    attach_known_labels,
    load_takeout_catalog,
    load_training_frame,
)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_CSV = OUTPUT_DIR / "takeout_play_books_03_16_25_unread_for_external_search.csv"


def normalize_filename(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\(\d+\)(?=\.[a-z0-9]+$)", "", text)
    return text


def normalize_filename_stem(value: object) -> str:
    text = normalize_filename(value)
    text = re.sub(r"\.[a-z0-9]+$", "", text)
    text = text.replace("_", " ")
    return normalize_text(text)


def normalize_author_key(value: object) -> str:
    normalized = normalize_text(value)
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    return " ".join(sorted(tokens))


def normalize_title_key(title: object, author: object) -> str:
    return normalize_text(clean_title_for_search(title or "", author or ""))


def build_training_match_frame(training: pd.DataFrame) -> pd.DataFrame:
    prepared = training.copy()
    prepared["title_key"] = prepared["title"].map(normalize_text)
    prepared["search_title_key"] = prepared.apply(
        lambda row: normalize_title_key(row.get("title", ""), row.get("author", "")),
        axis=1,
    )
    prepared["author_key"] = prepared.get("author", pd.Series("", index=prepared.index)).map(
        normalize_author_key
    )
    prepared["filename_key"] = prepared.get(
        "filename", pd.Series("", index=prepared.index)
    ).map(normalize_filename)
    prepared["filename_stem_key"] = prepared.get(
        "filename", pd.Series("", index=prepared.index)
    ).map(normalize_filename_stem)
    return prepared[
        ["title_key", "search_title_key", "author_key", "filename_key", "filename_stem_key"]
    ].drop_duplicates()


def build_exact_match_sets(training_keys: pd.DataFrame) -> dict[str, set[str]]:
    return {
        "filename_key": {value for value in training_keys["filename_key"] if value},
        "filename_stem_key": {value for value in training_keys["filename_stem_key"] if value},
        "title_key": {value for value in training_keys["title_key"] if value},
        "search_title_key": {value for value in training_keys["search_title_key"] if value},
    }


def titles_compatible(candidate_title: str, training_title: str) -> bool:
    shorter = min(len(candidate_title), len(training_title))
    if shorter < 8:
        return False
    return candidate_title in training_title or training_title in candidate_title


def has_training_match(
    candidate: pd.Series, training_keys: pd.DataFrame, exact_sets: dict[str, set[str]]
) -> bool:
    if candidate["filename_key"] and candidate["filename_key"] in exact_sets["filename_key"]:
        return True
    if (
        candidate["filename_stem_key"]
        and candidate["filename_stem_key"] in exact_sets["filename_stem_key"]
    ):
        return True
    if candidate["title_key"] and candidate["title_key"] in exact_sets["title_key"]:
        return True
    if (
        candidate["search_title_key"]
        and candidate["search_title_key"] in exact_sets["search_title_key"]
    ):
        return True

    if candidate["author_key"]:
        pool = training_keys[
            training_keys["author_key"].eq(candidate["author_key"]) | training_keys["author_key"].eq("")
        ]
    else:
        pool = training_keys

    for row in pool.itertuples(index=False):
        if titles_compatible(candidate["title_key"], row.title_key):
            return True
        if titles_compatible(candidate["search_title_key"], row.search_title_key):
            return True

    for row in training_keys.itertuples(index=False):
        if titles_compatible(candidate["title_key"], row.title_key):
            return True
        if titles_compatible(candidate["search_title_key"], row.search_title_key):
            return True
    return False


def build_export_frame() -> pd.DataFrame:
    catalog = load_takeout_catalog()
    training = load_training_frame(include_new_holdout=True)
    candidates = attach_known_labels(catalog, training)
    training_keys = build_training_match_frame(training)
    exact_sets = build_exact_match_sets(training_keys)

    candidates["title_key"] = candidates["title"].map(normalize_text)
    candidates["search_title_key"] = candidates.apply(
        lambda row: normalize_title_key(row.get("title", ""), row.get("author", "")),
        axis=1,
    )
    candidates["author_key"] = candidates["author"].map(normalize_author_key)
    candidates["filename_key"] = candidates["filename"].map(normalize_filename)
    candidates["filename_stem_key"] = candidates["filename"].map(normalize_filename_stem)
    candidates["already_read_robust"] = candidates.apply(
        lambda row: bool(row["already_labeled"])
        or has_training_match(row, training_keys, exact_sets),
        axis=1,
    )

    unread = candidates[~candidates["already_read_robust"]].copy()

    unread["search_title"] = unread.apply(
        lambda row: clean_title_for_search(row.get("title", ""), row.get("author", "")),
        axis=1,
    )
    unread["search_author"] = unread["author"].fillna("").astype(str).str.strip()

    for column in [
        "goodreads_rating",
        "goodreads_rating_count",
        "goodreads_url",
        "amazon_rating",
        "amazon_rating_count",
        "amazon_url",
        "openlibrary_rating",
        "openlibrary_rating_count",
        "openlibrary_url",
        "search_notes",
    ]:
        unread[column] = ""

    unread = unread.sort_values(
        ["Bookshelf", "play_status", "latest_modified", "title"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    return unread[
        [
            "title",
            "author",
            "Bookshelf",
            "play_status",
            "earliest_modified",
            "latest_modified",
            "filename",
            "canonical_key",
            "search_title",
            "search_author",
            "goodreads_rating",
            "goodreads_rating_count",
            "goodreads_url",
            "amazon_rating",
            "amazon_rating_count",
            "amazon_url",
            "openlibrary_rating",
            "openlibrary_rating_count",
            "openlibrary_url",
            "search_notes",
        ]
    ].copy()


def main() -> None:
    export = build_export_frame()
    export.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV.name}")
    print(f"Rows: {len(export)}")


if __name__ == "__main__":
    main()
