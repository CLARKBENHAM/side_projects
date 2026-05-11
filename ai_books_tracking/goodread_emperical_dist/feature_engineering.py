from __future__ import annotations

import math
import re

import pandas as pd

CATEGORY_KEYWORDS = {
    "Machine Learning": [
        "machine learning",
        "deep learning",
        "neural network",
        "artificial intelligence",
        "reinforcement learning",
    ],
    "Computer Science": [
        "computer science",
        "programming",
        "software",
        "algorithm",
        "data structure",
        "distributed systems",
        "operating systems",
        "compiler",
        "database",
        "code",
    ],
    "Math": [
        "mathematics",
        "math",
        "statistics",
        "probability",
        "geometry",
        "algebra",
        "calculus",
        "topology",
        "real analysis",
        "complex analysis",
        "functional analysis",
        "mathematical analysis",
    ],
    "Business, management": [
        "business",
        "economics",
        "finance",
        "management",
        "marketing",
        "investing",
        "productivity",
        "leadership",
        "entrepreneur",
    ],
    "Histories": [
        "history",
        "historical",
        "war",
        "empire",
        "biography",
        "memoir",
        "civilization",
        "politics",
        "political",
    ],
    "fiction": [
        "fiction",
        "novel",
        "fantasy",
        "science fiction",
        "sci-fi",
        "mystery",
        "thriller",
        "romance",
        "story collection",
        "stories",
    ],
    "Literature": [
        "literature",
        "literary",
        "poetry",
        "play",
        "essay",
        "essays",
        "classic",
        "criticism",
    ],
}
SPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def infer_category(
    title: object,
    author_name: object,
    user_shelves: object,
    description: object,
) -> str:
    combined = " ".join(
        normalize_text(value)
        for value in [title, author_name, user_shelves, description]
        if normalize_text(value)
    )
    if not combined:
        return "General Reading"

    scores: dict[str, int] = {category: 0 for category in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                scores[category] += 1

    if "fiction" in combined and scores["fiction"] == 0:
        scores["fiction"] += 1
    if "novella" in combined:
        scores["fiction"] += 1

    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "General Reading"
    return best_category


def category_bucket(category: object) -> str:
    text = "" if pd.isna(category) else str(category).strip()
    if text in {"fiction", "Literature"}:
        return "Fiction/Literature"
    if text in {"Computer Science", "Machine Learning", "Math"}:
        return "Technical/Other"
    if text == "Histories":
        return "General/Business"
    if text == "Business, management":
        return "General/Business"
    return "General/Business"


def _coalesce_date_columns(frame: pd.DataFrame) -> pd.Series:
    candidates = [
        pd.to_datetime(frame["user_read_at"], errors="coerce", utc=True),
        pd.to_datetime(frame["user_date_added"], errors="coerce", utc=True),
        pd.to_datetime(frame["pub_date"], errors="coerce", utc=True),
    ]
    output = candidates[0]
    for candidate in candidates[1:]:
        output = output.fillna(candidate)
    return output


def prepare_profile_books(raw_books: pd.DataFrame) -> pd.DataFrame:
    frame = raw_books.copy()
    if frame.empty:
        return frame

    frame.columns = frame.columns.str.strip()
    frame["user_rating"] = pd.to_numeric(frame["user_rating"], errors="coerce")
    frame["average_rating"] = pd.to_numeric(frame["average_rating"], errors="coerce")
    frame["book_published"] = pd.to_numeric(frame["book_published"], errors="coerce")
    frame = frame[frame["user_rating"].fillna(0) > 0].copy()

    frame["event_date"] = _coalesce_date_columns(frame)
    frame["event_year"] = frame["event_date"].dt.year
    frame["description_length"] = frame["book_description"].fillna("").str.len()
    frame["category"] = frame.apply(
        lambda row: infer_category(
            row.get("title", ""),
            row.get("author_name", ""),
            row.get("user_shelves", ""),
            row.get("book_description", ""),
        ),
        axis=1,
    )
    frame["category_bucket"] = frame["category"].map(category_bucket)

    dedupe_sort = frame.sort_values(
        ["event_date", "user_date_added", "title"],
        ascending=[True, True, True],
        na_position="last",
    )
    dedupe_key = dedupe_sort["book_id"].where(
        dedupe_sort["book_id"].astype(str).str.len() > 0,
        dedupe_sort["title"].fillna("").str.lower(),
    )
    dedupe_sort["dedupe_key"] = dedupe_key
    frame = dedupe_sort.drop_duplicates(subset=["dedupe_key"], keep="last").copy()
    frame = frame.sort_values(
        ["event_date", "title"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    frame["profile_rating_mean"] = frame.groupby("profile_slug")[
        "user_rating"
    ].transform("mean")
    frame["profile_rating_std"] = frame.groupby("profile_slug")[
        "user_rating"
    ].transform(lambda series: float(series.std()) if len(series) > 1 else math.nan)
    return frame
