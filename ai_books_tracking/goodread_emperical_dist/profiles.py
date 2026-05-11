from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd

from ai_books_tracking.goodread_emperical_dist.config import (
    ANALYSIS_PROFILE_MANIFEST_CSV,
    PROFILE_MANIFEST_CSV,
)

USER_ID_RE = re.compile(r"/user/show/(?P<user_id>\d+)")
MANIFEST_COLUMNS = [
    "display_name",
    "profile_slug",
    "goodreads_url",
    "books_on_goodreads",
    "public_ratings",
    "public_reviews",
    "read_shelf_count",
    "source_group",
    "fit_notes",
]


def _to_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def extract_user_id(goodreads_url: str) -> str:
    match = USER_ID_RE.search(str(goodreads_url))
    if not match:
        raise ValueError(f"Could not parse Goodreads user id from URL: {goodreads_url}")
    return match.group("user_id")


@dataclass(frozen=True)
class GoodreadsProfile:
    display_name: str
    profile_slug: str
    goodreads_url: str
    user_id: str
    books_on_goodreads: int | None
    public_ratings: int | None
    public_reviews: int | None
    read_shelf_count: int | None
    source_group: str
    fit_notes: str

    @property
    def rss_url_template(self) -> str:
        return (
            "https://www.goodreads.com/review/list_rss/"
            f"{self.user_id}?shelf=read&page={{page}}"
        )

    def feed_path(self, base_dir: Path) -> Path:
        return base_dir / f"{self.profile_slug}.csv"


def _standardize_manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = normalized.columns.str.strip()
    for column in MANIFEST_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["public_ratings"] = pd.to_numeric(
        normalized["public_ratings"], errors="coerce"
    )
    normalized["public_reviews"] = pd.to_numeric(
        normalized["public_reviews"], errors="coerce"
    )
    normalized["books_on_goodreads"] = pd.to_numeric(
        normalized["books_on_goodreads"], errors="coerce"
    )
    normalized["read_shelf_count"] = pd.to_numeric(
        normalized["read_shelf_count"], errors="coerce"
    )
    return normalized[MANIFEST_COLUMNS].copy()


def _filter_seed_manifest_frame(
    frame: pd.DataFrame,
    minimum_public_ratings: int = 100,
    limit_profiles: int | None = None,
) -> pd.DataFrame:
    filtered = _standardize_manifest_frame(frame)
    filtered = filtered[filtered["public_ratings"].fillna(0) >= minimum_public_ratings]
    filtered = filtered.sort_values(
        ["source_group", "public_ratings", "display_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    if limit_profiles is not None:
        filtered = filtered.head(limit_profiles).copy()
    return filtered


def _ensure_unique_profile_slugs(frame: pd.DataFrame) -> pd.DataFrame:
    if (
        frame.empty
        or "profile_slug" not in frame.columns
        or "user_id" not in frame.columns
    ):
        return frame
    adjusted = frame.copy()
    seen_counts: dict[str, int] = {}
    unique_slugs: list[str] = []
    for row in adjusted.itertuples(index=False):
        base_slug = str(row.profile_slug).strip()
        user_id = str(row.user_id).strip()
        if not base_slug:
            base_slug = f"user-{user_id}"
        count = seen_counts.get(base_slug, 0)
        if count == 0:
            candidate = base_slug
        else:
            candidate = f"{base_slug}-{user_id}"
        while candidate in seen_counts:
            count += 1
            candidate = f"{base_slug}-{user_id}-{count}"
        seen_counts[base_slug] = count + 1
        seen_counts[candidate] = 1
        unique_slugs.append(candidate)
    adjusted["profile_slug"] = unique_slugs
    return adjusted


def build_analysis_manifest_frame(
    seed_frame: pd.DataFrame,
    network_validated_frame: pd.DataFrame | None = None,
    include_network_profiles: bool = False,
    analysis_target_total: int | None = None,
) -> pd.DataFrame:
    analysis_frame = _standardize_manifest_frame(seed_frame).copy()
    analysis_frame["user_id"] = analysis_frame["goodreads_url"].map(extract_user_id)
    analysis_frame["source_priority"] = 0
    if not include_network_profiles or network_validated_frame is None:
        return analysis_frame[MANIFEST_COLUMNS].copy().reset_index(drop=True)

    network_frame = network_validated_frame.copy()
    if network_frame.empty or "passes_network_filter" not in network_frame.columns:
        return analysis_frame[MANIFEST_COLUMNS].copy().reset_index(drop=True)
    network_frame = network_frame[network_frame["passes_network_filter"].fillna(False)]
    if network_frame.empty:
        return analysis_frame[MANIFEST_COLUMNS].copy().reset_index(drop=True)

    network_candidates = pd.DataFrame(
        {
            "display_name": network_frame["display_name"].astype(str).str.strip(),
            "profile_slug": network_frame["profile_slug"].astype(str).str.strip(),
            "goodreads_url": network_frame["goodreads_url"].astype(str).str.strip(),
            "books_on_goodreads": pd.to_numeric(
                network_frame.get("books_on_goodreads"), errors="coerce"
            ),
            "public_ratings": pd.to_numeric(
                network_frame.get("public_ratings"), errors="coerce"
            ),
            "public_reviews": pd.to_numeric(
                network_frame.get("public_reviews"), errors="coerce"
            ),
            "read_shelf_count": pd.NA,
            "source_group": "network_discovered",
            "fit_notes": (
                "Public-profile network expansion;"
                + " source="
                + network_frame["source_profile_slug"].fillna("").astype(str)
                + "; depth="
                + network_frame["discovery_depth"].fillna(-1).astype(int).astype(str)
                + "; observed_years="
                + network_frame["observed_distinct_years"]
                .fillna(0)
                .astype(int)
                .astype(str)
            ),
            "user_id": network_frame["goodreads_url"].map(extract_user_id),
            "observed_distinct_years": pd.to_numeric(
                network_frame.get("observed_distinct_years"), errors="coerce"
            ),
            "followers_count": pd.to_numeric(
                network_frame.get("followers_count"), errors="coerce"
            ),
            "discovery_depth": pd.to_numeric(
                network_frame.get("discovery_depth"), errors="coerce"
            ),
            "source_priority": 1,
        }
    )
    combined = pd.concat([analysis_frame, network_candidates], ignore_index=True)
    combined = combined.sort_values(
        [
            "source_priority",
            "discovery_depth",
            "public_ratings",
            "observed_distinct_years",
            "followers_count",
            "display_name",
        ],
        ascending=[True, True, False, False, False, True],
        na_position="last",
    ).drop_duplicates(subset=["user_id"], keep="first")

    if analysis_target_total is not None and analysis_target_total > 0:
        seed_count = int((combined["source_group"] != "network_discovered").sum())
        keep_total = max(seed_count, analysis_target_total)
        seeds = combined[combined["source_group"] != "network_discovered"]
        network_only = combined[combined["source_group"] == "network_discovered"]
        slots = max(0, keep_total - len(seeds))
        combined = pd.concat(
            [seeds, network_only.head(slots)],
            ignore_index=True,
        )

    combined = _ensure_unique_profile_slugs(combined)
    return combined[MANIFEST_COLUMNS].reset_index(drop=True)


def build_analysis_manifest(
    minimum_public_ratings: int = 100,
    limit_profiles: int | None = None,
) -> pd.DataFrame:
    seed_frame = pd.read_csv(PROFILE_MANIFEST_CSV)
    return _filter_seed_manifest_frame(
        seed_frame,
        minimum_public_ratings=minimum_public_ratings,
        limit_profiles=limit_profiles,
    )


def profiles_from_frame(frame: pd.DataFrame) -> list[GoodreadsProfile]:
    standardized = _standardize_manifest_frame(frame)

    profiles: list[GoodreadsProfile] = []
    for row in standardized.to_dict(orient="records"):
        profiles.append(
            GoodreadsProfile(
                display_name=str(row["display_name"]).strip(),
                profile_slug=str(row["profile_slug"]).strip(),
                goodreads_url=str(row["goodreads_url"]).strip(),
                user_id=extract_user_id(str(row["goodreads_url"])),
                books_on_goodreads=_to_int(row.get("books_on_goodreads")),
                public_ratings=_to_int(row.get("public_ratings")),
                public_reviews=_to_int(row.get("public_reviews")),
                read_shelf_count=_to_int(row.get("read_shelf_count")),
                source_group=str(row.get("source_group", "")).strip(),
                fit_notes=str(row.get("fit_notes", "")).strip(),
            )
        )
    return profiles


def load_profiles(
    manifest_path: Path = PROFILE_MANIFEST_CSV,
    minimum_public_ratings: int = 100,
    limit_profiles: int | None = None,
) -> list[GoodreadsProfile]:
    frame = pd.read_csv(manifest_path)
    filtered = _filter_seed_manifest_frame(
        frame,
        minimum_public_ratings=minimum_public_ratings,
        limit_profiles=limit_profiles,
    )
    return profiles_from_frame(filtered)


def write_analysis_manifest(
    network_validated_frame: pd.DataFrame | None = None,
    include_network_profiles: bool = False,
    minimum_public_ratings: int = 100,
    limit_profiles: int | None = None,
    analysis_target_total: int | None = None,
    output_path: Path = ANALYSIS_PROFILE_MANIFEST_CSV,
) -> pd.DataFrame:
    seed_frame = build_analysis_manifest(
        minimum_public_ratings=minimum_public_ratings,
        limit_profiles=limit_profiles,
    )
    analysis_frame = build_analysis_manifest_frame(
        seed_frame=seed_frame,
        network_validated_frame=network_validated_frame,
        include_network_profiles=include_network_profiles,
        analysis_target_total=analysis_target_total,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_frame.to_csv(output_path, index=False)
    return analysis_frame
