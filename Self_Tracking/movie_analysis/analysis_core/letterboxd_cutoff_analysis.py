from __future__ import annotations

import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "data" / "mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)
LETTERBOXD_NS = "https://letterboxd.com"
TMDB_NS = "https://themoviedb.org"
JSON_LD_PATTERN = re.compile(
    r'<script[^>]+type="application/ld\+json"[^>]*>(?P<body>.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
CDATA_PREFIX = "/* <![CDATA[ */"
CDATA_SUFFIX = "/* ]]> */"


@dataclass(frozen=True)
class LetterboxdFilmMetadata:
    film_title: str
    film_year: int | None
    average_rating: float
    rating_count: int | None
    review_count: int | None
    canonical_film_url: str


def build_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def _parse_timestamp(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed if not pd.isna(parsed) else pd.NaT


def _parse_date(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    return parsed if not pd.isna(parsed) else pd.NaT


def _parse_yes_no(value: str | None) -> bool:
    return str(value or "").strip().lower() == "yes"


def canonicalize_film_url(item_link: str) -> str:
    parsed = urlparse(item_link)
    path_parts = [part for part in parsed.path.split("/") if part]
    if "film" not in path_parts:
        return ""
    film_index = path_parts.index("film")
    if film_index + 1 >= len(path_parts):
        return ""
    film_slug = path_parts[film_index + 1]
    return f"https://letterboxd.com/film/{film_slug}/"


def fetch_profile_rss(
    profile_slug: str,
    session: requests.Session,
) -> str:
    response = session.get(
        f"https://letterboxd.com/{profile_slug}/rss/",
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def parse_profile_rss(
    xml_text: str,
    *,
    profile_slug: str,
) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    channel = root.find("./channel")
    display_name = ""
    if channel is not None:
        title_text = channel.findtext("title") or ""
        display_name = title_text.removeprefix("Letterboxd - ").strip()

    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(tz=UTC).isoformat()
    for item in root.findall(".//item"):
        film_title = item.findtext(f"{{{LETTERBOXD_NS}}}filmTitle") or ""
        item_link = (item.findtext("link") or "").strip()
        canonical_film_url = canonicalize_film_url(item_link)
        if not film_title or not canonical_film_url:
            continue
        tmdb_movie_id = pd.to_numeric(
            item.findtext(f"{{{TMDB_NS}}}movieId"),
            errors="coerce",
        )
        member_rating = pd.to_numeric(
            item.findtext(f"{{{LETTERBOXD_NS}}}memberRating"),
            errors="coerce",
        )
        rows.append(
            {
                "profile_slug": profile_slug,
                "display_name": display_name or profile_slug,
                "fetched_at_utc": fetched_at,
                "item_title": (item.findtext("title") or "").strip(),
                "item_link": item_link,
                "guid": (item.findtext("guid") or "").strip(),
                "pub_date_utc": _parse_timestamp(item.findtext("pubDate")),
                "watched_date": _parse_date(
                    item.findtext(f"{{{LETTERBOXD_NS}}}watchedDate")
                ),
                "rewatch": _parse_yes_no(item.findtext(f"{{{LETTERBOXD_NS}}}rewatch")),
                "film_title": film_title.strip(),
                "film_year": pd.to_numeric(
                    item.findtext(f"{{{LETTERBOXD_NS}}}filmYear"),
                    errors="coerce",
                ),
                "member_rating": member_rating,
                "member_like": _parse_yes_no(
                    item.findtext(f"{{{LETTERBOXD_NS}}}memberLike")
                ),
                "tmdb_movie_id": (
                    int(tmdb_movie_id) if not pd.isna(tmdb_movie_id) else pd.NA
                ),
                "canonical_film_url": canonical_film_url,
                "description_html": item.findtext("description") or "",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["film_key"] = frame["tmdb_movie_id"].astype("string").fillna("")
    missing_key = frame["film_key"] == ""
    frame.loc[missing_key, "film_key"] = frame.loc[missing_key, "canonical_film_url"]
    return frame


def dedupe_latest_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.loc[frame["member_rating"].notna()].copy()
    if working.empty:
        return working
    watched_date = pd.to_datetime(working["watched_date"], errors="coerce", utc=True)
    pub_date_utc = pd.to_datetime(working["pub_date_utc"], errors="coerce", utc=True)
    working["sort_date"] = watched_date.combine_first(pub_date_utc)
    return (
        working.sort_values(
            ["profile_slug", "sort_date", "pub_date_utc"],
            ascending=[True, False, False],
        )
        .drop_duplicates(["profile_slug", "film_key"], keep="first")
        .drop(columns=["sort_date"])
        .reset_index(drop=True)
    )


def _clean_json_ld(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith(CDATA_PREFIX):
        cleaned = cleaned[len(CDATA_PREFIX) :].strip()
    if cleaned.endswith(CDATA_SUFFIX):
        cleaned = cleaned[: -len(CDATA_SUFFIX)].strip()
    return cleaned


def _parse_movie_year(payload: dict[str, Any]) -> int | None:
    released_event = payload.get("releasedEvent")
    if isinstance(released_event, list) and released_event:
        start_date = str(released_event[0].get("startDate") or "")
        match = re.search(r"\b(\d{4})\b", start_date)
        if match is not None:
            return int(match.group(1))
    date_created = str(payload.get("dateCreated") or "")
    match = re.search(r"\b(\d{4})\b", date_created)
    if match is not None:
        return int(match.group(1))
    return None


def parse_film_page_metadata(
    html_text: str,
    *,
    canonical_film_url: str,
) -> LetterboxdFilmMetadata:
    for match in JSON_LD_PATTERN.finditer(html_text):
        raw_body = _clean_json_ld(match.group("body"))
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            continue
        payloads = payload if isinstance(payload, list) else [payload]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            aggregate = item.get("aggregateRating")
            if not isinstance(aggregate, dict):
                continue
            rating_value = aggregate.get("ratingValue")
            if rating_value is None:
                continue
            return LetterboxdFilmMetadata(
                film_title=str(item.get("name") or "").strip(),
                film_year=_parse_movie_year(item),
                average_rating=float(rating_value),
                rating_count=(
                    int(aggregate["ratingCount"])
                    if aggregate.get("ratingCount") is not None
                    else None
                ),
                review_count=(
                    int(aggregate["reviewCount"])
                    if aggregate.get("reviewCount") is not None
                    else None
                ),
                canonical_film_url=canonical_film_url,
            )
    raise ValueError(
        f"Could not find aggregateRating JSON-LD for {canonical_film_url!r}"
    )


def fetch_film_metadata(
    canonical_film_url: str,
    session: requests.Session,
) -> LetterboxdFilmMetadata:
    response = session.get(canonical_film_url, timeout=30)
    response.raise_for_status()
    return parse_film_page_metadata(
        response.text,
        canonical_film_url=canonical_film_url,
    )


def load_json_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def enrich_with_film_metadata(
    ratings: pd.DataFrame,
    *,
    cache_path: Path,
    refresh: bool = False,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    if ratings.empty:
        return ratings.copy()
    cache = load_json_cache(cache_path)
    session = build_requests_session()
    metadata_rows: list[dict[str, Any]] = []
    unique_urls = sorted(ratings["canonical_film_url"].dropna().unique().tolist())
    for index, canonical_film_url in enumerate(unique_urls):
        cached_payload = cache.get(canonical_film_url)
        if refresh or cached_payload is None:
            metadata = fetch_film_metadata(canonical_film_url, session)
            cached_payload = {
                "film_title": metadata.film_title,
                "film_year": metadata.film_year,
                "average_rating": metadata.average_rating,
                "rating_count": metadata.rating_count,
                "review_count": metadata.review_count,
                "canonical_film_url": metadata.canonical_film_url,
            }
            cache[canonical_film_url] = cached_payload
            save_json_cache(cache_path, cache)
            if sleep_seconds > 0 and index < len(unique_urls) - 1:
                time.sleep(sleep_seconds)
        metadata_rows.append(cached_payload)
    metadata_df = pd.DataFrame(metadata_rows)
    return ratings.merge(metadata_df, on="canonical_film_url", how="left")


def build_threshold_grid(
    frame: pd.DataFrame,
    score_column: str,
) -> np.ndarray:
    values = frame[score_column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    lower = math.floor(values.min() * 10.0) / 10.0
    upper = math.ceil(values.max() * 10.0) / 10.0
    return np.round(np.arange(lower, upper + 0.001, 0.1), 1)


def build_threshold_curve(
    frame: pd.DataFrame,
    *,
    score_column: str = "average_rating",
    rating_column: str = "member_rating",
    liked_threshold: float = 4.0,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame[[score_column, rating_column]].dropna().copy()
    if working.empty:
        return pd.DataFrame()
    baseline_mean = float(working[rating_column].mean())
    thresholds = build_threshold_grid(working, score_column)
    rows: list[dict[str, Any]] = []
    total_rows = len(working)
    for threshold in thresholds:
        kept = working.loc[working[score_column] >= threshold].copy()
        if kept.empty:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "kept_rows": int(len(kept)),
                "keep_rate": float(len(kept) / total_rows),
                "mean_member_rating": float(kept[rating_column].mean()),
                "gain_vs_all": float(kept[rating_column].mean() - baseline_mean),
                "liked_rate": float((kept[rating_column] >= liked_threshold).mean()),
                "mean_crowd_rating": float(kept[score_column].mean()),
                "baseline_mean_member_rating": baseline_mean,
            }
        )
    return pd.DataFrame(rows)


def build_percentile_curve(
    frame: pd.DataFrame,
    *,
    score_column: str = "average_rating",
    rating_column: str = "member_rating",
    liked_threshold: float = 4.0,
    keep_fractions: np.ndarray | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame[[score_column, rating_column]].dropna().copy()
    if working.empty:
        return pd.DataFrame()
    if keep_fractions is None:
        keep_fractions = np.linspace(0.2, 1.0, 17)
    working = working.sort_values(score_column, ascending=False).reset_index(drop=True)
    baseline_mean = float(working[rating_column].mean())
    total_rows = len(working)
    rows: list[dict[str, Any]] = []
    for keep_fraction in keep_fractions:
        keep_count = max(1, math.ceil(total_rows * float(keep_fraction)))
        kept = working.head(keep_count)
        rows.append(
            {
                "keep_fraction": float(keep_fraction),
                "drop_fraction": float(1.0 - keep_fraction),
                "kept_rows": int(keep_count),
                "cutoff_score": float(kept[score_column].iloc[-1]),
                "mean_member_rating": float(kept[rating_column].mean()),
                "gain_vs_all": float(kept[rating_column].mean() - baseline_mean),
                "liked_rate": float((kept[rating_column] >= liked_threshold).mean()),
                "baseline_mean_member_rating": baseline_mean,
            }
        )
    return pd.DataFrame(rows)


def summarize_profile_curves(
    threshold_curve: pd.DataFrame,
    percentile_curve: pd.DataFrame,
    *,
    profile_slug: str,
    display_name: str,
    min_keep_rate: float = 0.1,
) -> dict[str, Any]:
    profile_summary: dict[str, Any] = {
        "profile_slug": profile_slug,
        "display_name": display_name,
        "n_rated_films": 0,
        "baseline_mean_member_rating": np.nan,
        "best_threshold": np.nan,
        "best_threshold_keep_rate": np.nan,
        "best_threshold_gain": np.nan,
        "best_percentile_drop_fraction": np.nan,
        "best_percentile_cutoff_score": np.nan,
        "best_percentile_gain": np.nan,
    }
    if threshold_curve.empty or percentile_curve.empty:
        return profile_summary
    filtered_thresholds = threshold_curve.loc[
        threshold_curve["keep_rate"] >= min_keep_rate
    ].copy()
    filtered_percentiles = percentile_curve.loc[
        percentile_curve["keep_fraction"] >= min_keep_rate
    ].copy()
    if filtered_thresholds.empty or filtered_percentiles.empty:
        return profile_summary
    best_threshold = filtered_thresholds.sort_values(
        ["gain_vs_all", "threshold"],
        ascending=[False, True],
    ).iloc[0]
    best_percentile = filtered_percentiles.sort_values(
        ["gain_vs_all", "drop_fraction"],
        ascending=[False, True],
    ).iloc[0]
    profile_summary.update(
        {
            "n_rated_films": int(threshold_curve["kept_rows"].max()),
            "baseline_mean_member_rating": float(
                threshold_curve["baseline_mean_member_rating"].iloc[0]
            ),
            "best_threshold": float(best_threshold["threshold"]),
            "best_threshold_keep_rate": float(best_threshold["keep_rate"]),
            "best_threshold_gain": float(best_threshold["gain_vs_all"]),
            "best_percentile_drop_fraction": float(best_percentile["drop_fraction"]),
            "best_percentile_cutoff_score": float(best_percentile["cutoff_score"]),
            "best_percentile_gain": float(best_percentile["gain_vs_all"]),
        }
    )
    return profile_summary


def build_profile_summary(
    enriched_ratings: pd.DataFrame,
    threshold_curves: pd.DataFrame,
    percentile_curves: pd.DataFrame,
    *,
    min_keep_rate: float = 0.1,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if enriched_ratings.empty:
        return pd.DataFrame(rows)
    for (profile_slug, display_name), profile_frame in enriched_ratings.groupby(
        ["profile_slug", "display_name"]
    ):
        threshold_curve = threshold_curves.loc[
            threshold_curves["profile_slug"] == profile_slug
        ].copy()
        percentile_curve = percentile_curves.loc[
            percentile_curves["profile_slug"] == profile_slug
        ].copy()
        summary = summarize_profile_curves(
            threshold_curve,
            percentile_curve,
            profile_slug=profile_slug,
            display_name=display_name,
            min_keep_rate=min_keep_rate,
        )
        summary["n_rated_films"] = int(len(profile_frame))
        rows.append(summary)
    return pd.DataFrame(rows)


def aggregate_curve_metrics(
    curve: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    keep_column = "keep_rate" if "keep_rate" in curve.columns else "keep_fraction"
    aggregated = (
        curve.groupby(group_column)
        .agg(
            profiles=("profile_slug", "nunique"),
            mean_gain=("gain_vs_all", "mean"),
            median_gain=("gain_vs_all", "median"),
            mean_keep_rate=(keep_column, "mean"),
            mean_member_rating=("mean_member_rating", "mean"),
        )
        .reset_index()
    )
    return aggregated


def plot_threshold_tradeoffs(curve: pd.DataFrame, output_path: Path) -> None:
    profiles = curve["profile_slug"].drop_duplicates().tolist()
    n_profiles = len(profiles)
    n_cols = 2 if n_profiles > 1 else 1
    n_rows = math.ceil(n_profiles / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4.5 * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for axis, profile_slug in zip(axes.flat, profiles, strict=False):
        subset = curve.loc[curve["profile_slug"] == profile_slug].copy()
        axis.plot(
            subset["threshold"],
            subset["mean_member_rating"],
            marker="o",
            markersize=3,
        )
        baseline = float(subset["baseline_mean_member_rating"].iloc[0])
        axis.axhline(baseline, color="#555", linestyle="--", linewidth=1)
        axis.set_title(profile_slug)
        axis.set_xlabel("Minimum Letterboxd average rating")
        axis.set_ylabel("Mean personal rating kept")
        axis.grid(alpha=0.25)
    for axis in axes.flat[n_profiles:]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_percentile_tradeoffs(curve: pd.DataFrame, output_path: Path) -> None:
    profiles = curve["profile_slug"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile_slug in profiles:
        subset = curve.loc[curve["profile_slug"] == profile_slug].copy()
        ax.plot(
            subset["drop_fraction"] * 100.0,
            subset["mean_member_rating"],
            marker="o",
            markersize=3,
            label=profile_slug,
        )
    ax.set_xlabel("Dropped lowest-average films (%)")
    ax.set_ylabel("Mean personal rating kept")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_letterboxd_profiles(
    profile_slugs: list[str],
    *,
    output_dir: Path,
    refresh_film_cache: bool = False,
    sleep_seconds: float = 0.0,
    min_keep_rate: float = 0.1,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = build_requests_session()
    rss_rows: list[pd.DataFrame] = []
    for profile_slug in profile_slugs:
        xml_text = fetch_profile_rss(profile_slug, session)
        rss_rows.append(parse_profile_rss(xml_text, profile_slug=profile_slug))
    rss_df = pd.concat(rss_rows, ignore_index=True) if rss_rows else pd.DataFrame()
    deduped_ratings = dedupe_latest_ratings(rss_df)
    enriched = enrich_with_film_metadata(
        deduped_ratings,
        cache_path=output_dir / "film_metadata_cache.json",
        refresh=refresh_film_cache,
        sleep_seconds=sleep_seconds,
    )

    threshold_frames: list[pd.DataFrame] = []
    percentile_frames: list[pd.DataFrame] = []
    if not enriched.empty:
        for (profile_slug, display_name), profile_frame in enriched.groupby(
            ["profile_slug", "display_name"]
        ):
            threshold_curve = build_threshold_curve(profile_frame)
            if not threshold_curve.empty:
                threshold_curve.insert(0, "display_name", display_name)
                threshold_curve.insert(0, "profile_slug", profile_slug)
                threshold_frames.append(threshold_curve)
            percentile_curve = build_percentile_curve(profile_frame)
            if not percentile_curve.empty:
                percentile_curve.insert(0, "display_name", display_name)
                percentile_curve.insert(0, "profile_slug", profile_slug)
                percentile_frames.append(percentile_curve)

    threshold_curves = (
        pd.concat(threshold_frames, ignore_index=True)
        if threshold_frames
        else pd.DataFrame()
    )
    percentile_curves = (
        pd.concat(percentile_frames, ignore_index=True)
        if percentile_frames
        else pd.DataFrame()
    )
    profile_summary = build_profile_summary(
        enriched,
        threshold_curves,
        percentile_curves,
        min_keep_rate=min_keep_rate,
    )
    aggregate_thresholds = aggregate_curve_metrics(
        threshold_curves,
        group_column="threshold",
    )
    aggregate_percentiles = aggregate_curve_metrics(
        percentile_curves,
        group_column="drop_fraction",
    )

    rss_df.to_csv(output_dir / "profile_rss_items.csv", index=False)
    deduped_ratings.to_csv(output_dir / "profile_rated_films.csv", index=False)
    enriched.to_csv(output_dir / "profile_rated_films_enriched.csv", index=False)
    threshold_curves.to_csv(output_dir / "threshold_curves.csv", index=False)
    percentile_curves.to_csv(output_dir / "percentile_curves.csv", index=False)
    profile_summary.to_csv(output_dir / "profile_summary.csv", index=False)
    aggregate_thresholds.to_csv(
        output_dir / "aggregate_threshold_summary.csv",
        index=False,
    )
    aggregate_percentiles.to_csv(
        output_dir / "aggregate_percentile_summary.csv",
        index=False,
    )

    if not threshold_curves.empty:
        plot_threshold_tradeoffs(
            threshold_curves,
            output_dir / "letterboxd_threshold_tradeoffs.png",
        )
    if not percentile_curves.empty:
        plot_percentile_tradeoffs(
            percentile_curves,
            output_dir / "letterboxd_percentile_tradeoffs.png",
        )

    return {
        "rss_items": rss_df,
        "rated_films": deduped_ratings,
        "enriched_ratings": enriched,
        "threshold_curves": threshold_curves,
        "percentile_curves": percentile_curves,
        "profile_summary": profile_summary,
        "aggregate_threshold_summary": aggregate_thresholds,
        "aggregate_percentile_summary": aggregate_percentiles,
    }
