from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd
from pandas.errors import EmptyDataError
import requests

from ai_books_tracking.goodread_emperical_dist.config import (
    NETWORK_PROFILE_FEEDS_DIR,
    NETWORK_PROFILE_PAGE_CACHE_DIR,
)
from ai_books_tracking.goodread_emperical_dist.feature_engineering import (
    prepare_profile_books,
)
from ai_books_tracking.goodread_emperical_dist.goodreads_rss import (
    USER_AGENT,
    fetch_rss_page,
    parse_rss_page,
)
from ai_books_tracking.goodread_emperical_dist.profiles import (
    GoodreadsProfile,
    extract_user_id,
    load_profiles,
)

BASE_URL = "https://www.goodreads.com"
BOOKS_RE = re.compile(r"\((?P<count>[\d,]+)\s+books\)")
RATINGS_RE = re.compile(r"(?P<count>[\d,]+)\s+ratings\b", re.I)
REVIEWS_RE = re.compile(r"(?P<count>[\d,]+)\s+reviews\b", re.I)
FOLLOWERS_RE = re.compile(r"(?P<count>[\d,]+)\s+people are following\b", re.I)
FRIENDS_HEADER_RE = re.compile(r"Friends\s+\((?P<count>[\d,]+)\)", re.I)


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    digits = re.sub(r"[^\d]", "", value)
    if not digits:
        return None
    return int(digits)


def _slug_from_url(goodreads_url: str) -> str:
    tail = str(goodreads_url).rstrip("/").split("/")[-1]
    cleaned = re.sub(r"^\d+-?", "", tail).strip("-")
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned.lower()).strip("-")
    return cleaned or extract_user_id(goodreads_url)


@dataclass(frozen=True)
class PublicProfilePage:
    display_name: str
    profile_slug: str
    user_id: str
    goodreads_url: str
    books_on_goodreads: int | None
    public_ratings: int | None
    public_reviews: int | None
    followers_count: int | None
    friends_count: int | None
    visible_user_urls: tuple[str, ...]


def profile_page_cache_path(user_id: str) -> Path:
    return NETWORK_PROFILE_PAGE_CACHE_DIR / f"{user_id}.html"


def _fetch_profile_html(
    goodreads_url: str,
    session: requests.Session,
    refresh: bool = False,
) -> tuple[str, bool]:
    cache_path = profile_page_cache_path(extract_user_id(goodreads_url))
    if cache_path.exists() and not refresh:
        return cache_path.read_text(), True
    response = session.get(
        goodreads_url,
        timeout=30,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text)
    return response.text, False


def parse_public_profile_html(
    html_text: str,
    goodreads_url: str,
) -> PublicProfilePage:
    soup = BeautifulSoup(html_text, "html.parser")
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title_match = re.match(r"(?P<name>.+?)\s+\((?P<count>[\d,]+)\s+books\)", page_title)
    display_name = (
        title_match.group("name").strip()
        if title_match
        else soup.find("meta", attrs={"property": "og:title"})
        .get("content", "")
        .strip()
    )
    books_on_goodreads = (
        _parse_int(title_match.group("count"))
        if title_match
        else _parse_int(
            BOOKS_RE.search(page_title).group("count")
            if BOOKS_RE.search(page_title)
            else None
        )
    )

    visible_urls: list[str] = []
    self_url = f"/user/show/{extract_user_id(goodreads_url)}"
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if "/user/show/" not in href:
            continue
        full_url = urljoin(BASE_URL, href)
        if href.startswith(self_url):
            continue
        if full_url not in visible_urls:
            visible_urls.append(full_url)

    stripped_text = " ".join(soup.stripped_strings)
    ratings_match = RATINGS_RE.search(stripped_text)
    reviews_match = REVIEWS_RE.search(stripped_text)
    followers_match = FOLLOWERS_RE.search(stripped_text)
    friends_match = FRIENDS_HEADER_RE.search(stripped_text)
    return PublicProfilePage(
        display_name=display_name,
        profile_slug=_slug_from_url(goodreads_url),
        user_id=extract_user_id(goodreads_url),
        goodreads_url=goodreads_url,
        books_on_goodreads=books_on_goodreads,
        public_ratings=_parse_int(
            ratings_match.group("count") if ratings_match else None
        ),
        public_reviews=_parse_int(
            reviews_match.group("count") if reviews_match else None
        ),
        followers_count=_parse_int(
            followers_match.group("count") if followers_match else None
        ),
        friends_count=_parse_int(
            friends_match.group("count") if friends_match else None
        ),
        visible_user_urls=tuple(visible_urls),
    )


def crawl_public_profile_network(
    seed_profiles: list[GoodreadsProfile] | None = None,
    max_depth: int = 2,
    max_profiles: int = 600,
    refresh: bool = False,
    sleep_seconds: float = 0.1,
    verbose: bool = False,
) -> pd.DataFrame:
    profiles = seed_profiles or load_profiles()
    seed_url_map = {profile.goodreads_url: profile.profile_slug for profile in profiles}
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    queue: deque[tuple[str, int, str]] = deque(
        (profile.goodreads_url, 0, profile.profile_slug) for profile in profiles
    )
    seen_urls: set[str] = set()
    discovered_rows: list[dict[str, object]] = []

    while queue and len(seen_urls) < max_profiles:
        goodreads_url, depth, source_slug = queue.popleft()
        if goodreads_url in seen_urls or depth > max_depth:
            continue
        try:
            html_text, from_cache = _fetch_profile_html(
                goodreads_url=goodreads_url,
                session=session,
                refresh=refresh,
            )
            parsed = parse_public_profile_html(html_text, goodreads_url)
        except requests.RequestException:
            if verbose:
                print(f"Failed profile page fetch for {goodreads_url}")
            continue
        seen_urls.add(goodreads_url)
        discovered_rows.append(
            {
                **asdict(parsed),
                "discovery_depth": depth,
                "source_profile_slug": source_slug,
                "is_seed_profile": goodreads_url in seed_url_map,
                "visible_user_link_count": len(parsed.visible_user_urls),
            }
        )
        if verbose:
            print(
                f"Network crawl depth={depth} slug={parsed.profile_slug} "
                f"visible_links={len(parsed.visible_user_urls)}"
            )
        if depth == max_depth:
            continue
        for visible_url in parsed.visible_user_urls:
            if visible_url not in seen_urls:
                queue.append((visible_url, depth + 1, parsed.profile_slug))
        if not from_cache:
            time.sleep(sleep_seconds)

    frame = pd.DataFrame(discovered_rows)
    if frame.empty:
        return frame
    return (
        frame.sort_values(
            ["discovery_depth", "public_ratings", "books_on_goodreads", "display_name"],
            ascending=[True, False, False, True],
            na_position="last",
        )
        .drop_duplicates(subset=["user_id"], keep="first")
        .reset_index(drop=True)
    )


def _fetch_rss_until_two_years(
    profile: GoodreadsProfile,
    max_pages: int = 20,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    output_path = NETWORK_PROFILE_FEEDS_DIR / f"{profile.profile_slug}.csv"
    if output_path.exists():
        try:
            cached = pd.read_csv(output_path)
        except EmptyDataError:
            output_path.unlink(missing_ok=True)
        else:
            cached.columns = cached.columns.str.strip()
            return cached

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict[str, object]] = []
    for page in range(1, max_pages + 1):
        try:
            xml_text = fetch_rss_page(profile, page, session)
        except requests.RequestException:
            break
        page_rows = parse_rss_page(xml_text, profile, page)
        if not page_rows:
            break
        rows.extend(page_rows)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            prepared = prepare_profile_books(frame)
            if prepared["event_year"].dropna().nunique() >= 2:
                break
        time.sleep(sleep_seconds)

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def validate_network_candidates(
    discovered_profiles: pd.DataFrame,
    minimum_public_ratings: int = 100,
    minimum_years: int = 2,
    max_rss_pages: int = 20,
    sleep_seconds: float = 0.05,
    verbose: bool = False,
) -> pd.DataFrame:
    if discovered_profiles.empty:
        return pd.DataFrame()

    candidate_rows: list[dict[str, object]] = []
    for row in discovered_profiles.itertuples(index=False):
        meets_public_ratings = (
            pd.notna(row.public_ratings)
            and float(row.public_ratings) >= minimum_public_ratings
        )
        observed_years = 0
        year_span = math.nan
        observed_books = 0
        if meets_public_ratings:
            profile = GoodreadsProfile(
                display_name=row.display_name,
                profile_slug=row.profile_slug,
                goodreads_url=row.goodreads_url,
                user_id=str(row.user_id),
                books_on_goodreads=(
                    int(row.books_on_goodreads)
                    if pd.notna(row.books_on_goodreads)
                    else None
                ),
                public_ratings=int(row.public_ratings),
                public_reviews=(
                    int(row.public_reviews) if pd.notna(row.public_reviews) else None
                ),
                read_shelf_count=None,
                source_group="network_discovered",
                fit_notes="Visible public-profile network expansion",
            )
            raw_profile = _fetch_rss_until_two_years(
                profile,
                max_pages=max_rss_pages,
                sleep_seconds=sleep_seconds,
            )
            prepared = prepare_profile_books(raw_profile)
            observed_books = int(len(prepared))
            if not prepared.empty and "event_year" in prepared.columns:
                years = prepared["event_year"].dropna()
                observed_years = int(years.nunique()) if not years.empty else 0
                if observed_years >= 2:
                    year_span = float(int(years.max()) - int(years.min()))
            if verbose:
                print(
                    f"Validated {row.profile_slug}: public_ratings={row.public_ratings} "
                    f"observed_years={observed_years} observed_books={observed_books}"
                )

        passes_filter = meets_public_ratings and observed_years >= minimum_years
        candidate_rows.append(
            {
                **row._asdict(),
                "meets_minimum_public_ratings": meets_public_ratings,
                "observed_rss_books": observed_books,
                "observed_distinct_years": observed_years,
                "observed_year_span": year_span,
                "passes_minimum_years": observed_years >= minimum_years,
                "passes_network_filter": passes_filter,
            }
        )

    frame = pd.DataFrame(candidate_rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["passes_network_filter", "public_ratings", "observed_distinct_years"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def expand_network_until_target_passes(
    seed_profiles: list[GoodreadsProfile] | None = None,
    target_pass_count: int = 200,
    start_max_profiles: int = 200,
    profile_step_size: int = 200,
    max_profiles_cap: int = 4000,
    max_depth: int = 2,
    crawl_sleep_seconds: float = 0.5,
    validation_sleep_seconds: float = 0.2,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = seed_profiles or load_profiles()
    current_max_profiles = max(len(profiles), start_max_profiles)
    latest_discovered = pd.DataFrame()
    latest_validated = pd.DataFrame()

    while True:
        latest_discovered = crawl_public_profile_network(
            seed_profiles=profiles,
            max_depth=max_depth,
            max_profiles=current_max_profiles,
            sleep_seconds=crawl_sleep_seconds,
            verbose=verbose,
        )
        latest_validated = validate_network_candidates(
            latest_discovered,
            sleep_seconds=validation_sleep_seconds,
            verbose=verbose,
        )
        pass_count = (
            int(latest_validated["passes_network_filter"].fillna(False).sum())
            if not latest_validated.empty
            else 0
        )
        if verbose:
            print(
                "Network expansion status: "
                f"discovered={len(latest_discovered)} "
                f"validated={len(latest_validated)} "
                f"passing={pass_count} "
                f"target={target_pass_count}"
            )
        if pass_count >= target_pass_count:
            break
        if len(latest_discovered) < current_max_profiles:
            break
        if current_max_profiles >= max_profiles_cap:
            break
        current_max_profiles = min(
            current_max_profiles + profile_step_size,
            max_profiles_cap,
        )

    return latest_discovered, latest_validated
