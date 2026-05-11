from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
import re
import time
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd
from pandas.errors import EmptyDataError
import requests
from pathlib import Path

from ai_books_tracking.goodread_emperical_dist.profiles import GoodreadsProfile

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_html_text(value: str | None) -> str:
    if not value:
        return ""
    cleaned = unescape(str(value))
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def fetch_rss_page(
    profile: GoodreadsProfile,
    page: int,
    session: requests.Session,
) -> str:
    response = session.get(
        profile.rss_url_template.format(page=page),
        timeout=30,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    response.raise_for_status()
    return response.text


def parse_rss_page(
    xml_text: str,
    profile: GoodreadsProfile,
    page: int,
) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(tz=UTC).isoformat()
    for item in root.findall(".//item"):
        rows.append(
            {
                "profile_slug": profile.profile_slug,
                "display_name": profile.display_name,
                "user_id": profile.user_id,
                "goodreads_url": profile.goodreads_url,
                "rss_page": page,
                "fetched_at_utc": fetched_at,
                "guid": (item.findtext("guid") or "").strip(),
                "pub_date": _parse_date(item.findtext("pubDate")),
                "title": _clean_html_text(item.findtext("title")),
                "link": (item.findtext("link") or "").strip(),
                "book_id": (item.findtext("book_id") or "").strip(),
                "book_description": _clean_html_text(item.findtext("book_description")),
                "author_name": _clean_html_text(item.findtext("author_name")),
                "isbn": (item.findtext("isbn") or "").strip(),
                "user_name": _clean_html_text(item.findtext("user_name")),
                "user_rating": pd.to_numeric(
                    item.findtext("user_rating"), errors="coerce"
                ),
                "user_read_at": _parse_date(item.findtext("user_read_at")),
                "user_date_added": _parse_date(item.findtext("user_date_added")),
                "user_date_created": _parse_date(item.findtext("user_date_created")),
                "user_shelves": _clean_html_text(item.findtext("user_shelves")),
                "user_review": _clean_html_text(item.findtext("user_review")),
                "average_rating": pd.to_numeric(
                    item.findtext("average_rating"), errors="coerce"
                ),
                "book_published": pd.to_numeric(
                    item.findtext("book_published"), errors="coerce"
                ),
            }
        )
    return rows


def fetch_profile_books(
    profile: GoodreadsProfile,
    output_path,
    refresh: bool = False,
    max_pages: int | None = None,
    sleep_seconds: float = 0.1,
    verbose: bool = False,
) -> pd.DataFrame:
    output_path = Path(output_path)
    if not refresh and output_path.exists():
        try:
            existing = pd.read_csv(output_path)
        except EmptyDataError:
            output_path.unlink(missing_ok=True)
        else:
            existing.columns = existing.columns.str.strip()
            return existing

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    page = 1
    all_rows: list[dict[str, Any]] = []
    while True:
        if max_pages is not None and page > max_pages:
            break
        try:
            xml_text = fetch_rss_page(profile, page, session)
        except requests.RequestException:
            if verbose:
                print(
                    f"Stopping {profile.profile_slug} at page {page} due to fetch error"
                )
            break
        rows = parse_rss_page(xml_text, profile, page)
        if not rows:
            break
        all_rows.extend(rows)
        if verbose:
            print(
                f"Fetched {profile.profile_slug} page {page} "
                f"({len(rows)} items; total {len(all_rows)})"
            )
        page += 1
        time.sleep(sleep_seconds)

    frame = pd.DataFrame(all_rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["guid"], keep="first").reset_index(
            drop=True
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
