from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis_core.letterboxd_cutoff_analysis import (  # noqa: E402
    build_threshold_curve,
    canonicalize_film_url,
    dedupe_latest_ratings,
    parse_film_page_metadata,
    parse_profile_rss,
)


def test_canonicalize_film_url_strips_user_prefix() -> None:
    assert (
        canonicalize_film_url("https://letterboxd.com/brat/film/dead-poets-society/")
        == "https://letterboxd.com/film/dead-poets-society/"
    )


def test_parse_profile_rss_reads_namespaced_member_rating() -> None:
    xml_text = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"
    xmlns:atom="http://www.w3.org/2005/Atom"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:letterboxd="https://letterboxd.com"
    xmlns:tmdb="https://themoviedb.org">
  <channel>
    <title>Letterboxd - Example User</title>
    <item>
      <title>Dead Poets Society, 1989 - ★★★★</title>
      <link>https://letterboxd.com/example/film/dead-poets-society/</link>
      <guid isPermaLink="false">letterboxd-review-1</guid>
      <pubDate>Sun, 5 Apr 2026 19:36:56 +1200</pubDate>
      <letterboxd:watchedDate>2026-04-04</letterboxd:watchedDate>
      <letterboxd:rewatch>No</letterboxd:rewatch>
      <letterboxd:filmTitle>Dead Poets Society</letterboxd:filmTitle>
      <letterboxd:filmYear>1989</letterboxd:filmYear>
      <letterboxd:memberRating>4.0</letterboxd:memberRating>
      <letterboxd:memberLike>Yes</letterboxd:memberLike>
      <tmdb:movieId>207</tmdb:movieId>
      <description><![CDATA[<p>Watched on Friday March 27, 2026.</p>]]></description>
      <dc:creator>Example User</dc:creator>
    </item>
  </channel>
</rss>
"""
    frame = parse_profile_rss(xml_text, profile_slug="example")

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["display_name"] == "Example User"
    assert row["film_title"] == "Dead Poets Society"
    assert row["member_rating"] == 4.0
    assert row["tmdb_movie_id"] == 207
    assert row["canonical_film_url"] == (
        "https://letterboxd.com/film/dead-poets-society/"
    )


def test_dedupe_latest_ratings_keeps_most_recent_rewatch() -> None:
    frame = pd.DataFrame(
        [
            {
                "profile_slug": "example",
                "film_key": "207",
                "member_rating": 3.5,
                "watched_date": pd.Timestamp("2025-01-01"),
                "pub_date_utc": pd.Timestamp("2025-01-01", tz="UTC"),
            },
            {
                "profile_slug": "example",
                "film_key": "207",
                "member_rating": 4.0,
                "watched_date": pd.Timestamp("2026-01-01"),
                "pub_date_utc": pd.Timestamp("2026-01-01", tz="UTC"),
            },
        ]
    )

    deduped = dedupe_latest_ratings(frame)

    assert len(deduped) == 1
    assert deduped.iloc[0]["member_rating"] == 4.0


def test_parse_film_page_metadata_reads_letterboxd_json_ld() -> None:
    html_text = """
<html>
  <head>
    <script type="application/ld+json">
/* <![CDATA[ */
{"@type":"Movie","name":"Parasite","releasedEvent":[{"startDate":"2019"}],"aggregateRating":{"bestRating":5,"reviewCount":682581,"@type":"aggregateRating","ratingValue":4.53,"ratingCount":5213623,"worstRating":0}}
/* ]]> */
    </script>
  </head>
</html>
"""
    metadata = parse_film_page_metadata(
        html_text,
        canonical_film_url="https://letterboxd.com/film/parasite-2019/",
    )

    assert metadata.film_title == "Parasite"
    assert metadata.film_year == 2019
    assert metadata.average_rating == 4.53
    assert metadata.rating_count == 5213623
    assert metadata.review_count == 682581


def test_build_threshold_curve_computes_gain_vs_baseline() -> None:
    frame = pd.DataFrame(
        [
            {"average_rating": 2.5, "member_rating": 2.0},
            {"average_rating": 3.0, "member_rating": 3.0},
            {"average_rating": 4.0, "member_rating": 4.5},
            {"average_rating": 4.5, "member_rating": 5.0},
        ]
    )

    curve = build_threshold_curve(frame)
    threshold_row = curve.loc[curve["threshold"] == 4.0].iloc[0]

    assert threshold_row["kept_rows"] == 2
    assert threshold_row["mean_member_rating"] == 4.75
    assert threshold_row["gain_vs_all"] > 0
