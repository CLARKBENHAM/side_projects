from __future__ import annotations

import math

import pandas as pd

from tree_leads.movie_rt_analysis import (
    evaluate_models,
    parse_movie_entries,
    parse_reviews_data,
    parse_search_results,
)


def test_parse_movie_entries_handles_multiline_notes() -> None:
    raw = """
Movie One 8; first note
continuation line

Movie Two 4; second note

stray comment
"""
    entries = parse_movie_entries(raw)
    assert [entry.title for entry in entries] == ["Movie One", "Movie Two"]
    assert entries[0].rating == 8
    assert entries[0].notes == "first note\ncontinuation line"
    assert entries[1].notes == "second note\nstray comment"


def test_parse_search_results_and_reviews_data_extract_scores() -> None:
    search_html = """
    <search-page-result skeleton="panel" type="movie" data-qa="search-result">
      <ul slot="list">
        <search-page-media-row cast="Actor One,Actor Two" release-year="2008" tomatometer-score="84">
          <a href="https://www.rottentomatoes.com/m/in_bruges" class="unset" data-qa="thumbnail-link" slot="thumbnail"></a>
          <a href="https://www.rottentomatoes.com/m/in_bruges" class="unset" data-qa="info-name" slot="title">
            In Bruges
          </a>
        </search-page-media-row>
      </ul>
    </search-page-result>
    """
    candidates = parse_search_results(search_html)
    assert len(candidates) == 1
    assert candidates[0].title == "In Bruges"
    assert candidates[0].release_year == 2008
    assert candidates[0].cast == ("Actor One", "Actor Two")
    assert candidates[0].tomatometer_score == 84

    movie_html = """
    <script type="application/json" data-json="reviewsData">
    {"audienceScore":{"score":"87"},"criticsScore":{"score":"84"},"title":"In Bruges"}
    </script>
    """
    reviews = parse_reviews_data(movie_html)
    assert reviews["audienceScore"]["score"] == "87"
    assert reviews["criticsScore"]["score"] == "84"
    assert reviews["title"] == "In Bruges"


def test_evaluate_models_uses_last_twenty_percent_holdout() -> None:
    critic_on_10 = [2.0, 3.0, 4.5, 5.0, 6.5, 7.5, 8.0, 9.0, 4.0, 8.5]
    audience_on_10 = [4.0, 4.5, 5.0, 6.5, 7.0, 8.0, 8.5, 9.5, 6.0, 5.0]
    my_rating = [
        1.2 + 0.35 * critic + 0.45 * audience
        for critic, audience in zip(critic_on_10, audience_on_10)
    ]

    df = pd.DataFrame(
        {
            "movie_title": [f"Movie {index}" for index in range(10)],
            "my_rating": my_rating,
            "rt_critic_on_10": critic_on_10,
            "rt_audience_on_10": audience_on_10,
        }
    )
    df["rt_critic_rating"] = df["rt_critic_on_10"] * 10
    df["rt_audience_rating"] = df["rt_audience_on_10"] * 10
    df["rt_average_on_10"] = df[["rt_critic_on_10", "rt_audience_on_10"]].mean(axis=1)
    df["rt_average_rating"] = df["rt_average_on_10"] * 10
    df["liked"] = df["my_rating"] >= 7.0

    results = evaluate_models(df)
    assert results["train_size"] == 8
    assert results["holdout_size"] == 2
    assert math.isclose(results["linear_coefficients"]["intercept"], 1.2, abs_tol=1e-10)
    assert math.isclose(results["linear_coefficients"]["critic_weight"], 0.35, abs_tol=1e-10)
    assert math.isclose(results["linear_coefficients"]["audience_weight"], 0.45, abs_tol=1e-10)
    assert results["linear_metrics"]["mae"] < 1e-10
    assert set(results["holdout_predictions"]["movie_title"]) == {"Movie 8", "Movie 9"}
