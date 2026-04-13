from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis_core.movie_rt_analysis import (  # noqa: E402
    cross_validate_models,
    evaluate_models,
    evaluate_threshold_baselines,
    parse_movie_entries,
    parse_reviews_data,
    parse_search_results,
    simulate_data_quality_impact,
)
from analysis_core.movie_rt_followup_analysis import (  # noqa: E402
    bootstrap_threshold_uncertainty,
    build_provisional_full_imdb_dataset,
    build_gemini_comparison_table,
    compare_thresholds_by_keep_rate,
    compute_threshold_tradeoff,
    load_gemini_rating_tables,
    personal_bucket_3,
    personal_bucket_4,
    personal_bucket_5,
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
    assert math.isclose(
        results["linear_coefficients"]["critic_weight"], 0.35, abs_tol=1e-10
    )
    assert math.isclose(
        results["linear_coefficients"]["audience_weight"], 0.45, abs_tol=1e-10
    )
    assert results["linear_metrics"]["mae"] < 1e-10
    assert set(results["holdout_predictions"]["movie_title"]) == {"Movie 8", "Movie 9"}


def test_threshold_baselines_counts_confusion_matrix() -> None:
    df = pd.DataFrame(
        {
            "movie_title": ["A", "B", "C", "D"],
            "my_rating": [8.0, 3.0, 7.0, 4.0],
            "rt_audience_rating": [80, 40, 65, 75],
            "rt_critic_rating": [70, 55, 60, 80],
        }
    )
    df["rt_audience_on_10"] = df["rt_audience_rating"] / 10.0
    df["rt_critic_on_10"] = df["rt_critic_rating"] / 10.0
    df["rt_average_rating"] = df[["rt_audience_rating", "rt_critic_rating"]].mean(
        axis=1
    )
    df["rt_average_on_10"] = df["rt_average_rating"] / 10.0
    df["liked"] = df["my_rating"] >= 7.0

    results = evaluate_threshold_baselines(df, thresholds=(60,))
    audience_row = results.loc[
        (results["score_source"] == "audience") & (results["threshold"] == 60)
    ].iloc[0]
    assert audience_row["true_positive"] == 2
    assert audience_row["true_negative"] == 1
    assert audience_row["false_positive"] == 1
    assert audience_row["false_negative"] == 0
    assert math.isclose(audience_row["accuracy"], 0.75)


def test_cross_validate_models_and_sensitivity_are_deterministic() -> None:
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

    cv_results = cross_validate_models(df, n_splits=5, n_repeats=2, random_state=0)
    assert set(cv_results) == {
        "n_splits",
        "n_repeats",
        "regression_summary",
        "classification_summary",
    }
    assert set(cv_results["regression_summary"]["model"]) == {"average_rt", "linear_rt"}
    assert "logistic_rt" in set(cv_results["classification_summary"]["model"])

    sensitivity = simulate_data_quality_impact(
        df,
        deltas=(0,),
        iterations=20,
        random_state=0,
    )
    assert len(sensitivity) == 1
    assert math.isclose(
        sensitivity[0]["audience_correlation_percentiles"]["p50"],
        df["my_rating"].corr(df["rt_audience_on_10"]),
    )


def test_load_gemini_tables_and_comparison_match_verified_titles(
    tmp_path: Path,
) -> None:
    markdown = """
### The 25-Movie Dataset (CSV)

```
Movie Title,User Rating,RT Audience,RT Critic,IMDb,Letterboxd
Batman v Superman,7,63,29,65,52
```

```
Movie Title,User Rating,RT Audience,RT Critic,IMDb (Actual),Letterboxd (Actual)
Batman v Superman,7,63,28,65,54
```
## Prompt:
"""
    markdown_path = tmp_path / "gemini.md"
    markdown_path.write_text(markdown)
    tables = load_gemini_rating_tables(markdown_path)
    assert list(tables["approximate"]["Movie Title"]) == ["Batman v Superman"]
    assert list(tables["corrected"]["Movie Title"]) == ["Batman v Superman"]

    verified = pd.DataFrame(
        {
            "movie_title": ["Batman vs Superman: Dawn of Justice"],
            "my_rating": [7.0],
            "rt_audience_rating": [63],
            "rt_critic_rating": [28],
        }
    )
    comparison = build_gemini_comparison_table(
        tables["approximate"],
        tables["corrected"],
        verified,
    )
    assert (
        comparison.iloc[0]["verified_movie_title"]
        == "Batman vs Superman: Dawn of Justice"
    )
    assert comparison.iloc[0]["rt_critic_abs_error_vs_verified"] == 1
    assert comparison.iloc[0]["letterboxd_abs_error_vs_corrected"] == 2


def test_bucket_helpers_and_threshold_tradeoff() -> None:
    assert [personal_bucket_4(value) for value in [5, 6, 8, 10]] == [0, 1, 2, 3]
    assert [personal_bucket_3(value) for value in [5, 6, 8, 10]] == [0, 1, 2, 2]
    assert [personal_bucket_5(value) for value in [4, 5, 7, 8, 10]] == [1, 2, 3, 4, 5]

    df = pd.DataFrame(
        {
            "my_rating": [4.0, 6.0, 8.0, 10.0],
            "rt_audience_rating": [50, 65, 75, 95],
        }
    )
    tradeoff = compute_threshold_tradeoff(df, "rt_audience_rating", thresholds=[60, 80])
    assert list(tradeoff["kept_movies"]) == [3, 1]
    assert math.isclose(tradeoff.iloc[0]["gain_vs_all"], 1.0)
    assert math.isclose(tradeoff.iloc[1]["gain_vs_all"], 3.0)
    assert math.isclose(tradeoff.iloc[1]["kept_mean_rating"], 10.0)


def test_bootstrap_threshold_uncertainty_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "my_rating": [4.0, 6.0, 8.0, 10.0],
            "rt_audience_rating": [50, 65, 75, 95],
        }
    )
    summary = bootstrap_threshold_uncertainty(
        df,
        "rt_audience_rating",
        thresholds=[60, 80],
        iterations=40,
        random_state=0,
    )
    assert list(summary["threshold"]) == [60, 80]
    assert math.isclose(summary.iloc[0]["keep_rate"], 0.75)
    assert math.isclose(summary.iloc[1]["keep_rate"], 0.25)
    assert math.isclose(summary.iloc[0]["kept_mean_rating"], 8.0)
    assert math.isclose(summary.iloc[1]["kept_mean_rating"], 10.0)
    for metric in ("gain_vs_all", "liked_rate", "mean_bucket5"):
        assert summary.iloc[0][f"{metric}_p05"] <= summary.iloc[0][metric]
        assert summary.iloc[0][f"{metric}_p95"] >= summary.iloc[0][metric]


def test_build_provisional_imdb_dataset_and_keep_rate_matching() -> None:
    provisional = pd.DataFrame(
        {
            "Movie Title": ["Batman v Superman", "In Bruges"],
            "My Rating": [7.0, 8.0],
            "RT Audience": [63, 87],
            "RT Critic": [28, 84],
            "IMDb Rating": [65.0, 79.0],
            "Letterboxd Rating": [54.0, 78.0],
        }
    )
    verified = pd.DataFrame(
        {
            "movie_title": [
                "Batman vs Superman: Dawn of Justice",
                "In Bruges",
            ],
            "my_rating": [7.0, 8.0],
            "rt_audience_rating": [63, 87],
            "rt_critic_rating": [28, 84],
            "rt_average_rating": [45.5, 85.5],
        }
    )
    merged = build_provisional_full_imdb_dataset(provisional, verified)
    assert list(merged["movie_title"]) == [
        "Batman vs Superman: Dawn of Justice",
        "In Bruges",
    ]
    assert math.isclose(merged.iloc[0]["imdb_on_10"], 6.5)
    assert math.isclose(merged.iloc[1]["rt_audience_on_10"], 8.7)
    assert math.isclose(merged.iloc[0]["combo_avg_on_10"], 6.4)

    comparison = compare_thresholds_by_keep_rate(
        pd.DataFrame(
            {
                "threshold": [6.0, 7.0],
                "keep_rate": [0.80, 0.50],
                "gain_vs_all": [0.2, 0.8],
                "liked_rate": [0.60, 0.90],
            }
        ),
        pd.DataFrame(
            {
                "threshold": [5.5, 6.5],
                "keep_rate": [0.78, 0.52],
                "gain_vs_all": [0.1, 0.7],
                "liked_rate": [0.58, 0.88],
            }
        ),
        comparison_label="combo",
        reference_label="audience",
    )
    assert list(comparison["audience_threshold_nearest_keep"]) == [5.5, 6.5]
    assert math.isclose(comparison.iloc[0]["keep_pct_gap"], 2.0)
