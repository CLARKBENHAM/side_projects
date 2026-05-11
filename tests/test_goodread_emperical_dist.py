from __future__ import annotations

import pandas as pd

from ai_books_tracking.goodread_emperical_dist.cross_profile_analysis import (
    compute_goodreads_rating_count_accuracy,
    compute_book_overlap_metrics,
    compute_reviewer_volume_metrics,
)
from ai_books_tracking.goodread_emperical_dist.evaluation import (
    _design_matrix,
    choose_holdout_split,
    drop_curve,
    expected_gain_from_correlation,
)
from ai_books_tracking.goodread_emperical_dist.feature_engineering import infer_category
from ai_books_tracking.goodread_emperical_dist.fixed_rule_analysis import (
    assign_fixed_rule_correlation_bands,
    compute_fixed_rule_outputs,
    compute_fixed_rule_parameter_comparison,
    filter_fixed_rule_profiles_by_percentile_window,
)
from ai_books_tracking.goodread_emperical_dist.policy_analysis import (
    ProfileSplitBundle,
    build_goodreads_cutoff_sweep,
    build_cutoff_curve,
    choose_threshold_for_target,
    select_small_multiples_profiles,
    summarize_cutoff_sweep,
)
from ai_books_tracking.goodread_emperical_dist.profile_network import (
    parse_public_profile_html,
)
from ai_books_tracking.goodread_emperical_dist.profiles import (
    build_analysis_manifest_frame,
    extract_user_id,
)
from ai_books_tracking.goodread_emperical_dist.temporal_analysis import (
    compute_temporal_picking_profile_metrics,
)
from ai_books_tracking.goodread_emperical_dist.yearly_consistency import (
    compute_yearly_consistency_outputs,
)


def test_extract_user_id_reads_numeric_prefix() -> None:
    assert (
        extract_user_id("https://www.goodreads.com/user/show/11004626-gwern")
        == "11004626"
    )


def test_infer_category_prefers_technical_keywords() -> None:
    category = infer_category(
        title="Designing Data-Intensive Applications",
        author_name="Martin Kleppmann",
        user_shelves="distributed-systems, programming",
        description="A software systems book about databases and distributed systems.",
    )
    assert category == "Computer Science"


def test_infer_category_data_analysis_prefers_business_context() -> None:
    category = infer_category(
        title="Data Analysis for Business",
        author_name="Foster Provost",
        user_shelves="business, analytics",
        description="A business book about decision-making and data-driven management.",
    )
    assert category == "Business, management"


def test_choose_holdout_split_prefers_recent_calendar_year() -> None:
    early_rows = [
        {
            "profile_slug": "demo",
            "display_name": "Demo",
            "title": f"Early {idx}",
            "user_rating": 3 + idx % 2,
            "average_rating": 3.5,
            "event_date": pd.Timestamp(f"2022-01-{(idx % 28) + 1:02d}", tz="UTC"),
            "event_year": 2022,
        }
        for idx in range(70)
    ]
    recent_rows = [
        {
            "profile_slug": "demo",
            "display_name": "Demo",
            "title": f"Recent {idx}",
            "user_rating": 4,
            "average_rating": 4.1,
            "event_date": pd.Timestamp(f"2025-02-{(idx % 28) + 1:02d}", tz="UTC"),
            "event_year": 2025,
        }
        for idx in range(25)
    ]
    frame = pd.DataFrame(early_rows + recent_rows)

    split = choose_holdout_split(frame)

    assert split is not None
    _, holdout, info = split
    assert info.split_type == "calendar_year"
    assert info.holdout_label == "2025"
    assert len(holdout) == 25


def test_choose_holdout_split_excludes_future_years_from_calendar_train() -> None:
    rows = []
    for idx in range(70):
        rows.append(
            {
                "profile_slug": "demo",
                "display_name": "Demo",
                "title": f"Old {idx}",
                "user_rating": 3 + idx % 2,
                "average_rating": 3.8,
                "event_date": pd.Timestamp(f"2022-01-{(idx % 28) + 1:02d}", tz="UTC"),
                "event_year": 2022,
            }
        )
    for idx in range(25):
        rows.append(
            {
                "profile_slug": "demo",
                "display_name": "Demo",
                "title": f"Holdout {idx}",
                "user_rating": 4,
                "average_rating": 4.0,
                "event_date": pd.Timestamp(f"2025-02-{(idx % 28) + 1:02d}", tz="UTC"),
                "event_year": 2025,
            }
        )
    for idx in range(10):
        rows.append(
            {
                "profile_slug": "demo",
                "display_name": "Demo",
                "title": f"Future {idx}",
                "user_rating": 5,
                "average_rating": 4.2,
                "event_date": pd.Timestamp(f"2026-03-{(idx % 28) + 1:02d}", tz="UTC"),
                "event_year": 2026,
            }
        )
    frame = pd.DataFrame(rows)

    split = choose_holdout_split(frame)

    assert split is not None
    train, holdout, info = split
    assert info.split_type == "calendar_year"
    assert info.holdout_label == "2025"
    assert set(train["event_year"].unique()) == {2022}
    assert holdout["event_year"].eq(2025).all()


def test_drop_curve_computes_rating_gain_for_sorted_scores() -> None:
    actual = pd.Series([1, 2, 3, 4, 5], dtype=float)
    score = pd.Series([1, 2, 3, 4, 5], dtype=float)

    curve = drop_curve(actual, score, drop_fractions=(0.4,))

    assert len(curve) == 1
    row = curve.iloc[0]
    assert row["n_kept"] == 3
    assert row["rating_gain"] == 1.0


def test_drop_curve_uses_non_actual_tie_breaking() -> None:
    actual = pd.Series([1, 5, 3, 4], dtype=float)
    score = pd.Series([1, 1, 2, 2], dtype=float)

    curve = drop_curve(actual, score, drop_fractions=(0.25,))

    row = curve.iloc[0]
    assert round(float(row["rating_gain"]), 3) == -0.583


def test_expected_gain_from_correlation_matches_known_drop50_factor() -> None:
    gain = expected_gain_from_correlation(std_dev=1.0, correlation=0.3, keep_share=0.5)
    assert round(gain, 3) == 0.239


def test_choose_threshold_for_target_prefers_matching_drop_rate() -> None:
    actual = pd.Series([1, 2, 4, 5], dtype=float)
    goodreads = pd.Series([3.0, 3.2, 4.0, 4.5], dtype=float)

    curve = build_cutoff_curve(actual, goodreads)
    choice = choose_threshold_for_target(curve, target_drop_fraction=0.5)

    assert choice["threshold"] == 4.0
    assert choice["train_drop_share"] == 0.5
    assert choice["train_gain"] == 1.5


def test_select_small_multiples_profiles_ranks_by_temporal_quality() -> None:
    base_frame = pd.DataFrame(
        {
            "profile_slug": ["demo"],
            "display_name": ["Demo"],
            "title": ["Book"],
            "user_rating": [4.0],
            "average_rating": [4.1],
            "event_date": [pd.Timestamp("2025-01-01", tz="UTC")],
            "event_year": [2025],
        }
    )
    split_stub = choose_holdout_split(
        pd.concat(
            [
                base_frame.assign(
                    title=[f"Book {idx}"],
                    event_date=[
                        pd.Timestamp(f"2023-01-{(idx % 28) + 1:02d}", tz="UTC")
                    ],
                    event_year=[2023],
                )
                for idx in range(85)
            ]
            + [
                base_frame.assign(
                    title=[f"Recent {idx}"],
                    event_date=[
                        pd.Timestamp(f"2025-02-{(idx % 28) + 1:02d}", tz="UTC")
                    ],
                    event_year=[2025],
                )
                for idx in range(25)
            ],
            ignore_index=True,
        )
    )
    assert split_stub is not None
    train_df, holdout_df, split_info = split_stub

    bundles = [
        ProfileSplitBundle(
            profile_slug="best",
            display_name="Best",
            full_df=train_df,
            train_df=train_df,
            holdout_df=holdout_df,
            split_info=split_info,
            n_rated_books=110,
            latest_event_year=2025,
            holdout_year=2025,
            recency_gap=0.0,
            dominant_year_share=0.2,
        ),
        ProfileSplitBundle(
            profile_slug="mid",
            display_name="Mid",
            full_df=train_df,
            train_df=train_df,
            holdout_df=holdout_df,
            split_info=split_info,
            n_rated_books=200,
            latest_event_year=2025,
            holdout_year=2023,
            recency_gap=2.0,
            dominant_year_share=0.15,
        ),
        ProfileSplitBundle(
            profile_slug="worst",
            display_name="Worst",
            full_df=train_df,
            train_df=train_df,
            holdout_df=holdout_df,
            split_info=split_info,
            n_rated_books=400,
            latest_event_year=2025,
            holdout_year=2018,
            recency_gap=7.0,
            dominant_year_share=0.7,
        ),
    ]

    selection = select_small_multiples_profiles(bundles, max_profiles=2)

    assert selection["profile_slug"].tolist() == ["best", "mid"]


def test_summarize_cutoff_sweep_computes_gain_percentiles() -> None:
    profile_metrics = pd.DataFrame(
        {
            "profile_slug": ["a", "b", "c", "a", "b", "c"],
            "threshold": [3.8, 3.8, 3.8, 4.0, 4.0, 4.0],
            "actual_drop_share": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "rating_gain": [0.0, 0.2, 0.4, 0.1, 0.3, 0.5],
        }
    )

    summary = summarize_cutoff_sweep(profile_metrics, gain_percentiles=(15, 85))

    row_38 = summary.loc[summary["threshold"] == 3.8].iloc[0]
    row_40 = summary.loc[summary["threshold"] == 4.0].iloc[0]

    assert round(row_38["drop_share_median"], 3) == 0.2
    assert round(row_38["drop_p15"], 3) == 0.13
    assert round(row_38["drop_p85"], 3) == 0.27
    assert round(row_38["gain_p15"], 3) == 0.06
    assert round(row_38["gain_p85"], 3) == 0.34
    assert round(row_40["drop_share_median"], 3) == 0.5
    assert round(row_40["drop_p15"], 3) == 0.43
    assert round(row_40["drop_p85"], 3) == 0.57
    assert round(row_40["gain_p15"], 3) == 0.16
    assert round(row_40["gain_p85"], 3) == 0.44


def test_parse_public_profile_html_extracts_stats_and_visible_users() -> None:
    html = """
    <html>
      <head>
        <title>Demo Person (314 books)</title>
        <meta property="og:title" content="Demo Person" />
      </head>
      <body>
        <div>219 ratings</div>
        <div>53 reviews</div>
        <div>2983 people are following Demo Person</div>
        <div>Friends (351)</div>
        <a href="/user/show/123-demo-person">Self</a>
        <a href="/user/show/456-friend-one">Friend One</a>
        <a href="/user/show/789-friend-two">Friend Two</a>
      </body>
    </html>
    """

    parsed = parse_public_profile_html(
        html,
        "https://www.goodreads.com/user/show/123-demo-person",
    )

    assert parsed.display_name == "Demo Person"
    assert parsed.books_on_goodreads == 314
    assert parsed.public_ratings == 219
    assert parsed.public_reviews == 53
    assert parsed.followers_count == 2983
    assert parsed.friends_count == 351
    assert parsed.visible_user_urls == (
        "https://www.goodreads.com/user/show/456-friend-one",
        "https://www.goodreads.com/user/show/789-friend-two",
    )


def test_compute_temporal_picking_profile_metrics_detects_improving_taste() -> None:
    frame = pd.DataFrame(
        {
            "profile_slug": ["demo"] * 120,
            "display_name": ["Demo"] * 120,
            "title": [f"Book {idx}" for idx in range(120)],
            "user_rating": [3.0 + 0.01 * idx for idx in range(120)],
            "average_rating": [3.5 + 0.005 * idx for idx in range(120)],
            "event_date": pd.date_range(
                "2020-01-01",
                periods=120,
                freq="15D",
                tz="UTC",
            ),
            "event_year": [2020 + (idx // 30) for idx in range(120)],
        }
    )

    metrics = compute_temporal_picking_profile_metrics(frame)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["passes_temporal_filter"]
    assert row["user_rating_slope_full_span"] > 1.0
    assert row["goodreads_rating_slope_full_span"] > 0.5
    assert row["user_rating_late_minus_early"] > 0.8
    assert row["goodreads_rating_late_minus_early"] > 0.4


def test_build_analysis_manifest_frame_adds_network_profiles_up_to_target() -> None:
    seed_frame = pd.DataFrame(
        {
            "display_name": ["Seed One", "Seed Two"],
            "profile_slug": ["seed-one", "seed-two"],
            "goodreads_url": [
                "https://www.goodreads.com/user/show/1-seed-one",
                "https://www.goodreads.com/user/show/2-seed-two",
            ],
            "books_on_goodreads": [120, 150],
            "public_ratings": [110, 130],
            "public_reviews": [5, 8],
            "read_shelf_count": [120, 150],
            "source_group": ["seed", "seed"],
            "fit_notes": ["", ""],
        }
    )
    network_frame = pd.DataFrame(
        {
            "display_name": ["Net A", "Net B", "Net C", "Seed One Copy"],
            "profile_slug": ["net-a", "net-a", "net-c", "seed-one-copy"],
            "goodreads_url": [
                "https://www.goodreads.com/user/show/10-net-a",
                "https://www.goodreads.com/user/show/11-net-b",
                "https://www.goodreads.com/user/show/12-net-c",
                "https://www.goodreads.com/user/show/1-seed-one-copy",
            ],
            "books_on_goodreads": [500, 400, 300, 999],
            "public_ratings": [500, 450, 200, 999],
            "public_reviews": [10, 9, 7, 99],
            "followers_count": [50, 100, 20, 1000],
            "source_profile_slug": ["seed-one", "seed-one", "seed-two", "seed-one"],
            "discovery_depth": [1, 2, 1, 1],
            "observed_distinct_years": [6, 4, 8, 10],
            "passes_network_filter": [True, True, True, True],
        }
    )

    manifest = build_analysis_manifest_frame(
        seed_frame=seed_frame,
        network_validated_frame=network_frame,
        include_network_profiles=True,
        analysis_target_total=4,
    )

    assert set(manifest["profile_slug"]) == {"seed-one", "seed-two", "net-a", "net-c"}
    assert "seed-one-copy" not in manifest["profile_slug"].tolist()
    assert "net-a-11" not in manifest["profile_slug"].tolist()


def test_build_goodreads_cutoff_sweep_all_books_does_not_require_holdout_split() -> (
    None
):
    frame = pd.DataFrame(
        {
            "profile_slug": ["demo"] * 12,
            "display_name": ["Demo"] * 12,
            "title": [f"Book {idx}" for idx in range(12)],
            "user_rating": [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 3.0, 4.0, 2.0, 5.0, 4.0, 3.0],
            "average_rating": [
                3.5,
                4.1,
                3.2,
                4.5,
                3.6,
                4.0,
                3.7,
                4.2,
                3.1,
                4.6,
                4.3,
                3.8,
            ],
            "event_date": pd.date_range("2025-01-01", periods=12, freq="7D", tz="UTC"),
            "event_year": [2025] * 12,
        }
    )

    holdout_outputs = build_goodreads_cutoff_sweep(frame, thresholds=(3.8,))
    all_books_outputs = build_goodreads_cutoff_sweep(
        frame,
        thresholds=(3.8,),
        evaluation_scope="all_books",
    )

    assert holdout_outputs["cutoff_sweep_profiles"].empty
    assert len(all_books_outputs["cutoff_sweep_profiles"]) == 1
    row = all_books_outputs["cutoff_sweep_profiles"].iloc[0]
    assert row["evaluation_scope"] == "all_books"
    assert row["split_type"] == "all_books"
    assert row["holdout_label"] == "all_books"


def test_design_matrix_fills_all_nan_goodreads_average() -> None:
    frame = pd.DataFrame(
        {
            "average_rating": [float("nan"), float("nan")],
            "category": ["General Reading", "Histories"],
        }
    )

    matrix, columns = _design_matrix(frame, include_category=True)

    assert columns[0] == "goodreads_avg_rating"
    assert matrix["goodreads_avg_rating"].eq(0.0).all()
    assert matrix.notna().all().all()


def test_compute_reviewer_volume_metrics_counts_profiles_and_years() -> None:
    frame = pd.DataFrame(
        {
            "profile_slug": ["a", "a", "a", "b", "b"],
            "display_name": ["A", "A", "A", "B", "B"],
            "title": ["t1", "t2", "t3", "u1", "u2"],
            "event_year": [2024, 2024, 2025, 2025, 2026],
        }
    )

    outputs = compute_reviewer_volume_metrics(frame)

    summary = outputs["reviewer_volume_summary"].set_index("profile_slug")
    assert summary.loc["a", "n_books"] == 3
    assert summary.loc["a", "n_active_years"] == 2
    assert summary.loc["b", "books_per_active_year"] == 1.0
    assert len(outputs["reviewer_year_counts"]) == 4


def test_compute_book_overlap_metrics_tracks_overlap_and_pairwise_correlation() -> None:
    frame = pd.DataFrame(
        {
            "book_id": ["x", "x", "x", "y", "y", "y", "z", "z", "z"],
            "title": [
                "Book X",
                "Book X",
                "Book X",
                "Book Y",
                "Book Y",
                "Book Y",
                "Book Z",
                "Book Z",
                "Book Z",
            ],
            "author_name": ["Auth"] * 9,
            "profile_slug": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
            "display_name": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "user_rating": [1, 2, 3, 2, 4, 2, 3, 6, 1],
            "average_rating": [4.0] * 9,
        }
    )

    outputs = compute_book_overlap_metrics(
        frame,
        minimum_shared_books_for_pair_corr=3,
    )

    overlap_books = outputs["overlap_books"].set_index("book_id")
    reviewer_pairs = outputs["reviewer_pair_correlations"]
    summary = outputs["overlap_rater_count_summary"].set_index("n_raters")

    assert reviewer_pairs["n_shared_books"].max() == 3
    assert round(float(overlap_books.loc["x", "rating_sd"]), 3) == 1.0
    assert round(float(overlap_books.loc["x", "mean_pairwise_profile_corr"]), 3) == (
        -0.333
    )
    assert summary.loc[3, "n_books"] == 3


def test_compute_goodreads_rating_count_accuracy_bins_by_total_rating_count() -> None:
    rows: list[dict[str, object]] = []
    for book_id, count, goodreads_rating, sample_rating in [
        ("low_a", 10, 1.0, 5.0),
        ("low_b", 20, 2.0, 4.0),
        ("low_c", 30, 3.0, 3.0),
        ("high_a", 1000, 1.0, 1.0),
        ("high_b", 2000, 2.0, 2.0),
        ("high_c", 3000, 3.0, 3.0),
    ]:
        for profile_slug in ["reader_1", "reader_2"]:
            rows.append(
                {
                    "book_id": book_id,
                    "title": book_id,
                    "author_name": "Auth",
                    "profile_slug": profile_slug,
                    "display_name": profile_slug,
                    "user_rating": sample_rating,
                    "average_rating": goodreads_rating,
                }
            )
    frame = pd.DataFrame(rows)
    counts = pd.DataFrame(
        {
            "book_id": ["low_a", "low_b", "low_c", "high_a", "high_b", "high_c"],
            "goodreads_rating_count": [10, 20, 30, 1000, 2000, 3000],
        }
    )

    outputs = compute_goodreads_rating_count_accuracy(
        frame,
        counts,
        n_bins=2,
        minimum_books_per_bin=1,
        bootstrap_samples=50,
        random_seed=42,
    )

    summary = outputs["rating_count_bin_summary"].sort_values("rating_count_median")

    assert summary["rating_count_median"].tolist() == [20.0, 2000.0]
    assert summary["n_books"].tolist() == [3, 3]
    assert float(summary.iloc[0]["row_spearman_rho"]) < 0
    assert float(summary.iloc[1]["row_spearman_rho"]) > 0.9
    assert float(summary.iloc[1]["mean_book_absolute_error"]) == 0.0
    assert "book_mean_spearman_rho_p80_low" in summary.columns
    assert "book_mean_spearman_rho_p95_high" in summary.columns
    assert (
        float(summary.iloc[1]["book_mean_spearman_rho_p80_low"])
        <= float(summary.iloc[1]["book_mean_spearman_rho"])
        <= float(summary.iloc[1]["book_mean_spearman_rho_p80_high"])
    )


def test_compute_yearly_consistency_outputs_reports_stable_transfer() -> None:
    rows: list[dict[str, object]] = []
    for year in [2024, 2025]:
        for idx in range(40):
            rating = 1.0 if idx < 20 else 5.0
            score = 3.0 + idx * 0.05
            rows.append(
                {
                    "profile_slug": "demo",
                    "display_name": "Demo",
                    "title": f"{year}-{idx}",
                    "user_rating": rating,
                    "average_rating": score,
                    "event_date": pd.Timestamp(f"{year}-01-01", tz="UTC")
                    + pd.Timedelta(days=idx),
                    "event_year": year,
                }
            )
    frame = pd.DataFrame(rows)

    outputs = compute_yearly_consistency_outputs(
        frame,
        minimum_books_per_year=30,
    )

    transfer = outputs["yearly_transfer_metrics"].iloc[0]
    gain_differences = outputs["yearly_gain_differences"]
    correlations = outputs["yearly_correlation_pairs"].iloc[0]

    assert transfer["year_1"] == 2024
    assert transfer["year_2"] == 2025
    assert round(float(transfer["year_2_optimal_efficiency"]), 3) == 1.0
    assert gain_differences["gain_difference_year_2_minus_year_1"].abs().max() == 0.0
    assert float(correlations["spearman_rho_year_1"]) > 0.86
    assert float(correlations["spearman_rho_year_2"]) > 0.86


def test_compute_fixed_rule_outputs_compares_cutoff_and_lower_half() -> None:
    frame = pd.DataFrame(
        {
            "profile_slug": ["demo"] * 6,
            "display_name": ["Demo"] * 6,
            "title": [f"Book {idx}" for idx in range(6)],
            "book_id": [f"id-{idx}" for idx in range(6)],
            "user_rating": [1, 2, 3, 4, 5, 5],
            "average_rating": [3.4, 3.6, 3.8, 4.0, 4.2, 4.4],
            "category": ["General Reading"] * 6,
            "user_shelves": ["history"] * 6,
            "author_name": ["Author"] * 6,
        }
    )

    outputs = compute_fixed_rule_outputs(frame)

    metrics = outputs["profile_metrics"].set_index("rule_name")
    shares = outputs["rating_shares"]
    summary = outputs["summary"].set_index("rule_name")

    assert metrics.loc["goodreads_cutoff_4_0", "n_kept"] == 3
    assert round(float(metrics.loc["goodreads_cutoff_4_0", "rating_gain"]), 3) == 1.333
    assert metrics.loc["drop_lower_goodreads_half", "n_kept"] == 3
    assert metrics.loc["drop_lower_goodreads_half", "actual_drop_share"] == 0.5
    assert (
        round(float(metrics.loc["drop_lower_goodreads_half", "rating_gain"]), 3)
        == 1.333
    )
    kept_five_share = shares[
        (shares["rule_name"] == "drop_lower_goodreads_half")
        & (shares["rating_value"] == 5)
    ]["share_of_books"].iloc[0]
    assert round(float(kept_five_share), 3) == 0.667
    assert summary.loc["comparison", "positive_gain_share"] == 0.0
    assert summary.loc["comparison", "negative_gain_share"] == 0.0


def test_compute_fixed_rule_outputs_handles_ties_by_count_not_threshold() -> None:
    frame = pd.DataFrame(
        {
            "profile_slug": ["demo"] * 5,
            "display_name": ["Demo"] * 5,
            "title": [f"Book {idx}" for idx in range(5)],
            "book_id": [f"id-{idx}" for idx in range(5)],
            "user_rating": [1, 2, 3, 4, 5],
            "average_rating": [3.8, 3.8, 3.8, 4.2, 4.2],
            "category": ["General Reading"] * 5,
            "user_shelves": [""] * 5,
            "author_name": ["Author"] * 5,
        }
    )

    outputs = compute_fixed_rule_outputs(frame)
    metrics = outputs["profile_metrics"].set_index("rule_name")

    assert metrics.loc["drop_lower_goodreads_half", "n_dropped"] == 2
    assert metrics.loc["drop_lower_goodreads_half", "n_kept"] == 3
    assert (
        round(float(metrics.loc["drop_lower_goodreads_half", "actual_drop_share"]), 3)
        == 0.4
    )


def test_compute_fixed_rule_parameter_comparison_summarizes_cutoff_vs_drop_rule() -> (
    None
):
    frame = pd.DataFrame(
        {
            "profile_slug": ["a"] * 6 + ["b"] * 6,
            "display_name": ["A"] * 6 + ["B"] * 6,
            "title": [f"Book {idx}" for idx in range(12)],
            "book_id": [f"id-{idx}" for idx in range(12)],
            "user_rating": [1, 1, 2, 4, 5, 5, 1, 2, 3, 4, 5, 5],
            "average_rating": [
                3.0,
                3.2,
                3.4,
                4.0,
                4.2,
                4.4,
                3.95,
                4.01,
                4.02,
                4.03,
                4.04,
                4.05,
            ],
            "category": ["General Reading"] * 12,
            "user_shelves": [""] * 12,
            "author_name": ["Author"] * 12,
        }
    )

    outputs = compute_fixed_rule_parameter_comparison(
        frame,
        cutoff_thresholds=(4.0,),
        drop_fractions=(1 / 3,),
    )

    summary = outputs["comparison_summary"].iloc[0]

    assert summary["n_profiles"] == 2
    assert round(float(summary["benefit_share_cutoff_beats_target_drop"]), 3) == 0.5
    assert (
        round(float(summary["mean_gain_difference_cutoff_minus_target_drop"]), 3)
        == 0.108
    )
    assert round(float(summary["gain_difference_p25"]), 3) == -0.171
    assert round(float(summary["gain_difference_p75"]), 3) == 0.388


def test_filter_fixed_rule_profiles_by_percentile_window_uses_profile_metric() -> None:
    rows: list[dict[str, object]] = []
    for idx, mean_rating in enumerate([2.0, 3.0, 4.0, 5.0], start=1):
        for rule_name in [
            "original_distribution",
            "goodreads_cutoff_4_0",
            "drop_lower_goodreads_half",
        ]:
            rows.append(
                {
                    "profile_slug": f"p{idx}",
                    "display_name": f"P{idx}",
                    "rule_name": rule_name,
                    "baseline_mean": mean_rating,
                    "profile_rating_std_all_books": 0.5 + idx,
                    "goodreads_spearman_rho_all_books": 0.1 * idx,
                }
            )
    profile_metrics = pd.DataFrame(rows)
    rating_shares = pd.DataFrame(
        {
            "profile_slug": ["p1", "p2", "p3", "p4"] * 3,
            "rule_name": (
                ["original_distribution"] * 4
                + ["goodreads_cutoff_4_0"] * 4
                + ["drop_lower_goodreads_half"] * 4
            ),
            "rating_value": [5] * 12,
            "share_percent": [100.0] * 12,
        }
    )

    outputs = filter_fixed_rule_profiles_by_percentile_window(
        profile_metrics,
        rating_shares,
        metric_column="baseline_mean",
        lower_quantile=0.25,
        upper_quantile=0.75,
    )

    assert set(outputs["profile_metrics"]["profile_slug"]) == {"p2", "p3"}
    assert set(outputs["rating_shares"]["profile_slug"]) == {"p2", "p3"}
    assert outputs["summary"]["n_profiles"] == 2


def test_assign_fixed_rule_correlation_bands_labels_profiles_by_quantiles() -> None:
    rows: list[dict[str, object]] = []
    for idx, rho in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=1):
        for rule_name in [
            "original_distribution",
            "goodreads_cutoff_4_0",
            "drop_lower_goodreads_half",
        ]:
            rows.append(
                {
                    "profile_slug": f"p{idx}",
                    "display_name": f"P{idx}",
                    "rule_name": rule_name,
                    "baseline_mean": 3.0,
                    "profile_rating_std_all_books": 1.0,
                    "goodreads_spearman_rho_all_books": rho,
                }
            )
    profile_metrics = pd.DataFrame(rows)
    rating_shares = pd.DataFrame(
        {
            "profile_slug": ["p1", "p2", "p3", "p4", "p5"] * 3,
            "rule_name": (
                ["original_distribution"] * 5
                + ["goodreads_cutoff_4_0"] * 5
                + ["drop_lower_goodreads_half"] * 5
            ),
            "rating_value": [5] * 15,
            "share_percent": [100.0] * 15,
        }
    )

    outputs = assign_fixed_rule_correlation_bands(
        profile_metrics,
        rating_shares,
        quantile_ranges=((0.2, 0.4), (0.4, 0.8)),
    )

    assert set(outputs["band_summary"]["band_name"]) == {"p20_to_p40", "p40_to_p80"}
    assert "correlation_band" in outputs["profile_metrics"].columns
    assert outputs["profile_metrics"]["correlation_band"].notna().all()
