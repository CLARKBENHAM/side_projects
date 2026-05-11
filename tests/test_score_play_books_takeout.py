from __future__ import annotations

import numpy as np
import pandas as pd

from ai_books_tracking import score_play_books_takeout as takeout


def test_load_takeout_catalog_merges_and_normalizes(tmp_path) -> None:
    finished = pd.DataFrame(
        [
            {
                "title": "Book One",
                "author": "by",
                "bookshelf": "General Reading",
                "earliest_modified": "2026-03-01 10:00:00",
                "latest_modified": "2026-03-02 11:00:00",
                "filename": "book_one.pdf",
            }
        ]
    )
    unfinished = pd.DataFrame(
        [
            {
                "title": "Book Two",
                "author": "Jane Doe",
                "bookshelf": "Literature",
                "earliest_modified": "",
                "latest_modified": "",
                "filename": "book_two.epub",
            }
        ]
    )
    finished_path = tmp_path / "finished.csv"
    unfinished_path = tmp_path / "unfinished.csv"
    finished.to_csv(finished_path, index=False)
    unfinished.to_csv(unfinished_path, index=False)

    catalog = takeout.load_takeout_catalog(finished_path, unfinished_path)

    assert len(catalog) == 2
    assert set(catalog["play_status"]) == {"finished", "unfinished"}
    assert "Bookshelf" in catalog.columns
    assert catalog.loc[catalog["title"] == "Book One", "author"].item() == ""
    assert catalog["canonical_key"].notna().all()


def test_prepare_candidate_frame_adds_required_prediction_columns() -> None:
    catalog = pd.DataFrame(
        [
            {
                "title": "Book One",
                "author": "Jane Doe",
                "Bookshelf": "General Reading",
                "earliest_modified": pd.Timestamp("2026-03-01 10:00:00"),
                "latest_modified": pd.Timestamp("2026-03-02 11:00:00"),
                "filename": "book_one.pdf",
                "goodreads_author": "Jane Doe",
                "goodreads_rating": 4.2,
                "goodreads_rating_count": 1000,
                "goodreads_rating_raw_best": 4.2,
                "goodreads_rating_count_raw_best": 1000,
            }
        ]
    )

    prepared = takeout.prepare_candidate_frame(catalog)

    assert "avg_enjoyment" in prepared.columns
    assert "avg_usefulness" in prepared.columns
    assert "year_finished" in prepared.columns
    assert prepared["year_finished"].item() == 2026
    assert prepared["canonical_author"].item() == "jane doe"
    assert pd.isna(prepared["avg_enjoyment"].item())


def test_apply_scoring_rules_adds_decision_columns(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_model_predictions_for_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        spec,
        model_name: str,
    ) -> np.ndarray:
        calls.append((target_col, model_name))
        base = {
            ("avg_enjoyment", "GBM"): np.array([3.9, 3.6]),
            ("avg_enjoyment", "Lasso"): np.array([3.7, 3.4]),
            ("avg_usefulness", "Lasso"): np.array([2.5, 2.1]),
            ("avg_usefulness", "Ridge"): np.array([2.8, 2.0]),
            ("avg_usefulness", "Random Forest"): np.array([2.3, 1.7]),
        }
        return base[(target_col, model_name)]

    monkeypatch.setattr(
        takeout, "model_predictions_for_split", fake_model_predictions_for_split
    )

    training = pd.DataFrame(
        [
            {"avg_enjoyment": 3.0, "avg_usefulness": 2.0},
            {"avg_enjoyment": 4.0, "avg_usefulness": 3.0},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"title": "A"},
            {"title": "B"},
        ]
    )

    scored = takeout.apply_scoring_rules(training, candidates)

    assert len(calls) == len(takeout.SCORING_RULES)
    assert scored["decision_enjoyment_gbm_3p8"].tolist() == [True, False]
    assert scored["decision_usefulness_lasso_2p4"].tolist() == [True, False]
    assert scored["primary_decision"].tolist() == ["read", "skip"]
    assert scored["primary_decision_with_usefulness"].tolist() == [
        "read_high_priority",
        "skip",
    ]


def test_add_goodreads_from_existing_cache_uses_cached_matches(
    monkeypatch, tmp_path
) -> None:
    catalog = pd.DataFrame(
        [
            {
                "title": "Book One",
                "author": "Jane Doe",
                "Bookshelf": "General Reading",
                "earliest_modified": pd.Timestamp("2026-03-01 10:00:00"),
                "latest_modified": pd.Timestamp("2026-03-02 11:00:00"),
                "filename": "book_one.pdf",
                "canonical_key": "book one key",
            },
            {
                "title": "Book Two",
                "author": "John Doe",
                "Bookshelf": "Literature",
                "earliest_modified": pd.Timestamp("2026-03-01 10:00:00"),
                "latest_modified": pd.Timestamp("2026-03-02 11:00:00"),
                "filename": "book_two.epub",
                "canonical_key": "book two key",
            },
        ]
    )

    monkeypatch.setattr(
        takeout,
        "load_match_cache_with_retry",
        lambda: {
            "book one key": {
                "goodreads_status": "matched",
                "goodreads_match_method": "cached",
                "goodreads_rating": 4.4,
                "goodreads_rating_count": 5000,
                "goodreads_match_score": 0.9,
                "goodreads_candidates_json": "[]",
            }
        },
    )

    merged = takeout.add_goodreads_from_existing_cache(
        catalog, output_path=tmp_path / "goodreads.csv"
    )

    first = merged.loc[merged["title"] == "Book One"].iloc[0]
    second = merged.loc[merged["title"] == "Book Two"].iloc[0]
    assert first["goodreads_status"] == "matched"
    assert first["goodreads_rating"] == 4.4
    assert second["goodreads_status"] == "unmatched"
