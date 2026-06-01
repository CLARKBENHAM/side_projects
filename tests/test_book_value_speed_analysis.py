from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_books_tracking.book_value_speed_analysis import build_value_speed_dataset


def test_value_speed_dataset_drops_main_outliers_and_matches_ratings(
    tmp_path: Path,
) -> None:
    speed_csv = tmp_path / "speed.csv"
    ratings_csv = tmp_path / "ratings.csv"
    output_dir = tmp_path / "outputs"
    pd.DataFrame(
        [
            {
                "title": "Example Fast",
                "matched_finish_date": "2026-01-01",
                "category": "Business",
                "usable_for_visual_reading_speed": True,
                "wpm_visual_after_audio_350wpm": 300,
                "primary_first_pass_minutes": 120,
                "non_audio_primary_minutes": 120,
            },
            {
                "title": "Example Slow",
                "matched_finish_date": "2026-01-02",
                "category": "Business",
                "usable_for_visual_reading_speed": True,
                "wpm_visual_after_audio_350wpm": 200,
                "primary_first_pass_minutes": 240,
                "non_audio_primary_minutes": 240,
            },
            {
                "title": "Example Long",
                "matched_finish_date": "2026-01-03",
                "category": "Business",
                "usable_for_visual_reading_speed": True,
                "wpm_visual_after_audio_350wpm": 250,
                "primary_first_pass_minutes": 360,
                "non_audio_primary_minutes": 360,
            },
            {
                "title": "Dropped Outlier",
                "matched_finish_date": "2026-01-04",
                "category": "Business",
                "usable_for_visual_reading_speed": True,
                "wpm_visual_after_audio_350wpm": 700,
                "primary_first_pass_minutes": 100,
                "non_audio_primary_minutes": 100,
            },
        ]
    ).to_csv(speed_csv, index=False)
    pd.DataFrame(
        [
            {
                "title": "Example Fast",
                "date_finished": "2026-01-01",
                "Enjoyment (/5)": 4,
                "Usefulness /5 to Me": 3,
                "Enjoyment (/5) 2nd": 4,
                "Usefulness /5 to Me.1": 3,
                "Bookshelf": "Business",
            },
            {
                "title": "Example Slow",
                "date_finished": "2026-01-02",
                "Enjoyment (/5)": 3,
                "Usefulness /5 to Me": 4,
                "Enjoyment (/5) 2nd": 3,
                "Usefulness /5 to Me.1": 4,
                "Bookshelf": "Business",
            },
            {
                "title": "Example Long",
                "date_finished": "2026-01-03",
                "Enjoyment (/5)": 2,
                "Usefulness /5 to Me": 5,
                "Enjoyment (/5) 2nd": 2,
                "Usefulness /5 to Me.1": 5,
                "Bookshelf": "Business",
            },
        ]
    ).to_csv(ratings_csv, index=False)

    joined, summary, category = build_value_speed_dataset(
        speed_csv=speed_csv,
        output_dir=output_dir,
        master_ratings=tmp_path / "missing_master.csv",
        new_ratings=ratings_csv,
        ratings2=tmp_path / "missing_ratings2.csv",
    )

    assert joined["title"].tolist() == ["Example Fast", "Example Slow", "Example Long"]
    assert joined["has_value_rating"].all()
    assert "Dropped Outlier" not in set(joined["title"])
    assert (
        summary[
            summary["predictor"].eq("Primary reading hours")
            & summary["outcome"].eq("Usefulness rating")
        ]["pearson_r"].iloc[0]
        > 0
    )
    assert category.iloc[0]["n_books"] == 3
    assert (output_dir / "book_value_speed_scatter_matrix.png").exists()
