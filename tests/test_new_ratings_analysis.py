from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from ai_books_tracking.new_ratings_analysis import load_new


def test_load_new_handles_author_column_and_renames_rating_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "new_ratings.csv"
    pd.DataFrame(
        {
            "title": ["Example"],
            "author": ["Author"],
            "date_finished": ["2026-03-01"],
            "Enjoyment (/5)": [4.0],
            "Usefulness /5 to Me": [2.0],
            "Enjoyment (/5) 2nd": [3.0],
            "Usefulness /5 to Me.1": [1.5],
            "Bookshelf": ["Literature"],
            "Long Term Effects": [None],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr("ai_books_tracking.new_ratings_analysis.NEW_RATINGS", csv_path)
    loaded = load_new()

    assert "author" not in loaded.columns
    assert loaded.loc[0, "enjoy1"] == pytest.approx(4.0)
    assert loaded.loc[0, "useful2"] == pytest.approx(1.5)
    assert loaded.loc[0, "enjoy_avg"] == pytest.approx(3.5)
