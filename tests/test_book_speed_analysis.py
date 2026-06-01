from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_books_tracking.book_speed_analysis import write_rolling_category_percentiles


def test_write_rolling_category_percentiles_uses_six_month_windows(
    tmp_path: Path,
) -> None:
    usable = pd.DataFrame(
        [
            {
                "title": "A",
                "matched_finish_date": "2026-01-10",
                "category": "Business",
                "wpm_first_pass_primary": 200,
            },
            {
                "title": "B",
                "matched_finish_date": "2026-02-10",
                "category": "Business",
                "wpm_first_pass_primary": 300,
            },
            {
                "title": "C",
                "matched_finish_date": "2026-07-10",
                "category": "Business",
                "wpm_first_pass_primary": 400,
            },
            {
                "title": "D",
                "matched_finish_date": "2026-07-20",
                "category": "Business",
                "wpm_first_pass_primary": 500,
            },
        ]
    )

    result = write_rolling_category_percentiles(
        usable,
        tmp_path,
        min_books_per_window=2,
        min_books_per_category=3,
    )

    july = result[result["window_end"].eq("2026-07-31")].iloc[0]
    assert july["rolling_n"] == 3
    assert july["wpm_p50"] == 400
    assert (tmp_path / "book_speed_rolling_6mo_category_percentiles.png").exists()
