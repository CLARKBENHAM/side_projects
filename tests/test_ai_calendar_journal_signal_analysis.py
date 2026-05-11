import pandas as pd

from Self_Tracking.ai_calendar.journal_signal_analysis import add_theme_features


def test_add_theme_features_extracts_expected_flags() -> None:
    entries = pd.DataFrame(
        [
            {
                "entry_id": "a",
                "entry_type": "weekly_summary",
                "week_start": "2025-03-10",
                "date": "2025-03-17",
                "text": (
                    "Fight with Amelia, drank too much, scrolled twitter and blogs, "
                    "then re-plan life and feel pathetic."
                ),
            },
            {
                "entry_id": "b",
                "entry_type": "daily_entry",
                "week_start": "2025-03-10",
                "date": "2025-03-16",
                "text": "Great work at WeWork, good routine, happy after gym.",
            },
        ]
    )

    featured = add_theme_features(entries)
    first = featured.iloc[0]
    second = featured.iloc[1]

    assert first["relationship_any"] == 1
    assert first["conflict_any"] == 1
    assert first["relationship_conflict_any"] == 1
    assert first["alcohol_any"] == 1
    assert first["internet_drift_any"] == 1
    assert first["analysis_any"] == 1
    assert first["negative_affect_any"] == 1
    assert first["analysis_negative_any"] == 1
    assert first["drift_any"] == 1

    assert second["work_any"] == 1
    assert second["structure_any"] == 1
    assert second["work_structure_any"] == 1
    assert second["exercise_any"] == 1
    assert second["positive_affect_any"] == 1
