from pathlib import Path

import pandas as pd

from Self_Tracking.ai_calendar.parse_journals import parse_file


def write_journal(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "journal.md"
    path.write_text(content.strip() + "\n")
    return path


def test_parse_weekly_summaries_reset_year_context(tmp_path: Path) -> None:
    path = write_journal(
        tmp_path,
        """
        ## 2026
        #### Weekly Summaries, starting on Monday inclusive of Sunday.
        1. 03/16 Most recent week

        ## 2025
        #### Weekly Summaries, starting on Monday inclusive of Sunday.
        2. 12/30 Cross-year week
        3. 01/06 First full 2025 week

        ## 2023
        #### Weekly Summaries, starting on Monday inclusive of Sunday.
        1. 07/24 First tracked 2023 week
        2. 07/31 Follow-up week
        """,
    )

    records = pd.DataFrame(record.__dict__ for record in parse_file(path))
    weekly = records[records["entry_type"] == "weekly_summary"].reset_index(drop=True)

    assert weekly["date"].tolist() == [
        "2026-03-16",
        "2024-12-30",
        "2025-01-06",
        "2023-07-24",
        "2023-07-31",
    ]


def test_parse_daily_entries_use_month_section_context(tmp_path: Path) -> None:
    path = write_journal(
        tmp_path,
        """
        ## 2024
        #### December 2024
        12/31 End of month
        12/27 Earlier in month

        #### January 2024
        01/02 New year
        01/01 First day

        ## 2022
        #### January 2022
        01/01 Old entry
        """,
    )

    records = pd.DataFrame(record.__dict__ for record in parse_file(path))
    daily = records[records["entry_type"] == "daily_entry"].reset_index(drop=True)

    assert daily["date"].tolist() == [
        "2024-12-31",
        "2024-12-27",
        "2024-01-02",
        "2024-01-01",
        "2022-01-01",
    ]


def test_parse_ignores_review_placeholders_and_image_footnotes(tmp_path: Path) -> None:
    path = write_journal(
        tmp_path,
        """
        ## 2025
        #### Weekly Summaries, starting on Monday inclusive of Sunday.
        1. 11/10 Real weekly summary
        **Quarterly Review:**
        1. 03/31
        2. 12/31
        #### November 2025
        11/17 Real daily entry
        [image1]: <data:image/png;base64,AAAA>
        """,
    )

    records = pd.DataFrame(record.__dict__ for record in parse_file(path))

    assert records["entry_type"].value_counts().to_dict() == {
        "weekly_summary": 1,
        "daily_entry": 1,
    }
