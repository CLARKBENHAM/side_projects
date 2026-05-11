"""Tests for productivity_analysis.py core data loading and feature engineering."""

import pandas as pd

from Self_Tracking.ai_calendar.productivity_analysis import (
    build_analysis_df,
    extract_daily_supplements,
    extract_work_start_time,
    load_distracted_stacked,
    parse_supplement_dict,
)


class TestParseSupplementDict:
    def test_basic_dict(self) -> None:
        result = parse_supplement_dict("{s:7, caf:80, med:0.5}")
        assert result == {"s": 7.0, "caf": 80.0, "med": 0.5}

    def test_negative_value(self) -> None:
        result = parse_supplement_dict("{s:-1.5, caf:80}")
        assert result["s"] == -1.5
        assert result["caf"] == 80.0

    def test_boolean_values(self) -> None:
        result = parse_supplement_dict("{plug:True, gym:False}")
        assert result["plug"] == 1.0
        assert result["gym"] == 0.0

    def test_empty_string(self) -> None:
        assert parse_supplement_dict("") == {}

    def test_no_braces(self) -> None:
        assert parse_supplement_dict("just some text") == {}

    def test_empty_braces(self) -> None:
        assert parse_supplement_dict("{}") == {}

    def test_hyphenated_key(self) -> None:
        result = parse_supplement_dict("{l-th:1}")
        assert result["l_th"] == 1.0

    def test_key_normalization(self) -> None:
        result = parse_supplement_dict("{CAF:80}")
        assert result["caf"] == 80.0

    def test_integer_value(self) -> None:
        result = parse_supplement_dict("{a:20}")
        assert result["a"] == 20.0

    def test_whitespace_handling(self) -> None:
        result = parse_supplement_dict("  { s : 7 , caf : 80 }  ")
        assert result == {"s": 7.0, "caf": 80.0}


class TestExtractDailySupplements:
    def test_extracts_supplements_from_start_entries(self) -> None:
        distracted = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "type": ["s", "d", "s"],
                "comment": ["{s:7, caf:80}", "blog", "{s:6, a:20}"],
            }
        )
        result = extract_daily_supplements(distracted)
        assert len(result) == 2

        day1 = result[result["date"] == pd.Timestamp("2024-01-01")].iloc[0]
        assert day1["sleep_hours"] == 7.0
        assert day1["caffeine"] == 80.0

        day2 = result[result["date"] == pd.Timestamp("2024-01-02")].iloc[0]
        assert day2["sleep_hours"] == 6.0
        assert day2["adderall"] == 20.0

    def test_sleep_takes_first_valid_report(self) -> None:
        distracted = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "type": ["s", "c"],
                "comment": ["{s:7}", "{s:5}"],
            }
        )
        result = extract_daily_supplements(distracted)
        assert result.iloc[0]["sleep_hours"] == 7.0

    def test_sleep_rejects_outliers(self) -> None:
        distracted = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "type": ["s"],
                "comment": ["{s:20}"],
            }
        )
        result = extract_daily_supplements(distracted)
        assert result.iloc[0].get("sleep_hours", 0) == 0

    def test_naps_accumulate(self) -> None:
        distracted = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "type": ["c", "c"],
                "comment": ["{n:0.5}", "{n:1}"],
            }
        )
        result = extract_daily_supplements(distracted)
        assert result.iloc[0]["nap_hours"] == 1.5

    def test_ignores_distraction_entries(self) -> None:
        distracted = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "type": ["d"],
                "comment": ["{caf:80}"],
            }
        )
        result = extract_daily_supplements(distracted)
        assert result.empty


class TestExtractWorkStartTime:
    def test_finds_earliest_start(self) -> None:
        distracted = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
                "type": ["s", "d", "s"],
                "time": ["9:30 AM", "10:00 AM", "2:00 PM"],
            }
        )
        result = extract_work_start_time(distracted)
        assert len(result) == 1
        assert result.iloc[0]["work_start_hour"] == 9.5


class TestStreakCalculation:
    def _build_df(self, dates: list[str], hours: list[float]) -> pd.DataFrame:
        """Build a minimal daily summary + supplements for build_analysis_df."""
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "regime": "Mats",
                "Energy": 5.0,
                "Focus": 5.0,
                "Value": 5.0,
                "Hours Working": hours,
                "# Distractions": 0,
                "Length Distractions": 0.0,
                "# Unfocused": 0,
                "work_productivity": [v * h for v, h in zip([5.0] * len(hours), hours)],
            }
        )
        supplements = pd.DataFrame({"date": pd.to_datetime(dates[:1])})
        calendar = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "calendar_sleep_hours": 7.0,
                "calendar_blue_hours": 0.0,
                "calendar_waste_hours": 0.0,
                "wake_hour": 8.0,
            }
        )
        return build_analysis_df(daily, supplements, calendar)

    def test_streak_resets_on_gap(self) -> None:
        # Days 1-3 consecutive with work, then gap, then day 5 with work
        result = self._build_df(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-06"],
            [3.0, 3.0, 3.0, 3.0],
        )
        # Day 4 (Jan 6): streak should be 1 (reset due to gap), shifted by 1
        # work_streak is yesterday's streak, so Jan 6 sees Jan 3's streak
        # But there's a gap, so the streak at Jan 6 itself is 1
        jan6 = result[result["date"] == pd.Timestamp("2024-01-06")]
        # The streak counter at index 3 (Jan 6) should be 1, not 4
        # work_streak is shifted, so it shows previous day's value
        assert jan6["work_streak"].iloc[0] <= 1  # should NOT be 3+

    def test_streak_continues_consecutive(self) -> None:
        result = self._build_df(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [3.0, 3.0, 3.0],
        )
        jan3 = result[result["date"] == pd.Timestamp("2024-01-03")]
        assert jan3["work_streak"].iloc[0] == 2  # yesterday's streak

    def test_streak_resets_on_rest_day(self) -> None:
        result = self._build_df(
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [3.0, 1.0, 3.0],  # Day 2 is rest (< 2h)
        )
        jan3 = result[result["date"] == pd.Timestamp("2024-01-03")]
        assert jan3["work_streak"].iloc[0] == 0


class TestLoadDistractedStacked:
    def test_parses_three_ranges(self, tmp_path) -> None:
        # Create a minimal CSV with data in all 3 column ranges
        header = ",".join([""] * 34)
        # Range 1 (cols 0-6), range 2 (cols 17-23), range 3 (cols 25-31)
        row = [""] * 34
        # Range 1
        row[0] = "1/1/2025"
        row[1] = "diesl"
        row[2] = "9:00 AM"
        row[3] = "s"
        row[4] = "{s:7}"
        row[5] = ""
        row[6] = "1"
        # Range 2
        row[17] = "6/1/2021"
        row[18] = "Hive"
        row[19] = "10:00 AM"
        row[20] = "s"
        row[21] = "{caf:80}"
        row[22] = ""
        row[23] = "1"
        # Range 3
        row[25] = "3/1/2023"
        row[26] = "Mats"
        row[27] = "8:00 AM"
        row[28] = "s"
        row[29] = "{a:20}"
        row[30] = ""
        row[31] = "1"

        csv_path = tmp_path / "distracted.csv"
        csv_path.write_text(header + "\n" + ",".join(row) + "\n")

        result = load_distracted_stacked(str(csv_path))
        assert len(result) == 3
        assert set(result["for"].unique()) == {"diesl", "Hive", "Mats"}
