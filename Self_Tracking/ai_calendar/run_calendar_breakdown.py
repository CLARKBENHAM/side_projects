# %% use conda env: side_projects
"""Run just the calendar time breakdown analysis for different time periods."""
import matplotlib
matplotlib.use("Agg")

from productivity_analysis import (
    load_calendar_full,
    load_daily_summary_full,
    load_distracted_stacked,
    analyze_calendar_time_breakdown,
    CALENDAR_DIR,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
)

print("Loading data...")
daily = load_daily_summary_full(DAILY_SUMMARY_CSV)
print(f"  Daily summary: {len(daily)} rows")

distracted = load_distracted_stacked(DISTRACTED_CSV)
print(f"  Distracted entries: {len(distracted)}")

print("  Loading full calendar (with overlap/slash handling)...")
cal_full = load_calendar_full(CALENDAR_DIR)
print(f"  Calendar events: {len(cal_full)}")

analyze_calendar_time_breakdown(cal_full, daily, distracted,
                                period_label="All Time")
analyze_calendar_time_breakdown(cal_full, daily, distracted,
                                period_label="Past Year (Mar 2025 - Mar 2026)",
                                start_date="2025-03-15", end_date="2026-03-15")
analyze_calendar_time_breakdown(cal_full, daily, distracted,
                                period_label="Past 6 Months (Sep 2025 - Mar 2026)",
                                start_date="2025-09-15", end_date="2026-03-15")
analyze_calendar_time_breakdown(cal_full, daily, distracted,
                                period_label="Past 3 Months (Dec 2025 - Mar 2026)",
                                start_date="2025-12-15", end_date="2026-03-15")
