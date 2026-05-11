
import pandas as pd
from data_fixes import load_daily_summary_improved, load_distracted_stacked_improved, load_calendar_sleep_improved, fix_streaks
from productivity_analysis import extract_daily_supplements, extract_work_start_time, load_calendar_full, DAILY_SUMMARY_CSV, DISTRACTED_CSV, CALENDAR_DIR

def analyze_low_days():
    print("Loading data...")
    daily = load_daily_summary_improved(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked_improved(DISTRACTED_CSV)
    supplements = extract_daily_supplements(distracted)
    work_starts = extract_work_start_time(distracted)
    cal_sleep = load_calendar_sleep_improved(CALENDAR_DIR)
    cal_full = load_calendar_full(CALENDAR_DIR)
    
    # Merge
    df = daily.merge(supplements, on="date", how="left")
    df = df.merge(cal_sleep, on="date", how="left")
    df = df.merge(work_starts, on="date", how="left")
    df = fix_streaks(df, cal_full)
    
    low_days = df[df["hours_working"] < 2].copy()
    print(f"Total Low Work days (<2h): {len(low_days)}")
    
    print("\nCommon Tasks Summary on Low Work Days:")
    low_tasks = low_days["tasks_summary"].dropna()
    all_low_tasks = " | ".join(low_tasks).lower()
    
    keywords = ["rank", "journal", "nothing", "anki", "reading", "review", "planning", "misc", "cleanup", "emails"]
    for k in keywords:
        count = all_low_tasks.count(k)
        pct = count / len(low_days) if len(low_days) > 0 else 0
        print(f"  {k:15s}: {count} times ({pct:.1%})")

if __name__ == "__main__":
    analyze_low_days()
