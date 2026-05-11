
import pandas as pd
import numpy as np
from data_fixes import load_daily_summary_improved, load_distracted_stacked_improved, load_calendar_sleep_improved, fix_streaks
from productivity_analysis import extract_daily_supplements, extract_work_start_time, load_calendar_full, DAILY_SUMMARY_CSV, DISTRACTED_CSV, CALENDAR_DIR

def analyze_recovery_days():
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
    
    df = df.sort_values("date")
    
    # Define "Big Push" days
    big_push = df[df["cal_blue_hours"] >= 8].copy()
    print(f"Total Big Push days: {len(big_push)}")
    
    # Analyze the day AFTER a big push
    recovery_dates = big_push["date"] + pd.Timedelta(days=1)
    recovery_days = df[df["date"].isin(recovery_dates)].copy()
    
    # Analyze the "Tasks Summary" for recovery days
    print("\nCommon Tasks Summary on Recovery Days (Day after 8h+ Blue):")
    recovery_tasks = recovery_days["tasks_summary"].dropna()
    all_tasks = " | ".join(recovery_tasks).lower()
    
    # Count keywords
    keywords = ["rank", "journal", "nothing", "anki", "reading", "review", "planning"]
    for k in keywords:
        count = all_tasks.count(k)
        pct = count / len(recovery_days) if len(recovery_days) > 0 else 0
        print(f"  {k:15s}: {count} times ({pct:.1%})")
        
    # Compare to non-recovery days
    non_recovery = df[~df["date"].isin(recovery_dates)].copy()
    print("\nCommon Tasks Summary on Non-Recovery Days:")
    non_recovery_tasks = non_recovery["tasks_summary"].dropna()
    all_non_tasks = " | ".join(non_recovery_tasks).lower()
    for k in keywords:
        count = all_non_tasks.count(k)
        pct = count / len(non_recovery) if len(non_recovery) > 0 else 0
        print(f"  {k:15s}: {count} times ({pct:.1%})")

if __name__ == "__main__":
    analyze_recovery_days()
