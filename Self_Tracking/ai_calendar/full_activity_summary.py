
import pandas as pd
import numpy as np
from datetime import datetime, date
import pytz
import os
from productivity_analysis import (
    CALENDAR_DIR,
    CALENDAR_NAME_ALIASES,
    DAILY_SUMMARY_CSV,
    DISTRACTED_CSV,
    WORK_CALENDAR_NAMES,
    parse_ics_files,
    process_book_events,
    process_overlaps,
    process_sleep_events,
    process_slash_events,
)
from data_fixes import load_distracted_stacked_improved, load_daily_summary_improved, load_calendar_sleep_improved

def generate_full_activity_summary_pct():
    print("Loading all data sources...")
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2026, 3, 17)
    
    # 1. Calendar Data
    df = parse_ics_files(CALENDAR_DIR, start_date, end_date)
    df["calendar_name"] = df["calendar_name"].replace(CALENDAR_NAME_ALIASES)
    df = process_sleep_events(df)
    df = process_slash_events(df)
    df = process_book_events(df)
    df = process_overlaps(df)
    
    # Category Assignment
    color_map = {
        "Things": "things",
        "Meals, Supplements, Sleep": "green",
        "Waste Time": "waste",
    }
    for calendar_name in WORK_CALENDAR_NAMES:
        color_map[calendar_name] = "blue"
    df["category"] = "other"
    for key, cat in color_map.items():
        df.loc[
            df["calendar_name"].str.contains(key, case=False, na=False, regex=False),
            "category",
        ] = cat

    ct = pytz.timezone("US/Central")
    df["start_local"] = df["start_time"].dt.tz_convert(ct)
    df["date"] = pd.to_datetime(df["start_local"].dt.date)

    # 2. Daily Summary (Work ground truth)
    daily = load_daily_summary_improved(DAILY_SUMMARY_CSV)
    
    # 3. Drunk Days
    distracted = load_distracted_stacked_improved(DISTRACTED_CSV)
    drunk_mask = (
        distracted["type"].str.contains("drink", case=False, na=False) |
        distracted["comment"].str.contains("drink|drank|drunk", case=False, na=False) |
        distracted["notes"].str.contains("drink|drank|drunk", case=False, na=False) |
        distracted["productivity_value"].str.contains("drink|drank|drunk", case=False, na=False)
    )
    drunk_dates = set(distracted[drunk_mask]["date"].dt.date)
    drunk_dates |= set(df[df["event_name"].str.lower().str.contains("drink|drank|drunk", na=False)]["date"].dt.date)

    # 4. Activity Flags
    df["is_book"] = df["event_name"].str.lower().str.contains("book:", na=False)
    df["is_blog"] = df["event_name"].str.lower().str.contains("blog", na=False)
    df["is_waste"] = df["category"] == "waste"
    df["is_sleep"] = df["event_name"].str.lower() == "sleep"
    df["is_drunk_day"] = df["date"].dt.date.isin(drunk_dates)

    # 5. Aggregate Daily stats first to merge with work
    cal_daily = df.groupby("date").apply(lambda d: pd.Series({
        "books_h": d[d["is_book"]]["duration"].sum(),
        "blogs_h": d[d["is_blog"]]["duration"].sum(),
        "waste_h": d[d["is_waste"]]["duration"].sum(),
        "waste_drunk_h": d[d["is_waste"] & d["is_drunk_day"]]["duration"].sum(),
        "sleep_h": d[d["is_sleep"]]["duration"].sum(),
        "is_drunk": d["is_drunk_day"].any()
    })).reset_index()

    full_daily = cal_daily.merge(daily[["date", "hours_working"]], on="date", how="outer").fillna(0)
    full_daily["month_period"] = full_daily["date"].dt.to_period("M")
    
    # 6. Monthly Aggregation
    monthly = full_daily.groupby("month_period").apply(lambda m: pd.Series({
        "Books": m["books_h"].sum(),
        "Blogs": m["blogs_h"].sum(),
        "Waste": m["waste_h"].sum(),
        "Drunk Waste": m["waste_drunk_h"].sum(),
        "Total Work": m["hours_working"].sum(),
        "Sleep": m["sleep_h"].sum(),
        "Days": m["date"].nunique()
    })).reset_index()

    # Calculate Waking Hours
    # Note: If Sleep is missing for a day, we assume 8h to be conservative
    # But let's check the data.
    monthly["Total Hours"] = monthly["Days"] * 24
    # If sleep is suspiciously low (e.g. < 4h average), we might be missing data.
    # Let's use a floor of 7h per day for waking hour calculation to avoid 100%+ stats on missing data.
    monthly["Adjusted Sleep"] = monthly["Sleep"].clip(lower=monthly["Days"] * 6)
    monthly["Waking Hours"] = monthly["Total Hours"] - monthly["Adjusted Sleep"]
    
    # Convert to Percentages
    cols_to_pct = ["Books", "Blogs", "Waste", "Drunk Waste", "Total Work"]
    for col in cols_to_pct:
        monthly[f"{col} (%)"] = (monthly[col] / monthly["Waking Hours"]) * 100

    monthly["Month"] = monthly["month_period"].astype(str)
    
    final_cols = ["Month", "Total Work (%)", "Books (%)", "Blogs (%)", "Waste (%)", "Drunk Waste (%)"]
    
    print("\nFULL ACTIVITY SUMMARY SINCE 2021-01-01 (AS % OF WAKING HOURS)")
    print(monthly[final_cols].to_string(index=False, float_format="%.1f"))

if __name__ == "__main__":
    generate_full_activity_summary_pct()
