
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from productivity_analysis import (
    CALENDAR_NAME_ALIASES,
    WORK_CALENDAR_NAMES,
    parse_ics_files,
    process_book_events,
    process_overlaps,
    process_sleep_events,
    process_slash_events,
)

def load_calendar_sleep_improved(calendar_dir: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Improved version of load_calendar_sleep that correctly handles:
    1. Naps across day boundaries (gap < 30m).
    2. Naps before sleep (gap < 30m).
    3. Multiple sleep events per day.
    4. Wake time assignment to the correct day.
    """
    if start_date is None:
        start_date = datetime(2021, 5, 24)
    if end_date is None:
        end_date = datetime(2026, 3, 17)
    
    df = parse_ics_files(calendar_dir, start_date, end_date)
    if df.empty:
        return pd.DataFrame()

    # Normalize calendar names
    df["calendar_name"] = df["calendar_name"].replace(CALENDAR_NAME_ALIASES)

    df = process_sleep_events(df)
    
    # Separate sleep and nap
    sleep_df = df[df["event_name"].str.lower() == "sleep"].copy()
    nap_df = df[df["event_name"].str.lower() == "nap"].copy()
    
    # Sort for gap checking
    all_rest = pd.concat([sleep_df, nap_df]).sort_values("start_time")
    
    # Improved Reclassification Logic
    # We want to merge any 'nap' that is within 30m of a 'sleep' event (before or after)
    # or even within 30m of another 'nap' that is itself merged with sleep.
    
    # Let's iterate and merge overlapping or near-contiguous rest events
    merged_events = []
    if not all_rest.empty:
        current_event = all_rest.iloc[0].to_dict()
        for i in range(1, len(all_rest)):
            next_event = all_rest.iloc[i].to_dict()
            gap = (next_event["start_time"] - current_event["end_time"]).total_seconds() / 3600
            
            if gap <= 0.5: # 30 minutes
                # Merge
                current_event["end_time"] = max(current_event["end_time"], next_event["end_time"])
                current_event["duration"] = (current_event["end_time"] - current_event["start_time"]).total_seconds() / 3600
                # If either is 'sleep', the merged one is 'sleep'
                if next_event["event_name"].lower() == "sleep":
                    current_event["event_name"] = "sleep"
            else:
                merged_events.append(current_event)
                current_event = next_event
        merged_events.append(current_event)
    
    merged_df = pd.DataFrame(merged_events)
    
    # Now separate them again
    sleep_df = merged_df[merged_df["event_name"].str.lower() == "sleep"].copy()
    nap_df = merged_df[merged_df["event_name"].str.lower() == "nap"].copy()
    
    # Wake time (end of sleep)
    ct = pytz.timezone("US/Central")
    sleep_df["end_local"] = sleep_df["end_time"].dt.tz_convert(ct)
    sleep_df["wake_hour"] = sleep_df["end_local"].dt.hour + sleep_df["end_local"].dt.minute / 60
    sleep_df["date"] = pd.to_datetime(sleep_df["end_local"].dt.date)
    
    # Assign sleep to the day it ENDED (since that's the day it affects)
    # Wait, usually sleep is assigned to the day it started or the day it ended?
    # productivity_analysis.py says: sleep_df["date"] = pd.to_datetime(sleep_df["start_time"].dt.date)
    # But for "wake_hour" it uses: wake_times = sleep_df.groupby("wake_date")["wake_hour"].min().reset_index()
    # where wake_date is end_local.dt.date.
    
    # Let's be consistent with the outcome we are predicting.
    # If we are predicting "today's work", we care about the sleep that ended today.
    
    sleep_daily = sleep_df.groupby("date")["duration"].sum().reset_index()
    sleep_daily = sleep_daily.rename(columns={"duration": "cal_sleep_improved"})
    
    wake_daily = sleep_df.groupby("date")["wake_hour"].last().reset_index()
    wake_daily = wake_daily.rename(columns={"wake_hour": "cal_wake_hour_improved"})
    
    nap_df["date"] = pd.to_datetime(nap_df["start_time"].dt.tz_convert(ct).dt.date)
    nap_daily = nap_df.groupby("date")["duration"].sum().reset_index()
    nap_daily = nap_daily.rename(columns={"duration": "cal_nap_improved"})
    
    result = sleep_daily.merge(wake_daily, on="date", how="outer").merge(nap_daily, on="date", how="outer")
    result = result.fillna(0)
    return result

def load_distracted_stacked_improved(path: str) -> pd.DataFrame:
    """Load the When Distracted CSV, stacking all 3 date ranges into one DataFrame.
    Includes Productivity Value (reflection text) and Notes.
    """
    import csv

    # Standardized names
    std_cols = [
        "date",
        "for",
        "time",
        "type",
        "comment",
        "length_minutes",
        "work_increment",
        "productivity_value",
        "notes"
    ]

    ranges_config = [
        { # Range 1 (current)
            "date": 0, "for": 1, "time": 2, "type": 3, "comment": 4, 
            "length_minutes": 5, "work_increment": 6, "productivity_value": 7, "notes": 8
        },
        { # Range 2 (oldest)
            "date": 17, "for": 18, "time": 19, "type": 20, "comment": 21, 
            "length_minutes": 22, "work_increment": 23, "productivity_value": -1, "notes": 24
        },
        { # Range 3 (middle)
            "date": 25, "for": 26, "time": 27, "type": 28, "comment": 29, 
            "length_minutes": 30, "work_increment": 31, "productivity_value": 32, "notes": 33
        },
    ]

    all_rows: list[dict[str, str]] = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            for config in ranges_config:
                date_ix = config["date"]
                if date_ix < len(row) and row[date_ix].strip() and "/" in row[date_ix]:
                    entry = {}
                    for col_name, col_ix in config.items():
                        if col_ix == -1 or col_ix >= len(row):
                            entry[col_name] = ""
                        else:
                            entry[col_name] = row[col_ix].strip()
                    
                    # Sanity check: if it's the header row repeated (can happen in stacked CSVs)
                    if entry["date"].lower() == "date":
                        continue
                        
                    all_rows.append(entry)

    df = pd.DataFrame(all_rows, columns=std_cols)
    # Filter out empty date rows
    df = df[df["date"] != ""].copy()
    
    # Standardize types
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False, errors="coerce")
    df = df.dropna(subset=["date"])
    
    df["length_minutes"] = pd.to_numeric(df["length_minutes"], errors="coerce").fillna(0)
    df["work_increment"] = pd.to_numeric(df["work_increment"], errors="coerce").fillna(0)
    
    return df.sort_values(["date", "time"]).reset_index(drop=True)

def load_daily_summary_improved(path: str) -> pd.DataFrame:
    """Load Daily Summary with text columns preserved."""
    import pandas as pd
    df = pd.read_csv(path)
    
    # Rename columns to standard names
    col_map = {
        "Start Date": "date",
        "For": "regime",
        "Tasks Summary": "tasks_summary",
        "Energy": "energy",
        "Focus": "focus",
        "Value": "value",
        "Hours Working": "hours_working",
        "# Distractions": "n_distractions",
        "Length Distractions": "len_distractions",
        "# Unfocused": "n_unfocused",
        "Things Learned": "things_learned",
        "Reflections": "reflections",
        "Productivity Notes": "productivity_notes"
    }
    
    # Check which columns exist (some might be missing in older versions)
    existing_cols = {orig: std for orig, std in col_map.items() if orig in df.columns}
    df = df[list(existing_cols.keys())].copy()
    df = df.rename(columns=existing_cols)
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    numeric_cols = ["energy", "focus", "value", "hours_working", "n_distractions", "len_distractions", "n_unfocused"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    # Aggregate text by joining with " | "
    def join_text(x):
        return " | ".join(str(v) for v in x if pd.notna(v) and str(v).strip())

    agg: dict = {
        "regime": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        "energy": "mean",
        "focus": "mean",
        "value": "mean",
        "hours_working": "sum",
        "n_distractions": "sum",
        "len_distractions": "sum",
        "n_unfocused": "sum",
        "tasks_summary": join_text,
        "things_learned": join_text,
        "reflections": join_text,
        "productivity_notes": join_text
    }
    
    # Only aggregate columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}
    
    daily = df.groupby("date").agg(agg).reset_index()
    daily["work_productivity"] = daily["value"] * daily["hours_working"]
    return daily

def fix_streaks(df: pd.DataFrame, cal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Improved streak calculation that considers both CSV work and Calendar work.
    """
    # Merge CSV hours and Calendar blue hours
    df = df.copy()
    cal_work = cal_df[cal_df["calendar_name"].isin(WORK_CALENDAR_NAMES)].copy()
    ct = pytz.timezone("US/Central")
    cal_work["date"] = pd.to_datetime(cal_work["start_time"].dt.tz_convert(ct).dt.date)
    cal_daily = cal_work.groupby("date")["duration"].sum().reset_index()
    cal_daily = cal_daily.rename(columns={"duration": "cal_blue_hours"})
    
    df = df.merge(cal_daily, on="date", how="left")
    df["cal_blue_hours"] = df["cal_blue_hours"].fillna(0)
    
    # A day is "worked" if CSV Hours > 2 OR Calendar Blue Hours > 2
    df["any_work_hours"] = df[["hours_working", "cal_blue_hours"]].max(axis=1)
    df["is_worked_day"] = (df["any_work_hours"] >= 2).astype(int)
    
    # Calculate streak with gap awareness
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].values
    worked = df["is_worked_day"].values
    
    streak_vals = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if i > 0:
            day_gap = (dates[i] - dates[i-1]) / np.timedelta64(1, 'D')
            if day_gap > 1:
                # We missed some days. We don't know if we worked them.
                # BUT if we have calendar data for those days, we could check!
                # For now, let's assume if it's missing from BOTH it's a gap.
                streak_vals[i] = 1 if worked[i] else 0
                continue
        
        if worked[i]:
            streak_vals[i] = (streak_vals[i-1] + 1) if i > 0 else 1
        else:
            streak_vals[i] = 0
            
    df["streak_improved"] = streak_vals
    return df
