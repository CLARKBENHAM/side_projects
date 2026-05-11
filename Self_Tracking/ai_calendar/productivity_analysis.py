# %% use conda env: side_projects
"""
Productivity analysis: what predicts how well I work?
Builds on calendar_analysis.py but focuses on the work tracking CSV data.

Primary outcome: Hours Working (from Daily Summary)
Secondary outcomes: work_productivity (Value * Hours), Value, Energy, Focus
"""
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import types


def _import_functions_from(
    module_path: str, function_names: list[str]
) -> dict[str, object]:
    """Import specific functions from a module without executing top-level code
    that depends on runtime variables. Loads only up to the first bare (non-guarded)
    assignment that would fail."""
    import ast

    with open(module_path) as f:
        source = f.read()

    tree = ast.parse(source)

    # Keep only imports, function/class defs, and if __name__ == "__main__" blocks
    safe_stmts = []
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            safe_stmts.append(node)
        elif isinstance(node, ast.If):
            # Keep if __name__ == "__main__" but skip its body
            pass
        elif isinstance(node, ast.Assign):
            # Keep simple constant assignments (like color_map)
            # but skip anything that references runtime variables
            try:
                code = ast.get_source_segment(source, node)
                if code and not any(
                    v in code for v in ["calendar_df", "work_df", "df.", "result ="]
                ):
                    safe_stmts.append(node)
            except Exception:
                pass

    new_tree = ast.Module(body=safe_stmts, type_ignores=[])
    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, module_path, "exec")

    mod = types.ModuleType("calendar_analysis")
    mod.__file__ = module_path
    exec(code, mod.__dict__)
    return {name: getattr(mod, name) for name in function_names}


BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
REPO_DIR = PARENT_DIR.parent
WORK_CALENDAR_NAMES = tuple(
    name.strip()
    for name in os.environ.get("SELF_TRACKING_WORK_CALENDAR_NAMES", "Work").split(",")
    if name.strip()
)
PRIMARY_WORK_CALENDAR_NAME = WORK_CALENDAR_NAMES[0] if WORK_CALENDAR_NAMES else "Work"
CALENDAR_NAME_ALIASES = {
    os.environ.get("SELF_TRACKING_WASTE_CALENDAR_NAME", "Waste Time"): "Waste Time",
    os.environ.get(
        "SELF_TRACKING_MEALS_SLEEP_CALENDAR_NAME", "Meals, Supplements, Sleep"
    ): "Meals, Supplements, Sleep",
    os.environ.get("SELF_TRACKING_THINGS_CALENDAR_NAME", "Things"): "Things",
}

_ca_candidates = [
    BASE_DIR / "calendar_analysis.py",
    PARENT_DIR / "calendar_analysis.py",
]
_ca_path = next((str(path) for path in _ca_candidates if path.exists()), None)
if _ca_path is None:
    raise FileNotFoundError(
        "Could not find calendar_analysis.py in ai_calendar/ or Self_Tracking/."
    )
_ca_funcs = _import_functions_from(
    _ca_path,
    [
        "parse_ics_files",
        "process_sleep_events",
        "process_slash_events",
        "process_book_events",
        "process_overlaps",
    ],
)
parse_ics_files = _ca_funcs["parse_ics_files"]
process_sleep_events = _ca_funcs["process_sleep_events"]
process_slash_events = _ca_funcs["process_slash_events"]
process_book_events = _ca_funcs["process_book_events"]
process_overlaps = _ca_funcs["process_overlaps"]

# ============================================================================
# Data Loading
# ============================================================================

DISTRACTED_CSV = os.path.join(REPO_DIR, "data", "Work Summary  - When Distracted.csv")
DAILY_SUMMARY_CSV = os.path.join(REPO_DIR, "data", "Work Summary  - Daily Summary.csv")
CALENDAR_DIR = os.path.join(REPO_DIR, "data", "Takeout 5", "Calendar")

# Supplement key normalization: map variant keys to canonical names
SUPPLEMENT_ALIASES: dict[str, str] = {
    "a": "adderall",
    "ad": "adderall",
    "add": "adderall",
    "caf": "caffeine",
    "mod": "modafinil",
    "bro": "bronkaid",
    "bronkaid": "bronkaid",
    "nic": "nicotine",
    "sub": "sulbutiamine",
    "pir": "piracetam",
    "ash": "ashwagandha",
    "cho": "choline",
    "cre": "creatine",
    "lav": "lavender",
    "zem": "zembrin",
    "kglu": "potassium_gluconate",
    "kgu": "potassium_gluconate",
    "th": "l_theanine",
    "l_th": "l_theanine",
    "zinc": "zinc",
}

# Keys in comment dicts that are NOT supplements
NON_SUPPLEMENT_KEYS = {"s", "n", "nap", "m", "sn", "med", "plug", "gym", "p", "c", "1"}

# Stimulant subset for focused analysis
STIMULANT_NAMES = ["caffeine", "adderall", "modafinil", "bronkaid", "nicotine"]


def parse_supplement_dict(comment: str) -> dict[str, float]:
    """Parse a comment like '{s:2, caf:80, med:0.5, m:0.8}' into a dict."""
    comment = comment.strip()
    if not comment.startswith("{"):
        return {}
    inner = comment.strip("{}").strip()
    if not inner:
        return {}
    result: dict[str, float] = {}
    # Match key:value pairs, handling negative numbers and decimals
    for match in re.finditer(r"(\w[\w-]*)\s*:\s*(-?[0-9]*\.?[0-9]+|True|False)", inner):
        key = match.group(1).lower().replace("-", "_")
        val_str = match.group(2)
        if val_str in ("True", "False"):
            result[key] = 1.0 if val_str == "True" else 0.0
        else:
            result[key] = float(val_str)
    return result


def load_distracted_stacked(path: str) -> pd.DataFrame:
    """Load the When Distracted CSV, stacking all 3 date ranges into one DataFrame."""
    import csv

    # Column mappings for each range -> standardized names
    # Range 1 (current, 2025+): cols 0-8
    # Range 2 (06/21-12/22): cols 17-24 (no Productivity Value)
    # Range 3 (1/23-12/24): cols 25-33
    std_cols = [
        "date",
        "for",
        "time",
        "type",
        "comment",
        "length_minutes",
        "work_increment",
    ]

    ranges_config = [
        {
            "date": 0,
            "for": 1,
            "time": 2,
            "type": 3,
            "comment": 4,
            "length_minutes": 5,
            "work_increment": 6,
        },
        {
            "date": 17,
            "for": 18,
            "time": 19,
            "type": 20,
            "comment": 21,
            "length_minutes": 22,
            "work_increment": 23,
        },
        {
            "date": 25,
            "for": 26,
            "time": 27,
            "type": 28,
            "comment": 29,
            "length_minutes": 30,
            "work_increment": 31,
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
                        entry[col_name] = (
                            row[col_ix].strip() if col_ix < len(row) else ""
                        )
                    all_rows.append(entry)

    df = pd.DataFrame(all_rows, columns=std_cols)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["length_minutes"] = pd.to_numeric(df["length_minutes"], errors="coerce")
    df["work_increment"] = pd.to_numeric(df["work_increment"], errors="coerce")
    return df.sort_values(["date", "time"]).reset_index(drop=True)


def extract_daily_supplements(distracted_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the stacked distracted CSV, extract per-day supplement doses,
    self-reported sleep, naps, meals, and meditation.
    Only looks at 's' (start) and 'c' (consume) type entries.
    """
    # Filter to start and consume entries which contain the supplement dicts
    mask = distracted_df["type"].isin(["s", "c", "s/c"])
    relevant = distracted_df[mask].copy()

    daily_data: dict[pd.Timestamp, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for _, row in relevant.iterrows():
        d = row["date"]
        parsed = parse_supplement_dict(row["comment"])
        if not parsed:
            continue

        day_dict = daily_data[d]

        for key, value in parsed.items():
            if key == "s":
                # Sleep: take the first report of the day (from the start entry)
                # Filter out obvious data entry errors (>14h or <1h)
                if 1 <= value <= 14:
                    if "sleep_hours" not in day_dict or day_dict["sleep_hours"] == 0:
                        day_dict["sleep_hours"] = value
            elif key in ("n", "nap"):
                day_dict["nap_hours"] += value
            elif key == "m":
                day_dict["meals_kcal"] += value  # units are 1000 cal
            elif key == "sn":
                day_dict["snacks"] += value  # units are 100 cal
            elif key == "med":
                day_dict["meditation_min"] += value
            elif key == "plug":
                day_dict["is_plug"] = 1.0
            elif key in SUPPLEMENT_ALIASES:
                canonical = SUPPLEMENT_ALIASES[key]
                day_dict[canonical] += value
            # else: ignore unknown keys

    rows = []
    for date, data in daily_data.items():
        row = {"date": date}
        row.update(data)
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)
    # Fill NaN supplement columns with 0
    supplement_cols = [c for c in result.columns if c != "date"]
    result[supplement_cols] = result[supplement_cols].fillna(0)
    return result


def extract_work_start_time(distracted_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the time of first 's' (start) entry per day as work start hour."""
    starts = distracted_df[distracted_df["type"] == "s"].copy()
    starts["time_parsed"] = pd.to_datetime(
        starts["time"], format="mixed", errors="coerce"
    )
    starts["start_hour"] = (
        starts["time_parsed"].dt.hour + starts["time_parsed"].dt.minute / 60
    )
    # Take earliest start per day
    first = (
        starts.sort_values("start_hour")
        .groupby("date")["start_hour"]
        .first()
        .reset_index()
    )
    first = first.rename(columns={"start_hour": "work_start_hour"})
    return first


def load_daily_summary_full(path: str) -> pd.DataFrame:
    """Load Daily Summary with all useful columns, aggregating multiple sessions per day."""
    df = pd.read_csv(path)
    # Select useful columns
    cols = [
        "Start Date",
        "For",
        "Energy",
        "Focus",
        "Value",
        "Hours Working",
        "# Distractions",
        "Length Distractions",
        "# Unfocused",
    ]
    df = df[cols].copy()
    df = df.rename(columns={"Start Date": "date", "For": "regime"})

    # Convert types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in [
        "Energy",
        "Focus",
        "Value",
        "Hours Working",
        "# Distractions",
        "Length Distractions",
        "# Unfocused",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"])

    # Aggregate by day: sum hours/distractions, mean for ratings
    # Keep regime as mode (most common that day)
    agg: dict[str, str | tuple[str, str]] = {
        "regime": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        "Energy": "mean",
        "Focus": "mean",
        "Value": "mean",
        "Hours Working": "sum",
        "# Distractions": "sum",
        "Length Distractions": "sum",
        "# Unfocused": "sum",
    }
    daily = df.groupby("date").agg(agg).reset_index()
    daily["work_productivity"] = daily["Value"] * daily["Hours Working"]
    return daily


def load_calendar_sleep(calendar_dir: str) -> pd.DataFrame:
    """Load calendar-derived sleep data for comparison with self-reported."""
    start_date = datetime(2021, 5, 24)
    end_date = datetime(2026, 3, 15)
    df = parse_ics_files(calendar_dir, start_date, end_date)
    if df.empty:
        return pd.DataFrame(columns=["date", "calendar_sleep_hours"])
    # Rename calendar columns
    df["calendar_name"] = df["calendar_name"].replace(CALENDAR_NAME_ALIASES)
    df = process_sleep_events(df)
    sleep_df = df[df["event_name"].str.lower() == "sleep"].copy()
    nap_df = df[df["event_name"].str.lower() == "nap"].copy()

    # Reclassify naps within 30min of sleep end as interrupted sleep
    reclassified_indices: set[int] = set()
    if not nap_df.empty and not sleep_df.empty:
        for nap_idx, nap_row in nap_df.iterrows():
            nap_start = nap_row["start_time"]
            day_sleeps = sleep_df[sleep_df["start_time"].dt.date == nap_start.date()]
            for _, sleep_row in day_sleeps.iterrows():
                gap_hours = (nap_start - sleep_row["end_time"]).total_seconds() / 3600
                if 0 <= gap_hours <= 0.5:
                    reclassified_indices.add(nap_idx)
                    break
        if reclassified_indices:
            reclassed = nap_df.loc[list(reclassified_indices)].copy()
            reclassed["event_name"] = "sleep"
            sleep_df = pd.concat([sleep_df, reclassed], ignore_index=True)
            nap_df = nap_df.drop(index=list(reclassified_indices))

    sleep_df["date"] = pd.to_datetime(sleep_df["start_time"].dt.date)
    cal_sleep = sleep_df.groupby("date")["duration"].sum().reset_index()
    cal_sleep = cal_sleep.rename(columns={"duration": "calendar_sleep_hours"})

    # Also extract calendar blue (productive) time
    df = process_slash_events(df)
    df = process_book_events(df)
    df = process_overlaps(df)
    df.loc[df["event_name"].str.lower().str.contains(": hive"), "calendar_name"] = (
        PRIMARY_WORK_CALENDAR_NAME
    )
    df.loc[df["event_name"].str.lower() == "job", "calendar_name"] = (
        PRIMARY_WORK_CALENDAR_NAME
    )
    blue_df = df[df["calendar_name"].isin(WORK_CALENDAR_NAMES)].copy()
    blue_df["date"] = pd.to_datetime(blue_df["start_time"].dt.date)
    cal_blue = blue_df.groupby("date")["duration"].sum().reset_index()
    cal_blue = cal_blue.rename(columns={"duration": "calendar_blue_hours"})

    # Also extract waste time
    waste_df = df[df["calendar_name"] == "Waste Time"].copy()
    waste_df["date"] = pd.to_datetime(waste_df["start_time"].dt.date)
    cal_waste = waste_df.groupby("date")["duration"].sum().reset_index()
    cal_waste = cal_waste.rename(columns={"duration": "calendar_waste_hours"})

    result = cal_sleep.merge(cal_blue, on="date", how="outer").merge(
        cal_waste, on="date", how="outer"
    )
    result = result.fillna(0)

    # Extract wake time (sleep end in local time)
    import pytz

    ct = pytz.timezone("US/Central")
    sleep_df["end_local"] = sleep_df["end_time"].dt.tz_convert(ct)
    sleep_df["wake_hour"] = (
        sleep_df["end_local"].dt.hour + sleep_df["end_local"].dt.minute / 60
    )
    sleep_df["wake_date"] = pd.to_datetime(sleep_df["end_local"].dt.date)
    wake_times = sleep_df.groupby("wake_date")["wake_hour"].min().reset_index()
    wake_times = wake_times.rename(columns={"wake_date": "date"})
    result = result.merge(wake_times, on="date", how="outer")

    # Extract gym time of day (local, excluding implausible durations)
    gym_df = df[
        (df["event_name"].str.lower().str.contains("gym"))
        & (df["duration"] < 4)
        & (df["duration"] > 0.1)
    ].copy()
    if not gym_df.empty:
        gym_df["start_local"] = gym_df["start_time"].dt.tz_convert(ct)
        gym_df["gym_hour"] = (
            gym_df["start_local"].dt.hour + gym_df["start_local"].dt.minute / 60
        )
        gym_df["date"] = pd.to_datetime(gym_df["start_local"].dt.date)
        gym_daily = (
            gym_df.groupby("date")
            .agg(gym_hour=("gym_hour", "first"), gym_duration=("duration", "sum"))
            .reset_index()
        )
        result = result.merge(gym_daily, on="date", how="outer")

    result = result.fillna(0)
    return result


def load_calendar_full(calendar_dir: str) -> pd.DataFrame:
    """Load and process all calendar events with overlap handling and slash splitting."""
    start_date = datetime(2021, 5, 24)
    end_date = datetime(2026, 3, 15)
    df = parse_ics_files(calendar_dir, start_date, end_date)
    if df.empty:
        return df

    # Normalize calendar names
    df["calendar_name"] = df["calendar_name"].replace(CALENDAR_NAME_ALIASES)

    df = process_sleep_events(df)
    df = process_slash_events(df)
    df = process_book_events(df)
    df = process_overlaps(df)

    # Reclassify hive/job events as blue/productive
    df.loc[df["event_name"].str.lower().str.contains(": hive"), "calendar_name"] = (
        PRIMARY_WORK_CALENDAR_NAME
    )
    df.loc[df["event_name"].str.lower() == "job", "calendar_name"] = (
        PRIMARY_WORK_CALENDAR_NAME
    )

    # Assign category
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

    # Convert to local time for date assignment
    import pytz

    ct = pytz.timezone("US/Central")
    df["start_local"] = df["start_time"].dt.tz_convert(ct)
    df["date"] = pd.to_datetime(df["start_local"].dt.date)

    # Normalize event names for grouping
    df["event_lower"] = df["event_name"].str.lower().str.strip()

    return df


def analyze_calendar_time_breakdown(
    cal_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    distracted_df: pd.DataFrame | None = None,
    period_label: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Detailed breakdown of where time goes in a given period.

    Uses overlap-adjusted calendar data and work sheet data to account for 24h/day.
    """
    df = cal_df.copy()
    daily = daily_df.copy()

    if start_date:
        sd = pd.Timestamp(start_date)
        df = df[df["date"] >= sd]
        daily = daily[daily["date"] >= sd]
    if end_date:
        ed = pd.Timestamp(end_date)
        df = df[df["date"] <= ed]
        daily = daily[daily["date"] <= ed]

    if df.empty or daily.empty:
        print(f"\n{'='*60}")
        print(f"  {period_label}: No data")
        return

    # Work days only: days where hours_working > 0
    work_days = set(daily[daily["Hours Working"] > 0]["date"])
    n_work_days = len(work_days)
    n_total_days = len(daily["date"].unique())

    print(f"\n{'='*70}")
    print(f"  CALENDAR TIME BREAKDOWN: {period_label}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Total days: {n_total_days}, Work days: {n_work_days}")
    print(f"{'='*70}")

    # --- Per-category daily totals ---
    cat_daily = df.groupby(["date", "category"])["duration"].sum().unstack(fill_value=0)

    for cat in ["green", "things", "waste", "blue", "other"]:
        if cat not in cat_daily.columns:
            cat_daily[cat] = 0.0

    # Overall averages (all days)
    print(f"\n--- Average hours/day (ALL {n_total_days} days) ---")
    for cat in ["green", "blue", "things", "waste", "other"]:
        mean_val = cat_daily[cat].mean()
        label = {
            "green": "Sleep/meals/gym",
            "blue": "Productive (blue)",
            "things": "Things (social/personal)",
            "waste": "Waste Time",
            "other": "Other calendars",
        }[cat]
        print(f"  {label:35s}: {mean_val:.1f}h")
    print(f"  {'Calendar total':35s}: {cat_daily.sum(axis=1).mean():.1f}h")

    # Work day averages
    cat_work = cat_daily[cat_daily.index.isin(work_days)]
    if not cat_work.empty:
        print(f"\n--- Average hours/day (WORK days only, n={n_work_days}) ---")
        for cat in ["green", "blue", "things", "waste", "other"]:
            mean_val = cat_work[cat].mean()
            label = {
                "green": "Sleep/meals/gym",
                "blue": "Productive (blue)",
                "things": "Things (social/personal)",
                "waste": "Waste Time",
                "other": "Other calendars",
            }[cat]
            print(f"  {label:35s}: {mean_val:.1f}h")

        # Add work sheet data
        daily_work = daily[daily["date"].isin(work_days)]
        avg_hours = daily_work["Hours Working"].mean()
        avg_distr = (
            daily_work["Length Distractions"].mean() / 60
            if "Length Distractions" in daily_work.columns
            else 0
        )
        print(f"  {'Sheet focused work':35s}: {avg_hours:.1f}h")
        print(f"  {'Sheet distractions':35s}: {avg_distr:.1f}h")
        total_cal = cat_work.sum(axis=1).mean()
        total_sheet = avg_hours + avg_distr
        print(f"  {'Calendar total':35s}: {total_cal:.1f}h")
        print(
            f"  {'Unaccounted (24 - cal - sheet)':35s}: {24 - total_cal - total_sheet + cat_work['blue'].mean():.1f}h"
        )
        # Note: blue time and sheet time overlap, so unaccounted = 24 - green - things - waste - other - sheet_total

    # --- Detailed event breakdown within each category ---
    for cat, cat_label in [
        ("waste", "WASTE TIME"),
        ("things", "THINGS (social/personal)"),
        ("green", "SLEEP/MEALS/GYM"),
        ("blue", "PRODUCTIVE (blue)"),
    ]:
        cat_events = df[df["category"] == cat].copy()
        if cat_events.empty:
            continue

        print(
            f"\n--- {cat_label} breakdown (top events, avg hours/day across all days) ---"
        )
        # Group by normalized event name
        event_totals = cat_events.groupby("event_lower")["duration"].sum()
        event_totals = event_totals.sort_values(ascending=False)

        # Per-day average
        for event_name, total_hours in event_totals.head(20).items():
            avg = total_hours / n_total_days
            if avg < 0.01:
                continue
            print(
                f"    {str(event_name):35s}: {avg:.2f}h/day  ({total_hours:.0f}h total)"
            )

    # --- Weekend vs weekday comparison ---
    cat_daily_wd = cat_daily.copy()
    cat_daily_wd["dow"] = cat_daily_wd.index.dayofweek
    weekday = cat_daily_wd[cat_daily_wd["dow"] < 5]
    weekend = cat_daily_wd[cat_daily_wd["dow"] >= 5]

    if not weekend.empty and not weekday.empty:
        print("\n--- Weekday vs Weekend (hours/day) ---")
        print(f"  {'Category':35s} {'Weekday':>10s} {'Weekend':>10s} {'Diff':>10s}")
        for cat in ["green", "blue", "things", "waste"]:
            wd_mean = weekday[cat].mean()
            we_mean = weekend[cat].mean()
            label = {
                "green": "Sleep/meals/gym",
                "blue": "Productive",
                "things": "Things",
                "waste": "Waste",
            }[cat]
            print(
                f"  {label:35s} {wd_mean:10.1f} {we_mean:10.1f} {we_mean - wd_mean:+10.1f}"
            )

    # --- Monthly trend ---
    cat_daily_m = cat_daily.copy()
    cat_daily_m["month"] = cat_daily_m.index.to_period("M")
    monthly = cat_daily_m.groupby("month").mean(numeric_only=True)

    if len(monthly) > 1:
        print("\n--- Monthly trend (hours/day) ---")
        print(
            f"  {'Month':10s} {'Sleep/meal':>10s} {'Blue':>10s} {'Things':>10s} {'Waste':>10s}"
        )
        for month, row in monthly.tail(12).iterrows():
            print(
                f"  {str(month):10s} {row['green']:10.1f} {row['blue']:10.1f} {row['things']:10.1f} {row['waste']:10.1f}"
            )


def build_analysis_df(
    daily_summary: pd.DataFrame,
    supplements: pd.DataFrame,
    calendar_data: pd.DataFrame,
    distracted_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all data sources into one analysis-ready DataFrame."""
    df = daily_summary.merge(supplements, on="date", how="left")
    df = df.merge(calendar_data, on="date", how="left")

    # Merge work start time from distracted CSV
    if distracted_df is not None:
        work_starts = extract_work_start_time(distracted_df)
        df = df.merge(work_starts, on="date", how="left")

    # Fill missing supplement/calendar values with 0
    fill_cols = [c for c in df.columns if c not in daily_summary.columns or c == "date"]
    for c in fill_cols:
        if df[c].dtype in [np.float64, np.int64, float, int]:
            df[c] = df[c].fillna(0)

    # Add derived features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    # Regime indicators
    df["is_hive"] = (df["regime"] == "Hive").astype(int)
    df["is_mats"] = (df["regime"] == "Mats").astype(int)
    df["is_diesl"] = (df["regime"] == "diesl").astype(int)

    # Binary supplement indicators (did you take it at all?)
    all_supps = list(SUPPLEMENT_ALIASES.values())
    for s in set(all_supps):
        if s in df.columns:
            df[f"{s}_any"] = (df[s] > 0).astype(int)

    # Lag features
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_hours"] = df["Hours Working"].shift(1)
    df["prev_productivity"] = df["work_productivity"].shift(1)
    df["hours_7d_ago"] = df["Hours Working"].shift(7)

    # Rolling consistency features
    df["hours_rolling_7d_mean"] = (
        df["Hours Working"].rolling(7, min_periods=3).mean().shift(1)
    )
    df["hours_rolling_7d_std"] = (
        df["Hours Working"].rolling(7, min_periods=3).std().shift(1)
    )
    df["hours_rolling_30d_mean"] = (
        df["Hours Working"].rolling(30, min_periods=10).mean().shift(1)
    )

    # Streak: consecutive days worked (>=2 hours)
    # Reset streak when there's a gap > 1 day between rows (missing CSV days)
    dates = df["date"].values
    worked = (df["Hours Working"] >= 2).values
    streak_vals = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if i > 0:
            day_gap = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
            if day_gap > 1:
                streak_vals[i] = 1 if worked[i] else 0
                continue
        if worked[i]:
            streak_vals[i] = (streak_vals[i - 1] + 1) if i > 0 else 1
        else:
            streak_vals[i] = 0
    # work_streak = yesterday's streak, but NaN if previous row isn't yesterday
    shifted_streak = pd.Series(streak_vals, dtype=float).shift(1).values
    for i in range(1, len(df)):
        day_gap = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
        if day_gap > 1:
            shifted_streak[i] = 0.0
    df["work_streak"] = shifted_streak

    # Days since last rest day (also gap-aware)
    rest_vals = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        is_rest = df["Hours Working"].iloc[i] < 2
        if is_rest:
            rest_vals[i] = 0
        elif i == 0:
            rest_vals[i] = 0
        else:
            day_gap = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
            if day_gap > 1:
                # Gap contains unknown days; conservatively reset
                rest_vals[i] = 0
            else:
                rest_vals[i] = rest_vals[i - 1] + 1
    df["days_since_rest"] = rest_vals

    return df


# ============================================================================
# Analysis 1: Sleep -> Work Quality
# ============================================================================


def analyze_sleep(df: pd.DataFrame) -> None:
    """How does sleep predict work output?"""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: SLEEP -> WORK OUTPUT")
    print("=" * 80)

    # Filter to days with valid sleep data and actual work
    mask = (
        (df["sleep_hours"] > 0)
        & (df["Hours Working"] > 0)
        & (~df["is_weekend"].astype(bool))
    )
    work_days = df[mask].copy()
    print(f"\nDays with sleep data & work (weekdays): {len(work_days)}")

    # 1a. Self-reported sleep stats
    print(
        f"\nSelf-reported sleep: mean={work_days['sleep_hours'].mean():.1f}h, "
        f"std={work_days['sleep_hours'].std():.1f}h, "
        f"median={work_days['sleep_hours'].median():.1f}h"
    )

    # 1b. Compare self-reported vs calendar-derived sleep
    has_both = work_days[
        (work_days["sleep_hours"] > 0) & (work_days["calendar_sleep_hours"] > 0)
    ]
    if len(has_both) > 20:
        r, p = stats.pearsonr(has_both["sleep_hours"], has_both["calendar_sleep_hours"])
        diff = has_both["calendar_sleep_hours"] - has_both["sleep_hours"]
        print(f"\nSelf-reported vs Calendar sleep (n={len(has_both)}):")
        print(f"  Correlation: r={r:.3f} (p={p:.1e})")
        print(
            f"  Mean difference (cal - self): {diff.mean():.2f}h +/- {diff.std():.2f}h"
        )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            has_both["sleep_hours"], has_both["calendar_sleep_hours"], alpha=0.3, s=15
        )
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="y=x")
        ax.set_xlabel("Self-reported sleep (hours)")
        ax.set_ylabel("Calendar-derived sleep (hours)")
        ax.set_title(f"Sleep measurement comparison (r={r:.2f})")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # 1c. Sleep vs Hours Working - regression
    print("\n--- Sleep -> Hours Working (OLS) ---")
    # Simple regression
    X_simple = sm.add_constant(work_days[["sleep_hours"]])
    model_simple = sm.OLS(work_days["Hours Working"], X_simple).fit()
    print(
        f"  Simple: coef={model_simple.params['sleep_hours']:.3f}, "
        f"R²={model_simple.rsquared:.3f}, p={model_simple.pvalues['sleep_hours']:.3e}"
    )

    # With nap
    if (work_days["nap_hours"] > 0).sum() > 20:
        X_nap = sm.add_constant(work_days[["sleep_hours", "nap_hours"]])
        model_nap = sm.OLS(work_days["Hours Working"], X_nap).fit()
        print(
            f"  With nap: sleep coef={model_nap.params['sleep_hours']:.3f}, "
            f"nap coef={model_nap.params['nap_hours']:.3f}, R²={model_nap.rsquared:.3f}"
        )

    # Quadratic (optimal sleep amount?)
    work_days["sleep_sq"] = work_days["sleep_hours"] ** 2
    X_quad = sm.add_constant(work_days[["sleep_hours", "sleep_sq"]])
    model_quad = sm.OLS(work_days["Hours Working"], X_quad).fit()
    if model_quad.pvalues["sleep_sq"] < 0.1:
        optimal = -model_quad.params["sleep_hours"] / (
            2 * model_quad.params["sleep_sq"]
        )
        print(
            f"  Quadratic fit suggests optimal sleep ~{optimal:.1f}h "
            f"(sq term p={model_quad.pvalues['sleep_sq']:.3f})"
        )

    # 1d. Sleep buckets
    print("\n--- Hours Working by sleep bucket ---")
    bins = [0, 5, 6, 7, 8, 9, 15]
    labels = ["<5h", "5-6h", "6-7h", "7-8h", "8-9h", "9h+"]
    work_days["sleep_bucket"] = pd.cut(
        work_days["sleep_hours"], bins=bins, labels=labels, right=False
    )
    bucket_stats = work_days.groupby("sleep_bucket", observed=True).agg(
        hours_mean=("Hours Working", "mean"),
        hours_std=("Hours Working", "std"),
        prod_mean=("work_productivity", "mean"),
        n=("Hours Working", "count"),
    )
    print(bucket_stats.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(work_days["sleep_hours"], work_days["Hours Working"], alpha=0.15, s=10)
    # Add regression line
    x_range = np.linspace(
        work_days["sleep_hours"].min(), work_days["sleep_hours"].max(), 100
    )
    ax.plot(
        x_range,
        model_simple.predict(sm.add_constant(x_range)),
        "r-",
        lw=2,
        label=f"R²={model_simple.rsquared:.3f}",
    )
    ax.set_xlabel("Self-reported sleep (hours)")
    ax.set_ylabel("Hours Working")
    ax.set_title("Sleep vs Hours Working")
    ax.legend()

    ax = axes[1]
    bucket_stats["hours_mean"].plot(
        kind="bar", ax=ax, yerr=bucket_stats["hours_std"] / np.sqrt(bucket_stats["n"])
    )
    ax.set_ylabel("Hours Working (mean +/- SE)")
    ax.set_title("Hours Working by Sleep Bucket")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()

    # 1e. Calendar sleep vs work (if available)
    cal_mask = (
        (df["calendar_sleep_hours"] > 0)
        & (df["Hours Working"] > 0)
        & (~df["is_weekend"].astype(bool))
    )
    cal_days = df[cal_mask]
    if len(cal_days) > 50:
        X_cal = sm.add_constant(cal_days[["calendar_sleep_hours"]])
        model_cal = sm.OLS(cal_days["Hours Working"], X_cal).fit()
        print(
            f"\n  Calendar sleep -> Hours Working: coef={model_cal.params['calendar_sleep_hours']:.3f}, "
            f"R²={model_cal.rsquared:.3f}, p={model_cal.pvalues['calendar_sleep_hours']:.3e}"
        )


# ============================================================================
# Analysis 2: Stimulants -> Work Quality
# ============================================================================


def analyze_stimulants(df: pd.DataFrame) -> None:
    """How do various stimulants affect work output?"""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: STIMULANTS -> WORK OUTPUT")
    print("=" * 80)

    work_days = df[(df["Hours Working"] > 0) & (~df["is_weekend"].astype(bool))].copy()
    print(f"\nWeekday work days: {len(work_days)}")

    # 2a. Usage frequency for each supplement
    print("\n--- Supplement usage frequency ---")
    all_supps_in_data = sorted(set(SUPPLEMENT_ALIASES.values()) & set(df.columns))
    usage_stats = []
    for s in all_supps_in_data:
        n_used = (work_days[s] > 0).sum()
        if n_used > 0:
            mean_dose = work_days.loc[work_days[s] > 0, s].mean()
            usage_stats.append(
                {
                    "supplement": s,
                    "days_used": n_used,
                    "pct": n_used / len(work_days) * 100,
                    "mean_dose_when_used": mean_dose,
                }
            )
    usage_df = pd.DataFrame(usage_stats).sort_values("days_used", ascending=False)
    print(usage_df.to_string(index=False))

    # 2b. Simple with/without comparison for each supplement
    print("\n--- With vs Without comparison (Hours Working) ---")
    comparisons = []
    for s in all_supps_in_data:
        n_with = (work_days[s] > 0).sum()
        if n_with < 10:
            continue
        with_hours = work_days.loc[work_days[s] > 0, "Hours Working"]
        without_hours = work_days.loc[work_days[s] == 0, "Hours Working"]
        diff = with_hours.mean() - without_hours.mean()
        t_stat, p_val = stats.ttest_ind(with_hours, without_hours)
        # Also Mann-Whitney for robustness
        u_stat, u_p = stats.mannwhitneyu(
            with_hours, without_hours, alternative="two-sided"
        )
        comparisons.append(
            {
                "supplement": s,
                "with_mean": with_hours.mean(),
                "without_mean": without_hours.mean(),
                "diff": diff,
                "t_pval": p_val,
                "mw_pval": u_p,
                "n_with": n_with,
                "n_without": len(without_hours),
            }
        )
    comp_df = pd.DataFrame(comparisons).sort_values("diff", ascending=False)
    print(comp_df.to_string(index=False, float_format="%.3f"))

    # Confounding warning for supplements used only in specific regimes
    print(
        "\n  WARNING: piracetam/choline were almost exclusively used during early Hive period"
    )
    print("  (65/71 days overlap). Their apparent effect is confounded with regime.")
    # Regime-controlled comparison for supplements with enough data
    print("\n--- With vs Without, controlling for regime (Hive-only subset) ---")
    hive_days = work_days[work_days["regime"] == "Hive"]
    for s in all_supps_in_data:
        n_with = (hive_days[s] > 0).sum()
        n_without = (hive_days[s] == 0).sum()
        if n_with < 5 or n_without < 5:
            continue
        with_h = hive_days.loc[hive_days[s] > 0, "Hours Working"]
        without_h = hive_days.loc[hive_days[s] == 0, "Hours Working"]
        t, p = stats.ttest_ind(with_h, without_h)
        print(
            f"  {s:20s}: with={with_h.mean():.2f}h without={without_h.mean():.2f}h "
            f"diff={with_h.mean() - without_h.mean():+.2f}h p={p:.3f} (n={n_with}/{n_without})"
        )

    # 2c. Same for Value (quality)
    quality_days = work_days.dropna(subset=["Value"])
    print("\n--- With vs Without comparison (Value) ---")
    q_comparisons = []
    for s in all_supps_in_data:
        n_with = (quality_days[s] > 0).sum()
        if n_with < 10:
            continue
        with_val = quality_days.loc[quality_days[s] > 0, "Value"]
        without_val = quality_days.loc[quality_days[s] == 0, "Value"]
        diff = with_val.mean() - without_val.mean()
        t_stat, p_val = stats.ttest_ind(with_val, without_val)
        q_comparisons.append(
            {
                "supplement": s,
                "with_mean": with_val.mean(),
                "without_mean": without_val.mean(),
                "diff": diff,
                "t_pval": p_val,
                "n_with": n_with,
            }
        )
    q_df = pd.DataFrame(q_comparisons).sort_values("diff", ascending=False)
    print(q_df.to_string(index=False, float_format="%.3f"))

    # 2d. Dose-response for caffeine and adderall
    print("\n--- Dose-response analysis ---")
    for s in ["caffeine", "adderall"]:
        if s not in work_days.columns:
            continue
        used = work_days[work_days[s] > 0]
        if len(used) < 20:
            continue
        r, p = stats.pearsonr(used[s], used["Hours Working"])
        print(f"  {s} dose vs Hours Working (n={len(used)}): r={r:.3f}, p={p:.3f}")
        r2, p2 = (
            stats.pearsonr(used[s], used["Value"].dropna())
            if used["Value"].notna().sum() > 20
            else (0, 1)
        )
        print(f"  {s} dose vs Value: r={r2:.3f}, p={p2:.3f}")

    # 2e. Multiple regression controlling for confounds
    print("\n--- Multiple regression: stimulants controlling for sleep, regime ---")
    feature_cols = ["sleep_hours", "is_hive", "is_mats", "is_diesl"]
    for s in STIMULANT_NAMES:
        col = f"{s}_any"
        if col in work_days.columns and (work_days[col] > 0).sum() >= 10:
            feature_cols.append(col)

    valid = work_days.dropna(subset=feature_cols + ["Hours Working"])
    if len(valid) > 50:
        X = sm.add_constant(valid[feature_cols])
        model = sm.OLS(valid["Hours Working"], X).fit()
        print(model.summary2().tables[1].to_string())

    # 2f. Supplement interaction: caffeine + adderall
    if "caffeine_any" in work_days.columns and "adderall_any" in work_days.columns:
        caf_only = work_days[
            (work_days["caffeine_any"] == 1) & (work_days["adderall_any"] == 0)
        ]
        add_only = work_days[
            (work_days["adderall_any"] == 1) & (work_days["caffeine_any"] == 0)
        ]
        both = work_days[
            (work_days["caffeine_any"] == 1) & (work_days["adderall_any"] == 1)
        ]
        neither = work_days[
            (work_days["caffeine_any"] == 0) & (work_days["adderall_any"] == 0)
        ]
        if min(len(caf_only), len(add_only), len(both), len(neither)) >= 5:
            print("\n--- Caffeine x Adderall interaction ---")
            print(
                f"  Neither:          {neither['Hours Working'].mean():.2f}h (n={len(neither)})"
            )
            print(
                f"  Caffeine only:    {caf_only['Hours Working'].mean():.2f}h (n={len(caf_only)})"
            )
            print(
                f"  Adderall only:    {add_only['Hours Working'].mean():.2f}h (n={len(add_only)})"
            )
            print(
                f"  Both:             {both['Hours Working'].mean():.2f}h (n={len(both)})"
            )

    # Plot: bar chart of with/without for top supplements
    if not comp_df.empty:
        plot_supps = comp_df.head(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(plot_supps))
        width = 0.35
        ax.bar(
            x - width / 2,
            plot_supps["without_mean"],
            width,
            label="Without",
            color="lightblue",
        )
        ax.bar(
            x + width / 2, plot_supps["with_mean"], width, label="With", color="coral"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_supps["supplement"], rotation=30, ha="right")
        ax.set_ylabel("Hours Working")
        ax.set_title("Hours Working: With vs Without Supplement")
        ax.legend()
        # Add significance stars
        for i, row in enumerate(plot_supps.itertuples()):
            if row.t_pval < 0.01:
                ax.text(
                    i, max(row.with_mean, row.without_mean) + 0.1, "**", ha="center"
                )
            elif row.t_pval < 0.05:
                ax.text(i, max(row.with_mean, row.without_mean) + 0.1, "*", ha="center")
        plt.tight_layout()
        plt.show()


# ============================================================================
# Analysis 3: Consistency -> Work Quality
# ============================================================================


def analyze_consistency(df: pd.DataFrame) -> None:
    """Does consistency in working predict better output?"""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: CONSISTENCY -> WORK OUTPUT")
    print("=" * 80)

    work_days = df[(df["Hours Working"] > 0) & (~df["is_weekend"].astype(bool))].copy()
    work_days = work_days.dropna(subset=["hours_rolling_7d_mean", "prev_hours"])

    print(f"\nWeekday work days with lag data: {len(work_days)}")

    # 3a. Previous day's hours predicting today
    print("\n--- Previous day -> Today ---")
    r, p = stats.pearsonr(work_days["prev_hours"], work_days["Hours Working"])
    print(f"  Yesterday's hours vs today: r={r:.3f}, p={p:.1e}")
    r7, p7 = stats.pearsonr(
        work_days["hours_7d_ago"].dropna(),
        work_days.loc[work_days["hours_7d_ago"].notna(), "Hours Working"],
    )
    print(f"  7 days ago vs today: r={r7:.3f}, p={p7:.1e}")

    # 3b. Rolling mean (momentum)
    r_roll, p_roll = stats.pearsonr(
        work_days["hours_rolling_7d_mean"], work_days["Hours Working"]
    )
    print(f"  7-day rolling mean vs today: r={r_roll:.3f}, p={p_roll:.1e}")

    valid_30 = work_days.dropna(subset=["hours_rolling_30d_mean"])
    if len(valid_30) > 50:
        r_30, p_30 = stats.pearsonr(
            valid_30["hours_rolling_30d_mean"], valid_30["Hours Working"]
        )
        print(f"  30-day rolling mean vs today: r={r_30:.3f}, p={p_30:.1e}")

    # 3c. Variability (low std = consistent)
    valid_std = work_days.dropna(subset=["hours_rolling_7d_std"])
    if len(valid_std) > 50:
        r_std, p_std = stats.pearsonr(
            valid_std["hours_rolling_7d_std"], valid_std["Hours Working"]
        )
        print(f"\n  7-day rolling std vs today: r={r_std:.3f}, p={p_std:.1e}")
        print("  (Negative means: more consistent past week -> more work today)")

    # 3d. Streak length
    valid_streak = work_days.dropna(subset=["work_streak"])
    if len(valid_streak) > 50:
        r_str, p_str = stats.pearsonr(
            valid_streak["work_streak"], valid_streak["Hours Working"]
        )
        print(f"\n  Work streak length vs today: r={r_str:.3f}, p={p_str:.1e}")

        # Bucket by streak
        bins = [0, 1, 3, 5, 10, 100]
        labels = ["0d", "1-2d", "3-4d", "5-9d", "10d+"]
        valid_streak["streak_bucket"] = pd.cut(
            valid_streak["work_streak"], bins=bins, labels=labels, right=False
        )
        streak_stats = valid_streak.groupby("streak_bucket", observed=True).agg(
            hours_mean=("Hours Working", "mean"),
            n=("Hours Working", "count"),
        )
        print("\n  Hours Working by streak length:")
        print(streak_stats.to_string())

    # 3e. Days since rest
    r_rest, p_rest = stats.pearsonr(
        work_days["days_since_rest"], work_days["Hours Working"]
    )
    print(f"\n  Days since rest day vs today: r={r_rest:.3f}, p={p_rest:.1e}")
    bins_rest = [0, 1, 3, 5, 7, 14, 100]
    labels_rest = ["0", "1-2", "3-4", "5-6", "7-13", "14+"]
    work_days["rest_bucket"] = pd.cut(
        work_days["days_since_rest"], bins=bins_rest, labels=labels_rest, right=False
    )
    rest_stats = work_days.groupby("rest_bucket", observed=True).agg(
        hours_mean=("Hours Working", "mean"),
        n=("Hours Working", "count"),
    )
    print("\n  Hours Working by days since rest:")
    print(rest_stats.to_string())

    # 3f. Multiple regression with consistency features
    print("\n--- Consistency regression ---")
    consistency_cols = [
        "prev_hours",
        "hours_rolling_7d_mean",
        "hours_rolling_7d_std",
        "work_streak",
        "days_since_rest",
        "sleep_hours",
    ]
    valid = work_days.dropna(subset=consistency_cols)
    if len(valid) > 50:
        X = sm.add_constant(valid[consistency_cols])
        model = sm.OLS(valid["Hours Working"], X).fit()
        print(model.summary2().tables[1].to_string())

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(work_days["prev_hours"], work_days["Hours Working"], alpha=0.15, s=10)
    ax.set_xlabel("Yesterday's Hours")
    ax.set_ylabel("Today's Hours")
    ax.set_title(f"Day-to-day autocorrelation (r={r:.2f})")

    ax = axes[0, 1]
    ax.scatter(
        work_days["hours_rolling_7d_mean"], work_days["Hours Working"], alpha=0.15, s=10
    )
    ax.set_xlabel("7-day Rolling Mean Hours")
    ax.set_ylabel("Today's Hours")
    ax.set_title(f"Momentum effect (r={r_roll:.2f})")

    if len(valid_streak) > 50:
        ax = axes[1, 0]
        streak_stats["hours_mean"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Hours Working")
        ax.set_title("Hours by Streak Length")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax = axes[1, 1]
    rest_stats["hours_mean"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Hours Working")
    ax.set_title("Hours by Days Since Rest")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Analysis 3b: Split by Regime
# ============================================================================


def analyze_by_regime(df: pd.DataFrame) -> None:
    """How do baselines and effects differ by regime?"""
    print("\n" + "=" * 80)
    print("ANALYSIS 3b: SPLIT BY REGIME")
    print("=" * 80)

    work_days = df[(df["Hours Working"] > 0) & (~df["is_weekend"].astype(bool))].copy()

    # 3b.1 Baseline stats by regime
    print("\n--- Baseline stats by regime ---")
    regime_stats = work_days.groupby("regime").agg(
        n=("Hours Working", "count"),
        hours_mean=("Hours Working", "mean"),
        hours_std=("Hours Working", "std"),
        hours_median=("Hours Working", "median"),
        value_mean=("Value", "mean"),
        prod_mean=("work_productivity", "mean"),
        sleep_mean=(
            "sleep_hours",
            lambda x: x[x > 0].mean() if (x > 0).any() else np.nan,
        ),
    )
    print(regime_stats.to_string(float_format="%.2f"))

    # 3b.2 Sleep effect by regime
    print("\n--- Sleep -> Hours Working by regime ---")
    for regime in ["Hive", "Mats", "diesl", "Personal"]:
        subset = work_days[
            (work_days["regime"] == regime) & (work_days["sleep_hours"] > 0)
        ]
        if len(subset) < 20:
            print(f"  {regime}: insufficient data (n={len(subset)})")
            continue
        r, p = stats.pearsonr(subset["sleep_hours"], subset["Hours Working"])
        print(f"  {regime} (n={len(subset)}): sleep->hours r={r:.3f}, p={p:.3f}")

    # 3b.3 Consistency by regime
    print("\n--- Consistency by regime ---")
    for regime in ["Hive", "Mats", "diesl", "Personal"]:
        subset = work_days[(work_days["regime"] == regime)].dropna(
            subset=["prev_hours"]
        )
        if len(subset) < 20:
            continue
        r, p = stats.pearsonr(subset["prev_hours"], subset["Hours Working"])
        print(f"  {regime} (n={len(subset)}): prev_day->today r={r:.3f}, p={p:.3f}")

    # 3b.4 Stimulant effects by regime
    print("\n--- Stimulant effects by regime ---")
    for s in STIMULANT_NAMES:
        col = f"{s}_any"
        if col not in work_days.columns:
            continue
        for regime in ["Hive", "Personal", "diesl"]:
            subset = work_days[work_days["regime"] == regime]
            n_with = (subset[col] == 1).sum()
            n_without = (subset[col] == 0).sum()
            if n_with < 5 or n_without < 5:
                continue
            with_h = subset.loc[subset[col] == 1, "Hours Working"].mean()
            without_h = subset.loc[subset[col] == 0, "Hours Working"].mean()
            diff = with_h - without_h
            t, p = stats.ttest_ind(
                subset.loc[subset[col] == 1, "Hours Working"],
                subset.loc[subset[col] == 0, "Hours Working"],
            )
            sig = "*" if p < 0.05 else ""
            print(
                f"  {s:12s} | {regime:8s} | with={with_h:.1f}h without={without_h:.1f}h "
                f"diff={diff:+.1f}h p={p:.3f}{sig}"
            )

    # Plot: regime comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    regime_order = ["Hive", "Mats", "diesl", "Personal"]
    regime_data = [
        work_days[work_days["regime"] == r]["Hours Working"] for r in regime_order
    ]
    bp = ax.boxplot(regime_data, labels=regime_order, patch_artist=True)
    colors = ["#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Hours Working")
    ax.set_title("Hours Working by Regime")

    ax = axes[1]
    # Time series of monthly average hours by regime
    work_days["year_month"] = work_days["date"].dt.to_period("M")
    monthly = (
        work_days.groupby(["year_month", "regime"])["Hours Working"].mean().unstack()
    )
    monthly.index = monthly.index.to_timestamp()
    for col in monthly.columns:
        ax.plot(monthly.index, monthly[col], label=col, alpha=0.7)
    ax.set_ylabel("Avg Hours Working")
    ax.set_title("Monthly Avg Hours by Regime")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Analysis 4: Additional Factors
# ============================================================================


def analyze_additional_factors(df: pd.DataFrame) -> None:
    """Other factors: day of week, meals, meditation, distractions, season."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: ADDITIONAL FACTORS")
    print("=" * 80)

    work_days = df[(df["Hours Working"] > 0)].copy()

    # 4a. Day of week
    print("\n--- Day of week ---")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_stats = work_days.groupby("day_of_week").agg(
        hours_mean=("Hours Working", "mean"),
        n=("Hours Working", "count"),
    )
    dow_stats.index = [dow_names[i] for i in dow_stats.index]
    print(dow_stats.to_string(float_format="%.2f"))

    # 4b. Meals before/during work
    print("\n--- Meals (1000 cal units) vs Hours Working ---")
    meal_days = work_days[work_days["meals_kcal"] > 0]
    if len(meal_days) > 20:
        r, p = stats.pearsonr(meal_days["meals_kcal"], meal_days["Hours Working"])
        print(f"  Meal intake vs Hours (n={len(meal_days)}): r={r:.3f}, p={p:.3f}")
        bins = [0, 0.5, 1.0, 1.5, 2.0, 10]
        labels = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2+"]
        meal_days = meal_days.copy()
        meal_days["meal_bucket"] = pd.cut(
            meal_days["meals_kcal"], bins=bins, labels=labels, right=False
        )
        meal_stats = meal_days.groupby("meal_bucket", observed=True).agg(
            hours_mean=("Hours Working", "mean"), n=("Hours Working", "count")
        )
        print(meal_stats.to_string(float_format="%.2f"))

    # 4c. Meditation
    print("\n--- Meditation vs Hours Working ---")
    med_any = (work_days["meditation_min"] > 0).sum()
    if med_any > 10:
        with_med = work_days[work_days["meditation_min"] > 0]["Hours Working"]
        without_med = work_days[work_days["meditation_min"] == 0]["Hours Working"]
        t, p = stats.ttest_ind(with_med, without_med)
        print(f"  With meditation: {with_med.mean():.2f}h (n={len(with_med)})")
        print(f"  Without:         {without_med.mean():.2f}h (n={len(without_med)})")
        print(f"  Diff: {with_med.mean() - without_med.mean():.2f}h, p={p:.3f}")

    # 4d. Season/month
    print("\n--- Month of year ---")
    month_stats = work_days.groupby("month").agg(
        hours_mean=("Hours Working", "mean"), n=("Hours Working", "count")
    )
    print(month_stats.to_string(float_format="%.2f"))

    # 4e. Distraction patterns as predictor of next day
    print("\n--- Previous day distractions -> Today's hours ---")
    work_days["prev_distractions"] = work_days["# Distractions"].shift(1)
    valid = work_days.dropna(subset=["prev_distractions"])
    if len(valid) > 50:
        r, p = stats.pearsonr(valid["prev_distractions"], valid["Hours Working"])
        print(f"  Yesterday's distractions vs today's hours: r={r:.3f}, p={p:.3f}")

    # 4f. Calendar waste time vs work hours (same day)
    if "calendar_waste_hours" in work_days.columns:
        waste_days = work_days[work_days["calendar_waste_hours"] > 0]
        if len(waste_days) > 20:
            r, p = stats.pearsonr(
                waste_days["calendar_waste_hours"], waste_days["Hours Working"]
            )
            print(
                f"\n--- Calendar waste time vs Hours Working (n={len(waste_days)}): r={r:.3f}, p={p:.3f}"
            )

    # 4g. Calendar blue time vs sheet hours working (validation)
    if "calendar_blue_hours" in work_days.columns:
        blue_days = work_days[work_days["calendar_blue_hours"] > 0]
        if len(blue_days) > 20:
            r, p = stats.pearsonr(
                blue_days["calendar_blue_hours"], blue_days["Hours Working"]
            )
            print(
                f"\n--- Calendar blue hours vs Sheet Hours Working (n={len(blue_days)}): "
                f"r={r:.3f}, p={p:.3f}"
            )
            print(
                f"  Cal blue mean: {blue_days['calendar_blue_hours'].mean():.2f}h, "
                f"Sheet hours mean: {blue_days['Hours Working'].mean():.2f}h"
            )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    dow_stats["hours_mean"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg Hours Working")
    ax.set_title("Hours by Day of Week")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax = axes[1]
    month_stats["hours_mean"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg Hours Working")
    ax.set_title("Hours by Month")

    if "calendar_blue_hours" in work_days.columns and len(blue_days) > 20:
        ax = axes[2]
        ax.scatter(
            blue_days["calendar_blue_hours"],
            blue_days["Hours Working"],
            alpha=0.15,
            s=10,
        )
        ax.plot([0, 15], [0, 15], "r--", alpha=0.5)
        ax.set_xlabel("Calendar Blue Hours")
        ax.set_ylabel("Sheet Hours Working")
        ax.set_title(f"Calendar vs Sheet (r={r:.2f})")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Analysis 5: Wake Time & Gym Timing
# ============================================================================


def analyze_timing(df: pd.DataFrame) -> None:
    """What wake-up time and gym timing optimize work output?"""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: WAKE TIME & GYM TIMING")
    print("=" * 80)

    work_days = df[(df["Hours Working"] > 0) & (~df["is_weekend"].astype(bool))].copy()

    # ---- 5a. Work start time (from distracted CSV) ----
    has_start = work_days[
        (work_days["work_start_hour"] > 4) & (work_days["work_start_hour"] < 16)
    ].copy()
    print(f"\n--- Work start time -> Hours Working (n={len(has_start)}) ---")
    if len(has_start) > 50:
        r, p = stats.pearsonr(has_start["work_start_hour"], has_start["Hours Working"])
        print(f"  Correlation: r={r:.3f}, p={p:.1e}")
        print(
            f"  Start time: mean={has_start['work_start_hour'].mean():.1f}, "
            f"median={has_start['work_start_hour'].median():.1f}"
        )

        bins = [4, 6, 7, 8, 9, 10, 11, 16]
        labels = ["4-6am", "6-7am", "7-8am", "8-9am", "9-10am", "10-11am", "11am+"]
        has_start["start_bucket"] = pd.cut(
            has_start["work_start_hour"], bins=bins, labels=labels, right=False
        )
        start_stats = has_start.groupby("start_bucket", observed=True).agg(
            hours_mean=("Hours Working", "mean"),
            hours_std=("Hours Working", "std"),
            value_mean=("Value", "mean"),
            n=("Hours Working", "count"),
        )
        print("\n  Hours Working by work start time:")
        print(start_stats.to_string(float_format="%.2f"))

        # Regression with quadratic
        has_start["start_sq"] = has_start["work_start_hour"] ** 2
        X = sm.add_constant(has_start[["work_start_hour", "start_sq"]])
        model = sm.OLS(has_start["Hours Working"], X).fit()
        if model.pvalues["start_sq"] < 0.1:
            optimal = -model.params["work_start_hour"] / (2 * model.params["start_sq"])
            print(
                f"\n  Quadratic fit: optimal work start ~{int(optimal)}:"
                f"{int((optimal % 1) * 60):02d} "
                f"(sq term p={model.pvalues['start_sq']:.3f})"
            )

    # ---- 5b. Calendar wake time ----
    has_wake = work_days[
        (work_days["wake_hour"] > 3) & (work_days["wake_hour"] < 14)
    ].copy()
    print(f"\n--- Calendar wake time -> Hours Working (n={len(has_wake)}) ---")
    if len(has_wake) > 50:
        r, p = stats.pearsonr(has_wake["wake_hour"], has_wake["Hours Working"])
        print(f"  Correlation: r={r:.3f}, p={p:.1e}")
        print(
            f"  Wake time: mean={has_wake['wake_hour'].mean():.1f}, "
            f"median={has_wake['wake_hour'].median():.1f}"
        )

        bins_w = [3, 6, 7, 8, 9, 10, 11, 14]
        labels_w = ["3-6am", "6-7am", "7-8am", "8-9am", "9-10am", "10-11am", "11am+"]
        has_wake["wake_bucket"] = pd.cut(
            has_wake["wake_hour"], bins=bins_w, labels=labels_w, right=False
        )
        wake_stats = has_wake.groupby("wake_bucket", observed=True).agg(
            hours_mean=("Hours Working", "mean"),
            hours_std=("Hours Working", "std"),
            value_mean=("Value", "mean"),
            n=("Hours Working", "count"),
        )
        print("\n  Hours Working by wake time:")
        print(wake_stats.to_string(float_format="%.2f"))

    # ---- 5c. Wake time by regime ----
    if len(has_wake) > 50:
        print("\n--- Wake time -> Hours Working by regime ---")
        for regime in ["Hive", "Personal", "diesl"]:
            subset = has_wake[has_wake["regime"] == regime]
            if len(subset) < 20:
                continue
            r, p = stats.pearsonr(subset["wake_hour"], subset["Hours Working"])
            print(
                f"  {regime:10s} (n={len(subset)}): r={r:.3f}, p={p:.3f}, "
                f"mean wake={subset['wake_hour'].mean():.1f}"
            )

    # ---- 5d. Gym timing ----
    has_gym = work_days[(work_days.get("gym_hour", pd.Series(dtype=float)) > 0)].copy()
    print(f"\n--- Gym timing -> Hours Working (n={len(has_gym)}) ---")
    if len(has_gym) > 20:
        bins_g = [0, 10, 14, 17, 24]
        labels_g = [
            "morning(<10am)",
            "midday(10-2pm)",
            "afternoon(2-5pm)",
            "evening(5pm+)",
        ]
        has_gym["gym_period"] = pd.cut(
            has_gym["gym_hour"], bins=bins_g, labels=labels_g, right=False
        )
        gym_stats = has_gym.groupby("gym_period", observed=True).agg(
            hours_mean=("Hours Working", "mean"),
            hours_std=("Hours Working", "std"),
            value_mean=("Value", "mean"),
            n=("Hours Working", "count"),
        )
        print("\n  Hours Working by gym timing:")
        print(gym_stats.to_string(float_format="%.2f"))

        # Also: does working out at all help vs not?
        gym_dates = set(has_gym["date"])
        no_gym = work_days[~work_days["date"].isin(gym_dates)]
        _, p_gym = stats.ttest_ind(has_gym["Hours Working"], no_gym["Hours Working"])
        print(
            f"\n  Gym days: {has_gym['Hours Working'].mean():.2f}h (n={len(has_gym)})"
        )
        print(f"  No gym:   {no_gym['Hours Working'].mean():.2f}h (n={len(no_gym)})")
        print(
            f"  Diff: {has_gym['Hours Working'].mean() - no_gym['Hours Working'].mean():+.2f}h, p={p_gym:.3f}"
        )

        # Gym timing by regime
        print("\n--- Gym timing by regime ---")
        for regime in ["Hive", "Personal", "diesl"]:
            subset = has_gym[has_gym["regime"] == regime]
            if len(subset) < 10:
                continue
            period_stats = subset.groupby("gym_period", observed=True).agg(
                hours_mean=("Hours Working", "mean"), n=("Hours Working", "count")
            )
            print(f"\n  {regime}:")
            print(period_stats.to_string(float_format="%.2f"))

    # ---- 5e. Wake-to-work delay ----
    has_both = work_days[
        (work_days["wake_hour"] > 3)
        & (work_days["wake_hour"] < 14)
        & (work_days["work_start_hour"] > 4)
        & (work_days["work_start_hour"] < 16)
    ].copy()
    if len(has_both) > 30:
        has_both["wake_to_work"] = has_both["work_start_hour"] - has_both["wake_hour"]
        # Filter to reasonable delays (0-6h)
        has_both = has_both[
            (has_both["wake_to_work"] >= 0) & (has_both["wake_to_work"] < 6)
        ]
        if len(has_both) > 20:
            r, p = stats.pearsonr(has_both["wake_to_work"], has_both["Hours Working"])
            print(f"\n--- Wake-to-work delay -> Hours Working (n={len(has_both)}) ---")
            print(f"  Correlation: r={r:.3f}, p={p:.1e}")
            print(
                f"  Mean delay: {has_both['wake_to_work'].mean():.1f}h, "
                f"median: {has_both['wake_to_work'].median():.1f}h"
            )
            bins_d = [0, 0.5, 1, 1.5, 2, 3, 6]
            labels_d = ["<30m", "30-60m", "1-1.5h", "1.5-2h", "2-3h", "3h+"]
            has_both["delay_bucket"] = pd.cut(
                has_both["wake_to_work"],
                bins=bins_d,
                labels=labels_d,
                right=False,
            )
            delay_stats = has_both.groupby("delay_bucket", observed=True).agg(
                hours_mean=("Hours Working", "mean"),
                n=("Hours Working", "count"),
            )
            print("\n  Hours Working by wake-to-work delay:")
            print(delay_stats.to_string(float_format="%.2f"))

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    if len(has_start) > 50:
        ax = axes[0, 0]
        start_stats["hours_mean"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Hours Working")
        ax.set_title("Hours by Work Start Time")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    if len(has_wake) > 50:
        ax = axes[0, 1]
        wake_stats["hours_mean"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Hours Working")
        ax.set_title("Hours by Wake Time")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    if len(has_gym) > 20:
        ax = axes[1, 0]
        gym_stats["hours_mean"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Hours Working")
        ax.set_title("Hours by Gym Time of Day")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    if len(has_both) > 20:
        ax = axes[1, 1]
        delay_stats["hours_mean"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Hours Working")
        ax.set_title("Hours by Wake-to-Work Delay")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Analysis 6: Full Model
# ============================================================================


def full_regression(
    df: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Kitchen sink regression with all factors."""
    print("\n" + "=" * 80)
    print("ANALYSIS 6: FULL REGRESSION MODEL")
    print("=" * 80)

    work_days = df[(df["Hours Working"] > 0) & (~df["is_weekend"].astype(bool))].copy()

    # Build feature set
    feature_candidates = [
        "sleep_hours",
        "nap_hours",
        "is_hive",
        "is_mats",
        "is_diesl",
        "prev_hours",
        "hours_rolling_7d_mean",
        "hours_rolling_7d_std",
        "work_streak",
        "days_since_rest",
        "meals_kcal",
        "meditation_min",
        "day_of_week",
        "work_start_hour",
        "wake_hour",
    ]

    # Add supplements that have enough data
    for s in set(SUPPLEMENT_ALIASES.values()):
        col = f"{s}_any"
        if col in work_days.columns and (work_days[col] > 0).sum() >= 15:
            feature_candidates.append(col)

    # Filter to features that exist and have variance
    features = []
    for f in feature_candidates:
        if f in work_days.columns and work_days[f].std() > 0.01:
            features.append(f)

    valid = work_days.dropna(subset=features + ["Hours Working"])
    print(f"\nFeatures: {features}")
    print(f"Observations: {len(valid)}")

    X = sm.add_constant(valid[features])
    y = valid["Hours Working"]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Standardized coefficients for comparison
    print("\n--- Standardized coefficients (effect size comparison) ---")
    X_std = (valid[features] - valid[features].mean()) / valid[features].std()
    X_std = sm.add_constant(X_std)
    model_std = sm.OLS(y, X_std).fit()
    std_coefs = model_std.params.drop("const").sort_values(key=abs, ascending=False)
    for feat, coef in std_coefs.items():
        pval = model_std.pvalues[feat]
        sig = (
            "***"
            if pval < 0.001
            else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        )
        print(f"  {feat:30s}: {coef:+.3f} {sig}")

    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Loading data...")

    distracted = load_distracted_stacked(DISTRACTED_CSV)
    print(
        f"  Distracted entries: {len(distracted)} ({distracted['date'].min().date()} to {distracted['date'].max().date()})"
    )

    supplements = extract_daily_supplements(distracted)
    print(f"  Days with supplement data: {len(supplements)}")

    daily = load_daily_summary_full(DAILY_SUMMARY_CSV)
    print(
        f"  Daily summary rows: {len(daily)} ({daily['date'].min().date()} to {daily['date'].max().date()})"
    )

    print("  Loading calendar data...")
    calendar = load_calendar_sleep(CALENDAR_DIR)
    print(f"  Calendar days: {len(calendar)}")

    df = build_analysis_df(daily, supplements, calendar, distracted)
    print(f"\nFinal analysis DataFrame: {len(df)} days")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Regimes: {df['regime'].value_counts().to_dict()}")

    # Run all analyses
    analyze_sleep(df)
    analyze_stimulants(df)
    analyze_consistency(df)
    analyze_by_regime(df)
    analyze_timing(df)
    analyze_additional_factors(df)
    model = full_regression(df)

    # Calendar time breakdown by period
    print("\n\n" + "#" * 70)
    print("#  DETAILED CALENDAR TIME BREAKDOWN BY PERIOD")
    print("#" * 70)
    print("  Loading full calendar data with overlap/slash handling...")
    cal_full = load_calendar_full(CALENDAR_DIR)
    print(f"  Total calendar events: {len(cal_full)}")

    analyze_calendar_time_breakdown(
        cal_full, daily, distracted, period_label="All Time"
    )
    analyze_calendar_time_breakdown(
        cal_full,
        daily,
        distracted,
        period_label="Past Year (Mar 2025 - Mar 2026)",
        start_date="2025-03-15",
        end_date="2026-03-15",
    )
    analyze_calendar_time_breakdown(
        cal_full,
        daily,
        distracted,
        period_label="Past 6 Months (Sep 2025 - Mar 2026)",
        start_date="2025-09-15",
        end_date="2026-03-15",
    )
    analyze_calendar_time_breakdown(
        cal_full,
        daily,
        distracted,
        period_label="Past 3 Months (Dec 2025 - Mar 2026)",
        start_date="2025-12-15",
        end_date="2026-03-15",
    )
