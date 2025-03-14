# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from icalendar import Calendar
from datetime import datetime, timedelta
import recurring_ical_events
import pytz

color_map = {
    "Things": "grey",
    "Meals, Supplements, Sleep": "green",
    "Waste Time": "red",
    "clark.benham@gmail.com": "blue",
    "cb5ye@virginia.edu": "blue",
}


def parse_ics_files(directory, start_date, end_date):
    all_events = []
    directory = os.path.expanduser(directory)
    ics_files = [
        f for f in os.listdir(directory) if f.endswith(".ics") and "Personal Dates" not in f
    ]
    for ics_file in ics_files:
        calendar_name = os.path.splitext(ics_file)[0]
        file_path = os.path.join(directory, ics_file)
        try:
            with open(file_path, "rb") as f:
                cal = Calendar.from_ical(f.read())
                events = recurring_ical_events.of(cal).between(start_date, end_date)
                for event in events:
                    if event.get("status") and str(event.get("status")).upper() == "CANCELLED":
                        continue
                    dtstart = event.get("dtstart").dt
                    dtend = (
                        event.get("dtend").dt
                        if event.get("dtend")
                        else dtstart + timedelta(hours=1)
                    )
                    duration = (dtend - dtstart).total_seconds() / 3600.0
                    all_events.append(
                        {
                            "event_name": str(event.get("summary")),
                            "calendar_name": calendar_name,
                            "start_time": dtstart,
                            "end_time": dtend,
                            "duration": duration,
                            "metadata": {
                                k: str(event.get(k))
                                for k in event.keys()
                                if k not in ["dtstart", "dtend", "summary"]
                            },
                        }
                    )
        except Exception as e:
            print(f"Error processing {ics_file}: {e}")
    df = pd.DataFrame(all_events)
    if not df.empty:
        for col in ["start_time", "end_time"]:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def process_sleep_events(df):
    df = df.copy()
    bed_df = df[df["event_name"].str.lower() == "bed"].sort_values("start_time")
    woke_df = df[df["event_name"].str.lower() == "woke up"].sort_values("start_time")
    sleep_events = []
    for i, bed_row in bed_df.iterrows():
        bed_time = bed_row["start_time"]
        woke_candidates = woke_df[woke_df["start_time"] > bed_time]
        if woke_candidates.empty:
            continue
        woke_time = woke_candidates.iloc[0]["start_time"]
        base_duration = (woke_time - bed_time).total_seconds() / 3600.0
        intervening = df[(df["start_time"] > bed_time) & (df["end_time"] < woke_time)]
        subtract_duration = intervening["duration"].sum()
        sleep_duration = base_duration - subtract_duration
        sleep_events.append(
            {
                "event_name": "sleep",
                "calendar_name": "Meals, Supplements, Sleep",
                "start_time": bed_time,
                "end_time": woke_time,
                "duration": sleep_duration,
                "metadata": {"subtracted": subtract_duration},
            }
        )
    df = df[~df["event_name"].str.lower().isin(["bed", "woke up"])]
    sleep_df = pd.DataFrame(sleep_events)
    df = pd.concat([df, sleep_df], ignore_index=True)
    return df


def process_slash_events(df):
    df = df.copy()
    new_rows = []
    drop_indices = []
    for idx, row in df.iterrows():
        name = row["event_name"]
        if "/" in name:
            # Special case: "knee/gym" becomes "gym"
            if name.lower() == "knee/gym":
                df.at[idx, "event_name"] = "gym"
                if isinstance(row["metadata"], dict):
                    row["metadata"]["original_name"] = name
                else:
                    row["metadata"] = {"original_name": name}
            else:
                parts = [p.strip() for p in name.split("/")]
                n = len(parts)
                orig_start = row["start_time"]
                orig_end = row["end_time"]
                total_seconds = (orig_end - orig_start).total_seconds()
                segment_seconds = total_seconds / n
                for i, part in enumerate(parts):
                    new_row = row.copy()
                    new_row["event_name"] = part
                    new_row["start_time"] = orig_start + timedelta(seconds=i * segment_seconds)
                    new_row["end_time"] = orig_start + timedelta(seconds=(i + 1) * segment_seconds)
                    new_row["duration"] = segment_seconds / 3600.0  # convert to hours
                    if isinstance(new_row["metadata"], dict):
                        new_row["metadata"]["original_name"] = name
                    else:
                        new_row["metadata"] = {"original_name": name}
                    new_rows.append(new_row)
                drop_indices.append(idx)
    df = df.drop(drop_indices)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def process_book_events(df):
    """this won't handle books with numbers in their title: eg `Kings 12` would be `k1`
    TODO would be to standardize this.
    """
    df = df.copy()
    for idx, row in df.iterrows():
        name = row["event_name"]
        # Process only events that start with "book:".
        if name.lower().startswith("book:"):
            remainder = name[len("book:") :].strip()
            # If the remainder has more than one word, create an abbreviation.
            if " " in remainder:
                abbrev = "".join([w[0] for w in remainder.split()]).lower()
                new_name = "book:" + abbrev  # no space after colon
                df.at[idx, "event_name"] = new_name
                if isinstance(row["metadata"], dict):
                    row["metadata"]["full_title"] = remainder
                else:
                    df.at[idx, "metadata"] = {"full_title": remainder}
            else:
                # remove extra spaces: `book: abc`
                df.loc[idx, "event_name"] = name.replace(": ", ":")
    return df


def process_overlaps(df):
    df = df.copy().reset_index(drop=True)
    events = df.sort_values("start_time").reset_index(drop=True)
    boundaries = sorted(set(events["start_time"].tolist() + events["end_time"].tolist()))
    alloc = [0] * len(events)
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_dur = (seg_end - seg_start).total_seconds() / 3600.0
        active = events[
            (events["start_time"] <= seg_start) & (events["end_time"] >= seg_end)
        ].index.tolist()
        if active:
            share = seg_dur / len(active)
            for j in active:
                alloc[j] += share
    events["adjusted_duration"] = alloc
    events["duration"] = events["adjusted_duration"]
    return events


# --- Tests ---
def run_tests():
    print("Running tests...")
    # Test 1: Slash events with sequential times.
    from datetime import datetime

    test_event_slash = {
        "event_name": "cook/eat/nap",
        "calendar_name": "TestCalendar",
        "start_time": datetime(2023, 1, 1, 13, 0, 0),
        "end_time": datetime(2023, 1, 1, 14, 0, 0),
        "duration": 1.0,
        "metadata": {},
    }
    df_test_slash = pd.DataFrame([test_event_slash])
    df_slash_processed = process_slash_events(df_test_slash)
    # Expect three events.
    assert len(df_slash_processed) == 3, "Slash event splitting did not produce 3 events"
    # Sort by start time
    df_slash_processed = df_slash_processed.sort_values("start_time").reset_index(drop=True)
    expected_starts = [
        datetime(2023, 1, 1, 13, 0, 0),
        datetime(2023, 1, 1, 13, 20, 0),
        datetime(2023, 1, 1, 13, 40, 0),
    ]
    expected_ends = [
        datetime(2023, 1, 1, 13, 20, 0),
        datetime(2023, 1, 1, 13, 40, 0),
        datetime(2023, 1, 1, 14, 0, 0),
    ]
    for i, row in df_slash_processed.iterrows():
        assert row["event_name"] in ["cook", "eat", "nap"], "Unexpected event name after splitting"
        assert (
            row["start_time"] == expected_starts[i]
        ), f"Start time for {row['event_name']} incorrect"
        assert row["end_time"] == expected_ends[i], f"End time for {row['event_name']} incorrect"
        # Duration should be ~0.3333 hours (20 minutes)
        assert abs(row["duration"] - 0.3333) < 0.001, "Duration for sub-event not correct"

    # Test 2: Knee/gym event special case.
    test_event_knee = {
        "event_name": "knee/gym",
        "calendar_name": "TestCalendar",
        "start_time": datetime(2023, 1, 3, 18, 0, 0),
        "end_time": datetime(2023, 1, 3, 19, 0, 0),
        "duration": 1.0,
        "metadata": {},
    }
    df_test_knee = pd.DataFrame([test_event_knee])
    df_knee_processed = process_slash_events(df_test_knee)
    assert len(df_knee_processed) == 1, "Knee/gym event should remain a single event"
    row = df_knee_processed.iloc[0]
    assert row["event_name"].lower() == "gym", "Knee/gym event not converted to gym"
    assert (
        "original_name" in row["metadata"] and row["metadata"]["original_name"] == "knee/gym"
    ), "Metadata not set for knee/gym event"

    # Test 3: Book event processing.
    # this won't handle books with numbers in their title: eg `Kings 12` would be `k1`
    for event_name in ["Life and Fate", "laf"]:
        test_event_book = {
            "event_name": f"book: {event_name}",
            "calendar_name": "TestCalendar",
            "start_time": datetime(2023, 1, 2, 10, 0, 0),
            "end_time": datetime(2023, 1, 2, 11, 0, 0),
            "duration": 1.0,
            "metadata": {},
        }
        df_test_book = pd.DataFrame([test_event_book])
        df_book_processed = process_book_events(df_test_book)
        row = df_book_processed.iloc[0]
        assert row["event_name"] == "book:laf", (
            "Book event not processed correctly (incorrect abbreviation or extra space) "
            + row["event_name"]
        )
    print("All tests passed!")


def debug_sleep_patterns(df):
    """Debug missing sleep entries and bed/wakeup patterns"""
    # First, get all bed and wakeup entries
    bed_entries = df[df["event_name"].str.lower() == "bed"].copy()
    wakeup_entries = df[df["event_name"].str.lower().str.contains("woke up|wake up")].copy()
    sleep_entries = df[df["event_name"].str.lower() == "sleep"].copy()

    print(f"Total bed entries: {len(bed_entries)}")
    print(f"Total wakeup entries: {len(wakeup_entries)}")
    print(f"Total sleep entries: {len(sleep_entries)}")

    if bed_entries.empty:
        print("No bed entries found to analyze!")
        return

    # Sort by start time
    bed_entries = bed_entries.sort_values("start_time")
    wakeup_entries = wakeup_entries.sort_values("start_time")

    # 1. Find consecutive bed entries without wakeup in between
    consecutive_beds = []
    for i in range(len(bed_entries) - 1):
        current_bed = bed_entries.iloc[i]
        next_bed = bed_entries.iloc[i + 1]

        # Check if there's any wakeup between these beds
        wakeups_between = wakeup_entries[
            (wakeup_entries["start_time"] > current_bed["start_time"])
            & (wakeup_entries["start_time"] < next_bed["start_time"])
        ]

        if wakeups_between.empty:
            time_diff = (next_bed["start_time"] - current_bed["start_time"]).total_seconds() / 3600
            if time_diff > 20:  # Only count if beds are more than 20 hours apart
                consecutive_beds.append(
                    {
                        "first_bed_time": current_bed["start_time"],
                        "second_bed_time": next_bed["start_time"],
                        "hours_between": time_diff,
                    }
                )

    print(
        f"\nFound {len(consecutive_beds)} instances of consecutive bed entries without wakeup in"
        " between"
    )
    if consecutive_beds:
        consec_df = pd.DataFrame(consecutive_beds)
        print(consec_df["first_bed_time"].tail(20))
        print("Hours between consecutive beds:")
        print(f"  Mean: {consec_df['hours_between'].mean():.2f}")
        print(f"  Min: {consec_df['hours_between'].min():.2f}")
        print(f"  Max: {consec_df['hours_between'].max():.2f}")

        # Sample some consecutive beds
        print("\nSample consecutive bed entries:")
        for i, row in consec_df.tail(5).iterrows():
            print(
                f"  Bed at {row['first_bed_time']} → Bed at"
                f" {row['second_bed_time']} ({row['hours_between']:.2f} hours)"
            )

    # 2. Find bed entries without corresponding wakeup
    orphaned_beds = []
    for _, bed_row in bed_entries.iterrows():
        bed_time = bed_row["start_time"]

        # Look for a wakeup within 24 hours
        next_wakeup = wakeup_entries[
            (wakeup_entries["start_time"] > bed_time)
            & (wakeup_entries["start_time"] <= bed_time + timedelta(hours=24))
        ]

        if next_wakeup.empty:
            orphaned_beds.append({"bed_time": bed_time, "bed_date": bed_time.date()})

    print(f"\nFound {len(orphaned_beds)} bed entries without corresponding wakeup within 24 hours")
    if orphaned_beds:
        orphan_df = pd.DataFrame(orphaned_beds)
        # Count by month to see if there's a pattern
        orphan_df["month"] = orphan_df["bed_time"].dt.strftime("%Y-%m")
        monthly_counts = orphan_df["month"].value_counts().sort_index()
        print(orphan_df["bed_time"].tail(20))

        print("\nOrphaned beds by month:")
        for month, count in monthly_counts.items():
            print(f"  {month}: {count}")

    # 3. Check time distribution of bed and wakeup entries
    if not bed_entries.empty:
        bed_entries["hour"] = bed_entries["start_time"].dt.hour
        plt.figure(figsize=(10, 6))
        plt.hist(bed_entries["hour"], bins=24, alpha=0.7, label="Bed")
        if not wakeup_entries.empty:
            wakeup_entries["hour"] = wakeup_entries["start_time"].dt.hour
            plt.hist(wakeup_entries["hour"], bins=24, alpha=0.7, label="Wakeup")
        plt.title("Distribution of Bed and Wakeup Times by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Count")
        plt.legend()
        plt.xticks(range(0, 24))
        plt.savefig("bed_wakeup_hour_distribution.png")

    # 4. Check if there are days with bed entries but no sleep entries
    if not sleep_entries.empty:
        bed_dates = set(bed_entries["start_time"].dt.date)
        sleep_dates = set(sleep_entries["start_time"].dt.date)

        bed_no_sleep = bed_dates - sleep_dates
        print(f"\nDays with bed entries but no sleep entries: {len(bed_no_sleep)}")

        if bed_no_sleep:
            print("Sample dates:", sorted(list(bed_no_sleep))[:5])

    # 5. Analyze the gap between consecutive entries
    if len(bed_entries) > 1:
        bed_entries = bed_entries.sort_values("start_time")
        bed_entries["next_bed"] = bed_entries["start_time"].shift(-1)
        bed_entries["days_to_next"] = (
            bed_entries["next_bed"] - bed_entries["start_time"]
        ).dt.total_seconds() / (3600 * 24)

        # Find large gaps (more than 3 days)
        large_gaps = bed_entries[bed_entries["days_to_next"] > 3].copy()
        print(f"\nFound {len(large_gaps)} large gaps (>3 days) between consecutive bed entries")

        if not large_gaps.empty:
            print("\nSample large gaps:")
            for i, row in large_gaps.tail(5).iterrows():
                print(
                    f"  Gap from {row['start_time'].date()} to"
                    f" {row['next_bed'].date()} ({row['days_to_next']:.1f} days)"
                )

            # Plot gaps over time
            plt.figure(figsize=(12, 6))
            plt.plot(bed_entries["start_time"], bed_entries["days_to_next"])
            plt.title("Days Between Consecutive Bed Entries")
            plt.xlabel("Date")
            plt.ylabel("Days to Next Bed Entry")
            plt.axhline(y=1, color="r", linestyle="--", alpha=0.5)
            plt.savefig("bed_entry_gaps.png")

            # Also group by month
            bed_entries["month"] = bed_entries["start_time"].dt.strftime("%Y-%m")
            monthly_gap_avg = bed_entries.groupby("month")["days_to_next"].mean()

            print("\nAverage days between bed entries by month:")
            for month, avg_gap in monthly_gap_avg.items():
                print(f"  {month}: {avg_gap:.2f} days")

    return {"consecutive_beds": consecutive_beds, "orphaned_beds": orphaned_beds}


# === New Feature Functions ===


# 1. Weekly, Monthly, and Overall Time Summary by Category
def _utc_date(dt):
    """Return a timezone-aware datetime in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=pytz.UTC)
    return dt


def overall_time_summary(df, start_date, end_date):
    start_date = _utc_date(start_date)
    end_date = _utc_date(end_date)

    # Filter by dates.
    df_filtered = df[(df["start_time"] >= start_date) & (df["end_time"] <= end_date)].copy()

    # Optionally, filter out implausible book events.
    # For example, if a book event lasts more than 2 hours, it is likely an error.
    df_filtered = df_filtered[
        ~(
            (df_filtered["event_name"].str.lower().str.startswith("book:"))
            & (df_filtered["duration"] > 2)
        )
    ]

    # Group by event_name and compute total time and count of separate entries.
    summary = (
        df_filtered.groupby("event_name")
        .agg(total_time=("duration", "sum"), count=("duration", "count"))
        .reset_index()
    )

    # Top 20 by total time.
    summary_hours = summary.sort_values("total_time", ascending=False).head(20)
    # Top 20 by count of entries.
    summary_count = summary.sort_values("count", ascending=False).head(20)

    print("Overall Time Summary (Top 20 by Hours):")
    for _, row in summary_hours.iterrows():
        print(f"  {row['event_name']}: {row['total_time']:.2f} hours, {row['count']} entries")

    print("\nOverall Time Summary (Top 20 by Entry Count):")
    for _, row in summary_count.iterrows():
        print(f"  {row['event_name']}: {row['count']} entries, {row['total_time']:.2f} hours")

    return summary_hours, summary_count


def weekly_top_events(df, start_date, end_date, top_n=10):
    start_date = _utc_date(start_date)
    end_date = _utc_date(end_date)
    df_filtered = df[(df["start_time"] >= start_date) & (df["end_time"] <= end_date)].copy()
    # Create a week column (the starting date of the week)
    df_filtered["week"] = df_filtered["start_time"].dt.to_period("W").apply(lambda r: r.start_time)

    overall = (
        df_filtered.groupby(["week", "event_name"])
        .agg(total_time=("duration", "sum"), count=("duration", "count"))
        .reset_index()
    )
    overall_top = (
        overall.groupby("week")
        .apply(lambda g: g.sort_values("total_time", ascending=False).head(top_n))
        .reset_index(drop=True)
    )

    by_calendar = (
        df_filtered.groupby(["week", "calendar_name", "event_name"])
        .agg(total_time=("duration", "sum"), count=("duration", "count"))
        .reset_index()
    )
    calendar_top = (
        by_calendar.groupby(["week", "calendar_name"])
        .apply(lambda g: g.sort_values("total_time", ascending=False).head(top_n))
        .reset_index(drop=True)
    )

    print("Weekly Top Events Overall:")
    print(overall_top)
    print("\nWeekly Top Events by Calendar:")
    print(calendar_top)
    return overall_top, calendar_top


def monthly_top_events(df, start_date, end_date, top_n=10):
    start_date = _utc_date(start_date)
    end_date = _utc_date(end_date)
    df_filtered = df[(df["start_time"] >= start_date) & (df["end_time"] <= end_date)].copy()
    # Create a month column (the starting date of the month)
    df_filtered["month"] = df_filtered["start_time"].dt.to_period("M").apply(lambda r: r.start_time)

    overall = (
        df_filtered.groupby(["month", "event_name"])
        .agg(total_time=("duration", "sum"), count=("duration", "count"))
        .reset_index()
    )
    overall_top = (
        overall.groupby("month")
        .apply(lambda g: g.sort_values("total_time", ascending=False).head(top_n))
        .reset_index(drop=True)
    )

    by_calendar = (
        df_filtered.groupby(["month", "calendar_name", "event_name"])
        .agg(total_time=("duration", "sum"), count=("duration", "count"))
        .reset_index()
    )
    calendar_top = (
        by_calendar.groupby(["month", "calendar_name"])
        .apply(lambda g: g.sort_values("total_time", ascending=False).head(top_n))
        .reset_index(drop=True)
    )

    print("Monthly Top Events Overall:")
    print(overall_top)
    print("\nMonthly Top Events by Calendar:")
    print(calendar_top)
    return overall_top, calendar_top


def graph_waste_days(df):
    df = df.copy()
    # Define waste events as those whose event_name contains 'waste' (case-insensitive)
    df_waste = df[df["calendar_name"] == "Waste Time"].copy()
    df_waste["date"] = df_waste["start_time"].dt.date
    days = []
    for date_val, group in df_waste.groupby("date"):
        group = group.sort_values("start_time")
        current_start = None
        current_end = None
        max_block = 0
        for _, row in group.iterrows():
            if current_start is None:
                current_start = row["start_time"]
                current_end = row["end_time"]
            else:
                gap = (row["start_time"] - current_end).total_seconds() / 60.0
                if gap <= 30:
                    current_end = max(current_end, row["end_time"])
                else:
                    block_duration = (current_end - current_start).total_seconds() / 3600.0
                    max_block = max(max_block, block_duration)
                    current_start = row["start_time"]
                    current_end = row["end_time"]
        if current_start is not None:
            block_duration = (current_end - current_start).total_seconds() / 3600.0
            max_block = max(max_block, block_duration)
        days.append({"date": date_val, "max_waste_block": max_block})
    df_days = pd.DataFrame(days)
    df_days["waste_flag"] = df_days["max_waste_block"] >= 4
    plt.figure(figsize=(10, 5))
    plt.bar(df_days["date"], df_days["waste_flag"].astype(int))
    plt.title("Days with ≥4 Hours Contiguous Waste")
    plt.xlabel("Date")
    plt.ylabel("1 if wasted ≥4 hours continuously")
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def graph_gym_count(df, average_for=1):
    # Filter for gym events.
    df_gym = df[df["event_name"].str.lower().str.contains("gym")].copy()
    df_gym["date"] = pd.to_datetime(df_gym["start_time"].dt.date)
    # Group by date: count events and sum durations (assumed in hours)
    daily_agg = df_gym.groupby("date").agg(count=("date", "size"), hours=("duration", "sum"))
    # Create a full date range.
    full_index = pd.date_range(start=daily_agg.index.min(), end=daily_agg.index.max(), freq="D")
    daily_agg = daily_agg.reindex(full_index, fill_value=0)
    daily_agg.index.name = "date"
    daily_agg = daily_agg.reset_index()

    if average_for > 1:
        daily_agg = daily_agg.set_index("date").resample(f"{average_for}D").mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.scatter(daily_agg["date"], daily_agg["count"])
    plt.title(f"Daily 'Gym' Entry Count (averaged over {average_for} days)")
    plt.xlabel("Date")
    plt.ylabel("Average Count")
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return daily_agg


def graph_drink_count(df):
    df = df.copy()
    mask = df["event_name"].str.lower().str.contains("drink")
    df_drink = df[mask]
    df_drink["date"] = df_drink["start_time"].dt.date
    daily_count = df_drink.groupby("date").size().reset_index(name="count")
    plt.figure(figsize=(10, 5))
    plt.bar(daily_count["date"], daily_count["count"] > 0)
    plt.title("Daily 'Drink' Entry Count")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.ylim(bottom=0)
    plt.show()


def scatter_calendar_time(df, average_for=1):
    df = df.copy()
    df["date"] = pd.to_datetime(df["start_time"].dt.date)
    calendars = df["calendar_name"].unique()
    results = {}
    for cal in calendars:
        df_cal = df[df["calendar_name"] == cal].copy()
        daily = df_cal.groupby("date")["duration"].sum()
        full_index = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
        daily = daily.reindex(full_index, fill_value=0)
        daily.index.name = "date"
        daily = daily.to_frame(name="total_time")
        if average_for > 1:
            daily = daily.resample(f"{average_for}D").mean()
        daily = daily.reset_index().rename(columns={"index": "date"})
        plt.figure(figsize=(10, 5))
        plt.scatter(daily["date"], daily["total_time"])
        plt.title(f"Daily Time Spent in {cal} (averaged over {average_for} days)")
        plt.xlabel("Date")
        plt.ylabel("Total Time (hours)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        results[cal] = daily
    return results


def graph_sleep_nap(df, average_for=1):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df.copy()
    # Ensure date is a proper datetime (date only)
    df["date"] = pd.to_datetime(df["start_time"].dt.date)

    # Separate sleep and nap events.
    df_sleep = df[df["event_name"].str.lower() == "sleep"].copy()
    df_nap = df[df["event_name"].str.lower() == "nap"].copy()

    # Sum durations per day.
    sleep_daily = df_sleep.groupby("date")["duration"].sum()
    nap_daily = df_nap.groupby("date")["duration"].sum()

    # Create a full daily index spanning the data range.
    start_date = min(
        sleep_daily.index.min() if not sleep_daily.empty else df["date"].min(),
        nap_daily.index.min() if not nap_daily.empty else df["date"].min(),
    )
    end_date = max(
        sleep_daily.index.max() if not sleep_daily.empty else df["date"].max(),
        nap_daily.index.max() if not nap_daily.empty else df["date"].max(),
    )
    full_index = pd.date_range(start=start_date, end=end_date, freq="D")

    # Reindex to include missing days (filling with zeros).
    sleep_daily = sleep_daily.reindex(full_index, fill_value=0)
    nap_daily = nap_daily.reindex(full_index, fill_value=0)

    # Combine into one DataFrame.
    daily = pd.DataFrame({"sleep_time": sleep_daily, "nap_time": nap_daily}, index=full_index)

    # Compute averaged data if needed.
    if average_for > 1:
        daily_avg = daily.resample(f"{average_for}D").mean()
        # Reset index so that 'date' is a column.
        daily_avg = daily_avg.reset_index().rename(columns={"index": "date"})
    else:
        daily_avg = daily.reset_index().rename(columns={"index": "date"})

    plt.figure(figsize=(10, 5))

    if average_for > 1:
        # Plot stacked bars using the averaged data.
        bar_width = average_for  # width in days for the bar (no gap between periods)
        plt.bar(
            daily_avg["date"],
            daily_avg["sleep_time"],
            width=bar_width,
            align="edge",
            label="Sleep (avg)",
        )
        plt.bar(
            daily_avg["date"],
            daily_avg["nap_time"],
            bottom=daily_avg["sleep_time"],
            width=bar_width,
            align="edge",
            label="Nap (avg)",
        )
        # Plot original daily datapoints in the background as small, translucent markers.
        plt.scatter(
            daily.index, daily["sleep_time"], s=3, color="blue", alpha=0.2, label="Sleep (daily)"
        )
        plt.scatter(
            daily.index, daily["nap_time"], s=3, color="orange", alpha=0.2, label="Nap (daily)"
        )
    else:
        # For daily plotting (no averaging), plot just the stacked bars.
        bar_width = 1
        plt.bar(
            daily_avg["date"], daily_avg["sleep_time"], width=bar_width, align="edge", label="Sleep"
        )
        plt.bar(
            daily_avg["date"],
            daily_avg["nap_time"],
            bottom=daily_avg["sleep_time"],
            width=bar_width,
            align="edge",
            label="Nap",
        )

    plt.title(
        "Daily Sleep & Nap Durations (averaged over"
        f" {average_for} day{'s' if average_for>1 else ''})"
    )
    plt.xlabel("Date")
    plt.ylabel("Hours")
    plt.ylim((4, 9.5))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    daily_avg["total_rest"] = daily_avg["sleep_time"] + daily_avg["nap_time"]
    return daily_avg


####                               #### Graph of all time combined
def bar_calendar_time(df, average_for=1):
    """
    Produces a stacked bar plot of total time per calendar with a continuous, evenly spaced x-axis.
    Bars are aggregated by day (or averaged over nonoverlapping groups of days if average_for > 1)
    and are plotted with no gaps. Uses a fixed color mapping for specific calendar names:
      - "Things": Grey
      - "Meals, Supplements, Sleep": Green
      - "Waste Time": Red
      - "clark.benham@gmail.com": Blue
      - "cb5ye@virginia.edu": Blue
    """
    # Fixed color mapping.
    # Ensure each row has a proper 'date'
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["start_time"].dt.date)

    # Group by date and calendar_name: sum durations (in hours)
    grouped = df2.groupby(["date", "calendar_name"])["duration"].sum().unstack(fill_value=0)

    # Reindex to a full daily date range (so days with no events appear as zeros)
    full_index = pd.date_range(start=grouped.index.min(), end=grouped.index.max(), freq="D")
    grouped = grouped.reindex(full_index, fill_value=0)
    grouped.index.name = "date"

    # Average over nonoverlapping windows if requested.
    if average_for > 1:
        grouped = grouped.resample(f"{average_for}D").mean()

    # Convert the index (dates) to matplotlib numeric dates.
    dates = mdates.date2num(grouped.index.to_pydatetime())

    # Prepare for a stacked bar plot.
    categories = grouped.columns
    bottom = np.zeros(len(grouped))

    plt.figure(figsize=(10, 5))
    # Set bar width equal to the averaging period (in days) to ensure no gaps.
    width = average_for

    for cat in categories:
        values = grouped[cat].values
        # Get color from mapping; if not found, default to grey.
        col = color_map.get(cat, "grey")
        plt.bar(dates, values, bottom=bottom, width=width, align="edge", color=col, label=cat)
        bottom += values

    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.title(
        f"Total Time per Calendar (averaged over {average_for} day{'s' if average_for>1 else ''})"
    )
    plt.xlabel("Date")
    plt.ylabel("Average Time (hours)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return grouped


def graph_activity_breakdown(df, average_for=1):
    """
    Produces a stacked bar plot showing, for each day (or an average over nonoverlapping windows of days),
    how much time is spent on each activity (by event_name). Any remaining time out of 24 hours is labeled
    as 'uncategorized'. Event names are truncated to 10 characters, ensuring uniqueness.

    Colors are assigned based on calendar_name.
    Uses a fixed color mapping:
      - "Things"                   -> Grey
      - "Meals, Supplements, Sleep"-> Green
      - "Waste Time"               -> Red
      - "clark.benham@gmail.com"   -> Blue
      - "cb5ye@virginia.edu"       -> Blue

    Bars are sorted by duration and plotted with no gaps. Text is added directly on each
    segment if the average is >=0.5 hours (30 minutes).
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Fixed color mapping for calendar names
    color_map = {
        "Things": "grey",
        "Meals, Supplements, Sleep": "green",
        "Waste Time": "red",
        "clark.benham@gmail.com": "blue",
        "cb5ye@virginia.edu": "blue",
    }

    # Prepare data for truncation and lowercase conversion
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["start_time"].dt.date)

    # Create a combined column for grouping - convert event_name to lowercase
    df2["event_cal"] = df2["event_name"].str.lower() + " (" + df2["calendar_name"] + ")"

    # Group by date and combined event_cal: sum durations (in hours)
    grouped = df2.groupby(["date", "event_cal"])["duration"].sum().unstack(fill_value=0)

    # Add 'uncategorized' time: remaining hours in the day
    grouped["uncategorized"] = (24 - grouped.sum(axis=1)).clip(lower=0)

    # Reindex over the full daily date range
    full_index = pd.date_range(start=grouped.index.min(), end=grouped.index.max(), freq="D")
    grouped = grouped.reindex(full_index, fill_value=0)
    grouped.index.name = "date"

    # Average over nonoverlapping windows if needed
    if average_for > 1:
        grouped = grouped.resample(f"{average_for}D").mean()

    daily = grouped.reset_index()

    # Create mapping between event_cal columns and their calendar names
    event_cal_to_calendar = {}
    for event_cal in [col for col in daily.columns if col != "date" and col != "uncategorized"]:
        # Extract calendar name from the combined string (format: "event_name (calendar_name)")
        try:
            calendar_name = event_cal.split("(")[1].rstrip(")")
            event_cal_to_calendar[event_cal] = calendar_name
        except:
            # Fallback for any columns that don't match the expected format
            event_cal_to_calendar[event_cal] = "unknown"

    # Set calendar for uncategorized
    event_cal_to_calendar["uncategorized"] = "uncategorized"

    # Create unique truncated column names for all columns except 'date'
    seen = {}

    def unique_truncate(col, max_len=15):
        base = col[:max_len]
        if base in seen:
            seen[base] += 1
            return f"{base}_{seen[base]}"
        else:
            seen[base] = 1
            return base

    # Build mapping from original column to unique truncated name
    trunc_map = {}
    for col in daily.columns:
        if col == "date":
            trunc_map[col] = col
        else:
            # Get the event name part without the calendar part
            if col != "uncategorized":
                try:
                    event_name = col.split(" (")[0]
                    trunc = event_name[:15]  # Truncate to 15 chars
                except:
                    trunc = col[:10]
            else:
                trunc = col

            # Ensure uniqueness
            trunc = unique_truncate(trunc)
            trunc_map[col] = trunc

    # Rename columns
    daily_trunc = daily.rename(columns=trunc_map)

    # Create a reverse mapping from truncated name back to original
    # In case of multiple originals mapping to the same truncated name,
    # we'll just take the first one for color assignment purposes
    trunc_to_orig = {}
    for orig, trunc in trunc_map.items():
        if trunc not in trunc_to_orig and orig != "date":
            trunc_to_orig[trunc] = orig

    # Calculate total duration for each event across all days (for sorting)
    event_totals = daily.drop(columns=["date"]).sum().sort_values(ascending=False)

    # Sort columns by total duration (excluding date and keeping uncategorized at the end)
    sorted_events = event_totals.index.tolist()
    if "uncategorized" in sorted_events:
        sorted_events.remove("uncategorized")
        sorted_events.append("uncategorized")

    # Get truncated column names in sorted order
    sorted_trunc_events = [trunc_map[ev] for ev in sorted_events if ev in trunc_map]

    # Build a mapping for colors based on calendar name
    trunc_color = {}
    for trunc, orig in trunc_to_orig.items():
        if orig == "date":
            continue

        calendar = event_cal_to_calendar.get(orig, "unknown")

        # Assign color based on calendar name
        assigned_color = "grey"  # Default color
        for key, color in color_map.items():
            if key.lower() in calendar.lower():
                assigned_color = color
                break

        trunc_color[trunc] = assigned_color

    # Special color for uncategorized
    if "uncategorized" in trunc_map.values():
        trunc_color["uncategorized"] = "lightgrey"

    # Plot manually using plt.bar to control spacing
    # Dynamic figure width based on number of rows: 2 + 3 * number_of_rows
    num_rows = len(daily_trunc)
    fig_width = 2 + 3 * num_rows
    fig, ax = plt.subplots(figsize=(fig_width, 15))

    # Fix the x-axis positioning to ensure even spacing
    x_positions = np.arange(len(daily_trunc))
    bar_width = 0.9  # Width to accommodate 15 characters

    # For each day/time chunk, sort the events by their duration within that specific chunk
    for i, (_, row) in enumerate(daily_trunc.iterrows()):
        day_data = row.drop("date").to_dict()
        # Sort events by duration for this specific day/chunk
        day_sorted_events = sorted(
            [(event, duration) for event, duration in day_data.items()],
            key=lambda x: x[1],
            reverse=False,  # Sort ascending so largest are on top of stack
        )

        bottom = 0
        for event, height in day_sorted_events:
            if height > 0:  # Only plot if there's any duration
                color = trunc_color.get(event, "grey")
                bar = ax.bar(
                    x_positions[i],
                    height,
                    bottom=bottom,
                    width=bar_width,
                    align="center",
                    color=color,
                )

                # Add text label if duration is >= 0.1667 hours (10 minutes)
                if height >= 0.1667:
                    ax.text(
                        x_positions[i],
                        bottom + height / 2,
                        event,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

                bottom += height

    # Set x-axis labels as the dates with even spacing
    ax.set_xticks(x_positions)
    ax.set_xticklabels(daily_trunc["date"].dt.strftime("%Y-%m-%d"), rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Time (hours)")
    ax.set_title(
        f"Daily Activity Breakdown (averaged over {average_for} day{'s' if average_for>1 else ''})"
    )

    # Create legend: group by calendar name and only include those with events >= 0.5 hours
    legend_items = {}

    # Calculate average duration by event
    event_averages = daily_trunc.drop(columns=["date"]).mean()

    # Group legend items by calendar and only include those with sufficient average time
    for trunc, orig in trunc_to_orig.items():
        if trunc == "date":
            continue

        # Skip events with average less than 10 minutes
        if event_averages.get(trunc, 0) < 0.1667:
            continue

        calendar = event_cal_to_calendar.get(orig, "unknown")
        color = trunc_color.get(trunc, "grey")

        if calendar not in legend_items:
            legend_items[calendar] = (color, [])

        legend_items[calendar][1].append(trunc)

    # Build the legend
    legend_handles = []
    legend_labels = []

    for calendar, (color, events) in sorted(legend_items.items()):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
        if len(events) <= 3:
            # For few events, list them all
            legend_labels.append(f"{calendar}: {', '.join(events)}")
        else:
            # For many events, just show calendar name
            legend_labels.append(calendar)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right")

    plt.tight_layout()
    plt.show()

    return daily_trunc


filtered_df = df
graph_activity_breakdown(filtered_df, average_for=28)
# %%
if __name__ == "__main__":
    # Run tests.
    # run_tests()

    # Example usage (comment out tests if processing real calendar data)
    calendar_dir = "data/Calendar Takeout/Calendar/"
    start_date = datetime(2021, 5, 24)
    end_date = datetime.combine(datetime.today(), datetime.max.time())
    df = parse_ics_files(calendar_dir, start_date, end_date)
    _df = df.copy()
    r = debug_sleep_patterns(df)

    df = process_sleep_events(df)
    df = process_slash_events(df)
    df = process_book_events(df)
    df = process_overlaps(df)
    # # count work as productive time
    df.loc[df["event_name"].str.lower() == "job: hive", "calendar_name"] = "clark.benham@gmail.com"
    # # reminders don't count

    # Print overall time summary.
    overall_time_summary(df, start_date, end_date)
    if False:  # graph of overall, just look at tabl instead
        agg = bar_calendar_time(df, average_for=7)
        plt.scatter(agg.index, agg["Waste Time"], color="red")
        plt.title("avg Waste Time/day over Week")
        plt.show()

        # start_date = pd.to_datetime("2025-01-01", utc=True)
        # filtered_df = df.query("start_time >= @start_date")
        filtered_df = df
        graph_activity_breakdown(filtered_df, average_for=28)

    # %%
    # 2. Top-N events per week and month.
    weekly_top_events(df, start_date, end_date, top_n=10)
    monthly_top_events(df, start_date, end_date, top_n=10)

    # Graphs
    scatter_calendar_time(df, average_for=14)
    graph_drink_count(df)
    graph_waste_days(df)

    graph_sleep_nap(df, average_for=7)
    graph_sleep_nap(df, average_for=28)
    graph_sleep_nap(df, average_for=120)

    graph_gym_count(df, average_for=7)
    graph_gym_count(df, average_for=28)
    graph_gym_count(df, average_for=100)

    # Most and least ever worked out and slept in a month
    # For Gym – using graph_gym_count
    gym_data = graph_gym_count(df, average_for=28)
    most_worked_out = gym_data.loc[gym_data["count"].idxmax()]
    least_worked_out = gym_data.loc[gym_data["count"].idxmin()]

    print("Most worked out month (28-day average):")
    print(most_worked_out)
    print("Least worked out month (28-day average):")
    print(least_worked_out)

    # For Sleep – using graph_sleep_nap
    sleep_data = graph_sleep_nap(df, average_for=28)
    most_slept = sleep_data.loc[sleep_data["total_rest"].idxmax()]
    least_slept = sleep_data.loc[sleep_data["total_rest"].idxmin()]

    print("Most slept month (28-day average) – based on total rest:")
    print(most_slept)
    print("Least slept month (28-day average) – based on total rest:")
    print(least_slept)


##############          Extract data from work tracking csv
def load_work_summary(csv_file):
    """
    Loads the work summary CSV (which has extra header rows) and extracts only the
    columns 'Start Date', 'Value', and 'Hours Working'. It converts the date and numeric
    columns appropriately, and computes the work productivity as Value * Hours Working.
    """
    # Read CSV while skipping the second row (which contains extra header info).
    # Adjust skiprows if your file structure differs.
    df = pd.read_csv(csv_file, skiprows=[1])

    # Select the needed columns.
    # These must match the CSV column names exactly.
    df = df[["Start Date", "Value", "Hours Working"]].copy()

    # Convert Start Date to datetime.
    df["date"] = pd.to_datetime(df["Start Date"], errors="coerce")

    # Convert Value and Hours Working to numeric.
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Hours Working"] = pd.to_numeric(df["Hours Working"], errors="coerce")

    # Compute the productivity metric.
    df["work_productivity"] = df["Value"] * df["Hours Working"]

    # Drop rows with missing date or productivity.
    df = df.dropna(subset=["date", "work_productivity"])
    return df


# %%
# Analysis of sleep: does napping reduce? How does impact hours worked?


def scatter_total_rest_vs_nap(sleep_data):
    """
    Given sleep_data DataFrame (with columns: date, sleep_time, nap_time),
    compute total rest = sleep_time + nap_time, then scatter-plot total rest (y)
    versus nap_time (x). A linear regression line is added along with its slope,
    intercept, and R² value.
    """
    # Compute total rest.
    sleep_data = sleep_data.copy()
    sleep_data["total_rest"] = sleep_data["sleep_time"] + sleep_data["nap_time"]

    # Define x and y for regression.
    x = sleep_data["nap_time"].values
    y = sleep_data["total_rest"].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, label="Data points")

    # Compute linear regression (degree 1 polynomial).
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    # Generate regression line using the sorted x values.
    x_sorted = np.sort(x)
    plt.plot(
        x_sorted, poly1d_fn(x_sorted), color="red", label=f"Fit: y={coef[0]:.2f}x+{coef[1]:.2f}"
    )

    # Compute R^2.
    y_pred = poly1d_fn(x)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    plt.text(
        0.05,
        0.95,
        f"R² = {r2:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.xlabel("Nap Time (hours)")
    plt.ylabel("Total Rest Time (hours)")
    plt.title("Scatter: Total Rest vs. Nap Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return sleep_data


def scatter_total_rest_vs_clark(df, sleep_data):
    """
    Given the overall DataFrame (df) and sleep_data (with columns: date, sleep_time, nap_time),
    this function computes total rest as sleep_time + nap_time, then extracts the total duration
    per day for events from calendar 'clark.benham@gmail.com'. It merges the two by date and plots
    total rest (x) versus total clark event duration (y), along with a best-fit line and R².
    """
    # Compute total rest in sleep_data.
    sleep_data = sleep_data.copy()
    sleep_data["total_rest"] = sleep_data["sleep_time"] + sleep_data["nap_time"]

    # Get clark events.
    clark_df = df[df["calendar_name"] == "clark.benham@gmail.com"].copy()
    clark_df["date"] = pd.to_datetime(clark_df["start_time"].dt.date)
    # Group by date: sum durations.
    clark_daily = clark_df.groupby("date")["duration"].sum().reset_index()

    # Merge sleep_data and clark_daily on date.
    merged = pd.merge(sleep_data, clark_daily, on="date", how="inner")
    # Rename for clarity.
    merged = merged.rename(columns={"duration": "clark_duration"})

    # Define x and y for regression.
    x = merged["total_rest"].values
    y = merged["clark_duration"].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, label="Data points")

    # Compute regression.
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    x_sorted = np.sort(x)
    plt.plot(
        x_sorted, poly1d_fn(x_sorted), color="red", label=f"Fit: y={coef[0]:.2f}x+{coef[1]:.2f}"
    )

    # Compute R².
    y_pred = poly1d_fn(x)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    plt.text(
        0.05,
        0.95,
        f"R² = {r2:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.xlabel("Total Rest Time (hours)")
    plt.ylabel("Total 'clark.benham@gmail.com' Duration (hours)")
    plt.title("Scatter: Total Rest vs. 'clark.benham@gmail.com' Events")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return merged


def scatter_sleep_vs_work_aggregated(sleep_data, work_data):
    """
    Aggregates work_data over periods defined by sleep_data dates.

    For each sleep_data entry, assume its period is from that date (inclusive)
    to the next sleep_data date. For the last sleep_data entry, use the period
    from that date to work_data['date'].max(), but assert the gap isn't longer
    than the maximum of the previous gaps.

    Then, compute for each period:
      - total_rest = sleep_time + nap_time (from sleep_data at the period start)
      - average work_productivity and average Hours Working (from work_data)

    Finally, plot two scatter plots:
      1. total_rest (x) vs. avg work_productivity (y)
      2. total_rest (x) vs. avg Hours Working (y)

    In both, the aggregated points are color-coded from red (oldest) to blue (newest)
    and a linear best-fit line is added (with slope, intercept, and R² shown).

    Returns a DataFrame with one row per aggregated period.
    """
    # Ensure dates are datetime and sort
    sleep_data = sleep_data.copy()
    sleep_data["date"] = pd.to_datetime(sleep_data["date"])
    sleep_data = sleep_data.sort_values("date").reset_index(drop=True)

    work_data = work_data.copy()
    work_data["date"] = pd.to_datetime(work_data["date"])
    work_data = work_data.sort_values("date").reset_index(drop=True)

    # Compute total_rest in sleep_data.
    sleep_data["total_rest"] = sleep_data["sleep_time"] + sleep_data["nap_time"]

    # Build intervals from sleep_data dates.
    sleep_dates = sleep_data["date"].tolist()
    intervals = []
    for i in range(len(sleep_dates) - 1):
        intervals.append((sleep_dates[i], sleep_dates[i + 1]))
    # For the last interval: from last sleep date to max date in work_data.
    last_interval = (sleep_dates[-1], work_data["date"].max())
    # Check gap lengths.
    previous_gaps = [
        (sleep_dates[i + 1] - sleep_dates[i]).days for i in range(len(sleep_dates) - 1)
    ]
    max_gap = max(previous_gaps) if previous_gaps else 0
    last_gap = (last_interval[1] - last_interval[0]).days
    if max_gap > 0 and last_gap > max_gap:
        raise ValueError(
            f"Last interval gap ({last_gap} days) exceeds maximum previous gap ({max_gap} days)."
        )
    intervals.append(last_interval)

    # For each interval, compute average work_productivity and average Hours Working.
    agg_list = []
    for start, end in intervals:
        mask = (work_data["date"] >= start) & (work_data["date"] < end)
        sub = work_data.loc[mask]
        if not sub.empty:
            avg_prod = sub["work_productivity"].mean()
            avg_hours = sub["Hours Working"].mean()  # assumes work_data has this column
        else:
            avg_prod = np.nan
            avg_hours = np.nan
        # Get sleep metrics from the sleep_data row at the start date.
        sleep_row = sleep_data[sleep_data["date"] == start].iloc[0]
        agg_list.append(
            {
                "date": start,
                "total_rest": sleep_row["total_rest"],
                "sleep_time": sleep_row["sleep_time"],
                "nap_time": sleep_row["nap_time"],
                "avg_work_productivity": avg_prod,
                "avg_hours_working": avg_hours,
            }
        )
    agg_df = pd.DataFrame(agg_list).dropna()

    # Build a colormap: map date timestamps to a value from 0 (old) to 1 (new)
    norm = plt.Normalize(agg_df["date"].min().timestamp(), agg_df["date"].max().timestamp())
    cmap = plt.get_cmap("RdBu")  # red (old) to blue (new)
    colors = [cmap(norm(d.timestamp())) for d in agg_df["date"]]

    # Create subplots: one for productivity and one for hours working.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: total_rest vs. avg_work_productivity
    x = agg_df["total_rest"].values
    y1 = agg_df["avg_work_productivity"].values
    axs[0].scatter(x, y1, color=colors, s=80, alpha=0.8, label="Data")
    # Linear regression.
    coef1 = np.polyfit(x, y1, 1)
    poly1 = np.poly1d(coef1)
    x_sorted = np.sort(x)
    axs[0].plot(
        x_sorted,
        poly1(x_sorted),
        color="black",
        lw=2,
        label=f"Fit: y={coef1[0]:.2f}x+{coef1[1]:.2f}",
    )
    y1_pred = poly1(x)
    r2_1 = 1 - np.sum((y1 - y1_pred) ** 2) / np.sum((y1 - np.mean(y1)) ** 2)
    axs[0].text(
        0.05,
        0.95,
        f"R² = {r2_1:.2f}",
        transform=axs[0].transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    axs[0].set_xlabel("Total Rest (hours)")
    axs[0].set_ylabel("Avg Work Productivity")
    axs[0].set_title("Sleep vs. Work Productivity")
    axs[0].legend()

    # Plot 2: total_rest vs. avg_hours_working
    y2 = agg_df["avg_hours_working"].values
    axs[1].scatter(x, y2, color=colors, s=80, alpha=0.8, label="Data")
    coef2 = np.polyfit(x, y2, 1)
    poly2 = np.poly1d(coef2)
    axs[1].plot(
        x_sorted,
        poly2(x_sorted),
        color="black",
        lw=2,
        label=f"Fit: y={coef2[0]:.2f}x+{coef2[1]:.2f}",
    )
    y2_pred = poly2(x)
    r2_2 = 1 - np.sum((y2 - y2_pred) ** 2) / np.sum((y2 - np.mean(y2)) ** 2)
    axs[1].text(
        0.05,
        0.95,
        f"R² = {r2_2:.2f}",
        transform=axs[1].transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    axs[1].set_xlabel("Total Rest (hours)")
    axs[1].set_ylabel("Avg Hours Working")
    axs[1].set_title("Sleep vs. Hours Working")
    axs[1].legend()

    # Create a colorbar showing the date scale.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("Date (red=old, blue=new)")

    plt.tight_layout()
    plt.show()

    return agg_df


if __name__ == "__main__":
    sleep_data = graph_sleep_nap(df, average_for=3)
    scatter_total_rest_vs_nap(sleep_data)
    scatter_total_rest_vs_clark(df, sleep_data)

    work_data = load_work_summary("data/Work Summary  - Daily Summary.csv")
    merged_data = scatter_sleep_vs_work_aggregated(sleep_data, work_data)

    # Only from tue or wed start of aggr
    weekday_start = merged_data[merged_data["date"].dt.dayofweek.isin([1, 2])]
    for c1 in ["avg_work_productivity", "avg_hours_working"]:
        for c2 in ["total_rest", "sleep_time"]:
            plt.scatter(weekday_start[c2], weekday_start[c1])
            plt.title(f"{c1}- {c2}")
            plt.show()


# %% Impact of workouts on job performance
def scatter_workout_vs_work_aggregated(workout_data, work_data):
    """
    Aggregates workout_data over periods defined by its 'date' entries.
    For each period (from one workout date to the next), it computes:
      - Average workout_count and workout_hours from workout_data
      - Average work_productivity and average Hours Working from work_data
    The last period runs from the last workout date to work_data['date'].max() – an assertion
    is raised if that final gap is longer than any previous gap.

    Then, two scatter plots are produced:
      (1) Average workout count vs. average work productivity.
      (2) Average workout hours vs. average Hours Working.

    Data points are color-coded from red (old) to blue (new) according to their start date.
    Regression lines (with slope, intercept, and R²) are added to each plot.

    Returns:
        A DataFrame with one row per aggregated interval.
    """
    # Ensure the date columns are datetime and sort both DataFrames.
    workout_data = workout_data.copy()
    workout_data["date"] = pd.to_datetime(workout_data["date"])
    workout_data = workout_data.sort_values("date").reset_index(drop=True)

    work_data = work_data.copy()
    work_data["date"] = pd.to_datetime(work_data["date"])
    work_data = work_data.sort_values("date").reset_index(drop=True)

    # Build intervals using workout_data's dates.
    workout_dates = workout_data["date"].tolist()
    intervals = []
    for i in range(len(workout_dates) - 1):
        intervals.append((workout_dates[i], workout_dates[i + 1]))
    # For the final interval: from last workout date to work_data's max date.
    last_interval = (workout_dates[-1], work_data["date"].max())
    previous_gaps = [
        (workout_dates[i + 1] - workout_dates[i]).days for i in range(len(workout_dates) - 1)
    ]
    max_gap = max(previous_gaps) if previous_gaps else 0
    last_gap = (last_interval[1] - last_interval[0]).days
    if max_gap > 0 and last_gap > max_gap:
        raise ValueError(
            f"Last interval gap ({last_gap} days) exceeds maximum previous gap ({max_gap} days)."
        )
    intervals.append(last_interval)

    # Aggregate data for each interval.
    agg_list = []
    for start, end in intervals:
        # Aggregate workout_data for this period.
        mask_w = (workout_data["date"] >= start) & (workout_data["date"] < end)
        sub_w = workout_data.loc[mask_w]
        if not sub_w.empty:
            avg_workout_count = sub_w["count"].mean()
            avg_workout_hours = sub_w["hours"].mean()
        else:
            avg_workout_count = np.nan
            avg_workout_hours = np.nan

        # Aggregate work_data for this period.
        mask_work = (work_data["date"] >= start) & (work_data["date"] < end)
        sub_work = work_data.loc[mask_work]
        if not sub_work.empty:
            avg_work_productivity = sub_work["work_productivity"].mean()
            avg_hours_working = sub_work["Hours Working"].mean()
        else:
            avg_work_productivity = np.nan
            avg_hours_working = np.nan

        agg_list.append(
            {
                "date": start,
                "avg_workout_count": avg_workout_count,
                "avg_workout_hours": avg_workout_hours,
                "avg_work_productivity": avg_work_productivity,
                "avg_hours_working": avg_hours_working,
            }
        )
    agg_df = pd.DataFrame(agg_list).dropna()

    # Color coding: map each aggregated date to a color (red=old, blue=new).
    norm = plt.Normalize(agg_df["date"].min().timestamp(), agg_df["date"].max().timestamp())
    cmap = plt.get_cmap("RdBu")
    colors = [cmap(norm(d.timestamp())) for d in agg_df["date"]]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: avg_workout_count vs. avg_work_productivity.
    x1 = agg_df["avg_workout_count"].values
    y1 = agg_df["avg_work_productivity"].values
    axs[0].scatter(x1, y1, color=colors, s=80, alpha=0.8, label="Data")
    coef1 = np.polyfit(x1, y1, 1)
    poly1 = np.poly1d(coef1)
    x1_sorted = np.sort(x1)
    axs[0].plot(
        x1_sorted,
        poly1(x1_sorted),
        color="black",
        lw=2,
        label=f"Fit: y={coef1[0]:.2f}x+{coef1[1]:.2f}",
    )
    y1_pred = poly1(x1)
    r2_1 = 1 - np.sum((y1 - y1_pred) ** 2) / np.sum((y1 - np.mean(y1)) ** 2)
    axs[0].text(
        0.05,
        0.95,
        f"R² = {r2_1:.2f}",
        transform=axs[0].transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    axs[0].set_xlabel("Avg Workout Count")
    axs[0].set_ylabel("Avg Work Productivity")
    axs[0].set_title("Workout Count vs. Work Productivity")
    axs[0].legend()

    # Plot 2: avg_workout_hours vs. avg_hours_working.
    x2 = agg_df["avg_workout_hours"].values
    y2 = agg_df["avg_hours_working"].values
    axs[1].scatter(x2, y2, color=colors, s=80, alpha=0.8, label="Data")
    coef2 = np.polyfit(x2, y2, 1)
    poly2 = np.poly1d(coef2)
    x2_sorted = np.sort(x2)
    axs[1].plot(
        x2_sorted,
        poly2(x2_sorted),
        color="black",
        lw=2,
        label=f"Fit: y={coef2[0]:.2f}x+{coef2[1]:.2f}",
    )
    y2_pred = poly2(x2)
    r2_2 = 1 - np.sum((y2 - y2_pred) ** 2) / np.sum((y2 - np.mean(y2)) ** 2)
    axs[1].text(
        0.05,
        0.95,
        f"R² = {r2_2:.2f}",
        transform=axs[1].transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    axs[1].set_xlabel("Avg Workout Hours")
    axs[1].set_ylabel("Avg Hours Working")
    axs[1].set_title("Workout Hours vs. Hours Working")
    axs[1].legend()

    # Add a colorbar for the date scale.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("Date (red=old, blue=new)")

    plt.tight_layout()
    plt.show()

    return agg_df


if __name__ == "__main__":
    # not "correct" way to aggr by weekday, but close enough
    gym_data = graph_gym_count(df, average_for=2)
    agg_workout = scatter_workout_vs_work_aggregated(gym_data, work_data)

    # weekday_start = agg_workout[agg_workout["date"].dt.dayofweek.isin([1, 2])]
    weekday_start = agg_workout[agg_workout["date"].dt.dayofweek.isin([0])]
    norm = plt.Normalize(
        weekday_start["date"].min().timestamp(), weekday_start["date"].max().timestamp()
    )
    cmap = plt.get_cmap("RdBu")
    colors = [cmap(norm(d.timestamp())) for d in weekday_start["date"]]
    for c1 in ["avg_work_productivity", "avg_hours_working"]:
        for c2 in ["avg_workout_count", "avg_workout_hours"]:
            plt.scatter(weekday_start[c2], weekday_start[c1], color=colors)
            plt.title(f"{c1}- {c2}")
            plt.show()
