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


# %% Correlation between calendar and daily work tracker
def partition_calendar_by_sleep_with_boundaries(df):
    df = df.sort_values("start_time").reset_index(drop=True)
    sleep_mask = df["event_name"].str.lower() == "sleep"
    if not sleep_mask.any():
        return []
    sleep_indices = df.loc[sleep_mask].index.tolist()
    partitions = []
    for i, idx in enumerate(sleep_indices):
        start_time = df.loc[idx, "start_time"]
        sleep_date = start_time.date()
        if i < len(sleep_indices) - 1:
            end_time = df.loc[sleep_indices[i + 1], "start_time"]
        else:
            end_time = df["start_time"].max() + pd.Timedelta(seconds=1)
        block = df[(df["start_time"] >= start_time) & (df["start_time"] < end_time)]
        partitions.append(
            {"date": sleep_date, "start_time": start_time, "end_time": end_time, "block": block}
        )
    return partitions


def compute_clark_duration_by_sleep_partition(calendar_df):
    partitions = partition_calendar_by_sleep_with_boundaries(calendar_df)
    rows = []
    for part in partitions:
        d = part["date"]
        block = part["block"]
        # Sum duration for events in the Clark calendar
        clark_duration = block.loc[
            block["calendar_name"].str.lower() == "clark.benham@gmail.com", "duration"
        ].sum()
        rows.append({"date": d, "clark_duration": clark_duration})
    return pd.DataFrame(rows)


# --- Usage ---
if __name__ == "__main__":
    # Compute Clark's daily duration using sleep partitions.
    calendar_df = df.copy()
    work_df = work_data.copy()
    clark_df = compute_clark_duration_by_sleep_partition(calendar_df)

    # Ensure work_df dates are date objects.
    work_df["date"] = pd.to_datetime(work_df["date"]).dt.date

    # Merge on the partition 'date'
    merged = pd.merge(work_df, clark_df, on="date", how="inner")

    # Compute the correlation between Clark's duration and Hours Working.
    correlation = merged["clark_duration"].corr(merged["Hours Working"])
    print(
        "Correlation between Clark event duration (split by sleep) and Hours Working:", correlation
    )

    plt.title("difference in worksheet vs hours there")
    plt.hist(merged["clark_duration"] - merged["Hours Working"])
    print("More than 5 hours wrong")
    print(merged[(merged["clark_duration"] - merged["Hours Working"]) > 5])
    print(merged[(merged["clark_duration"] - merged["Hours Working"]) < -1])

    calendar_df["start_date"] = pd.to_datetime(calendar_df["start_time"])
    # Define the target date
    target_date = pd.to_datetime("2024-03-19").date()
    # Filter for calendar entries on March 19, 2024
    entries_mar19 = calendar_df[calendar_df["start_date"].dt.date == target_date]

    print(entries_mar19.query("calendar_name=='clark.benham@gmail.com'"))

    target_date = pd.to_datetime("2023-07-10").date()
    work_df[work_df["date"] == target_date]
