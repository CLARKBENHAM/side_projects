import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from icalendar import Calendar
from datetime import datetime, timedelta
import pytz
import re
import json
from pathlib import Path


# Function to parse .ics files and create a DataFrame
def parse_ics_files(directory):
    all_events = []

    # Get all .ics files in the directory
    directory = os.path.expanduser(directory)
    ics_files = [f for f in os.listdir(directory) if f.endswith(".ics")]

    for ics_file in ics_files:
        calendar_name = os.path.splitext(ics_file)[0]  # Get calendar name without extension
        file_path = os.path.join(directory, ics_file)

        try:
            with open(file_path, "rb") as f:
                cal = Calendar.from_ical(f.read())

                for component in cal.walk():
                    if component.name == "VEVENT":
                        event_name = str(component.get("summary", "No Title"))
                        start_time = component.get("dtstart").dt
                        end_time = component.get("dtend").dt

                        # Convert to datetime if it's a date
                        if not isinstance(start_time, datetime):
                            start_time = datetime.combine(start_time, datetime.min.time())
                        if not isinstance(end_time, datetime):
                            end_time = datetime.combine(end_time, datetime.min.time())

                        # Calculate duration in hours
                        duration = (end_time - start_time).total_seconds() / 3600

                        # Extract color if available
                        color = None
                        for key in component:
                            if "COLOR" in key or "color" in key.lower():
                                color = str(component[key])
                                break

                        # Store other metadata
                        metadata = {}
                        for key in component:
                            if key not in ["SUMMARY", "DTSTART", "DTEND"]:
                                metadata[key] = str(component[key])

                        all_events.append(
                            {
                                "event_name": event_name,
                                "calendar_name": calendar_name,
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration": duration,
                                "color": color,
                                "metadata": metadata,
                            }
                        )

            print(f"Processed {ics_file} - found {len(all_events)} events")

        except Exception as e:
            print(f"Error processing {ics_file}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_events)
    return df


# Transform data based on requirements
def transform_data(df):
    print("Transforming data...")
    df_copy = df.copy()

    # Add date columns for easier filtering
    df_copy["date"] = df_copy["start_time"].dt.date

    # 1b. Handle 'bed' and 'Woke up' entries
    sleep_entries = []
    print("Processing sleep entries...")

    # Group by date to handle sleep entries
    for date, day_df in df_copy.groupby("date"):
        # Get 'bed' entries for this day
        bed_entries = day_df[day_df["event_name"].str.lower() == "bed"].copy()

        if not bed_entries.empty:
            # Get all 'Woke up' entries from this day and the next day
            next_day = date + timedelta(days=1)
            woke_up_entries = df_copy[
                (df_copy["event_name"].str.contains("Woke up", case=False, na=False))
                & ((df_copy["date"] == date) | (df_copy["date"] == next_day))
            ].copy()

            # Process each bed entry
            for _, bed_row in bed_entries.iterrows():
                bed_time = bed_row["start_time"]

                # Find the next 'Woke up' entry after this bed time
                next_woke_ups = woke_up_entries[
                    woke_up_entries["start_time"] > bed_time
                ].sort_values("start_time")

                if not next_woke_ups.empty:
                    next_woke_up = next_woke_ups.iloc[0]
                    woke_up_time = next_woke_up["start_time"]
                    sleep_duration = (woke_up_time - bed_time).total_seconds() / 3600

                    # Create a new sleep entry
                    sleep_entry = {
                        "event_name": "sleep",
                        "calendar_name": "Meals, Supplements, Sleep",
                        "start_time": bed_time,
                        "end_time": woke_up_time,
                        "duration": sleep_duration,
                        "color": "green",
                        "metadata": {
                            "original_bed": bed_row["event_name"],
                            "original_woke_up": next_woke_up["event_name"],
                        },
                        "date": bed_time.date(),
                    }

                    sleep_entries.append(sleep_entry)

    # Remove 'bed' and 'Woke up' entries
    print(f"Removing {len(df_copy[df_copy['event_name'].str.lower() == 'bed'])} bed entries")
    print(
        "Removing"
        f" {len(df_copy[df_copy['event_name'].str.contains('Woke up', case=False, na=False)])} woke"
        " up entries"
    )

    df_copy = df_copy[
        ~(
            (df_copy["event_name"].str.lower() == "bed")
            | (df_copy["event_name"].str.contains("Woke up", case=False, na=False))
        )
    ].copy()

    # Add sleep entries
    print(f"Adding {len(sleep_entries)} sleep entries")
    if sleep_entries:
        sleep_df = pd.DataFrame(sleep_entries)
        df_copy = pd.concat([df_copy, sleep_df], ignore_index=True)

    # 1c & 1d. Handle event names with slashes
    print("Processing slashed event names...")
    split_events = []
    to_remove = []

    for idx, row in df_copy.iterrows():
        event_name = row["event_name"]

        if isinstance(event_name, str) and "/" in event_name:
            # Special case for 'knee/gym'
            if event_name.lower() == "knee/gym":
                df_copy.at[idx, "event_name"] = "gym"
                if isinstance(row["metadata"], dict):
                    row["metadata"]["original_name"] = event_name
                else:
                    df_copy.at[idx, "metadata"] = {"original_name": event_name}
            else:
                # Split other slashed names
                to_remove.append(idx)
                splits = event_name.split("/")
                duration_per_split = row["duration"] / len(splits)

                for split_name in splits:
                    new_row = row.copy()
                    new_row["event_name"] = split_name.strip()
                    new_row["duration"] = duration_per_split
                    if isinstance(new_row["metadata"], dict):
                        new_row["metadata"]["original_name"] = event_name
                    else:
                        new_row["metadata"] = {"original_name": event_name}
                    split_events.append(new_row)

    # Remove original slash entries and add split entries
    if to_remove:
        df_copy = df_copy.drop(to_remove).reset_index(drop=True)
        if split_events:
            split_df = pd.DataFrame(split_events)
            df_copy = pd.concat([df_copy, split_df], ignore_index=True)

    # 1e. Handle book entries
    print("Processing book entries...")
    # First, create a mapping of first letter abbreviations to full titles
    book_mapping = {}

    # Find full book titles
    for idx, row in df_copy.iterrows():
        event_name = row["event_name"]
        if isinstance(event_name, str) and event_name.startswith("book:"):
            full_title = event_name.replace("book:", "").strip()
            if full_title:
                first_letter = full_title[0].lower()
                book_mapping[first_letter] = full_title

    # Transform book entries
    for idx, row in df_copy.iterrows():
        event_name = row["event_name"]
        if isinstance(event_name, str):
            # Look for abbreviations like '12rfl'
            match = re.match(r"^([a-zA-Z]).*?$", event_name)
            if match and match.group(1).lower() in book_mapping:
                first_letter = match.group(1).lower()
                full_title = book_mapping[first_letter]
                df_copy.at[idx, "event_name"] = f"book: {first_letter}"

                if isinstance(row["metadata"], dict):
                    row["metadata"]["full_book_title"] = full_title
                    df_copy.at[idx, "metadata"] = row["metadata"]
                else:
                    df_copy.at[idx, "metadata"] = {"full_book_title": full_title}

    print(f"Transformation complete. Final dataframe has {len(df_copy)} entries.")
    return df_copy


# Analysis functions
def analyze_time_by_calendar(df, start_date, end_date, period="week"):
    """Analyze time spent on each calendar category for a given period."""
    # Filter by date range
    filtered_df = df[(df["start_time"] >= start_date) & (df["start_time"] <= end_date)].copy()

    # Group by calendar and calculate total duration
    if period == "week":
        filtered_df["week"] = filtered_df["start_time"].dt.isocalendar().week
        filtered_df["year"] = filtered_df["start_time"].dt.year
        result = (
            filtered_df.groupby(["year", "week", "calendar_name"])["duration"].sum().reset_index()
        )
        result = result.sort_values(["year", "week", "duration"], ascending=[True, True, False])
    elif period == "month":
        filtered_df["month"] = filtered_df["start_time"].dt.month
        filtered_df["year"] = filtered_df["start_time"].dt.year
        result = (
            filtered_df.groupby(["year", "month", "calendar_name"])["duration"].sum().reset_index()
        )
        result = result.sort_values(["year", "month", "duration"], ascending=[True, True, False])
    else:  # all time
        result = filtered_df.groupby(["calendar_name"])["duration"].sum().reset_index()
        result = result.sort_values("duration", ascending=False)

    return result


def analyze_top_events(df, start_date, end_date, top_n=10, calendar_name=None, period="week"):
    """Analyze top N event names for a given period."""
    # Filter by date range
    filtered_df = df[(df["start_time"] >= start_date) & (df["start_time"] <= end_date)].copy()

    # Filter by calendar if specified
    if calendar_name:
        filtered_df = filtered_df[filtered_df["calendar_name"] == calendar_name]

    # Group by event name and calculate total duration
    if period == "week":
        filtered_df["week"] = filtered_df["start_time"].dt.isocalendar().week
        filtered_df["year"] = filtered_df["start_time"].dt.year
        result = filtered_df.groupby(["year", "week", "event_name"])["duration"].sum().reset_index()

        # Get top N for each week
        top_events = []
        for (year, week), group in result.groupby(["year", "week"]):
            group_sorted = group.sort_values("duration", ascending=False)
            top_events.append(group_sorted.head(top_n))

        if top_events:
            result = pd.concat(top_events)
            result = result.sort_values(["year", "week", "duration"], ascending=[True, True, False])

    elif period == "month":
        filtered_df["month"] = filtered_df["start_time"].dt.month
        filtered_df["year"] = filtered_df["start_time"].dt.year
        result = (
            filtered_df.groupby(["year", "month", "event_name"])["duration"].sum().reset_index()
        )

        # Get top N for each month
        top_events = []
        for (year, month), group in result.groupby(["year", "month"]):
            group_sorted = group.sort_values("duration", ascending=False)
            top_events.append(group_sorted.head(top_n))

        if top_events:
            result = pd.concat(top_events)
            result = result.sort_values(
                ["year", "month", "duration"], ascending=[True, True, False]
            )
    else:  # all time
        result = filtered_df.groupby(["event_name"])["duration"].sum().reset_index()
        result = result.sort_values("duration", ascending=False).head(top_n)

    return result


# Visualization functions
def plot_calendar_scatter(df, calendar_name, start_date, end_date):
    """Create a scatter plot of time spent on events for a specific calendar."""
    filtered_df = df[
        (df["calendar_name"] == calendar_name)
        & (df["start_time"] >= start_date)
        & (df["start_time"] <= end_date)
    ].copy()

    filtered_df["date"] = filtered_df["start_time"].dt.date

    # Aggregate by date and event name
    agg_df = filtered_df.groupby(["date", "event_name"])["duration"].sum().reset_index()

    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=agg_df,
        x="date",
        y="duration",
        hue="event_name",
        size="duration",
        sizes=(20, 200),
        alpha=0.7,
    )

    plt.title(f"Time Spent on {calendar_name} Events")
    plt.xlabel("Date")
    plt.ylabel("Duration (hours)")
    plt.xticks(rotation=45)

    # If there are too many unique event names, limit the legend
    if len(agg_df["event_name"].unique()) > 15:
        handles, labels = scatter.get_legend_handles_labels()
        top_events = agg_df.groupby("event_name")["duration"].sum().nlargest(15).index
        legend_handles = [h for h, l in zip(handles, labels) if l in top_events]
        legend_labels = [l for l in labels if l in top_events]
        plt.legend(legend_handles, legend_labels, title="Top 15 Events")

    plt.tight_layout()
    return plt


def plot_drink_count(df, start_date, end_date):
    """Graph the number of 'drink' entries over time."""
    filtered_df = df[
        (df["event_name"].str.lower() == "drink")
        & (df["start_time"] >= start_date)
        & (df["start_time"] <= end_date)
    ].copy()

    filtered_df["date"] = filtered_df["start_time"].dt.date

    # Count drinks per day
    date_range = pd.date_range(start=start_date, end=end_date)
    date_df = pd.DataFrame({"date": date_range})

    drink_counts = filtered_df.groupby("date").size().reset_index(name="count")

    # Merge with date range to include zeros
    drink_counts = date_df.merge(drink_counts, on="date", how="left").fillna(0)

    plt.figure(figsize=(12, 6))
    plt.bar(drink_counts["date"], drink_counts["count"])
    plt.title("Number of Drink Entries per Day")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt


def plot_waste_time_days(df, start_date, end_date):
    """Graph days with >4 hours of continuous waste time."""
    filtered_df = df[
        (df["calendar_name"] == "Waste Time")
        & (df["start_time"] >= start_date)
        & (df["start_time"] <= end_date)
    ].copy()

    # Find days with continuous waste time > 4 hours
    waste_days = []
    unique_dates = filtered_df["start_time"].dt.date.unique()

    for date in unique_dates:
        day_df = filtered_df[filtered_df["start_time"].dt.date == date].sort_values("start_time")

        # Get waste time events for that day
        waste_events = []
        for _, row in day_df.iterrows():
            waste_events.append((row["start_time"], row["end_time"]))

        # Merge overlapping or close intervals (< 30 minutes apart)
        waste_events.sort()
        merged_events = []

        for start, end in waste_events:
            if not merged_events or (start - merged_events[-1][1]).total_seconds() / 60 > 30:
                merged_events.append([start, end])
            else:
                merged_events[-1][1] = max(merged_events[-1][1], end)

        # Check if any merged interval is longer than 4 hours
        for start, end in merged_events:
            duration = (end - start).total_seconds() / 3600
            if duration >= 4:
                waste_days.append(date)
                break

    # Create a date range for the full period
    date_range = pd.date_range(start=start_date, end=end_date)
    all_dates_df = pd.DataFrame({"date": date_range})

    # Mark waste days
    all_dates_df["wasted"] = all_dates_df["date"].isin(waste_days).astype(int)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(all_dates_df["date"], all_dates_df["wasted"], width=1.0)
    plt.title("Days with >4 Hours of Continuous Waste Time")
    plt.xlabel("Date")
    plt.ylabel("Wasted (1=Yes, 0=No)")
    plt.yticks([0, 1])
    plt.tight_layout()
    return plt


def plot_sleep_and_naps(df, start_date, end_date):
    """Graph sleep, naps, and total rest time."""
    # Filter by date range
    filtered_df = df[(df["start_time"] >= start_date) & (df["start_time"] <= end_date)].copy()

    # Get sleep entries
    sleep_df = filtered_df[filtered_df["event_name"] == "sleep"].copy()
    sleep_df["date"] = sleep_df["start_time"].dt.date

    # Get nap entries
    nap_df = filtered_df[filtered_df["event_name"].str.lower() == "nap"].copy()
    nap_df["date"] = nap_df["start_time"].dt.date

    # Create a date range for all days
    date_range = pd.date_range(start=start_date, end=end_date)
    all_dates_df = pd.DataFrame({"date": date_range})

    # Aggregate sleep by date
    sleep_by_date = sleep_df.groupby("date")["duration"].sum().reset_index()
    sleep_by_date.columns = ["date", "sleep_hours"]

    # Aggregate naps by date
    nap_by_date = nap_df.groupby("date")["duration"].sum().reset_index()
    nap_by_date.columns = ["date", "nap_hours"]

    # Merge with all dates
    merged_df = all_dates_df.merge(sleep_by_date, on="date", how="left").fillna(0)
    merged_df = merged_df.merge(nap_by_date, on="date", how="left").fillna(0)

    # Calculate total rest time
    merged_df["total_rest"] = merged_df["sleep_hours"] + merged_df["nap_hours"]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(merged_df["date"], merged_df["sleep_hours"], label="Sleep", linewidth=2)
    plt.plot(merged_df["date"], merged_df["nap_hours"], label="Naps", linewidth=2)
    plt.plot(merged_df["date"], merged_df["total_rest"], label="Total Rest", linewidth=2.5)
    plt.title("Sleep and Rest Time")
    plt.xlabel("Date")
    plt.ylabel("Hours")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_gym_entries(df, start_date, end_date):
    """Graph the number of entries with 'gym' in them."""
    filtered_df = df[
        (df["event_name"].str.contains("gym", case=False, na=False))
        & (df["start_time"] >= start_date)
        & (df["start_time"] <= end_date)
    ].copy()

    filtered_df["date"] = filtered_df["start_time"].dt.date

    # Count gym entries per day
    date_range = pd.date_range(start=start_date, end=end_date)
    date_df = pd.DataFrame({"date": date_range})

    gym_counts = filtered_df.groupby("date").size().reset_index(name="count")

    # Merge with date range to include zeros
    gym_counts = date_df.merge(gym_counts, on="date", how="left").fillna(0)

    # Also calculate duration
    gym_duration = filtered_df.groupby("date")["duration"].sum().reset_index()
    gym_counts = gym_counts.merge(gym_duration, on="date", how="left").fillna(0)

    # Plot count
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = "tab:blue"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Count", color=color)
    ax1.bar(gym_counts["date"], gym_counts["count"], color=color, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor=color)

    # Add duration line on secondary y-axis
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Duration (hours)", color=color)
    ax2.plot(gym_counts["date"], gym_counts["duration"], color=color, linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Gym Entries and Duration")
    plt.tight_layout()
    return plt


# Function to save all plots
def save_plots(plots_dict, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, plot in plots_dict.items():
        filename = output_dir / f"{name}.png"
        plot.savefig(filename)
        plt.close(plot.figure)
        print(f"Saved {filename}")


# Main function to run the analysis
def analyze_calendar_data(directory, start_date, end_date, output_dir="./calendar_analysis_output"):
    """
    Analyze calendar data exported from Google Calendar.

    Parameters:
    - directory: Path to the directory containing .ics files
    - start_date: Start date for analysis (datetime object)
    - end_date: End date for analysis (datetime object)
    - output_dir: Directory to save output plots and data
    """
    print(f"Analyzing calendar data from {start_date} to {end_date}")

    # Create output directory
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Parse .ics files
    df = parse_ics_files(directory)

    # Save raw data
    df.to_csv(os.path.join(output_dir, "raw_calendar_data.csv"), index=False)

    # Apply transformations
    transformed_df = transform_data(df)

    # Save transformed data
    transformed_df.to_csv(os.path.join(output_dir, "transformed_calendar_data.csv"), index=False)

    # 1. Analyze time by calendar
    print("Analyzing time spent by calendar...")
    weekly_time = analyze_time_by_calendar(transformed_df, start_date, end_date, period="week")
    monthly_time = analyze_time_by_calendar(transformed_df, start_date, end_date, period="month")
    all_time = analyze_time_by_calendar(transformed_df, start_date, end_date, period="all")

    # Save time analyses
    weekly_time.to_csv(os.path.join(output_dir, "weekly_time_by_calendar.csv"), index=False)
    monthly_time.to_csv(os.path.join(output_dir, "monthly_time_by_calendar.csv"), index=False)
    all_time.to_csv(os.path.join(output_dir, "all_time_by_calendar.csv"), index=False)

    # 2. Analyze top events
    print("Analyzing top events...")
    top_events_weekly = analyze_top_events(
        transformed_df, start_date, end_date, top_n=10, period="week"
    )
    top_events_monthly = analyze_top_events(
        transformed_df, start_date, end_date, top_n=10, period="month"
    )
    top_events_all = analyze_top_events(
        transformed_df, start_date, end_date, top_n=10, period="all"
    )

    # Save top events analyses
    top_events_weekly.to_csv(os.path.join(output_dir, "top_events_weekly.csv"), index=False)
    top_events_monthly.to_csv(os.path.join(output_dir, "top_events_monthly.csv"), index=False)
    top_events_all.to_csv(os.path.join(output_dir, "top_events_all.csv"), index=False)

    # Analyze top events for each calendar
    calendars = transformed_df["calendar_name"].unique()
    for calendar in calendars:
        print(f"Analyzing top events for calendar: {calendar}")
        calendar_top_events = analyze_top_events(
            transformed_df, start_date, end_date, top_n=10, calendar_name=calendar, period="all"
        )
        calendar_file_name = calendar.replace(" ", "_").replace(",", "").lower()
        calendar_top_events.to_csv(
            os.path.join(output_dir, f"top_events_{calendar_file_name}.csv"), index=False
        )

    # 4-8. Create visualizations
    print("Creating visualizations...")
    plots = {}

    # 4. Scatter plots for each calendar
    for calendar in calendars:
        print(f"Creating scatter plot for {calendar}")
        plots[f"scatter_{calendar.replace(' ', '_').replace(',', '').lower()}"] = (
            plot_calendar_scatter(transformed_df, calendar, start_date, end_date)
        )

    # 5. Plot drink entries
    print("Creating drink entries plot")
    plots["drink_entries"] = plot_drink_count(transformed_df, start_date, end_date)

    # 6. Plot waste time days
    print("Creating waste time plot")
    plots["waste_time_days"] = plot_waste_time_days(transformed_df, start_date, end_date)

    # 7. Plot sleep and naps
    print("Creating sleep and nap plot")
    plots["sleep_and_naps"] = plot_sleep_and_naps(transformed_df, start_date, end_date)

    # 8. Plot gym entries
    print("Creating gym entries plot")
    plots["gym_entries"] = plot_gym_entries(transformed_df, start_date, end_date)

    # Save all plots
    print("Saving plots...")
    save_plots(plots, output_dir)

    print(f"Analysis complete! Results saved to {output_dir}")
    return {
        "dataframe": transformed_df,
        "weekly_time": weekly_time,
        "monthly_time": monthly_time,
        "all_time": all_time,
        "top_events_weekly": top_events_weekly,
        "top_events_monthly": top_events_monthly,
        "top_events_all": top_events_all,
        "plots": plots,
    }


# Example usage:
if __name__ == "__main__":
    # Replace with actual directory path
    calendar_dir = "~/Downloads/calendar-takeout-20250217T231511Z-001/Calendar/"

    # Set your date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 2, 17)

    # Run the analysis
    results = analyze_calendar_data(
        calendar_dir, start_date, end_date, output_dir="~/calendar_analysis_results"
    )

    print("Analysis complete!")
