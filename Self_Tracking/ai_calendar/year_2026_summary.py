from datetime import datetime

import pandas as pd
from data_fixes import load_distracted_stacked_improved
from productivity_analysis import CALENDAR_DIR, DISTRACTED_CSV, load_calendar_full


def generate_2026_summary():
    print("Loading 2026 data...")
    # Load calendar data for 2026
    df = load_calendar_full(CALENDAR_DIR)
    df = df[df["date"].dt.year == 2026].copy()

    # Load distracted data to find "drunk days"
    distracted = load_distracted_stacked_improved(DISTRACTED_CSV)
    drunk_dates = set(
        distracted[
            distracted["comment"].str.contains("drink", case=False, na=False)
            | distracted["type"].str.contains("drink", case=False, na=False)
        ]["date"].dt.date
    )

    # Also check for "drink" events in calendar (just in case)
    drunk_dates |= set(
        df[df["event_name"].str.lower().str.contains("drink", na=False)]["date"].dt.date
    )

    # Categories: Books, Blogs, Waste Time
    df["is_book"] = df["event_name"].str.lower().str.contains("book:", na=False)
    df["is_blog"] = df["event_name"].str.lower().str.contains("blog", na=False)
    df["is_waste"] = df["category"] == "waste"

    # Drunk day flag
    df["is_drunk_day"] = df["date"].dt.date.isin(drunk_dates)

    # Monthly aggregates
    df["month"] = df["date"].dt.month

    summary = []
    for month in sorted(df["month"].unique()):
        m_df = df[df["month"] == month]

        books_h = m_df[m_df["is_book"]]["duration"].sum()
        blogs_h = m_df[m_df["is_blog"]]["duration"].sum()
        total_waste_h = m_df[m_df["is_waste"]]["duration"].sum()

        drunk_waste_h = m_df[m_df["is_waste"] & m_df["is_drunk_day"]]["duration"].sum()

        summary.append(
            {
                "Month": datetime(2026, month, 1).strftime("%B"),
                "Books (h)": books_h,
                "Blogs (h)": blogs_h,
                "Total Waste (h)": total_waste_h,
                "Waste on Drunk Days (h)": drunk_waste_h,
            }
        )

    summary_df = pd.DataFrame(summary)
    print("\n2026 Monthly Activity Table:")
    print(summary_df.to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    generate_2026_summary()
