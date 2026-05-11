
import pandas as pd
import numpy as np

def analyze_distraction_cascades():
    from data_fixes import load_distracted_stacked_improved
    from productivity_analysis import DISTRACTED_CSV
    
    print("Loading distracted data...")
    df = load_distracted_stacked_improved(DISTRACTED_CSV)
    
    # Filter to distractions and task starts
    # Type 'd' = distraction, 'u' = unfocused, 't' = task start
    relevant = df[df["type"].isin(["d", "u", "t", "s"])].copy()
    relevant["time_parsed"] = pd.to_datetime(relevant["time"], format="mixed", errors="coerce")
    
    # Calculate time since previous event in the same work increment
    relevant = relevant.sort_values(["date", "time_parsed"])
    relevant["prev_time"] = relevant.groupby(["date", "work_increment"])["time_parsed"].shift(1)
    relevant["gap_minutes"] = (relevant["time_parsed"] - relevant["prev_time"]).dt.total_seconds() / 60
    
    # Distraction number in the session
    # We'll count 'd' and 'u' types
    relevant["is_distraction"] = relevant["type"].isin(["d", "u"]).astype(int)
    relevant["distraction_num"] = relevant.groupby(["date", "work_increment"])["is_distraction"].cumsum()
    
    # Time of day
    relevant["hour"] = relevant["time_parsed"].dt.hour
    
    # Analyze gaps for distractions
    distractions = relevant[relevant["is_distraction"] == 1].copy()
    distractions = distractions.dropna(subset=["gap_minutes"])
    
    print("\n--- Distraction Gap Analysis ---")
    print(f"{'Distraction #':15s} {'Mean Gap (min)':>15s} {'Count':>10s}")
    stats = distractions.groupby("distraction_num")["gap_minutes"].agg(["mean", "count"])
    print(stats.head(10).to_string())
    
    print("\n--- Interaction with Time of Day ---")
    distractions["time_period"] = pd.cut(distractions["hour"], bins=[0, 12, 17, 24], labels=["Morning", "Afternoon", "Evening"])
    
    pivot = distractions.groupby(["time_period", "distraction_num"], observed=True)["gap_minutes"].mean().unstack()
    print(pivot.iloc[:, :5].to_string()) # Show first 5 distractions

if __name__ == "__main__":
    analyze_distraction_cascades()
