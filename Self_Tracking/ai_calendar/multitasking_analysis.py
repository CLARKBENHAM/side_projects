
import pandas as pd
import re

def analyze_multitasking():
    weekly_csv = pd.read_csv("data/Work Summary  - Weekly Summary_Projects.csv", header=None, skiprows=3)
    weekly = pd.DataFrame({
        "start": pd.to_datetime(weekly_csv.iloc[:, 1], format="mixed", errors="coerce"),
        "hours": pd.to_numeric(weekly_csv.iloc[:, 5], errors="coerce"),
        "summary": weekly_csv.iloc[:, 11].astype(str)
    })
    weekly = weekly.dropna(subset=["start", "hours"])
    
    def count_projects(text):
        if text.lower() == "nothing" or text.lower() == "n/a": return 0
        # Split by semicolon or comma if they seem to separate items
        parts = [p.strip() for p in re.split(r'[;]', text) if p.strip()]
        return len(parts)

    weekly["n_projects"] = weekly["summary"].apply(count_projects)
    
    print("\n--- Multitasking Analysis ---")
    print(f"Mean hours by number of projects:")
    stats = weekly.groupby("n_projects")["hours"].agg(["mean", "count", "std"])
    print(stats.to_string())
    
    print("\nCorrelation (n_projects vs hours):", weekly["n_projects"].corr(weekly["hours"]))

if __name__ == "__main__":
    analyze_multitasking()
