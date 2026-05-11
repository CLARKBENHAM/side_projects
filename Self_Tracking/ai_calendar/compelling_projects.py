
import pandas as pd
from collections import Counter
import re

def analyze_compelling_projects():
    # Load weekly summary CSV (skipping header rows and handling formatting)
    weekly_csv = pd.read_csv("data/Work Summary  - Weekly Summary_Projects.csv", header=None, skiprows=3)
    
    # Standard columns from tail output earlier:
    # Col 0: Regime, Col 1: Start, Col 2: End, Col 5: Hours, Col 11: Summary
    weekly = pd.DataFrame({
        "regime": weekly_csv.iloc[:, 0],
        "start": pd.to_datetime(weekly_csv.iloc[:, 1], format="mixed", errors="coerce"),
        "hours": pd.to_numeric(weekly_csv.iloc[:, 5], errors="coerce"),
        "summary": weekly_csv.iloc[:, 11].astype(str)
    })
    
    weekly = weekly.dropna(subset=["start", "hours"])
    
    high_weeks = weekly[weekly["hours"] >= 40].copy()
    low_weeks = weekly[weekly["hours"] < 20].copy()
    
    print(f"Total Weeks: {len(weekly)}")
    print(f"High Weeks (40h+): {len(high_weeks)}")
    print(f"Low Weeks (<20h): {len(low_weeks)}")
    
    def get_keywords(text_series):
        words = []
        for text in text_series:
            # Extract potential project names (semicolon separated usually)
            parts = re.split(r'[;,\.]', text)
            for p in parts:
                p = p.strip().lower()
                if len(p) > 2 and "nothing" not in p and "n/a" not in p:
                    words.append(p)
        return Counter(words)

    high_counts = get_keywords(high_weeks["summary"])
    low_counts = get_keywords(low_weeks["summary"])
    
    print("\nProject Keywords in HIGH Weeks (count):")
    for k, v in high_counts.most_common(20):
        print(f"  {k:30s}: {v}")
        
    print("\nProject Keywords in LOW Weeks (count):")
    for k, v in low_counts.most_common(20):
        print(f"  {k:30s}: {v}")

if __name__ == "__main__":
    analyze_compelling_projects()
