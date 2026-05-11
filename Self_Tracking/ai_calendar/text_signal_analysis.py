
import pandas as pd
import numpy as np
from pathlib import Path
import os
from collections import Counter
import re

from data_fixes import load_daily_summary_improved, load_distracted_stacked_improved, load_calendar_sleep_improved, fix_streaks
from productivity_analysis import extract_daily_supplements, extract_work_start_time, load_calendar_full, REPO_DIR, DAILY_SUMMARY_CSV, DISTRACTED_CSV, CALENDAR_DIR

def analyze_text_signals():
    print("Loading improved data...")
    daily = load_daily_summary_improved(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked_improved(DISTRACTED_CSV)
    supplements = extract_daily_supplements(distracted)
    work_starts = extract_work_start_time(distracted)
    cal_sleep = load_calendar_sleep_improved(CALENDAR_DIR)
    cal_full = load_calendar_full(CALENDAR_DIR)
    
    # Merge
    df = daily.merge(supplements, on="date", how="left")
    df = df.merge(cal_sleep, on="date", how="left")
    df = df.merge(work_starts, on="date", how="left")
    
    # Fix streaks using both CSV and Calendar
    df = fix_streaks(df, cal_full)
    
    print(f"Final Analysis DF: {len(df)} days")
    
    # 1. Analyze Tasks Summary
    print("\n--- Tasks Summary Analysis ---")
    high_value = df[df["value"] >= 4].copy()
    low_value = df[df["value"] < 2].copy()
    
    def get_top_words(text_series, top_n=50):
        words = []
        for text in text_series:
            if pd.isna(text): continue
            # Basic cleaning
            clean = re.sub(r'[^\w\s]', ' ', str(text).lower())
            words.extend([w for w in clean.split() if len(w) > 3])
        return Counter(words).most_common(top_n)

    high_words = dict(get_top_words(high_value["tasks_summary"]))
    low_words = dict(get_top_words(low_value["tasks_summary"]))
    
    # Find words that are more common in high value vs low value
    all_words = set(high_words.keys()) | set(low_words.keys())
    diffs = []
    for w in all_words:
        h_freq = high_words.get(w, 0) / len(high_value) if len(high_value) > 0 else 0
        l_freq = low_words.get(w, 0) / len(low_value) if len(low_value) > 0 else 0
        diffs.append((w, h_freq, l_freq, h_freq - l_freq))
        
    diffs.sort(key=lambda x: x[3], reverse=True)
    
    print("\nTop words for HIGH Value days (relative to low value):")
    for w, h, l, d in diffs[:20]:
        print(f"  {w:15s}: {h:.3f} vs {l:.3f} (diff: {d:+.3f})")
        
    print("\nTop words for LOW Value days (relative to high value):")
    for w, h, l, d in diffs[-20:]:
        print(f"  {w:15s}: {h:.3f} vs {l:.3f} (diff: {d:+.3f})")

    # 2. Analyze Productivity Value (Reflections)
    print("\n--- Micro-Reflections (Productivity Value) Analysis ---")
    # Join reflections per day
    daily_reflections = distracted[distracted["productivity_value"] != ""].groupby("date")["productivity_value"].apply(lambda x: " | ".join(x)).reset_index()
    
    df = df.merge(daily_reflections, on="date", how="left")
    
    high_prod = df[df["hours_working"] >= 6].copy()
    low_prod = df[df["hours_working"] < 2].copy()
    
    high_ref_words = dict(get_top_words(high_prod["productivity_value"]))
    low_ref_words = dict(get_top_words(low_prod["productivity_value"]))
    
    ref_all_words = set(high_ref_words.keys()) | set(low_ref_words.keys())
    ref_diffs = []
    for w in ref_all_words:
        h_freq = high_ref_words.get(w, 0) / len(high_prod) if len(high_prod) > 0 else 0
        l_freq = low_ref_words.get(w, 0) / len(low_prod) if len(low_prod) > 0 else 0
        ref_diffs.append((w, h_freq, l_freq, h_freq - l_freq))
        
    ref_diffs.sort(key=lambda x: x[3], reverse=True)
    
    print("\nTop reflection words for HIGH Productivity days:")
    for w, h, l, d in ref_diffs[:20]:
        print(f"  {w:15s}: {h:.3f} vs {l:.3f} (diff: {d:+.3f})")
        
    print("\nTop reflection words for LOW Productivity days:")
    for w, h, l, d in ref_diffs[-20:]:
        print(f"  {w:15s}: {h:.3f} vs {l:.3f} (diff: {d:+.3f})")

if __name__ == "__main__":
    analyze_text_signals()
