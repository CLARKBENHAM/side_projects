
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from data_fixes import load_daily_summary_improved, load_distracted_stacked_improved, load_calendar_sleep_improved
from productivity_analysis import extract_daily_supplements, extract_work_start_time, load_calendar_full, DAILY_SUMMARY_CSV, DISTRACTED_CSV, CALENDAR_DIR

def analyze_efficiency_and_sleep():
    print("Loading data...")
    daily = load_daily_summary_improved(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked_improved(DISTRACTED_CSV)
    cal_sleep_new = load_calendar_sleep_improved(CALENDAR_DIR)
    
    # Let's get the original sleep/nap too for comparison
    from productivity_analysis import load_calendar_sleep as load_calendar_sleep_orig
    cal_sleep_orig = load_calendar_sleep_orig(CALENDAR_DIR)
    # The original load_calendar_sleep also returns a result with calendar_sleep_hours
    # Let's extract the raw nap from cal_full for a truly 'unimproved' baseline
    cal_full = load_calendar_full(CALENDAR_DIR)
    raw_nap = cal_full[cal_full["event_lower"] == "nap"].groupby("date")["duration"].sum().reset_index(name="cal_nap_raw")
    
    # Merge
    df = daily.merge(cal_sleep_new, on="date", how="left")
    df = df.merge(cal_sleep_orig[["date", "calendar_sleep_hours"]], on="date", how="left")
    df = df.merge(raw_nap, on="date", how="left")
    
    # 1. Efficiency Analysis
    df["total_tracked_hours"] = df["hours_working"] + (df["len_distractions"] / 60)
    df["efficiency"] = df["hours_working"] / df["total_tracked_hours"]
    
    # Filter to days with significant tracked time
    valid_eff = df[df["total_tracked_hours"] >= 1].copy()
    valid_eff = valid_eff.sort_values("date")
    valid_eff["eff_rolling_30d"] = valid_eff["efficiency"].rolling(30).mean()
    
    print("\n--- Efficiency Evolution ---")
    print(f"Overall Mean Efficiency: {valid_eff['efficiency'].mean():.1%}")
    
    # Trends by year
    valid_eff["year"] = valid_eff["date"].dt.year
    yearly_eff = valid_eff.groupby("year")["efficiency"].mean()
    print("\nYearly Mean Efficiency:")
    print(yearly_eff)
    
    # 2. Sleep vs Nap Prediction Comparison
    print("\n--- Sleep & Nap Prediction Comparison (OLS) ---")
    df_work = df[df["hours_working"] > 0].copy()
    
    # Model A: Original/Raw-ish
    # We use calendar_sleep_hours (orig) and cal_nap_raw
    sub_a = df_work[["hours_working", "calendar_sleep_hours", "cal_nap_raw"]].dropna()
    X_a = sm.add_constant(sub_a[["calendar_sleep_hours", "cal_nap_raw"]])
    model_a = sm.OLS(sub_a["hours_working"], X_a).fit()
    
    # Model B: Improved
    sub_b = df_work[["hours_working", "cal_sleep_improved", "cal_nap_improved"]].dropna()
    X_b = sm.add_constant(sub_b[["cal_sleep_improved", "cal_nap_improved"]])
    model_b = sm.OLS(sub_b["hours_working"], X_b).fit()
    
    print("\nMODEL A (Original Logic):")
    print(model_a.summary2().tables[1].to_string())
    print(f"R-squared: {model_a.rsquared:.4f}")
    
    print("\nMODEL B (Improved Logic):")
    print(model_b.summary2().tables[1].to_string())
    print(f"R-squared: {model_b.rsquared:.4f}")

    # Plot Efficiency over time
    plt.figure(figsize=(12, 6))
    plt.plot(valid_eff["date"], valid_eff["eff_rolling_30d"], label="30-day Rolling Efficiency")
    plt.axhline(valid_eff["efficiency"].mean(), color='r', linestyle='--', label="Global Mean")
    plt.title("Work Efficiency Evolution (Hours Working / Total Tracked Hours)")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Self_Tracking/efficiency_evolution.png")
    print("\nSaved efficiency plot to Self_Tracking/efficiency_evolution.png")

if __name__ == "__main__":
    analyze_efficiency_and_sleep()
