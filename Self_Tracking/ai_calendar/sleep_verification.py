
import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_fixes import load_daily_summary_improved, load_distracted_stacked_improved, load_calendar_sleep_improved, fix_streaks
from productivity_analysis import extract_daily_supplements, extract_work_start_time, load_calendar_full, DAILY_SUMMARY_CSV, DISTRACTED_CSV, CALENDAR_DIR

def verify_sleep_fix():
    print("Loading data...")
    daily = load_daily_summary_improved(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked_improved(DISTRACTED_CSV)
    supplements = extract_daily_supplements(distracted)
    work_starts = extract_work_start_time(distracted)
    cal_sleep_new = load_calendar_sleep_improved(CALENDAR_DIR)
    
    # Let's get the original sleep too for comparison
    from productivity_analysis import load_calendar_sleep as load_calendar_sleep_orig
    cal_sleep_orig = load_calendar_sleep_orig(CALENDAR_DIR)
    
    # Merge
    df = daily.merge(supplements, on="date", how="left")
    df = df.merge(cal_sleep_new, on="date", how="left")
    df = df.merge(cal_sleep_orig, on="date", how="left")
    
    # Correlation analysis
    df = df[df["hours_working"] > 0].copy()
    
    print("\n--- Correlation with Hours Working (n={}) ---".format(len(df)))
    for col in ["cal_sleep_improved", "calendar_sleep_hours"]:
        if col in df.columns:
            corr = df[col].corr(df["hours_working"])
            print(f"  {col:25s}: r={corr:.3f}")
            
    # Regression analysis
    print("\n--- OLS Regression: Hours Working ~ Sleep ---")
    for col in ["cal_sleep_improved", "calendar_sleep_hours"]:
        if col in df.columns:
            sub = df[["hours_working", col]].dropna()
            X = sm.add_constant(sub[col])
            y = sub["hours_working"]
            model = sm.OLS(y, X).fit()
            print(f"  {col:25s}: coef={model.params[col]:+.3f} (p={model.pvalues[col]:.3f}), R^2={model.rsquared:.3f}")

    # Check "Early-Morning Nap" reclassification instances
    print("\n--- Examples of Improved Sleep (diff > 1h) ---")
    df["sleep_diff"] = df["cal_sleep_improved"] - df["calendar_sleep_hours"]
    changed = df[abs(df["sleep_diff"]) > 1].sort_values("sleep_diff", ascending=False)
    for _, row in changed.head(10).iterrows():
        print(f"  {row['date'].date()}: Orig={row['calendar_sleep_hours']:.1f}h, Improved={row['cal_sleep_improved']:.1f}h, Diff={row['sleep_diff']:+.1f}h")

if __name__ == "__main__":
    verify_sleep_fix()
