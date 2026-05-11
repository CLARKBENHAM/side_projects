
import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_fixes import load_daily_summary_improved, load_distracted_stacked_improved, load_calendar_sleep_improved, fix_streaks
from productivity_analysis import extract_daily_supplements, extract_work_start_time, load_calendar_full, DAILY_SUMMARY_CSV, DISTRACTED_CSV, CALENDAR_DIR, SUPPLEMENT_ALIASES

def analyze_supp_interactions():
    print("Loading data...")
    daily = load_daily_summary_improved(DAILY_SUMMARY_CSV)
    distracted = load_distracted_stacked_improved(DISTRACTED_CSV)
    supplements = extract_daily_supplements(distracted)
    
    # Merge
    df = daily.merge(supplements, on="date", how="left")
    
    # Check for Caffeine and Adderall
    if "caffeine" not in df.columns or "adderall" not in df.columns:
        print("Missing caffeine/adderall columns.")
        return
        
    df["has_caf"] = (df["caffeine"] > 0).astype(int)
    df["has_add"] = (df["adderall"] > 0).astype(int)
    df["has_both"] = (df["has_caf"] & df["has_add"]).astype(int)
    
    df = df[df["hours_working"] > 0].copy()
    
    print("\n--- Caffeine & Adderall Interaction (n={}) ---".format(len(df)))
    groups = {
        "Neither": df[(df["has_caf"] == 0) & (df["has_add"] == 0)],
        "Caf Only": df[(df["has_caf"] == 1) & (df["has_add"] == 0)],
        "Add Only": df[(df["has_caf"] == 0) & (df["has_add"] == 1)],
        "Both": df[(df["has_caf"] == 1) & (df["has_add"] == 1)]
    }
    
    print(f"{'Group':15s} {'Mean Hours':>12s} {'Mean Value':>12s} {'Count':>6s}")
    for name, g in groups.items():
        if len(g) > 0:
            print(f"{name:15s} {g['hours_working'].mean():12.2f} {g['value'].mean():12.2f} {len(g):6d}")
            
    # OLS with interaction term
    X = df[["has_caf", "has_add", "has_both"]]
    X = sm.add_constant(X)
    y = df["hours_working"]
    model = sm.OLS(y, X).fit()
    print("\nRegression Output (Hours Working):")
    print(model.summary2().tables[1].to_string())
    
    # Also check Modafinil
    if "modafinil" in df.columns:
        df["has_mod"] = (df["modafinil"] > 0).astype(int)
        print("\n--- Modafinil vs others ---")
        mod_groups = {
            "Mod Only": df[(df["has_mod"] == 1) & (df["has_caf"] == 0) & (df["has_add"] == 0)],
            "Mod + Caf": df[(df["has_mod"] == 1) & (df["has_caf"] == 1) & (df["has_add"] == 0)],
            "Mod + Add": df[(df["has_mod"] == 1) & (df["has_caf"] == 0) & (df["has_add"] == 1)],
            "Mod + Both": df[(df["has_mod"] == 1) & (df["has_caf"] == 1) & (df["has_add"] == 1)]
        }
        for name, g in mod_groups.items():
            if len(g) > 0:
                print(f"{name:15s} {g['hours_working'].mean():12.2f} {len(g):6d}")

if __name__ == "__main__":
    analyze_supp_interactions()
