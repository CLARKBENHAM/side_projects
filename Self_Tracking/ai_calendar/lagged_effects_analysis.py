
import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_fixes import load_daily_summary_improved, load_distracted_stacked_improved, load_calendar_sleep_improved, fix_streaks
from productivity_analysis import extract_daily_supplements, extract_work_start_time, load_calendar_full, DAILY_SUMMARY_CSV, DISTRACTED_CSV, CALENDAR_DIR

def analyze_lagged_effects():
    print("Loading data...")
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
    df = fix_streaks(df, cal_full)
    
    # 1. Add Amelia subcategories (from next_analysis.py)
    amelia = cal_full[cal_full["event_lower"].str.contains("amelia", case=False, na=False)].copy()
    negative_keywords = ["argue", "fight", "rant", "tiff", "yell", "scream", "flip", "stern"]
    negative_pattern = r"(?<!\w)(" + "|".join(negative_keywords) + r")(?!\w)"
    neg_mask = amelia["event_name"].str.contains(negative_pattern, case=False, regex=True, na=False)
    
    amelia["subtype"] = "quality_time"
    amelia.loc[neg_mask, "subtype"] = "negative"
    
    sub_daily = amelia.groupby(["date", "subtype"])["duration"].sum().unstack(fill_value=0)
    for st in ["negative", "quality_time"]:
        if st not in sub_daily.columns: sub_daily[st] = 0.0
    sub_daily = sub_daily.rename(columns={"negative": "amelia_neg", "quality_time": "amelia_qt"})
    
    df = df.merge(sub_daily, left_on="date", right_index=True, how="left")
    df[["amelia_neg", "amelia_qt"]] = df[["amelia_neg", "amelia_qt"]].fillna(0)
    
    # 2. Add other calendar categories
    cat_daily = cal_full.groupby(["date", "category"])["duration"].sum().unstack(fill_value=0)
    cat_daily = cat_daily.rename(columns={c: f"cal_{c}" for c in cat_daily.columns})
    df = df.merge(cat_daily, left_on="date", right_index=True, how="left")
    
    # 3. Create Lagged Features
    df = df.sort_values("date")
    outcomes = ["hours_working", "value", "energy", "focus"]
    predictors = ["amelia_neg", "amelia_qt", "cal_waste", "cal_things", "cal_blue", "cal_green", "cal_sleep_improved"]
    
    for p in predictors:
        if p in df.columns:
            df[f"prev_{p}"] = df[p].shift(1)
            
    # Also add lagged outcomes
    for o in outcomes:
        df[f"prev_{o}"] = df[o].shift(1)
        
    df = df.dropna(subset=[f"prev_{p}" for p in predictors if p in df.columns])
    
    print("\n--- Lagged Effects Analysis (Yesterday -> Today) ---")
    
    controls = ["is_weekend", "regime"] # We'll handle regime with dummies if needed
    
    for outcome in outcomes:
        print(f"\nPredicting {outcome.upper()}:")
        features = [f"prev_{p}" for p in predictors if p in df.columns] + [f"prev_{outcome}"]
        
        # Add regime dummies
        for r in df["regime"].unique():
            if pd.notna(r):
                df[f"regime_{r}"] = (df["regime"] == r).astype(int)
                features.append(f"regime_{r}")
        
        df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
        features.append("is_weekend")
        
        X = df[features].copy()
        X = sm.add_constant(X)
        y = df[outcome]
        
        # Filter to valid rows for this specific model
        valid_mask = y.notna() & X.notna().all(axis=1)
        if valid_mask.sum() < 50: continue
        
        model = sm.OLS(y[valid_mask], X[valid_mask]).fit()
        
        print(f"  R^2: {model.rsquared:.3f}, N: {valid_mask.sum()}")
        for feat in features:
            coef = model.params[feat]
            pval = model.pvalues[feat]
            if pval < 0.1: # Show significant or borderline
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "."
                print(f"    {feat:25s}: {coef:+.3f} (p={pval:.3f}) {sig}")

if __name__ == "__main__":
    analyze_lagged_effects()
