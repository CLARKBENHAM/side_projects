import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Load scored books
df = pd.read_csv("gwern_scored.csv")

def get_curve(books_df, score_col):
    df_sorted = books_df.sort_values(score_col)
    n = len(df_sorted)
    baseline = df_sorted['trueEnjoy'].mean()
    res = {}
    for pct in [20, 40, 50, 60, 80]:
        keep = df_sorted.iloc[int(n*pct/100):]
        res[pct] = (keep['trueEnjoy'].mean() - baseline) if len(keep) > 0 else 0
    return res

print("=== 1. Non-category baseline (Overall Gwern vs User) ===")
# See previous correlations
corr, _ = pearsonr(df['grRating'], df['trueEnjoy'])
print(f"Overall Pearson Correlation (GR vs Enjoy): {corr:.4f}")
overall_gains = get_curve(df, 'heuristic_sum')
print(f"Overall Drop Gains (GR heuristic): Drop 20%: +{overall_gains[20]:.4f}, Drop 50%: +{overall_gains[50]:.4f}, Drop 80%: +{overall_gains[80]:.4f}")

print("\n=== 2. Category Segmented Analysis ===")
for cat in df['category'].unique():
    subset = df[df['category'] == cat]
    if len(subset) < 10: continue
    cat_corr, _ = pearsonr(subset['grRating'], subset['trueEnjoy'])
    # Ridge
    ridge_gains = get_curve(subset, 'ridge_full_enjoy')
    print(f"Category: {cat:20} (n={len(subset):4d}) | Corr: {cat_corr:+.4f} | Ridge Drop 50% Gain: +{ridge_gains[50]:.4f}")

print("\n=== 2c. Year Segmented Analysis (Gains by Year) ===")
# Drop books with no read year
df['read_year'] = pd.to_numeric(df['read_year'], errors='coerce')
df_valid_yr = df.dropna(subset=['read_year']).copy()
df_valid_yr['read_year'] = df_valid_yr['read_year'].astype(int)

# Group by year
for yr in sorted(df_valid_yr['read_year'].unique()):
    subset = df_valid_yr[df_valid_yr['read_year'] == yr]
    if len(subset) < 20: continue
    gr_gains = get_curve(subset, 'heuristic_sum')
    rf_gains = get_curve(subset, 'rf_enjoy')
    print(f"Year {yr} (n={len(subset):3d}) | GR D20: +{gr_gains[20]:.3f}, D40: +{gr_gains[40]:.3f}, D60: +{gr_gains[60]:.3f}, D80: +{gr_gains[80]:.3f}")
    print(f"   [RF]       | RF D20: +{rf_gains[20]:.3f}, D40: +{rf_gains[40]:.3f}, D60: +{rf_gains[60]:.3f}, D80: +{rf_gains[80]:.3f}")

print("\n=== Scraping aborted due to Cloudflare ===")
