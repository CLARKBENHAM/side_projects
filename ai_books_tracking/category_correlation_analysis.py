
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def main():
    path = Path('ai_books_tracking/FINAL_CONSOLIDATED_MASTER.csv')
    df = pd.read_csv(path)
    
    # Target prep (Average Enjoyment and Usefulness)
    def get_avg(row, prefix):
        vals = [v for k, v in row.items() if k.startswith(prefix)]
        valid_vals = [pd.to_numeric(v, errors='coerce') for v in vals if pd.notna(v)]
        return np.mean(valid_vals) if valid_vals else np.nan

    df['avg_enjoy'] = df.apply(lambda r: get_avg(r, 'enjoyment_'), axis=1)
    df['avg_useful'] = df.apply(lambda r: get_avg(r, 'usefulness_'), axis=1)
    
    # Utility
    df['util_enjoy'] = np.power(np.maximum(0, df['avg_enjoy'] - 1), 1.3)
    df['util_useful'] = np.power(np.maximum(0, df['avg_useful'] - 1), 1.8)
    
    # External ratings
    df['gr'] = pd.to_numeric(df['gr_rating'], errors='coerce')
    df['amz'] = pd.to_numeric(df['amz_rating'], errors='coerce')
    
    categories = df['final_category'].unique()
    results = []
    
    for cat in categories:
        subset = df[df['final_category'] == cat]
        if len(subset) < 5: continue
        
        # Enjoyment correlation with GR
        valid_gr = subset[subset['gr'].notna() & subset['util_enjoy'].notna()]
        if len(valid_gr) >= 5:
            rho_e, _ = stats.spearmanr(valid_gr['gr'], valid_gr['util_enjoy'])
            rho_u, _ = stats.spearmanr(valid_gr['gr'], valid_gr['util_useful'])
        else:
            rho_e, rho_u = np.nan, np.nan
            
        results.append({
            'category': cat,
            'n': len(subset),
            'enjoy_rho': rho_e,
            'useful_rho': rho_u
        })
        
    res_df = pd.DataFrame(results).sort_values('enjoy_rho', ascending=False)
    print("=== Category Signal Analysis (GR Rating vs Utility) ===")
    print(res_df.to_string(index=False))
    
    res_df.to_csv('ai_books_tracking/CATEGORY_CORRELATIONS.csv', index=False)

if __name__ == "__main__":
    main()
