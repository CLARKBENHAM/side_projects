
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

def main():
    # 1. Load Data
    path = Path('ai_books_tracking/FINAL_CONSOLIDATED_MASTER.csv')
    df = pd.read_csv(path)

    # 2. Target Engineering (Average Enjoyment and Usefulness)
    def get_avg_target(row, target_prefix):
        if row['source_original'] == 'Holdout 2026':
            vals = [row.get(f'{target_prefix}_h2026'), row.get(f'{target_prefix}_h2026_2nd')]
        else:
            enjoy_map = {'enjoyment': ['enjoyment_play', 'enjoyment_r2'],
                         'usefulness': ['usefulness_play', 'usefulness_r2']}
            vals = [row.get(c) for c in enjoy_map[target_prefix]]
        valid_vals = [pd.to_numeric(v, errors='coerce') for v in vals if pd.notna(v)]
        return np.mean(valid_vals) if valid_vals else np.nan

    df['avg_enjoyment'] = df.apply(lambda r: get_avg_target(r, 'enjoyment'), axis=1)
    df['avg_usefulness'] = df.apply(lambda r: get_avg_target(r, 'usefulness'), axis=1)

    # 3. Utility Functions
    # (enjoyment - 1)^1.3
    # (usefulness - 1)^1.8
    df['util_enjoyment'] = np.power(np.maximum(0, df['avg_enjoyment'] - 1), 1.3)
    df['util_usefulness'] = np.power(np.maximum(0, df['avg_usefulness'] - 1), 1.8)

    # 4. Feature Engineering & Imputation
    df['gr_rating'] = pd.to_numeric(df['gr_rating'], errors='coerce')
    df['amz_rating'] = pd.to_numeric(df['amz_rating'], errors='coerce')
    df['log_gr_count'] = np.log10(pd.to_numeric(df['gr_count'], errors='coerce') + 1)
    df['log_amz_count'] = np.log10(pd.to_numeric(df['amz_count'], errors='coerce') + 1)
    df['category'] = df['dropbox_category'].fillna('Unknown')
    
    # Fill NAs
    df['gr_rating'] = df['gr_rating'].fillna(df['gr_rating'].mean())
    df['amz_rating'] = df['amz_rating'].fillna(df['gr_rating'])
    df['log_gr_count'] = df['log_gr_count'].fillna(df['log_gr_count'].median())
    df['log_amz_count'] = df['log_amz_count'].fillna(df['log_amz_count'].median())

    numerical_cols = ['gr_rating', 'amz_rating', 'log_gr_count', 'log_amz_count']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def run_model(target_name):
        df_valid = df[df[target_name].notna()].copy()
        df_train = df_valid[df_valid['source_original'] != 'Holdout 2026'].copy()
        df_holdout = df_valid[df_valid['source_original'] == 'Holdout 2026'].copy()
        
        cat_train = encoder.fit_transform(df_train[['category']])
        cat_holdout = encoder.transform(df_holdout[['category']])
        
        X_train = np.hstack([df_train[numerical_cols].values, cat_train])
        y_train = df_train[target_name].values
        X_holdout = np.hstack([df_holdout[numerical_cols].values, cat_holdout])
        y_holdout = df_holdout[target_name].values
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        df_train['prediction'] = model.predict(X_train)
        df_holdout['prediction'] = model.predict(X_holdout)
        
        return df_train, df_holdout

    # Train on Utility
    train_u_enjoy, holdout_u_enjoy = run_model('util_enjoyment')
    train_u_useful, holdout_u_useful = run_model('util_usefulness')

    # 5. Advanced Drop Analysis
    def advanced_drop_analysis(df, target_col, pred_col):
        df_sorted = df.sort_values(pred_col, ascending=False).copy()
        df_perfect = df.sort_values(target_col, ascending=False).copy()
        n = len(df_sorted)
        results = []
        for p in [0, 10, 25, 50, 60, 75, 80, 90, 95]:
            n_keep = max(1, int(n * (1 - p/100)))
            model_avg = df_sorted.head(n_keep)[target_col].mean()
            perfect_avg = df_perfect.head(n_keep)[target_col].mean()
            results.append({
                'drop_pct': p,
                'model_avg': model_avg,
                'perfect_avg': perfect_avg,
                'efficiency': model_avg / perfect_avg if perfect_avg > 0 else 0
            })
        return pd.DataFrame(results)

    print("=== DROP ANALYSIS: ENJOYMENT UTILITY ===")
    enjoy_stats = advanced_drop_analysis(holdout_u_enjoy, 'util_enjoyment', 'prediction')
    print(enjoy_stats.to_string(index=False))

    print("\n=== DROP ANALYSIS: USEFULNESS UTILITY ===")
    useful_stats = advanced_drop_analysis(holdout_u_useful, 'util_usefulness', 'prediction')
    print(useful_stats.to_string(index=False))

    # Save detailed predictions
    holdout_u_enjoy.to_csv('ai_books_tracking/UTILITY_ENJOY_HOLDOUT.csv', index=False)
    holdout_u_useful.to_csv('ai_books_tracking/UTILITY_USEFUL_HOLDOUT.csv', index=False)

if __name__ == "__main__":
    main()
