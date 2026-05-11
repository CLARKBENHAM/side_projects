
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

def main():
    # 1. Load Data
    path = Path('ai_books_tracking/FINAL_CONSOLIDATED_MASTER.csv')
    if not path.exists():
        print(f"Error: {path} not found. Run build_final_golden_master.py first.")
        return
    df = pd.read_csv(path)

    # 2. Target Engineering (Average Enjoyment and Usefulness)
    def get_avg_target(row, target_prefix):
        if row['source_original'] == 'Holdout 2026':
            vals = [row.get(f'{target_prefix}_h2026'), row.get(f'{target_prefix}_h2026_2nd')]
        else:
            # Map user column names to prefixes
            enjoy_map = {'enjoyment': ['enjoyment_play', 'enjoyment_r2'],
                         'usefulness': ['usefulness_play', 'usefulness_r2']}
            vals = [row.get(c) for c in enjoy_map[target_prefix]]
        
        valid_vals = [pd.to_numeric(v, errors='coerce') for v in vals if pd.notna(v)]
        return np.mean(valid_vals) if valid_vals else np.nan

    df['avg_enjoyment'] = df.apply(lambda r: get_avg_target(r, 'enjoyment'), axis=1)
    df['avg_usefulness'] = df.apply(lambda r: get_avg_target(r, 'usefulness'), axis=1)

    # 3. Feature Engineering
    df['gr_rating'] = pd.to_numeric(df['gr_rating'], errors='coerce')
    df['amz_rating'] = pd.to_numeric(df['amz_rating'], errors='coerce')
    
    # Log counts
    df['log_gr_count'] = np.log10(pd.to_numeric(df['gr_count'], errors='coerce') + 1)
    df['log_amz_count'] = np.log10(pd.to_numeric(df['amz_count'], errors='coerce') + 1)
    
    # Categories (Handle NAs)
    df['category'] = df['dropbox_category'].fillna('Unknown')
    
    # 4. Imputation
    df['gr_rating'] = df['gr_rating'].fillna(df['gr_rating'].mean())
    df['amz_rating'] = df['amz_rating'].fillna(df['gr_rating'])
    df['log_gr_count'] = df['log_gr_count'].fillna(df['log_gr_count'].median())
    df['log_amz_count'] = df['log_amz_count'].fillna(df['log_amz_count'].median())

    # 5. Split Data
    train_mask = df['source_original'] != 'Holdout 2026'
    holdout_mask = df['source_original'] == 'Holdout 2026'
    
    numerical_cols = ['gr_rating', 'amz_rating', 'log_gr_count', 'log_amz_count']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    def run_target_model(target_name):
        df_target = df[df[target_name].notna()].copy()
        df_train = df_target[df_target['source_original'] != 'Holdout 2026'].copy()
        df_holdout = df_target[df_target['source_original'] == 'Holdout 2026'].copy()
        
        cat_train = encoder.fit_transform(df_train[['category']])
        cat_holdout = encoder.transform(df_holdout[['category']])
        cat_names = encoder.get_feature_names_out(['category'])
        
        X_train = np.hstack([df_train[numerical_cols].values, cat_train])
        y_train = df_train[target_name].values
        X_holdout = np.hstack([df_holdout[numerical_cols].values, cat_holdout])
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        df_train['prediction'] = model.predict(X_train)
        df_holdout['prediction'] = model.predict(X_holdout)
        
        # Also predict for the whole dataset for the drop analysis
        cat_all = encoder.transform(df[['category']])
        X_all = np.hstack([df[numerical_cols].values, cat_all])
        df[f'pred_{target_name}'] = model.predict(X_all)
        
        print(f"\n=== {target_name.upper()} COEFFICIENTS ===")
        features = numerical_cols + list(cat_names)
        coef_df = pd.DataFrame({'feature': features, 'coefficient': model.coef_})
        print(coef_df.to_string(index=False))
        
        return df_train, df_holdout

    train_enjoy, holdout_enjoy = run_target_model('avg_enjoyment')
    train_useful, holdout_useful = run_target_model('avg_usefulness')

    # Save for plotting
    train_enjoy.to_csv('ai_books_tracking/TRAIN_ENJOYMENT_PREDS.csv', index=False)
    holdout_enjoy.to_csv('ai_books_tracking/HOLDOUT_ENJOYMENT_PREDS.csv', index=False)
    train_useful.to_csv('ai_books_tracking/TRAIN_USEFULNESS_PREDS.csv', index=False)
    holdout_useful.to_csv('ai_books_tracking/HOLDOUT_USEFULNESS_PREDS.csv', index=False)
    
    # Global Preds for dropping analysis
    df.to_csv('ai_books_tracking/ALL_BOOKS_PREDICTIONS.csv', index=False)

if __name__ == "__main__":
    main()
