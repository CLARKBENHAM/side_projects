
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

def main():
    # 1. Load the Golden Master
    path = Path('data/Books Read and their effects - master_book_metadata_cleaned.csv')
    df = pd.read_csv(path)
    
    # 2. Extract and Clean Columns
    # Goodreads (Truth)
    df['gr_rating'] = pd.to_numeric(df['goodread ratings'], errors='coerce')
    
    # Amazon Ratings
    # Based on: 'Amazon combined with links', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Amazon no links', 'amazon ratings', 'amzon number reviews'
    # Column indices from header:
    # 14: Amazon combined with links (4.8|153494|URL)
    # 15: Unnamed: 15 (4.8)
    # 16: Unnamed: 16 (153494)
    # 21: Amazon no links (4.8|153494)
    # 22: amazon ratings (4.8)
    # 23: amzon number reviews (153494)
    
    df['amz_r1'] = pd.to_numeric(df['amazon ratings'], errors='coerce')
    df['amz_r2'] = pd.to_numeric(df['Unnamed: 15'], errors='coerce')
    
    # Open Library Ratings
    # 10: open library combined (4.2|185|URL)
    # 11: open library ratings (4.2)
    # 12: open library number reviews (185)
    # 18: Open LIbrary No links (4.2|185)
    # 19: open library rating (4.2)
    # 20: open library num reviews (185)
    
    df['ol_r1'] = pd.to_numeric(df['open library ratings'], errors='coerce')
    df['ol_r2'] = pd.to_numeric(df['open library rating'], errors='coerce')
    
    # 3. Combine Ratings (Average of available)
    def combine_ratings(row, r1_col, r2_col):
        v1 = row[r1_col]
        v2 = row[r2_col]
        if pd.notna(v1) and pd.notna(v2):
            return (v1 + v2) / 2
        elif pd.notna(v1):
            return v1
        elif pd.notna(v2):
            return v2
        return np.nan

    df['amz_combined'] = df.apply(lambda r: combine_ratings(r, 'amz_r1', 'amz_r2'), axis=1)
    df['ol_combined'] = df.apply(lambda r: combine_ratings(r, 'ol_r1', 'ol_r2'), axis=1)
    
    # 4. Linear Regression Imputation
    # We impute Amazon and OL from Goodreads
    gr_valid = df[df['gr_rating'].notna()].copy()
    
    # Amazon Imputation
    amz_model_set = gr_valid[gr_valid['amz_combined'].notna()]
    if len(amz_model_set) > 5:
        amz_reg = LinearRegression().fit(amz_model_set[['gr_rating']], amz_model_set['amz_combined'])
        df['amz_predicted'] = amz_reg.predict(df[['gr_rating']].fillna(df['gr_rating'].mean()))
        df['amz_final'] = df['amz_combined'].fillna(df['amz_predicted'])
        print(f"Amazon Regression: slope={amz_reg.coef_[0]:.3f}, intercept={amz_reg.intercept_:.3f}, N={len(amz_model_set)}")
    else:
        df['amz_final'] = df['amz_combined']
        
    # Open Library Imputation
    ol_model_set = gr_valid[gr_valid['ol_combined'].notna()]
    if len(ol_model_set) > 5:
        ol_reg = LinearRegression().fit(ol_model_set[['gr_rating']], ol_model_set['ol_combined'])
        df['ol_predicted'] = ol_reg.predict(df[['gr_rating']].fillna(df['gr_rating'].mean()))
        df['ol_final'] = df['ol_combined'].fillna(df['ol_predicted'])
        print(f"Open Library Regression: slope={ol_reg.coef_[0]:.3f}, intercept={ol_reg.intercept_:.3f}, N={len(ol_model_set)}")
    else:
        df['ol_final'] = df['ol_combined']

    # 5. Outlier/Error Analysis
    df['amz_diff'] = (df['amz_combined'] - df['amz_predicted']).abs()
    df['ol_diff'] = (df['ol_combined'] - df['ol_predicted']).abs()
    
    print("\n=== Outlier Analysis (Top 10 Amazon Discrepancies) ===")
    print(df[df['amz_combined'].notna()].sort_values('amz_diff', ascending=False)[['title', 'gr_rating', 'amz_combined', 'amz_predicted']].head(10).to_string(index=False))
    
    print("\n=== Outlier Analysis (Top 10 Open Library Discrepancies) ===")
    print(df[df['ol_combined'].notna()].sort_values('ol_diff', ascending=False)[['title', 'gr_rating', 'ol_combined', 'ol_predicted']].head(10).to_string(index=False))

    # 6. Save the imputed results
    df.to_csv('ai_books_tracking/imputed_ratings_audit.csv', index=False)
    print("\nSaved imputed ratings and audit to: ai_books_tracking/imputed_ratings_audit.csv")

if __name__ == "__main__":
    main()
