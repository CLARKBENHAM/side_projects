
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder
import re

def normalize_title(t):
    if not isinstance(t, str): return ""
    t = t.replace('.pdf', '').replace('.epub', '').replace('.mobi', '').replace('.txt', '').replace('.html', '')
    return re.sub(r'[^\w\s]', '', t.lower()).strip()

def main():
    # 1. Load Data
    master_path = 'ai_books_tracking/FINAL_CONSOLIDATED_MASTER.csv'
    df = pd.read_csv(master_path)
    
    # Calculate target utilities
    def get_avg(row, prefix):
        vals = [v for k, v in row.items() if k.startswith(prefix)]
        valid_vals = [pd.to_numeric(v, errors='coerce') for v in vals if pd.notna(v)]
        return np.mean(valid_vals) if valid_vals else np.nan

    df['avg_enjoyment'] = df.apply(lambda r: get_avg(r, 'enjoyment_'), axis=1)
    df['avg_usefulness'] = df.apply(lambda r: get_avg(r, 'usefulness_'), axis=1)
    
    df['util_enjoy'] = np.power(np.maximum(0, df['avg_enjoyment'] - 1), 1.3)
    df['util_useful'] = np.power(np.maximum(0, df['avg_usefulness'] - 1), 1.8)

    # Clean Features
    df['gr_rating'] = pd.to_numeric(df['gr_rating'], errors='coerce')
    df['amz_rating'] = pd.to_numeric(df['amz_rating'], errors='coerce')
    df['ol_rating'] = pd.to_numeric(df['ol_rating'], errors='coerce')
    df['log_gr_count'] = np.log10(pd.to_numeric(df['gr_count'], errors='coerce') + 1)
    df['log_amz_count'] = np.log10(pd.to_numeric(df['amz_count'], errors='coerce') + 1)
    # Use 'source_original' to distinguish train/holdout
    df['is_holdout'] = df['source_original'] == 'Holdout 2026'
    
    # Determine Categories (using 'source_original' categories if dropbox missing)
    # Actually, let's just use a fixed set of major categories to keep columns manageable
    df['final_category'] = df['dropbox_category'].fillna('Other')
    major_cats = df['final_category'].value_counts().head(6).index.tolist()
    df.loc[~df['final_category'].isin(major_cats), 'final_category'] = 'Other'
    categories = sorted(df['final_category'].unique())

    def plot_small_multiples(df_subset, target_col, title_prefix, filename, use_imputation=False):
        # Impute if requested
        data = df_subset.copy()
        if use_imputation:
            data['gr_rating'] = data['gr_rating'].fillna(data['gr_rating'].mean())
            data['amz_rating'] = data['amz_rating'].fillna(data['gr_rating'])
            data['ol_rating'] = data['ol_rating'].fillna(data['gr_rating'])
            data['log_gr_count'] = data['log_gr_count'].fillna(data['log_gr_count'].median())
            data['log_amz_count'] = data['log_amz_count'].fillna(data['log_amz_count'].median())
        else:
            # Complete cases only
            data = data[data['gr_rating'].notna() & data['amz_rating'].notna() & data['ol_rating'].notna()]

        data = data[data[target_col].notna()]
        if len(data) < 10: 
            print(f"Skipping {filename}: too few rows ({len(data)})")
            return

        # Train Model on Train Set, Predict on both
        train = data[~data['is_holdout']]
        holdout = data[data['is_holdout']]
        
        if len(train) < 5: return

        numerical_cols = ['gr_rating', 'amz_rating', 'ol_rating', 'log_gr_count', 'log_amz_count']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_train = encoder.fit_transform(train[['final_category']])
        X_train = np.hstack([train[numerical_cols].values, cat_train])
        y_train = train[target_col].values
        
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        
        train['prediction'] = model.predict(X_train)
        if len(holdout) > 0:
            cat_holdout = encoder.transform(holdout[['final_category']])
            X_holdout = np.hstack([holdout[numerical_cols].values, cat_holdout])
            holdout['prediction'] = model.predict(X_holdout)

        # Plot
        n_rows = 2
        n_cols = len(categories)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 8), sharex=True, sharey=True)
        fig.suptitle(f'{title_prefix} (Target: {target_col})', fontsize=16)

        sets = [(train, "Training"), (holdout, "Holdout")]
        
        for r_idx, (d_set, set_name) in enumerate(sets):
            for c_idx, cat in enumerate(categories):
                ax = axes[r_idx, c_idx]
                cat_subset = d_set[d_set['final_category'] == cat]
                
                if len(cat_subset) > 0:
                    ax.scatter(cat_subset['prediction'], cat_subset[target_col], alpha=0.6)
                    # Regression line for this specific facet
                    if len(cat_subset) > 2:
                        reg = LinearRegression().fit(cat_subset[['prediction']], cat_subset[target_col])
                        r2 = reg.score(cat_subset[['prediction']], cat_subset[target_col])
                        x_range = np.linspace(cat_subset['prediction'].min(), cat_subset['prediction'].max(), 50)
                        y_range = reg.predict(x_range.reshape(-1, 1))
                        ax.plot(x_range, y_range, 'r-', label=f'R2: {r2:.2f}')
                        ax.legend(fontsize='small')
                
                if r_idx == 0: ax.set_title(cat)
                if c_idx == 0: ax.set_ylabel(f'{set_name}\nActual Utility')
                if r_idx == 1: ax.set_xlabel('Predicted Utility')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'ai_books_tracking/{filename}')
        plt.close()

    # Generate the 4 combinations
    plot_small_multiples(df, 'util_enjoy', 'Complete Cases Only', 'sm_enjoy_complete.png', use_imputation=False)
    plot_small_multiples(df, 'util_enjoy', 'Imputed Data', 'sm_enjoy_imputed.png', use_imputation=True)
    plot_small_multiples(df, 'util_useful', 'Complete Cases Only', 'sm_useful_complete.png', use_imputation=False)
    plot_small_multiples(df, 'util_useful', 'Imputed Data', 'sm_useful_imputed.png', use_imputation=True)

    # 6. Annotated Drop Curve
    def generate_annotated_drop_curve(target_col, filename):
        # Use imputed data for full curve
        data = df[df['gr_rating'].notna() & df[target_col].notna()].copy()
        data['gr_rating'] = data['gr_rating'].fillna(data['gr_rating'].mean())
        data['amz_rating'] = data['amz_rating'].fillna(data['gr_rating'])
        data['ol_rating'] = data['ol_rating'].fillna(data['gr_rating'])
        data['log_gr_count'] = data['log_gr_count'].fillna(data['log_gr_count'].median())
        data['log_amz_count'] = data['log_amz_count'].fillna(data['log_amz_count'].median())
        
        train = data[~data['is_holdout']]
        holdout = data[data['is_holdout']]
        
        numerical_cols = ['gr_rating', 'amz_rating', 'ol_rating', 'log_gr_count', 'log_amz_count']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_train = encoder.fit_transform(train[['final_category']])
        X_train = np.hstack([train[numerical_cols].values, cat_train])
        y_train = train[target_col].values
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        
        cat_holdout = encoder.transform(holdout[['final_category']])
        X_holdout = np.hstack([holdout[numerical_cols].values, cat_holdout])
        holdout['prediction'] = model.predict(X_holdout)
        
        holdout_sorted = holdout.sort_values('prediction', ascending=False)
        n = len(holdout_sorted)
        
        x_axis, y_axis, labels = [], [], []
        for p in [0, 10, 25, 43, 50, 60, 75, 80, 90, 95]:
            n_keep = max(1, int(n * (1 - p/100)))
            subset = holdout_sorted.head(n_keep)
            avg_util = subset[target_col].mean()
            # The rating of the book at this percentile
            threshold_idx = min(n-1, int(n * (1 - p/100)))
            # We'll use the GR rating as the human-readable 'rating' label
            threshold_rating = holdout_sorted.iloc[threshold_idx]['gr_rating']
            
            x_axis.append(p)
            y_axis.append(avg_util)
            labels.append(f'{threshold_rating:.1f}')

        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, y_axis, 'b-o')
        for i, txt in enumerate(labels):
            plt.annotate(txt, (x_axis[i], y_axis[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.xlabel('Percentage of Books Dropped')
        plt.ylabel(f'Avg {target_col} of Kept Books')
        plt.title(f'Annotated Drop Curve (Holdout Set) - Labels: GR Rating Cutoff')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'ai_books_tracking/{filename}')
        plt.close()

    generate_annotated_drop_curve('util_enjoy', 'plot_drop_enjoy_annotated.png')
    generate_annotated_drop_curve('util_useful', 'plot_drop_useful_annotated.png')

    print("Small multiples and annotated curves generated.")

if __name__ == "__main__":
    main()
