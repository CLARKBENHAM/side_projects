
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
import re

def normalize_title(t):
    if not isinstance(t, str): return ""
    t = t.replace('.pdf', '').replace('.epub', '').replace('.mobi', '').replace('.txt', '').replace('.html', '')
    return re.sub(r'[^\w\s]', '', t.lower()).strip()

def main():
    # 1. Load Data
    gold_path = 'data/Books Read and their effects - master_book_metadata_cleaned.csv'
    master_path = 'ai_books_tracking/FINAL_CONSOLIDATED_MASTER.csv' # Has the historical ratings
    df_gold = pd.read_csv(gold_path)
    df_master = pd.read_csv(master_path)
    
    # Calculate avg enjoyment and usefulness in master if not already there
    # This logic matches build_final_golden_master.py
    def get_avg(row, prefix):
        # Collect all columns starting with prefix (e.g. 'enjoyment_')
        vals = [v for k, v in row.items() if k.startswith(prefix)]
        valid_vals = [pd.to_numeric(v, errors='coerce') for v in vals if pd.notna(v)]
        return np.mean(valid_vals) if valid_vals else np.nan

    df_master['avg_enjoyment'] = df_master.apply(lambda r: get_avg(r, 'enjoyment_'), axis=1)
    df_master['avg_usefulness'] = df_master.apply(lambda r: get_avg(r, 'usefulness_'), axis=1)

    # Merge them to get everything in one place
    df_gold['title_norm'] = df_gold['title'].apply(normalize_title)
    df_master['title_norm'] = df_master['title'].apply(normalize_title)
    df = pd.merge(df_gold, df_master[['title_norm', 'avg_enjoyment', 'avg_usefulness', 'dropbox_category']], on='title_norm', how='inner')
    
    # 2. Cleanup Ratings
    df['gr_rating'] = pd.to_numeric(df['goodread ratings'], errors='coerce')
    df['amz_rating'] = pd.to_numeric(df['amazon ratings'], errors='coerce')
    df['ol_rating'] = pd.to_numeric(df['open library ratings'], errors='coerce')
    df['category'] = df['dropbox_category'].fillna('Unknown')
    
    # Imputation for the prediction model (Plot 4)
    df['gr_imputed'] = df['gr_rating'].fillna(df['gr_rating'].mean())
    df['amz_imputed'] = df['amz_rating'].fillna(df['gr_imputed'])
    df['ol_imputed'] = df['ol_rating'].fillna(df['gr_imputed'])
    
    # 3. Calculate Utilities
    df['util_enjoy'] = np.power(np.maximum(0, df['avg_enjoyment'] - 1), 1.3)
    df['util_useful'] = np.power(np.maximum(0, df['avg_usefulness'] - 1), 1.8)

    def generate_2x2(target_col, target_label, filename):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Ratings vs {target_label}', fontsize=16)
        
        # Sources for first 3 plots
        sources = [
            ('gr_rating', 'Goodreads Rating', axes[0, 0]),
            ('amz_rating', 'Amazon Rating', axes[0, 1]),
            ('ol_rating', 'Open Library Rating', axes[1, 0])
        ]
        
        # Helper for regression annotation
        def add_stats(ax, x, y):
            mask = x.notna() & y.notna()
            if mask.sum() > 2:
                model = LinearRegression().fit(x[mask].values.reshape(-1, 1), y[mask])
                r2 = model.score(x[mask].values.reshape(-1, 1), y[mask])
                x_range = np.linspace(x[mask].min(), x[mask].max(), 100)
                y_pred = model.predict(x_range.reshape(-1, 1))
                ax.plot(x_range, y_pred, 'r-', alpha=0.8, label=f'Coef: {model.coef_[0]:.2f}, R2: {r2:.2f}')
                ax.legend()

        # Plot 1-3: Raw Ratings (Dropped N/As)
        for col, label, ax in sources:
            subset = df[df[col].notna() & df[target_col].notna()]
            for cat in subset['category'].unique():
                cat_data = subset[subset['category'] == cat]
                ax.scatter(cat_data[col], cat_data[target_col], alpha=0.6, label=cat)
            ax.set_xlabel(label)
            ax.set_ylabel(target_label)
            ax.set_title(f'{label} vs {target_label}')
            add_stats(ax, subset[col], subset[target_col])

        # Plot 4: Full Model Prediction (Imputed N/As, drop if no GR)
        valid_model = df[df['gr_rating'].notna() & df[target_col].notna()].copy()
        # Train simple model for this plot
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(valid_model[['category']])
        X = np.hstack([valid_model[['gr_imputed', 'amz_imputed', 'ol_imputed']].values, cat_encoded])
        y = valid_model[target_col].values
        
        reg = Ridge(alpha=1.0).fit(X, y)
        valid_model['pred_utility'] = reg.predict(X)
        
        ax = axes[1, 1]
        for cat in valid_model['category'].unique():
            cat_data = valid_model[valid_model['category'] == cat]
            ax.scatter(cat_data['pred_utility'], cat_data[target_col], alpha=0.6, label=cat)
        ax.set_xlabel('Model Predicted Utility')
        ax.set_ylabel(target_label)
        ax.set_title(f'Full Model Prediction vs {target_label}')
        add_stats(ax, valid_model['pred_utility'], valid_model[target_col])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'ai_books_tracking/{filename}')
        plt.close()

    # Generate the main 2x2s
    generate_2x2('util_enjoy', 'Enjoyment Utility', 'plot_2x2_enjoyment.png')
    generate_2x2('util_useful', 'Usefulness Utility', 'plot_2x2_usefulness.png')

    # Generate the "Dropped" plots (Cumulative Avg Utility)
    def generate_drop_curves(target_col, target_label, filename):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Selection Efficiency: Avg {target_label} after dropping bottom X%', fontsize=16)
        
        metrics = [
            ('gr_rating', 'Goodreads Rating', axes[0, 0]),
            ('amz_rating', 'Amazon Rating', axes[0, 1]),
            ('ol_rating', 'Open Library Rating', axes[1, 0]),
            ('pred_utility', 'Model Prediction', axes[1, 1])
        ]
        
        # We need predictions for everything to plot plot 4
        valid_model = df[df['gr_rating'].notna() & df[target_col].notna()].copy()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(valid_model[['category']])
        X = np.hstack([valid_model[['gr_imputed', 'amz_imputed', 'ol_imputed']].values, cat_encoded])
        y = valid_model[target_col].values
        reg = Ridge(alpha=1.0).fit(X, y)
        valid_model['pred_utility'] = reg.predict(X)

        for col, label, ax in metrics:
            # Use valid_model for all so we compare same population
            data = valid_model[valid_model[col].notna()].sort_values(col, ascending=False)
            n = len(data)
            x_axis = []
            y_axis = []
            for p in range(0, 96, 5):
                n_keep = max(1, int(n * (1 - p/100)))
                avg_util = data.head(n_keep)[target_col].mean()
                x_axis.append(p)
                y_axis.append(avg_util)
            
            ax.plot(x_axis, y_axis, 'b-o')
            ax.set_xlabel('Percentage of Books Dropped (Bottom Predicted)')
            ax.set_ylabel(f'Avg {target_label} of Kept Books')
            ax.set_title(f'Drop Curve via {label}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'ai_books_tracking/{filename}')
        plt.close()

    generate_drop_curves('util_enjoy', 'Enjoyment Utility', 'plot_2x2_drop_enjoy.png')
    generate_drop_curves('util_useful', 'Usefulness Utility', 'plot_2x2_drop_useful.png')

    print("All 2x2 plots generated successfully.")

if __name__ == "__main__":
    main()
