
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
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
    
    # Target prep
    def get_avg(row, prefix):
        vals = [v for k, v in row.items() if k.startswith(prefix)]
        valid_vals = [pd.to_numeric(v, errors='coerce') for v in vals if pd.notna(v)]
        return np.mean(valid_vals) if valid_vals else np.nan

    df['avg_enjoy'] = df.apply(lambda r: get_avg(r, 'enjoyment_'), axis=1)
    df['avg_useful'] = df.apply(lambda r: get_avg(r, 'usefulness_'), axis=1)
    df['util_enjoy'] = np.power(np.maximum(0, df['avg_enjoy'] - 1), 1.3)
    df['util_useful'] = np.power(np.maximum(0, df['avg_useful'] - 1), 1.8)

    # Feature prep
    df['gr'] = pd.to_numeric(df['gr_rating'], errors='coerce')
    df['amz'] = pd.to_numeric(df['amz_rating'], errors='coerce')
    df['ol'] = pd.to_numeric(df['ol_rating'], errors='coerce')
    df['log_gr_count'] = np.log10(pd.to_numeric(df['gr_count'], errors='coerce') + 1)
    df['log_amz_count'] = np.log10(pd.to_numeric(df['amz_count'], errors='coerce') + 1)
    df['is_holdout'] = df['source_original'] == 'Holdout 2026'
    df['category'] = df['final_category']

    # 2. Imputation for "Full Model"
    df['gr_imp'] = df['gr'].fillna(df['gr'].mean())
    df['amz_imp'] = df['amz'].fillna(df['gr_imp'])
    df['ol_imp'] = df['ol'].fillna(df['gr_imp'])
    df['log_gr_imp'] = df['log_gr_count'].fillna(df['log_gr_count'].median())
    df['log_amz_imp'] = df['log_amz_count'].fillna(df['log_amz_count'].median())

    # 3. Two-Model Approach
    # High signal vs Low signal categories
    high_signal_cats = ['Advanced Finance', 'Literature', 'Business, management', 'General Reading', 'Computer Science']
    df['is_high_signal'] = df['category'].isin(high_signal_cats)

    def run_two_model_prediction(target_col):
        # Split into high and low signal
        df_target = df[df[target_col].notna()].copy()
        
        # Train on Training set only
        train = df_target[~df_target['is_holdout']]
        holdout = df_target[df_target['is_holdout']]
        
        preds_all = pd.Series(index=df_target.index, dtype=float)
        
        numerical_cols = ['gr_imp', 'amz_imp', 'ol_imp', 'log_gr_imp', 'log_amz_imp']
        
        for signal_type in [True, False]:
            mask_train = train['is_high_signal'] == signal_type
            mask_full = df_target['is_high_signal'] == signal_type
            
            if mask_train.sum() < 5: continue
            
            t_subset = train[mask_train]
            f_subset = df_target[mask_full]
            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_train = encoder.fit_transform(t_subset[['category']])
            X_train = np.hstack([t_subset[numerical_cols].values, cat_train])
            y_train = t_subset[target_col].values
            
            model = Ridge(alpha=1.0).fit(X_train, y_train)
            
            cat_full = encoder.transform(f_subset[['category']])
            X_full = np.hstack([f_subset[numerical_cols].values, cat_full])
            preds_all.loc[mask_full] = model.predict(X_full)
            
        return preds_all

    df.loc[df['util_enjoy'].notna(), 'pred_enjoy'] = run_two_model_prediction('util_enjoy')
    df.loc[df['util_useful'].notna(), 'pred_useful'] = run_two_model_prediction('util_useful')

    # 4. Generate 2x2 Drop Curves with Labeled Percentiles
    def plot_labeled_2x2_drop(target_col, target_label, filename):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Selection Efficiency (Holdout 2026): {target_label}', fontsize=16)
        
        # We only plot holdout for the drop curve forward test
        holdout = df[df['is_holdout'] & df[target_col].notna()].copy()
        
        # Column 'pred_enjoy' or 'pred_useful'
        pred_col = 'pred_enjoy' if 'enjoy' in target_col else 'pred_useful'
        
        metrics = [
            ('gr', 'Goodreads Rating', axes[0, 0]),
            ('amz', 'Amazon Rating', axes[0, 1]),
            ('ol', 'Open Library Rating', axes[1, 0]),
            (pred_col, 'Two-Model Prediction', axes[1, 1])
        ]
        
        percentiles = [0, 10, 25, 43, 50, 60, 75, 80, 90, 95]
        
        for col, label, ax in metrics:
            data = holdout[holdout[col].notna()].sort_values(col, ascending=False)
            n = len(data)
            if n == 0: continue
            
            x_pts, y_pts, annots = [], [], []
            for p in percentiles:
                n_keep = max(1, int(n * (1 - p/100)))
                avg_util = data.head(n_keep)[target_col].mean()
                
                # The rating of the book at this cutoff
                threshold_idx = min(n-1, int(n * (1 - p/100)))
                threshold_val = data.iloc[threshold_idx][col]
                
                x_pts.append(p)
                y_pts.append(avg_util)
                annots.append(f'{threshold_val:.1f}')
                
            ax.plot(x_pts, y_pts, 'b-o')
            for i, txt in enumerate(annots):
                ax.annotate(txt, (x_pts[i], y_pts[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
            ax.set_xlabel('Percentage of Books Dropped')
            ax.set_ylabel(f'Avg {target_label}')
            ax.set_title(f'Drop via {label}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'ai_books_tracking/{filename}')
        plt.close()

    plot_labeled_2x2_drop('util_enjoy', 'Enjoyment Utility', 'plot_final_drop_enjoy.png')
    plot_labeled_2x2_drop('util_useful', 'Usefulness Utility', 'plot_final_drop_useful.png')

    # 5. Small Multiples (Facet Plots)
    def plot_sm_facets(target_col, title_prefix, filename):
        # Major categories only for legibility
        major_cats = df['category'].value_counts().head(8).index.tolist()
        subset = df[df['category'].isin(major_cats) & df[target_col].notna()].copy()
        
        pred_col = 'pred_enjoy' if 'enjoy' in target_col else 'pred_useful'
        
        n_rows = 2
        n_cols = len(major_cats)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 8), sharex=True, sharey=True)
        fig.suptitle(f'{title_prefix}: Predicted vs Actual {target_col}', fontsize=16)
        
        train = subset[~subset['is_holdout']]
        holdout = subset[subset['is_holdout']]
        
        sets = [(train, "Training"), (holdout, "Holdout")]
        
        for r_idx, (d_set, set_name) in enumerate(sets):
            for c_idx, cat in enumerate(major_cats):
                ax = axes[r_idx, c_idx]
                cat_data = d_set[d_set['category'] == cat]
                
                if len(cat_data) > 0:
                    ax.scatter(cat_data[pred_col], cat_data[target_col], alpha=0.6)
                    if len(cat_data) > 2:
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(cat_data[[pred_col]], cat_data[target_col])
                        r2 = reg.score(cat_data[[pred_col]], cat_data[target_col])
                        x_range = np.linspace(cat_data[pred_col].min(), cat_data[pred_col].max(), 20)
                        ax.plot(x_range, reg.predict(x_range.reshape(-1, 1)), 'r-', alpha=0.5)
                        ax.set_title(f'{cat}\nR2: {r2:.2f}', fontsize=10)
                    else:
                        ax.set_title(cat, fontsize=10)
                
                if c_idx == 0: ax.set_ylabel(f'{set_name}')
                if r_idx == 1: ax.set_xlabel('Predicted')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'ai_books_tracking/{filename}')
        plt.close()

    plot_sm_facets('util_enjoy', 'Two-Model Small Multiples', 'sm_final_enjoy.png')
    plot_sm_facets('util_useful', 'Two-Model Small Multiples', 'sm_final_useful.png')

    print("Final labeled 2x2s and small multiples generated.")

if __name__ == "__main__":
    main()
