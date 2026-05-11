
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_actual_vs_pred(df, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['prediction'], df['avg_enjoyment'], alpha=0.5)
    
    # Diagonal line
    lims = [1, 5]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    
    plt.xlabel('Estimated Enjoyment')
    plt.ylabel('Actual Enjoyment')
    plt.title(f'Actual vs Estimated Ratings: {title}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'ai_books_tracking/{filename}')
    plt.close()

def drop_analysis(df, target_col, pred_col):
    df_sorted = df.sort_values(pred_col, ascending=False).copy()
    n = len(df_sorted)
    
    results = []
    # Drop percentages from 0 to 90%
    for p in range(0, 95, 5):
        n_keep = max(1, int(n * (1 - p/100)))
        kept = df_sorted.head(n_keep)
        
        results.append({
            'drop_pct': p,
            'avg_enjoyment': kept[target_col].mean(),
            'n_kept': n_keep
        })
    return pd.DataFrame(results)

def main():
    # 1. Load predicted data
    train_enjoy = pd.read_csv('ai_books_tracking/TRAIN_ENJOYMENT_PREDS.csv')
    holdout_enjoy = pd.read_csv('ai_books_tracking/HOLDOUT_ENJOYMENT_PREDS.csv')
    
    train_useful = pd.read_csv('ai_books_tracking/TRAIN_USEFULNESS_PREDS.csv')
    holdout_useful = pd.read_csv('ai_books_tracking/HOLDOUT_USEFULNESS_PREDS.csv')

    # 2. Plots
    plot_actual_vs_pred(train_enjoy, 'Training Set', 'plot_train_actual_vs_pred.png')
    plot_actual_vs_pred(holdout_enjoy, 'Holdout Set (2026)', 'plot_holdout_actual_vs_pred.png')

    # 3. Drop Analysis
    # We want to see how much we gain by dropping the bottom x%
    train_stats = drop_analysis(train_enjoy, 'avg_enjoyment', 'prediction')
    holdout_stats = drop_analysis(holdout_enjoy, 'avg_enjoyment', 'prediction')
    
    # Also do for Usefulness
    train_useful_stats = drop_analysis(train_useful, 'avg_usefulness', 'prediction')
    holdout_useful_stats = drop_analysis(holdout_useful, 'avg_usefulness', 'prediction')

    print("=== DROP ANALYSIS: ENJOYMENT GAIN ===")
    print("Drop % | Train Avg | Holdout Avg")
    print("-" * 30)
    for p in [0, 10, 25, 50]:
        t_val = train_stats[train_stats['drop_pct'] == p]['avg_enjoyment'].values[0]
        h_val = holdout_stats[holdout_stats['drop_pct'] == p]['avg_enjoyment'].values[0]
        print(f"{p:5}% | {t_val:9.2f} | {h_val:11.2f}")

    print("\n=== DROP ANALYSIS: USEFULNESS GAIN ===")
    print("Drop % | Train Avg | Holdout Avg")
    print("-" * 30)
    for p in [0, 10, 25, 50]:
        t_val = train_useful_stats[train_useful_stats['drop_pct'] == p]['avg_enjoyment'].values[0] # Note: column is misnamed in drop_analysis but contains correct mean
        h_val = holdout_useful_stats[holdout_useful_stats['drop_pct'] == p]['avg_enjoyment'].values[0]
        print(f"{p:5}% | {t_val:9.2f} | {h_val:11.2f}")

    # Plot Drop Curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_stats['drop_pct'], train_stats['avg_enjoyment'], label='Train Enjoyment', marker='o')
    plt.plot(holdout_stats['drop_pct'], holdout_stats['avg_enjoyment'], label='Holdout Enjoyment', marker='s')
    plt.xlabel('Percentage of Books Dropped (Bottom Predicted)')
    plt.ylabel('Average Rating of Kept Books')
    plt.title('Impact of Selective Reading on Average Enjoyment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ai_books_tracking/plot_drop_tradeoffs.png')
    plt.close()

if __name__ == "__main__":
    main()
