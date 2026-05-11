
import pandas as pd
import re
from pathlib import Path

def normalize_title(t):
    if not isinstance(t, str): return ""
    # Strip extensions and normalized
    t = t.replace('.pdf', '').replace('.epub', '').replace('.mobi', '').replace('.txt', '').replace('.html', '')
    return re.sub(r'[^\w\s]', '', t.lower()).strip()

def main():
    # 1. Paths
    gold_path = Path('data/Books Read and their effects - master_book_metadata_cleaned.csv')
    play_export_path = Path('data/Books Read and their effects - Play Export.csv')
    ratings_2_path = Path('data/Books Read and their effects - Ratings 2.csv')
    holdout_path = Path('data/Books Read and their effects - new_books_to_rate 2026.csv')
    dropbox_path = Path('data/dropbox_old_reading_list.txt')
    
    # Alias map for renames (Normalized Old Name -> Normalized New Name in Gold)
    alias_map = {
        normalize_title("buckley"): normalize_title("Buckley: The Life and the Revolution That Changed America"),
        normalize_title("David Foster Wallace"): normalize_title("David Foster Wallace the last interview"),
        normalize_title("dfw_tv"): normalize_title("e unibus pluram television and u.s. fiction"),
        normalize_title("GoodOldNeon.pdf"): normalize_title("Good Old Neon"),
        normalize_title("colinbennenttradingvolatilityall sorts of infopdf"): normalize_title("Trading Volatility: Trading Volatility, Correlation, Term Structure and Skew"),
        normalize_title("cambridge military histories dr phillips payson obrien  how the war was won_ airsea power and allied victory in world war iicambridge university press 2015"): normalize_title("How the War was Won_ Air-Sea Power and Allied Victory in World War II-Cambridge University Press (2015).pdf"),
        normalize_title("Kelly"): normalize_title("Kelly: More Than My Share of It All"),
        normalize_title("kelly my share of it all"): normalize_title("Kelly: More Than My Share of It All"),
        normalize_title("shaping up"): normalize_title("Shape Up Stop Running in Circles and Ship Work that Matters"),
        normalize_title("now it can be told"): normalize_title("Now It Can Be Told: The Story of the Manhattan Project"),
    }
    
    # 2. Load Golden Master (truth for GR and corrected authors)
    df_gold = pd.read_csv(gold_path)
    df_gold['title_norm'] = df_gold['title'].apply(normalize_title)
    
    # 3. Load all ratings sources
    df_play = pd.read_csv(play_export_path)
    df_r2 = pd.read_csv(ratings_2_path)
    df_h2026 = pd.read_csv(holdout_path)
    
    def apply_aliases(tn):
        return alias_map.get(tn, tn)

    df_play['title_norm'] = df_play['title'].apply(normalize_title).apply(apply_aliases)
    df_r2['title_norm'] = df_r2['title'].apply(normalize_title).apply(apply_aliases)
    df_h2026['title_norm'] = df_h2026['title'].apply(normalize_title).apply(apply_aliases)
    
    # 4. Extract Timing and Categories from Dropbox
    # Format: filename, date. Headers are categories.
    dropbox_data = []
    current_cat = "Unknown"
    if dropbox_path.exists():
        with open(dropbox_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if ',' not in line:
                    current_cat = line
                else:
                    parts = line.split(',')
                    fname = parts[0].strip()
                    dt = parts[1].strip() if len(parts) > 1 else None
                    dropbox_data.append({
                        'filename': fname,
                        'dropbox_date': dt,
                        'dropbox_category': current_cat,
                        'title_norm': normalize_title(fname.replace('.pdf', '').replace('.epub', '').replace('.mobi', ''))
                    })
    df_dropbox = pd.DataFrame(dropbox_data)
    # Deduplicate dropbox by title_norm
    df_dropbox = df_dropbox.drop_duplicates(subset='title_norm')

    # 5. Build the Unified Ratings Map
    # We want to collect all ratings for a given normalized title
    ratings_map = {}
    
    def add_to_map(df, enjoy_col, use_col, source_name, cat_col='Bookshelf'):
        for _, row in df.iterrows():
            tn = row['title_norm']
            if tn not in ratings_map:
                ratings_map[tn] = {'sources': []}
            ratings_map[tn]['sources'].append(source_name)
            if enjoy_col in row and pd.notna(row[enjoy_col]):
                ratings_map[tn][f'enjoyment_{source_name}'] = row[enjoy_col]
            if use_col in row and pd.notna(row[use_col]):
                ratings_map[tn][f'usefulness_{source_name}'] = row[use_col]
            if cat_col in row and pd.notna(row[cat_col]):
                ratings_map[tn][f'category_{source_name}'] = row[cat_col]
            # Add dates if available
            for dcol in ['earliest_modified', 'latest_modified', 'date_finished']:
                if dcol in row and pd.notna(row[dcol]):
                    ratings_map[tn][f'{dcol}_{source_name}'] = row[dcol]

    add_to_map(df_play, 'Enjoyment (/5)', 'Usefulness /5 to Me', 'play')
    add_to_map(df_r2, 'Enjoyment (/5)', 'Usefulness /5 to Me', 'r2')
    add_to_map(df_h2026, 'Enjoyment (/5)', 'Usefulness /5 to Me', 'h2026')
    add_to_map(df_h2026, 'Enjoyment (/5) 2nd', 'Usefulness /5 to Me.1', 'h2026_2nd')

    # 6. Assemble the Final Golden Master
    final_rows = []
    seen_norms = set()
    
    for _, row in df_gold.iterrows():
        tn = row['title_norm']
        seen_norms.add(tn)
        
        # Base from gold
        new_row = {
            'title': row['title'],
            'corrected_author': row['corrected_author'],
            'gr_rating': row['goodread ratings'],
            'gr_count': row['goodreads number ratings'],
            'amz_rating': row['amazon ratings'],
            'amz_count': row['amzon number reviews'],
            'ol_rating': row['open library ratings'],
            'ol_count': row['open library number reviews'],
            'filename': row['filename'],
            'source_original': row['source']
        }
        
        # Add ratings and categories
        if tn in ratings_map:
            rm = ratings_map[tn]
            new_row.update({k: v for k, v in rm.items() if k != 'sources'})
            
        # Add dropbox info
        db_match = df_dropbox[df_dropbox['title_norm'] == tn]
        if not db_match.empty:
            new_row['dropbox_date'] = db_match.iloc[0]['dropbox_date']
            new_row['dropbox_category'] = db_match.iloc[0]['dropbox_category']
            
        # Unified Category logic: Prioritize dropbox, then h2026, then r2, then play
        cat_order = ['dropbox_category', 'category_h2026', 'category_r2', 'category_play']
        final_cat = None
        for c in cat_order:
            if c in new_row and pd.notna(new_row[c]):
                final_cat = new_row[c]
                break
        new_row['final_category'] = final_cat or "Unknown"
            
        final_rows.append(new_row)
        
    # Check for books in ratings but not in gold (ambiguous)
    ambiguous = []
    for tn, data in ratings_map.items():
        if tn not in seen_norms:
            ambiguous.append({'title_norm': tn, 'data': data})
            
    # 7. Final Polish
    df_final = pd.DataFrame(final_rows)
    
    # Save the master
    out_path = Path('ai_books_tracking/FINAL_CONSOLIDATED_MASTER.csv')
    df_final.to_csv(out_path, index=False)
    
    print(f"Created Golden Master with {len(df_final)} books.")
    print(f"Found {len(ambiguous)} ambiguous books not in the gold standard.")
    
    if ambiguous:
        print("\nTOP AMBIGUOUS (Needs Correction):")
        for a in ambiguous[:5]:
            print(f"- {a['title_norm']} (Sources: {a['data']['sources']})")
            
    # Save ambiguous for correction
    pd.DataFrame([{'title_norm': a['title_norm'], 'sources': a['data']['sources']} for a in ambiguous]).to_csv('ai_books_tracking/AMBIGUOUS_BOOKS.csv', index=False)

if __name__ == "__main__":
    main()
