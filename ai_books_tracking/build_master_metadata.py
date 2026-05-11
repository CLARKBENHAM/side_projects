
import pandas as pd
import os
import re
from pathlib import Path

def clean_filename_author(filename):
    # Try to extract author from filename using patterns like "Title-Author.epub" or "Title by Author"
    if not isinstance(filename, str): return None
    
    # Pattern: Title - Author - Extra
    match = re.search(r' - ([\w\.\s\']+)\s*-\s*', filename)
    if match: return match.group(1).strip()
    
    # Pattern: Title by Author
    match = re.search(r' by ([\w\.\s\']+)', filename, re.IGNORECASE)
    if match:
        author = match.group(1).strip()
        # Remove file extension if it was caught
        author = re.sub(r'\.(pdf|epub|mobi|txt|html)$', '', author, flags=re.IGNORECASE)
        return author
        
    return None

def main():
    data_dir = Path('data')
    
    # 1. Collect all filenames from Takeout folders
    takeout_files = []
    for pt in [1, 2, 3]:
        pt_dir = data_dir / f'Takeout_Play_books_03_16_25_pt{pt}'
        if pt_dir.exists():
            for root, dirs, files in os.walk(pt_dir):
                for f in files:
                    if f.lower().endswith(('.pdf', '.epub', '.mobi', '.txt', '.html')):
                        takeout_files.append(f)
    
    # 2. Collect from dropbox_old_reading_list.txt
    dropbox_mappings = {}
    dropbox_path = data_dir / 'dropbox_old_reading_list.txt'
    if dropbox_path.exists():
        with open(dropbox_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ',' in line and '.' in line: # Likely a file line
                    parts = line.split(',')
                    filename = parts[0].strip()
                    # The second part is often a date
                    dropbox_mappings[filename] = filename # We just want the filename for now
                    
    # 3. Load Play Export and Ratings 2
    play_export = pd.read_csv(data_dir / 'Books Read and their effects - Play Export.csv')
    ratings_2 = pd.read_csv(data_dir / 'Books Read and their effects - Ratings 2.csv')
    
    # 4. Load Holdout (2026)
    holdout = pd.read_csv(data_dir / 'Books Read and their effects - new_books_to_rate 2026.csv')
    
    # 5. Build Master List
    # We want Title, Original Author, Filename (if known), Source
    master = []
    
    # From Play Export
    for _, row in play_export.iterrows():
        master.append({
            'title': row['title'],
            'author': row['author'],
            'filename': row['filename'],
            'source': 'Play Export'
        })
        
    # From Ratings 2
    for _, row in ratings_2.iterrows():
        master.append({
            'title': row['title'],
            'author': row['author'],
            'filename': row['filename'],
            'source': 'Ratings 2'
        })
        
    # From Holdout
    for _, row in holdout.iterrows():
        master.append({
            'title': row['title'],
            'author': row['author'],
            'filename': None, # Holdout doesn't have filename column
            'source': 'Holdout 2026'
        })
        
    df_master = pd.DataFrame(master)
    
    # Deduplicate by Title and Author
    df_master = df_master.drop_duplicates(subset=['title', 'author'])
    
    # Add filenames from takeout and dropbox if missing
    # This is fuzzy but better than nothing
    all_known_filenames = set(takeout_files) | set(dropbox_mappings.keys())
    
    def find_filename(title):
        title_clean = re.sub(r'[^\w\s]', '', str(title).lower())
        for f in all_known_filenames:
            f_clean = re.sub(r'[^\w\s]', '', f.lower())
            if title_clean in f_clean or f_clean in title_clean:
                return f
        return None
        
    df_master.loc[df_master['filename'].isna(), 'filename'] = df_master[df_master['filename'].isna()]['title'].apply(find_filename)
    
    # Output for LLM correction
    df_master.to_csv('ai_books_tracking/master_book_metadata_raw.csv', index=False)
    print(f"Created master list with {len(df_master)} unique books.")

if __name__ == "__main__":
    main()
