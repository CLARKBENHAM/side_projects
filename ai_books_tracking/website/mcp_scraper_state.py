import sys
import json
import os
import argparse

CACHE_FILE = "mcp_scraper_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def init_target_list(csv_file="gwern_scored.csv"):
    import csv
    cache = load_cache()
    if 'targets' not in cache:
        targets = []
        if os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    if r['link'] and 'goodreads.com' in r['link']:
                        targets.append(r['link'])
        cache['targets'] = targets
        cache['results'] = {}
        save_cache(cache)
        return len(targets)
    return len(cache['targets'])

def get_next(batch=5):
    cache = load_cache()
    targets = cache.get('targets', [])
    results = cache.get('results', {})
    
    pending = [t for t in targets if t not in results]
    return pending[:batch]

def save_result(url, rating_count, review_count, category):
    cache = load_cache()
    if 'results' not in cache:
        cache['results'] = {}
    
    # Try casting numericals
    try: rc = int(str(rating_count).replace(',',''))
    except: rc = 0
    try: rvc = int(str(review_count).replace(',',''))
    except: rvc = 0
        
    cache['results'][url] = {
        'ratingCount': rc,
        'reviewCount': rvc,
        'category': category
    }
    save_cache(cache)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="State and Cache manager for Gemini MCP Scraper")
    parser.add_argument('--init', action='store_true', help="Initialize cache from gwern_scored.csv")
    parser.add_argument('--next', type=int, metavar='N', help="Get next N pending urls")
    parser.add_argument('--save', nargs=4, metavar=('URL', 'RATING_COUNT', 'REVIEW_COUNT', 'CATEGORY'), help="Save a result")
    parser.add_argument('--status', action='store_true', help="Print progress status")
    
    args = parser.parse_args()
    
    if args.init:
        n = init_target_list()
        print(f"Initialized cache with {n} target URLs.")
    elif args.next:
        urls = get_next(args.next)
        for u in urls:
            print(u)
    elif args.save:
        save_result(args.save[0], args.save[1], args.save[2], args.save[3])
        print(f"Successfully cached data for {args.save[0]}")
    elif args.status:
        c = load_cache()
        t = len(c.get('targets', []))
        r = len(c.get('results', {}))
        print(f"Progress: {r} / {t} URLs scraped ({(r/t*100) if t>0 else 0:.1f}%)")
    else:
        parser.print_help()
