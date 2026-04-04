import urllib.request
import xml.etree.ElementTree as ET
import csv
import time
import math

base_url = "https://www.goodreads.com/review/list_rss/11004626?shelf=read&page={}"

books = []
page = 1
print("Fetching Goodreads RSS feed for Gwern...")

while True:
    url = base_url.format(page)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        xml_data = response.read()
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
        break

    root = ET.fromstring(xml_data)
    items = root.findall('.//item')
    
    if not items:
        break
        
    for item in items:
        title = item.find('title').text
        user_rating = int(item.find('user_rating').text)
        avg_rating = float(item.find('average_rating').text)
        
        category = 'General Reading'
        shelves_elem = item.find('user_shelves')
        if shelves_elem is not None and shelves_elem.text:
            shelves = shelves_elem.text.lower()
            if any(s in shelves for s in ['fiction', 'sci-fi', 'fantasy', 'literature', 'poetry', 'manga', 'novel']):
                category = 'Fiction'
            elif any(s in shelves for s in ['math', 'computer', 'science', 'tech', 'programming']):
                category = 'Computer Science' # Technically maps to 'Technical_Other' in predict.js
            elif any(s in shelves for s in ['business', 'history', 'economics', 'biography', 'psychology']):
                category = 'Business'
        
        # Only include books he has actually rated (1-5)
        if user_rating > 0:
            books.append({
                'title': title,
                'avg_enjoyment': user_rating,
                'goodreads_rating': avg_rating,
                'category': category
            })
            
    # print(f"Page {page} done, total rated books so far: {len(books)}")
    page += 1
    time.sleep(1)

def mean(vals):
    if not vals: return 0
    return sum(vals)/len(vals)

def corr(x, y):
    mx = mean(x)
    my = mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y))
    return num / den if den != 0 else 0

x = [b['goodreads_rating'] for b in books]
y = [b['avg_enjoyment'] for b in books]
correlation = corr(x, y)

print(f"\n--- Analysis ---")
print(f"Total books rated by Gwern: {len(books)}")
print(f"Correlation between Goodreads avg and Gwern's rating: {correlation:.4f}")

baseline = mean(y)
print(f"Gwern's average rating (Baseline): {baseline:.4f}")

# Sort by goodreads rating ascending
sorted_by_gr = sorted(books, key=lambda b: b['goodreads_rating'])

print("\nTheoretical Improvement by dropping bottom X% (using raw Goodreads filtering):")
for pct in range(5, 55, 5):
    drop_count = int(len(sorted_by_gr) * (pct / 100))
    kept_avg = mean([b['avg_enjoyment'] for b in sorted_by_gr[drop_count:]])
    print(f"Drop bottom {pct:2d}% -> Kept Avg: {kept_avg:.4f} (Rating Gain: +{kept_avg - baseline:.4f})")

csv_file = "gwern_books.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['title', 'avg_enjoyment', 'goodreads_rating', 'amazon_rating', 'category'])
    writer.writeheader()
    for b in books:
        b['amazon_rating'] = ''
        if not b.get('category'):
            b['category'] = 'General Reading'
        # Sanitize html in title if any
        if b['title']:
            b['title'] = b['title'].replace('<![CDATA[', '').replace(']]>', '').strip()
        writer.writerow(b)

print(f"\nExported {len(books)} rows to {csv_file}")
