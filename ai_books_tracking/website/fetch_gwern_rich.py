import urllib.request
import xml.etree.ElementTree as ET
import csv
import time
import math
import datetime

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
        link = item.find('link').text
        
        # Read year
        read_at = item.find('user_read_at').text
        added_at = item.find('user_date_added').text
        year_str = read_at if read_at else added_at
        year = ''
        if year_str:
            # Format: 'Sun, 15 Feb 2026 08:22:14 -0800'
            try:
                # Extract year using simple string splitting, it's always the 4th token
                parts = year_str.split(' ')
                year = parts[3]
            except:
                pass

        category = 'General Reading'
        shelves_elem = item.find('user_shelves')
        if shelves_elem is not None and shelves_elem.text:
            shelves = shelves_elem.text.lower()
            if any(s in shelves for s in ['fiction', 'sci-fi', 'fantasy', 'literature', 'poetry', 'manga', 'novel']):
                category = 'Fiction'
            elif any(s in shelves for s in ['math', 'computer', 'science', 'tech', 'programming']):
                category = 'Computer Science'
            elif any(s in shelves for s in ['business', 'history', 'economics', 'biography', 'psychology']):
                category = 'Business'
        
        if user_rating > 0:
            books.append({
                'title': title,
                'avg_enjoyment': user_rating,
                'goodreads_rating': avg_rating,
                'category': category,
                'read_year': year,
                'link': link
            })
            
    # print(f"Page {page} done, total {len(books)} books")
    page += 1
    time.sleep(1)

csv_file = "gwern_books_rich.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['title', 'avg_enjoyment', 'goodreads_rating', 'category', 'read_year', 'link'])
    writer.writeheader()
    for b in books:
        if b['title']:
            b['title'] = b['title'].replace('<![CDATA[', '').replace(']]>', '').strip()
        writer.writerow(b)

print(f"Exported {len(books)} rows to {csv_file}")
