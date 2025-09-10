#!/usr/bin/env python3

import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import unquote
from pathlib import Path
from datetime import datetime
import unicodedata

class ArxivDownloader:
    def __init__(self, base_dir="~/Downloads/Papers"):
        self.base_dir = Path(base_dir).expanduser()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
    def sanitize_filename(self, filename):
        """Remove special characters and limit length"""
        filename = unicodedata.normalize('NFKD', filename)
        filename = re.sub(r'[^\w\s\-\.]', '', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        return filename[:200]  # Reasonable length limit
    
    def extract_arxiv_id(self, url):
        """Extract arxiv ID from various URL formats"""
        patterns = [
            r'arxiv\.org/abs/([^/?]+)',
            r'arxiv\.org/pdf/([^/?]+)',
            r'arxiv\.org/html/([^/?]+)',
            r'arxiv\.org/[^/]+/([^/?]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                arxiv_id = match.group(1)
                if arxiv_id.endswith('.pdf'):
                    arxiv_id = arxiv_id[:-4]
                return arxiv_id
        return None
    
    def get_paper_metadata(self, arxiv_id):
        """Fetch paper metadata from arxiv API"""
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        
        try:
            time.sleep(3)  # Rate limiting - arxiv requests 3 sec between calls
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            
            if entry is None:
                return None
                
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            title = re.sub(r'\s+', ' ', title)
            
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
            
            return {
                'title': title,
                'published': pub_date,
                'date_str': pub_date.strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"Error fetching metadata for {arxiv_id}: {e}")
            return None
    
    def download_paper(self, arxiv_id, metadata, folder_path):
        """Download paper PDF"""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        title = self.sanitize_filename(metadata['title'])
        filename = f"{metadata['date_str']} - {title}.pdf"
        filepath = folder_path / filename
        
        if filepath.exists():
            print(f"Already exists: {filename}")
            return True
            
        try:
            print(f"Downloading: {filename}")
            time.sleep(2)  # Additional rate limiting for downloads
            
            response = self.session.get(pdf_url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {arxiv_id}: {e}")
            return False
    
    def parse_reading_list(self, html_file_path):
        """Parse Chrome reading list HTML export"""
        html_path = Path(html_file_path)
        if not html_path.exists():
            raise FileNotFoundError(f"Reading list file not found: {html_file_path}")
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Decode URL-encoded path if needed
        content = unquote(content)
        
        soup = BeautifulSoup(content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        arxiv_urls = []
        for link in links:
            href = link['href']
            if 'arxiv.org' in href:
                arxiv_urls.append(href)
        
        print(f"Found {len(arxiv_urls)} arxiv links in reading list")
        return arxiv_urls
    
    def hardcode_urls(self):
        return ["https://arxiv.org/html/2506.12928v1",
                "https://arxiv.org/html/2410.11163v2",
                "https://arxiv.org/html/2502.04510v1",
                "https://arxiv.org/html/2406.07155v3",
                "https://arxiv.org/html/2402.05120v2",
                "https://arxiv.org/html/2503.05473v1#:~:text=learning.%20%20nature%2C%20518%287540%29%3A529%E2%80%93533.%20,Cross",
                "https://arxiv.org/html/2503.05473v1",
                "https://arxiv.org/html/2402.01680v2",
                "https://arxiv.org/html/2402.03578v2",
                "https://arxiv.org/html/2403.17636v2",
                "https://arxiv.org/html/2505.23433v1",
                "https://arxiv.org/abs/2405.02345",
                "https://arxiv.org/abs/2406.04692",
                "https://arxiv.org/html/2501.06322v1",
                "https://arxiv.org/html/2506.00066v1",
                "https://arxiv.org/html/2506.10910v1",
                "https://arxiv.org/pdf/2412.19437",
                "https://arxiv.org/abs/2405.04434",
                "https://arxiv.org/pdf/2501.12948",
                "https://arxiv.org/abs/2503.11486",
                ]

    def process_papers(self, html_file_path):
        """Main processing function"""
        # urls = self.parse_reading_list(html_file_path)
        urls = self.hardcode_urls() # self.parse_reading_list(html_file_path)
        
        # Extract arxiv IDs and get metadata
        papers_data = []
        print("Fetching paper metadata...")
        
        for i, url in enumerate(urls, 1):
            arxiv_id = self.extract_arxiv_id(url)
            if not arxiv_id:
                print(f"Could not extract arxiv ID from: {url}")
                continue
                
            print(f"Processing {i}/{len(urls)}: {arxiv_id}")
            metadata = self.get_paper_metadata(arxiv_id)
            
            if metadata:
                papers_data.append({
                    'arxiv_id': arxiv_id,
                    'url': url,
                    'metadata': metadata
                })
            else:
                print(f"Could not fetch metadata for: {arxiv_id}")
        
        # Sort by publication date (most recent first)
        papers_data.sort(key=lambda x: x['metadata']['published'], reverse=True)
        
        # Create folders
        reading_list_dir = self.base_dir / "Reading List Agents"
        temp_dir = self.base_dir / "temp"
        
        reading_list_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading {len(papers_data)} papers...")
        print(f"Reading List folder: {reading_list_dir}")
        print(f"Temp folder: {temp_dir}")
        
        # Download papers (all to Reading List for now - adjust logic if needed)
        successful = 0
        for i, paper in enumerate(papers_data, 1):
            print(f"\n[{i}/{len(papers_data)}]")
            if self.download_paper(paper['arxiv_id'], paper['metadata'], reading_list_dir):
                successful += 1
        
        print(f"\n✓ Successfully downloaded {successful}/{len(papers_data)} papers")
        print(f"Papers saved to: {reading_list_dir}")
        
        # Print summary sorted by date
        print(f"\n📚 Paper Summary (sorted by date, newest first):")
        for paper in papers_data:
            meta = paper['metadata']
            print(f"{meta['date_str']} - {meta['title'][:80]}...")

def main():
    downloader = ArxivDownloader()
    html_file = "/Users/clarkbenham/Downloads/Takeout 4/Chrome/Reading List.html"
    
    try:
        downloader.process_papers(html_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
