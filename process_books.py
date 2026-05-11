import csv
import json
import time
import os
from ddgs import DDGS
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Ensure we have the API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

class RatingData(BaseModel):
    matched: bool
    canonical_title: str | None = None
    canonical_author: str | None = None
    rating: float | None = None
    review_count: int | None = None
    url: str | None = None
    confidence: float
    notes: str

class OverallAssessment(BaseModel):
    recommended_action: str = Field(pattern="^(keep_existing|replace_with_found_values|manual_review)$")
    why: str

class BookVerification(BaseModel):
    input_title: str
    input_author: str
    goodreads: RatingData
    open_library: RatingData
    amazon: RatingData
    overall_assessment: OverallAssessment

def search_ddg(query: str, max_results: int = 3) -> str:
    try:
        results = DDGS().text(query, max_results=max_results)
        return "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}" for r in results])
    except Exception as e:
        return f"Error searching DDG: {e}"

def process_book(row):
    title = row['title']
    author = row['author']
    source = row['source']
    category = row['category']
    gr_rating = row['goodreads_rating']
    gr_reviews = row['goodreads_review_count']
    ol_rating = row['open_library_rating_consensus']
    ol_reviews = row['open_library_review_count_consensus']
    az_rating = row['amazon_rating_consensus']
    az_reviews = row['amazon_review_count_consensus']
    issues = row['issues']
    
    # Gather info from duckduckgo
    print(f"Searching for {title} by {author}...")
    gr_context = search_ddg(f'site:goodreads.com/book/show "{title}" "{author}" rating')
    ol_context = search_ddg(f'site:openlibrary.org/works "{title}" "{author}"')
    az_context = search_ddg(f'site:amazon.com "{title}" "{author}" out of 5 stars')

    prompt = f"""
You are verifying suspicious external rating data for one book.
Task:
Use the provided web search context to find the best current public rating values for this exact book on Goodreads, Open Library, and Amazon.
Return corrected values only if you can confidently match the same work.
If uncertain, return null for that site rather than guessing.

Book to verify:
- title: "{title}"
- author: "{author}"
- source: "{source}"
- category: "{category}"

Current suspicious row values:
- goodreads_rating: {gr_rating}
- goodreads_review_count: {gr_reviews}
- open_library_rating_consensus: {ol_rating}
- open_library_review_count_consensus: {ol_reviews}
- amazon_rating_consensus: {az_rating}
- amazon_review_count_consensus: {az_reviews}
- issues: {issues}

Matching rules:
1. Match the same book/work by title and author.
2. Avoid summaries, study guides, workbooks, audiobooks, translated adaptations unless the title clearly refers to the same main book.
3. Prefer the main consumer book page, not reseller/SEO/spam pages.
4. For Goodreads: use the main Goodreads book/work page.
5. For Open Library: use the main Open Library work/book page with public ratings if available.
6. For Amazon: use the main Amazon book product page with customer star rating.
7. If multiple editions exist, use the edition/work page with the clearest match to title+author.
8. If counts are abbreviated like "4.7K", convert to an integer.
9. If the site does not clearly expose a usable rating/count, return null for that site.
10. Be conservative. Precision is more important than coverage.

Search Context:
=== Goodreads Search ===
{gr_context}

=== Open Library Search ===
{ol_context}

=== Amazon Search ===
{az_context}
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BookVerification,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating content for {title}: {e}")
        return None

results = []
input_file = 'ai_books_tracking/multi_source_suspicious_rows.csv'
output_file = 'ai_books_tracking/multi_source_suspicious_rows_verified.json'

with open(input_file, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for i, row in enumerate(reader):
        res = process_book(row)
        if res:
            results.append(res)
        time.sleep(1) # Simple rate limiting for DuckDuckGo/Gemini

with open(output_file, mode='w', encoding='utf-8') as outfile:
    json.dump(results, outfile, indent=2)

print(f"Processed {len(results)} books and saved to {output_file}.")
if len(results) > 0:
    print("\nFirst result as example:")
    print(json.dumps(results[0], indent=2))
