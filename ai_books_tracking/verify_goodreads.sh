#!/bin/bash

# verify_goodreads.sh
# Usage: ./verify_goodreads.sh "Book Title" "Author" [model]

TITLE="$1"
AUTHOR="$2"
MODEL="${3:-flash}"

if [[ -z "$TITLE" ]]; then
    echo "Usage: $0 \"Book Title\" \"Author\" [model]"
    exit 1
fi

PROMPT="You are a book metadata expert. Find the EXACT Goodreads information for this book.
Title: $TITLE
Author: $AUTHOR

Return ONLY a JSON object with these keys:
- goodreads_url: The canonical URL for the book
- goodreads_title: The exact title on Goodreads
- goodreads_author: The primary author on Goodreads
- goodreads_rating: The current average rating (e.g. 4.23)
- goodreads_rating_count: The number of ratings (e.g. 10500)
- confidence: A score from 0 to 1 on how sure you are this is the correct book (not a children's version or different author with same title)

JSON:"

# Use the gemini CLI to get the data
# -m specifies the model, -p is the prompt, -o json for structured output
# We use --raw-output to avoid ANSI codes in the pipe
RESPONSE=$(gemini -m "$MODEL" -p "$PROMPT" --raw-output)

# Extract JSON from response (sometimes models add markdown blocks)
CLEAN_JSON=$(echo "$RESPONSE" | sed -n '/{/,/}/p')

echo "$CLEAN_JSON"
