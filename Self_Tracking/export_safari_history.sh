#!/usr/bin/env bash
set -euo pipefail

SRC_DB="${HOME}/Library/Safari/History.db"
DEST_DIR="${HOME}/del"
DEST_DB_UC="${DEST_DIR}/History.db"
DEST_DB_LC="${DEST_DIR}/history.db"
OUT_DIR="${HOME}/data/safair_history"

mkdir -p "$DEST_DIR" "$OUT_DIR"

# Check for sqlite3
if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 is required. Install via 'xcode-select --install' or Homebrew." >&2
  exit 1
fi

# Copy the live DB to ~/del using plain cp (your requested method)
if ! cp -f "$SRC_DB" "$DEST_DB_UC" 2>/dev/null; then
  echo "Could not copy $SRC_DB to $DEST_DB_UC. If denied, grant Terminal Full Disk Access." >&2
  exit 1
fi

# If a lowercase file exists (per your example), prefer it; otherwise use the uppercase one we just wrote.
DB_TO_QUERY="$DEST_DB_UC"
if [ -f "$DEST_DB_LC" ]; then
  DB_TO_QUERY="$DEST_DB_LC"
fi

# Output CSV path with timestamp
 ts="$(date +%Y%m%d_%H%M%S)"
out_csv="${OUT_DIR}/safari_history_${ts}.csv"

# Query: visits with local timestamps, URL, and title
SQL=$'SELECT datetime(v.visit_time+978307200,\'unixepoch\',\'localtime\') AS visited_at, i.url FROM history_visits v JOIN history_items i ON i.id=v.history_item ORDER BY v.visit_time DESC;'

if ! sqlite3 -header -csv "$DB_TO_QUERY" "$SQL" > "$out_csv"; then
  echo "sqlite3 query failed on $DB_TO_QUERY." >&2
  exit 1
fi

# Report summary and small preview
rows=$(wc -l < "$out_csv" | tr -d ' ')
echo "Copied DB: ${DB_TO_QUERY}"
echo "Wrote CSV: ${out_csv}"
echo "Rows (including header): ${rows}"
echo "Preview:"
head -n 5 "$out_csv" || true
