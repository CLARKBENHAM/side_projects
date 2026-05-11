"""Cross-reference calendar book entries with Play Books takeout and training CSV.

Extracts "started book:", "finished book:", and "book:" entries from the calendar,
matches them to Play Books titles, and identifies:
- Books newly finished (not in original 208-book training CSV)
- Books started but not finished (per calendar)
- Discrepancies between calendar and Play Books data
"""

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import recurring_ical_events
from icalendar import Calendar

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"
FINISHED_CSV = DATA_DIR / "finished_books_2025_03_16.csv"
UNFINISHED_CSV = DATA_DIR / "unfinished_books_2025_03_16.csv"

CALENDAR_DIR = Path.home() / "Downloads" / "Takeout 5" / "Calendar"
CALENDAR_DIR_ALT = Path.home() / "Downloads" / "Takeout 5 11_08_2025" / "Calendar"

# Manual overrides: reading session refs that are actually finished books
# but lack a "finished book:" calendar entry. Maps ref -> canonical title.
MANUAL_FINISHED: dict[str, str] = {
    "toboss": "The Oxford Book of Short Stories",
    "mml": "Mathematics for Machine Learning",
    "asme y14.41": "ASME Y14.41",
}

# Manual abbreviation → title mappings that override automatic matching.
MANUAL_ABBREVS: dict[str, str] = {
    "h": "Hyperion",
    "htiymwtai": "how to improve your marriage without talking about it",
    "tgd": "the great democracies",
}

# Manual title equivalences: pairs that should be treated as the same book
# but don't match via automated normalization (e.g., stem differences).
TITLE_ALIASES: list[tuple[str, str]] = [
    ("shape up", "shaping up"),
    (
        "the history of the english speaking peoples volume 4",
        "the great democracies",
    ),
    ("night watch", "discworld 29 night watch"),
    ("fall of hyperion", "hyperion cantos 02 the fall of hyperion"),
    ("the mind body prescription", "the mindbody prescription"),
    ("autobiography of ben franklin", "the autobiography of benjamin franklin"),
    ("into the aquarium", "inside the aquarium"),
    ("deaths end", "death's end"),
]

# Title rewrites applied to calendar refs BEFORE any matching.
# Used to split multi-volume works tracked under a single abbreviation.
# Maps (ref, date_range) -> canonical title. Date ranges are inclusive.
VOLUME_SPLITS: list[tuple[str, str, str, str]] = [
    # "twc" covers The World Crisis vols 1-4; the "finished" entries
    # give us volume boundaries.
    # Vol 1: through 2025-12-27
    ("twc", "2000-01-01", "2025-12-27", "the world crisis"),
    # Vol 2: 2025-12-28 through 2026-01-04
    ("twc", "2025-12-28", "2026-01-04", "the world crisis volume 2"),
    # Vol 3+4: 2026-01-05 through 2026-01-10
    ("twc", "2026-01-05", "2026-01-10", "the world crisis volume 4"),
]

_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
}


def normalize_title(t: str) -> str:
    """Normalize a title for fuzzy comparison: lowercase, strip punctuation/articles."""
    t = t.lower().strip()
    # Remove file extensions
    t = re.sub(r"\.(pdf|epub|html|txt)$", "", t)
    # Remove parenthetical year/edition info
    t = re.sub(r"\(\d{4}\)", "", t)
    # Remove "Day N/M" suffixes from multi-day finish entries
    t = re.sub(r"\s*\(day\s*\d+/\d+\)\s*$", "", t, flags=re.IGNORECASE)
    # Normalize apostrophes and quotes
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", "").replace("\u201d", "")
    # Strip punctuation except apostrophes
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    # Normalize number words to digits
    for word, digit in _NUMBER_WORDS.items():
        t = re.sub(rf"\b{word}\b", digit, t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _containment_match(shorter: str, longer: str) -> bool:
    """Check if shorter being contained in longer is a real title match.

    Rejects matches where the remainder is a volume/part number or
    the shorter string is less than half the longer one.
    """
    remainder = longer.replace(shorter, "", 1).strip()
    if re.match(r"(volume|vol|part|book)\s*\d", remainder):
        return False
    if remainder.endswith((" volume", " vol", " part", " book")):
        return False
    if len(shorter) < len(longer) * 0.5:
        return False
    return True


def titles_match(a: str, b: str) -> bool:
    """Check if two titles refer to the same book after normalization."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # One contains the other (handles subtitle differences)
    if len(na) > 5 and len(nb) > 5:
        if na in nb or nb in na:
            shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
            if _containment_match(shorter, longer):
                return True
    # First N significant words match, but not if the longer title
    # continues with a volume/part number
    skip = {"the", "a", "an", "of", "and"}
    wa = [w for w in na.split() if w not in skip]
    wb = [w for w in nb.split() if w not in skip]
    n = min(len(wa), len(wb), 3)
    if n >= 2 and wa[:n] == wb[:n]:
        shorter_w, longer_w = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
        extra = longer_w[n:]
        if extra and extra[0] in ("volume", "vol", "part", "book"):
            pass  # Different volumes, not the same book
        elif wa[:n][-1] in ("volume", "vol", "part", "book") and wa[n:] != wb[n:]:
            pass  # Same prefix up to "volume" but different volume numbers
        else:
            return True
    # Handle abbreviation differences like "Ms" vs "Mrs", "mr" vs "mr"
    # by comparing with common abbreviations expanded
    na_exp = _expand_abbreviations(na)
    nb_exp = _expand_abbreviations(nb)
    if na_exp == nb_exp:
        return True
    if len(na_exp) > 5 and len(nb_exp) > 5:
        if na_exp in nb_exp or nb_exp in na_exp:
            shorter_exp = min(na_exp, nb_exp, key=len)
            longer_exp = max(na_exp, nb_exp, key=len)
            if _containment_match(shorter_exp, longer_exp):
                return True
    # Strip all spaces/apostrophes and compare (catches "dont" vs "don't",
    # "everyman" vs "every man", etc.)
    na_stripped = re.sub(r"[' ]", "", na)
    nb_stripped = re.sub(r"[' ]", "", nb)
    if len(na_stripped) > 5 and len(nb_stripped) > 5:
        if na_stripped == nb_stripped:
            return True
        if na_stripped in nb_stripped or nb_stripped in na_stripped:
            shorter_s = min(na_stripped, nb_stripped, key=len)
            longer_s = max(na_stripped, nb_stripped, key=len)
            if _containment_match(shorter_s, longer_s):
                return True
    # Word overlap: if both titles have 4+ significant words and 70%+ of
    # the shorter title's significant words appear in the longer one, treat
    # as match (catches typos like "elevyn" vs "evelyn" when most other
    # words match). Uses significant words to avoid false matches from
    # common articles like "the", "of", "a". Also require the titles to be
    # roughly similar in length to avoid "my early life" matching
    # "surprised by joy the shape of my early life".
    wa_sig = {w for w in na.split() if w not in skip}
    wb_sig = {w for w in nb.split() if w not in skip}
    min_sig = min(len(wa_sig), len(wb_sig))
    max_sig = max(len(wa_sig), len(wb_sig))
    if min_sig >= 3 and min_sig >= max_sig * 0.5:
        sig_overlap = len(wa_sig & wb_sig)
        # Don't match if the difference is just different volume numbers
        # (e.g., "world crisis volume 2" vs "world crisis volume 4")
        diff_a = wa_sig - wb_sig
        diff_b = wb_sig - wa_sig
        volume_words = {"volume", "vol", "part", "book"}
        has_volume_context = bool((wa_sig | wb_sig) & volume_words)
        diff_a_nums = all(w.isdigit() for w in diff_a)
        diff_b_nums = all(w.isdigit() for w in diff_b)
        if sig_overlap / min_sig >= 0.7 and not (
            has_volume_context and diff_a_nums and diff_b_nums and diff_a != diff_b
        ):
            return True
    # Check manual aliases (also match if one title starts with an alias)
    for alias_a, alias_b in TITLE_ALIASES:
        na_alias = normalize_title(alias_a)
        nb_alias = normalize_title(alias_b)
        a_matches = na == na_alias or na.startswith(na_alias + " ")
        b_matches = nb == nb_alias or nb.startswith(nb_alias + " ")
        a_matches_b = na == nb_alias or na.startswith(nb_alias + " ")
        b_matches_a = nb == na_alias or nb.startswith(na_alias + " ")
        if (a_matches and b_matches) or (a_matches_b and b_matches_a):
            return True
        if (a_matches and b_matches_a) or (a_matches_b and b_matches):
            return True
    return False


def _expand_abbreviations(title: str) -> str:
    """Expand common abbreviations in a normalized title for matching."""
    t = title
    t = re.sub(r"\bms\b", "mrs", t)
    t = re.sub(r"\bmr\b", "mister", t)
    t = re.sub(r"\bdr\b", "doctor", t)
    t = re.sub(r"\bst\b", "saint", t)
    return t


def parse_book_events(
    calendar_dir: Path, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Extract all book-related calendar events."""
    ics_path = calendar_dir / "Things.ics"
    if not ics_path.exists():
        raise FileNotFoundError(f"Things.ics not found at {ics_path}")

    with open(ics_path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    events = recurring_ical_events.of(cal).between(start_date, end_date)
    book_events = []
    for event in events:
        summary = str(event.get("summary", ""))
        if not (
            re.search(r"book", summary, re.IGNORECASE)
            or re.match(r"finished\s*:", summary, re.IGNORECASE)
            or re.match(r"started\s*:", summary, re.IGNORECASE)
        ):
            continue
        if re.match(
            r"^(get |search |toil:? |misc |find ).*book",
            summary,
            re.IGNORECASE,
        ):
            continue
        dtstart = event.get("dtstart")
        if dtstart:
            dt = dtstart.dt
            if hasattr(dt, "date"):
                date = dt.date()
            else:
                date = dt
            book_events.append({"date": date, "summary": summary.strip()})

    return pd.DataFrame(book_events)


def classify_event(summary: str) -> tuple[str, str]:
    """Classify a calendar event and extract the book reference.

    Returns (event_type, book_ref) where event_type is one of:
    'finished', 'started', 'reading', 'audiobook', 'other'
    """
    s = summary.strip()

    # "Finished book: X" / "finished audiobook: X" / "end of book: X"
    # Also handles prefixes like "ai qs/finished book: X"
    m = re.search(
        r"(?:finished|end of)\s+(?:audio\s*)?book\s*:\s*(.+?)$",
        s,
        re.IGNORECASE,
    )
    if m:
        return "finished", m.group(1).strip()

    # "Finished: X" (no "book" word)
    m = re.match(r"finished\s*:\s*(.+)", s, re.IGNORECASE)
    if m:
        ref = m.group(1).strip()
        # Only accept if it looks like a book title (not "finished: task")
        if len(ref) > 2:
            return "finished", ref

    # "Finished Audiobook: X"
    m = re.search(r"finished\s+audiobook\s*:\s*(.+?)$", s, re.IGNORECASE)
    if m:
        return "finished", m.group(1).strip()

    # "Finished Book: X" (with space variant)
    m = re.search(r"finished\s+book\s*:\s*(.+?)$", s, re.IGNORECASE)
    if m:
        return "finished", m.group(1).strip()

    # "Book finished: X"
    m = re.match(r"book\s+finished\s*:\s*(.+)", s, re.IGNORECASE)
    if m:
        return "finished", m.group(1).strip()

    # "elon book finished" style
    m = re.match(r"(.+?)\s+book\s+finished", s, re.IGNORECASE)
    if m:
        return "finished", m.group(1).strip()

    # "Started book: X" (possibly after prefix like "History/started book: X")
    m = re.search(r"started\s+book\s*:\s*(.+?)$", s, re.IGNORECASE)
    if m:
        return "started", m.group(1).strip()

    # "audiobook: X"
    m = re.match(r"audio\s*book\s*:\s*(.+)", s, re.IGNORECASE)
    if m:
        return "audiobook", m.group(1).strip()

    # "Book: X" (regular reading session)
    m = re.match(r"book\s*:\s*(.+)", s, re.IGNORECASE)
    if m:
        ref = m.group(1).strip()
        # Handle "book: X/book: Y" — take first
        if "/" in ref and "book:" in ref.lower():
            ref = ref.split("/")[0].strip()
        return "reading", ref

    # "Book" alone
    if re.match(r"^books?$", s, re.IGNORECASE):
        return "other", ""

    # Compound: "Blogs/book: X"
    m = re.search(r"book\s*:\s*(.+?)(?:/|$)", s, re.IGNORECASE)
    if m:
        return "reading", m.group(1).strip()

    return "other", ""


def is_abbreviation(ref: str) -> bool:
    """Heuristic: short references that look like initials."""
    ref = ref.strip()
    # Pure lowercase letters (2-8 chars), or single letter, or digits+letters (like "12rfl")
    if re.match(r"^[a-z]{1,8}$", ref):
        return True
    if re.match(r"^[a-z0-9]{2,8}$", ref) and re.search(r"[a-z]", ref):
        return True
    return False


def _split_words(title: str) -> list[str]:
    """Split title into words, treating contractions as single words.

    "it's" -> ["its"], not ["it", "s"].
    Numbers kept as full tokens: "12 rules" -> ["12", "rules"].
    """
    t = title.lower()
    # Normalize apostrophes then collapse contractions (it's -> its)
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"'", "", t)
    return re.findall(r"[a-z0-9]+", t)


def _word_initial(w: str) -> str:
    """Get the 'initial' of a word: first letter for alpha, full string for numbers."""
    if w.isdigit():
        return w
    return w[0]


def get_initials(title: str) -> str:
    """Get first-letter initials of a title (all words). Numbers kept whole."""
    return "".join(_word_initial(w) for w in _split_words(title))


def get_sig_initials(title: str) -> str:
    """Get first-letter initials of significant words (skip articles)."""
    skip = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "by", "its"}
    return "".join(_word_initial(w) for w in _split_words(title) if w not in skip)


ARTICLES = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "by"}


def _article_subset_initials(title: str) -> set[str]:
    """Generate initials for all subsets of article inclusion/exclusion.

    Users inconsistently include/skip articles in abbreviations:
    "the price of victory" -> "tpov" (all), "pv" (no articles),
    "tpv" (include 'the' but skip 'of'), etc.
    """
    words = _split_words(title)
    # Identify which word indices are articles
    article_indices = [i for i, w in enumerate(words) if w in ARTICLES]
    non_article_indices = [i for i, w in enumerate(words) if w not in ARTICLES]

    if not non_article_indices:
        return {get_initials(title)}

    results: set[str] = set()
    # Try all 2^len(article_indices) subsets of article inclusion
    for mask in range(1 << len(article_indices)):
        included = set(non_article_indices)
        for bit, idx in enumerate(article_indices):
            if mask & (1 << bit):
                included.add(idx)
        initials = "".join(_word_initial(words[i]) for i in sorted(included))
        if len(initials) >= 1:
            results.add(initials)
    return results


def _title_prefixes(title: str) -> list[str]:
    """Extract title prefixes before subtitle separators (: · - by)."""
    prefixes = [title]
    for sep in [":", "·", " - ", " — "]:
        if sep in title:
            prefixes.append(title.split(sep)[0].strip())
    return prefixes


def match_abbreviation_to_title(
    abbrev: str,
    candidate_titles: list[str],
) -> list[str]:
    """Match an abbreviation to known titles. Returns ALL matches for ambiguity detection."""
    abbrev_lower = abbrev.lower().strip()
    matches = []

    for title in candidate_titles:
        # Try full title and title prefixes (before subtitle separators)
        found = False
        for variant in _title_prefixes(title):
            all_initials = _article_subset_initials(variant)
            if abbrev_lower in all_initials:
                if title not in matches:
                    matches.append(title)
                found = True
                break
            if len(abbrev_lower) >= 4:
                for init in all_initials:
                    if _edit_distance_one(abbrev_lower, init):
                        if title not in matches:
                            matches.append(title)
                        found = True
                        break
                if found:
                    break
        if found:
            continue
        # Prefix match: only if abbrev IS the first word(s) of the title,
        # not just a string prefix that splits a word boundary.
        # Also reject if the remainder starts with a word that looks like a
        # sequel indicator (e.g. "dune" should not prefix-match "dune messiah").
        if len(abbrev_lower) >= 3:
            t_norm = normalize_title(title)
            if t_norm.startswith(abbrev_lower):
                remainder = t_norm[len(abbrev_lower):].strip()
                # Only match if remainder is empty or starts with a subtitle
                # separator word, not a sequel/different-book word
                if not remainder:
                    if title not in matches:
                        matches.append(title)
                elif t_norm[len(abbrev_lower)] == " ":
                    if title not in matches:
                        matches.append(title)

    return matches


def _edit_distance_one(a: str, b: str) -> bool:
    """Check if two strings differ by exactly one insertion, deletion, or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        # Substitution
        return sum(ca != cb for ca, cb in zip(a, b)) == 1
    # Insertion/deletion: the longer string has one extra char
    short, long = (a, b) if len(a) < len(b) else (b, a)
    diffs = 0
    si = 0
    for li in range(len(long)):
        if si < len(short) and short[si] == long[li]:
            si += 1
        else:
            diffs += 1
    return diffs <= 1


def _apply_volume_splits(ref: str, date_str: str) -> str:
    """Rewrite a ref based on VOLUME_SPLITS if it matches a date range."""
    ref_lower = ref.lower()
    for split_ref, start, end, canonical in VOLUME_SPLITS:
        if ref_lower == split_ref and start <= date_str <= end:
            return canonical
    return ref


def build_book_timeline(events_df: pd.DataFrame) -> dict:
    """Build a timeline of books from calendar events.

    Audiobook sessions are merged into reading_sessions (the user reads
    across both mediums). The audiobooks dict is kept for display only.
    """
    finished: list[tuple[str, str]] = []
    started: list[tuple[str, str]] = []
    reading_sessions: dict[str, list[str]] = defaultdict(list)
    audiobooks: dict[str, list[str]] = defaultdict(list)

    for _, row in events_df.iterrows():
        date_str = str(row["date"])
        event_type, book_ref = classify_event(row["summary"])

        if not book_ref:
            continue

        # Strip "Day N/M" from finished entries
        book_ref = re.sub(
            r"\s*\(day\s*\d+/\d+\)\s*$", "", book_ref, flags=re.IGNORECASE
        )

        # Apply volume splits to reading/audiobook refs
        if event_type in ("reading", "audiobook"):
            book_ref = _apply_volume_splits(book_ref.lower(), date_str)

        if event_type == "finished":
            finished.append((date_str, book_ref))
        elif event_type == "started":
            started.append((date_str, book_ref))
        elif event_type == "reading":
            reading_sessions[book_ref.lower()].append(date_str)
        elif event_type == "audiobook":
            ref_lower = book_ref.lower()
            audiobooks[ref_lower].append(date_str)
            # Merge audiobook sessions into reading_sessions
            reading_sessions[ref_lower].append(date_str)

    return {
        "finished": finished,
        "started": started,
        "reading_sessions": dict(reading_sessions),
        "audiobooks": dict(audiobooks),
    }


def deduplicate_finished(
    finished: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Deduplicate finished entries: same book finished on same/adjacent days
    (e.g., multi-day splits) or duplicate entries."""
    seen: dict[str, str] = {}  # normalized_title -> first date
    deduped: list[tuple[str, str]] = []
    for date, title in sorted(finished):
        norm = normalize_title(title)
        if norm in seen:
            continue
        # Check for near-matches already seen
        already = False
        for prev_norm in seen:
            if titles_match(title, prev_norm):
                already = True
                break
        if not already:
            seen[norm] = date
            deduped.append((date, title))
    return deduped


def resolve_finished_abbreviations(
    finished: list[tuple[str, str]],
    started: list[tuple[str, str]],
    reading_sessions: dict[str, list[str]],
    all_known_titles: list[str],
) -> list[tuple[str, str]]:
    """Resolve abbreviations in 'finished book:' entries.

    E.g., "finished book: tpv" -> "the price of victory" if "tpv" matches
    initials of a known started/reading book.
    """
    # Build pool of full titles from started entries, reading sessions with
    # full titles, and all known titles.
    # Also include short reading refs that are real title words (not just
    # initials) — e.g. "dune" is a real title, not an abbreviation.
    known_title_words = set()
    for t in all_known_titles:
        known_title_words.update(normalize_title(t).split())

    full_titles_pool: list[str] = []
    for _, t in started:
        full_titles_pool.append(t)
    for ref in reading_sessions:
        if not is_abbreviation(ref):
            full_titles_pool.append(ref)
        elif len(ref) >= 4 and ref.lower() in known_title_words:
            full_titles_pool.append(ref)
    full_titles_pool.extend(all_known_titles)

    resolved: list[tuple[str, str]] = []
    warnings: list[str] = []

    for date, title in finished:
        if is_abbreviation(title.lower()):
            matches = match_abbreviation_to_title(title, full_titles_pool)
            if len(matches) == 1:
                resolved.append((date, matches[0]))
            elif len(matches) > 1:
                # Disambiguate: prefer the match where the title is the
                # most complete match (highest word overlap ratio)
                title_norm = normalize_title(title)
                scored = []
                for m in matches:
                    m_norm = normalize_title(m)
                    # Exact normalized match
                    if m_norm == title_norm:
                        scored.append((100, m))
                    # Title is a complete word in the match
                    elif title_norm in m_norm.split():
                        scored.append((50, m))
                    # Title appears as prefix but not sole content
                    elif m_norm.startswith(title_norm):
                        scored.append((10, m))
                    else:
                        scored.append((1, m))
                scored.sort(key=lambda x: -x[0])
                best = scored[0][1]
                resolved.append((date, best))
                if scored[0][0] < 100:
                    warnings.append(
                        f"  WARNING: '{title}' matched multiple titles: {matches}. "
                        f"Using '{best}'."
                    )
            else:
                resolved.append((date, title))
                warnings.append(
                    f"  WARNING: Could not resolve finished abbreviation '{title}' "
                    f"(date: {date})"
                )
        else:
            resolved.append((date, title))

    for w in warnings:
        print(w)

    return resolved


def build_abbreviation_map(
    reading_sessions: dict[str, list[str]],
    audiobooks: dict[str, list[str]],
    cal_finished_titles: list[str],
    cal_started_titles: list[str],
    all_known_titles: list[str],
) -> dict[str, str | None]:
    """Build abbreviation -> resolved title map with duplicate warnings."""
    # Pool: calendar finished/started titles first (highest priority),
    # then non-abbreviated reading session refs, then all known titles
    manual_refs = set(MANUAL_ABBREVS.keys())
    reading_full_titles = [
        ref
        for ref in reading_sessions
        if not is_abbreviation(ref) and ref not in manual_refs
    ]
    audio_full_titles = [
        ref for ref in audiobooks if not is_abbreviation(ref) and ref not in manual_refs
    ]
    pool = (
        cal_finished_titles
        + cal_started_titles
        + reading_full_titles
        + audio_full_titles
        + all_known_titles
    )

    abbrev_map: dict[str, str | None] = {}
    all_refs = set(reading_sessions.keys()) | set(audiobooks.keys())

    # Apply manual overrides first
    for ref in sorted(all_refs):
        if ref in MANUAL_ABBREVS:
            abbrev_map[ref] = MANUAL_ABBREVS[ref]

    for ref in sorted(all_refs):
        if ref in abbrev_map:
            continue
        if not is_abbreviation(ref):
            continue
        matches = match_abbreviation_to_title(ref, pool)
        if len(matches) == 1:
            abbrev_map[ref] = matches[0]
        elif len(matches) > 1:
            # Deduplicate matches that normalize to the same thing
            unique = []
            seen_norm: set[str] = set()
            for m in matches:
                n = normalize_title(m)
                if n not in seen_norm:
                    seen_norm.add(n)
                    unique.append(m)
            if len(unique) == 1:
                abbrev_map[ref] = unique[0]
            else:
                print(
                    f"  WARNING: abbreviation '{ref}' matches multiple titles: "
                    f"{unique}. Using first."
                )
                abbrev_map[ref] = unique[0]
        else:
            abbrev_map[ref] = None

    return abbrev_map


def check_in_title_set(title: str, title_set: set[str]) -> bool:
    """Check if a title matches anything in a set using normalized comparison."""
    for t in title_set:
        if titles_match(title, t):
            return True
    return False


def reconcile(
    cal_dir: Path | None = None,
    start_year: int = 2025,
) -> None:
    """Main reconciliation: compare calendar, Play Books, and training data."""
    if cal_dir is None:
        cal_dir = CALENDAR_DIR if CALENDAR_DIR.exists() else CALENDAR_DIR_ALT

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(2026, 12, 31)

    print("=" * 70)
    print(f"BOOK RECONCILIATION: Calendar vs Play Books ({start_year}+)")
    print("=" * 70)

    # Calendar events
    events_df = parse_book_events(cal_dir, start_date, end_date)
    print(f"\nCalendar: {len(events_df)} book-related events from {start_year}")

    timeline = build_book_timeline(events_df)
    cal_finished_raw = timeline["finished"]
    cal_started = timeline["started"]
    reading_sessions = timeline["reading_sessions"]
    audiobooks = timeline["audiobooks"]

    print(f"  'Finished book:' entries (raw): {len(cal_finished_raw)}")
    print(f"  'Started book:' entries: {len(cal_started)}")
    print(f"  Unique reading refs: {len(reading_sessions)}")
    print(f"  Audiobook refs: {len(audiobooks)}")

    # Load data sources
    train_df = pd.read_csv(TRAIN_EXPORT)
    train_df.columns = train_df.columns.str.strip()
    train_titles_raw = [str(t).strip() for t in train_df["title"]]
    train_titles_norm = set(normalize_title(t) for t in train_titles_raw)
    print(f"\nOriginal training CSV: {len(train_df)} books")

    pb_finished = pd.read_csv(FINISHED_CSV) if FINISHED_CSV.exists() else pd.DataFrame()
    pb_unfinished = (
        pd.read_csv(UNFINISHED_CSV) if UNFINISHED_CSV.exists() else pd.DataFrame()
    )
    if not pb_finished.empty:
        pb_finished.columns = pb_finished.columns.str.strip()
    if not pb_unfinished.empty:
        pb_unfinished.columns = pb_unfinished.columns.str.strip()

    pb_fin_raw = (
        [str(t).strip() for t in pb_finished["title"]] if not pb_finished.empty else []
    )
    pb_unfin_raw = (
        [str(t).strip() for t in pb_unfinished["title"]]
        if not pb_unfinished.empty
        else []
    )
    all_pb_titles = pb_fin_raw + pb_unfin_raw
    print(
        f"Play Books takeout: {len(pb_fin_raw)} finished, {len(pb_unfin_raw)} unfinished"
    )

    # --- Resolve abbreviations in finished entries ---
    print("\nResolving abbreviations in finished entries...")
    cal_started_titles = [t for _, t in cal_started]
    cal_finished_resolved = resolve_finished_abbreviations(
        cal_finished_raw,
        cal_started,
        reading_sessions,
        all_pb_titles + train_titles_raw + cal_started_titles,
    )

    # Deduplicate (multi-day finishes, etc.)
    cal_finished = deduplicate_finished(cal_finished_resolved)

    # Add manual finished overrides (books confirmed finished by user
    # but lacking a "finished book:" calendar entry)
    existing_norm = {normalize_title(t) for _, t in cal_finished}
    for ref_key, canonical in MANUAL_FINISHED.items():
        if normalize_title(canonical) not in existing_norm:
            # Use the latest reading session date as the finish date
            # Try exact key match first, then case-insensitive
            dates = reading_sessions.get(ref_key, [])
            if not dates:
                for k, v in reading_sessions.items():
                    if normalize_title(k) == normalize_title(ref_key):
                        dates = v
                        break
            date = max(dates) if dates else "unknown"
            cal_finished.append((date, canonical))
            print(f"  Manual override: added '{canonical}' as finished (date: {date})")

    print(f"  After dedup: {len(cal_finished)} unique finished books")

    # --- Build abbreviation map ---
    cal_finished_titles = [t for _, t in cal_finished]
    abbrev_map = build_abbreviation_map(
        reading_sessions,
        audiobooks,
        cal_finished_titles,
        cal_started_titles,
        all_pb_titles + train_titles_raw,
    )

    # --- Section 1: Calendar-finished books ---
    print(f"\n{'=' * 70}")
    print("SECTION 1: BOOKS FINISHED PER CALENDAR")
    print(f"{'=' * 70}")

    new_finished = []
    old_finished = []
    for date, title in sorted(cal_finished):
        in_train = normalize_title(title) in train_titles_norm or any(
            titles_match(title, t) for t in train_titles_raw
        )
        if in_train:
            old_finished.append((date, title))
        else:
            new_finished.append((date, title))

    print(f"\n  Already in training set: {len(old_finished)}")
    print(f"  NEW (not in training set): {len(new_finished)}")

    print(f"\n  Newly finished books ({start_year}+):")
    for date, title in sorted(new_finished):
        in_pb = any(titles_match(title, t) for t in pb_fin_raw)
        pb_flag = "[PB:yes]" if in_pb else "[PB:no]"
        print(f"    {date}  {title:<55} {pb_flag}")

    # --- Section 2: Audiobooks ---
    print(f"\n{'=' * 70}")
    print("SECTION 2: AUDIOBOOKS")
    print(f"{'=' * 70}")

    print("\n  Audiobooks referenced in calendar:")
    for ref, dates in sorted(audiobooks.items(), key=lambda x: min(x[1])):
        resolved = abbrev_map.get(ref, ref) if is_abbreviation(ref) else ref
        label = f" -> {resolved}" if resolved and resolved != ref else ""
        print(f"    {ref:<30}{label:<40} sessions: {len(dates)}")

    # --- Section 3: Started but not finished ---
    print(f"\n{'=' * 70}")
    print("SECTION 3: BOOKS STARTED BUT NOT FINISHED (per calendar)")
    print(f"{'=' * 70}")

    finished_titles = [t for _, t in cal_finished]

    def is_known_finished(ref: str) -> bool:
        """Check if a reading ref matches any finished book."""
        for ft in finished_titles:
            if titles_match(ref, ft):
                return True
        # Check via abbreviation map (includes manual overrides)
        resolved = abbrev_map.get(ref)
        if not resolved and is_abbreviation(ref):
            resolved = abbrev_map.get(ref)
        if resolved:
            for ft in finished_titles:
                if titles_match(resolved, ft):
                    return True
        return False

    unfinished_reading: dict[str, dict] = {}
    for ref, dates in reading_sessions.items():
        if is_known_finished(ref):
            continue
        resolved = abbrev_map.get(ref)
        if resolved and is_known_finished(resolved):
            continue
        label = resolved if resolved else ref
        unfinished_reading[ref] = {
            "dates": dates,
            "n_sessions": len(dates),
            "first": min(dates),
            "last": max(dates),
            "resolved": label,
        }

    explicitly_started_unfinished = []
    for date, title in cal_started:
        if not is_known_finished(title):
            explicitly_started_unfinished.append((date, title))

    print("\n  Explicitly started (via 'started book:') but not finished:")
    for date, title in sorted(explicitly_started_unfinished):
        sessions = 0
        for ref, info in unfinished_reading.items():
            if titles_match(ref, title) or (
                info["resolved"] and titles_match(info["resolved"], title)
            ):
                sessions = info["n_sessions"]
                break
        print(f"    {date}  {title:<50} reading sessions: {sessions}")

    print("\n  Other books with reading sessions but no 'finished' entry:")
    for ref, info in sorted(unfinished_reading.items(), key=lambda x: x[1]["first"]):
        label = info["resolved"]
        print(
            f"    {label:<50} sessions: {info['n_sessions']:>3}  "
            f"({info['first']} to {info['last']})"
        )

    # --- Section 4: Unresolved abbreviations ---
    print(f"\n{'=' * 70}")
    print("SECTION 4: UNRESOLVED ABBREVIATIONS")
    print(f"{'=' * 70}")

    unresolved = {
        ref: reading_sessions.get(ref, [])
        for ref in abbrev_map
        if abbrev_map[ref] is None
    }
    audio_unresolved = {
        ref: audiobooks.get(ref, [])
        for ref in abbrev_map
        if abbrev_map[ref] is None and ref in audiobooks
    }
    unresolved.update(audio_unresolved)

    if unresolved:
        print("\n  These abbreviations couldn't be matched to a known title:")
        for ref in sorted(unresolved):
            dates = reading_sessions.get(ref, audiobooks.get(ref, []))
            if dates:
                print(
                    f"    '{ref}' ({len(dates)} sessions, {min(dates)} to {max(dates)})"
                )
            else:
                print(f"    '{ref}' (no sessions found)")
    else:
        print("\n  All abbreviations resolved.")

    # --- Section 5: Play Books vs Calendar comparison ---
    print(f"\n{'=' * 70}")
    print("SECTION 5: PLAY BOOKS FINISHED vs CALENDAR FINISHED")
    print(f"{'=' * 70}")

    pb_not_cal = []
    if not pb_finished.empty:
        for _, row in pb_finished.iterrows():
            title = str(row["title"]).strip()
            latest = str(row.get("latest_modified", ""))
            if latest and latest != "nan" and latest >= f"{start_year}":
                in_cal = any(titles_match(title, ft) for ft in finished_titles)
                if not in_cal:
                    in_train = any(titles_match(title, t) for t in train_titles_raw)
                    pb_not_cal.append((latest[:10], title, in_train))

    if pb_not_cal:
        print(
            f"\n  Play Books says finished (modified {start_year}+) "
            f"but NO calendar 'finished' entry:"
        )
        for date, title, in_train in sorted(pb_not_cal):
            flag = "[in training]" if in_train else "[NEW]"
            print(f"    {date}  {title[:55]:<55} {flag}")
    else:
        print("\n  All Play Books finished books have calendar entries.")

    cal_not_pb = []
    for date, title in cal_finished:
        in_pb = any(titles_match(title, t) for t in pb_fin_raw)
        if not in_pb:
            cal_not_pb.append((date, title))

    if cal_not_pb:
        print("\n  Calendar says finished but NOT in Play Books finished list:")
        for date, title in sorted(cal_not_pb):
            print(f"    {date}  {title}")

    # --- Write summary file ---
    out_path = OUTPUT_DIR / "book_reconciliation_summary.txt"
    with open(out_path, "w") as f:
        f.write(
            f"Book Reconciliation Summary "
            f"(generated {datetime.now().strftime('%Y-%m-%d')})\n"
        )
        f.write("=" * 70 + "\n\n")

        f.write("NEWLY FINISHED BOOKS (not in original 208-book training set)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Date':<12} {'Title':<55} {'Source'}\n")
        for date, title in sorted(new_finished):
            in_pb = any(titles_match(title, t) for t in pb_fin_raw)
            source = "Calendar+PlayBooks" if in_pb else "Calendar only"
            f.write(f"{date:<12} {title:<55} {source}\n")
        f.write(f"\nTotal new: {len(new_finished)}\n")

        f.write("\n\nSTARTED BUT NOT FINISHED (per calendar)\n")
        f.write("-" * 70 + "\n")
        for date, title in sorted(explicitly_started_unfinished):
            f.write(f"  {date}  {title}\n")
        for ref, info in sorted(
            unfinished_reading.items(), key=lambda x: x[1]["first"]
        ):
            f.write(
                f"  {info['first']}  {info['resolved']:<50} "
                f"({info['n_sessions']} sessions)\n"
            )

        f.write("\n\nALL CALENDAR-FINISHED BOOKS (for verification)\n")
        f.write("-" * 70 + "\n")
        for date, title in sorted(cal_finished):
            in_train = any(titles_match(title, t) for t in train_titles_raw)
            status = "OLD" if in_train else "NEW"
            f.write(f"  {date}  [{status}] {title}\n")

    print(f"\n\nSaved summary to {out_path}")

    # --- Write new_books_to_rate.csv ---
    # Collect all new finished books: calendar-finished + PB-only new
    all_new: list[tuple[str, str]] = list(new_finished)  # (date, title)

    # Add PB-only finished books that are genuinely new
    new_finished_norm = {normalize_title(t) for _, t in new_finished}
    for date, title, in_train in pb_not_cal:
        if not in_train and normalize_title(title) not in new_finished_norm:
            # Check it doesn't match any existing new_finished entry
            if not any(titles_match(title, t) for _, t in new_finished):
                all_new.append((date, title))

    # Build bookshelf lookup from PB data
    pb_bookshelf: dict[str, str] = {}
    if not pb_finished.empty:
        for _, row in pb_finished.iterrows():
            t = str(row["title"]).strip()
            shelf = str(row.get("bookshelf", "")).strip()
            if shelf and shelf != "nan":
                pb_bookshelf[normalize_title(t)] = shelf

    # Also check training CSV for bookshelf
    train_bookshelf: dict[str, str] = {}
    if "Bookshelf" in train_df.columns:
        for _, row in train_df.iterrows():
            t = str(row["title"]).strip()
            shelf = str(row.get("Bookshelf", "")).strip()
            if shelf and shelf != "nan":
                train_bookshelf[normalize_title(t)] = shelf

    def find_bookshelf(title: str) -> str:
        nt = normalize_title(title)
        if nt in pb_bookshelf:
            return pb_bookshelf[nt]
        if nt in train_bookshelf:
            return train_bookshelf[nt]
        # Fuzzy match against PB titles
        for pb_t, shelf in pb_bookshelf.items():
            if titles_match(title, pb_t):
                return shelf
        return ""

    csv_path = OUTPUT_DIR / "new_books_to_rate.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "title",
                "date_finished",
                "Enjoyment (/5)",
                "Usefulness /5 to Me",
                "Bookshelf",
                "Long Term Effects",
            ]
        )
        for date, title in sorted(all_new):
            shelf = find_bookshelf(title)
            writer.writerow([title, date, "", "", shelf, ""])

    print(f"Saved {len(all_new)} new books to {csv_path}")


if __name__ == "__main__":
    reconcile()
