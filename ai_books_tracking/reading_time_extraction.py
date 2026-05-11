"""Extract per-book reading time from Google Calendar ICS data.

Strategy: Use "finished book:" calendar events as anchors. For each finished
book in the golden master, find the corresponding "finished" event, then
attribute all "book:" reading sessions that match (by ref) between the
previous finished-book event and this one.

This avoids ambiguous abbreviation matching by using temporal boundaries.
For books without a "finished" event, fall back to matching by date range
from the golden master's estimated_start/estimated_finish.

Heavy validation: cross-checks dates, flags speed outliers, reports unmatched.
"""

import re
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import recurring_ical_events
from icalendar import Calendar

sys.path.insert(0, str(Path(__file__).parent))
from calendar_book_reconciliation import (
    MANUAL_ABBREVS,
    TITLE_ALIASES,
    VOLUME_SPLITS,
    _apply_volume_splits,
    build_abbreviation_map,
    classify_event,
    deduplicate_finished,
    is_abbreviation,
    match_abbreviation_to_title,
    normalize_title,
    resolve_finished_abbreviations,
    titles_match,
)

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"
GOLDEN_MASTER = OUTPUT_DIR / "golden_master_multi_source.csv"
PLAY_EXPORT = DATA_DIR / "Books Read and their effects - Play Export.csv"

CALENDAR_DIRS = [
    DATA_DIR / "Takeout 5" / "Calendar",
    DATA_DIR / "Calendar Takeout" / "Calendar",
    Path.home() / "Downloads" / "Takeout 5" / "Calendar",
    Path.home() / "Downloads" / "Takeout 5 11_08_2025" / "Calendar",
]


def find_calendar_dir() -> Path:
    for d in CALENDAR_DIRS:
        if d.exists() and (d / "Things.ics").exists():
            return d
    raise FileNotFoundError(
        f"No calendar directory with Things.ics found. Tried: {CALENDAR_DIRS}"
    )


def parse_all_events(calendar_dir: Path) -> pd.DataFrame:
    """Extract all book-related events WITH duration from Things.ics."""
    ics_path = calendar_dir / "Things.ics"
    with open(ics_path, "rb") as f:
        cal = Calendar.from_ical(f.read())

    start_date = datetime(2018, 1, 1)
    end_date = datetime(2026, 12, 31)
    events = recurring_ical_events.of(cal).between(start_date, end_date)

    book_events = []
    for event in events:
        summary = str(event.get("summary", ""))
        # Match "book:" reading events, "finished [book]:" events, "started book:" events
        if not (
            re.search(r"book", summary, re.IGNORECASE)
            or re.match(r"finished\s*:", summary, re.IGNORECASE)
            or re.match(r"started\s*:", summary, re.IGNORECASE)
        ):
            continue
        if re.match(
            r"^(get |search |toil:? |misc |find ).*book", summary, re.IGNORECASE
        ):
            continue

        dtstart = event.get("dtstart")
        dtend = event.get("dtend")
        if not dtstart:
            continue

        dt_s = dtstart.dt
        if dtend:
            dt_e = dtend.dt
        else:
            dt_e = dt_s + timedelta(hours=1)

        if hasattr(dt_s, "hour"):
            if dt_s.tzinfo is None:
                dt_s = dt_s.replace(tzinfo=pytz.UTC)
            if dt_e.tzinfo is None:
                dt_e = dt_e.replace(tzinfo=pytz.UTC)
            duration_hours = (dt_e - dt_s).total_seconds() / 3600.0
        else:
            duration_hours = 1.0

        if hasattr(dt_s, "date"):
            ev_date = dt_s.date()
        else:
            ev_date = dt_s

        event_type, book_ref = classify_event(summary)
        if not book_ref:
            continue

        book_ref = re.sub(
            r"\s*\(day\s*\d+/\d+\)\s*$", "", book_ref, flags=re.IGNORECASE
        )

        if event_type in ("reading", "audiobook"):
            book_ref = _apply_volume_splits(book_ref.lower(), str(ev_date))

        if duration_hours > 7:
            print(f"  SKIP implausible: {summary} ({duration_hours:.1f}h on {ev_date})")
            continue

        # Handle slash events
        if "/" in summary and event_type == "reading":
            parts = [p.strip() for p in summary.split("/")]
            duration_hours = duration_hours / len(parts)

        book_events.append({
            "date": ev_date,
            "summary": summary,
            "event_type": event_type,
            "book_ref": book_ref.lower().strip(),
            "duration_hours": duration_hours,
        })

    return pd.DataFrame(book_events)


def build_finish_timeline(events_df: pd.DataFrame, golden_titles: list[str],
                          play_titles: list[str]) -> list[dict]:
    """Build a timeline of finished books from calendar events.

    Returns list of {date, cal_title, golden_title} sorted by date.
    Each finished event is matched to a golden master title.
    """
    finished_events = events_df[events_df["event_type"] == "finished"].copy()
    started_events = events_df[events_df["event_type"] == "started"].copy()
    reading_sessions = defaultdict(list)
    for _, row in events_df[events_df["event_type"].isin(["reading", "audiobook"])].iterrows():
        reading_sessions[row["book_ref"]].append(str(row["date"]))

    # Resolve abbreviations in finished entries
    finished_raw = [(str(r["date"]), r["book_ref"]) for _, r in finished_events.iterrows()]
    started_raw = [(str(r["date"]), r["book_ref"]) for _, r in started_events.iterrows()]
    cal_started_titles = [t for _, t in started_raw]

    resolved = resolve_finished_abbreviations(
        finished_raw, started_raw, dict(reading_sessions),
        play_titles + golden_titles + cal_started_titles,
    )
    deduped = deduplicate_finished(resolved)

    # Match each finished entry to golden master
    # First pass: strict titles_match
    # Second pass: lenient matching for subtitle differences and typos
    # Build a map from the raw finished refs (before abbreviation resolution)
    # to their resolved forms. This lets us detect cases where e.g. "dune"
    # was wrongly resolved to "dune messiah" and try the original ref too.
    raw_to_resolved = {}
    for (_, raw_title), (_, resolved_title) in zip(sorted(finished_raw), sorted(resolved)):
        raw_norm = normalize_title(raw_title)
        res_norm = normalize_title(resolved_title)
        if raw_norm != res_norm:
            raw_to_resolved[res_norm] = raw_title

    timeline = []
    for date_str, cal_title in sorted(deduped):
        golden_match = None
        cal_norm = normalize_title(cal_title)

        # Strict match — find ALL matches and pick the best one
        # (prefer shorter golden titles / higher overlap ratio)
        strict_matches = []
        for gt in golden_titles:
            if titles_match(cal_title, gt):
                gt_norm = normalize_title(gt)
                cal_len = len(cal_norm.split())
                gt_len = len(gt_norm.split())
                ratio = cal_len / max(gt_len, 1)
                strict_matches.append((ratio, gt))
        if strict_matches:
            strict_matches.sort(key=lambda x: -x[0])
            golden_match = strict_matches[0][1]

        # Lenient: cal_title words all appear in golden title as word-boundary match
        # e.g. "Dune" matches "Frank Herbert - Dune 1 - Dune.pdf"
        # e.g. "the goal" matches "The Goal: A Process of Ongoing Improvement"
        if golden_match is None and len(cal_norm) >= 3:
            cal_words = set(cal_norm.split()) - {"the", "a", "an", "of", "and"}
            if cal_words:
                best_score = 0
                best_gt = None
                for gt in golden_titles:
                    gt_norm = normalize_title(gt)
                    gt_words = set(gt_norm.split())
                    overlap = cal_words & gt_words
                    if overlap == cal_words and len(cal_words) >= 1:
                        # Penalize if golden title has sequel/volume words
                        # not present in cal_title (e.g. "dune" shouldn't
                        # prefer "dune messiah" over "dune 1 dune")
                        extra_words = gt_words - cal_words - {"the", "a", "an", "of", "and"}
                        sequel_penalty = 0
                        for w in extra_words:
                            if w in ("messiah", "children", "emperor", "god",
                                     "heretics", "chapterhouse", "end", "forest",
                                     "death", "dark", "2", "3", "4", "5"):
                                sequel_penalty += 1
                        # Heavy penalty: a single extra sequel word should
                        # disqualify since it likely indicates a different book
                        score = len(cal_words) / len(gt_words) - sequel_penalty * 0.5
                        if score > best_score:
                            best_score = score
                            best_gt = gt
                    # Also try matching cal against golden title prefixes
                    # (before subtitle separators like : · -)
                    else:
                        from calendar_book_reconciliation import _title_prefixes
                        for prefix in _title_prefixes(gt)[1:]:
                            prefix_norm = normalize_title(prefix)
                            prefix_words = set(prefix_norm.split())
                            p_overlap = cal_words & prefix_words
                            if p_overlap == cal_words and len(cal_words) >= 1:
                                score = len(cal_words) / len(prefix_words)
                                if score > best_score:
                                    best_score = score
                                    best_gt = gt
                                break
                if best_gt is not None:
                    golden_match = best_gt

        # If still no match and this title was an abbreviation resolution,
        # also try the original raw ref
        if golden_match is None and cal_norm in raw_to_resolved:
            raw_ref = raw_to_resolved[cal_norm]
            raw_norm = normalize_title(raw_ref)
            raw_words = set(raw_norm.split()) - {"the", "a", "an", "of", "and"}
            if raw_words and len(raw_norm) >= 3:
                best_score = 0
                for gt in golden_titles:
                    gt_norm = normalize_title(gt)
                    gt_words = set(gt_norm.split())
                    overlap = raw_words & gt_words
                    if overlap == raw_words and len(raw_words) >= 1:
                        extra_words = gt_words - raw_words - {"the", "a", "an", "of", "and"}
                        sequel_penalty = sum(
                            1 for w in extra_words
                            if w in ("messiah", "children", "emperor", "god",
                                     "heretics", "chapterhouse", "end", "forest",
                                     "death", "dark", "2", "3", "4", "5")
                        )
                        score = len(raw_words) / len(gt_words) - sequel_penalty * 0.1
                        if score > best_score:
                            best_score = score
                            golden_match = gt

        # Lenient: fuzzy — allow 1-2 char difference for typos
        if golden_match is None and len(cal_norm) >= 6:
            for gt in golden_titles:
                gt_norm = normalize_title(gt)
                # Compare first N chars where N = len(cal_norm)
                gt_prefix = gt_norm[:len(cal_norm)]
                if len(cal_norm) == len(gt_prefix):
                    diffs = sum(a != b for a, b in zip(cal_norm, gt_prefix))
                    if diffs <= 2 and diffs > 0:
                        golden_match = gt
                        break

        timeline.append({
            "date": date_str,
            "cal_title": cal_title,
            "golden_title": golden_match,
        })

    return timeline


def resolve_ref_for_date(
    ref: str,
    ref_date: date,
    finish_timeline: list[dict],
    abbrev_map: dict,
    golden_titles: list[str],
) -> str | None:
    """Resolve a reading session ref to a golden master title, using temporal context.

    Strategy:
    1. Find which "finished book" windows this date falls into.
    2. Among golden-matched finished books whose window includes ref_date,
       check if the ref matches (via title or abbreviation).
    3. If multiple match, pick the one whose finish date is closest.
    """
    ref_date_str = str(ref_date)

    # Build windows: each finished book "owns" the period from the previous
    # finish to its own finish date (+ 1 day grace).
    matched_finishes = [f for f in finish_timeline if f["golden_title"] is not None]

    candidates = []
    for i, f in enumerate(matched_finishes):
        f_date = f["date"]
        # Window start: previous finish date, or very early
        if i > 0:
            window_start = matched_finishes[i - 1]["date"]
        else:
            window_start = "2000-01-01"

        # Only consider if ref_date is in window (with 2 day grace after finish)
        grace_date = str(
            pd.to_datetime(f_date).date() + timedelta(days=2)
        )
        if ref_date_str < window_start or ref_date_str > grace_date:
            continue

        gt = f["golden_title"]
        cal_t = f["cal_title"]

        # Check if this ref matches the book
        if _ref_matches_title(ref, cal_t, gt, abbrev_map):
            days_to_finish = abs(
                (pd.to_datetime(f_date).date() - ref_date).days
            )
            candidates.append((gt, days_to_finish))

    if candidates:
        # Pick closest finish date
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    # Fallback: direct title match without temporal constraint
    norm_ref = normalize_title(ref)
    for gt in golden_titles:
        if titles_match(ref, gt):
            return gt

    # Abbreviation map fallback
    if ref in abbrev_map and abbrev_map[ref] is not None:
        resolved = abbrev_map[ref]
        for gt in golden_titles:
            if titles_match(resolved, gt):
                return gt

    return None


def _ref_matches_title(ref: str, cal_title: str, golden_title: str,
                       abbrev_map: dict) -> bool:
    """Check if a reading session ref plausibly refers to a specific book."""
    if titles_match(ref, golden_title):
        return True
    if titles_match(ref, cal_title):
        return True

    # Check abbreviation: ref could be initials of the title
    if is_abbreviation(ref):
        matches = match_abbreviation_to_title(ref, [cal_title, golden_title])
        if matches:
            return True
        # Also check via abbrev_map
        if ref in abbrev_map and abbrev_map[ref] is not None:
            resolved = abbrev_map[ref]
            if titles_match(resolved, golden_title) or titles_match(resolved, cal_title):
                return True

    return False


def main():
    print("=" * 70)
    print("READING TIME EXTRACTION FROM CALENDAR (timestamp-based)")
    print("=" * 70)

    cal_dir = find_calendar_dir()
    print(f"\nUsing calendar: {cal_dir}")

    # Load golden master
    gm = pd.read_csv(GOLDEN_MASTER)
    gm.columns = gm.columns.str.strip()
    golden_titles = gm["title"].tolist()

    # Load Play Export
    play_df = pd.read_csv(PLAY_EXPORT)
    play_df.columns = play_df.columns.str.strip()
    play_titles = [str(t).strip() for t in play_df["title"]]

    # Parse all calendar events
    events_df = parse_all_events(cal_dir)
    print(f"\nTotal book events: {len(events_df)}")
    for et in ["reading", "audiobook", "finished", "started", "other"]:
        print(f"  {et}: {(events_df['event_type'] == et).sum()}")

    reading_events = events_df[
        events_df["event_type"].isin(["reading", "audiobook"])
    ].copy()
    print(f"\nReading+audiobook events: {len(reading_events)}")
    print(f"  Total hours: {reading_events['duration_hours'].sum():.1f}")

    # Build finish timeline
    finish_timeline = build_finish_timeline(events_df, golden_titles, play_titles)
    n_matched_finishes = sum(1 for f in finish_timeline if f["golden_title"] is not None)
    print(f"\nFinished-book events: {len(finish_timeline)}")
    print(f"  Matched to golden master: {n_matched_finishes}")

    # Build abbreviation map (for fallback)
    reading_sessions = defaultdict(list)
    audiobooks = defaultdict(list)
    finished_titles_raw = []
    started_titles_raw = []
    for _, row in events_df.iterrows():
        ref = row["book_ref"]
        if row["event_type"] == "reading":
            reading_sessions[ref].append(str(row["date"]))
        elif row["event_type"] == "audiobook":
            audiobooks[ref].append(str(row["date"]))
            reading_sessions[ref].append(str(row["date"]))
        elif row["event_type"] == "finished":
            finished_titles_raw.append(ref)
        elif row["event_type"] == "started":
            started_titles_raw.append(ref)

    abbrev_map = build_abbreviation_map(
        dict(reading_sessions), dict(audiobooks),
        finished_titles_raw, started_titles_raw,
        play_titles + golden_titles,
    )

    # --- MATCH READING EVENTS TO GOLDEN TITLES ---
    gm["est_start"] = pd.to_datetime(gm["estimated_start"], errors="coerce")
    gm["est_finish"] = pd.to_datetime(gm["estimated_finish"], errors="coerce")

    # Pre-compute: for each finished golden title, what refs match it?
    # This avoids O(events * titles * abbreviation_matching) at runtime.
    matched_finishes = [f for f in finish_timeline if f["golden_title"] is not None]
    print(f"\nPre-computing ref matches for {len(matched_finishes)} finished books...")

    # Build ref->golden mapping per time window
    # For each unique ref, find which golden titles it could match
    unique_refs = reading_events["book_ref"].unique()
    ref_possible_titles = {}
    for ref in unique_refs:
        possible = set()
        for f in matched_finishes:
            gt = f["golden_title"]
            cal_t = f["cal_title"]
            if _ref_matches_title(ref, cal_t, gt, abbrev_map):
                possible.add(gt)
        # Also direct title match
        for gt in golden_titles:
            if titles_match(ref, gt):
                possible.add(gt)
        if ref in abbrev_map and abbrev_map[ref] is not None:
            resolved = abbrev_map[ref]
            for gt in golden_titles:
                if titles_match(resolved, gt):
                    possible.add(gt)
        ref_possible_titles[ref] = possible

    # Now resolve each event using temporal context + pre-computed possibilities
    def resolve_event(ref, ref_date):
        possible = ref_possible_titles.get(ref, set())
        if not possible:
            return None
        if len(possible) == 1:
            return next(iter(possible))

        # Multiple possibilities: assign to the EARLIEST finished book
        # whose finish date is on or after the event date.
        # This implements "sessions before Book A finishes go to Book A,
        # sessions after go to Book B".
        candidates = []
        for gt in possible:
            for f in matched_finishes:
                if f["golden_title"] == gt:
                    f_date = pd.to_datetime(f["date"]).date()
                    if ref_date <= f_date + timedelta(days=2):
                        candidates.append((f_date, gt))
                        break
        if candidates:
            # Pick the earliest finish that's still >= ref_date
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return next(iter(possible))

    reading_events["golden_title"] = reading_events.apply(
        lambda r: resolve_event(r["book_ref"], r["date"]), axis=1
    )

    matched_events = reading_events[reading_events["golden_title"].notna()].copy()
    unmatched_events = reading_events[reading_events["golden_title"].isna()].copy()

    # For books with multiple reads, only count the FIRST read-through
    # (from first session to first "finished" event)
    finish_dates = {}
    for f in finish_timeline:
        if f["golden_title"] is not None:
            gt = f["golden_title"]
            fd = f["date"]
            if gt not in finish_dates or fd < finish_dates[gt]:
                finish_dates[gt] = fd

    # Filter matched events to only those on or before the first finish date
    def is_first_read(row):
        gt = row["golden_title"]
        if gt in finish_dates:
            return str(row["date"]) <= str(
                pd.to_datetime(finish_dates[gt]).date() + timedelta(days=1)
            )
        return True

    matched_first_read = matched_events[matched_events.apply(is_first_read, axis=1)].copy()
    reread_events = matched_events[~matched_events.apply(is_first_read, axis=1)]

    n_unique_matched = matched_first_read["golden_title"].nunique()
    print(f"\n--- MATCHING RESULTS ---")
    print(f"Unique calendar refs: {reading_events['book_ref'].nunique()}")
    print(f"Matched events: {len(matched_events)} ({len(matched_first_read)} first-read, {len(reread_events)} re-read)")
    print(f"Unique golden titles matched: {n_unique_matched}")
    print(f"Unmatched events: {len(unmatched_events)}")

    # Aggregate per-book reading time (first read only)
    book_time = (
        matched_first_read.groupby("golden_title")
        .agg(
            total_reading_hours=("duration_hours", "sum"),
            n_sessions=("duration_hours", "count"),
            first_session=("date", "min"),
            last_session=("date", "max"),
            mean_session_hours=("duration_hours", "mean"),
        )
        .reset_index()
    )
    book_time.rename(columns={"golden_title": "title"}, inplace=True)

    # Merge with golden master
    merged = gm.merge(book_time, on="title", how="left")

    # Compute reading speed
    merged["page_count_num"] = pd.to_numeric(merged["page_count"], errors="coerce")
    has_both = (
        merged["total_reading_hours"].notna()
        & merged["page_count_num"].notna()
        & (merged["page_count_num"] > 0)
    )
    merged["total_reading_minutes"] = merged["total_reading_hours"] * 60
    merged.loc[has_both, "minutes_per_page"] = (
        merged.loc[has_both, "total_reading_minutes"]
        / merged.loc[has_both, "page_count_num"]
    )

    # --- VALIDATION ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # 1. Coverage analysis
    print("\n--- Coverage Analysis ---")
    has_time = merged["total_reading_hours"].notna()
    cal_start = date(2021, 5, 24)

    before_cal = merged["est_finish"].notna() & (merged["est_finish"].dt.date < cal_start)
    after_cal = merged["est_finish"].notna() & (merged["est_finish"].dt.date >= cal_start)
    no_dates = merged["est_finish"].isna()

    print(f"  Total golden master books: {len(gm)}")
    print(f"  Finished before calendar tracking (pre {cal_start}): {before_cal.sum()}")
    print(f"  Finished after calendar tracking started: {after_cal.sum()}")
    print(f"  No finish date: {no_dates.sum()}")
    print(f"  Books with calendar reading time: {has_time.sum()}")
    print(f"  Coverage of post-calendar books: {has_time[after_cal].sum()}/{after_cal.sum()} "
          f"({has_time[after_cal].sum()/after_cal.sum()*100:.0f}%)")
    print(f"  Coverage of pre-calendar books: {has_time[before_cal].sum()}/{before_cal.sum()} "
          f"({has_time[before_cal].sum()/before_cal.sum()*100:.0f}% - expected ~0)")

    # Post-calendar books NOT matched
    post_cal_unmatched = merged[after_cal & ~has_time].copy()
    if len(post_cal_unmatched) > 0:
        print(f"\n  Post-calendar books WITHOUT reading time ({len(post_cal_unmatched)}):")
        for _, row in post_cal_unmatched.sort_values("est_finish").iterrows():
            print(f"    {row['est_finish'].date() if pd.notna(row['est_finish']) else '?'} "
                  f"{str(row['title'])[:55]}")

    # 2. Reading speed outliers
    print("\n--- Reading Speed Outliers ---")
    speed_df = merged[has_both].copy()
    if len(speed_df) > 0:
        median_speed = speed_df["minutes_per_page"].median()
        mad = np.median(np.abs(speed_df["minutes_per_page"] - median_speed))
        speed_df["speed_z"] = (speed_df["minutes_per_page"] - median_speed) / (mad * 1.4826 + 0.01)

        outliers = speed_df[speed_df["speed_z"].abs() > 3]
        print(f"  Median: {median_speed:.2f} min/page, MAD: {mad:.2f}")
        print(f"  Outliers (|z|>3): {len(outliers)}")
        for _, row in outliers.iterrows():
            print(
                f"    {str(row['title'])[:45]:<45} "
                f"{row['minutes_per_page']:.2f} min/pg "
                f"({row['total_reading_hours']:.1f}h, {row['page_count_num']:.0f}pg) "
                f"z={row['speed_z']:.1f}"
            )

    # 3. Unmatched sessions
    print("\n--- Unmatched Calendar Refs (top 15 by hours) ---")
    unmatched_summary = (
        unmatched_events.groupby("book_ref")
        .agg(hours=("duration_hours", "sum"), sessions=("duration_hours", "count"))
        .sort_values("hours", ascending=False)
    )
    total_unmatched_h = unmatched_summary["hours"].sum()
    print(f"  {len(unmatched_summary)} unmatched refs, {total_unmatched_h:.1f} total hours")
    for ref, row in unmatched_summary.head(15).iterrows():
        resolved = abbrev_map.get(ref, "?")
        print(f"    {ref:<25} -> {str(resolved)[:30]:<30} {row['hours']:.1f}h ({row['sessions']:.0f})")

    # 4. Re-read events excluded
    if len(reread_events) > 0:
        reread_summary = (
            reread_events.groupby("golden_title")
            .agg(hours=("duration_hours", "sum"), sessions=("duration_hours", "count"))
            .sort_values("hours", ascending=False)
        )
        print(f"\n--- Re-read events excluded ({len(reread_events)} events, {reread_events['duration_hours'].sum():.1f}h) ---")
        for title, row in reread_summary.head(10).iterrows():
            print(f"    {str(title)[:50]:<50} {row['hours']:.1f}h ({row['sessions']:.0f} sessions)")

    # 5. Spot-check specific books
    print("\n--- Spot Checks ---")
    for check_title in ["Dune Messiah", "Harry Potter and the Methods of Rationality",
                        "THE 48 LAWS OF POWER", "Apollo"]:
        match = merged[merged["title"].str.contains(check_title, case=False, na=False)]
        if len(match) > 0:
            r = match.iloc[0]
            events_for = matched_first_read[
                matched_first_read["golden_title"] == r["title"]
            ] if pd.notna(r.get("total_reading_hours")) else pd.DataFrame()
            print(f"  {r['title'][:55]}:")
            print(f"    hours={r.get('total_reading_hours', 'N/A')}, "
                  f"pages={r.get('page_count_num', 'N/A')}, "
                  f"min/pg={r.get('minutes_per_page', 'N/A')}")
            if len(events_for) > 0:
                print(f"    Sessions ({len(events_for)}):")
                for _, e in events_for.sort_values("date").iterrows():
                    print(f"      {e['date']} ref=\"{e['book_ref']}\" {e['duration_hours']:.2f}h")

    # --- SUMMARY ---
    print(f"\n--- Summary ---")
    print(f"  Books in golden master: {len(gm)}")
    print(f"  Books with calendar reading time: {has_time.sum()}")
    print(f"  Books with page_count: {merged['page_count_num'].notna().sum()}")
    print(f"  Books with BOTH time + pages: {has_both.sum()}")
    if has_time.sum() > 0:
        print(f"  Mean reading hours: {merged.loc[has_time, 'total_reading_hours'].mean():.1f}")
        print(f"  Median reading hours: {merged.loc[has_time, 'total_reading_hours'].median():.1f}")

    # --- OUTPUT ---
    out_cols = [
        "title", "author", "category", "estimated_start", "estimated_finish",
        "avg_enjoyment", "avg_usefulness", "page_count", "page_count_num",
        "goodreads_rating", "goodreads_rating_count",
        "total_reading_hours", "total_reading_minutes", "n_sessions",
        "first_session", "last_session", "mean_session_hours",
        "minutes_per_page", "Bookshelf", "year_finished",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out_df = merged[out_cols].copy()

    out_path = OUTPUT_DIR / "book_reading_times.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} books to {out_path}")

    return merged


if __name__ == "__main__":
    main()
