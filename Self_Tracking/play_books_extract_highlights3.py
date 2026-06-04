#!/usr/bin/env python3
"""
Google Play Books highlights extractor (Android 11-14, non-root) using adb + uiautomator dump.

Workflow:
  1) On the tablet, open Google Play Books and navigate to the per-book highlights list screen.
  2) Run:
       python3 dump_playbooks_highlights.py --out out
  3) The script will:
       - Dump UI hierarchy
       - Extract likely highlight text nodes
       - Swipe to scroll
       - Repeat until no new highlights found for a few consecutive pages
       - Write Markdown + JSONL exports

Notes:
  - This does NOT require root.
  - This does NOT modify Play Books data.
  - It is resilient but not perfect: it may capture some UI chrome. You can refine filters if needed.
  - To attempt full-library export, start from the Library screen and add --all-books.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

"""Most annotations only via tablet and not in play books
    Local Markdown export for Every Man for Himself and God Against All.


# passage of power has 836 yellow in play books vs 141 in tablet
so table is proper subset of play books
master of the sentate 541 vs 164 in table

pdf page numbering is off, starts at 927 past the end of real book.
but quotes are correct. Table is a continuous subset from the start

making of the atomic bomb 370 yellow vs couldn't open
(didn't click into them?)

dark sun 172 yellow (201 total) vs
(and they're not the same subsets)

"""

# Titles to export. Keep these as lowercase substrings for easy edits.
TARGET_TITLE_SUBSTRINGS = [
    "walking with destiny",
    "the world crisis",
    "great contemporaries",
    "second world war",
    "prime movers",
    "business adventures",
    "vibe coding",
    "revolt in the",
    "the price of victory",
    "of the ocean",
    "thoughts and",
    "six",
    "my early life",
    "systems performance",
    "richard nixon",
    # "man for himself",
    "dark sun",
    "improve your marriage",
    "atomic bomb",  # why couldn't the notes list here be opened?
    # "bureaucracy",
    "passage of",
    "AI Engineering",
    "Practical ML",
    "master of the senate",
    "path to power",
    "means of ascent",
    "apple in china",
    "building microservices",
    "linux kernel",
    "life and fate",
    "kelly",
]

ALWAYS_SKIP_TITLE_SUBSTRINGS = [
    "the abolition of man and the great divorce",
]


def run(
    cmd: List[str],
    check: bool = True,
    capture: bool = True,
    text: bool = True,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=text,
        timeout=timeout,
    )


def adb(args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return run(["adb"] + args, timeout=timeout)


def ensure_adb_ok() -> None:
    try:
        p = adb(["devices"])
    except FileNotFoundError:
        print(
            "ERROR: adb not found in PATH. Install Android platform-tools.",
            file=sys.stderr,
        )
        sys.exit(1)

    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        print(
            "ERROR: No devices detected. Check USB cable + USB debugging prompt.",
            file=sys.stderr,
        )
        sys.exit(1)
    if any("\tunauthorized" in ln for ln in lines[1:]):
        print(
            "ERROR: Device unauthorized. Unlock tablet and accept the USB debugging prompt.",
            file=sys.stderr,
        )
        sys.exit(1)


def get_screen_size() -> Tuple[int, int]:
    p = adb(["shell", "wm", "size"])
    # Example: "Physical size: 1200x2000"
    m = re.search(r"Physical size:\s*(\d+)x(\d+)", p.stdout)
    if not m:
        # Fallback: "Override size: ..."
        m = re.search(r"Override size:\s*(\d+)x(\d+)", p.stdout)
    if not m:
        raise RuntimeError(f"Could not parse screen size from: {p.stdout!r}")
    return int(m.group(1)), int(m.group(2))


def parse_data_free_bytes_from_df(output: str) -> int:
    lines = [line.split() for line in output.splitlines() if line.strip()]
    for parts in lines[1:]:
        if len(parts) < 6:
            continue
        mount = parts[-1]
        if mount.startswith("/data"):
            try:
                return int(parts[3]) * 1024
            except ValueError as exc:
                raise RuntimeError(
                    f"Could not parse free space from df: {output!r}"
                ) from exc
    raise RuntimeError(f"Could not find /data free space in df output: {output!r}")


def get_data_free_bytes() -> int:
    p = adb(["shell", "df", "-k", "/data"])
    return parse_data_free_bytes_from_df(p.stdout)


def format_gib(byte_count: int) -> str:
    return f"{byte_count / (1024**3):.1f} GiB"


def format_storage_delta(before: int, after: int) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{format_gib(abs(delta))}"


def min_free_bytes(args: argparse.Namespace) -> int:
    return int(max(getattr(args, "min_free_gb", 0.0), 0.0) * (1024**3))


def enforce_min_free_space(args: argparse.Namespace, context: str) -> None:
    required = min_free_bytes(args)
    if required <= 0:
        return
    free = get_data_free_bytes()
    if free < required:
        raise RuntimeError(
            f"Refusing to {context}: tablet has {format_gib(free)} free, "
            f"below --min-free-gb {args.min_free_gb:g}. "
            "Run cleanup or free space before continuing."
        )


def uiautomator_dump_xml() -> str:
    """
    Returns UI hierarchy XML as a string.
    Dumps under /data/local/tmp instead of /sdcard so low shared-storage
    conditions do not block UI inspection.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(3):
        try:
            adb(["shell", "uiautomator", "dump", UI_DUMP_DEVICE_PATH], timeout=60)
            p = adb(["shell", "cat", UI_DUMP_DEVICE_PATH], timeout=60)
            xml = p.stdout
            if "<hierarchy" in xml:
                return xml
            last_error = RuntimeError(
                "uiautomator dump did not return hierarchy XML. "
                "Make sure the device is unlocked."
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc

        time.sleep(1 + attempt)

    if last_error:
        raise last_error
    raise RuntimeError("uiautomator dump failed for an unknown reason.")


@dataclass(frozen=True)
class UiNode:
    text: str
    resource_id: str
    class_name: str
    content_desc: str
    bounds: Tuple[int, int, int, int]  # left, top, right, bottom


def parse_bounds(bounds_str: str) -> Tuple[int, int, int, int]:
    # bounds like: "[0,123][1080,456]"
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str.strip())
    if not m:
        return (0, 0, 0, 0)
    return tuple(int(x) for x in m.groups())  # type: ignore


def iter_nodes(xml: str) -> Iterable[UiNode]:
    root = ET.fromstring(xml)
    for elem in root.iter():
        if elem.tag != "node":
            continue
        text = elem.attrib.get("text", "") or ""
        rid = elem.attrib.get("resource-id", "") or ""
        cls = elem.attrib.get("class", "") or ""
        desc = elem.attrib.get("content-desc", "") or ""
        bounds = parse_bounds(elem.attrib.get("bounds", "") or "")
        yield UiNode(
            text=text, resource_id=rid, class_name=cls, content_desc=desc, bounds=bounds
        )


# Heuristics: what "looks like highlight text"
UI_CHROME_EXACT = {
    "Search",
    "Library",
    "Home",
    "Shop",
    "Read now",
    "More options",
    "Back",
    "Done",
    "Cancel",
    "OK",
    "Highlights",
    "Notes",
    "Notes & highlights",
    "Notes and highlights",
    "Chapters",
    "Bookmarks",
}
UI_CHROME_REGEX = [
    re.compile(r"^\d+\s+of\s+\d+$", re.IGNORECASE),
    re.compile(r"^Page\s+\d+", re.IGNORECASE),
    re.compile(r"^Chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)?$", re.IGNORECASE),
    re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", re.IGNORECASE),
]


def is_probably_chrome(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return True
    if s2 in UI_CHROME_EXACT:
        return True
    if len(s2) <= 2:
        return True
    for rgx in UI_CHROME_REGEX:
        if rgx.search(s2):
            return True
    return False


def normalize_text(s: str) -> str:
    # Collapse whitespace, keep punctuation.
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# UI wait tuning
UI_WAIT_SHORT = 0.35
UI_WAIT_MED = 0.6
UI_WAIT_LONG = 0.9
UI_DUMP_DEVICE_PATH = "/data/local/tmp/play_books_ua_dump.xml"
LIBRARY_CONTENT_TOP_FRACTION = 0.2
LIBRARY_CONTENT_BOTTOM_FRACTION = 0.875


def title_matches_target(title: str) -> bool:
    t = normalize_text(title).lower()
    return any(substr.lower() in t for substr in TARGET_TITLE_SUBSTRINGS)


def book_key(title: str) -> str:
    return normalize_text(title).lower()


def title_is_always_skipped(title: str) -> bool:
    t = book_key(title)
    return any(substr in t for substr in ALWAYS_SKIP_TITLE_SUBSTRINGS)


def is_above_library_bottom_overlay(
    bounds: Tuple[int, int, int, int], screen_height: int
) -> bool:
    return center_y(bounds) < int(screen_height * LIBRARY_CONTENT_BOTTOM_FRACTION)


def is_library_content_bounds(
    bounds: Tuple[int, int, int, int], screen_height: int
) -> bool:
    cy = center_y(bounds)
    return cy >= int(
        screen_height * LIBRARY_CONTENT_TOP_FRACTION
    ) and is_above_library_bottom_overlay(bounds, screen_height)


def configured_title_substrings(args: argparse.Namespace) -> List[str]:
    only = [
        normalize_text(item).lower() for item in getattr(args, "only_title", []) or []
    ]
    if only:
        return [item for item in only if item]
    extras = [
        normalize_text(item).lower()
        for item in getattr(args, "include_title", []) or []
    ]
    return TARGET_TITLE_SUBSTRINGS + [item for item in extras if item]


def title_is_selected(title: str, args: argparse.Namespace) -> bool:
    if getattr(args, "no_title_filter", False):
        return True
    t = normalize_text(title).lower()
    return any(substr in t for substr in configured_title_substrings(args))


def parse_log_title(raw_title: str) -> Optional[str]:
    raw_title = raw_title.strip()
    try:
        parsed = ast.literal_eval(raw_title)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, str):
        return parsed
    return None


def load_titles_from_run_log(path: str) -> Set[str]:
    titles: Set[str] = set()
    download_pattern = re.compile(r"^Download state before export for (.+): ")
    skip_pattern = re.compile(r"^Skipping (.+): ")
    no_highlights_pattern = re.compile(r"^No highlights found for (.+)\.$")
    wrote_pattern = re.compile(r"^Wrote: .+\.jsonl$")
    pending_title: Optional[str] = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = download_pattern.search(line)
            if match:
                pending_title = parse_log_title(match.group(1))
                continue

            match = skip_pattern.search(line)
            if match:
                title = parse_log_title(match.group(1))
                if title:
                    titles.add(book_key(title))
                pending_title = None
                continue

            match = no_highlights_pattern.search(line)
            if match:
                title = parse_log_title(match.group(1))
                if title:
                    titles.add(book_key(title))
                pending_title = None
                continue

            if pending_title and wrote_pattern.search(line):
                titles.add(book_key(pending_title))
                pending_title = None
    return titles


def load_new_download_titles_and_cleaned_keys(
    path: str,
) -> Tuple[dict[str, str], Set[str]]:
    titles_by_key: dict[str, str] = {}
    cleaned_keys: Set[str] = set()
    patterns_new = (
        re.compile(r"^Download state before export for (.+): not downloaded\.$"),
    )
    patterns_cleaned = (
        re.compile(r"^Removed local download for (.+)\.$"),
        re.compile(r"^No local download left for (.+)\.$"),
    )

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            for pattern in patterns_new:
                match = pattern.search(line)
                if not match:
                    continue
                title = parse_log_title(match.group(1))
                if title:
                    key = book_key(title)
                    titles_by_key[key] = title
                    cleaned_keys.discard(key)
                break
            for pattern in patterns_cleaned:
                match = pattern.search(line)
                if not match:
                    continue
                title = parse_log_title(match.group(1))
                if title:
                    cleaned_keys.add(book_key(title))
                break

    return titles_by_key, cleaned_keys


def load_unremoved_new_download_titles_from_run_log(path: str) -> List[str]:
    titles_by_key, cleaned_keys = load_new_download_titles_and_cleaned_keys(path)
    return [title for key, title in titles_by_key.items() if key not in cleaned_keys]


def compact_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", book_key(title))


def title_keys_match(left: str, right: str) -> bool:
    left_key = book_key(left)
    right_key = book_key(right)
    if left_key == right_key:
        return True
    if min(len(left_key), len(right_key)) >= 24 and (
        left_key in right_key or right_key in left_key
    ):
        return True

    left_compact = compact_title_key(left)
    right_compact = compact_title_key(right)
    if not left_compact or not right_compact:
        return False
    prefix_len = min(len(left_compact), len(right_compact), 48)
    return prefix_len >= 24 and left_compact[:prefix_len] == right_compact[:prefix_len]


def matching_title_key(title: str, target_titles: dict[str, str]) -> Optional[str]:
    for key, target_title in target_titles.items():
        if title_keys_match(title, target_title):
            return key

    visible_compact = compact_title_key(title)
    if len(visible_compact) < 10:
        return None

    prefix_matches = [
        key
        for key, target_title in target_titles.items()
        if compact_title_key(target_title).startswith(visible_compact)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def matching_cleanup_title_key(
    title: str, target_titles: dict[str, str]
) -> Optional[str]:
    for key, target_title in target_titles.items():
        if title_keys_match(title, target_title):
            return key
    return None


def tap(x: int, y: int) -> None:
    adb(["shell", "input", "tap", str(x), str(y)])


def tap_bounds(bounds: Tuple[int, int, int, int]) -> None:
    left, top, right, bottom = bounds
    tap((left + right) // 2, (top + bottom) // 2)


def long_press_bounds(
    bounds: Tuple[int, int, int, int], duration_ms: int = 700
) -> None:
    left, top, right, bottom = bounds
    x = (left + right) // 2
    y = (top + bottom) // 2
    adb(
        [
            "shell",
            "input",
            "swipe",
            str(x),
            str(y),
            str(x),
            str(y),
            str(duration_ms),
        ]
    )
    time.sleep(UI_WAIT_MED)


def press_back() -> None:
    adb(["shell", "input", "keyevent", "4"])


def launch_play_books() -> None:
    try:
        adb(
            [
                "shell",
                "am",
                "start",
                "-n",
                "com.google.android.apps.books/"
                "com.google.android.apps.play.books.home.HomeActivity",
            ],
            timeout=10,
        )
    except subprocess.CalledProcessError:
        adb(
            [
                "shell",
                "monkey",
                "-p",
                "com.google.android.apps.books",
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ]
        )
    time.sleep(UI_WAIT_LONG)


def is_notes_list_screen(xml: str) -> bool:
    texts = {normalize_text(n.text) for n in iter_nodes(xml) if normalize_text(n.text)}
    if "Add your own thoughts" in texts and any(
        text.startswith("Touch & hold the text") for text in texts
    ):
        return True
    for n in iter_nodes(xml):
        if n.resource_id.endswith(":id/notes_list_view"):
            return True
        if n.resource_id.endswith(":id/snippet"):
            return True
    return False


def is_library_screen(xml: str) -> bool:
    for n in iter_nodes(xml):
        if n.resource_id.endswith(":id/library_root"):
            return True
        if n.resource_id.endswith(":id/books_view_pager"):
            return True
        if n.resource_id.endswith(":id/library_tabs"):
            return True
    return False


def tap_by_text_or_desc(xml: str, labels: Iterable[str]) -> bool:
    targets = {label.lower() for label in labels}
    matches: List[UiNode] = []
    for n in iter_nodes(xml):
        txt = normalize_text(n.text).lower() if n.text else ""
        desc = normalize_text(n.content_desc).lower() if n.content_desc else ""
        if txt in targets or desc in targets:
            matches.append(n)
    if not matches:
        return False
    matches.sort(key=lambda n: center_y(n.bounds))
    tap_bounds(matches[0].bounds)
    return True


def tap_by_text_or_desc_contains(
    xml: str, labels: Iterable[str], prefer_bottom: bool = False
) -> bool:
    targets = [label.lower() for label in labels]
    matches: List[UiNode] = []
    for n in iter_nodes(xml):
        txt = normalize_text(n.text).lower() if n.text else ""
        desc = normalize_text(n.content_desc).lower() if n.content_desc else ""
        rid = (n.resource_id or "").lower()
        for label in targets:
            if label in txt or label in desc or label in rid:
                matches.append(n)
                break
    if not matches:
        return False
    matches.sort(key=lambda n: center_y(n.bounds), reverse=prefer_bottom)
    tap_bounds(matches[0].bounds)
    return True


def find_bottom_nav_tab(xml: str, screen_height: int, label: str) -> Optional[UiNode]:
    label_key = normalize_text(label).lower()
    candidates: List[UiNode] = []
    for n in iter_nodes(xml):
        cy = center_y(n.bounds)
        if cy < int(screen_height * 0.72):
            continue
        txt = normalize_text(n.text).lower() if n.text else ""
        desc = normalize_text(n.content_desc).lower() if n.content_desc else ""
        if txt == label_key or desc == label_key:
            candidates.append(n)
            continue
        # Selected bottom tabs often expose combined accessibility labels.
        if desc.startswith(label_key + ","):
            candidates.append(n)
    if not candidates:
        return None
    candidates.sort(key=lambda n: center_y(n.bounds), reverse=True)
    return candidates[0]


def tap_bottom_nav_tab(xml: str, screen_height: int, label: str) -> bool:
    node = find_bottom_nav_tab(xml, screen_height, label)
    if node is None:
        return False
    tap_bounds(node.bounds)
    return True


def xml_has_text_or_desc_contains(xml: str, labels: Iterable[str]) -> bool:
    targets = [label.lower() for label in labels]
    for n in iter_nodes(xml):
        txt = normalize_text(n.text).lower() if n.text else ""
        desc = normalize_text(n.content_desc).lower() if n.content_desc else ""
        rid = (n.resource_id or "").lower()
        for label in targets:
            if label in txt or label in desc or label in rid:
                return True
    return False


def open_library_tab(width: int, height: int, max_tries: int = 10) -> Optional[str]:
    for attempt in range(max_tries):
        xml = uiautomator_dump_xml()
        tapped = tap_bottom_nav_tab(xml, height, "Library")
        if not tapped:
            tapped = tap_by_text_or_desc(xml, ["Library"])
        if tapped:
            time.sleep(UI_WAIT_MED)
            xml = uiautomator_dump_xml()
            xml = ensure_your_books_tab(xml)
            books = find_library_books(xml, height)
            if is_library_screen(xml) and books:
                return xml

        books = find_library_books(xml, height)
        if is_library_screen(xml) and books:
            xml = ensure_your_books_tab(xml)
            return xml

        if attempt == max_tries // 2:
            launch_play_books()
        else:
            press_back()
        time.sleep(UI_WAIT_SHORT)
    return None


def ensure_library_screen(width: int, height: int, max_tries: int = 8) -> Optional[str]:
    for attempt in range(max_tries):
        xml = uiautomator_dump_xml()
        books = find_library_books(xml, height)
        if is_library_screen(xml) and books:
            return xml
        if books and tap_by_text_or_desc_contains(xml, ["library"], prefer_bottom=True):
            # If the Library tab exists, tap it to be safe and re-check.
            time.sleep(UI_WAIT_SHORT)
            xml = uiautomator_dump_xml()
            books = find_library_books(xml, height)
        if books:
            return xml
        if tap_by_text_or_desc(xml, ["Library"]):
            time.sleep(UI_WAIT_MED)
            xml = uiautomator_dump_xml()
            books = find_library_books(xml, height)
            if books:
                return xml
            continue
        if tap_bottom_nav_tab(xml, height, "Library"):
            time.sleep(UI_WAIT_MED)
            xml = uiautomator_dump_xml()
            books = find_library_books(xml, height)
            if books:
                return xml
            continue
        if tap_by_text_or_desc_contains(xml, ["library"], prefer_bottom=True):
            time.sleep(UI_WAIT_MED)
            xml = uiautomator_dump_xml()
            books = find_library_books(xml, height)
            if books:
                return xml
            continue
        if attempt == max_tries // 2:
            launch_play_books()
        else:
            press_back()
        time.sleep(UI_WAIT_SHORT)
    return None


LIBRARY_SORT_LABELS = {
    "recent": ("Most recent", "Recent"),
    "title": ("Title", "A-Z", "Alphabetical"),
}
DOWNLOAD_STATUS_FILTER_LABELS = {
    "all": ("All",),
    "downloaded": ("Downloaded",),
    "not-downloaded": ("Not downloaded",),
}


def ensure_your_books_tab(xml: str) -> str:
    if tap_by_text_or_desc(xml, ["Your books"]):
        time.sleep(UI_WAIT_MED)
        return uiautomator_dump_xml()
    return xml


def library_filters_visible(xml: str) -> bool:
    return xml_has_text_or_desc_contains(xml, ["library filters", "sort filter"])


def library_page_signature(xml: str, height: int) -> Tuple[str, ...]:
    return tuple(book_key(title) for title, _ in find_library_books(xml, height)[:10])


def scroll_library_to_top(width: int, height: int, max_swipes: int = 60) -> str:
    xml = uiautomator_dump_xml()
    stable_pages = 0
    for _ in range(max_swipes):
        if not is_library_screen(xml):
            return xml
        before = library_page_signature(xml, height)
        x = width // 2
        y1 = int(height * 0.25)
        y2 = int(height * 0.88)
        adb(["shell", "input", "swipe", str(x), str(y1), str(x), str(y2), "450"])
        time.sleep(UI_WAIT_SHORT)
        next_xml = uiautomator_dump_xml()
        after = library_page_signature(next_xml, height)
        if before and after == before:
            stable_pages += 1
            if stable_pages >= 2:
                return next_xml
        else:
            stable_pages = 0
        xml = next_xml
    return xml


def ensure_download_status_filter(xml: str, status: str) -> bool:
    if status == "current":
        return True
    if status == "all" and not xml_has_text_or_desc_contains(
        xml, ["download status filter"]
    ):
        return True

    labels = DOWNLOAD_STATUS_FILTER_LABELS.get(status)
    if not labels:
        raise ValueError(f"Unknown download status filter: {status!r}")

    if not tap_by_text_or_desc_contains(xml, ["download status filter"]):
        return status == "all"

    time.sleep(UI_WAIT_MED)
    xml2 = uiautomator_dump_xml()
    if tap_by_text_or_desc(xml2, labels):
        time.sleep(UI_WAIT_MED)
        return True
    if tap_by_text_or_desc_contains(xml2, labels):
        time.sleep(UI_WAIT_MED)
        return True

    press_back()
    time.sleep(UI_WAIT_SHORT)
    return False


def ensure_library_sort(xml: str, sort_name: str) -> bool:
    if sort_name == "none":
        return True

    labels = LIBRARY_SORT_LABELS.get(sort_name)
    if not labels:
        raise ValueError(f"Unknown library sort: {sort_name!r}")

    if tap_by_text_or_desc(xml, ["Sort", "Sort by", "More options"]) or (
        tap_by_text_or_desc_contains(xml, ["sort filter"])
    ):
        time.sleep(UI_WAIT_MED)
        xml2 = uiautomator_dump_xml()
        if tap_by_text_or_desc(xml2, labels):
            time.sleep(UI_WAIT_MED)
            return True
        if tap_by_text_or_desc_contains(xml2, labels):
            time.sleep(UI_WAIT_MED)
            return True
        press_back()
        time.sleep(UI_WAIT_SHORT)
    return False


def ensure_sort_most_recent(xml: str) -> bool:
    return ensure_library_sort(xml, "recent")


def detect_book_title(xml: str, screen_height: int) -> Optional[str]:
    cutoff = min(220, int(screen_height * 0.2))
    candidates: List[Tuple[int, int, str]] = []
    for n in iter_nodes(xml):
        if not n.text:
            continue
        txt = normalize_text(n.text)
        if not txt or is_probably_chrome(txt):
            continue
        cy = center_y(n.bounds)
        if cy > cutoff:
            continue
        candidates.append((len(txt), cy, txt))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][2]


def book_title_from_card_desc(desc: str) -> str:
    title = normalize_text(desc)
    title = re.sub(r", \d+% complete(?:, \d+% downloaded)?$", "", title)
    title = re.sub(r", \d+% downloaded$", "", title)
    title = re.sub(r", (?:not )?downloaded$", "", title, flags=re.IGNORECASE)
    title = re.sub(r", download$", "", title, flags=re.IGNORECASE)
    return title.strip()


def progress_percent_from_text(text: str) -> Optional[int]:
    match = re.search(r"\b(\d{1,3})%\s+complete\b", normalize_text(text), re.I)
    if not match:
        return None
    return min(int(match.group(1)), 100)


def horizontal_overlap(
    left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]
) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0]))


def progress_percent_for_title(
    xml: str, screen_height: int, book_title: str
) -> Optional[int]:
    for n in iter_nodes(xml):
        if not n.resource_id.endswith(":id/book_card_root") or not n.content_desc:
            continue
        title = book_title_from_card_desc(n.content_desc)
        if title_keys_match(title, book_title):
            progress = progress_percent_from_text(n.content_desc)
            if progress is not None:
                return progress

    bounds = find_library_book_bounds(xml, screen_height, book_title)
    if not bounds:
        return None

    candidates: List[Tuple[int, int]] = []
    _, top, _, bottom = bounds
    for n in iter_nodes(xml):
        progress = progress_percent_from_text(f"{n.text} {n.content_desc}")
        if progress is None:
            continue
        if horizontal_overlap(bounds, n.bounds) <= 0:
            continue
        cy = center_y(n.bounds)
        if top - 8 <= cy <= bottom + 140:
            candidates.append((abs(cy - center_y(bounds)), progress))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def find_library_books(
    xml: str, screen_height: int
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    if not is_library_screen(xml):
        return []

    candidates: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
    desc_candidates: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
    id_candidates: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
    list_title_candidates: List[Tuple[int, str, Tuple[int, int, int, int]]] = []
    download_status_tops = {
        n.bounds[1]
        for n in iter_nodes(xml)
        if is_library_download_status(n.content_desc)
        and is_library_content_bounds(n.bounds, screen_height)
    }
    for n in iter_nodes(xml):
        if n.resource_id.endswith(":id/book_card_root") and n.content_desc:
            if not is_above_library_bottom_overlay(n.bounds, screen_height):
                continue
            title = book_title_from_card_desc(n.content_desc)
            if title and not is_probably_chrome(title):
                desc_candidates.append((center_y(n.bounds), title, n.bounds))
                continue
        if n.class_name != "android.widget.TextView":
            continue
        txt = normalize_text(n.text)
        if not txt or is_probably_chrome(txt):
            continue
        cy = center_y(n.bounds)
        if not is_library_content_bounds(n.bounds, screen_height):
            continue
        if n.resource_id.endswith(":id/title") or n.resource_id.endswith(
            ":id/volume_title"
        ):
            id_candidates.append((cy, txt, n.bounds))
            continue
        if any(abs(n.bounds[1] - row_top) <= 8 for row_top in download_status_tops):
            list_title_candidates.append((cy, txt, n.bounds))
            continue
        # Fallback: accept reasonable-looking titles even without known IDs
        if len(txt) >= 4:
            candidates.append((cy, txt, n.bounds))

    picked = (
        desc_candidates
        if desc_candidates
        else id_candidates
        if id_candidates
        else list_title_candidates or candidates
    )
    picked.sort(key=lambda t: t[0])
    return [(txt, bounds) for _, txt, bounds in picked]


def find_library_book_bounds(
    xml: str, screen_height: int, target_title: str
) -> Optional[Tuple[int, int, int, int]]:
    target = book_key(target_title)
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for title, bounds in find_library_books(xml, screen_height):
        title_key = book_key(title)
        if target == title_key:
            return bounds
        if title_keys_match(title, target_title):
            candidates.append((abs(len(title_key) - len(target)), bounds))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def is_library_download_status(desc: str) -> bool:
    d = normalize_text(desc).lower()
    return d in {"download", "downloaded", "not downloaded"} or "download" in d


def download_state_from_status_desc(desc: str) -> Optional[bool]:
    d = normalize_text(desc).lower()
    if d == "downloaded":
        return True
    if d in {"download", "not downloaded"}:
        return False
    if "downloaded" in d:
        return False
    return None


def download_status_node_for_title(
    xml: str, screen_height: int, book_title: str
) -> Optional[UiNode]:
    bounds = find_library_book_bounds(xml, screen_height, book_title)
    if not bounds:
        return None
    row_top = bounds[1]
    for n in iter_nodes(xml):
        if abs(n.bounds[1] - row_top) > 8:
            continue
        if is_library_download_status(n.content_desc):
            return n
    return None


def download_status_desc_for_title(
    xml: str, screen_height: int, book_title: str
) -> Optional[str]:
    node = download_status_node_for_title(xml, screen_height, book_title)
    if node is None:
        return None
    return normalize_text(node.content_desc)


def detect_download_state_from_library_row(
    xml: str, screen_height: int, book_title: str
) -> Optional[bool]:
    desc = download_status_desc_for_title(xml, screen_height, book_title)
    if desc is None:
        return None
    return download_state_from_status_desc(desc)


def is_plain_not_downloaded_status(desc: str) -> bool:
    return normalize_text(desc).lower() in {"download", "not downloaded"}


def is_local_or_partial_download_status(desc: str) -> bool:
    d = normalize_text(desc).lower()
    return d == "downloaded" or bool(re.search(r"\d+%\s+downloaded", d))


def title_meets_progress_filter(
    xml: str, screen_height: int, title: str, min_progress_percent: int
) -> bool:
    if min_progress_percent <= 0:
        return True
    progress = progress_percent_for_title(xml, screen_height, title)
    if progress is None:
        print(
            f"Skipping {title!r}: no visible progress percent for "
            f"--min-progress-percent {min_progress_percent}."
        )
        return False
    if progress < min_progress_percent:
        print(
            f"Skipping {title!r}: progress {progress}% below "
            f"--min-progress-percent {min_progress_percent}."
        )
        return False
    return True


def find_book_bounds_on_current_library_page(
    title: str,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    xml = ensure_library_screen(width, height)
    if not xml:
        return None
    return find_library_book_bounds(xml, height, title)


def ensure_notes_list(width: int, height: int, max_tries: int = 4) -> Optional[str]:
    for _ in range(max_tries):
        try:
            xml = uiautomator_dump_xml()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"WARN: could not inspect notes list UI: {exc}")
            return None
        if is_notes_list_screen(xml):
            return xml
        if tap_by_text_or_desc(
            xml, ["Notes", "Notes & highlights", "Notes and highlights"]
        ):
            time.sleep(UI_WAIT_MED)
            continue
        if tap_by_text_or_desc(xml, ["Table of contents", "Contents", "TOC"]):
            time.sleep(UI_WAIT_MED)
            continue
        tap(width // 2, height // 4)
        time.sleep(UI_WAIT_SHORT)
    return None


def center_y(bounds: Tuple[int, int, int, int]) -> int:
    _, top, _, bottom = bounds
    return (top + bottom) // 2


def is_relevant_resource_id(rid: str) -> bool:
    return rid.endswith(":id/snippet") or rid.endswith(":id/note")


@dataclass
class ExtractedItem:
    highlight: str
    note: Optional[str] = None


def extract_highlights_and_notes(xml: str) -> List[ExtractedItem]:
    """
    Extract (highlight snippet, optional note) pairs from the highlights list UI.

    Heuristic:
      - Collect all nodes with resource-id ending in :id/snippet or :id/note
      - Sort by vertical position (top to bottom)
      - When we see a snippet, start a new item
      - When we see a note, attach it to the most recent snippet above it
    """
    nodes = []
    for n in iter_nodes(xml):
        if not n.text:
            continue
        if not is_relevant_resource_id(n.resource_id):
            continue
        txt = normalize_text(n.text)
        if not txt:
            continue
        nodes.append((center_y(n.bounds), n.resource_id, txt))

    nodes.sort(key=lambda t: t[0])  # sort top->bottom

    items: List[ExtractedItem] = []
    for _, rid, txt in nodes:
        if rid.endswith(":id/snippet"):
            # Start a new highlight item
            items.append(ExtractedItem(highlight=txt, note=None))
        elif rid.endswith(":id/note"):
            # Attach to most recent highlight if it exists
            if items:
                # If multiple notes occur, concatenate with newline
                if items[-1].note:
                    items[-1].note = items[-1].note + "\n" + txt
                else:
                    items[-1].note = txt
            else:
                # Edge case: note appears without a snippet; keep it anyway
                items.append(ExtractedItem(highlight="", note=txt))

    # Drop empty highlight-only artifacts if desired (rare)
    items = [it for it in items if it.highlight or it.note]

    # Deduplicate within a page (Play Books sometimes repeats visible nodes)
    seen: Set[Tuple[str, Optional[str]]] = set()
    out: List[ExtractedItem] = []
    for it in items:
        key = (it.highlight, it.note)
        if key not in seen:
            seen.add(key)
            out.append(it)

    return out


def swipe_notes_scroll(width: int, height: int) -> None:
    """
    Swipe up to scroll down a notes/highlights list.
    """
    x = width // 2
    y1 = int(height * 0.78)
    y2 = int(height * 0.22)
    duration_ms = 350
    adb(["shell", "input", "swipe", str(x), str(y1), str(x), str(y2), str(duration_ms)])
    time.sleep(0.6)  # allow render


def swipe_library_scroll(width: int, height: int) -> None:
    """
    Use a short overlapping Library scroll so no book rows are skipped between dumps.
    """
    x = width // 2
    y1 = int(height * 0.72)
    y2 = int(height * 0.50)
    duration_ms = 350
    adb(["shell", "input", "swipe", str(x), str(y1), str(x), str(y2), str(duration_ms)])
    time.sleep(0.6)  # allow render


def maybe_jiggle_scroll(width: int, height: int) -> None:
    """
    Tiny scroll to trigger lazy loading / re-render if needed.
    """
    x = width // 2
    y1 = int(height * 0.55)
    y2 = int(height * 0.50)
    adb(["shell", "input", "swipe", str(x), str(y1), str(x), str(y2), "120"])
    time.sleep(0.4)


REMOVE_DOWNLOAD_LABELS = (
    "cancel download",
    "cancel downloading",
    "remove download",
    "remove from device",
    "remove downloaded",
    "delete download",
    "delete from device",
)
DOWNLOAD_LABELS = (
    "download",
    "download book",
    "download to device",
    "make available offline",
    "save offline",
)
MORE_OPTIONS_LABELS = ("more options", "options", "menu")


def infer_download_state_from_menu_xml(xml: str) -> Optional[bool]:
    if xml_has_text_or_desc_contains(xml, REMOVE_DOWNLOAD_LABELS):
        return True
    if xml_has_text_or_desc_contains(xml, DOWNLOAD_LABELS):
        return False
    return None


def detect_download_state_for_book(
    book_title: str,
    width: int,
    height: int,
    max_tries: int = 2,
) -> Optional[bool]:
    xml = ensure_library_screen(width, height)
    if not xml:
        print(
            f"WARN: could not return to Library to detect download state for {book_title!r}."
        )
        return None

    bounds = find_library_book_bounds(xml, height, book_title)
    if not bounds:
        print(
            f"WARN: could not find {book_title!r} in Library to detect download state."
        )
        return None

    row_state = detect_download_state_from_library_row(xml, height, book_title)
    if row_state is not None:
        return row_state

    for _ in range(max_tries):
        long_press_bounds(bounds)
        xml_menu = uiautomator_dump_xml()
        state = infer_download_state_from_menu_xml(xml_menu)
        if state is not None:
            press_back()
            time.sleep(UI_WAIT_SHORT)
            return state

        if tap_by_text_or_desc_contains(xml_menu, MORE_OPTIONS_LABELS):
            time.sleep(UI_WAIT_MED)
            xml_more = uiautomator_dump_xml()
            state = infer_download_state_from_menu_xml(xml_more)
            press_back()
            time.sleep(UI_WAIT_SHORT)
            if state is not None:
                return state

        press_back()
        time.sleep(UI_WAIT_SHORT)
        xml = ensure_library_screen(width, height)
        if not xml:
            return None
        bounds = find_library_book_bounds(xml, height, book_title)
        if not bounds:
            return None

    return None


def maybe_remove_download(
    book_title: str,
    width: int,
    height: int,
    max_tries: int = 2,
) -> bool:
    xml = ensure_library_screen(width, height)
    if not xml:
        print(
            f"WARN: could not return to Library to remove download for {book_title!r}."
        )
        return False

    bounds = find_library_book_bounds(xml, height, book_title)
    if not bounds:
        print(f"WARN: could not find {book_title!r} in Library to remove download.")
        return False

    status_node = download_status_node_for_title(xml, height, book_title)
    if status_node:
        if is_plain_not_downloaded_status(status_node.content_desc):
            print(f"No local download left for {book_title!r}.")
            return True
        if is_local_or_partial_download_status(status_node.content_desc):
            tap_bounds(status_node.bounds)
            time.sleep(UI_WAIT_MED)
            xml_confirm = uiautomator_dump_xml()
            if tap_by_text_or_desc(xml_confirm, ["Remove", "Delete", "OK"]):
                time.sleep(UI_WAIT_MED)
                print(f"Removed local download for {book_title!r}.")
                return True
            press_back()
            time.sleep(UI_WAIT_SHORT)
            xml = ensure_library_screen(width, height)
            if not xml:
                return False
            bounds = find_library_book_bounds(xml, height, book_title)
            if not bounds:
                return False

    for attempt in range(max_tries):
        long_press_bounds(bounds)
        xml = uiautomator_dump_xml()
        state = infer_download_state_from_menu_xml(xml)
        if state is False:
            press_back()
            time.sleep(UI_WAIT_SHORT)
            print(f"No local download left for {book_title!r}.")
            return True
        if tap_by_text_or_desc_contains(xml, REMOVE_DOWNLOAD_LABELS):
            time.sleep(UI_WAIT_MED)
            xml_confirm = uiautomator_dump_xml()
            tap_by_text_or_desc(xml_confirm, ["Remove", "Delete", "OK"])
            time.sleep(UI_WAIT_MED)
            print(f"Removed local download for {book_title!r}.")
            return True
        if tap_by_text_or_desc_contains(xml, MORE_OPTIONS_LABELS):
            time.sleep(UI_WAIT_MED)
            xml_menu = uiautomator_dump_xml()
            if tap_by_text_or_desc_contains(xml_menu, REMOVE_DOWNLOAD_LABELS):
                time.sleep(UI_WAIT_MED)
                xml_confirm = uiautomator_dump_xml()
                tap_by_text_or_desc(xml_confirm, ["Remove", "Delete", "OK"])
                time.sleep(UI_WAIT_MED)
                print(f"Removed local download for {book_title!r}.")
                return True
        press_back()
        time.sleep(UI_WAIT_SHORT)
        xml = ensure_library_screen(width, height)
        if not xml:
            break
        bounds = find_library_book_bounds(xml, height, book_title)
        if not bounds:
            break

    print(
        f"WARN: could not find a remove-download action for {book_title!r}. "
        "Check the Play Books UI wording on the device."
    )
    return False


def cleanup_download_after_book(
    args: argparse.Namespace,
    book_title: str,
    width: int,
    height: int,
    was_downloaded: Optional[bool],
) -> None:
    if args.remove_downloads:
        maybe_remove_download(book_title, width, height)
    elif args.remove_new_downloads and was_downloaded is False:
        xml = ensure_library_screen(width, height)
        if xml:
            desc = download_status_desc_for_title(xml, height, book_title)
            if desc and is_plain_not_downloaded_status(desc):
                print(f"No local download left for {book_title!r}.")
                return
        maybe_remove_download(book_title, width, height)


def return_to_library_after_book(width: int, height: int) -> None:
    press_back()
    time.sleep(UI_WAIT_SHORT)
    press_back()
    time.sleep(UI_WAIT_SHORT)
    try:
        tap_by_text_or_desc(uiautomator_dump_xml(), ["Library"])
        time.sleep(UI_WAIT_SHORT)
    except Exception as exc:
        print(f"WARN: could not tap Library after leaving book: {exc}")
    ensure_library_screen(width, height)


def cleanup_new_downloads_from_logs(
    args: argparse.Namespace, width: int, height: int
) -> None:
    target_titles: dict[str, str] = {}
    cleaned_keys: Set[str] = set()
    for log_path in args.cleanup_new_downloads_from_log:
        titles_by_key, log_cleaned_keys = load_new_download_titles_and_cleaned_keys(
            log_path
        )
        target_titles.update(titles_by_key)
        cleaned_keys.update(log_cleaned_keys)
        print(
            f"Loaded {len(titles_by_key)} new-download titles and "
            f"{len(log_cleaned_keys)} removals from {log_path}."
        )

    target_titles = {
        key: title for key, title in target_titles.items() if key not in cleaned_keys
    }

    if not target_titles:
        print("No unremoved new-download titles found in the provided log files.")
        return

    xml = open_library_tab(width, height)
    if not xml:
        launch_play_books()
        xml = open_library_tab(width, height)
    if not xml:
        print("WARN: could not reach Library for cleanup.")
        return

    xml = ensure_your_books_tab(xml)
    if args.resume_from_current_position:
        print(
            "Resuming cleanup scan from the current Library scroll position; "
            "assuming existing filters and sort are already correct."
        )
    else:
        xml = scroll_library_to_top(width, height)
        if not ensure_download_status_filter(xml, args.download_status):
            print(
                f"WARN: could not set download status filter to {args.download_status!r}."
            )
        xml = ensure_library_screen(width, height) or xml
        if not ensure_library_sort(xml, args.library_sort):
            print(f"WARN: could not set library sort to {args.library_sort!r}.")

    cleaned: Set[str] = set()
    failed: Set[str] = set()
    last_signature: Optional[Tuple[str, ...]] = None
    stalls = 0

    for page_i in range(args.max_swipes):
        xml = ensure_library_screen(width, height)
        if not xml:
            launch_play_books()
            xml = ensure_library_screen(width, height)
        if not xml:
            print("WARN: could not reach Library during cleanup scan; retrying.")
            stalls += 1
            if stalls >= args.stall_pages:
                break
            continue

        books = find_library_books(xml, height)
        visible_titles = [title for title, _ in books]
        signature = tuple(visible_titles[:10])
        remaining = {
            key: title
            for key, title in target_titles.items()
            if key not in cleaned and key not in failed
        }
        if not remaining:
            break

        removed_on_page = False
        for visible_title in visible_titles:
            key = matching_cleanup_title_key(visible_title, remaining)
            if not key:
                continue
            if maybe_remove_download(visible_title, width, height):
                cleaned.add(key)
                removed_on_page = True
                break
            failed.add(key)

        if removed_on_page:
            stalls = 0
            time.sleep(UI_WAIT_MED)
            continue

        if signature == last_signature:
            stalls += 1
        else:
            stalls = 0
            last_signature = signature

        print(
            f"[cleanup {page_i:03d}] visible={len(visible_titles)} "
            f"cleaned={len(cleaned)} remaining={len(remaining)} "
            f"stalls={stalls}"
        )
        if stalls >= args.stall_pages:
            break
        swipe_library_scroll(width, height)

    remaining_titles = [
        title
        for key, title in target_titles.items()
        if key not in cleaned and key not in failed
    ]
    print(
        f"Cleanup summary: removed={len(cleaned)}, failed={len(failed)}, "
        f"not_seen={len(remaining_titles)}."
    )
    if remaining_titles:
        for title in remaining_titles[:30]:
            print(f"Cleanup not seen: {title!r}")


def output_paths(book: str, outdir: str) -> Tuple[str, str]:
    base = safe_filename(book) or "Unknown Book"
    md_path = os.path.join(outdir, f"{base}.md")
    jsonl_path = os.path.join(outdir, f"{base}.jsonl")
    if not os.path.exists(md_path) and not os.path.exists(jsonl_path):
        return md_path, jsonl_path

    for i in range(2, 1000):
        numbered_base = f"{base} ({i})"
        md_path = os.path.join(outdir, f"{numbered_base}.md")
        jsonl_path = os.path.join(outdir, f"{numbered_base}.jsonl")
        if not os.path.exists(md_path) and not os.path.exists(jsonl_path):
            return md_path, jsonl_path

    raise RuntimeError(f"Could not find an unused output filename for {book!r}")


def write_outputs(book: str, outdir: str, items: List[ExtractedItem]) -> None:
    os.makedirs(outdir, exist_ok=True)

    md_path, jsonl_path = output_paths(book, outdir)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {book}\n\n")
        for i, it in enumerate(items, start=1):
            f.write(f"## Highlight {i}\n\n")
            if it.highlight:
                f.write("> " + it.highlight + "\n\n")
            if it.note:
                f.write("**Note:** " + it.note + "\n\n")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(
                json.dumps(
                    {"book": book, "highlight": it.highlight, "note": it.note},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote: {md_path}")
    print(f"Wrote: {jsonl_path}")


def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\- .()]+", "_", s)
    s = re.sub(r"\s+", " ", s)
    return s[:140] if len(s) > 140 else s


def extract_book_highlights(
    args: argparse.Namespace,
    width: int,
    height: int,
    book: str,
    initial_xml: Optional[str] = None,
) -> List[ExtractedItem]:
    all_highlights: List[ExtractedItem] = []
    seen: Set[Tuple[str, str]] = set()
    stalls = 0

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    for swipe_i in range(args.max_swipes):
        xml = initial_xml if swipe_i == 0 and initial_xml else uiautomator_dump_xml()

        if args.debug_dumps:
            dump_path = os.path.join(
                outdir, f"{safe_filename(book)}__dump_{swipe_i:04d}.xml"
            )
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(xml)

        items = extract_highlights_and_notes(xml)

        new_count = 0
        for it in items:
            key = (it.highlight, it.note or "")
            if key not in seen:
                seen.add(key)
                all_highlights.append(it)
                new_count += 1

        print(
            f"[{swipe_i:03d}] +{new_count} new (total {len(all_highlights)}), stalls={stalls}"
        )

        if new_count == 0:
            stalls += 1
            # Sometimes Play Books needs a tiny "jiggle" to load next items
            if stalls == 2:
                maybe_jiggle_scroll(width, height)
        else:
            stalls = 0

        if stalls >= args.stall_pages:
            print("Stopping: no new highlights after repeated scrolls.")
            break

        swipe_notes_scroll(width, height)

    return all_highlights


def run_all_books(args: argparse.Namespace, width: int, height: int) -> None:
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    enforce_min_free_space(args, "start all-books extraction")

    seen_titles: Set[str] = set()
    seen_targets: Set[str] = set()
    prior_new_downloads: Set[str] = set()
    last_no_new_signature: Optional[Tuple[str, ...]] = None
    stalls = 0
    processed_books = 0
    batch_start_free = get_data_free_bytes()
    print(f"Batch starting free space: {format_gib(batch_start_free)}.")

    for log_path in getattr(args, "skip_titles_from_log", []) or []:
        loaded = load_titles_from_run_log(log_path)
        seen_targets.update(loaded)
        new_downloads, cleaned_downloads = load_new_download_titles_and_cleaned_keys(
            log_path
        )
        prior_new_downloads.update(set(new_downloads) - cleaned_downloads)
        print(f"Loaded {len(loaded)} previously attempted titles from {log_path}.")

    xml = open_library_tab(width, height)
    if not xml:
        print("Could not reach Library screen. Aborting all-books run.")
        return

    xml = ensure_your_books_tab(xml)

    if args.resume_from_current_position:
        print(
            "Resuming Library scan from the current scroll position; "
            "assuming existing filters and sort are already correct."
        )
    else:
        xml = scroll_library_to_top(width, height)
        if not ensure_download_status_filter(xml, args.download_status):
            print(
                f"WARN: could not set Download status filter to {args.download_status!r}; "
                "continuing with the current filter."
            )
        else:
            xml = uiautomator_dump_xml()
            xml = scroll_library_to_top(width, height)

        if not ensure_library_sort(xml, args.library_sort):
            print(
                f"WARN: could not set Library sort to {args.library_sort!r}; "
                "continuing with the current sort."
            )
        else:
            scroll_library_to_top(width, height)

    for page_i in range(args.max_swipes):
        xml = ensure_library_screen(width, height)
        if not xml:
            print("WARN: could not reach Library during scan; retrying.")
            continue
        books = find_library_books(xml, height)
        # Remember everything we've seen to avoid ping-pong in "most recent" sorting.
        for title, _ in books:
            seen_titles.add(book_key(title))
        new_books: List[Tuple[str, Tuple[int, int, int, int], str]] = []
        for title, bounds in books:
            if not title_is_selected(title, args):
                continue
            key = book_key(title)
            if key in seen_targets:
                continue
            if title_is_always_skipped(title):
                print(f"Skipping {title!r}: configured always-skip title.")
                seen_targets.add(key)
                continue
            if not title_meets_progress_filter(
                xml, height, title, args.min_progress_percent
            ):
                seen_targets.add(key)
                continue
            new_books.append((title, bounds, key))

        if not new_books:
            signature = tuple(book_key(title) for title, _ in books)
            if signature == last_no_new_signature:
                stalls += 1
            else:
                stalls = 0
                last_no_new_signature = signature
            if stalls >= args.stall_pages:
                print("Stopping: no new books discovered after repeated scrolls.")
                break
            swipe_library_scroll(width, height)
            continue

        stalls = 0
        last_no_new_signature = None

        title, _bounds, key = new_books[0]
        current_bounds = find_library_book_bounds(xml, height, title)
        if not current_bounds:
            print(f"Skipping {title!r}: could not find it on the current Library page.")
            continue

        was_downloaded = None
        if args.remove_downloads or args.remove_new_downloads:
            if key in prior_new_downloads:
                was_downloaded = False
            else:
                was_downloaded = detect_download_state_from_library_row(
                    xml, height, title
                )
            if was_downloaded is None:
                print(
                    f"WARN: download state unknown for {title!r}; "
                    "will leave any local copy in place."
                )
            else:
                state_label = "downloaded" if was_downloaded else "not downloaded"
                print(f"Download state before export for {title!r}: {state_label}.")

        enforce_min_free_space(args, f"open {title!r}")
        free_before_open = get_data_free_bytes()
        print(f"Storage before opening {title!r}: {format_gib(free_before_open)} free.")
        seen_targets.add(key)
        book_title_for_cleanup = title
        should_pause = False
        tap_bounds(current_bounds)
        try:
            time.sleep(args.open_wait)
            free_after_open = get_data_free_bytes()
            print(
                f"Storage after opening {title!r}: "
                f"{format_gib(free_after_open)} free "
                f"({format_storage_delta(free_before_open, free_after_open)})."
            )
            enforce_min_free_space(args, f"continue after opening {title!r}")

            xml_notes = ensure_notes_list(width, height)
            if not xml_notes:
                print(f"Skipping {title!r}: could not open notes list.")
            else:
                book_title = detect_book_title(xml_notes, height) or title
                book_title_for_cleanup = book_title
                seen_targets.add(book_key(book_title))
                items = extract_book_highlights(
                    args, width, height, book_title, initial_xml=xml_notes
                )
                if items:
                    write_outputs(book_title, outdir, items)
                else:
                    print(f"No highlights found for {book_title!r}.")
        finally:
            free_before_cleanup = get_data_free_bytes()
            print(
                f"Storage before cleanup for {book_title_for_cleanup!r}: "
                f"{format_gib(free_before_cleanup)} free."
            )
            try:
                return_to_library_after_book(width, height)
                cleanup_download_after_book(
                    args, book_title_for_cleanup, width, height, was_downloaded
                )
            except Exception as exc:
                print(f"WARN: cleanup after {book_title_for_cleanup!r} failed: {exc}")
            free_after_cleanup = get_data_free_bytes()
            print(
                f"Storage after cleanup for {book_title_for_cleanup!r}: "
                f"{format_gib(free_after_cleanup)} free "
                f"({format_storage_delta(free_before_cleanup, free_after_cleanup)})."
            )
            processed_books += 1
            pause_after_books = getattr(args, "pause_after_books", 0)
            if pause_after_books and processed_books % pause_after_books == 0:
                print(
                    f"Pausing after {processed_books} opened books. "
                    f"Batch free-space change: "
                    f"{format_storage_delta(batch_start_free, free_after_cleanup)} "
                    f"({format_gib(batch_start_free)} -> "
                    f"{format_gib(free_after_cleanup)})."
                )
                should_pause = True
        if should_pause:
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", help="Book title (used for output filenames)")
    ap.add_argument(
        "--all-books", action="store_true", help="Auto-open each book from Library"
    )
    ap.add_argument(
        "--include-title",
        action="append",
        default=[],
        help="Extra lowercase or mixed-case title substring to include with --all-books. Repeatable.",
    )
    ap.add_argument(
        "--only-title",
        action="append",
        default=[],
        help=(
            "With --all-books, process only these title substrings instead of the "
            "built-in target list. Repeatable."
        ),
    )
    ap.add_argument(
        "--no-title-filter",
        action="store_true",
        help="With --all-books, process every visible library title instead of only configured substrings.",
    )
    ap.add_argument("--out", default="playbooks_export", help="Output directory")
    ap.add_argument(
        "--max-swipes", type=int, default=500, help="Hard cap to avoid infinite loops"
    )
    ap.add_argument(
        "--stall-pages",
        type=int,
        default=6,
        help="Stop after this many consecutive swipes with no new highlights",
    )
    ap.add_argument(
        "--debug-dumps", action="store_true", help="Save raw UI XML dumps (large)"
    )
    ap.add_argument(
        "--skip-titles-from-log",
        action="append",
        default=[],
        help="With --all-books, skip titles already attempted in a prior run.log.",
    )
    ap.add_argument(
        "--cleanup-new-downloads-from-log",
        action="append",
        default=[],
        help=(
            "Remove local downloads for titles that a prior run.log recorded as "
            "not downloaded before opening and did not later record as removed. "
            "Repeatable."
        ),
    )
    ap.add_argument(
        "--open-wait",
        type=float,
        default=UI_WAIT_LONG,
        help="Seconds to wait after opening a book before looking for notes.",
    )
    ap.add_argument(
        "--library-sort",
        choices=("none", "recent", "title"),
        default="recent",
        help="With --all-books, set the Library sort before scanning.",
    )
    ap.add_argument(
        "--download-status",
        choices=("current", "all", "downloaded", "not-downloaded"),
        default="current",
        help="With --all-books, set the Library download-status filter before scanning.",
    )
    ap.add_argument(
        "--resume-from-current-position",
        action="store_true",
        help=(
            "With --all-books, continue scanning from the current Library scroll "
            "position instead of scrolling to the top and reapplying sort/filter. "
            "Assumes the current Library filter and sort are already correct."
        ),
    )
    ap.add_argument(
        "--min-progress-percent",
        type=int,
        default=0,
        help=(
            "With --all-books, open only books whose visible Library progress is "
            "at least this percent. Use 10 to skip barely-started books."
        ),
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help=(
            "Minimum free space required on the tablet's /data partition before "
            "starting or opening the next book. Cleanup-only runs ignore this so "
            "they can recover space."
        ),
    )
    ap.add_argument(
        "--pause-after-books",
        type=int,
        default=0,
        help=(
            "With --all-books, stop after this many opened books so storage can "
            "be inspected before resuming. Use 3 for cautious batches."
        ),
    )
    ap.add_argument(
        "--remove-downloads",
        action="store_true",
        help=(
            "With --all-books, try to remove each book's local download after "
            "opening/export, including books that were already downloaded."
        ),
    )
    ap.add_argument(
        "--remove-new-downloads",
        action="store_true",
        help=(
            "With --all-books, remove a local download after export only when the "
            "book was not downloaded before opening it."
        ),
    )
    args = ap.parse_args()
    if args.remove_downloads and args.remove_new_downloads:
        ap.error(
            "--remove-downloads and --remove-new-downloads cannot be used together"
        )
    if (
        args.all_books
        and args.min_free_gb > 0
        and not args.remove_downloads
        and not args.remove_new_downloads
    ):
        ap.error(
            "--all-books with --min-free-gb requires --remove-downloads or "
            "--remove-new-downloads so downloads are cleaned before continuing"
        )

    ensure_adb_ok()
    width, height = get_screen_size()
    print(f"Device screen: {width}x{height}")

    if args.all_books:
        run_all_books(args, width, height)
        return

    if args.cleanup_new_downloads_from_log:
        cleanup_new_downloads_from_logs(args, width, height)
        return

    enforce_min_free_space(args, "start manual extraction")
    xml = uiautomator_dump_xml()
    if not is_notes_list_screen(xml):
        xml = ensure_notes_list(width, height) or xml

    book = args.book or detect_book_title(xml, height) or "Unknown Book"
    items = extract_book_highlights(args, width, height, book, initial_xml=xml)

    if not items:
        print(
            "\nNo highlights extracted.\nMake sure you are on the *per-book highlights list* screen"
            " in Play Books, with the highlights visibly listed.\nIf your highlights are very"
            " short, try enabling --debug-dumps and inspect nodes for :id/snippet entries.",
            file=sys.stderr,
        )
        sys.exit(2)

    write_outputs(book, args.out, items)


if __name__ == "__main__":
    main()
