#!/usr/bin/env python3
"""
Google Play Books highlights extractor (Android 11-14, non-root) using adb + uiautomator dump.

Workflow:
  1) On the tablet, open Google Play Books and navigate to the per-book highlights list screen.
  2) Run:
       python3 dump_playbooks_highlights.py --book "My Book Title" --out out
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
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple


def run(
    cmd: List[str], check: bool = True, capture: bool = True, text: bool = True, timeout: int = 30
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
        print("ERROR: adb not found in PATH. Install Android platform-tools.", file=sys.stderr)
        sys.exit(1)

    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        print(
            "ERROR: No devices detected. Check USB cable + USB debugging prompt.", file=sys.stderr
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


def uiautomator_dump_xml() -> str:
    """
    Returns UI hierarchy XML as a string.
    Uses a dump to /sdcard then cats it back (works reliably on Samsung).
    """
    # Dump to sdcard
    adb(["shell", "uiautomator", "dump", "/sdcard/ua_dump.xml"], timeout=60)
    # Read it back
    p = adb(["shell", "cat", "/sdcard/ua_dump.xml"], timeout=60)
    xml = p.stdout
    if "<hierarchy" not in xml:
        raise RuntimeError(
            "uiautomator dump did not return hierarchy XML. Make sure the device is unlocked."
        )
    return xml


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
        yield UiNode(text=text, resource_id=rid, class_name=cls, content_desc=desc, bounds=bounds)


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


def swipe_scroll(width: int, height: int) -> None:
    """
    Swipe up to scroll down the list.
    """
    x = width // 2
    y1 = int(height * 0.78)
    y2 = int(height * 0.22)
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


def write_outputs(book: str, outdir: str, items: List[ExtractedItem]) -> None:
    os.makedirs(outdir, exist_ok=True)

    md_path = os.path.join(outdir, f"{safe_filename(book)}.md")
    jsonl_path = os.path.join(outdir, f"{safe_filename(book)}.jsonl")

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="Book title (used for output filenames)")
    ap.add_argument("--out", default="playbooks_export", help="Output directory")
    ap.add_argument("--max-swipes", type=int, default=500, help="Hard cap to avoid infinite loops")
    ap.add_argument(
        "--stall-pages",
        type=int,
        default=6,
        help="Stop after this many consecutive swipes with no new highlights",
    )
    ap.add_argument("--debug-dumps", action="store_true", help="Save raw UI XML dumps (large)")
    args = ap.parse_args()

    ensure_adb_ok()
    width, height = get_screen_size()
    print(f"Device screen: {width}x{height}")

    all_highlights: List[ExtractedItem] = []
    seen: Set[Tuple[str, str]] = set()

    stalls = 0

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    for swipe_i in range(args.max_swipes):
        xml = uiautomator_dump_xml()

        if args.debug_dumps:
            dump_path = os.path.join(outdir, f"{safe_filename(args.book)}__dump_{swipe_i:04d}.xml")
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

        print(f"[{swipe_i:03d}] +{new_count} new (total {len(all_highlights)}), stalls={stalls}")

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

        swipe_scroll(width, height)

    if not all_highlights:
        print(
            "\nNo highlights extracted.\nMake sure you are on the *per-book highlights list* screen"
            " in Play Books, with the highlights visibly listed.\nIf your highlights are very"
            " short, try enabling --debug-dumps and inspect nodes for :id/snippet entries.",
            file=sys.stderr,
        )
        sys.exit(2)

    write_outputs(args.book, outdir, all_highlights)


if __name__ == "__main__":
    main()
