"""Estimate book word counts with local files as the first-class source.

The script produces a standalone word-count table. It tries local readable book
files first, then falls back to existing metadata page counts. Web/manual sources
can be added as explicit CSV rows so their provenance is visible.
"""

from __future__ import annotations

import argparse
import math
import posixpath
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_books_tracking.book_wpm_calendar_notes import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORDS_PER_PAGE,
    abbreviation_matches_title,
    count_words,
    find_metadata_record,
    load_metadata_records,
    normalize_title,
    title_match_score,
    titles_match,
)


DEFAULT_BOOK_ROOTS = [
    REPO_ROOT / "data" / "Books",
    REPO_ROOT / "data" / "Books" / "eleven_upload",
    REPO_ROOT / "data" / "Takeout_Play_books_03_16_25_pt1" / "Google Play Books",
    REPO_ROOT / "data" / "Takeout_Play_books_03_16_25_pt2" / "Google Play Books",
    Path("/Users/clarkbenham/Documents/Books"),
    Path("/Users/clarkbenham/Documents/Books/eleven_upload"),
]
DEFAULT_FILE_HINT_CSVS = [
    REPO_ROOT / "data" / "finished_books_2025_03_16.csv",
    REPO_ROOT
    / "data"
    / "Books Read and their effects - master_book_metadata_cleaned_final.csv",
    REPO_ROOT / "ai_books_tracking" / "master_book_metadata_cleaned.csv",
    REPO_ROOT / "ai_books_tracking" / "golden_master_multi_source.csv",
]
DEFAULT_WORD_SOURCE_CSV = DEFAULT_OUTPUT_DIR / "book_word_count_sources.csv"
WORD_SOURCE_COLUMNS = [
    "title",
    "source_type",
    "word_count",
    "url",
    "path",
    "start_marker",
    "end_marker",
    "subtract_header_words",
    "subtract_footer_words",
    "confidence",
    "notes",
]
TEXT_EXTENSIONS = {".txt", ".md"}
HTML_EXTENSIONS = {".html", ".htm"}
DOCX_EXTENSIONS = {".docx"}
PDF_EXTENSIONS = {".pdf"}
EPUB_EXTENSIONS = {".epub"}
CALIBRE_EXTENSIONS = {".mobi", ".azw3", ".lit"}
SAFE_SINGLE_WORD_PREFIX_HINTS = {"open"}
MIN_LOCAL_PDF_READING_WORDS = 1_000
NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
    "30": "thirty",
    "40": "forty",
    "50": "fifty",
    "60": "sixty",
    "70": "seventy",
    "80": "eighty",
    "90": "ninety",
}


@dataclass(frozen=True)
class LocalBookFile:
    path: Path
    name_norm: str
    stem_norm: str


@dataclass(frozen=True)
class LocalTextAudit:
    raw_word_count: int
    reading_word_count: int
    method: str
    warning: str
    section_rows: list[dict[str, object]]


def iter_book_files(roots: list[Path]) -> list[LocalBookFile]:
    files: list[LocalBookFile] = []
    seen: set[Path] = set()
    allowed = (
        TEXT_EXTENSIONS
        | HTML_EXTENSIONS
        | DOCX_EXTENSIONS
        | PDF_EXTENSIONS
        | EPUB_EXTENSIONS
        | CALIBRE_EXTENSIONS
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if (
                    should_skip_book_path(path)
                    or not path.is_file()
                    or path.suffix.lower() not in allowed
                    or not has_supported_book_signature(path)
                ):
                    continue
            except OSError:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(
                LocalBookFile(
                    path=path,
                    name_norm=normalize_title(path.name),
                    stem_norm=normalize_title(path.stem),
                )
            )
    return files


def should_skip_book_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if ".git" in parts or "__macosx" in parts:
        return True
    if "highlights" in parts or "play books notes" in parts:
        return True
    if path.suffix.lower() in HTML_EXTENSIONS and "google play books" in parts:
        return True
    if path.suffix.lower() in {".crdownload", ".jsonl", ".xml"}:
        return True
    return False


def has_supported_book_signature(path: Path) -> bool:
    if path.suffix.lower() not in PDF_EXTENSIONS:
        return True
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def inspect_book_roots(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in roots:
        row: dict[str, object] = {
            "root": str(root),
            "exists": root.exists(),
            "is_dir": root.is_dir(),
            "can_iterate": False,
            "visible_entry_count": 0,
            "error": "",
        }
        try:
            entries = list(root.iterdir())
            row["can_iterate"] = True
            row["visible_entry_count"] = len(entries)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def score_file_match(title: str, filename: str, file: LocalBookFile) -> float:
    title_norm = normalize_title(title)
    filename_norm = normalize_title(filename)
    candidates = [value for value in [title_norm, filename_norm] if value]
    score = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        if not candidate_numbers_match(candidate, file.stem_norm):
            continue
        candidate_words = candidate.split()
        if candidate == file.stem_norm or candidate == file.name_norm:
            score = max(score, 100.0)
        elif phrase_contains(candidate, file.stem_norm) or phrase_contains(
            file.stem_norm, candidate
        ):
            score = max(score, 90.0)
        elif (
            len(candidate_words) == 1
            and looks_like_initialism_token(candidate_words[0])
            and candidate_words[0] in file.stem_norm.split()
        ):
            score = max(score, 82.0)
        elif looks_like_abbreviation_candidate(
            candidate
        ) and abbreviation_matches_title(candidate, file.stem_norm):
            score = max(score, 82.0)
        elif titles_match(candidate, file.stem_norm):
            score = max(score, 85.0)
        elif len(candidate_words) >= 3:
            title_words = set(candidate.split())
            file_words = set(file.stem_norm.split())
            if title_words and file_words:
                overlap = len(title_words & file_words) / min(
                    len(title_words), len(file_words)
                )
                score = max(score, overlap * 80)
                title_stems = {singular_token(word) for word in title_words}
                file_stems = {singular_token(word) for word in file_words}
                stem_overlap = len(title_stems & file_stems) / min(
                    len(title_stems), len(file_stems)
                )
                score = max(score, stem_overlap * 80)
    return score


def candidate_numbers_match(candidate: str, file_stem: str) -> bool:
    candidate_volume_numbers = numbered_marker_numbers(candidate, {"vol", "volume"})
    file_volume_numbers = numbered_marker_numbers(file_stem, {"vol", "volume"})
    if candidate_volume_numbers and file_volume_numbers:
        if not candidate_volume_numbers & file_volume_numbers:
            return False
    candidate_numbers = set(re.findall(r"\d+", candidate))
    if not candidate_numbers:
        return True
    file_numbers = set(re.findall(r"\d+", file_stem))
    file_words = set(file_stem.split())
    return all(
        number in file_numbers or NUMBER_WORDS.get(number, "") in file_words
        for number in candidate_numbers
    )


def looks_like_initialism_token(token: str) -> bool:
    return 2 <= len(token) <= 8 and not re.search(r"[aeiou]", token)


def looks_like_abbreviation_candidate(candidate_norm: str) -> bool:
    return (
        " " not in candidate_norm
        and 2 <= len(candidate_norm) <= 12
        and bool(re.fullmatch(r"[a-z0-9]+", candidate_norm))
        and not candidate_norm.isdigit()
    )


def phrase_contains(longer: str, shorter: str) -> bool:
    shorter_words = shorter.split()
    if not shorter_words:
        return False
    if len(shorter_words) == 1:
        return False
    return bool(re.search(rf"\b{re.escape(shorter)}\b", longer))


def singular_token(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def find_local_file(
    title: str, filename: str, local_files: list[LocalBookFile]
) -> tuple[Path | None, float]:
    scored = [
        (score_file_match(title, filename, file), file.path) for file in local_files
    ]
    scored = [(score, path) for score, path in scored if score >= 70]
    if not scored:
        return None, 0.0
    scored.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return scored[0][1], scored[0][0]


def find_local_file_with_hints(
    title: str, filenames: list[str], local_files: list[LocalBookFile]
) -> tuple[Path | None, float, str]:
    scored: list[tuple[float, Path, str]] = []
    for filename in dict.fromkeys(value for value in filenames if str(value).strip()):
        for file in local_files:
            score = score_file_match(title, filename, file)
            if score >= 70:
                scored.append((score, file.path, filename))
    if not scored:
        return None, 0.0, ""
    scored.sort(key=lambda item: (-item[0], len(str(item[1]))))
    score, path, hint = scored[0]
    return path, score, hint


def text_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in HTML_EXTENSIONS:
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text(" ")
    if suffix in PDF_EXTENSIONS:
        return pdf_text(path)
    if suffix in DOCX_EXTENSIONS:
        return docx_text(path)
    if suffix in EPUB_EXTENSIONS:
        return epub_text(path)
    if suffix in CALIBRE_EXTENSIONS:
        return calibre_text(path)
    return ""


def html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ")


def pdf_text(path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp:
        process = subprocess.run(
            ["pdftotext", str(path), temp.name],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()
            raise RuntimeError(f"pdftotext failed: {detail}")
        return Path(temp.name).read_text(encoding="utf-8", errors="ignore")


def docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    return BeautifulSoup(xml, "xml").get_text(" ")


def epub_text(path: Path) -> str:
    texts: list[str] = []
    with ZipFile(path) as archive:
        for name in archive.namelist():
            if Path(name).suffix.lower() not in HTML_EXTENSIONS | {".xhtml"}:
                continue
            html = archive.read(name).decode("utf-8", errors="ignore")
            texts.append(html_text(html))
    return "\n".join(texts)


def local_text_audit(path: Path, title: str = "") -> LocalTextAudit:
    suffix = path.suffix.lower()
    if suffix in EPUB_EXTENSIONS:
        return epub_text_audit(path, title)
    if suffix in PDF_EXTENSIONS:
        return pdf_text_audit(path)
    raw_text = text_from_path(path)
    raw_count = cleaned_word_count(raw_text)
    return LocalTextAudit(
        raw_word_count=raw_count,
        reading_word_count=raw_count,
        method=f"{suffix.lstrip('.') or 'text'}_raw_text",
        warning="no_section_level_body_filter_available",
        section_rows=[],
    )


def epub_text_audit(path: Path, title: str = "") -> LocalTextAudit:
    rows: list[dict[str, object]] = []
    repair_warning = ""
    try:
        with ZipFile(path) as archive:
            names = epub_spine_names(archive)
            if not names:
                names = [
                    name
                    for name in archive.namelist()
                    if Path(name).suffix.lower() in HTML_EXTENSIONS | {".xhtml"}
                ]
            for order, name in enumerate(names):
                if name not in archive.namelist():
                    continue
                html = archive.read(name).decode("utf-8", errors="ignore")
                row = epub_section_audit_row(
                    html=html,
                    name=name,
                    order=order,
                    title=title,
                )
                rows.append(row)
            propagate_epub_back_matter_categories(rows)
            repair_warning = repair_all_excluded_epub_body(rows)
    except Exception as exc:
        text = text_from_path(path)
        raw_count = cleaned_word_count(text)
        return LocalTextAudit(
            raw_word_count=raw_count,
            reading_word_count=raw_count,
            method="epub_raw_text_fallback",
            warning=f"{type(exc).__name__}: {exc}",
            section_rows=[],
        )
    raw_count = int(sum(int(row["word_count"]) for row in rows))
    reading_count = int(
        sum(int(row["word_count"]) for row in rows if row["included_in_reading_count"])
    )
    warning = ""
    if reading_count <= 0 and raw_count > 0:
        reading_count = raw_count
        warning = "body_filter_removed_all_words_using_raw_count"
    if repair_warning:
        warning = "; ".join(value for value in [repair_warning, warning] if value)
    return LocalTextAudit(
        raw_word_count=raw_count,
        reading_word_count=reading_count,
        method="epub_spine_body_sections",
        warning=warning,
        section_rows=rows,
    )


def propagate_epub_back_matter_categories(rows: list[dict[str, object]]) -> None:
    active_back_matter = ""
    late_section_start = max(0, int(len(rows) * 0.55))
    for index, row in enumerate(rows):
        category = str(row.get("body_section_category", "") or "")
        if category in {"notes", "sources_references_bibliography"}:
            if active_back_matter or index >= late_section_start:
                active_back_matter = category
            continue
        if category == "index":
            active_back_matter = (
                "front_back_matter" if index >= late_section_start else ""
            )
            continue
        if category == "front_back_matter":
            continue
        if category == "body" and active_back_matter:
            row["body_section_category"] = active_back_matter
            row["included_in_reading_count"] = False


def repair_all_excluded_epub_body(rows: list[dict[str, object]]) -> str:
    candidate_rows = [
        row
        for row in rows
        if int(row.get("word_count", 0) or 0) > 0
        and row.get("body_section_category") not in {"front_back_matter", "empty"}
    ]
    included_words = sum(
        int(row.get("word_count", 0) or 0)
        for row in rows
        if row.get("included_in_reading_count")
    )
    if included_words > 0 or not candidate_rows:
        return ""
    for row in candidate_rows:
        row["body_section_category"] = "body"
        row["included_in_reading_count"] = True
    return "all_non_front_sections_reclassified_as_body"


def epub_spine_names(archive: ZipFile) -> list[str]:
    try:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
    except KeyError:
        return []
    ns = {"container": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = container.find(".//container:rootfile", ns)
    if rootfile is None:
        return []
    opf_name = rootfile.attrib.get("full-path", "")
    if not opf_name:
        return []
    opf = ET.fromstring(archive.read(opf_name))
    opf_ns = {"opf": "http://www.idpf.org/2007/opf"}
    manifest = {
        item.attrib.get("id"): item.attrib
        for item in opf.findall(".//opf:manifest/opf:item", opf_ns)
    }
    base = posixpath.dirname(opf_name)
    names: list[str] = []
    for itemref in opf.findall(".//opf:spine/opf:itemref", opf_ns):
        item = manifest.get(itemref.attrib.get("idref"))
        if not item:
            continue
        href = item.get("href", "")
        if not href:
            continue
        names.append(posixpath.normpath(posixpath.join(base, href)))
    return names


def epub_section_audit_row(
    *, html: str, name: str, order: int, title: str = ""
) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    headings = unique_nonempty(
        re.sub(r"\s+", " ", tag.get_text(" ")).strip()
        for tag in soup.find_all(["title", "h1", "h2", "h3"], limit=8)
    )
    heading_text = " | ".join(headings)
    word_count = cleaned_word_count(text)
    category = classify_body_section(name, heading_text, text, title)
    return {
        "section_order": order,
        "section_path": name,
        "section_heading": heading_text[:240],
        "section_sample": text[:240],
        "word_count": word_count,
        "body_section_category": category,
        "included_in_reading_count": category == "body",
    }


def unique_nonempty(values: object) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def classify_body_section(name: str, heading: str, text: str, title: str = "") -> str:
    if cleaned_word_count(text) == 0:
        return "empty"
    for part in significant_heading_parts(heading, title):
        category = classify_section_marker(part)
        if category:
            return category
    sample = strip_title_prefix(text[:260], title)
    category = classify_section_start(sample)
    if category:
        return category
    path_name = Path(name).name.lower()
    if re.match(r"^(?:footnotes?|endnotes?|notes)(?:[_\-.]|$)", path_name):
        return "notes"
    if re.search(
        r"(?:^|[_\-.])(?:toc|contents|cover|titlepage|title_page|copyright)(?:[_\-.]|$)",
        path_name,
    ):
        return "front_back_matter"
    return "body"


def significant_heading_parts(heading: str, title: str = "") -> list[str]:
    title_norm = normalize_title(title)
    parts: list[str] = []
    for part in re.split(r"\s+\|\s+", heading):
        part = part.strip()
        if not part:
            continue
        if title_norm and normalize_title(part) == title_norm:
            continue
        parts.append(part)
    return parts


def classify_section_marker(text: str) -> str:
    normalized = normalize_title(text)
    if not normalized:
        return ""
    if re.match(r"^(?:index|general index|name index)\b", normalized):
        return "index"
    if normalized in {
        "notes",
        "endnotes",
        "footnotes",
        "notes and references",
        "references and notes",
        "note on sources",
        "a note on sources",
        "selected interviews",
    }:
        return "notes"
    if re.match(
        r"^notes (?:abbreviations?|chapter|introduction|prologue|epilogue|part|\d+|"
        r"[ivxlcdm]+)\b",
        normalized,
    ):
        return "notes"
    if re.match(
        r"^(?:sources|source notes|references|bibliography|selected bibliography)\b",
        normalized,
    ):
        return "sources_references_bibliography"
    if re.match(
        r"^(?:cover|title page|copyright|contents|table of contents|dedication|"
        r"about the author|about the book|also by|praise for|acclaim for|"
        r"acknowledgments?|acknowledgements?|debts|photographic credits?|"
        r"photo credits?|illustration credits?|illustrations|permissions?|"
        r"advertisement)\b",
        normalized,
    ):
        return "front_back_matter"
    return ""


def classify_section_start(text: str) -> str:
    sample = re.sub(r"\s+", " ", text).strip()
    if not sample:
        return ""
    normalized = normalize_title(sample[:120])
    for category in ["index", "notes", "sources_references_bibliography"]:
        marker = classify_section_marker(normalized)
        if marker == category:
            return category
    if re.match(r"^\d+\.\s+.{1,80}\bSOURCES\b", sample, re.IGNORECASE):
        return "sources_references_bibliography"
    return ""


def strip_title_prefix(text: str, title: str = "") -> str:
    sample = re.sub(r"\s+", " ", text).strip()
    if not title:
        return sample
    title_words = re.escape(re.sub(r"\s+", " ", title).strip())
    return re.sub(rf"^{title_words}\b", "", sample, flags=re.IGNORECASE).strip()


def pdf_text_audit(path: Path) -> LocalTextAudit:
    pages = pdf_pages(path)
    if not pages:
        text = pdf_text(path)
        count = cleaned_word_count(text)
        return LocalTextAudit(
            raw_word_count=count,
            reading_word_count=count,
            method="pdf_raw_text_fallback",
            warning="pdftotext_page_split_failed",
            section_rows=[],
        )
    repeated_edges = repeated_pdf_edge_lines(pages)
    rows: list[dict[str, object]] = []
    for page_number, page_text in enumerate(pages, start=1):
        cleaned_page = remove_pdf_edge_lines(page_text, repeated_edges)
        category = classify_pdf_page(page_number, len(pages), cleaned_page)
        rows.append(
            {
                "section_order": page_number,
                "section_path": f"page_{page_number}",
                "section_heading": first_nonempty_line(cleaned_page)[:240],
                "section_sample": re.sub(r"\s+", " ", cleaned_page).strip()[:240],
                "word_count": cleaned_word_count(cleaned_page),
                "body_section_category": category,
                "included_in_reading_count": category == "body",
            }
        )
    raw_count = cleaned_word_count("\n".join(pages))
    reading_count = int(
        sum(int(row["word_count"]) for row in rows if row["included_in_reading_count"])
    )
    warning = ""
    if reading_count <= 0 and raw_count > 0:
        reading_count = raw_count
        warning = "body_filter_removed_all_words_using_raw_count"
    return LocalTextAudit(
        raw_word_count=raw_count,
        reading_word_count=reading_count,
        method="pdf_repeated_header_footer_body_pages",
        warning=warning,
        section_rows=rows,
    )


def pdf_pages(path: Path) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp:
        process = subprocess.run(
            ["pdftotext", "-layout", str(path), temp.name],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            return []
        text = Path(temp.name).read_text(encoding="utf-8", errors="ignore")
    return [page for page in text.split("\f") if page.strip()]


def repeated_pdf_edge_lines(pages: list[str]) -> set[str]:
    counts: dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        for line in lines[:2] + lines[-2:]:
            normalized = normalize_pdf_edge_line(line)
            if normalized:
                counts[normalized] = counts.get(normalized, 0) + 1
    threshold = max(5, int(len(pages) * 0.08))
    return {line for line, count in counts.items() if count >= threshold}


def normalize_pdf_edge_line(line: str) -> str:
    line = re.sub(r"\b\d+\b", "#", line.strip().lower())
    line = re.sub(r"\s+", " ", line)
    if not line or line == "#":
        return ""
    return line


def remove_pdf_edge_lines(text: str, repeated_edges: set[str]) -> str:
    lines = text.splitlines()
    if not repeated_edges:
        return text
    cleaned: list[str] = []
    for index, line in enumerate(lines):
        is_edge = index < 2 or index >= len(lines) - 2
        if is_edge and normalize_pdf_edge_line(line) in repeated_edges:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def classify_pdf_page(page_number: int, page_count: int, text: str) -> str:
    heading = first_nonempty_line(text)
    sample = re.sub(r"\s+", " ", text).strip()[:240]
    is_late_page = page_number > page_count * 0.75
    marker = classify_section_marker(heading)
    if marker and (is_late_page or page_number <= 5):
        return marker
    start_marker = classify_section_start(sample)
    if start_marker and (is_late_page or page_number <= 5):
        return start_marker
    return "body"


def calibre_text(path: Path) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "book.txt"
        process = subprocess.run(
            ["ebook-convert", str(path), str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()
            raise RuntimeError(f"ebook-convert failed: {detail}")
        return output.read_text(encoding="utf-8", errors="ignore")


def cleaned_word_count(text: str) -> int:
    text = re.sub(r"\s+", " ", text)
    return count_words(text)


def load_manual_word_counts(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["title", "word_count", "source", "notes"])
    frame = pd.read_csv(path)
    required = {"title", "word_count", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manual word-count CSV missing columns: {sorted(missing)}")
    return frame


def load_word_count_sources(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=WORD_SOURCE_COLUMNS)
    frame = pd.read_csv(path)
    missing = {"title", "source_type"} - set(frame.columns)
    if missing:
        raise ValueError(f"word-count source CSV missing columns: {sorted(missing)}")
    for column in WORD_SOURCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[WORD_SOURCE_COLUMNS]


def load_local_file_hints(paths: list[Path] | None = None) -> pd.DataFrame:
    hint_rows: list[dict[str, object]] = []
    for path in paths or DEFAULT_FILE_HINT_CSVS:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "title" not in frame.columns and "original_title" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            title = clean_source_value(row.get("title", row.get("original_title", "")))
            filename = clean_source_value(row.get("filename", ""))
            if not title and not filename:
                continue
            hint_rows.append(
                {
                    "hint_title": title,
                    "hint_filename": filename,
                    "hint_title_norm": normalize_title(title),
                    "hint_filename_norm": normalize_title(filename),
                    "hint_author": clean_source_value(
                        row.get(
                            "author",
                            row.get(
                                "corrected_author",
                                row.get(
                                    "canonical_author",
                                    row.get("author(old and wrong)", ""),
                                ),
                            ),
                        )
                    ),
                    "hint_finish_date": clean_source_value(
                        row.get(
                            "finished_date",
                            row.get(
                                "estimated_finish",
                                row.get(
                                    "latest_modified", row.get("earliest_modified", "")
                                ),
                            ),
                        )
                    ),
                    "hint_finish_date_parsed": pd.to_datetime(
                        row.get(
                            "finished_date",
                            row.get(
                                "estimated_finish",
                                row.get(
                                    "latest_modified",
                                    row.get("earliest_modified", ""),
                                ),
                            ),
                        ),
                        errors="coerce",
                    ),
                    "hint_source": str(path.relative_to(REPO_ROOT)),
                }
            )
    return pd.DataFrame(hint_rows)


def local_file_hints_for_title(
    hints: pd.DataFrame, title: str, ref: str = "", finish_date: object = ""
) -> list[str]:
    if hints.empty:
        return []
    candidate_groups = [[title], [ref]]
    target_date = pd.to_datetime(finish_date, errors="coerce")
    for candidates in candidate_groups:
        output = local_file_hints_for_candidates(hints, candidates, target_date)
        if output:
            return output
    return []


def local_file_hints_for_candidates(
    hints: pd.DataFrame, candidates: list[str], target_date: object
) -> list[str]:
    output: list[str] = []
    for _, row in hints.iterrows():
        hint_title = str(row.get("hint_title", "") or "")
        hint_filename = str(row.get("hint_filename", "") or "")
        if not hint_title and not hint_filename:
            continue
        date_bonus = False
        hint_date = row.get("hint_finish_date_parsed", pd.NaT)
        if pd.isna(hint_date):
            hint_date = pd.to_datetime(row.get("hint_finish_date", ""), errors="coerce")
        if pd.notna(target_date) and pd.notna(hint_date):
            date_bonus = abs((target_date.date() - hint_date.date()).days) <= 3
        for candidate in candidates:
            if not candidate:
                continue
            if hint_value_matches_candidate(
                candidate,
                hint_filename,
                date_bonus,
                str(row.get("hint_filename_norm", "") or ""),
            ):
                output.append(hint_filename)
            if hint_value_matches_candidate(
                candidate,
                hint_title,
                date_bonus,
                str(row.get("hint_title_norm", "") or ""),
            ):
                output.append(hint_title)
    return [value for value in dict.fromkeys(output) if value]


def hint_matches_candidate(
    candidate: str,
    hint_title: str,
    hint_filename: str,
    date_bonus: bool,
    hint_title_norm: str = "",
    hint_filename_norm: str = "",
) -> bool:
    return any(
        hint_value_matches_candidate(candidate, hint_value, date_bonus, hint_norm)
        for hint_value, hint_norm in [
            (hint_title, hint_title_norm or normalize_title(hint_title)),
            (hint_filename, hint_filename_norm or normalize_title(hint_filename)),
        ]
    )


def hint_value_matches_candidate(
    candidate: str,
    hint_value: str,
    date_bonus: bool,
    hint_norm: str = "",
) -> bool:
    candidate_norm = normalize_title(candidate)
    hint_norm = hint_norm or normalize_title(hint_value)
    if not candidate_norm or not hint_norm:
        return False
    if not hint_numbers_compatible(candidate_norm, hint_norm):
        return False
    if not should_score_hint_candidate(candidate_norm, hint_norm, date_bonus):
        return False
    if titles_match(candidate, hint_value) or (
        looks_like_abbreviation_candidate(candidate_norm)
        and abbreviation_matches_title(candidate_norm, hint_norm)
    ):
        return True
    score = title_match_score(candidate, hint_value)
    if score >= 78 or (date_bonus and score >= 60):
        return True
    if date_bonus and fuzzy_meaningful_overlap_ratio(candidate_norm, hint_norm) >= 0.8:
        return True
    if date_bonus and single_word_prefix_match(candidate_norm, hint_norm):
        return True
    if date_bonus and repeated_single_word_match(candidate_norm, hint_norm):
        return True
    return False


def hint_numbers_compatible(candidate_norm: str, hint_norm: str) -> bool:
    if not candidate_numbers_match(candidate_norm, hint_norm):
        return False
    candidate_numbers = set(re.findall(r"\d+", candidate_norm))
    if not candidate_numbers and numbered_volume_marker(hint_norm):
        return False
    return True


def numbered_volume_marker(text: str) -> bool:
    return bool(numbered_marker_numbers(text, {"vol", "volume", "book", "part"}))


def numbered_marker_numbers(text: str, markers: set[str]) -> set[str]:
    marker_pattern = "|".join(sorted(markers, key=len, reverse=True))
    return set(re.findall(rf"\b(?:{marker_pattern})\s+(\d+)\b", text))


def should_score_hint_candidate(
    candidate_norm: str, hint_norm: str, date_bonus: bool
) -> bool:
    if candidate_norm == hint_norm:
        return True
    if date_bonus and single_word_prefix_match(candidate_norm, hint_norm):
        return True
    if date_bonus and repeated_single_word_match(candidate_norm, hint_norm):
        return True
    candidate_all_words = candidate_norm.split()
    if len(candidate_all_words) == 1 and candidate_norm in hint_norm:
        return False
    if candidate_norm in hint_norm:
        return True
    if looks_like_abbreviation_candidate(candidate_norm):
        return True
    candidate_words = meaningful_hint_words(candidate_norm)
    hint_words = meaningful_hint_words(hint_norm)
    if not candidate_words or not hint_words:
        return False
    overlap = len(candidate_words & hint_words) / min(
        len(candidate_words), len(hint_words)
    )
    return overlap >= (0.34 if date_bonus else 0.5)


def meaningful_overlap_ratio(left: str, right: str) -> float:
    left_words = meaningful_hint_words(left)
    right_words = meaningful_hint_words(right)
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def fuzzy_meaningful_overlap_ratio(left: str, right: str) -> float:
    left_words = meaningful_hint_words(left)
    right_words = meaningful_hint_words(right)
    if not left_words or not right_words:
        return 0.0
    matched_right: set[str] = set()
    matches = 0
    for left_word in left_words:
        best_word = ""
        best_ratio = 0.0
        for right_word in right_words - matched_right:
            ratio = (
                1.0
                if left_word == right_word
                else word_similarity(left_word, right_word)
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_word = right_word
        if best_ratio >= 0.83:
            matches += 1
            matched_right.add(best_word)
    return matches / min(len(left_words), len(right_words))


def word_similarity(left: str, right: str) -> float:
    if min(len(left), len(right)) < 5:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def meaningful_hint_words(text: str) -> set[str]:
    return {
        word
        for word in text.split()
        if word not in {"the", "a", "an", "of", "and", "in", "on", "for", "to", "by"}
    }


def single_word_prefix_match(candidate_norm: str, hint_norm: str) -> bool:
    words = candidate_norm.split()
    return (
        len(words) == 1
        and words[0] in SAFE_SINGLE_WORD_PREFIX_HINTS
        and hint_norm.startswith(words[0])
    )


def repeated_single_word_match(candidate_norm: str, hint_norm: str) -> bool:
    words = candidate_norm.split()
    return (
        len(words) == 1
        and len(words[0]) >= 4
        and hint_norm.split().count(words[0]) >= 2
    )


def source_rows_for_title(sources: pd.DataFrame, title: str) -> pd.DataFrame:
    if sources.empty:
        return sources
    return sources[sources["title"].map(lambda value: titles_match(str(value), title))]


def numeric_or_zero(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clean_source_value(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def first_clean_value(*values: object) -> str:
    for value in values:
        text = clean_source_value(value)
        if text and text.lower() not in {"nan", "nat", "none"}:
            return text
    return ""


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(_markdown_table(df), encoding="utf-8")


def count_web_html_source(source: pd.Series) -> int:
    url = clean_source_value(source.get("url", ""))
    if not url:
        raise ValueError("web_html source missing url")
    with urllib.request.urlopen(url, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    start_marker = str(source.get("start_marker", "") or "").strip()
    end_marker = str(source.get("end_marker", "") or "").strip()
    if start_marker:
        start_index = text.find(start_marker)
        if start_index < 0:
            raise ValueError(f"start marker not found: {start_marker}")
        text = text[start_index:]
    if end_marker:
        end_index = text.find(end_marker)
        if end_index < 0:
            raise ValueError(f"end marker not found: {end_marker}")
        text = text[:end_index]
    word_count = cleaned_word_count(text)
    word_count -= int(numeric_or_zero(source.get("subtract_header_words", 0)))
    word_count -= int(numeric_or_zero(source.get("subtract_footer_words", 0)))
    if word_count <= 0:
        raise ValueError("web_html source produced non-positive word count")
    return word_count


def build_word_count_outputs(
    *,
    titles: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    book_roots: list[Path] | None = None,
    manual_word_counts: Path | None = None,
    word_source_csv: Path | None = DEFAULT_WORD_SOURCE_CSV,
    words_per_page: int = DEFAULT_WORDS_PER_PAGE,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = book_roots or DEFAULT_BOOK_ROOTS
    inspect_book_roots(roots).to_csv(
        output_dir / "book_word_count_local_roots.csv", index=False
    )
    local_files = iter_book_files(roots)
    metadata_records = load_metadata_records([])
    manual = load_manual_word_counts(manual_word_counts)
    sources = load_word_count_sources(word_source_csv)
    file_hints = load_local_file_hints()
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    local_text_audit_rows: list[dict[str, object]] = []
    local_section_audit_rows: list[dict[str, object]] = []
    local_text_cache: dict[tuple[Path, str], LocalTextAudit] = {}
    show_progress = len(titles) > 20
    for row_number, (_, title_row) in enumerate(titles.iterrows(), start=1):
        title = first_clean_value(title_row["title"])
        filename = first_clean_value(
            title_row.get("filename", ""), title_row.get("calendar_finish_ref", "")
        )
        finish_date = first_clean_value(
            title_row.get("finish_date", ""), title_row.get("matched_finish_date", "")
        )
        filename_hints = [
            title,
            filename,
            first_clean_value(title_row.get("calendar_finish_ref", "")),
            first_clean_value(title_row.get("cal_ref", "")),
        ]
        filename_hints.extend(
            local_file_hints_for_title(
                file_hints,
                title,
                filename,
                finish_date,
            )
        )
        manual_match = manual[
            manual["title"].map(lambda value: titles_match(str(value), title))
        ]
        source_matches = source_rows_for_title(sources, title)
        explicit_local_paths = [
            Path(str(value)).expanduser()
            for value in source_matches[
                source_matches["source_type"].str.lower().eq("local_file")
            ]["path"].dropna()
            if str(value).strip()
        ]
        local_path = explicit_local_paths[0] if explicit_local_paths else None
        local_score = 100.0 if local_path else 0.0
        local_hint = str(local_path) if local_path else ""
        if local_path is None:
            local_path, local_score, local_hint = find_local_file_with_hints(
                title, filename_hints, local_files
            )
        local_word_count = None
        local_raw_word_count = None
        local_excluded_word_count = None
        local_word_count_method = ""
        local_error = ""
        if local_path is not None:
            try:
                if show_progress:
                    print(
                        f"Extracting local text {row_number}/{len(titles)}: "
                        f"{title} <- {local_path}",
                        flush=True,
                    )
                cache_key = (local_path.resolve(), normalize_title(title))
                if cache_key not in local_text_cache:
                    local_text_cache[cache_key] = local_text_audit(local_path, title)
                local_audit = local_text_cache[cache_key]
                local_word_count = local_audit.reading_word_count
                local_raw_word_count = local_audit.raw_word_count
                local_excluded_word_count = (
                    local_audit.raw_word_count - local_audit.reading_word_count
                )
                local_word_count_method = local_audit.method
                local_error = local_audit.warning
                if local_word_count <= 0:
                    local_error = "no_text_extracted"
                elif (
                    local_word_count_method == "pdf_repeated_header_footer_body_pages"
                    and local_word_count < MIN_LOCAL_PDF_READING_WORDS
                ):
                    local_error = (
                        "insufficient_pdf_text_extracted:" f" {local_word_count} words"
                    )
                category_counts = section_category_counts(local_audit.section_rows)
                local_text_audit_rows.append(
                    {
                        "finish_id": title_row.get("finish_id", ""),
                        "title": title,
                        "local_file_path": str(local_path),
                        "local_file_word_count": local_word_count,
                        "local_raw_word_count": local_raw_word_count,
                        "local_excluded_word_count": local_excluded_word_count,
                        "local_word_count_method": local_word_count_method,
                        "local_word_count_warning": local_error,
                        **category_counts,
                    }
                )
                for section in local_audit.section_rows:
                    local_section_audit_rows.append(
                        {
                            "finish_id": title_row.get("finish_id", ""),
                            "title": title,
                            "local_file_path": str(local_path),
                            **section,
                        }
                    )
                if local_error.startswith("insufficient_pdf_text_extracted"):
                    local_word_count = None
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            audit_rows.append(
                {
                    "finish_id": title_row.get("finish_id", ""),
                    "title": title,
                    "source_type": "local_file",
                    "source": str(local_path),
                    "word_count": local_word_count,
                    "status": "ok" if local_word_count else "error",
                    "error": local_error,
                    "notes": local_word_count_method,
                }
            )
        external_counts: list[dict[str, object]] = []
        for _, source in source_matches.iterrows():
            source_type = str(source["source_type"]).strip().lower()
            if source_type == "local_file":
                continue
            source_label = clean_source_value(
                source.get("url", "")
            ) or clean_source_value(source.get("path", ""))
            try:
                if source_type in {"manual", "manual_count", "web_count", "ai_count"}:
                    word_count = int(numeric_or_zero(source.get("word_count", 0)))
                    if word_count <= 0:
                        raise ValueError("manual source missing positive word_count")
                elif source_type == "web_html":
                    word_count = count_web_html_source(source)
                else:
                    raise ValueError(f"unknown source_type: {source_type}")
                external_counts.append(
                    {
                        "word_count": word_count,
                        "source": source_label or source_type,
                        "source_type": source_type,
                        "confidence": str(source.get("confidence", "") or "high"),
                        "notes": str(source.get("notes", "") or ""),
                    }
                )
                audit_rows.append(
                    {
                        "finish_id": title_row.get("finish_id", ""),
                        "title": title,
                        "source_type": source_type,
                        "source": source_label,
                        "word_count": word_count,
                        "status": "ok",
                        "error": "",
                        "notes": str(source.get("notes", "") or ""),
                    }
                )
            except Exception as exc:
                audit_rows.append(
                    {
                        "finish_id": title_row.get("finish_id", ""),
                        "title": title,
                        "source_type": source_type,
                        "source": source_label,
                        "word_count": None,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "notes": str(source.get("notes", "") or ""),
                    }
                )
        metadata = find_metadata_record(title, metadata_records)
        metadata_words = (
            metadata.page_count * words_per_page
            if metadata and metadata.page_count
            else None
        )
        if local_word_count:
            chosen_words = float(local_word_count)
            chosen_source = "local_file_word_count"
            confidence = "high"
        elif not manual_match.empty:
            chosen_words = float(manual_match.iloc[0]["word_count"])
            chosen_source = str(manual_match.iloc[0]["source"])
            confidence = "high"
        elif external_counts:
            best_external = external_counts[0]
            chosen_words = float(best_external["word_count"])
            chosen_source = f"{best_external['source_type']}:{best_external['source']}"
            confidence = str(best_external["confidence"] or "high")
        elif metadata_words:
            chosen_words = float(metadata_words)
            chosen_source = "metadata_pages_x_words_per_page"
            confidence = "medium"
        else:
            chosen_words = None
            chosen_source = ""
            confidence = "low"
        rows.append(
            {
                "finish_id": title_row.get("finish_id", ""),
                "title": title,
                "finish_date": finish_date,
                "calendar_finish_ref": title_row.get("calendar_finish_ref", ""),
                "local_file_path": str(local_path) if local_path else "",
                "local_file_match_score": local_score,
                "local_file_match_hint": local_hint,
                "local_file_word_count": local_word_count,
                "local_raw_word_count": local_raw_word_count,
                "local_excluded_word_count": local_excluded_word_count,
                "local_word_count_method": local_word_count_method,
                "local_file_error": local_error,
                "external_word_count": (
                    external_counts[0]["word_count"] if external_counts else None
                ),
                "external_word_count_source": (
                    external_counts[0]["source"] if external_counts else ""
                ),
                "external_word_count_notes": (
                    external_counts[0]["notes"] if external_counts else ""
                ),
                "metadata_page_count": metadata.page_count if metadata else None,
                "metadata_word_estimate": metadata_words,
                "metadata_source": metadata.source if metadata else "",
                "chosen_word_count": chosen_words,
                "word_count_source": chosen_source,
                "word_count_confidence": confidence,
            }
        )
    result = pd.DataFrame(rows)
    add_online_error_columns(result)
    result.to_csv(output_dir / "book_word_counts.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "book_word_count_source_audit.csv", index=False
    )
    pd.DataFrame(local_text_audit_rows).to_csv(
        output_dir / "book_word_count_local_text_audit.csv", index=False
    )
    pd.DataFrame(local_section_audit_rows).to_csv(
        output_dir / "book_word_count_local_section_audit.csv", index=False
    )
    write_online_error_outputs(result, output_dir)
    write_validation_summary(result, output_dir)
    write_full_finished_projection_outputs(
        result,
        output_dir,
        output_stem="book_word_count_current_finished_projection",
        words_per_page=words_per_page,
    )
    return result


def section_category_counts(section_rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {
        "local_section_count": len(section_rows),
        "local_included_section_count": sum(
            bool(row.get("included_in_reading_count")) for row in section_rows
        ),
        "local_excluded_section_count": sum(
            not bool(row.get("included_in_reading_count")) for row in section_rows
        ),
    }
    category_totals: dict[str, int] = {}
    for row in section_rows:
        category = str(row.get("body_section_category", "") or "unknown")
        category_totals[category] = category_totals.get(category, 0) + int(
            row.get("word_count", 0) or 0
        )
    for category, total in sorted(category_totals.items()):
        output[f"local_{category}_word_count"] = total
    return output


def write_validation_summary(result: pd.DataFrame, output_dir: Path) -> None:
    frame = result[result["local_file_word_count"].fillna(0).gt(0)].copy()
    columns = [
        "title",
        "finish_date",
        "local_file_word_count",
        "local_raw_word_count",
        "local_excluded_word_count",
        "local_excluded_pct",
        "external_word_count",
        "online_error_pct_vs_body_local",
        "metadata_page_count",
        "local_word_count_method",
        "local_file_error",
        "validation_note",
        "local_file_path",
    ]
    if frame.empty:
        pd.DataFrame(columns=columns).to_csv(
            output_dir / "book_word_count_validation_summary.csv", index=False
        )
        return
    frame["local_excluded_pct"] = (
        frame["local_excluded_word_count"] / frame["local_raw_word_count"]
    ).where(frame["local_raw_word_count"].fillna(0).gt(0))
    frame["online_error_pct_vs_body_local"] = (
        (frame["external_word_count"] - frame["local_file_word_count"])
        / frame["local_file_word_count"]
        * 100
    ).where(frame["external_word_count"].fillna(0).gt(0))
    frame["validation_note"] = frame.apply(validation_note, axis=1)
    frame = frame.sort_values(
        ["online_error_pct_vs_body_local", "local_excluded_pct"],
        key=lambda series: series.abs() if "pct" in str(series.name) else series,
        ascending=False,
    )
    frame[columns].to_csv(
        output_dir / "book_word_count_validation_summary.csv", index=False
    )
    _write_markdown_table(
        frame[columns],
        output_dir / "book_word_count_validation_summary.md",
    )


def add_full_finished_projection_columns(
    result: pd.DataFrame, *, words_per_page: int = DEFAULT_WORDS_PER_PAGE
) -> pd.DataFrame:
    """Project a word count for every finished row, preserving source quality."""
    frame = result.copy()
    local = pd.to_numeric(frame.get("local_file_word_count"), errors="coerce")
    pages = pd.to_numeric(frame.get("metadata_page_count"), errors="coerce")
    external = pd.to_numeric(frame.get("external_word_count"), errors="coerce")
    chosen = pd.to_numeric(frame.get("chosen_word_count"), errors="coerce")

    local_page_mask = local.gt(0) & pages.gt(0)
    if local_page_mask.any() and pages[local_page_mask].sum() > 0:
        calibrated_words_per_page = float(
            local[local_page_mask].sum() / pages[local_page_mask].sum()
        )
    else:
        calibrated_words_per_page = float(words_per_page)

    local_external_mask = local.gt(0) & external.gt(0)
    if local_external_mask.any() and external[local_external_mask].sum() > 0:
        online_to_local_scale = float(
            local[local_external_mask].sum() / external[local_external_mask].sum()
        )
    else:
        online_to_local_scale = 1.0

    known_for_mean = local[local.gt(0)]
    if known_for_mean.empty:
        known_for_mean = chosen[chosen.gt(0)]
    global_mean_words = (
        float(known_for_mean.mean()) if not known_for_mean.empty else float("nan")
    )

    frame["projection_local_calibrated_words_per_page"] = calibrated_words_per_page
    frame["projection_online_to_local_scale_factor"] = online_to_local_scale
    frame["projection_global_mean_words"] = global_mean_words
    frame["projected_word_count"] = np.nan
    frame["projected_word_count_method"] = ""
    frame["projected_word_count_confidence"] = "low"
    frame["projection_is_observed_or_feature_based"] = False

    local_mask = local.gt(0)
    page_mask = ~local_mask & pages.gt(0)
    external_mask = ~local_mask & ~page_mask & external.gt(0)
    chosen_mask = ~local_mask & ~page_mask & ~external_mask & chosen.gt(0)
    mean_mask = (
        ~local_mask
        & ~page_mask
        & ~external_mask
        & ~chosen_mask
        & pd.notna(global_mean_words)
    )

    frame.loc[local_mask, "projected_word_count"] = local[local_mask]
    frame.loc[local_mask, "projected_word_count_method"] = "audited_local_file_text"
    frame.loc[local_mask, "projected_word_count_confidence"] = "high"
    frame.loc[local_mask, "projection_is_observed_or_feature_based"] = True

    frame.loc[page_mask, "projected_word_count"] = (
        pages[page_mask] * calibrated_words_per_page
    )
    frame.loc[page_mask, "projected_word_count_method"] = (
        "metadata_pages_x_local_calibrated_words_per_page"
    )
    frame.loc[page_mask, "projected_word_count_confidence"] = "medium"
    frame.loc[page_mask, "projection_is_observed_or_feature_based"] = True

    frame.loc[external_mask, "projected_word_count"] = (
        external[external_mask] * online_to_local_scale
    )
    frame.loc[external_mask, "projected_word_count_method"] = (
        "external_word_count_scaled_to_local_bias"
    )
    frame.loc[external_mask, "projected_word_count_confidence"] = "medium_low"
    frame.loc[external_mask, "projection_is_observed_or_feature_based"] = True

    frame.loc[chosen_mask, "projected_word_count"] = chosen[chosen_mask]
    frame.loc[chosen_mask, "projected_word_count_method"] = (
        "existing_chosen_word_count_no_local_calibration"
    )
    frame.loc[chosen_mask, "projected_word_count_confidence"] = frame.loc[
        chosen_mask, "word_count_confidence"
    ].fillna("low")
    frame.loc[chosen_mask, "projection_is_observed_or_feature_based"] = True

    frame.loc[mean_mask, "projected_word_count"] = global_mean_words
    frame.loc[mean_mask, "projected_word_count_method"] = (
        "global_mean_local_file_word_count_imputation"
    )
    frame.loc[mean_mask, "projected_word_count_confidence"] = "low"

    missing_mask = frame["projected_word_count"].isna()
    frame.loc[missing_mask, "projected_word_count_method"] = (
        "missing_all_word_count_inputs"
    )
    frame["projected_word_count"] = frame["projected_word_count"].round()
    frame["current_chosen_minus_projected_words"] = (
        chosen - frame["projected_word_count"]
    )
    frame["current_chosen_vs_projected_error_rate"] = (
        frame["current_chosen_minus_projected_words"] / frame["projected_word_count"]
    ).where(frame["projected_word_count"].gt(0) & chosen.notna())
    return frame


def build_projection_metric_summary(projection: pd.DataFrame) -> pd.DataFrame:
    projected = pd.to_numeric(projection["projected_word_count"], errors="coerce")
    chosen = pd.to_numeric(projection.get("chosen_word_count"), errors="coerce")
    local = pd.to_numeric(projection.get("local_file_word_count"), errors="coerce")
    pages = pd.to_numeric(projection.get("metadata_page_count"), errors="coerce")
    observed_or_feature = projection["projection_is_observed_or_feature_based"].fillna(
        False
    )
    metrics = {
        "finished_rows": len(projection),
        "unique_normalized_titles": projection["title"]
        .map(lambda value: normalize_title(str(value)))
        .nunique(),
        "rows_with_current_chosen_word_count": int(chosen.gt(0).sum()),
        "rows_with_local_file_word_count": int(local.gt(0).sum()),
        "rows_with_metadata_pages": int(pages.gt(0).sum()),
        "rows_projected_from_observed_or_feature_inputs": int(
            observed_or_feature.sum()
        ),
        "rows_low_confidence_global_mean_imputed": int(
            projection["projected_word_count_method"]
            .eq("global_mean_local_file_word_count_imputation")
            .sum()
        ),
        "current_chosen_total_words": float(chosen.sum()),
        "projected_total_words": float(projected.sum()),
        "projected_delta_words_vs_current_chosen": float(
            projected.sum() - chosen.sum()
        ),
        "projected_delta_pct_vs_current_chosen": (
            float((projected.sum() - chosen.sum()) / chosen.sum() * 100)
            if chosen.sum() > 0
            else np.nan
        ),
        "projected_low_confidence_global_mean_words": float(
            projected[
                projection["projected_word_count_method"].eq(
                    "global_mean_local_file_word_count_imputation"
                )
            ].sum()
        ),
        "local_calibrated_words_per_page": (
            float(projection["projection_local_calibrated_words_per_page"].iloc[0])
            if not projection.empty
            else np.nan
        ),
        "online_to_local_scale_factor": (
            float(projection["projection_online_to_local_scale_factor"].iloc[0])
            if not projection.empty
            else np.nan
        ),
        "global_mean_local_words": (
            float(projection["projection_global_mean_words"].iloc[0])
            if not projection.empty
            else np.nan
        ),
    }
    return pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in metrics.items()]
    )


def build_projection_method_summary(projection: pd.DataFrame) -> pd.DataFrame:
    if projection.empty:
        return pd.DataFrame(
            columns=[
                "projected_word_count_method",
                "rows",
                "projected_total_words",
                "current_chosen_total_words",
                "projected_delta_words_vs_current_chosen",
            ]
        )
    frame = projection.copy()
    frame["projected_word_count"] = pd.to_numeric(
        frame["projected_word_count"], errors="coerce"
    )
    frame["chosen_word_count"] = pd.to_numeric(
        frame.get("chosen_word_count"), errors="coerce"
    )
    grouped = (
        frame.groupby("projected_word_count_method", dropna=False)
        .agg(
            rows=("title", "size"),
            projected_total_words=("projected_word_count", "sum"),
            current_chosen_total_words=("chosen_word_count", "sum"),
        )
        .reset_index()
    )
    grouped["projected_delta_words_vs_current_chosen"] = (
        grouped["projected_total_words"] - grouped["current_chosen_total_words"]
    )
    return grouped.sort_values("rows", ascending=False).reset_index(drop=True)


def write_full_finished_projection_outputs(
    result: pd.DataFrame,
    output_dir: Path,
    *,
    output_stem: str,
    words_per_page: int = DEFAULT_WORDS_PER_PAGE,
) -> pd.DataFrame:
    projection = add_full_finished_projection_columns(
        result, words_per_page=words_per_page
    )
    summary = build_projection_metric_summary(projection)
    methods = build_projection_method_summary(projection)
    projection.to_csv(output_dir / f"{output_stem}.csv", index=False)
    summary.to_csv(output_dir / f"{output_stem}_summary.csv", index=False)
    methods.to_csv(output_dir / f"{output_stem}_methods.csv", index=False)
    _write_markdown_table(projection, output_dir / f"{output_stem}.md")
    _write_markdown_table(summary, output_dir / f"{output_stem}_summary.md")
    _write_markdown_table(methods, output_dir / f"{output_stem}_methods.md")
    return projection


def validation_note(row: pd.Series) -> str:
    notes: list[str] = []
    method = str(row.get("local_word_count_method", "") or "")
    warning = str(row.get("local_file_error", "") or "")
    excluded_pct = numeric_or_zero(row.get("local_excluded_word_count", 0)) / max(
        1, numeric_or_zero(row.get("local_raw_word_count", 0))
    )
    online = numeric_or_zero(row.get("external_word_count", 0))
    local = numeric_or_zero(row.get("local_file_word_count", 0))
    if method in {"epub_spine_body_sections", "pdf_repeated_header_footer_body_pages"}:
        notes.append("section-filtered local text")
    else:
        notes.append("raw local text; no reliable section filter")
    if excluded_pct >= 0.15:
        notes.append("large notes/index/front-back-matter exclusion")
    if online and local:
        error = (online - local) / local
        if error <= -0.25:
            notes.append("online estimate is much lower than body-local count")
        elif error >= 0.25:
            notes.append("online estimate is much higher than body-local count")
    if warning:
        notes.append(warning)
    return "; ".join(dict.fromkeys(notes))


def add_online_error_columns(result: pd.DataFrame) -> None:
    local = pd.to_numeric(result["local_file_word_count"], errors="coerce")
    online = pd.to_numeric(result["external_word_count"], errors="coerce")
    has_pair = local.gt(0) & online.gt(0)
    result["online_minus_local_words"] = np.nan
    result["online_error_rate_vs_local"] = np.nan
    result["online_abs_error_rate_vs_local"] = np.nan
    result.loc[has_pair, "online_minus_local_words"] = (
        online[has_pair] - local[has_pair]
    )
    result.loc[has_pair, "online_error_rate_vs_local"] = (
        online[has_pair] - local[has_pair]
    ) / local[has_pair]
    result.loc[has_pair, "online_abs_error_rate_vs_local"] = result.loc[
        has_pair, "online_error_rate_vs_local"
    ].abs()


def write_online_error_outputs(result: pd.DataFrame, output_dir: Path) -> None:
    compare = result[
        result["local_file_word_count"].fillna(0).gt(0)
        & result["external_word_count"].fillna(0).gt(0)
    ].copy()
    columns = [
        "title",
        "finish_date",
        "local_file_word_count",
        "external_word_count",
        "online_minus_local_words",
        "online_error_rate_vs_local",
        "online_abs_error_rate_vs_local",
        "external_word_count_source",
        "external_word_count_notes",
        "local_file_path",
    ]
    if compare.empty:
        pd.DataFrame(columns=columns).to_csv(
            output_dir / "book_word_count_online_error_rates.csv", index=False
        )
        return
    compare["online_error_pct_vs_local"] = compare["online_error_rate_vs_local"] * 100
    compare["online_abs_error_pct_vs_local"] = (
        compare["online_abs_error_rate_vs_local"] * 100
    )
    compare["_comparison_key"] = compare.apply(online_comparison_key, axis=1)
    compare = compare.drop_duplicates("_comparison_key").copy()
    compare["outlier_investigation"] = compare.apply(
        investigate_online_word_count_error, axis=1
    )
    output_columns = [
        "title",
        "finish_date",
        "local_file_word_count",
        "external_word_count",
        "online_minus_local_words",
        "online_error_pct_vs_local",
        "online_abs_error_pct_vs_local",
        "external_word_count_source",
        "external_word_count_notes",
        "local_file_path",
        "outlier_investigation",
    ]
    compare = compare.sort_values("online_abs_error_pct_vs_local", ascending=False)
    compare[output_columns].to_csv(
        output_dir / "book_word_count_online_error_rates.csv", index=False
    )
    _write_markdown_table(
        compare[output_columns],
        output_dir / "book_word_count_online_error_rates.md",
    )
    threshold = 25.0
    outliers = compare[compare["online_abs_error_pct_vs_local"].ge(threshold)].copy()
    outliers[output_columns].to_csv(
        output_dir / "book_word_count_online_error_outliers.csv", index=False
    )
    _write_markdown_table(
        outliers[output_columns],
        output_dir / "book_word_count_online_error_outliers.md",
    )
    plot_online_error_rates(
        compare,
        output_dir / "book_word_count_online_error_rates.png",
    )


def online_comparison_key(row: pd.Series) -> str:
    return "|".join(
        [
            str(
                row.get("local_file_path", "") or normalize_title(row.get("title", ""))
            ),
            str(int(float(row.get("local_file_word_count", 0) or 0))),
            str(int(float(row.get("external_word_count", 0) or 0))),
            str(row.get("external_word_count_source", "") or ""),
        ]
    )


def investigate_online_word_count_error(row: pd.Series) -> str:
    error_rate = float(row.get("online_error_rate_vs_local", math.nan))
    if math.isnan(error_rate):
        return ""
    notes = str(row.get("external_word_count_notes", "") or "").lower()
    source = str(row.get("external_word_count_source", "") or "").lower()
    path = str(row.get("local_file_path", "") or "").lower()
    local_count = float(row.get("local_file_word_count", 0) or 0)
    online_count = float(row.get("external_word_count", 0) or 0)
    if abs(error_rate) < 0.10:
        return "online count is within 10% of the local extracted text count"
    reasons: list[str] = []
    if "readinglength.com" in source:
        reasons.append(
            "ReadingLength marks this as an estimate, so treat it as an online-search baseline rather than a measured count"
        )
    if "audiobook" in notes:
        reasons.append("online count is inferred from audiobook duration")
    if "rough guess" in notes or "guess" in notes:
        reasons.append("online count appears to be a page-count-style rough guess")
    if online_count < local_count:
        reasons.append("online estimate is lower than the extracted local text")
    elif online_count > local_count:
        reasons.append("online estimate is higher than the extracted local text")
    if any(
        marker in path
        for marker in [
            "abridged",
            "annotated",
            "omnibus",
            "collection",
            "complete",
            "essays",
            "anthology",
            "edition",
        ]
    ):
        reasons.append("local filename suggests edition/front-back-matter differences")
    return (
        "; ".join(dict.fromkeys(reasons)) or "inspect local edition and online source"
    )


def plot_online_error_rates(frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = frame.sort_values("online_error_pct_vs_local").copy()
    if len(plot_frame) > 40:
        plot_frame = (
            plot_frame.reindex(
                plot_frame["online_abs_error_pct_vs_local"]
                .sort_values(ascending=False)
                .head(40)
                .index
            )
            .sort_values("online_error_pct_vs_local")
            .copy()
        )
    height = max(6, len(plot_frame) * 0.38 + 1.5)
    colors = [
        "#dc2626" if value < 0 else "#2563eb"
        for value in plot_frame["online_error_pct_vs_local"]
    ]
    plt.figure(figsize=(12, height))
    positions = np.arange(len(plot_frame))
    plt.barh(positions, plot_frame["online_error_pct_vs_local"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(positions, [truncate_title(title, 42) for title in plot_frame["title"]])
    plt.xlabel("Online word-count error vs local extracted text (%)")
    plt.title("Online Word-Count Search Error vs Local EPUB/PDF Text Extraction")
    max_abs = max(10, float(plot_frame["online_error_pct_vs_local"].abs().max()))
    plt.xlim(-max_abs * 1.15, max_abs * 1.15)
    for pos, value in zip(positions, plot_frame["online_error_pct_vs_local"]):
        offset = max_abs * 0.02
        ha = "left" if value >= 0 else "right"
        x = value + offset if value >= 0 else value - offset
        plt.text(x, pos, f"{value:+.0f}%", va="center", ha=ha, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def truncate_title(title: object, width: int) -> str:
    text = str(title)
    if len(text) <= width:
        return text
    return text[: width - 1] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--titles-csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "book_calendar_first_pass_finishes.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--book-root", action="append", type=Path, default=[])
    parser.add_argument("--manual-word-counts", type=Path)
    parser.add_argument("--word-source-csv", type=Path, default=DEFAULT_WORD_SOURCE_CSV)
    parser.add_argument("--words-per-page", type=int, default=DEFAULT_WORDS_PER_PAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    titles = pd.read_csv(args.titles_csv)
    result = build_word_count_outputs(
        titles=titles,
        output_dir=args.output_dir,
        book_roots=args.book_root or None,
        manual_word_counts=args.manual_word_counts,
        word_source_csv=args.word_source_csv,
        words_per_page=args.words_per_page,
    )
    with pd.option_context("display.max_rows", 30, "display.width", 180):
        print(
            result.sort_values("finish_date", ascending=False)
            .head(30)
            .to_string(index=False)
        )
    print(f"\nWrote word-count output to {args.output_dir / 'book_word_counts.csv'}")


if __name__ == "__main__":
    main()
