"""Estimate book word counts with local files as the first-class source.

The script produces a standalone word-count table. It tries local readable book
files first, then falls back to existing metadata page counts. Web/manual sources
can be added as explicit CSV rows so their provenance is visible.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_books_tracking.book_wpm_calendar_notes import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WORDS_PER_PAGE,
    count_words,
    find_metadata_record,
    load_metadata_records,
    normalize_title,
    titles_match,
)


DEFAULT_BOOK_ROOTS = [
    REPO_ROOT / "data" / "Books",
    REPO_ROOT / "data" / "Books" / "eleven_upload",
    Path("/Users/clarkbenham/Documents/Books"),
    Path("/Users/clarkbenham/Documents/Books/eleven_upload"),
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


@dataclass(frozen=True)
class LocalBookFile:
    path: Path
    name_norm: str
    stem_norm: str


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
    if path.suffix.lower() in {".crdownload", ".jsonl", ".xml"}:
        return True
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
        if candidate == file.stem_norm or candidate == file.name_norm:
            score = max(score, 100.0)
        elif phrase_contains(candidate, file.stem_norm) or phrase_contains(
            file.stem_norm, candidate
        ):
            score = max(score, 90.0)
        elif titles_match(candidate, file.stem_norm):
            score = max(score, 85.0)
        else:
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
            texts.append(BeautifulSoup(html, "html.parser").get_text(" "))
    return "\n".join(texts)


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
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for _, title_row in titles.iterrows():
        title = str(title_row["title"])
        filename = str(
            title_row.get("filename", "") or title_row.get("calendar_finish_ref", "")
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
        if local_path is None:
            local_path, local_score = find_local_file(title, filename, local_files)
        local_word_count = None
        local_error = ""
        if local_path is not None:
            try:
                local_word_count = cleaned_word_count(text_from_path(local_path))
                if local_word_count <= 0:
                    local_error = "no_text_extracted"
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
                    "notes": "",
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
        if not manual_match.empty:
            chosen_words = float(manual_match.iloc[0]["word_count"])
            chosen_source = str(manual_match.iloc[0]["source"])
            confidence = "high"
        elif local_word_count:
            chosen_words = float(local_word_count)
            chosen_source = "local_file_word_count"
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
                "finish_date": title_row.get("finish_date", ""),
                "calendar_finish_ref": title_row.get("calendar_finish_ref", ""),
                "local_file_path": str(local_path) if local_path else "",
                "local_file_match_score": local_score,
                "local_file_word_count": local_word_count,
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
    result.to_csv(output_dir / "book_word_counts.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "book_word_count_source_audit.csv", index=False
    )
    return result


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
