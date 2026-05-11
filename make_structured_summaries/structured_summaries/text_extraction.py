"""Text extraction for local book files."""

from __future__ import annotations

import posixpath
import re
import shutil
import subprocess
import zipfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from .models import BookRecord

HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}


class ExtractionError(RuntimeError):
    """Raised when book text cannot be extracted."""


class _HTMLToTextParser(HTMLParser):
    block_tags = {
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "footer",
        "li",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }
    ignored_tags = {"script", "style"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored_tags:
            self.ignored_depth += 1
            return
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self.ignored_depth:
            self.ignored_depth -= 1
            return
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def clean_extracted_text(text: str) -> str:
    lines = [
        line.strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(re.sub(r"[ \t]+", " ", line))
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def html_to_text(raw_html: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(raw_html)
    parser.close()
    return clean_extracted_text(unescape(parser.text()))


def extract_text_from_html(path: Path) -> str:
    return html_to_text(path.read_text(encoding="utf-8", errors="ignore"))


def _read_epub_package_path(epub: zipfile.ZipFile) -> str | None:
    try:
        container_xml = epub.read("META-INF/container.xml")
    except KeyError:
        return None
    root = ElementTree.fromstring(container_xml)
    rootfile = root.find(".//{*}rootfile")
    return rootfile.attrib.get("full-path") if rootfile is not None else None


def _resolve_epub_href(package_path: str, href: str) -> str:
    base_dir = posixpath.dirname(package_path)
    return posixpath.normpath(posixpath.join(base_dir, href))


def _ordered_epub_documents(epub: zipfile.ZipFile) -> list[str]:
    package_path = _read_epub_package_path(epub)
    if not package_path:
        return sorted(
            name
            for name in epub.namelist()
            if Path(name).suffix.lower() in HTML_EXTENSIONS
        )

    package_xml = ElementTree.fromstring(epub.read(package_path))
    manifest_items = package_xml.findall(".//{*}manifest/{*}item")
    manifest = {
        item.attrib["id"]: _resolve_epub_href(package_path, item.attrib["href"])
        for item in manifest_items
        if "id" in item.attrib and "href" in item.attrib
    }
    spine_items = package_xml.findall(".//{*}spine/{*}itemref")
    ordered = [
        manifest[item.attrib["idref"]]
        for item in spine_items
        if item.attrib.get("idref") in manifest
        and Path(manifest[item.attrib["idref"]]).suffix.lower() in HTML_EXTENSIONS
    ]
    if ordered:
        return ordered
    return sorted(
        name for name in epub.namelist() if Path(name).suffix.lower() in HTML_EXTENSIONS
    )


def extract_text_from_epub(path: Path) -> str:
    with zipfile.ZipFile(path) as epub:
        documents = _ordered_epub_documents(epub)
        chunks: list[str] = []
        for document in documents:
            try:
                raw = epub.read(document).decode("utf-8", errors="ignore")
            except KeyError:
                continue
            text = html_to_text(raw)
            if text:
                chunks.append(text)
    if not chunks:
        raise ExtractionError(f"No readable XHTML content found in {path}")
    return clean_extracted_text("\n\n".join(chunks))


def extract_text_from_pdf(path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise ExtractionError(
            "pdftotext is not installed; install poppler or use an EPUB/HTML source"
        )
    result = subprocess.run(
        [pdftotext, "-enc", "UTF-8", "-nopgbrk", str(path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ExtractionError(result.stderr.strip() or f"pdftotext failed for {path}")
    return clean_extracted_text(result.stdout)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return clean_extracted_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix in HTML_EXTENSIONS:
        return extract_text_from_html(path)
    if suffix == ".epub":
        return extract_text_from_epub(path)
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    raise ExtractionError(f"Unsupported file type: {path.suffix}")


def extract_book_text(book: BookRecord) -> str:
    try:
        return extract_text(book.primary_path)
    except ExtractionError:
        if book.companion_html_path and book.companion_html_path.exists():
            return extract_text(book.companion_html_path)
        raise
