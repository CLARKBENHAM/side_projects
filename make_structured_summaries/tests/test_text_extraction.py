from __future__ import annotations

import zipfile
from pathlib import Path

from structured_summaries.text_extraction import extract_text, html_to_text


def test_html_to_text_drops_script_content() -> None:
    text = html_to_text(
        "<html><body><h1>Title</h1><script>ignore()</script><p>Hello world.</p></body></html>"
    )
    assert "Title" in text
    assert "Hello world." in text
    assert "ignore()" not in text


def test_extract_text_from_epub_uses_spine_order(tmp_path: Path) -> None:
    epub_path = tmp_path / "sample.epub"
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml" />
              </rootfiles>
            </container>
            """,
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="2.0">
              <manifest>
                <item id="chap1" href="chap1.xhtml" media-type="application/xhtml+xml" />
                <item id="chap2" href="chap2.xhtml" media-type="application/xhtml+xml" />
              </manifest>
              <spine>
                <itemref idref="chap1" />
                <itemref idref="chap2" />
              </spine>
            </package>
            """,
        )
        archive.writestr(
            "OEBPS/chap1.xhtml",
            "<html><body><h1>Chapter One</h1><p>First paragraph.</p></body></html>",
        )
        archive.writestr(
            "OEBPS/chap2.xhtml",
            "<html><body><h1>Chapter Two</h1><p>Second paragraph.</p></body></html>",
        )

    extracted = extract_text(epub_path)

    assert "Chapter One" in extracted
    assert "First paragraph." in extracted
    assert extracted.index("Chapter One") < extracted.index("Chapter Two")
