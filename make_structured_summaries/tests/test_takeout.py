from __future__ import annotations

from pathlib import Path

from structured_summaries.takeout import scan_takeout_root


def test_scan_takeout_root_prefers_epub_and_tracks_html_companion(
    tmp_path: Path,
) -> None:
    book_dir = tmp_path / "The Power Broker"
    book_dir.mkdir()
    (book_dir / "The Power Broker.epub").write_text("epub bytes placeholder")
    (book_dir / "The Power Broker.html").write_text(
        "<html><body>html text</body></html>"
    )
    (book_dir / "The Power Broker.pdf").write_text("pdf placeholder")
    (book_dir / "The Power Broker(1).pdf").write_text("duplicate pdf placeholder")

    records = scan_takeout_root(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.primary_format == "epub"
    assert record.primary_path.name == "The Power Broker.epub"
    assert record.companion_html_path is not None
    assert record.companion_html_path.name == "The Power Broker.html"
    assert record.title == "The Power Broker"
