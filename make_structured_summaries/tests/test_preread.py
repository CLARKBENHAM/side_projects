from __future__ import annotations

import json
from pathlib import Path

from structured_summaries.models import BookRecord
from structured_summaries.preread import PrereadConfig, build_preread_brief


def _book(tmp_path: Path) -> BookRecord:
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    primary_path = book_dir / "sample.txt"
    primary_path.write_text(
        "The Senate operated through rules, seniority, and informal vetoes.\n\n"
        "Johnson changed the usable power of the majority leader by counting votes "
        "more precisely and making senators feel that he could deliver outcomes.\n",
        encoding="utf-8",
    )
    return BookRecord(
        book_id="sample-book",
        title="Sample Book",
        author="Test Author",
        book_dir=book_dir,
        primary_path=primary_path,
        primary_format="txt",
        companion_html_path=None,
        all_paths=(primary_path,),
        search_title="Sample Book",
    )


def test_build_preread_brief_writes_separate_artifacts(tmp_path: Path) -> None:
    config = PrereadConfig(
        chunk_backend="stub",
        synthesis_backend="stub",
        chunk_chars=120,
        overlap_chars=0,
        force=True,
    )

    artifacts = build_preread_brief(
        _book(tmp_path),
        project_root=tmp_path,
        config=config,
    )

    assert artifacts.extracted_text_path.exists()
    assert artifacts.chunk_dir == tmp_path / "data" / "preread_chunk_notes" / "sample-book"
    assert artifacts.summary_path == tmp_path / "data" / "preread_summaries" / "sample-book.md"
    assert artifacts.summary_path.exists()
    assert (artifacts.chunk_dir / "chunk_001_prompt.txt").exists()
    assert (artifacts.chunk_dir / "chunk_001_response.txt").exists()
    assert (artifacts.chunk_dir / "synthesis_prompt.txt").exists()

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["current_stage"] == "completed"
    assert manifest["chunk_count"] >= 1
    assert manifest["config"]["chunk_backend"] == "stub"
    assert manifest["paths"]["summary_path"] == str(artifacts.summary_path)

    prompt = (artifacts.chunk_dir / "chunk_001_prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "red highlights = most important factual/structural signal" in prompt
    assert "blue highlights = personal resonance" in prompt
