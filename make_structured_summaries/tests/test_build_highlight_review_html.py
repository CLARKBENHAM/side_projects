from __future__ import annotations

from pathlib import Path

from scripts.build_highlight_review_html import (
    extract_ai_chunks,
    parse_chunk_json,
    render_ai_chunk,
)


def test_parse_chunk_json_handles_fenced_json() -> None:
    parsed = parse_chunk_json(
        """```json
{"key_facts_and_mechanisms": ["A useful mechanism."]}
```"""
    )

    assert parsed == {"key_facts_and_mechanisms": ["A useful mechanism."]}


def test_extract_ai_chunks_preserves_chunk_boundaries(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_001_response.txt").write_text(
        """{
  "core_models": [
    {
      "name": "Constraint",
      "summary": "The bottleneck governs throughput.",
      "why_it_matters": "It changes where attention goes."
    }
  ],
  "key_facts_and_mechanisms": ["Local efficiency can hurt global flow."]
}""",
        encoding="utf-8",
    )
    (chunk_dir / "chunk_001_prompt.txt").write_text(
        "Prompt preface\nChunk text:\nThe bottleneck governs throughput.",
        encoding="utf-8",
    )
    (chunk_dir / "chunk_002_response.txt").write_text(
        """{
  "orientation_facts": [
    {
      "fact": "The plant is judged by throughput.",
      "why_it_matters": "It prevents local-efficiency confusion."
    }
  ],
  "key_facts_and_mechanisms": ["Buffers protect the constraint."]
}""",
        encoding="utf-8",
    )

    chunks = extract_ai_chunks(chunk_dir)

    assert [chunk.chunk_index for chunk in chunks] == [1, 2]
    assert len(chunks[0].items) == 2
    assert chunks[0].items[0].label == "Constraint"
    assert chunks[0].source_text == "The bottleneck governs throughput."
    assert chunks[1].items[0].label == "The plant is judged by throughput."
    assert chunks[1].items[1].text == "Buffers protect the constraint."


def test_render_ai_chunk_does_not_emit_match_scores(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_001_response.txt").write_text(
        """{"key_facts_and_mechanisms": ["One important item."]}""",
        encoding="utf-8",
    )

    html = render_ai_chunk(extract_ai_chunks(chunk_dir)[0])

    assert "One important item." in html
    assert "score" not in html
    assert "ai-match" not in html
