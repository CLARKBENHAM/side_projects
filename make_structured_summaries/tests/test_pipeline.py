from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from structured_summaries.chunk_companion import render_expanded_chunk_notes
from structured_summaries.llm_backends import PromptExecution
from structured_summaries.models import BookRecord
from structured_summaries.pipeline import SummaryConfig, summarize_book_async
from structured_summaries import pipeline as pipeline_module


def _book(tmp_path: Path) -> BookRecord:
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    primary_path = book_dir / "sample.txt"
    primary_path.write_text(
        "The core model is legibility.\n\n"
        "A second model is how power compounds through institutions.\n",
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


def test_async_pipeline_writes_completed_manifest(tmp_path: Path) -> None:
    book = _book(tmp_path)
    config = SummaryConfig(
        chunk_backend="stub",
        synthesis_backend="stub",
        critique_backend="stub",
        chunk_chars=80,
        overlap_chars=0,
        force=True,
    )

    artifacts = asyncio.run(
        summarize_book_async(
            book,
            project_root=tmp_path,
            config=config,
        )
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    stage_names = [stage["name"] for stage in manifest["stage_records"]]

    assert artifacts.expanded_notes_path is not None
    assert artifacts.expanded_notes_path.exists()
    assert artifacts.summary_path is not None and artifacts.summary_path.exists()
    assert artifacts.critique_path is not None and artifacts.critique_path.exists()
    assert manifest["status"] == "completed"
    assert manifest["current_stage"] == "completed"
    assert manifest["chunk_count"] >= 1
    assert stage_names[0] == "extract_text"
    assert "expanded_notes_render" in stage_names
    assert "synthesis_llm" in stage_names
    assert "critique_llm" in stage_names


def test_render_expanded_chunk_notes_strips_json_but_keeps_detail(
    tmp_path: Path,
) -> None:
    book = _book(tmp_path)
    chunk_outputs = [
        "```json\n"
        + json.dumps(
            {
                "core_models": [
                    {
                        "name": "Conduit of Power",
                        "summary": "Power comes from controlling access.",
                        "why_it_matters": "It explains why administrative work compounds.",
                    }
                ],
                "key_facts_and_mechanisms": [
                    "Johnson answered constituent mail the same day it arrived."
                ],
                "legibility_gains": [
                    "Administrative throughput can become political capital."
                ],
                "reasoning_methods": ["Map the bottleneck, then own it."],
                "practical_transfers": [
                    "Solve the tedious intake work no one else wants."
                ],
                "named_tools_and_metrics": [
                    {
                        "name": "Nose-counting",
                        "what_it_is": "A detailed audit of who will vote.",
                        "when_to_use": "When a low-turnout race can be won at the margin.",
                    }
                ],
                "best_examples": [
                    {
                        "label": "The office mail machine",
                        "supports": "Conduit of Power",
                        "setup": "A neglected congressional office handled constituent requests slowly.",
                        "mechanism": "Johnson forced same-day replies and tracked every request.",
                        "payoff": "Gratitude and dependency flowed through him instead of the Congressman.",
                    }
                ],
                "tactical_wins": [
                    {
                        "label": "Weekend blitz",
                        "setup": "A list of names had to be reached fast.",
                        "mechanism": "He split the work across loyalists for door-to-door contact.",
                        "why_it_matters": "It turned staff scale into electoral edge.",
                    }
                ],
                "non_obvious_claims": [
                    "Back-office speed can matter more than speechmaking."
                ],
                "pushback_points": [
                    "The narrative may over-attribute outcomes to Johnson alone."
                ],
                "checkable_claims": [
                    {
                        "claim": "He used staff to answer every letter the same day.",
                        "why_check": "It is central to the throughput argument.",
                    }
                ],
            }
        )
        + "\n```"
    ]

    rendered = render_expanded_chunk_notes(book, chunk_outputs)

    assert rendered.startswith("# Expanded Chunk Notes: Sample Book")
    assert '"core_models"' not in rendered
    assert "### Core Models" in rendered
    assert "### Key Facts And Mechanisms" in rendered
    assert "Johnson answered constituent mail the same day it arrived." in rendered
    assert "### Best Examples" in rendered
    assert (
        "Setup: A neglected congressional office handled constituent requests slowly."
        in rendered
    )
    assert "### Named Tools And Metrics" in rendered
    assert "When to use: When a low-turnout race can be won at the margin." in rendered


def test_async_pipeline_parallelizes_chunks_but_preserves_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    book_dir = tmp_path / "book_parallel"
    book_dir.mkdir()
    primary_path = book_dir / "parallel.txt"
    primary_path.write_text(
        "\n\n".join(
            [
                "chunk one carries one idea",
                "chunk two carries a second idea",
                "chunk three carries a third idea",
                "chunk four carries a fourth idea",
            ]
        ),
        encoding="utf-8",
    )
    book = BookRecord(
        book_id="parallel-book",
        title="Parallel Book",
        author="Test Author",
        book_dir=book_dir,
        primary_path=primary_path,
        primary_format="txt",
        companion_html_path=None,
        all_paths=(primary_path,),
        search_title="Parallel Book",
    )

    active_chunks = 0
    peak_chunks = 0
    active_lock = asyncio.Lock()

    async def fake_run_prompt_async(
        backend: str,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        timeout: int = 900,
        limiter=None,
    ) -> PromptExecution:
        nonlocal active_chunks, peak_chunks
        del backend, model, system_prompt, timeout, limiter
        now = pipeline_module._now_iso()
        match = re.search(r"- chunk: (\d+) of (\d+)", prompt)
        if not match:
            return PromptExecution(
                output=prompt,
                command=(),
                started_at=now,
                finished_at=now,
                duration_seconds=0.0,
            )

        chunk_index = int(match.group(1))
        async with active_lock:
            active_chunks += 1
            peak_chunks = max(peak_chunks, active_chunks)
        await asyncio.sleep(0.02 * (5 - chunk_index))
        async with active_lock:
            active_chunks -= 1
        output = json.dumps(
            {
                "core_models": [],
                "key_facts_and_mechanisms": [f"fact chunk {chunk_index}"],
                "legibility_gains": [],
                "reasoning_methods": [],
                "practical_transfers": [],
                "named_tools_and_metrics": [],
                "best_examples": [],
                "tactical_wins": [],
                "non_obvious_claims": [],
                "pushback_points": [],
                "checkable_claims": [],
            }
        )
        return PromptExecution(
            output=output,
            command=(),
            started_at=now,
            finished_at=pipeline_module._now_iso(),
            duration_seconds=0.01,
        )

    monkeypatch.setattr(pipeline_module, "run_prompt_async", fake_run_prompt_async)

    config = SummaryConfig(
        chunk_backend="gemini",
        chunk_model="fake",
        synthesis_backend="gemini",
        synthesis_model="fake",
        critique_backend=None,
        chunk_chars=24,
        overlap_chars=0,
        chunk_concurrency=3,
        force=True,
    )
    artifacts = asyncio.run(
        summarize_book_async(
            book,
            project_root=tmp_path,
            config=config,
        )
    )

    summary_text = artifacts.summary_path.read_text(encoding="utf-8")
    expanded_text = artifacts.expanded_notes_path.read_text(encoding="utf-8")

    assert peak_chunks > 1
    assert summary_text.index("fact chunk 1") < summary_text.index("fact chunk 2")
    assert summary_text.index("fact chunk 2") < summary_text.index("fact chunk 3")
    assert "## Chunk 1" in expanded_text
    assert expanded_text.index("fact chunk 1") < expanded_text.index("fact chunk 2")
