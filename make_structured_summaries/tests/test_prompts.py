from __future__ import annotations

from pathlib import Path

from structured_summaries.models import BookRecord
from structured_summaries.prompts import (
    build_color_alignment_review_prompt,
    build_chunk_analysis_prompt,
    build_preread_chunk_prompt,
    build_preread_synthesis_prompt,
    build_synthesis_prompt,
)


def _book() -> BookRecord:
    path = Path("/tmp/book.epub")
    return BookRecord(
        book_id="seeing-like-a-state",
        title="Seeing Like a State",
        author="James C. Scott",
        book_dir=Path("/tmp"),
        primary_path=path,
        primary_format="epub",
        companion_html_path=None,
        all_paths=(path,),
        search_title="Seeing Like a State",
    )


def test_chunk_prompt_prioritizes_models_over_recap() -> None:
    prompt = build_chunk_analysis_prompt(
        _book(),
        "Sample text",
        chunk_index=1,
        total_chunks=3,
    )

    assert "Do not write a chapter-by-chapter recap." in prompt
    assert "concrete facts, mechanisms, events, and institutional realities" in prompt
    assert "mental models and frameworks" in prompt
    assert "claims worth checking or pushing back against" in prompt
    assert "key_facts_and_mechanisms" in prompt
    assert "reasoning_methods" in prompt
    assert "practical_transfers" in prompt
    assert "named_tools_and_metrics" in prompt
    assert "tactical_wins" in prompt
    assert '"setup": "what the situation was and who was involved' in prompt
    assert '"mechanism": "what specifically happened and how it worked' in prompt
    assert '"what_it_is": "concrete definition in 1-2 sentences"' in prompt
    assert "prefer consequential institutional-scale events" in prompt
    assert "Do not promote a case just because it is vivid" in prompt


def test_synthesis_prompt_requests_pushback_and_legibility() -> None:
    prompt = build_synthesis_prompt(_book(), ['{"core_models": ["legibility"]}'])

    assert "What Actually Happened" in prompt
    assert "Best Illustrative Cases" in prompt
    assert "How The Author Thinks" in prompt
    assert "What This Book Makes Newly Legible" in prompt
    assert "Core Mental Models" in prompt
    assert "Where The Argument Is Strongest / Where To Push Back" in prompt
    assert "Do not collapse into a book report." in prompt
    assert "Named Tools, Metrics, And Procedural Tricks" in prompt
    assert "lead with concrete facts, mechanisms, and events" in prompt
    assert "mental models should be synthesized from the details" in prompt
    assert "must be self-contained" in prompt
    assert "sort examples by consequence and explanatory value" in prompt
    assert "omit local slang, quips, or period color" in prompt
    assert "do not infer importance from narrative vividness alone" in prompt
    assert "prefer specific named late-book cases, tools, and metrics" in prompt
    assert "do not generate psychological interpretations" in prompt


def test_preread_prompts_are_red_yellow_calibrated_and_concise() -> None:
    chunk_prompt = build_preread_chunk_prompt(
        _book(),
        "Sample text",
        chunk_index=1,
        total_chunks=2,
    )
    synthesis_prompt = build_preread_synthesis_prompt(
        _book(),
        ['{"arc_events": [{"event": "A thing happened"}]}'],
    )

    assert "red highlights = most important factual/structural signal" in chunk_prompt
    assert "yellow highlights = important normal signal" in chunk_prompt
    assert "blue highlights = personal resonance" in chunk_prompt
    assert "do not optimize for these in pre-reading" in chunk_prompt
    assert "orientation_facts" in chunk_prompt
    assert "arc_events" in chunk_prompt
    assert "load_bearing_scenes" in chunk_prompt
    assert "Stay within the item limits." in chunk_prompt
    assert "roughly 1,000-1,500 words" in synthesis_prompt
    assert "The Arc To Keep In Your Head" in synthesis_prompt
    assert "What To Watch For While Reading" in synthesis_prompt
    assert "Optimize for red highlights first" in synthesis_prompt
    assert "Do not add a generic \"questions to verify\" section." in synthesis_prompt


def test_color_alignment_prompt_prioritizes_red_grounding_and_chunk_mapping() -> None:
    prompt = build_color_alignment_review_prompt(
        title="The Goal",
        summary_text="Summary text",
        chunk_density_text="- Chunk 1: 8 highlights (normal); blue 1, red 1, green 0, yellow 6",
        yellow_chunk_review_text="### Chunk 1\nYellow highlights:\n- [p44 | 36.7% | chunk 1] Example",
        final_signal_text="Blue highlights\n- [p88 | 73.3% | chunk 2] Apply this\n\nRed highlights\n- [p44 | 36.7% | chunk 1] Fact",
        green_signal_text="Green highlights\n- [p55 | 45.8% | chunk 1] This is how Jonah thinks",
    )

    assert "1. Executive Summary" in prompt
    assert (
        "Treat red highlights as the strongest signal for factual fidelity." in prompt
    )
    assert "do not let blue dominate the judgment" in prompt
    assert (
        "Compare yellow highlights primarily against the estimated chunk extraction"
        in prompt
    )
    assert "If a chunk has very few highlights" in prompt
    assert "Detail Sufficiency Of The Summary's Examples" in prompt
    assert "lacks setup, mechanism, or payoff" in prompt
