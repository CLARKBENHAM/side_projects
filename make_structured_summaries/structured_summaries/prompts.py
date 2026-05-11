"""Prompt builders for summary and review tasks."""

from __future__ import annotations

from .models import BookRecord


def build_chunk_analysis_prompt(
    book: BookRecord,
    chunk_text: str,
    *,
    chunk_index: int,
    total_chunks: int,
) -> str:
    return f"""You are extracting the durable intellectual payload of a book.

Book:
- title: {book.title}
- author: {book.author or "unknown"}
- chunk: {chunk_index} of {total_chunks}

Do not write a chapter-by-chapter recap.
Do not prioritize chronology unless the chronology itself is the point.
Do prioritize:
- concrete facts, mechanisms, events, and institutional realities
- the few examples that actually carry the argument at meaningful scale
- mental models and frameworks
- what the book makes newly legible
- power, status, persuasion, coordination, or institutional dynamics
- decision procedures, diagnostic logic, or "if...then" reasoning methods
- practical transfers or personal applications
- later tactical wins, extensions, or counterintuitive allies
- claims worth checking or pushing back against

Additional rules:
- Do not discard later refinements just because the main model is already established.
- If this chunk contains a later-stage application, extension, or tactical breakthrough, capture it explicitly.
- Preserve the difference between a framework, a fact, an example, a reasoning method, and a personal application.
- If there is distinctive experiential or emotional texture, capture it inside the relevant fact or example, not as a separate abstraction.
- Environmental pressures such as geography, markets, technology, or institutions are facts about the system. Capture them concretely.
- If a strategy has a specific failure mode or counter-move, include it in the relevant model or example, not as a separate orphaned bullet.
- If a behavioral pattern drives outcomes, describe it as a concrete observation supported by the text, not as a speculative psychological guess.
- For biography or history, prefer consequential institutional-scale events over childhood foreshadowing or colorful but low-stakes anecdotes unless the anecdote is genuinely indispensable.
- Do not promote a case just because it is vivid; promote it because it teaches a mechanism a new reader can actually use.

Return JSON only with this schema:
{{
  "core_models": [
    {{
      "name": "short label",
      "summary": "2-4 sentence explanation",
      "why_it_matters": "what this changes in how a reader sees the world"
    }}
  ],
  "key_facts_and_mechanisms": [
    "specific factual claim, institutional mechanism, environmental condition, or behavioral pattern that a reader needs to understand the book's argument"
  ],
  "legibility_gains": ["what becomes easier to see after this chunk"],
  "reasoning_methods": [
    "specific diagnostic sequence, causal logic, or way of thinking used here"
  ],
  "practical_transfers": [
    "how a reader could apply this outside the immediate surface domain"
  ],
  "named_tools_and_metrics": [
    {{
      "name": "specific acronym, formula, named heuristic, or operational measure",
      "what_it_is": "concrete definition in 1-2 sentences",
      "when_to_use": "the situation in which this tool or metric applies"
    }}
  ],
  "best_examples": [
    {{
      "label": "short example label",
      "supports": "which model or claim it best illustrates",
      "setup": "what the situation was and who was involved in 2-3 sentences",
      "mechanism": "what specifically happened and how it worked in 2-3 sentences",
      "payoff": "what resulted and why it matters in 1-2 sentences"
    }}
  ],
  "tactical_wins": [
    {{
      "label": "specific later move, refinement, or implementation win",
      "setup": "context in 1-2 sentences",
      "mechanism": "what was done in 1-2 sentences",
      "why_it_matters": "why this is not just another example"
    }}
  ],
  "non_obvious_claims": ["claims that could update a smart reader"],
  "pushback_points": ["where the author may overreach or where evidence seems thin"],
  "checkable_claims": [
    {{
      "claim": "specific empirical or historical claim",
      "why_check": "why external verification would matter"
    }}
  ]
}}

Chunk text:
{chunk_text}
"""


def build_synthesis_prompt(book: BookRecord, chunk_outputs: list[str]) -> str:
    joined = "\n\n".join(
        f"--- chunk {index} extraction ---\n{output}"
        for index, output in enumerate(chunk_outputs, start=1)
    )
    return f"""You are writing a structured summary of a book.

Book:
- title: {book.title}
- author: {book.author or "unknown"}

Use the chunk extractions below.
Do not collapse into a book report.
Do not organize this as chapter 1, chapter 2, chapter 3.

Before writing, identify any later chunks that add genuinely new method, application,
or tactical guidance. Do not let famous early examples crowd them out.

Write a Markdown summary of roughly 2,000 to 3,000 words with these sections:

1. What Actually Happened
2. Best Illustrative Cases
3. Named Tools, Metrics, And Procedural Tricks
4. How The Author Thinks
5. What This Book Makes Newly Legible
6. Core Mental Models
7. Where The Argument Is Strongest / Where To Push Back
8. Claims Worth Verifying
9. Reading Payoff / Personal Application

Rules:
- lead with concrete facts, mechanisms, and events, not with abstractions
- do not organize as chapter 1, chapter 2, chapter 3, but do organize by importance
- mental models should be synthesized from the details, not stated before them
- keep only the strongest examples
- Section 1 should cover the 5-8 most important factual claims, institutional mechanisms, environmental conditions, or behavioral patterns in the book
- state observations concretely, not as thematic glosses
- in Section 1, prefer plain factual statements over giving every bullet a clever label or mini-theory name
- if the book is a biography or history, include enough context that a new reader understands who did what and why
- each example in "Best Illustrative Cases" must be self-contained: a reader who has not read the book should understand the setup, what happened, and why it matters
- if an example cannot be explained clearly in 3-5 sentences, either expand it or replace it with one that can
- do not include examples that are merely vivid; include examples that carry the argument or reveal a mechanism
- sort examples by consequence and explanatory value, not by charm, novelty, or early foreshadowing
- for biography or history, prefer mature or institutionally consequential cases unless an early anecdote is clearly one of the book's central proofs
- spell out transferable frameworks
- if a behavioral pattern drives outcomes, state the pattern as a factual observation inside the relevant section, not as a psychological speculation
- do not generate psychological interpretations that the book does not explicitly support
- preserve the methods of reasoning, not just the conclusions
- preserve what a reader could do differently after reading, not just what happened
- include distinctive later-stage material when it changes the practical payoff
- prefer specific named late-book cases, tools, and metrics over generic paraphrases
- if later chunks introduce concrete acronyms, formulas, or unusual implementation wins, include them directly
- only include a named tool, metric, or trick if the name teaches a real mechanism to a new reader; omit local slang, quips, or period color that are not explanatory on their own
- do not replace a specific case from the chunk notes with a broader cross-domain generalization unless the chunk notes already do that
- separate the author's strongest ideas from examples that merely decorate them
- do not infer importance from narrative vividness alone
- if there is a tension between elegant theory and messy reality, say so clearly

Chunk extractions:
{joined}
"""


def build_challenge_prompt(book: BookRecord, summary_markdown: str) -> str:
    return f"""You are a skeptical but fair reviewer of a book summary.

Book:
- title: {book.title}
- author: {book.author or "unknown"}

Review the summary below and produce Markdown with these sections:

1. Likely Overclaims
2. Missing Counterarguments
3. Missing Late-Book Methods Or Tactical Material
4. Places Where Examples May Be Doing Too Much Work
5. Claims That Need External Verification
6. What A Careful Reader Should Watch For In The Full Book

Summary to challenge:
{summary_markdown}
"""


def build_anki_prompt(
    title: str,
    highlights: list[str],
    *,
    card_format: str = "qa",
    max_cards: int = 20,
) -> str:
    highlights_blob = "\n".join(f"- {item}" for item in highlights)
    if card_format == "cloze":
        schema = '[{"text":"cloze text with {{c1::...}}","extra":"optional context","tags":["book","topic"]}]'
        guidance = "Prefer cloze cards for models, distinctions, and strategic lessons."
    else:
        schema = '[{"front":"question","back":"answer","tags":["book","topic"]}]'
        guidance = "Prefer question-answer cards that capture transferable ideas instead of trivia."

    return f"""Turn these highlights into Anki cards.

Book title: {title}
Maximum cards: {max_cards}
{guidance}
Avoid shallow factoids unless the highlight itself is obviously about a durable factual distinction.
Deduplicate aggressively.

Return JSON only matching this schema:
{schema}

Highlights:
{highlights_blob}
"""


def build_alignment_review_prompt(
    title: str,
    summary_text: str,
    highlights: list[str],
) -> str:
    highlights_blob = "\n".join(f"- {item}" for item in highlights)
    return f"""Compare a generated book summary against the reader's highlights.

Book title: {title}

Write Markdown with these sections:

1. Executive Summary
2. High-Signal Ideas Present In Both
3. Important Highlight Themes Missing From The Summary
4. Places Where The Summary Overweights Examples Versus Models
5. Detail Sufficiency Of The Summary's Examples
6. Prompt Fixes To Try Next
7. Whether This Summary Would Actually Prepare Someone To Read The Book

Reader highlights:
{highlights_blob}

Generated summary:
{summary_text}
"""


def build_color_alignment_review_prompt(
    *,
    title: str,
    summary_text: str,
    chunk_density_text: str,
    yellow_chunk_review_text: str,
    final_signal_text: str,
    green_signal_text: str,
) -> str:
    return f"""Compare a generated book summary against the reader's highlights.

Book title: {title}

The reader's highlight colors mean:
- Yellow: general notes and interesting passages
- Red: important facts or claims
- Green: how the author or characters think
- Blue: highest-value ideas to apply in the reader's own life

Write Markdown with these sections:

1. Executive Summary
2. Overall Fidelity To The Reader's Takeaways
3. Yellow Notes Vs Chunk Extractions
4. Red And Blue Signal Vs Final Summary
5. Green Mindset Coverage
6. Detail Sufficiency Of The Summary's Examples
7. Missing Or Distorted High-Signal Material
8. Prompt Fixes To Try Next
9. Confidence And Caveats

Rules:
- Compare yellow highlights primarily against the estimated chunk extraction they map to.
- Compare red and blue highlights primarily against the final summary, not the chunk notes.
- Treat red highlights as the strongest signal for factual fidelity.
- Treat blue highlights as useful but more idiosyncratic evidence about personal application; do not let blue dominate the judgment.
- Evaluate green highlights as evidence of whether the summary preserves the author's or characters' way of seeing.
- The page numbers and progress percentages are approximate. Use them as soft evidence, not exact proof.
- If a chunk has very few highlights, absence of an idea there is weak negative evidence, not strong evidence of failure.
- Do not assume the highlights are exhaustive.
- Check whether the summary's examples are understandable to a reader who has not read the book. If an example lacks setup, mechanism, or payoff, call that out explicitly.
- Prefer factual grounding and concrete mechanism over elegant gloss.

Chunk highlight density:
{chunk_density_text}

Yellow highlights grouped by estimated chunk, alongside each chunk extraction:
{yellow_chunk_review_text}

Red and blue highlights to compare against the final summary:
{final_signal_text}

Green highlights to compare against the final summary:
{green_signal_text}

Generated final summary:
{summary_text}
"""
