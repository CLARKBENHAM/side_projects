from __future__ import annotations

import json
from typing import Any

from .models import BookRecord


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _clean_text(item))]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _parse_chunk_output(raw_output: str) -> dict[str, Any] | None:
    candidates = [raw_output.strip()]
    stripped = raw_output.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _format_bullets(heading: str, items: list[str]) -> list[str]:
    if not items:
        return []
    lines = [f"### {heading}", ""]
    lines.extend(f"- {item}" for item in items)
    lines.append("")
    return lines


def _join_parts(parts: list[str]) -> str:
    cleaned = [part for part in parts if part]
    return " ".join(cleaned)


def _format_models(models: list[dict[str, Any]]) -> list[str]:
    if not models:
        return []
    lines = ["### Core Models", ""]
    for model in models:
        name = _clean_text(model.get("name")) or "Model"
        summary = _clean_text(model.get("summary"))
        why_it_matters = _clean_text(model.get("why_it_matters"))
        body = _join_parts(
            [
                summary,
                f"Why it matters: {why_it_matters}" if why_it_matters else "",
            ]
        )
        lines.append(f"- **{name}:** {body}" if body else f"- **{name}**")
    lines.append("")
    return lines


def _format_named_tools(tools: list[dict[str, Any]]) -> list[str]:
    if not tools:
        return []
    lines = ["### Named Tools And Metrics", ""]
    for tool in tools:
        name = _clean_text(tool.get("name")) or "Tool"
        what_it_is = _clean_text(tool.get("what_it_is"))
        when_to_use = _clean_text(tool.get("when_to_use"))
        body = _join_parts(
            [
                what_it_is,
                f"When to use: {when_to_use}" if when_to_use else "",
            ]
        )
        lines.append(f"- **{name}:** {body}" if body else f"- **{name}**")
    lines.append("")
    return lines


def _format_examples(
    heading: str,
    examples: list[dict[str, Any]],
    *,
    include_supports: bool,
) -> list[str]:
    if not examples:
        return []
    lines = [f"### {heading}", ""]
    for example in examples:
        label = _clean_text(example.get("label")) or "Example"
        supports = _clean_text(example.get("supports"))
        setup = _clean_text(example.get("setup"))
        mechanism = _clean_text(example.get("mechanism"))
        payoff = _clean_text(example.get("payoff"))
        why_it_matters = _clean_text(example.get("why_it_matters"))
        body = _join_parts(
            [
                f"Supports: {supports}" if include_supports and supports else "",
                f"Setup: {setup}" if setup else "",
                f"Mechanism: {mechanism}" if mechanism else "",
                f"Payoff: {payoff}" if payoff else "",
                f"Why it matters: {why_it_matters}" if why_it_matters else "",
            ]
        )
        lines.append(f"- **{label}:** {body}" if body else f"- **{label}**")
    lines.append("")
    return lines


def _format_checkable_claims(claims: list[dict[str, Any]]) -> list[str]:
    if not claims:
        return []
    lines = ["### Checkable Claims", ""]
    for claim in claims:
        claim_text = _clean_text(claim.get("claim")) or "Claim"
        why_check = _clean_text(claim.get("why_check"))
        body = f" Why check: {why_check}" if why_check else ""
        lines.append(f"- **{claim_text}.**{body}")
    lines.append("")
    return lines


def _format_chunk_output(chunk_index: int, raw_output: str) -> list[str]:
    lines = [f"## Chunk {chunk_index}", ""]
    parsed = _parse_chunk_output(raw_output)
    if parsed is None:
        lines.extend(["### Raw Chunk Output", "", raw_output.strip(), ""])
        return lines

    lines.extend(_format_models(_dict_list(parsed.get("core_models"))))
    lines.extend(
        _format_bullets(
            "Key Facts And Mechanisms",
            _string_list(parsed.get("key_facts_and_mechanisms")),
        )
    )
    lines.extend(
        _format_bullets(
            "Legibility Gains",
            _string_list(parsed.get("legibility_gains")),
        )
    )
    lines.extend(
        _format_bullets(
            "Reasoning Methods",
            _string_list(parsed.get("reasoning_methods")),
        )
    )
    lines.extend(
        _format_bullets(
            "Practical Transfers",
            _string_list(parsed.get("practical_transfers")),
        )
    )
    lines.extend(_format_named_tools(_dict_list(parsed.get("named_tools_and_metrics"))))
    lines.extend(
        _format_examples(
            "Best Examples",
            _dict_list(parsed.get("best_examples")),
            include_supports=True,
        )
    )
    lines.extend(
        _format_examples(
            "Tactical Wins",
            _dict_list(parsed.get("tactical_wins")),
            include_supports=False,
        )
    )
    lines.extend(
        _format_bullets(
            "Non-Obvious Claims",
            _string_list(parsed.get("non_obvious_claims")),
        )
    )
    lines.extend(
        _format_bullets(
            "Pushback Points",
            _string_list(parsed.get("pushback_points")),
        )
    )
    lines.extend(_format_checkable_claims(_dict_list(parsed.get("checkable_claims"))))
    return lines


def render_expanded_chunk_notes(book: BookRecord, chunk_outputs: list[str]) -> str:
    lines = [
        f"# Expanded Chunk Notes: {book.title}",
        "",
        "This is a lightly reformatted companion to the chunk analyses.",
        "It preserves chunk order and most of the detail, while stripping the JSON wrapper.",
        "",
    ]
    for index, output in enumerate(chunk_outputs, start=1):
        lines.extend(_format_chunk_output(index, output))
    return "\n".join(lines).rstrip() + "\n"
