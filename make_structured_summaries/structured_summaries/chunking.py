"""Chunking helpers for long book text."""

from __future__ import annotations

import re


def split_text_into_chunks(
    text: str,
    *,
    target_chars: int = 28_000,
    overlap_chars: int = 1_500,
    max_chunks: int | None = None,
) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    if not paragraphs:
        return [cleaned]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        addition_size = len(paragraph) + (2 if current else 0)
        if current and current_size + addition_size > target_chars:
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks
            overlap: list[str] = []
            overlap_size = 0
            for previous in reversed(current):
                paragraph_size = len(previous) + (2 if overlap else 0)
                if overlap and overlap_size + paragraph_size > overlap_chars:
                    break
                overlap.insert(0, previous)
                overlap_size += paragraph_size
            current = overlap
            current_size = len("\n\n".join(current))

        current.append(paragraph)
        current_size = len("\n\n".join(current))

    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks
