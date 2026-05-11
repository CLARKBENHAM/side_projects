from __future__ import annotations

from pathlib import Path

from structured_summaries.highlights import (
    load_structured_highlights,
    load_structured_highlights_from_paths,
    load_highlights_from_paths,
    parse_anki_cards,
    split_highlights,
)


def test_split_highlights_handles_bullets_and_blank_lines() -> None:
    text = """
    - Legibility beats local knowledge only on paper.

    > States like tidy abstractions.

    1. High modernism often destroys tacit knowledge.
    """
    highlights = split_highlights(text)

    assert len(highlights) == 3
    assert "Legibility beats local knowledge only on paper." in highlights
    assert "States like tidy abstractions." in highlights


def test_parse_anki_cards_reads_json_payload() -> None:
    cards = parse_anki_cards(
        '[{"front":"What is legibility?","back":"Making reality simpler for administration.","tags":["state"]}]'
    )

    assert cards == [
        {
            "front": "What is legibility?",
            "back": "Making reality simpler for administration.",
            "tags": "state",
        }
    ]


def test_split_highlights_strips_play_books_metadata() -> None:
    text = """
    Cover Image

    The Goal: A Process of Ongoing Improvement

    This document is overwritten when you make changes in Play Books.
    You should make a copy of this document before you edit it.

    Annotations by color
    2 yellow notes
    Created by Clark Benham   – Last synced October 12, 2024

    Yellow

    Why can’t we consistently get a quality product out the door on time?

    September 23, 2024
    33

    The real goal is making money, not keeping every machine busy.

    September 24, 2024
    44
    """

    highlights = split_highlights(text)

    assert highlights == [
        "Why can’t we consistently get a quality product out the door on time?",
        "The real goal is making money, not keeping every machine busy.",
    ]


def test_split_highlights_returns_empty_for_bookmark_only_export() -> None:
    text = """
    Cover Image

    Working Backwards.pdf

    6 bookmarks
    Created by Clark Benham   – Last synced August 13, 2022

    August 13, 2022
    1
    """

    assert split_highlights(text) == []


def test_load_highlights_from_paths_dedupes_across_sources(tmp_path: Path) -> None:
    path_one = tmp_path / "one.txt"
    path_two = tmp_path / "two.txt"
    path_one.write_text("- The bottleneck governs throughput.\n", encoding="utf-8")
    path_two.write_text(
        "- The bottleneck governs throughput.\n- Inventory is not the goal.\n",
        encoding="utf-8",
    )

    highlights = load_highlights_from_paths([path_one, path_two])

    assert highlights == [
        "The bottleneck governs throughput.",
        "Inventory is not the goal.",
    ]


def test_load_structured_highlights_preserves_color_and_pages(tmp_path: Path) -> None:
    highlights_file = tmp_path / "goal_notes.txt"
    highlights_file.write_text(
        """
        Annotations by color
        1 yellow notes • 1 blue notes • 1 red notes

        Yellow

        The bottleneck governs the system.

        September 23, 2024
        44

        Blue

        Improvement should change how I manage my own constraints.

        September 24, 2024
        88

        Red

        Throughput matters more than local efficiency.

        September 25, 2024
        120

        All your annotations

        The bottleneck governs the system.

        September 23, 2024
        44
        """,
        encoding="utf-8",
    )

    entries = load_structured_highlights(highlights_file, total_chunks=3)

    assert [entry.color for entry in entries] == ["yellow", "blue", "red"]
    assert [entry.page for entry in entries] == [44, 88, 120]
    assert [entry.estimated_chunk for entry in entries] == [2, 3, 3]


def test_load_structured_highlights_from_paths_dedupes_and_keeps_stronger_color(
    tmp_path: Path,
) -> None:
    path_one = tmp_path / "one.txt"
    path_two = tmp_path / "two.txt"
    path_one.write_text(
        """
        Annotations by color

        Yellow

        The bottleneck governs the system.

        September 23, 2024
        44
        """,
        encoding="utf-8",
    )
    path_two.write_text(
        """
        Annotations by color

        Blue

        The bottleneck governs the system.

        September 24, 2024
        45
        """,
        encoding="utf-8",
    )

    entries = load_structured_highlights_from_paths(
        [path_one, path_two], total_chunks=2
    )

    assert len(entries) == 1
    assert entries[0].color == "blue"
    assert entries[0].duplicate_count == 2
