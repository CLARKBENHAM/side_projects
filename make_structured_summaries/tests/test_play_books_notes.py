from __future__ import annotations

from pathlib import Path

from structured_summaries.play_books_notes import (
    merge_blue_highlights_sections,
    parse_blue_highlights_file,
    parse_blue_highlights_text,
    render_blue_highlights_markdown,
)


def test_parse_blue_highlights_text_extracts_title_author_and_entries() -> None:
    text = """\ufeffCover Image

My Early Life
Churchill, Winston S.
Project Gutenberg Canada

Annotations by color
133 yellow notes • 24 green notes • 19 blue notes • 12 red notes
Created by Clark Benham   – Last synced March 29, 2026

Yellow

something else

Blue

I found I could add nearly two hours to my working effort by going to bed for an hour after luncheon.

This should be applied to my own work blocks.

March 29, 2026
325

Continuity of work never harmed anyone.

March 29, 2026
326

Red

not part of blue
"""
    section = parse_blue_highlights_text(text, fallback_name="fallback")

    assert section is not None
    assert section.title == "My Early Life"
    assert section.author == "Churchill, Winston S."
    assert section.highlights == (
        "I found I could add nearly two hours to my working effort by going to bed for an hour after luncheon. This should be applied to my own work blocks.",
        "Continuity of work never harmed anyone.",
    )


def test_parse_blue_highlights_file_uses_existing_export_format(tmp_path: Path) -> None:
    note_path = tmp_path / 'Notes from "Example Book".txt'
    note_path.write_text(
        """Example Book
Example Author
Example Publisher

Annotations by color
1 yellow notes • 1 green notes • 1 blue notes • 0 red notes

Yellow

ignore me

Blue

First highlight.

May 5, 2025
44

Second highlight.

May 5, 2025
45
""",
        encoding="utf-8",
    )

    section = parse_blue_highlights_file(note_path)

    assert section is not None
    assert section.title == "Example Book"
    assert section.author == "Example Author"
    assert section.highlights == (
        "First highlight.",
        "Second highlight.",
    )


def test_render_blue_highlights_markdown_groups_title_author_and_highlights() -> None:
    section = parse_blue_highlights_text(
        """Example
Author

Blue

Only highlight.

May 1, 2025
1
""",
        fallback_name="Example",
    )
    assert section is not None

    rendered = render_blue_highlights_markdown([section])

    assert "# Compiled Blue Highlights" in rendered
    assert "## Example" in rendered
    assert "### Author" in rendered
    assert "Author" in rendered
    assert "### Blue Highlights" in rendered
    assert "- Only highlight." in rendered


def test_merge_blue_highlights_sections_combines_duplicate_titles() -> None:
    first = parse_blue_highlights_text(
        """Example
Author

Blue

First highlight.

May 1, 2025
1
""",
        fallback_name="Example",
    )
    second = parse_blue_highlights_text(
        """Example
Author

Blue

First highlight.

May 1, 2025
1

Second highlight.

May 2, 2025
2
""",
        fallback_name="Example",
    )

    assert first is not None
    assert second is not None

    merged = merge_blue_highlights_sections([first, second])

    assert merged == [
        first.__class__(
            title="Example",
            author="Author",
            highlights=("First highlight.", "Second highlight."),
            source_path=None,
        )
    ]
