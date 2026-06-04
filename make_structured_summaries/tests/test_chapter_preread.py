from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from structured_summaries.chapter_preread import (
    ChapterSpan,
    chapter_title_from_text,
    highlight_probes,
    locate_highlight_chapter,
    page_progress_chapter,
    place_highlights_by_chapter,
    render_highlights_markdown,
    split_long_chapters,
)
from structured_summaries.models import BookRecord
from structured_summaries.prompts import (
    build_chapter_highlight_judge_prompt,
    build_chapter_preread_summary_prompt,
)


@dataclass(frozen=True)
class FakeHighlight:
    text: str
    color: str = "yellow"
    page: int | None = None
    source_labels: tuple[str, ...] = ("test",)
    duplicate_count: int = 1


def sample_book() -> BookRecord:
    path = Path(__file__)
    return BookRecord(
        book_id="sample",
        title="Sample Book",
        author="Example Author",
        book_dir=path.parent,
        primary_path=path,
        primary_format="txt",
        companion_html_path=None,
        all_paths=(path,),
    )


def test_chapter_title_from_text_combines_epub_heading_lines() -> None:
    text = "\n".join(
        [
            "Sample Book",
            "ONE",
            "Directing the March of Other Nations",
            "Policy and Operations 1815-1840",
            "In the modern popular imagination, Britain was powerful.",
        ]
    )

    title = chapter_title_from_text(text, book_title="Sample Book", fallback="Chapter 1")

    assert title == (
        "ONE Directing the March of Other Nations Policy and Operations 1815-1840"
    )


def test_locate_highlight_chapter_uses_source_text_probe() -> None:
    chapters = [
        ChapterSpan(
            index=1,
            title="One",
            text="Alpha chapter about administration.",
            source_label="one.xhtml",
            start_char=0,
            end_char=35,
        ),
        ChapterSpan(
            index=2,
            title="Two",
            text=(
                "Johnson staged visible drama on the floor because the appearance "
                "of power made senators update their expectations."
            ),
            source_label="two.xhtml",
            start_char=36,
            end_char=150,
        ),
    ]
    highlight = (
        "Johnson staged visible drama on the floor because the appearance of power "
        "made senators update their expectations. Reader note appended later."
    )

    chapter_index, method = locate_highlight_chapter(highlight, chapters)

    assert chapter_index == 2
    assert method == "source text probe"
    assert highlight_probes(highlight)


def test_place_highlights_falls_back_to_page_progress() -> None:
    chapters = [
        ChapterSpan(1, "One", "alpha", "one", 0, 5),
        ChapterSpan(2, "Two", "beta", "two", 6, 10),
        ChapterSpan(3, "Three", "gamma", "three", 11, 16),
    ]
    highlights = [
        FakeHighlight("not present in source", page=50),
        FakeHighlight("also absent", page=120),
    ]

    placed = place_highlights_by_chapter(highlights, chapters)

    assert placed[2][0].method == "page progress"
    assert placed[3][0].method == "page progress"


def test_page_progress_handles_missing_pages() -> None:
    assert page_progress_chapter(None, max_page=100, chapter_count=3) is None
    assert page_progress_chapter(100, max_page=100, chapter_count=3) == 3


def test_chapter_prompts_keep_summary_blind_to_highlights() -> None:
    book = sample_book()
    summary_prompt = build_chapter_preread_summary_prompt(
        book,
        chapter_title="Chapter 1",
        chapter_index=1,
        total_chapters=3,
        chapter_text="The chapter text.",
    )
    judge_prompt = build_chapter_highlight_judge_prompt(
        book,
        chapter_title="Chapter 1",
        chapter_index=1,
        total_chapters=3,
        summary_markdown="A summary.",
        highlights_markdown="- [red] A highlight.",
    )

    assert "Do not mention reader highlights" in summary_prompt
    assert "Reader highlights from this chapter" not in summary_prompt
    assert "Reader highlights from this chapter" in judge_prompt
    assert "Main-point coverage" in judge_prompt


def test_render_highlights_markdown_includes_color_page_and_method() -> None:
    chapters = [ChapterSpan(1, "One", "alpha beta gamma", "one", 0, 16)]
    highlights = [FakeHighlight("alpha beta gamma", color="red", page=7)]

    placed = place_highlights_by_chapter(highlights, chapters)
    rendered = render_highlights_markdown(placed[1])

    assert "[red | p7 | source text | test]" in rendered


def test_split_long_chapters_preserves_order_and_titles() -> None:
    chapters = [
        ChapterSpan(
            1,
            "Long Chapter",
            "alpha\n\n" + ("body " * 30) + "\n\nomega",
            "source.xhtml",
            10,
            200,
        )
    ]

    split = split_long_chapters(chapters, max_chars=60)

    assert len(split) > 1
    assert [chapter.index for chapter in split] == list(range(1, len(split) + 1))
    assert split[0].title == "Long Chapter (part 1)"
    assert split[1].title == "Long Chapter (part 2)"
    assert split[0].start_char == 10
