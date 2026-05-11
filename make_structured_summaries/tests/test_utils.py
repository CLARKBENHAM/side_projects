from structured_summaries.utils import (
    clean_title_for_search,
    infer_author_from_filename,
)


def test_clean_title_for_search_preserves_hyphenated_title_segments() -> None:
    assert (
        clean_title_for_search(
            "Designing Data-Intensive Applications_ The Big Ide",
            "Intensive Applications_ The Big",
        )
        == "Designing Data-Intensive Applications The Big Ide"
    )
    assert clean_title_for_search("The-Day-of-the-Jackal") == "The-Day-of-the-Jackal"


def test_infer_author_from_filename_ignores_same_title_files() -> None:
    assert infer_author_from_filename("The Price of Victory.epub") == ""
    assert infer_author_from_filename("Sadly, Porn.epub") == ""
    assert (
        infer_author_from_filename("Mark Manson - Models (Revised and Updated).pdf")
        == "Mark Manson"
    )
