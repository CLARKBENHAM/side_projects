from __future__ import annotations

import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "Self_Tracking"
        / "play_books_extract_highlights3.py"
    )
    spec = importlib.util.spec_from_file_location(
        "play_books_extract_highlights3", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_title_is_selected_honors_default_targets_and_extra_titles() -> None:
    args = Namespace(include_title=["Nixon Agonistes"], no_title_filter=False)

    assert MODULE.title_is_selected("Master of the Senate", args)
    assert MODULE.title_is_selected("Nixon Agonistes", args)
    assert not MODULE.title_is_selected("Some Other Book", args)


def test_title_is_selected_can_disable_filter() -> None:
    args = Namespace(include_title=[], no_title_filter=True)

    assert MODULE.title_is_selected("Some Other Book", args)


def test_title_is_always_skipped_for_known_audiobook() -> None:
    assert MODULE.title_is_always_skipped("The Abolition of Man and The Great Divorce")
    assert not MODULE.title_is_always_skipped("The Great Divorce")


def test_only_title_overrides_default_targets() -> None:
    args = Namespace(
        include_title=[],
        only_title=["The Autobiography of Benjamin Franklin"],
        no_title_filter=False,
    )

    assert MODULE.title_is_selected("The Autobiography of Benjamin Franklin", args)
    assert not MODULE.title_is_selected("Churchill: Walking with Destiny", args)


def test_find_library_book_bounds_prefers_exact_match() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node
          text="Master of the Senate"
          resource-id="com.google.android.apps.books:id/title"
          class="android.widget.TextView"
          content-desc=""
          bounds="[10,500][300,600]"
        />
        <node
          text="Master"
          resource-id="com.google.android.apps.books:id/title"
          class="android.widget.TextView"
          content-desc=""
          bounds="[10,700][300,800]"
        />
      </node>
    </hierarchy>
    """

    bounds = MODULE.find_library_book_bounds(xml, 2000, "Master of the Senate")

    assert bounds == (10, 500, 300, 600)


def test_find_library_books_uses_download_status_rows_for_list_view() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node text="Recent" resource-id="" class="android.widget.TextView" content-desc="Recent, Sort filter" bounds="[120,240][300,280]" />
        <node text="The Real Book Title" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,500][900,534]" />
        <node text="Real Author" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,543][900,570]" />
        <node text="43% complete" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,579][900,606]" />
        <node text="" resource-id="" class="android.view.View" content-desc="Downloaded" bounds="[1098,500][1170,572]" />
      </node>
    </hierarchy>
    """

    books = MODULE.find_library_books(xml, 2000)

    assert books == [("The Real Book Title", (132, 500, 900, 534))]


def test_find_library_books_uses_partial_download_status_rows() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node text="Partially Downloaded Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,500][900,534]" />
        <node text="" resource-id="" class="android.view.View" content-desc="3% downloaded" bounds="[1056,500][1128,572]" />
      </node>
    </hierarchy>
    """

    books = MODULE.find_library_books(xml, 2000)

    assert books == [("Partially Downloaded Book", (132, 500, 900, 534))]


def test_find_library_books_uses_download_icon_rows() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node text="Cloud Only Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,500][900,534]" />
        <node text="" resource-id="" class="android.view.View" content-desc="Download" bounds="[1056,500][1128,572]" />
      </node>
    </hierarchy>
    """

    books = MODULE.find_library_books(xml, 2000)

    assert books == [("Cloud Only Book", (132, 500, 900, 534))]


def test_find_library_books_ignores_bottom_audio_play_bar_title() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node text="Visible Library Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,500][900,534]" />
        <node text="" resource-id="" class="android.view.View" content-desc="Download" bounds="[1056,500][1128,572]" />
        <node text="The Abolition of Man and The Great Divorce" resource-id="com.google.android.apps.books:id/title" class="android.widget.TextView" content-desc="" bounds="[103,1751][1050,1792]" />
      </node>
    </hierarchy>
    """

    books = MODULE.find_library_books(xml, 2000)

    assert books == [("Visible Library Book", (132, 500, 900, 534))]


def test_is_notes_list_screen_accepts_empty_notes_tab() -> None:
    xml = """
    <hierarchy>
      <node text="Chapters" resource-id="" class="android.widget.TextView" content-desc="" bounds="[70,147][170,188]" />
      <node text="Bookmarks" resource-id="" class="android.widget.TextView" content-desc="" bounds="[299,147][420,188]" />
      <node text="Notes" resource-id="" class="android.widget.TextView" content-desc="" bounds="[567,147][632,188]" />
      <node text="Add your own thoughts" resource-id="" class="android.widget.TextView" content-desc="" bounds="[72,1031][1128,1083]" />
      <node text="Touch &amp; hold the text to take some notes about what you're reading. We'll save them all right here." resource-id="" class="android.widget.TextView" content-desc="" bounds="[72,1113][1128,1154]" />
    </hierarchy>
    """

    assert MODULE.is_notes_list_screen(xml)


def test_find_library_books_uses_full_book_card_content_desc() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node
          text=""
          resource-id="com.google.android.apps.books:id/book_card_root"
          class="android.view.ViewGroup"
          content-desc="Computer Systems A Programmer's Perspective, Third Edition Instructor's Solution Manual.pdf, 6% complete, 10% downloaded"
          bounds="[386,135][685,593]"
        />
        <node
          text="Computer Systems A Pro..."
          resource-id="com.google.android.apps.books:id/multiline_description_widget_description_line"
          class="android.widget.TextView"
          content-desc=""
          bounds="[389,517][682,558]"
        />
      </node>
    </hierarchy>
    """

    books = MODULE.find_library_books(xml, 2000)

    assert books == [
        (
            "Computer Systems A Programmer's Perspective, Third Edition "
            "Instructor's Solution Manual.pdf",
            (386, 135, 685, 593),
        )
    ]


def test_progress_percent_for_title_uses_full_book_card_content_desc() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node
          text=""
          resource-id="com.google.android.apps.books:id/book_card_root"
          class="android.view.ViewGroup"
          content-desc="Partially Read Book, 12% complete, 4% downloaded"
          bounds="[386,135][685,593]"
        />
      </node>
    </hierarchy>
    """

    assert MODULE.progress_percent_for_title(xml, 2000, "Partially Read Book") == 12
    assert MODULE.title_meets_progress_filter(xml, 2000, "Partially Read Book", 10)
    assert not MODULE.title_meets_progress_filter(xml, 2000, "Partially Read Book", 20)


def test_progress_percent_for_title_uses_list_view_progress_row() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node text="List View Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,500][900,534]" />
        <node text="11% complete" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,579][900,606]" />
        <node text="" resource-id="" class="android.view.View" content-desc="Download" bounds="[1056,500][1128,572]" />
      </node>
    </hierarchy>
    """

    assert MODULE.progress_percent_for_title(xml, 2000, "List View Book") == 11


def test_detect_download_state_from_library_row() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node text="Local Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,500][900,534]" />
        <node text="" resource-id="" class="android.view.View" content-desc="Downloaded" bounds="[1056,500][1128,572]" />
        <node text="Cloud Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,650][900,684]" />
        <node text="" resource-id="" class="android.view.View" content-desc="Download" bounds="[1056,650][1128,722]" />
        <node text="Partial Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,800][900,834]" />
        <node text="" resource-id="" class="android.view.View" content-desc="3% downloaded" bounds="[1056,800][1128,872]" />
      </node>
    </hierarchy>
    """

    assert MODULE.detect_download_state_from_library_row(xml, 2000, "Local Book")
    assert not MODULE.detect_download_state_from_library_row(xml, 2000, "Cloud Book")
    assert not MODULE.detect_download_state_from_library_row(xml, 2000, "Partial Book")
    assert MODULE.download_status_desc_for_title(xml, 2000, "Cloud Book") == "Download"
    assert MODULE.is_plain_not_downloaded_status("Not downloaded")
    assert not MODULE.is_plain_not_downloaded_status("3% downloaded")
    assert MODULE.is_local_or_partial_download_status("Downloaded")
    assert MODULE.is_local_or_partial_download_status("3% downloaded")
    assert not MODULE.is_local_or_partial_download_status("Download")


def test_maybe_remove_download_treats_plain_download_as_cleaned(
    monkeypatch, capsys
) -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/library_root" class="" content-desc="" bounds="[0,0][1200,2000]">
        <node text="Cloud Book" resource-id="" class="android.widget.TextView" content-desc="" bounds="[132,500][900,534]" />
        <node text="" resource-id="" class="android.view.View" content-desc="Download" bounds="[1056,500][1128,572]" />
      </node>
    </hierarchy>
    """

    monkeypatch.setattr(MODULE, "ensure_library_screen", lambda _width, _height: xml)
    monkeypatch.setattr(
        MODULE,
        "tap_bounds",
        lambda _bounds: (_ for _ in ()).throw(AssertionError("should not tap")),
    )

    assert MODULE.maybe_remove_download("Cloud Book", 1200, 2000)
    assert "No local download left for 'Cloud Book'." in capsys.readouterr().out


def test_find_library_books_ignores_notes_list_titles() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="com.google.android.apps.books:id/notes_list_view" class="android.widget.ListView" content-desc="" bounds="[0,204][1200,1928]">
        <node text="Chapter Title" resource-id="com.google.android.apps.books:id/title" class="android.widget.TextView" content-desc="" bounds="[132,204][1068,206]" />
        <node text="A highlighted passage" resource-id="com.google.android.apps.books:id/snippet" class="android.widget.TextView" content-desc="" bounds="[132,265][1128,352]" />
      </node>
    </hierarchy>
    """

    assert MODULE.find_library_books(xml, 2000) == []


def test_library_filters_visible_detects_filter_bar() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="" class="android.view.View" content-desc="Library filters" bounds="[0,192][1200,309]" />
      <node text="" resource-id="" class="android.view.View" content-desc="Recent, Sort filter" bounds="[123,234][302,282]" />
    </hierarchy>
    """

    assert MODULE.library_filters_visible(xml)


def test_find_bottom_nav_tab_uses_bottom_library_label() -> None:
    xml = """
    <hierarchy>
      <node text="Library" resource-id="" class="android.widget.TextView" content-desc="" bounds="[24,144][160,196]" />
      <node text="Library" resource-id="" class="android.widget.TextView" content-desc="" bounds="[924,1848][1076,1904]" />
    </hierarchy>
    """

    node = MODULE.find_bottom_nav_tab(xml, 2000, "Library")

    assert node is not None
    assert node.bounds == (924, 1848, 1076, 1904)


def test_parse_data_free_bytes_from_df() -> None:
    output = """
Filesystem       1K-blocks     Used Available Use% Mounted on
/dev/block/dm-50  51412860 38946996  12334792  76% /data/user/0
"""

    assert MODULE.parse_data_free_bytes_from_df(output) == 12334792 * 1024


def test_format_storage_delta() -> None:
    gib = 1024**3

    assert MODULE.format_storage_delta(10 * gib, 12 * gib) == "+2.0 GiB"
    assert MODULE.format_storage_delta(12 * gib, 10 * gib) == "-2.0 GiB"


def test_infer_download_state_prefers_remove_download_over_download() -> None:
    xml = """
    <hierarchy>
      <node text="Remove download" resource-id="" class="android.widget.TextView" content-desc="" bounds="[0,0][1,1]" />
      <node text="Download" resource-id="" class="android.widget.TextView" content-desc="" bounds="[0,2][1,3]" />
    </hierarchy>
    """

    assert MODULE.infer_download_state_from_menu_xml(xml) is True


def test_infer_download_state_detects_not_downloaded() -> None:
    xml = """
    <hierarchy>
      <node text="Download" resource-id="" class="android.widget.TextView" content-desc="" bounds="[0,0][1,1]" />
    </hierarchy>
    """

    assert MODULE.infer_download_state_from_menu_xml(xml) is False


def test_infer_download_state_returns_none_without_signal() -> None:
    xml = """
    <hierarchy>
      <node text="Add to shelf" resource-id="" class="android.widget.TextView" content-desc="" bounds="[0,0][1,1]" />
    </hierarchy>
    """

    assert MODULE.infer_download_state_from_menu_xml(xml) is None


def test_output_paths_avoid_existing_files(tmp_path: Path) -> None:
    first_md, first_jsonl = MODULE.output_paths("Duplicate Title", str(tmp_path))
    Path(first_md).touch()
    Path(first_jsonl).touch()

    second_md, second_jsonl = MODULE.output_paths("Duplicate Title", str(tmp_path))

    assert second_md.endswith("Duplicate Title (2).md")
    assert second_jsonl.endswith("Duplicate Title (2).jsonl")


def test_load_titles_from_run_log(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Download state before export for 'Book One': not downloaded.",
                "Skipping 'Book One': could not open notes list.",
                'Skipping "Book Two": could not open notes list.',
                "No highlights found for 'Book Three'.",
                "Download state before export for 'Interrupted': downloaded.",
                "Download state before export for 'Written': downloaded.",
                "Wrote: out/Written.jsonl",
            ]
        ),
        encoding="utf-8",
    )

    titles = MODULE.load_titles_from_run_log(str(log_path))

    assert titles == {"book one", "book two", "book three", "written"}


def test_ensure_notes_list_returns_none_after_uiautomator_timeout(monkeypatch) -> None:
    def raise_timeout() -> str:
        raise subprocess.TimeoutExpired(["adb"], 60)

    monkeypatch.setattr(MODULE, "uiautomator_dump_xml", raise_timeout)

    assert MODULE.ensure_notes_list(1200, 2000, max_tries=1) is None


def test_library_scroll_is_shorter_than_notes_scroll(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_adb(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(["adb", *args], 0, "", "")

    monkeypatch.setattr(MODULE, "adb", fake_adb)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    MODULE.swipe_library_scroll(1200, 2000)
    MODULE.swipe_notes_scroll(1200, 2000)

    assert calls[0] == ["shell", "input", "swipe", "600", "1440", "600", "1000", "350"]
    assert calls[1] == ["shell", "input", "swipe", "600", "1560", "600", "440", "350"]


def test_run_all_books_resume_from_current_position_skips_top_reset(
    tmp_path: Path, monkeypatch
) -> None:
    args = Namespace(
        out=str(tmp_path),
        min_free_gb=0,
        skip_titles_from_log=[],
        resume_from_current_position=True,
        download_status="current",
        library_sort="title",
        max_swipes=0,
    )

    monkeypatch.setattr(MODULE, "get_data_free_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(MODULE, "open_library_tab", lambda _width, _height: "<xml />")
    monkeypatch.setattr(MODULE, "ensure_your_books_tab", lambda xml: xml)
    monkeypatch.setattr(
        MODULE,
        "scroll_library_to_top",
        lambda _width, _height: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        MODULE,
        "ensure_download_status_filter",
        lambda _xml, _status: (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        MODULE,
        "ensure_library_sort",
        lambda _xml, _sort: (_ for _ in ()).throw(AssertionError()),
    )

    MODULE.run_all_books(args, 1200, 2000)


def test_load_unremoved_new_download_titles_from_run_log(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Download state before export for 'Left Behind': not downloaded.",
                "Skipping 'Left Behind': could not open notes list.",
                "Download state before export for 'Cleaned': not downloaded.",
                "Removed local download for 'Cleaned'.",
                "Download state before export for 'Already Local': downloaded.",
            ]
        ),
        encoding="utf-8",
    )

    titles = MODULE.load_unremoved_new_download_titles_from_run_log(str(log_path))

    assert titles == ["Left Behind"]


def test_load_unremoved_new_download_titles_treats_redownload_as_outstanding(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Download state before export for 'Retried': not downloaded.",
                "Removed local download for 'Retried'.",
                "Download state before export for 'Retried': not downloaded.",
                "WARN: could not return to Library to remove download for 'Retried'.",
            ]
        ),
        encoding="utf-8",
    )

    titles = MODULE.load_unremoved_new_download_titles_from_run_log(str(log_path))

    assert titles == ["Retried"]


def test_title_keys_match_handles_encoding_damage_by_prefix() -> None:
    assert MODULE.title_keys_match(
        "Combinatorial problems and exercises-L. Lovász -North Holland.pdf",
        "Combinatorial problems and exercises-L. LovÃ¡sz -North Holland.pdf",
    )


def test_title_keys_match_rejects_short_substring_collision() -> None:
    assert not MODULE.title_keys_match(
        "Softwar",
        "99 Bottles of OOP: A Practical Guide to Object-Oriented Design "
        "(JavaScript Edition) - Sandi Metz, Katrina Owen, TJ Stankus - "
        "(2020, Potato Canyon Software).pdf",
    )


def test_matching_title_key_accepts_unique_visible_prefix() -> None:
    targets = {
        MODULE.book_key("Blitzscaling: The Lightning-Fast Path"): (
            "Blitzscaling: The Lightning-Fast Path"
        ),
        MODULE.book_key("Other Book"): "Other Book",
    }

    assert MODULE.matching_title_key("Blitzscaling", targets) == MODULE.book_key(
        "Blitzscaling: The Lightning-Fast Path"
    )


def test_matching_title_key_rejects_ambiguous_visible_prefix() -> None:
    targets = {
        MODULE.book_key("Advanced JavaScript"): "Advanced JavaScript",
        MODULE.book_key("Advanced Linear Algebra"): "Advanced Linear Algebra",
    }

    assert MODULE.matching_title_key("Advanced", targets) is None


def test_matching_cleanup_title_key_rejects_short_series_prefix() -> None:
    target = "The World Crisis, Vol. 4 (Winston Churchill's World Crisis Collection)"
    targets = {MODULE.book_key(target): target}

    assert MODULE.matching_cleanup_title_key("The world crisis", targets) is None


def test_matching_cleanup_title_key_accepts_long_truncated_title() -> None:
    target = "The World Crisis, Vol. 4 (Winston Churchill's World Crisis Collection)"
    targets = {MODULE.book_key(target): target}

    assert MODULE.matching_cleanup_title_key(
        "The World Crisis, Vol. 4 (Winston Churchill", targets
    ) == MODULE.book_key(target)
