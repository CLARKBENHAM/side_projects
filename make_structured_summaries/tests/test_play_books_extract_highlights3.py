from __future__ import annotations

import importlib.util
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


def test_find_library_book_bounds_prefers_exact_match() -> None:
    xml = """
    <hierarchy>
      <node text="" resource-id="" class="" content-desc="" bounds="[0,0][1200,2000]">
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
