from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ai_books_tracking"
    / "scripts"
    / "temp"
    / "gemini_holdout_fewshot_eval.py"
)


def load_module():
    spec = spec_from_file_location("gemini_holdout_fewshot_eval", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_json_payload_handles_fenced_json():
    module = load_module()
    payload = module.extract_json_payload(
        """```json
{"pred_enjoyment": 3.25, "pred_usefulness": 2.0, "confidence": 0.6, "reasoning": "test"}
```"""
    )
    assert payload["pred_enjoyment"] == 3.25
    assert payload["pred_usefulness"] == 2.0


def test_clean_display_title_removes_noise():
    module = load_module()
    assert (
        module.clean_display_title(
            "[Hyperion 1] Dan Simmons - Hyperion-Saga 1_ Hyperion (1990) - libgen.li.pdf"
        )
        == "Dan Simmons - Hyperion-Saga 1_ Hyperion (1990)"
    )


def test_build_prompt_includes_profile_examples_and_target():
    module = load_module()
    target = pd.Series(
        {
            "display_title": "Target Book",
            "display_author": "A. Author",
            "display_category": "Business, management",
            "goodreads_rating_verified": 4.2,
            "open_library_rating": 4.0,
            "amazon_rating_consensus": 4.6,
            "page_count": 320,
            "pub_year": 2020,
        }
    )
    examples = pd.DataFrame(
        [
            {
                "display_title": "Example One",
                "display_author": "Writer",
                "display_category": "fiction",
                "goodreads_rating_verified": 4.0,
                "open_library_rating": 3.9,
                "amazon_rating_consensus": 4.4,
                "page_count": 250,
                "pub_year": 2015,
                "avg_enjoyment": 4.5,
                "avg_usefulness": 2.0,
                "long_term_effects": "Memorable style.",
            }
        ]
    )
    prompt = module.build_prompt(target, examples, "Reader profile here.")
    assert "Reader profile here." in prompt
    assert "1. title=Example One" in prompt
    assert "Target Book" in prompt
    assert '"pred_enjoyment"' in prompt


def test_build_prompt_can_include_calibration_block():
    module = load_module()
    target = pd.Series(
        {
            "display_title": "Target Book",
            "display_author": "A. Author",
            "display_category": "Business, management",
            "goodreads_rating_verified": 4.2,
            "open_library_rating": 4.0,
            "amazon_rating_consensus": 4.6,
            "page_count": 320,
            "pub_year": 2020,
        }
    )
    examples = pd.DataFrame(
        [
            {
                "display_title": "Example One",
                "display_author": "Writer",
                "display_category": "fiction",
                "goodreads_rating_verified": 4.0,
                "open_library_rating": 3.9,
                "amazon_rating_consensus": 4.4,
                "page_count": 250,
                "pub_year": 2015,
                "avg_enjoyment": 4.5,
                "avg_usefulness": 2.0,
                "long_term_effects": "Memorable style.",
            }
        ]
    )
    prompt = module.build_prompt(
        target,
        examples,
        "Reader profile here.",
        "Calibration notes from train-only books:\n- Overall mean usefulness: 2.10",
    )
    assert "Calibration notes from train-only books" in prompt
    assert "Overall mean usefulness: 2.10" in prompt


def test_build_prompt_can_include_extra_profile():
    module = load_module()
    target = pd.Series(
        {
            "display_title": "Target Book",
            "display_author": "A. Author",
            "display_category": "Business, management",
            "goodreads_rating_verified": 4.2,
            "open_library_rating": 4.0,
            "amazon_rating_consensus": 4.6,
            "page_count": 320,
            "pub_year": 2020,
        }
    )
    examples = pd.DataFrame(
        [
            {
                "display_title": "Example One",
                "display_author": "Writer",
                "display_category": "fiction",
                "goodreads_rating_verified": 4.0,
                "open_library_rating": 3.9,
                "amazon_rating_consensus": 4.4,
                "page_count": 250,
                "pub_year": 2015,
                "avg_enjoyment": 4.5,
                "avg_usefulness": 2.0,
                "long_term_effects": "Memorable style.",
            }
        ]
    )
    prompt = module.build_prompt(
        target,
        examples,
        "Reader profile here.",
        "Calibration notes here.",
        "Extra taste profile here.",
    )
    assert "Extra taste profile here." in prompt


def test_sample_extreme_examples_dedupes_and_keeps_extremes():
    module = load_module()
    train = pd.DataFrame(
        [
            {
                "key": "a",
                "display_title": "Top Both",
                "avg_enjoyment": 5.0,
                "avg_usefulness": 5.0,
            },
            {
                "key": "b",
                "display_title": "Top Enjoy",
                "avg_enjoyment": 4.75,
                "avg_usefulness": 3.0,
            },
            {
                "key": "c",
                "display_title": "Top Useful",
                "avg_enjoyment": 3.0,
                "avg_usefulness": 4.75,
            },
            {
                "key": "d",
                "display_title": "Low Enjoy",
                "avg_enjoyment": 1.0,
                "avg_usefulness": 2.0,
            },
            {
                "key": "e",
                "display_title": "Low Useful",
                "avg_enjoyment": 2.0,
                "avg_usefulness": 1.0,
            },
        ]
    )
    sampled = module.sample_extreme_examples(train, bucket_size=2)
    assert sampled["key"].is_unique
    assert "a" in sampled["key"].tolist()
    assert "d" in sampled["key"].tolist()
    assert "e" in sampled["key"].tolist()
