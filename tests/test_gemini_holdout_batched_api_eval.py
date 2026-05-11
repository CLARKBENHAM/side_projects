from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ai_books_tracking.scripts.temp.gemini_holdout_batched_api_eval import (
    CallSpec,
    build_call_specs,
    combine_target_prompts,
    configure_experiment,
    load_prompt_override,
    normalize_batch_predictions,
    output_file,
    round_half,
    sample_count_curves,
)


def make_holdout(n_books: int = 68) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_books + 1):
        rows.append(
            {
                "book_id": f"B{idx:03d}",
                "key": f"book {idx} || author {idx}",
                "display_title": f"Book {idx}",
                "display_author": f"Author {idx}",
                "display_category": "fiction" if idx % 2 else "General Reading",
                "avg_enjoyment": 1.0 + (idx % 5),
                "avg_usefulness": 1.0 + ((idx + 1) % 4),
                "goodreads_rating_verified": 4.0,
                "open_library_rating": 4.1,
                "amazon_rating_consensus": 4.2,
            }
        )
    return pd.DataFrame(rows)


def test_build_call_specs_covers_every_book_once_per_run() -> None:
    holdout = make_holdout()
    specs = build_call_specs(holdout, batch_size=17, runs_per_prompt=5, seed=7)

    assert len(specs) == 2 * 5 * 4

    plan = pd.DataFrame(
        {
            "prompt_id": [spec.prompt_id for spec in specs],
            "replicate": [spec.replicate for spec in specs],
            "batch_size": [len(spec.batch_books) for spec in specs],
        }
    )
    assert set(plan["batch_size"]) == {17}

    for (prompt_id, replicate), run_specs in plan.groupby(["prompt_id", "replicate"]):
        del prompt_id, replicate
        book_ids = [
            book_id
            for spec in specs
            if spec.prompt_id == run_specs.iloc[0]["prompt_id"]
            and spec.replicate == run_specs.iloc[0]["replicate"]
            for book_id in spec.batch_books
        ]
        assert len(book_ids) == 68
        assert len(set(book_ids)) == 68


def test_normalize_batch_predictions_rounds_and_validates() -> None:
    batch = make_holdout(2).copy()
    spec = CallSpec(
        prompt_id=1, replicate=1, batch_index=1, batch_books=("B001", "B002")
    )
    records = [
        {
            "book_id": "B002",
            "brief_reason": "Second book",
            "confidence": "high",
            "predicted_enjoyment": 5.2,
            "predicted_usefulness": 0.7,
        },
        {
            "book_id": "B001",
            "brief_reason": "First book",
            "confidence": "low",
            "predicted_enjoyment": 3.26,
            "predicted_usefulness": 2.74,
        },
    ]

    normalized = normalize_batch_predictions(
        records=records,
        batch=batch,
        spec=spec,
        raw_text='[{"ok": true}]',
        payload={
            "modelVersion": "test-model",
            "usageMetadata": {"totalTokenCount": 99},
        },
    )

    assert normalized["book_id"].tolist() == ["B001", "B002"]
    assert normalized["predicted_enjoyment"].tolist() == [3.5, 5.0]
    assert normalized["predicted_usefulness"].tolist() == [2.5, 1.0]
    assert normalized["call_stem"].nunique() == 1


def test_sample_count_curves_are_stable_when_all_runs_match() -> None:
    frame = pd.DataFrame(
        {
            "book_id": ["B001", "B002", "B003", "B004"],
            "key": ["a", "b", "c", "d"],
            "display_title": ["A", "B", "C", "D"],
            "display_author": ["AA", "BB", "CC", "DD"],
            "display_category": ["fiction"] * 4,
            "avg_enjoyment": [1.0, 2.0, 3.0, 4.0],
            "avg_usefulness": [1.5, 2.5, 3.5, 4.5],
        }
    )
    for sample_id in [f"p1_r{idx:02d}" for idx in range(1, 6)] + [
        f"p2_r{idx:02d}" for idx in range(1, 6)
    ]:
        frame[f"enjoyment_{sample_id}"] = frame["avg_enjoyment"]
        frame[f"usefulness_{sample_id}"] = frame["avg_usefulness"]

    aggregate_metrics = pd.DataFrame(
        {
            "target": ["enjoyment", "usefulness"],
            "model": ["Gemini all10 mean", "Gemini all10 mean"],
            "spearman_rho": [1.0, 1.0],
            "mae": [0.0, 0.0],
        }
    )

    metric_curve, gain_curve = sample_count_curves(
        wide_df=frame,
        aggregate_metrics=aggregate_metrics,
        bootstrap_draws=50,
        projection_max_samples=12,
        seed=123,
    )

    rho_rows = metric_curve[metric_curve["metric"].eq("spearman_rho")]
    mae_rows = metric_curve[metric_curve["metric"].eq("mae")]
    assert np.allclose(rho_rows["value_mean"], 1.0)
    assert np.allclose(mae_rows["value_mean"], 0.0)

    gain_80 = gain_curve[gain_curve["drop_pct"].eq(80)]
    assert np.isfinite(gain_80["value_mean"]).all()


def test_round_half_clips_into_valid_range() -> None:
    assert round_half(0.4) == 1.0
    assert round_half(5.6) == 5.0
    assert math.isclose(round_half(3.24), 3.0)
    assert math.isclose(round_half(3.26), 3.5)


def test_configure_experiment_and_prompt_override(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("custom prompt body\n", encoding="utf-8")

    configure_experiment("gemini_holdout_test_run")
    try:
        assert output_file("summary.md").name == "gemini_holdout_test_run_summary.md"
        assert load_prompt_override(str(prompt_path)) == "custom prompt body"
        assert load_prompt_override(None) is None
    finally:
        configure_experiment("gemini_holdout_batched_api")


def test_build_call_specs_supports_single_prompt_variant() -> None:
    holdout = make_holdout()
    specs = build_call_specs(
        holdout, batch_size=17, runs_per_prompt=3, seed=7, prompt_ids=(1,)
    )

    assert len(specs) == 3 * 4
    assert {spec.prompt_id for spec in specs} == {1}


def test_combine_target_prompts_keeps_both_rubrics() -> None:
    combined = combine_target_prompts(
        "Enjoy prompt\nOutput each book with:\n- predicted_enjoyment",
        "Useful prompt\nOutput each book with:\n- predicted_usefulness",
    )

    assert "ENJOYMENT RUBRIC" in combined
    assert "USEFULNESS RUBRIC" in combined
    assert "Output each book with" not in combined
