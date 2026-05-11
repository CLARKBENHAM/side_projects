from __future__ import annotations

import math

import pandas as pd

from ai_books_tracking.scripts.temp import gemini_prompt_optimization_analysis as opt


def make_long_frame() -> pd.DataFrame:
    rows = []
    books = [
        ("B001", "Prestige History", "Histories", 2.0, 1.0, 4.5, 1.5, 4.0, 4.5),
        ("B002", "Essay Collection", "Literature", 2.5, 1.5, 4.0, 1.5, 3.5, 2.5),
        ("B003", "Applied Manual", "Computer Science", 3.0, 2.5, 2.5, 3.0, 2.5, 4.5),
    ]
    for book_id, title, category, actual_e, actual_u, p1e, p1u, p2e, p2u in books:
        for prompt_id, pred_e, pred_u, reason in (
            (
                1,
                p1e,
                p1u,
                "Primary-source power history with excellent prose.",
            ),
            (
                2,
                p2e,
                p2u,
                "Provides highly transferable models of power and incentives.",
            ),
        ):
            rows.append(
                {
                    "book_id": book_id,
                    "display_title": title,
                    "display_author": "Author",
                    "display_category": category,
                    "avg_enjoyment": actual_e,
                    "avg_usefulness": actual_u,
                    "prompt_id": prompt_id,
                    "predicted_enjoyment": pred_e,
                    "predicted_usefulness": pred_u,
                    "brief_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def test_build_candidate_metrics_reports_drop_gains_and_scores() -> None:
    wide = pd.DataFrame(
        {
            "avg_enjoyment": [1.0, 2.0, 3.0, 4.0],
            "avg_usefulness": [1.0, 2.0, 3.0, 4.0],
            "prompt1_enjoyment_mean": [1.0, 2.0, 3.0, 4.0],
            "prompt2_enjoyment_mean": [4.0, 3.0, 2.0, 1.0],
            "prompt1_usefulness_mean": [1.0, 2.0, 3.0, 4.0],
            "prompt2_usefulness_mean": [4.0, 3.0, 2.0, 1.0],
            "prompt_mean_both_enjoyment": [2.5, 2.5, 2.5, 2.5],
            "prompt_mean_both_usefulness": [2.5, 2.5, 2.5, 2.5],
            "targetwise_best_enjoyment": [4.0, 3.0, 2.0, 1.0],
            "targetwise_best_usefulness": [1.0, 2.0, 3.0, 4.0],
            "targetwise_conservative_usefulness": [1.0, 2.0, 2.0, 1.0],
        }
    )

    metrics = opt.build_candidate_metrics(wide)
    prompt1_enjoy = metrics[
        metrics["model"].eq("prompt1_mean") & metrics["target"].eq("enjoyment")
    ].iloc[0]
    prompt2_enjoy = metrics[
        metrics["model"].eq("prompt2_mean") & metrics["target"].eq("enjoyment")
    ].iloc[0]

    assert math.isclose(prompt1_enjoy["spearman_rho"], 1.0)
    assert math.isclose(prompt1_enjoy["mae"], 0.0)
    assert math.isclose(prompt1_enjoy["gain_drop_30"], 0.5)
    assert math.isclose(prompt2_enjoy["spearman_rho"], -1.0)


def test_assign_failure_mode_flags_prompt2_usefulness_inflation() -> None:
    failure_mode = opt.assign_failure_mode(
        prompt_id=2,
        target="usefulness",
        title="The Power State",
        category="Histories",
        error=2.4,
        reason_text="Provides highly transferable models of power, incentives, and institutional strategy.",
    )

    assert failure_mode == "seriousness_to_usefulness_inflation"


def test_build_failure_examples_labels_prose_collection_and_technical_cases() -> None:
    failure_df = opt.build_failure_examples(make_long_frame())

    essay_row = failure_df[
        failure_df["display_title"].eq("Essay Collection")
        & failure_df["target"].eq("enjoyment")
        & failure_df["prompt_id"].eq(1)
    ].iloc[0]
    manual_row = failure_df[
        failure_df["display_title"].eq("Applied Manual")
        & failure_df["target"].eq("usefulness")
        & failure_df["prompt_id"].eq(2)
    ].iloc[0]

    assert essay_row["failure_mode"] == "prose_collection_overrating"
    assert manual_row["failure_mode"] == "technical_applicability_overreach"


def test_prompt_texts_include_key_guardrails() -> None:
    assert "metadata" in opt.PROMPT_ENJOYMENT_V3.lower()
    assert "regress toward the category middle" in opt.PROMPT_ENJOYMENT_V3
    assert "A book should rarely exceed 2.5 usefulness" in opt.PROMPT_USEFULNESS_V3
    assert "Do not infer high usefulness" in opt.PROMPT_USEFULNESS_V3
