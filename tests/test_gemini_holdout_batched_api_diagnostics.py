from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_books_tracking.scripts.temp import (
    gemini_holdout_batched_api_diagnostics as diag,
)


def test_pipeline_audit_flags_duplicate_holdout_key(
    tmp_path: Path, monkeypatch
) -> None:
    master = pd.DataFrame(
        [
            {
                "title": "Kelly",
                "author": "Clarence",
                "source": "Holdout 2026",
            },
            {
                "title": "Kelly",
                "author": "Clarence",
                "source": "Play Export (unverified)",
            },
            {
                "title": "Train Book",
                "author": "Other",
                "source": "Play Export",
            },
        ]
    )
    master_path = tmp_path / "master.csv"
    master.to_csv(master_path, index=False)
    monkeypatch.setattr(diag, "MASTER_CSV", master_path)

    long_df = pd.DataFrame(
        {
            "book_id": ["B001"] * 10,
            "sample_id": [f"p1_r{i:02d}" for i in range(1, 11)],
            "call_stem": [f"stem_{i}" for i in range(10)],
            "confidence": ["high"] * 10,
            "predicted_enjoyment": [3.0] * 10,
            "predicted_usefulness": [2.0] * 10,
        }
    )
    audit = diag.pipeline_audit(long_df)
    duplicate_row = audit.loc[audit["check"].eq("holdout_key_leakage_duplicates")].iloc[
        0
    ]

    assert duplicate_row["value"] == 1
    assert duplicate_row["status"] == "warn"


def test_extract_batch_positions_reads_prompt_files(
    tmp_path: Path, monkeypatch
) -> None:
    call_dir = tmp_path / "calls"
    call_dir.mkdir()
    prompt = call_dir / "gemini_holdout_prompt1_rep01_batch01_prompt.txt"
    prompt.write_text(
        "\n".join(
            [
                "header",
                "Books to score:",
                "B001 | title=A",
                "B017 | title=B",
                "B003 | title=C",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(diag, "CALL_DIR", call_dir)

    positions = diag.extract_batch_positions().sort_values("batch_pos")

    assert positions["book_id"].tolist() == ["B001", "B017", "B003"]
    assert positions["batch_pos"].tolist() == [1, 2, 3]


def test_prompt_delta_by_book_computes_prompt_gap() -> None:
    long_df = pd.DataFrame(
        {
            "book_id": ["B001", "B001", "B002", "B002"],
            "display_title": ["A", "A", "B", "B"],
            "display_author": ["AA", "AA", "BB", "BB"],
            "display_category": ["fiction", "fiction", "History", "History"],
            "avg_enjoyment": [2.0, 2.0, 3.0, 3.0],
            "avg_usefulness": [1.0, 1.0, 2.0, 2.0],
            "prompt_id": [1, 2, 1, 2],
            "predicted_enjoyment": [2.5, 3.5, 3.0, 3.0],
            "predicted_usefulness": [1.0, 2.0, 1.5, 4.0],
        }
    )

    delta = diag.prompt_delta_by_book(long_df)
    row_a = delta.loc[delta["book_id"].eq("B001")].iloc[0]
    row_b = delta.loc[delta["book_id"].eq("B002")].iloc[0]

    assert row_a["delta_enjoyment"] == 1.0
    assert row_a["delta_usefulness"] == 1.0
    assert row_b["delta_usefulness"] == 2.5
