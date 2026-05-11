from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Iterable


QUESTION_LOG_FIELDNAMES = [
    "Date Seen",
    "Job ID",
    "Title",
    "Company",
    "Status",
    "Label",
    "Field Type",
    "Answer",
    "Previous Answer",
    "Answer Source",
]


def question_log_rows(
    questions_list: Iterable[tuple],
    job_id: str,
    title: str,
    company: str,
    status: str,
    seen_at: datetime | None = None,
) -> list[dict[str, str]]:
    seen_at = seen_at or datetime.now()
    rows = []
    for question in questions_list or []:
        if len(question) == 4:
            label, answer, field_type, previous_answer = question
            answer_source = "legacy"
        elif len(question) == 5:
            label, answer, field_type, previous_answer, answer_source = question
        else:
            continue
        rows.append(
            {
                "Date Seen": str(seen_at),
                "Job ID": job_id,
                "Title": title,
                "Company": company,
                "Status": status,
                "Label": str(label),
                "Field Type": str(field_type),
                "Answer": str(answer),
                "Previous Answer": str(previous_answer),
                "Answer Source": str(answer_source),
            }
        )
    return rows


def append_easy_apply_question_log(
    csv_path: str,
    questions_list: Iterable[tuple] | None,
    job_id: str,
    title: str,
    company: str,
    status: str,
    seen_at: datetime | None = None,
) -> int:
    if not questions_list:
        return 0

    rows = question_log_rows(
        questions_list,
        job_id=job_id,
        title=title,
        company=company,
        status=status,
        seen_at=seen_at,
    )
    if not rows:
        return 0

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    existing_keys = set()
    if file_exists:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            existing_keys = {
                (
                    row.get("Job ID", ""),
                    row.get("Status", ""),
                    row.get("Label", ""),
                    row.get("Field Type", ""),
                    row.get("Answer", ""),
                    row.get("Previous Answer", ""),
                    row.get("Answer Source", ""),
                )
                for row in csv.DictReader(handle)
            }
    rows = [
        row
        for row in rows
        if (
            row["Job ID"],
            row["Status"],
            row["Label"],
            row["Field Type"],
            row["Answer"],
            row["Previous Answer"],
            row["Answer Source"],
        )
        not in existing_keys
    ]
    if not rows:
        return 0

    with open(csv_path, mode="a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_LOG_FIELDNAMES)
        if not file_exists or handle.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)
