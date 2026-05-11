from datetime import datetime

from modules.easy_apply_question_log import (
    append_easy_apply_question_log,
    question_log_rows,
)


def test_question_log_rows_records_answer_source():
    rows = question_log_rows(
        {
            (
                "Cover letter",
                "Short answer",
                "textarea",
                "",
                "config:cover_letter",
            )
        },
        job_id="123",
        title="Engineer",
        company="Acme",
        status="submitted",
        seen_at=datetime(2026, 4, 29, 12, 0),
    )

    assert rows == [
        {
            "Date Seen": "2026-04-29 12:00:00",
            "Job ID": "123",
            "Title": "Engineer",
            "Company": "Acme",
            "Status": "submitted",
            "Label": "Cover letter",
            "Field Type": "textarea",
            "Answer": "Short answer",
            "Previous Answer": "",
            "Answer Source": "config:cover_letter",
        }
    ]


def test_append_easy_apply_question_log_writes_header_and_rows(tmp_path):
    path = tmp_path / "questions.csv"

    written = append_easy_apply_question_log(
        str(path),
        {("Major / Field of study", "Computer Science", "text", "", "question_bank")},
        "123",
        "Engineer",
        "Acme",
        "manual_writeup",
        seen_at=datetime(2026, 4, 29, 12, 0),
    )

    assert written == 1
    content = path.read_text(encoding="utf-8")
    assert "Date Seen,Job ID,Title,Company,Status,Label" in content
    assert "123,Engineer,Acme,manual_writeup,Major / Field of study" in content


def test_append_easy_apply_question_log_skips_existing_rows(tmp_path):
    path = tmp_path / "questions.csv"
    questions = {
        ("Major / Field of study", "Computer Science", "text", "", "question_bank")
    }

    first = append_easy_apply_question_log(
        str(path),
        questions,
        "123",
        "Engineer",
        "Acme",
        "submitted",
        seen_at=datetime(2026, 4, 29, 12, 0),
    )
    second = append_easy_apply_question_log(
        str(path),
        questions,
        "123",
        "Engineer",
        "Acme",
        "submitted",
        seen_at=datetime(2026, 4, 29, 12, 1),
    )

    assert first == 1
    assert second == 0
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
