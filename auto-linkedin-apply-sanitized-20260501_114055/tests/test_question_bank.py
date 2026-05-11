import json

from modules.question_bank import QuestionBank


def write_bank(tmp_path, answers):
    path = tmp_path / "question_bank.json"
    path.write_text(json.dumps({"answers": answers}), encoding="utf-8")
    return path


def test_question_bank_answers_major_field_of_study(tmp_path):
    path = write_bank(
        tmp_path,
        [
            {
                "key": "major",
                "enabled": True,
                "field_types": ["text"],
                "match": {"contains_all": ["major", "field", "study"]},
                "answer": "Computer Science and Mathematics",
            }
        ],
    )

    answer = QuestionBank.from_file(path).answer_for_question(
        "Major / Field of study", "text"
    )

    assert answer
    assert answer.answer == "Computer Science and Mathematics"


def test_question_bank_answers_date_selects_by_occurrence(tmp_path):
    path = write_bank(
        tmp_path,
        [
            {
                "key": "start_month",
                "field_types": ["select"],
                "option_kind": "month",
                "occurrence": 1,
                "match": {"contains_all": ["dates", "attended"]},
                "answer": "August",
            },
            {
                "key": "end_month",
                "field_types": ["select"],
                "option_kind": "month",
                "occurrence": 2,
                "match": {"contains_all": ["dates", "attended"]},
                "answer": "May",
            },
        ],
    )
    options = ["Month", "January", "February", "March", "April", "May", "August"]
    bank = QuestionBank.from_file(path)

    assert (
        bank.answer_for_question(
            "Dates attended", "select", options, occurrence=1
        ).answer
        == "August"
    )
    assert (
        bank.answer_for_question(
            "Dates attended", "select", options, occurrence=2
        ).answer
        == "May"
    )


def test_question_bank_skips_answer_not_in_select_options(tmp_path):
    path = write_bank(
        tmp_path,
        [
            {
                "key": "currently_attend",
                "field_types": ["select"],
                "match": {"contains_all": ["currently", "attend"]},
                "answer": "No",
            }
        ],
    )

    assert (
        QuestionBank.from_file(path).answer_for_question(
            "Do you currently attend this institution?",
            "select",
            ["Select an option", "Yes"],
        )
        is None
    )
