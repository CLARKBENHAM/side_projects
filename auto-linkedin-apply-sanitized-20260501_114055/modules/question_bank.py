from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION_BANK_PATH = ROOT / "config" / "question_bank.json"
MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def normalize_question_label(label: str) -> str:
    label = re.sub(r"\s+\[.*$", "", label)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", label.lower()).split())


def option_kind(options: list[str] | None) -> str:
    normalized_options = {
        " ".join(str(option).strip().lower().split()) for option in options or []
    }
    if len(MONTH_NAMES.intersection(normalized_options)) >= 6:
        return "month"
    year_count = sum(
        1 for option in normalized_options if re.fullmatch(r"20\d{2}|19\d{2}", option)
    )
    if year_count >= 5:
        return "year"
    return ""


def option_value(answer: str, options: list[str] | None) -> str | None:
    if not options:
        return answer

    normalized_answer = answer.strip().lower()
    for option in options:
        if normalized_answer == str(option).strip().lower():
            return option
    for option in options:
        normalized_option = str(option).strip().lower()
        if (
            normalized_answer in normalized_option
            or normalized_option in normalized_answer
        ):
            return option
    return None


@dataclass(frozen=True)
class QuestionBankAnswer:
    key: str
    answer: str


class QuestionBank:
    def __init__(self, rules: list[dict[str, Any]] | None = None):
        self.rules = rules or []

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_QUESTION_BANK_PATH) -> "QuestionBank":
        bank_path = Path(path)
        if not bank_path.exists():
            return cls()
        with bank_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        rules = data.get("answers", [])
        return cls(rules if isinstance(rules, list) else [])

    def answer_for_question(
        self,
        label: str,
        field_type: str,
        options: list[str] | None = None,
        occurrence: int | None = None,
    ) -> QuestionBankAnswer | None:
        normalized_label = normalize_question_label(label)
        detected_option_kind = option_kind(options)

        for rule in self.rules:
            if not rule.get("enabled", True):
                continue
            answer = str(rule.get("answer", "")).strip()
            if not answer:
                continue
            field_types = rule.get("field_types") or []
            if field_types and field_type not in field_types:
                continue
            if rule.get("option_kind") and rule["option_kind"] != detected_option_kind:
                continue
            if rule.get("occurrence") and rule["occurrence"] != occurrence:
                continue
            if not self._matches_rule(rule, normalized_label):
                continue

            resolved_answer = option_value(answer, options)
            if resolved_answer is None:
                continue
            return QuestionBankAnswer(
                key=str(rule.get("key", "")), answer=resolved_answer
            )

        return None

    @staticmethod
    def _matches_rule(rule: dict[str, Any], normalized_label: str) -> bool:
        match = rule.get("match") or {}
        contains_all = [
            normalize_question_label(str(item))
            for item in match.get("contains_all", [])
        ]
        contains_any = [
            normalize_question_label(str(item))
            for item in match.get("contains_any", [])
        ]
        exact = [normalize_question_label(str(item)) for item in match.get("exact", [])]

        if exact and normalized_label not in exact:
            return False
        if contains_all and not all(item in normalized_label for item in contains_all):
            return False
        if contains_any and not any(item in normalized_label for item in contains_any):
            return False
        return bool(exact or contains_all or contains_any)
