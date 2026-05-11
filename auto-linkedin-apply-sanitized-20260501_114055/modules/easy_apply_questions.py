import re


PLACEHOLDER_ANSWERS = {"", "select an option", "month", "year", "day"}


def is_placeholder_answer(answer: object) -> bool:
    normalized = " ".join(str(answer or "").strip().lower().split())
    return normalized in PLACEHOLDER_ANSWERS


def is_postal_code_label(label: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
    return bool(
        re.search(r"\bzip(?: code)?\b|\bpostal(?: code)?\b|\bpostcode\b", normalized)
    )


def manual_writeup_reason(field_type: str, label: str) -> str:
    cleaned_label = label.strip() if label else "Unknown"
    return f"{field_type}: {cleaned_label}"


def normalized_label(label: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", label.lower()).split())


def configured_free_text_answer(
    label: str, configured_cover_letter: str, configured_summary: str
) -> tuple[str, str] | None:
    normalized = normalized_label(label)
    cover = configured_cover_letter.strip()
    summary = configured_summary.strip()
    if cover and ("cover letter" in normalized or "motivation letter" in normalized):
        return cover, "config:cover_letter"
    if summary and (
        "summary" in normalized
        or "about yourself" in normalized
        or "tell us about yourself" in normalized
    ):
        return summary, "config:linkedin_summary"
    return None


def should_try_long_form_answer(label: str) -> bool:
    normalized = normalized_label(label)
    long_form_markers = (
        "additional",
        "anything else",
        "challenge",
        "contribute",
        "cover",
        "describe",
        "elaborate",
        "essay",
        "example",
        "explain",
        "fit",
        "interest",
        "interested",
        "motivation",
        "note",
        "proud",
        "project",
        "summary",
        "tell us",
        "why",
    )
    return any(marker in normalized for marker in long_form_markers)
