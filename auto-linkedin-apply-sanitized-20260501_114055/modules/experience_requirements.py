import re

_EXPERIENCE_SINGLE_PATTERN = re.compile(
    r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE
)
_EXPERIENCE_RANGE_PATTERN = re.compile(
    r"\b(\d{1,2})\s*(?:-|to|–|—)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)


def _is_plausible_experience_years(value: int) -> bool:
    return 0 <= value <= 30


def _extract_experience_lower_bounds(text: str) -> list[int]:
    values = []
    range_spans = []

    for match in _EXPERIENCE_RANGE_PATTERN.finditer(text):
        lower_bound = min(int(match.group(1)), int(match.group(2)))
        if _is_plausible_experience_years(lower_bound):
            values.append(lower_bound)
            range_spans.append(match.span())

    for match in _EXPERIENCE_SINGLE_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in range_spans):
            continue
        value = int(match.group(1))
        if _is_plausible_experience_years(value):
            values.append(value)

    return values


def _split_experience_requirement_chunks(text: str) -> list[str]:
    return [chunk for chunk in re.split(r"[\n.;]+", text) if chunk.strip()]


def estimate_required_years(text: str) -> int | None:
    """
    Estimate the experience floor to compare against current_experience.

    Ranges like "5-10 years" count as 5. Alternative requirements in the
    same chunk, like "2+ years with a degree or 6+ without", use the most
    permissive lower bound for that chunk.
    """
    required_by_chunk = []
    for chunk in _split_experience_requirement_chunks(text):
        lower_bounds = _extract_experience_lower_bounds(chunk)
        if not lower_bounds:
            continue
        if re.search(r"\bor\b", chunk, flags=re.IGNORECASE) and len(lower_bounds) > 1:
            required_by_chunk.append(min(lower_bounds))
        else:
            required_by_chunk.append(max(lower_bounds))

    if not required_by_chunk:
        return None
    return max(required_by_chunk)
