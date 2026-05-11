"""LinkedIn search preferences.

These defaults are generic marketing examples. Replace them with the
applicant's actual target roles, locations, salary floor, industries, and
companies before running.
"""

import os

BASE_SEARCH_TERMS = [
    "Marketing Manager",
    "Growth Marketing Manager",
    "Product Marketing Manager",
    "Demand Generation Manager",
    "Content Marketing Manager",
    "Brand Strategist",
]


def _parse_search_terms_override() -> list[str]:
    raw_terms = os.environ.get("AUTO_APPLIER_SEARCH_TERMS", "").strip()
    if not raw_terms:
        return []
    return [term.strip() for term in raw_terms.split(",") if term.strip()]


def _resolve_search_terms() -> list[str]:
    override_terms = _parse_search_terms_override()
    if override_terms:
        return override_terms

    start_at_term = os.environ.get("AUTO_APPLIER_START_AT_TERM", "").strip()
    if not start_at_term:
        return BASE_SEARCH_TERMS.copy()

    if start_at_term not in BASE_SEARCH_TERMS:
        raise ValueError(
            f"AUTO_APPLIER_START_AT_TERM={start_at_term!r} is not in BASE_SEARCH_TERMS"
        )

    start_index = BASE_SEARCH_TERMS.index(start_at_term)
    return BASE_SEARCH_TERMS[start_index:]


def _resolve_positive_int(env_name: str, default: int) -> int:
    raw_value = os.environ.get(env_name, "").strip()
    if not raw_value:
        return default

    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{env_name} must be >= 1")
    return value


search_terms = _resolve_search_terms()
search_location = "United States"
switch_number = _resolve_positive_int("AUTO_APPLIER_SWITCH_NUMBER", 10)
randomize_search_order = False

sort_by = ""
date_posted = "Past week"
salary = "$80,000+"
easy_apply_only = False

experience_level = ["Associate", "Mid-Senior level"]
job_type = ["Full-time"]
on_site = ["Remote", "Hybrid"]

companies = []
location = []
industry = []
job_function = []
job_titles = []
benefits = []
commitments = []

under_10_applicants = False
in_your_network = False
fair_chance_employer = False
pause_after_filters = False

about_company_bad_words = []
about_company_good_words = []
bad_words = []
security_clearance = False
did_masters = False
current_experience = 3
