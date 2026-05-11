from modules.experience_requirements import estimate_required_years


def test_experience_range_uses_lower_bound():
    assert estimate_required_years("5-10 years in solutions engineering") == 5
    assert estimate_required_years("5–10 yrs in customer-facing builds") == 5


def test_multiple_requirements_use_highest_required_floor():
    assert (
        estimate_required_years(
            "At least 7 years backend engineering. 2-3 years building LLM systems."
        )
        == 7
    )


def test_alternative_requirements_use_most_permissive_option():
    assert (
        estimate_required_years(
            "Bachelor's Degree with 2+ years relevant experience, or Graduate Degree "
            "with 0-2 years experience, or 6+ years without a degree"
        )
        == 0
    )


def test_no_experience_requirement_returns_none():
    assert (
        estimate_required_years("Participate in a 20 minute application process")
        is None
    )
