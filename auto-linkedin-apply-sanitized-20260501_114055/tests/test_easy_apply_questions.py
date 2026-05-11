from modules.easy_apply_questions import (
    configured_free_text_answer,
    is_placeholder_answer,
    is_postal_code_label,
    manual_writeup_reason,
    should_try_long_form_answer,
)
from modules.application_answer_safety import (
    answer_record_manual_review_reason,
    deterministic_answer_for_question,
    is_rating_scale_label,
    manual_review_reason_for_question,
    recommended_compensation_answer,
    should_override_existing_answer,
)


def test_postal_code_detection_does_not_match_infrastructure_as_code():
    assert is_postal_code_label("ZIP / Postal Code")
    assert is_postal_code_label("Postcode")
    assert not is_postal_code_label(
        "1-10 how strong are you in Infrastructure as Code?"
    )
    assert not is_postal_code_label(
        "On a scale of 1-10, how strong is your scientific Python code?"
    )


def test_rating_scale_overrides_zip_code_prefill():
    label = (
        "On a scale of 1-10, how would you rate your professional proficiency "
        "writing production-ready scientific python code?"
    )

    assert is_rating_scale_label(label)
    assert should_override_existing_answer(label, "12345", "text")
    assert deterministic_answer_for_question(label, "text", confidence_level="8") == (
        "8",
        "profile:rating_scale",
    )


def test_unsupported_claims_are_overridden_to_no():
    label = "Do you have a Doctor of Law (J.D.)?"

    assert should_override_existing_answer(label, "Yes", "radio")
    assert deterministic_answer_for_question(label, "radio", ["Yes", "No"]) == (
        "No",
        "legal:verified_no_claim",
    )


def test_specific_unsupported_years_queue_manual_review():
    assert (
        manual_review_reason_for_question(
            "How many years of work experience do you have with C#?"
        )
        == "unsupported required skill or employer-specific experience"
    )


def test_visa_sponsorship_polarity_is_not_inverted():
    assert deterministic_answer_for_question(
        "Are you legally authorized to work in the US?", "radio", ["Yes", "No"]
    ) == ("Yes", "legal:work_authorization")
    assert deterministic_answer_for_question(
        "Will you require visa sponsorship now or in the future?",
        "radio",
        ["Yes", "No"],
    ) == ("No", "legal:no_sponsorship_required")
    assert deterministic_answer_for_question(
        "Are you ok to work without Visa Sponsorship?", "radio", ["Yes", "No"]
    ) == ("Yes", "legal:ok_without_sponsorship")
    assert (
        deterministic_answer_for_question(
            "Are you a US Citizen?", "radio", ["Yes", "No"]
        )
        is None
    )
    assert (
        manual_review_reason_for_question("Are you a US Citizen?")
        == "citizenship or permanent resident status is not locked"
    )


def test_c2c_and_w2_polarity():
    assert deterministic_answer_for_question(
        "Do you require working on C2C? If you own a 1099 mark NO.",
        "radio",
        ["Yes", "No"],
    ) == ("No", "preference:no_c2c_required")
    assert deterministic_answer_for_question(
        "Are you willing to work W2?", "radio", ["Yes", "No"]
    ) == ("Yes", "preference:w2_ok")
    assert deterministic_answer_for_question(
        "Are you comfortable with a 3 year extending contract?",
        "select",
        ["Select an option", "Yes", "No"],
    ) == ("Yes", "preference:contract_ok")
    assert (
        manual_review_reason_for_question(
            "Are you comfortable with a 3 year extending contract?"
        )
        is None
    )


def test_clearance_answers_do_not_claim_current_clearance():
    assert deterministic_answer_for_question(
        "Do you currently hold an active security clearance?", "radio", ["Yes", "No"]
    ) == ("No", "legal:no_current_clearance")
    assert deterministic_answer_for_question(
        "Are you willing to obtain a security clearance?", "radio", ["Yes", "No"]
    ) == ("Yes", "legal:willing_to_obtain_clearance")
    assert deterministic_answer_for_question("Security clearance details", "text") == (
        "I do not currently have a clearance, but I am willing to obtain one.",
        "legal:no_current_clearance",
    )
    assert should_override_existing_answer(
        "Are you willing to obtain a security clearance?", "No", "radio"
    )
    assert should_override_existing_answer(
        "Do you currently hold an active security clearance?", "Yes", "radio"
    )


def test_skill_years_use_verified_map_only():
    expected_skill_years = {
        "Python": ("6", "profile:skill_years:python"),
        "PyTorch": ("4", "profile:skill_years:pytorch"),
        "Deep Learning": ("4", "profile:skill_years:deep_learning"),
        "Distributed Training": ("2", "profile:skill_years:distributed_training"),
        "LLM": ("3", "profile:skill_years:llm"),
        "RAG": ("2", "profile:skill_years:rag"),
        "Agentic AI": ("1", "profile:skill_years:agentic_ai"),
        "AWS": ("3", "profile:skill_years:aws"),
        "GCP": ("1", "profile:skill_years:gcp"),
        "Node.js": ("3", "profile:skill_years:node"),
        "React.js": ("3", "profile:skill_years:react"),
        "C++": ("1", "profile:skill_years:cpp"),
        "SQL": ("4", "profile:skill_years:sql"),
        "Docker": ("4", "profile:skill_years:docker"),
        "Kubernetes": ("1", "profile:skill_years:kubernetes"),
        "Jenkins": ("3", "profile:skill_years:jenkins"),
    }
    for skill, expected in expected_skill_years.items():
        assert (
            deterministic_answer_for_question(
                f"How many years of work experience do you have with {skill}?",
                "text",
            )
            == expected
        )
        assert not should_override_existing_answer(
            f"How many years of work experience do you have with {skill}?",
            expected[0],
            "text",
        )
    assert should_override_existing_answer(
        "How many years of work experience do you have with Jenkins?", "4", "text"
    )
    assert (
        answer_record_manual_review_reason(
            (
                "How many years of work experience do you have with Jenkins?",
                "4",
                "text",
                "4",
                "existing",
            )
        )
        == "text: How many years of work experience do you have with Jenkins? has an unsafe saved or generated answer"
    )
    assert (
        manual_review_reason_for_question(
            "How many years of experience do you have with Terraform?"
        )
        == "specific skill years are not in the verified skill map"
    )
    assert should_override_existing_answer(
        "How many years of experience do you have with Terraform?", "4", "text"
    )


def test_unmapped_skill_years_block_adjacent_verified_skill_answers():
    assert (
        deterministic_answer_for_question(
            "How many years of experience do you have with Python and Spark?",
            "text",
        )
        is None
    )
    assert (
        manual_review_reason_for_question(
            "How many years of experience do you have with Python and Spark?"
        )
        == "specific skill years are not in the verified skill map"
    )
    assert should_override_existing_answer(
        "How many years of experience do you have with Python and Spark?", "6", "text"
    )
    assert (
        deterministic_answer_for_question(
            "How many years of experience do you have with Python and Ray?",
            "text",
        )
        is None
    )
    assert (
        deterministic_answer_for_question(
            "How many years of experience do you have with Azure OpenAI?",
            "text",
        )
        is None
    )
    locked_additional_years = {
        "How many years of architect level Ai or LLM experience do you have?": (
            "2",
            "profile:skill_years:architect_ai_llm",
        ),
        "How many years of work experience do you have with Azure AI Foundry?": (
            "1",
            "profile:skill_years:azure_ai_foundry",
        ),
        "How many years of work experience do you have with Java?": (
            "1",
            "profile:skill_years:java",
        ),
        "How many years of work experience do you have with Claude Code Subagents?": (
            "1",
            "profile:skill_years:claude_code_subagents",
        ),
    }
    for label, expected in locked_additional_years.items():
        assert deterministic_answer_for_question(label, "text") == expected
        assert manual_review_reason_for_question(label) is None
        assert should_override_existing_answer(label, "4", "text")
        assert not should_override_existing_answer(label, expected[0], "text")
    assert (
        deterministic_answer_for_question(
            "How many years of work experience do you have with JavaScript?", "text"
        )
        is None
    )
    assert (
        deterministic_answer_for_question(
            "How many years of Financial Services experience do you currently have?",
            "text",
        )
        is None
    )
    assert (
        manual_review_reason_for_question(
            "How many years of Financial Services experience do you currently have?"
        )
        == "specific skill years are not in the verified skill map"
    )
    assert should_override_existing_answer(
        "How many years of Financial Services experience do you currently have?",
        "5",
        "text",
    )
    assert (
        deterministic_answer_for_question(
            "How many years of work experience do you have with Large Language Model Operations (LLMOps)?",
            "text",
        )
        is None
    )
    assert deterministic_answer_for_question(
        "How many years of experience do you have with Google Cloud?", "text"
    ) == ("1", "profile:skill_years:gcp")
    assert (
        deterministic_answer_for_question(
            "How many years of experience do you have with Go?", "text"
        )
        is None
    )


def test_existing_specialized_yes_answers_are_not_trusted():
    assert should_override_existing_answer(
        "Have you owned end-to-end speech AI systems?", "Yes", "radio"
    )
    assert should_override_existing_answer(
        "Do you have professional experience in computer vision?", "Yes", "radio"
    )
    assert not should_override_existing_answer(
        "Can you relocate to El Segundo and work onsite?", "Yes", "radio"
    )


def test_unsupported_domain_specific_questions_do_not_overclaim():
    assert deterministic_answer_for_question(
        "How many years of work experience do you have with UAV systems?", "text"
    ) == ("0", "profile:no_experience:domain_specific")
    assert deterministic_answer_for_question(
        "Have you built fine-tuned ASR/TTS audio models?", "radio", ["Yes", "No"]
    ) == ("No", "profile:no_experience:domain_specific")
    assert deterministic_answer_for_question(
        "Have you built fine-tuned ASR/TTS audio models?", "text"
    ) == ("No", "profile:no_experience:domain_specific")
    assert deterministic_answer_for_question(
        "Do you have professional experience in computer vision?",
        "select",
        ["Select an option", "Yes", "No"],
    ) == ("No", "profile:no_experience:domain_specific")
    assert deterministic_answer_for_question(
        "Have you worked with sensor fusion, signal processing, or other multimodal inference systems?",
        "select",
        ["Select an option", "Yes", "No"],
    ) == ("No", "profile:no_experience:domain_specific")
    assert should_override_existing_answer(
        "How many years of work experience do you have with SAR workflows?", "4", "text"
    )
    assert deterministic_answer_for_question(
        "Do you have CSSLP certification?", "radio", ["Yes", "No"]
    ) == ("No", "legal:verified_no_claim")
    assert deterministic_answer_for_question(
        "Do you have a professional certification?", "text"
    ) == ("No", "profile:no_certification_or_license")


def test_location_and_travel_preferences_are_locked():
    assert deterministic_answer_for_question(
        "Are you located within a few hours of NYC, SF Bay Area, or Seattle?",
        "radio",
        ["Yes", "No"],
    ) == ("No", "preference:location_mismatch")
    assert deterministic_answer_for_question(
        "Can you commute to El Segundo and work onsite 5 days per week?",
        "radio",
        ["Yes", "No"],
    ) == ("Yes", "preference:local_onsite_ok")
    assert deterministic_answer_for_question(
        "Are you willing to relocate to San Francisco?", "radio", ["Yes", "No"]
    ) == ("No", "preference:location_mismatch")
    assert deterministic_answer_for_question(
        "Are you willing to travel up to 75%?", "radio", ["Yes", "No"]
    ) == ("Yes", "preference:max_travel")


def test_role_aware_compensation_answers_do_not_undersell():
    assert recommended_compensation_answer(
        "Expected annual compensation",
        "Job title: Principal AI Engineer\nSalary range: $200k - $350k",
    ) == ("250000", "preference:listed_comp_high")
    assert recommended_compensation_answer(
        "Expected annual compensation",
        "Job title: Senior AI Engineer\nSalary range: $200k - $240k",
    ) == ("220000", "preference:listed_comp_mid")
    assert recommended_compensation_answer(
        "Hourly rate",
        "Job title: Contract ML Engineer\nType: hourly contract",
    ) == ("90", "preference:comp_floor:contract_hourly")


def test_untrusted_answer_sources_force_manual_review():
    assert (
        answer_record_manual_review_reason(
            ("Do you own end-to-end speech AI?", "Yes", "radio", "", "config")
        )
        == "radio: Do you own end-to-end speech AI? used untrusted answer source config"
    )
    assert (
        answer_record_manual_review_reason(
            ("Mobile phone number", "5555550100", "text", "", "profile:phone")
        )
        is None
    )


def test_placeholder_answers_include_linkedin_select_defaults():
    assert is_placeholder_answer("")
    assert is_placeholder_answer("Select an option")
    assert is_placeholder_answer("Month")
    assert is_placeholder_answer("Year")
    assert not is_placeholder_answer("January")


def test_manual_writeup_reason_is_stable_for_add_and_remove():
    assert (
        manual_writeup_reason("text", " Major / Field of study ")
        == "text: Major / Field of study"
    )


def test_configured_free_text_answer_uses_cover_letter_and_summary():
    assert configured_free_text_answer(
        "Cover letter", "cover text", "summary text"
    ) == (
        "cover text",
        "config:cover_letter",
    )
    assert configured_free_text_answer(
        "Tell us about yourself", "cover text", "summary text"
    ) == ("summary text", "config:linkedin_summary")


def test_should_try_long_form_answer_for_essay_like_questions():
    assert should_try_long_form_answer("Why are you interested in this role?")
    assert not should_try_long_form_answer("Mobile phone number")
