import os
import sys
import types

os.environ["AUTO_APPLIER_DISABLE_AI_CLI"] = "true"

sys.modules.setdefault(
    "pyautogui", types.SimpleNamespace(alert=lambda *args, **kwargs: None)
)

from modules.ai.cli_classifier import (  # noqa: E402
    MODEL_DECISIONS_FILE,
    _compensation_upper_bound_usd,
    _extract_json_object,
    answer_freeform_question,
    classify_job,
    hard_reject_reason,
    has_auto_apply_recruiter_signal,
)


def test_recruiter_signal_wins_before_network_domain():
    assert (
        classify_job(
            "Staff Software Engineer",
            "Oho Group",
            "Direct hire role for our client building AI systems for defense autonomy.",
        )
        == "auto_apply"
    )


def test_recruiter_signal_detects_staffing_description():
    assert has_auto_apply_recruiter_signal(
        "Insight Global",
        "Business consulting and staffing company. We mention Anthropic in passing.",
    )


def test_weak_company_suffix_does_not_override_direct_network_domain():
    assert not has_auto_apply_recruiter_signal(
        "Robotics Solutions Group",
        "Build autonomy software for deployed warehouse robots.",
    )
    assert (
        classify_job(
            "Robotics Software Engineer",
            "Robotics Solutions Group",
            "Build autonomy software for deployed warehouse robots.",
        )
        == "network"
    )


def test_generic_ai_engineer_at_b2b_saas_auto_applies():
    assert (
        classify_job(
            "Founding AI Engineer",
            "Early AI Startup",
            "Build B2B SaaS workflows for finance teams.",
        )
        == "auto_apply"
    )


def test_sub_170k_basic_saas_salary_auto_applies():
    assert (
        classify_job(
            "AI Engineer",
            "Normal SaaS Co",
            "Build B2B SaaS workflow automation for finance teams. "
            "Salary range: $145k - $165k base.",
        )
        == "auto_apply"
    )


def test_hard_rejects_attorney_role():
    assert hard_reject_reason(
        "Professional Liability Attorney",
        "Miller Waxler LLP",
        "J.D. required. Must be admitted to the bar.",
    )


def test_hard_rejects_csharp_dotnet_core_role():
    assert hard_reject_reason(
        "Senior Software Engineer (.NET)",
        "Prime Team Partners",
        "Requires strong C# and .NET experience.",
    )


def test_hard_rejects_phd_peptide_role():
    assert hard_reject_reason(
        "Senior AI Peptide and Protein Design Engineer",
        "Example Bio",
        "Requires a Ph.D. in Computational Biology.",
    )


def test_hard_rejects_ex_palantir_only_role():
    assert hard_reject_reason(
        "Forward Deployed Engineer",
        "Example Co",
        "Ex-Palantir only. Previously worked at Palantir is mandatory.",
    )


def test_hard_rejects_active_clearance_required_role():
    assert hard_reject_reason(
        "AI Engineer",
        "Defense Co",
        "This role requires active security clearance.",
    )


def test_hard_rejects_ts_sci_required_role():
    assert hard_reject_reason(
        "AI Engineer",
        "Defense Co",
        "TS/SCI clearance required.",
    )


def test_does_not_hard_reject_clearance_obtainable_role():
    assert (
        hard_reject_reason(
            "AI Engineer",
            "Defense Co",
            "This role requires ability to obtain a security clearance.",
        )
        is None
    )


def test_hard_rejects_strict_advanced_degree_requirement():
    assert hard_reject_reason(
        "Research Scientist",
        "Example Lab",
        "Minimum qualifications: Ph.D. required in Computer Science.",
    )
    assert hard_reject_reason(
        "ML Engineer",
        "Example Lab",
        "Master's degree required in Computer Science.",
    )


def test_does_not_hard_reject_advanced_degree_with_equivalent_experience():
    assert (
        hard_reject_reason(
            "ML Engineer",
            "Example Lab",
            "Master's degree or equivalent practical experience required.",
        )
        is None
    )


def test_bare_contract_compensation_range_is_treated_as_hourly():
    upper_bound, _ = _compensation_upper_bound_usd(
        "Compensation: $50 - $120\nType: Hourly contract"
    )
    assert upper_bound == 120 * 2080


def test_sub_170k_protected_domain_still_networks():
    assert (
        classify_job(
            "AI Engineer",
            "Industrial AI Co",
            "Build AI systems for manufacturing quality workflows. "
            "Salary range: $145k - $165k base.",
        )
        == "network"
    )


def test_domain_keyword_protects_direct_company_role():
    assert (
        classify_job(
            "Backend Engineer",
            "Example Co",
            "Build production AI systems for AEC and construction teams.",
        )
        == "network"
    )


def test_domain_keyword_in_title_protects_direct_company_role():
    assert (
        classify_job(
            "Robotics Software Engineer",
            "Example Co",
            "Build Linux services for fielded autonomy products.",
        )
        == "network"
    )


def test_extract_json_object_from_codex_output():
    assert _extract_json_object(
        'analysis text\n{"decision":"network","confidence":0.8,"reason":"AEC role"}\n'
    ) == {"decision": "network", "confidence": 0.8, "reason": "AEC role"}


def test_classify_job_without_job_id_does_not_write_global_audit_file():
    before = (
        os.path.getsize(MODEL_DECISIONS_FILE)
        if os.path.exists(MODEL_DECISIONS_FILE)
        else 0
    )

    assert (
        classify_job(
            "Backend Engineer",
            "Example Co",
            "Build production AI systems for AEC and construction teams.",
        )
        == "network"
    )

    after = (
        os.path.getsize(MODEL_DECISIONS_FILE)
        if os.path.exists(MODEL_DECISIONS_FILE)
        else 0
    )
    assert after == before


def test_classify_job_saves_decision_audit_row(tmp_path, monkeypatch):
    decisions_file = tmp_path / "model_decisions.csv"
    monkeypatch.setattr(
        "modules.ai.cli_classifier.MODEL_DECISIONS_FILE", str(decisions_file)
    )

    assert (
        classify_job(
            "Backend Engineer",
            "Example Co",
            "Build production AI systems for AEC and construction teams.",
            job_id="123",
            job_link="https://www.linkedin.com/jobs/view/123",
        )
        == "network"
    )

    content = decisions_file.read_text(encoding="utf-8")
    assert "Job ID,Title,Company,Decision,Confidence,Reason,Source" in content
    assert "123,Backend Engineer,Example Co,network," in content
    assert "deterministic_fallback" in content


def test_answer_freeform_question_returns_none_when_ai_cli_disabled():
    assert (
        answer_freeform_question(
            "Why are you interested in this role?",
            "Applicant has relevant marketing experience.",
            "Job title: Marketing Manager",
        )
        is None
    )
