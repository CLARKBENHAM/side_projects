"""Shared classifier and form-answer helpers for multi-site job appliers."""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import google.generativeai as genai

from modules.ai.cli_classifier import classify_job as classify_job_by_policy
from modules.application_answer_safety import (
    deterministic_answer_for_question,
    manual_review_reason_for_question,
    recommended_compensation_answer,
)

logger = logging.getLogger(__name__)

NETWORKING_JOBS_FILE = "all excels/networking_jobs.csv"

_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if _api_key:
    genai.configure(api_key=_api_key)
else:
    logger.warning("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment")

_gemini_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_model


def classify_job(
    title: str, company: str, description: str, job_id: str = "", job_link: str = ""
) -> str:
    """Returns 'network' or 'auto_apply'."""
    return classify_job_by_policy(
        title=title,
        company=company,
        description=description,
        job_id=job_id,
        job_link=job_link,
    )


_ANSWER_PROMPT = """You are filling out a job application form for a software engineer with 5 years of general software experience. Based in Los Angeles.

The form field label is: "{label}"
The job title is: "{job_title}" at "{company}"

Give a concise, professional answer for this field. If it's asking about:
- Years of experience: only use exact verified values: Python 6, PyTorch 4, Deep Learning 4, Distributed Training 2, LLM 3, RAG 2, Agentic AI 1, AWS 3, GCP 1, Node.js 3, React.js 3, C++ 1, SQL 4, Docker 4, Kubernetes 1, Jenkins 3
- Unsupported specific skills, certifications, regulated credentials, or domain-specific build claims: leave blank for manual review
- Willingness to relocate: "No" outside the Los Angeles commuting area; local onsite or hybrid near Los Angeles is acceptable
- Work authorization: "Yes, I am authorized to work in the US"
- Start date / availability: "Immediately" or "2 weeks notice"
- Salary expectations: "200000" to "250000" for annual AI/ML roles, or "90" for hourly contract roles
- Why interested / cover letter: Write 2-3 sentences about being excited about the role and bringing AI/ML expertise
- Anything about visa/sponsorship: "No, I do not require sponsorship"
- Skills or tools: List relevant ones from the verified years-of-experience map only
- Other questions: Give a reasonable, honest, brief answer

Respond with ONLY the answer text, nothing else. Keep it short."""


def answer_field(label: str, job_title: str = "", company: str = "") -> str:
    """Use Gemini to generate an answer for a free-response application field."""
    job_context = f"Job title: {job_title}\nCompany: {company}"
    safe_answer = recommended_compensation_answer(label, job_context)
    if safe_answer:
        return safe_answer[0]
    safe_answer = deterministic_answer_for_question(label, "text")
    if safe_answer:
        return safe_answer[0]
    if manual_review_reason_for_question(label):
        return ""

    model = _get_model()
    prompt = _ANSWER_PROMPT.format(label=label, job_title=job_title, company=company)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        logger.exception("Gemini answer_field failed for '%s'", label)
        return ""


def save_networking_job(
    job_id: str,
    title: str,
    company: str,
    work_location: str,
    work_style: str,
    description: str,
    hr_name: str,
    hr_link: str,
    job_link: str,
    date_listed: str,
    source: str = "",
) -> None:
    """Save a job flagged for networking to a shared CSV."""
    Path("all excels").mkdir(exist_ok=True)
    try:
        with open(NETWORKING_JOBS_FILE, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "Job ID",
                "Title",
                "Company",
                "Work Location",
                "Work Style",
                "About Job",
                "HR Name",
                "HR Link",
                "Job Link",
                "Date Posted",
                "Date Saved",
                "Source",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "Job ID": job_id,
                    "Title": title,
                    "Company": company,
                    "Work Location": work_location,
                    "Work Style": work_style,
                    "About Job": description[:500],
                    "HR Name": hr_name,
                    "HR Link": hr_link,
                    "Job Link": job_link,
                    "Date Posted": date_listed,
                    "Date Saved": datetime.now().isoformat(),
                    "Source": source,
                }
            )
        logger.info("Saved '%s | %s' for networking [%s]", title, company, source)
    except Exception:
        logger.exception("Failed to save networking job")
