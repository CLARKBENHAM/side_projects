import csv
import json
import os
import re
import shlex
import subprocess
import tempfile
import textwrap
from hashlib import sha256
from datetime import datetime

from modules.helpers import print_lg

NETWORKING_JOBS_FILE = "all excels/networking_jobs.csv"
MODEL_DECISIONS_FILE = "all excels/model_decisions.csv"
AUTO_APPLY_COMPANY_KEYWORDS = (
    "recruit",
    "staffing",
    "talent",
    "search",
)
AUTO_APPLY_COMPANY_CONTEXT_KEYWORDS = (
    "consulting",
    "solutions",
    "partners",
    "group",
)
AUTO_APPLY_DESCRIPTION_KEYWORDS = (
    "our client",
    "contract to hire",
    "contract role",
    "c2c",
    "corp to corp",
    "hourly",
    "staffing agency",
    "staffing company",
    "recruiting company",
    "talent company",
    "business consulting and services",
    "recruiter",
)
NETWORK_DOMAIN_KEYWORDS = (
    "aec",
    "architecture, engineering and construction",
    "architecture engineering and construction",
    "architectural engineering",
    "construction",
    "built environment",
    "bim",
    "revit",
    "cad",
    "digital twin",
    "manufacturing",
    "industrial automation",
    "factory",
    "supply chain",
    "robotics",
    "hardware",
    "embedded",
    "aerospace",
    "spacecraft",
    "satellite",
    "defense",
    "defence",
    "national security",
    "military",
    "clearance",
    "dod",
    "drone",
    "uav",
    "usv",
    "autonomy",
)
GENERIC_SAAS_KEYWORDS = (
    "saas",
    "software as a service",
    "b2b",
    "enterprise software",
    "internal tools",
    "workflow",
    "workflows",
    "platform",
    "product",
    "web app",
    "web application",
    "backend",
    "front end",
    "frontend",
    "full-stack",
    "full stack",
    "api",
    "cloud",
    "customer",
    "crm",
    "sales",
    "marketing",
    "martech",
    "adtech",
    "fintech",
    "productivity",
)
LEGAL_ROLE_PATTERN = re.compile(
    r"\b(attorney|lawyer|legal counsel|litigation counsel|professional liability)\b",
    re.IGNORECASE,
)
CSHARP_DOTNET_PATTERN = re.compile(
    r"(?:(?<![a-z0-9])c\s*#(?![a-z0-9])|\bc sharp\b|\.net\b|\bdotnet\b)",
    re.IGNORECASE,
)
EMBEDDED_FLIGHT_ROLE_PATTERN = re.compile(
    r"\b(embedded|firmware|flight software|spacecraft software)\b", re.IGNORECASE
)
PEPTIDE_BIOLOGY_PATTERN = re.compile(
    r"\b(peptide|protein design|computational biology|bioinformatics)\b",
    re.IGNORECASE,
)
PHD_REQUIREMENT_PATTERN = re.compile(r"\b(ph\.?\s*d\.?|doctorate)\b", re.IGNORECASE)
EXACT_EMPLOYER_REQUIREMENT_PATTERN = re.compile(
    r"\b(ex[- ]?palantir|palantir only|previously worked at palantir|worked at palantir)\b",
    re.IGNORECASE,
)
LICENSED_ROLE_PATTERN = re.compile(
    r"\b(cpa|certified public accountant|rn\b|registered nurse|m\.?d\.?|medical doctor|licensed)\b",
    re.IGNORECASE,
)
ACTIVE_CLEARANCE_REQUIRED_PATTERN = re.compile(
    r"\b(active|current|existing)\s+(security\s+)?clearance\b|"
    r"\bmust\s+(hold|have|possess)\b.{0,50}\b(clearance|polygraph)\b|"
    r"\brequires\b.{0,50}\b(active|current)\b.{0,30}\b(clearance|polygraph)\b",
    re.IGNORECASE,
)
CLEARANCE_REQUIREMENT_TERM_PATTERN = re.compile(
    r"\b(security\s+clearance|secret\s+clearance|top\s+secret|ts/sci|"
    r"sci\s+clearance|polygraph)\b",
    re.IGNORECASE,
)
CLEARANCE_OBTAINABLE_PATTERN = re.compile(
    r"\b(ability to obtain|able to obtain|eligible to obtain|willing to obtain|"
    r"obtain|sponsor)\b",
    re.IGNORECASE,
)
REQUIRED_CONTEXT_PATTERN = re.compile(
    r"\b(required|requires|requirement|mandatory|must|need|needs)\b",
    re.IGNORECASE,
)
ADVANCED_DEGREE_PATTERN = re.compile(
    r"\b(ph\.?\s*d\.?|doctorate|doctoral degree|master'?s degree|"
    r"m\.?\s*s\.? degree)\b",
    re.IGNORECASE,
)
ADVANCED_DEGREE_SAFE_ALTERNATIVE_PATTERN = re.compile(
    r"\b(preferred|nice to have|or equivalent|equivalent experience|"
    r"or relevant experience|or related experience|or equivalent practical experience|"
    r"in lieu|bachelor'?s?\b.{0,40}\bor\b|\bor\b.{0,40}\bbachelor'?s?)\b",
    re.IGNORECASE,
)
NETWORKING_MIN_ANNUAL_SALARY_USD = int(
    os.environ.get("AUTO_APPLIER_NETWORKING_MIN_ANNUAL_SALARY_USD", "170000")
)
_YEAR_DASH_PATTERN = r"(?:-|to|–|—)"
_HOURLY_RANGE_PATTERN = re.compile(
    rf"\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?\s*{_YEAR_DASH_PATTERN}\s*\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?[^$\n]{{0,20}}(?:/ ?hr|per hour|hourly)",
    re.IGNORECASE,
)
_HOURLY_SINGLE_PATTERN = re.compile(
    r"\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?[^$\n]{0,12}(?:/ ?hr|per hour|hourly)",
    re.IGNORECASE,
)
_ANNUAL_RANGE_PATTERN = re.compile(
    rf"\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?\s*{_YEAR_DASH_PATTERN}\s*\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?[^$\n]{{0,20}}(?:/ ?yr|/ ?year|per year|annually|annual|yearly)",
    re.IGNORECASE,
)
_ANNUAL_SINGLE_PATTERN = re.compile(
    r"\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?[^$\n]{0,12}(?:/ ?yr|/ ?year|per year|annually|annual|yearly)",
    re.IGNORECASE,
)
_ANNUAL_CONTEXT_RANGE_PATTERN = re.compile(
    rf"(?:salary range|compensation range|salary|compensation|base compensation|base)[^$\n]{{0,35}}\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?\s*{_YEAR_DASH_PATTERN}\s*\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?",
    re.IGNORECASE,
)

CODEX_CONDA_SCRIPT = os.environ.get(
    "AUTO_APPLIER_CONDA_SCRIPT",
    "",
)
CODEX_CONDA_ENV = os.environ.get("AUTO_APPLIER_CODEX_CONDA_ENV", "")
CODEX_MODEL = os.environ.get("AUTO_APPLIER_CODEX_MODEL", "").strip()
AI_CLI_TIMEOUT_SECONDS = int(os.environ.get("AUTO_APPLIER_AI_CLI_TIMEOUT_SECONDS", 90))
AI_CLI_MIN_CONFIDENCE = float(
    os.environ.get("AUTO_APPLIER_AI_CLI_MIN_CONFIDENCE", 0.55)
)
_AI_CLI_CACHE: dict[str, str | None] = {}


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _parse_money_amount(raw_value: str, suffix: str | None) -> float:
    value = float(raw_value.replace(",", ""))
    if suffix and suffix.lower() == "k":
        value *= 1000
    return value


def _looks_like_hourly_range(
    low: float, high: float, low_suffix: str | None, high_suffix: str | None
) -> bool:
    return not low_suffix and not high_suffix and max(low, high) < 1000


def _compensation_upper_bound_usd(text: str) -> tuple[float | None, str | None]:
    normalized_text = text.replace("–", "-").replace("—", "-")

    for pattern in (
        _ANNUAL_RANGE_PATTERN,
        _ANNUAL_CONTEXT_RANGE_PATTERN,
    ):
        match = pattern.search(normalized_text)
        if not match:
            continue
        low = _parse_money_amount(match.group(1), match.group(2))
        high = _parse_money_amount(match.group(3), match.group(4))
        if pattern is _ANNUAL_CONTEXT_RANGE_PATTERN and _looks_like_hourly_range(
            low, high, match.group(2), match.group(4)
        ):
            return max(low, high) * 2080, match.group(0)
        return max(low, high), match.group(0)

    match = _HOURLY_RANGE_PATTERN.search(normalized_text)
    if match:
        low = _parse_money_amount(match.group(1), match.group(2))
        high = _parse_money_amount(match.group(3), match.group(4))
        return max(low, high) * 2080, match.group(0)

    for pattern, multiplier in (
        (_ANNUAL_SINGLE_PATTERN, 1),
        (_HOURLY_SINGLE_PATTERN, 2080),
    ):
        match = pattern.search(normalized_text)
        if not match:
            continue
        amount = _parse_money_amount(match.group(1), match.group(2)) * multiplier
        return amount, match.group(0)

    return None, None


def _is_generic_saas_without_network_domain(
    title: str, company: str, description: str
) -> bool:
    searchable_text = f"{title}\n{company}\n{description[:5000]}".lower()
    return _contains_any(searchable_text, GENERIC_SAAS_KEYWORDS) and not _contains_any(
        searchable_text, NETWORK_DOMAIN_KEYWORDS
    )


def _basic_saas_salary_auto_apply_reason(
    title: str, company: str, description: str
) -> str | None:
    upper_bound, snippet = _compensation_upper_bound_usd(f"{title}\n{description}")
    if upper_bound is None or upper_bound >= NETWORKING_MIN_ANNUAL_SALARY_USD:
        return None
    if not _is_generic_saas_without_network_domain(title, company, description):
        return None
    return (
        f'Explicit compensation "{snippet}" tops out below '
        f"${NETWORKING_MIN_ANNUAL_SALARY_USD:,}; generic SaaS/software role without "
        "a protected niche domain, so do not spend networking effort."
    )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_codex_cli(
    prompt: str, timeout_seconds: int = AI_CLI_TIMEOUT_SECONDS
) -> str | None:
    if _truthy_env("AUTO_APPLIER_DISABLE_AI_CLI"):
        return None
    if prompt in _AI_CLI_CACHE:
        return _AI_CLI_CACHE[prompt]

    output_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as output_file:
            output_path = output_file.name

        model_arg = f" --model {shlex.quote(CODEX_MODEL)}" if CODEX_MODEL else ""
        setup_parts = []
        if CODEX_CONDA_SCRIPT:
            setup_parts.append(f"source {shlex.quote(CODEX_CONDA_SCRIPT)}")
        if CODEX_CONDA_ENV:
            setup_parts.append(f"conda activate {shlex.quote(CODEX_CONDA_ENV)}")
        setup_prefix = " && ".join(setup_parts)
        if setup_prefix:
            setup_prefix += " && "

        command = (
            setup_prefix
            + "codex exec --ignore-user-config --ignore-rules --ephemeral "
            f"--skip-git-repo-check --sandbox read-only{model_arg} "
            f"--output-last-message {shlex.quote(output_path)} -"
        )
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            cwd=os.getcwd(),
            env={**os.environ, "NO_COLOR": "1"},
            check=False,
        )
        result = ""
        if output_path and os.path.exists(output_path):
            with open(output_path, encoding="utf-8") as file:
                result = file.read().strip()
        if completed.returncode != 0 or not result:
            print_lg(
                "Codex CLI call failed; using deterministic safety fallback.",
                (completed.stderr or completed.stdout or "")[-1200:],
            )
            _AI_CLI_CACHE[prompt] = None
            return None
        _AI_CLI_CACHE[prompt] = result
        return result
    except subprocess.TimeoutExpired:
        print_lg("Codex CLI call timed out; using deterministic safety fallback.")
        _AI_CLI_CACHE[prompt] = None
        return None
    except Exception as e:
        print_lg("Codex CLI call failed; using deterministic safety fallback.", e)
        _AI_CLI_CACHE[prompt] = None
        return None
    finally:
        if output_path:
            try:
                os.unlink(output_path)
            except OSError:
                pass


def has_auto_apply_recruiter_signal(company: str, description: str) -> bool:
    company_lower = company.lower()
    description_lower = description[:3000].lower()
    combined_text = f"{company_lower}\n{description_lower}"
    if _contains_any(company_lower, AUTO_APPLY_COMPANY_KEYWORDS):
        return True
    if _contains_any(combined_text, AUTO_APPLY_DESCRIPTION_KEYWORDS):
        return True
    return _contains_any(
        company_lower, AUTO_APPLY_COMPANY_CONTEXT_KEYWORDS
    ) and _contains_any(description_lower, AUTO_APPLY_DESCRIPTION_KEYWORDS)


def _requires_existing_clearance(text: str) -> bool:
    if ACTIVE_CLEARANCE_REQUIRED_PATTERN.search(text):
        return True

    for match in CLEARANCE_REQUIREMENT_TERM_PATTERN.finditer(text):
        window = text[max(0, match.start() - 80) : match.end() + 80]
        if CLEARANCE_OBTAINABLE_PATTERN.search(window):
            continue
        if REQUIRED_CONTEXT_PATTERN.search(window):
            return True
    return False


def _requires_unsupported_advanced_degree(text: str) -> bool:
    for match in ADVANCED_DEGREE_PATTERN.finditer(text):
        window = text[max(0, match.start() - 120) : match.end() + 120]
        if ADVANCED_DEGREE_SAFE_ALTERNATIVE_PATTERN.search(window):
            continue
        if REQUIRED_CONTEXT_PATTERN.search(window):
            return True
    return False


def _save_model_decision(
    *,
    job_id: str,
    title: str,
    company: str,
    job_link: str,
    decision: str,
    confidence: float | None,
    reason: str,
    source: str,
    raw_response: str | None,
    description: str,
) -> None:
    if not job_id.strip() and not job_link.strip():
        return

    try:
        os.makedirs(os.path.dirname(MODEL_DECISIONS_FILE), exist_ok=True)
        with open(MODEL_DECISIONS_FILE, "a", newline="", encoding="utf-8") as f:
            fieldnames = [
                "Recorded At",
                "Job ID",
                "Title",
                "Company",
                "Decision",
                "Confidence",
                "Reason",
                "Source",
                "Job Link",
                "Description Hash",
                "Description Snippet",
                "Raw Response",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "Recorded At": datetime.now(),
                    "Job ID": job_id,
                    "Title": title,
                    "Company": company,
                    "Decision": decision,
                    "Confidence": "" if confidence is None else f"{confidence:.2f}",
                    "Reason": reason,
                    "Source": source,
                    "Job Link": job_link,
                    "Description Hash": sha256(
                        description.encode("utf-8", errors="ignore")
                    ).hexdigest(),
                    "Description Snippet": " ".join(description[:500].split()),
                    "Raw Response": raw_response or "",
                }
            )
    except Exception as e:
        print_lg("Failed to save model decision audit row.", e)


def _heuristic_classification(title: str, company: str, description: str) -> str | None:
    # Recruiter/vendor signals intentionally win before domain networking checks.
    if has_auto_apply_recruiter_signal(company, description):
        return "auto_apply"

    searchable_text = f"{title}\n{company}\n{description[:5000]}".lower()
    if _contains_any(searchable_text, NETWORK_DOMAIN_KEYWORDS):
        return "network"

    return None


def hard_reject_reason(title: str, company: str, description: str) -> str | None:
    searchable_text = f"{title}\n{company}\n{description[:5000]}"
    title_text = title.lower()

    if LEGAL_ROLE_PATTERN.search(searchable_text):
        return "Legal/attorney role; the applicant does not have a J.D. or bar admission."
    if EXACT_EMPLOYER_REQUIREMENT_PATTERN.search(searchable_text):
        return "Employer-specific background is required and is not in the applicant profile."
    if _requires_existing_clearance(searchable_text):
        return "Active/current clearance or polygraph appears required."
    if _requires_unsupported_advanced_degree(searchable_text):
        return "Strict advanced degree requirement is not in the applicant profile."
    if LICENSED_ROLE_PATTERN.search(searchable_text):
        return "Licensed/certified role appears to require credentials not in the applicant profile."
    if CSHARP_DOTNET_PATTERN.search(title) or (
        CSHARP_DOTNET_PATTERN.search(description[:1500])
        and "senior software engineer" in title_text
    ):
        return "C#/.NET appears to be a core requirement, not a configured strength."
    if EMBEDDED_FLIGHT_ROLE_PATTERN.search(title):
        return "Embedded/firmware/flight software role; avoid low-fit specialist applications."
    if PEPTIDE_BIOLOGY_PATTERN.search(
        searchable_text
    ) and PHD_REQUIREMENT_PATTERN.search(searchable_text):
        return "Computational biology/protein-design role requiring a Ph.D."
    return None


def classify_job(
    title: str,
    company: str,
    description: str,
    job_id: str = "",
    job_link: str = "",
) -> str:
    """
    Classify a job as 'network' or 'auto_apply'.

    Uses Codex CLI in headless mode for the primary decision. The deterministic
    path is only a safety fallback if the CLI is unavailable or returns invalid
    JSON.
    """
    salary_override_reason = _basic_saas_salary_auto_apply_reason(
        title, company, description
    )
    if salary_override_reason:
        _save_model_decision(
            job_id=job_id,
            title=title,
            company=company,
            job_link=job_link,
            decision="auto_apply",
            confidence=None,
            reason=salary_override_reason,
            source="deterministic_salary_policy",
            raw_response=None,
            description=description,
        )
        print_lg(
            f"Salary/networking policy decision: auto_apply. {salary_override_reason}"
        )
        return "auto_apply"

    prompt = textwrap.dedent(
        f"""
        You decide how the applicant's LinkedIn job bot should handle one job.
        Return ONLY valid JSON with this exact shape:
        {{"decision":"auto_apply|network","confidence":0.0,"reason":"short reason"}}

        Decision policy:
        - "auto_apply" means the bot may apply on LinkedIn if the application flow is otherwise safe.
        - "network" means save for the applicant to pursue with a referral/outreach instead of auto-applying.
        - Recruiter, staffing, talent agency, consulting-vendor, or "our client" postings should usually be auto_apply even if the client/domain sounds interesting.
        - Direct-employer roles should be network when the applicant likely has a special edge worth preserving for outreach.
        - Generic roles that match the configured search criteria should usually be auto_apply.
        - If explicit compensation tops out below the configured networking salary floor and the role is generic, auto_apply instead of network.
        - Do not decide from title alone. Use title, company, and description together.
        - Salary-floor and years-of-experience filters are handled elsewhere.

        Job title: {title}
        Company: {company}
        Description:
        {description[:7000]}
        """
    ).strip()
    response = _run_codex_cli(prompt)
    parsed = _extract_json_object(response or "")
    if parsed:
        decision = str(parsed.get("decision", "")).strip().lower()
        try:
            confidence = float(parsed.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(parsed.get("reason", "")).strip()
        if (
            decision in {"auto_apply", "network"}
            and confidence >= AI_CLI_MIN_CONFIDENCE
        ):
            _save_model_decision(
                job_id=job_id,
                title=title,
                company=company,
                job_link=job_link,
                decision=decision,
                confidence=confidence,
                reason=reason,
                source="codex_cli",
                raw_response=response,
                description=description,
            )
            print_lg(
                f"Codex job policy decision: {decision}"
                f" (confidence={confidence:.2f}). Reason: {reason}"
            )
            return decision
        print_lg(
            "Codex job policy response was low-confidence or invalid; using fallback.",
            parsed,
        )
    elif response:
        print_lg(
            "Codex job policy response was not valid JSON; using fallback.", response
        )

    heuristic = _heuristic_classification(title, company, description)
    if heuristic:
        _save_model_decision(
            job_id=job_id,
            title=title,
            company=company,
            job_link=job_link,
            decision=heuristic,
            confidence=None,
            reason="Codex CLI unavailable or invalid; deterministic safety fallback.",
            source="deterministic_fallback",
            raw_response=response,
            description=description,
        )
        return heuristic
    _save_model_decision(
        job_id=job_id,
        title=title,
        company=company,
        job_link=job_link,
        decision="auto_apply",
        confidence=None,
        reason="Codex CLI unavailable or invalid; deterministic default.",
        source="deterministic_fallback",
        raw_response=response,
        description=description,
    )
    return "auto_apply"


def answer_simple_question(
    question: str,
    user_context: str,
    options: list[str] | None = None,
    max_words: int = 10,
) -> str | None:
    """
    Use Codex CLI for simple, inferable application questions only.

    Returns:
    - exact option text for select/radio questions
    - short free-text answer for simple text questions
    - None when the answer is not safely inferable
    """
    options_block = ""
    if options:
        options_block = (
            "\nOptions, choose one exactly if safely inferable:\n"
            + "\n".join(f"- {option}" for option in options)
        )
    prompt = textwrap.dedent(
        f"""
        Answer a simple LinkedIn job application question for the applicant.
        Return ONLY valid JSON with this exact shape:
        {{"answer":null,"confidence":0.0,"reason":"short reason"}}

        Rules:
        - Use only the user context below and the question/options.
        - If the answer is not safely inferable, set answer to null.
        - For radio/select options, answer must be one of the provided options exactly.
        - For free text, answer must be at most {max_words} words and should not sound like a cover letter.

        User context:
        {user_context}

        Question:
        {question}
        {options_block}
        """
    ).strip()

    response = _run_codex_cli(prompt)
    parsed = _extract_json_object(response or "")
    if not parsed:
        if response:
            print_lg("Codex short-answer response was not valid JSON.", response)
        return None

    result = parsed.get("answer")
    if result is None:
        return None
    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < AI_CLI_MIN_CONFIDENCE:
        return None

    result = str(result).strip().strip('"').strip("'")
    if not result:
        return None

    if options:
        lowered_result = result.lower()
        for option in options:
            if lowered_result == option.lower():
                return option
        for option in options:
            option_lower = option.lower()
            if lowered_result in option_lower or option_lower in lowered_result:
                return option
        return None

    cleaned = " ".join(result.split())
    if len(cleaned.split()) > max_words:
        return None
    return cleaned


def answer_freeform_question(
    question: str,
    user_context: str,
    job_context: str,
    max_words: int = 140,
) -> str | None:
    """
    Use Codex CLI for longer free-text application questions.

    Returns None when the prompt asks for information that is not safely inferable.
    """
    prompt = textwrap.dedent(
        f"""
        Draft a concise answer to a LinkedIn Easy Apply free-text question for the applicant.
        Return ONLY valid JSON with this exact shape:
        {{"answer":null,"confidence":0.0,"reason":"short reason"}}

        Rules:
        - Use only the user context and job context below.
        - If the answer would require a fact not present in the context, set answer to null.
        - Do not invent degrees, employment history, certifications, work authorization, salary,
          availability, personal demographics, or links.
        - For work sample / writing sample requests, answer null unless a specific link in context
          directly fits.
        - Keep non-null answers under {max_words} words.
        - Write in first person, direct and professional, with no greeting or signature.

        User context:
        {user_context}

        Job context:
        {job_context[:5000]}

        Question:
        {question}
        """
    ).strip()

    response = _run_codex_cli(prompt)
    parsed = _extract_json_object(response or "")
    if not parsed:
        if response:
            print_lg("Codex freeform-answer response was not valid JSON.", response)
        return None

    result = parsed.get("answer")
    if result is None:
        return None
    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < AI_CLI_MIN_CONFIDENCE:
        return None

    cleaned = str(result).strip().strip('"').strip("'")
    if not cleaned:
        return None
    if len(cleaned.split()) > max_words:
        return None
    return cleaned


def _existing_networking_job_ids() -> set[str]:
    if not os.path.exists(NETWORKING_JOBS_FILE):
        return set()

    with open(NETWORKING_JOBS_FILE, newline="", encoding="utf-8") as f:
        return {
            row["Job ID"].strip()
            for row in csv.DictReader(f)
            if row.get("Job ID", "").strip()
        }


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
) -> None:
    """Save a job flagged for networking to a separate CSV."""
    try:
        if job_id in _existing_networking_job_ids():
            print_lg(
                f'Skipping duplicate networking job "{title} | {company}". Job ID: {job_id}'
            )
            return

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
                    "Date Saved": datetime.now(),
                }
            )
        print_lg(f'Saved "{title} | {company}" for networking.')
    except Exception as e:
        print_lg("Failed to save networking job!", e)
