from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


LINKEDIN_HOSTS = ("linkedin.com", "lnkd.in")


@dataclass(frozen=True)
class AtsPlatform:
    key: str
    label: str
    host_fragments: tuple[str, ...]


ATS_PLATFORMS = (
    AtsPlatform(
        "greenhouse",
        "Greenhouse",
        ("greenhouse.io", "job-boards.greenhouse.io", "boards.greenhouse.io"),
    ),
    AtsPlatform("lever", "Lever", ("lever.co", "jobs.lever.co", "postings.lever.co")),
    AtsPlatform("ashby", "Ashby", ("ashbyhq.com", "jobs.ashbyhq.com")),
    AtsPlatform("workable", "Workable", ("workable.com", "jobs.workable.com")),
    AtsPlatform("smartrecruiters", "SmartRecruiters", ("smartrecruiters.com",)),
    AtsPlatform("breezy", "Breezy HR", ("breezy.hr",)),
    AtsPlatform(
        "workday",
        "Workday",
        ("workdayjobs.com", "myworkdayjobs.com", "myworkdaysite.com"),
    ),
    AtsPlatform("icims", "iCIMS", ("icims.com",)),
    AtsPlatform("ukg", "UKG", ("ultipro.com", "ukg.com")),
    AtsPlatform("taleo", "Taleo", ("taleo.net",)),
    AtsPlatform("jobvite", "Jobvite", ("jobvite.com",)),
    AtsPlatform("paylocity", "Paylocity", ("paylocity.com",)),
    AtsPlatform("successfactors", "SAP SuccessFactors", ("successfactors.com",)),
    AtsPlatform("rippling", "Rippling", ("rippling.com",)),
    AtsPlatform("bamboohr", "BambooHR", ("bamboohr.com",)),
    AtsPlatform("jazzhr", "JazzHR", ("applytojob.com", "jazz.co")),
    AtsPlatform("pinpoint", "Pinpoint", ("pinpointhq.com",)),
    AtsPlatform("recruitee", "Recruitee", ("recruitee.com",)),
    AtsPlatform("amazon_jobs", "Amazon Jobs", ("amazon.jobs",)),
    AtsPlatform(
        "microsoft_careers", "Microsoft Careers", ("jobs.careers.microsoft.com",)
    ),
    AtsPlatform("tiktok_careers", "TikTok Careers", ("lifeattiktok.com",)),
    AtsPlatform("deloitte_careers", "Deloitte Careers", ("apply.deloitte.com",)),
    AtsPlatform("ey_careers", "EY Careers", ("careers.ey.com",)),
    AtsPlatform("wellfound", "Wellfound", ("wellfound.com",)),
    AtsPlatform("cybercoders", "CyberCoders", ("cybercoders.com",)),
    AtsPlatform("recruitcrm", "Recruit CRM", ("recruitcrm.io",)),
    AtsPlatform("remotehunter", "RemoteHunter", ("remotehunter.com",)),
    AtsPlatform("7seventy", "7Seventy", ("7seventy.net",)),
)

PLATFORM_LABELS = {platform.key: platform.label for platform in ATS_PLATFORMS}
PLATFORM_LABELS.update(
    {"linkedin": "LinkedIn", "unknown": "Unknown ATS", "invalid": "Invalid URL"}
)
ATS_QUERY_PARAMETER_PLATFORMS = {
    "ashby_jid": "ashby",
    "gh_jid": "greenhouse",
}

EXTERNAL_ATS_REVIEW_COLUMNS = (
    "Job ID",
    "Title",
    "Company",
    "Platform",
    "Platform Key",
    "Needs Recapture",
    "Resolved ATS URL",
    "Saved Application Link",
    "LinkedIn Job Link",
    "Date Saved",
)


@dataclass(frozen=True)
class ExternalApplyJob:
    job_id: str
    title: str
    company: str
    job_link: str
    application_link: str
    platform: str
    date_saved: str


def normalize_hostname(url: str) -> str:
    try:
        hostname = urlsplit(url.strip()).hostname or ""
    except ValueError:
        return ""
    return hostname.lower().removeprefix("www.")


def is_linkedin_url(url: str) -> bool:
    hostname = normalize_hostname(url)
    return any(
        hostname == host or hostname.endswith(f".{host}") for host in LINKEDIN_HOSTS
    )


def _is_direct_external_apply_url(url: str, linkedin_job_url: str = "") -> bool:
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    if is_linkedin_url(url):
        return False
    return url.strip().lower() != linkedin_job_url.strip().lower()


def extract_real_external_apply_url(url: str, linkedin_job_url: str = "") -> str | None:
    stripped_url = url.strip()
    if _is_direct_external_apply_url(stripped_url, linkedin_job_url):
        return stripped_url

    try:
        parsed = urlsplit(stripped_url)
    except ValueError:
        return None

    for parameter in ("url", "u", "target"):
        for candidate in parse_qs(parsed.query).get(parameter, []):
            if _is_direct_external_apply_url(candidate, linkedin_job_url):
                return candidate.strip()
    return None


def is_real_external_apply_url(url: str, linkedin_job_url: str = "") -> bool:
    return extract_real_external_apply_url(url, linkedin_job_url) is not None


def classify_ats_platform(url: str) -> str:
    if not url.strip():
        return "invalid"
    real_url = extract_real_external_apply_url(url)
    if real_url:
        url = real_url
    if is_linkedin_url(url):
        return "linkedin"
    hostname = normalize_hostname(url)
    if not hostname:
        return "invalid"
    try:
        query_parameters = parse_qs(urlsplit(url.strip()).query)
    except ValueError:
        query_parameters = {}
    for parameter, platform_key in ATS_QUERY_PARAMETER_PLATFORMS.items():
        if parameter in query_parameters:
            return platform_key
    for platform in ATS_PLATFORMS:
        if any(
            hostname == fragment or hostname.endswith(f".{fragment}")
            for fragment in platform.host_fragments
        ):
            return platform.key
    return "unknown"


def platform_label(platform_key: str) -> str:
    return PLATFORM_LABELS.get(platform_key, platform_key)


def external_url_needs_recapture(url: str) -> bool:
    return classify_ats_platform(url) in {"invalid", "linkedin"}


def load_external_apply_jobs(csv_path: str | Path) -> list[ExternalApplyJob]:
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("Backlog Type") == "external_apply"
        ]

    jobs = []
    for row in rows:
        application_link = row.get("Application Link", "").strip()
        jobs.append(
            ExternalApplyJob(
                job_id=row.get("Job ID", "").strip(),
                title=row.get("Title", "").strip(),
                company=row.get("Company", "").strip(),
                job_link=row.get("Job Link", "").strip(),
                application_link=application_link,
                platform=classify_ats_platform(application_link),
                date_saved=row.get("Date Saved", "").strip(),
            )
        )
    return jobs


def summarize_platforms(jobs: list[ExternalApplyJob]) -> Counter[str]:
    return Counter(job.platform for job in jobs)


def external_apply_review_rows(
    jobs: list[ExternalApplyJob],
) -> list[dict[str, str]]:
    rows = []
    for job in jobs:
        resolved_url = (
            extract_real_external_apply_url(job.application_link, job.job_link) or ""
        )
        platform = classify_ats_platform(resolved_url or job.application_link)
        rows.append(
            {
                "Job ID": job.job_id,
                "Title": job.title,
                "Company": job.company,
                "Platform": platform_label(platform),
                "Platform Key": platform,
                "Needs Recapture": (
                    "yes"
                    if external_url_needs_recapture(job.application_link)
                    else "no"
                ),
                "Resolved ATS URL": resolved_url,
                "Saved Application Link": job.application_link,
                "LinkedIn Job Link": job.job_link,
                "Date Saved": job.date_saved,
            }
        )
    return rows


def write_external_ats_review_csv(
    jobs: list[ExternalApplyJob], csv_path: str | Path
) -> list[dict[str, str]]:
    rows = external_apply_review_rows(jobs)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTERNAL_ATS_REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
