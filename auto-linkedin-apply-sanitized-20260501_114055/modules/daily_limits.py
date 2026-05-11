import os
import csv
from datetime import date, datetime, timedelta


def _resolve_positive_int_env(env_names: tuple[str, ...], default: int) -> int:
    raw_value = ""
    resolved_name = env_names[0]
    for env_name in env_names:
        raw_value = os.environ.get(env_name, "").strip()
        if raw_value:
            resolved_name = env_name
            break

    if not raw_value:
        return default

    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{resolved_name} must be >= 1")
    return value


def resolve_max_daily_linkedin_applications(default: int = 0) -> int:
    return _resolve_positive_int_env(
        (
            "AUTO_APPLIER_MAX_DAILY_LINKEDIN_APPLICATIONS",
            "AUTO_APPLIER_MAX_DAILY_APPLICATIONS",
            "AUTO_APPLIER_MAX_DAILY_EASY_APPLIES",
        ),
        default,
    )


def resolve_max_run_linkedin_applications(default: int = 0) -> int:
    return _resolve_positive_int_env(
        ("AUTO_APPLIER_MAX_RUN_LINKEDIN_APPLICATIONS",), default
    )


def resolve_max_result_pages_per_search(default: int = 0) -> int:
    return _resolve_positive_int_env(
        ("AUTO_APPLIER_MAX_RESULT_PAGES_PER_SEARCH",), default
    )


def resolve_daily_application_window_hours(default: int = 24) -> int:
    return _resolve_positive_int_env(
        ("AUTO_APPLIER_DAILY_LIMIT_WINDOW_HOURS",), default
    )


def daily_linkedin_application_cap_reached(
    linkedin_application_count: int, max_daily_linkedin_applications: int
) -> bool:
    return (
        max_daily_linkedin_applications > 0
        and linkedin_application_count >= max_daily_linkedin_applications
    )


def looks_like_linkedin_application_marker(application_link: str) -> bool:
    marker = application_link.strip().lower()
    if not marker:
        return True
    if "linkedin.com" in marker:
        return True
    if marker.startswith(("http://", "https://")):
        return False
    return True


def count_todays_linkedin_applications(
    history_path: str, today: date | None = None
) -> int:
    today = today or date.today()
    if not os.path.exists(history_path):
        return 0

    count = 0
    with open(history_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if not looks_like_linkedin_application_marker(
                row.get("External Job link") or ""
            ):
                continue

            raw_date = (row.get("Date Applied") or "").strip()
            if not raw_date or raw_date == "Pending":
                continue

            try:
                applied_at = datetime.fromisoformat(raw_date)
            except ValueError:
                continue

            if applied_at.date() == today:
                count += 1
    return count


def _todays_linkedin_application_ids(
    history_path: str, today: date | None = None
) -> set[str]:
    today = today or date.today()
    if not os.path.exists(history_path):
        return set()

    job_ids = set()
    with open(history_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if not looks_like_linkedin_application_marker(
                row.get("External Job link") or ""
            ):
                continue

            raw_date = (row.get("Date Applied") or "").strip()
            if not raw_date or raw_date == "Pending":
                continue

            try:
                applied_at = datetime.fromisoformat(raw_date)
            except ValueError:
                continue

            if applied_at.date() == today and row.get("Job ID"):
                job_ids.add(row["Job ID"])
    return job_ids


def _todays_recorded_linkedin_application_ids(
    actions_path: str, today: date | None = None
) -> set[str]:
    today = today or date.today()
    if not os.path.exists(actions_path):
        return set()

    job_ids = set()
    with open(actions_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            raw_date = (row.get("Recorded At") or "").strip()
            if not raw_date:
                continue

            try:
                recorded_at = datetime.fromisoformat(raw_date)
            except ValueError:
                continue

            if recorded_at.date() == today and row.get("Job ID"):
                job_ids.add(row["Job ID"])
    return job_ids


def _recent_linkedin_application_ids(
    history_path: str, cutoff: datetime, now: datetime
) -> set[str]:
    if not os.path.exists(history_path):
        return set()

    job_ids = set()
    with open(history_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if not looks_like_linkedin_application_marker(
                row.get("External Job link") or ""
            ):
                continue

            raw_date = (row.get("Date Applied") or "").strip()
            if not raw_date or raw_date == "Pending":
                continue

            try:
                applied_at = datetime.fromisoformat(raw_date)
            except ValueError:
                continue

            if cutoff <= applied_at <= now and row.get("Job ID"):
                job_ids.add(row["Job ID"])
    return job_ids


def _recent_recorded_linkedin_application_ids(
    actions_path: str, cutoff: datetime, now: datetime
) -> set[str]:
    if not os.path.exists(actions_path):
        return set()

    job_ids = set()
    with open(actions_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            raw_date = (row.get("Recorded At") or "").strip()
            if not raw_date:
                continue

            try:
                recorded_at = datetime.fromisoformat(raw_date)
            except ValueError:
                continue

            if cutoff <= recorded_at <= now and row.get("Job ID"):
                job_ids.add(row["Job ID"])
    return job_ids


def count_recent_linkedin_application_actions(
    history_path: str,
    actions_path: str,
    now: datetime | None = None,
    window_hours: int = 24,
) -> int:
    now = now or datetime.now()
    cutoff = now - timedelta(hours=window_hours)
    job_ids = _recent_linkedin_application_ids(history_path, cutoff, now)
    job_ids.update(_recent_recorded_linkedin_application_ids(actions_path, cutoff, now))
    return len(job_ids)


def count_todays_linkedin_application_actions(
    history_path: str, actions_path: str, today: date | None = None
) -> int:
    job_ids = _todays_linkedin_application_ids(history_path, today=today)
    job_ids.update(_todays_recorded_linkedin_application_ids(actions_path, today=today))
    return len(job_ids)


def record_linkedin_application_action(
    actions_path: str,
    job_id: str,
    title: str,
    company: str,
    recorded_at: datetime | None = None,
) -> None:
    recorded_at = recorded_at or datetime.now()
    fieldnames = ["Recorded At", "Job ID", "Title", "Company"]
    file_exists = os.path.exists(actions_path)
    os.makedirs(os.path.dirname(actions_path), exist_ok=True)
    with open(actions_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "Recorded At": recorded_at,
                "Job ID": job_id,
                "Title": title,
                "Company": company,
            }
        )
