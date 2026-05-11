"""
Author:     Sai Vignesh Golla
LinkedIn:   https://www.linkedin.com/in/saivigneshgolla/

Copyright (C) 2024 Sai Vignesh Golla

License:    GNU Affero General Public License
            https://www.gnu.org/licenses/agpl-3.0.en.html

GitHub:     https://github.com/GodsScion/Auto_job_applier_linkedIn

version:    24.12.29.12.30
"""

# ruff: noqa: F403,F405

# Imports
import arm64_patch  # noqa: F401 - must be before pyautogui
import os
import csv
import re
import pyautogui
import time
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from random import choice, shuffle
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.select import Select
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import (
    NoSuchElementException,
    ElementClickInterceptedException,
    NoSuchWindowException,
    ElementNotInteractableException,
    StaleElementReferenceException,
    InvalidSessionIdException,
)

from config.personals import *
from config.questions import *
from config.search import *
from config.secrets import use_AI, username, password
from config.settings import *

from modules.open_chrome import *
from modules.helpers import *
from modules.clickers_and_finders import *
from modules.daily_limits import (
    count_recent_linkedin_application_actions,
    daily_linkedin_application_cap_reached,
    looks_like_linkedin_application_marker,
    record_linkedin_application_action,
    resolve_daily_application_window_hours,
    resolve_max_daily_linkedin_applications,
    resolve_max_result_pages_per_search,
    resolve_max_run_linkedin_applications,
)
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
    is_specific_experience_label,
    manual_review_reason_for_question,
    recommended_compensation_answer,
    should_override_existing_answer,
)
from modules.easy_apply_question_log import append_easy_apply_question_log
from modules.experience_requirements import estimate_required_years
from modules.external_ats import (
    extract_real_external_apply_url,
    is_real_external_apply_url,
)
from modules.profile_context import load_profile_context
from modules.question_bank import QuestionBank, normalize_question_label, option_kind
from modules.validator import validate_config
from modules.ai.openaiConnections import *
from modules.ai.cli_classifier import (
    answer_freeform_question,
    answer_simple_question,
    classify_job,
    hard_reject_reason,
    has_auto_apply_recruiter_signal,
    save_networking_job,
)

from typing import Literal


pyautogui.FAILSAFE = False
# if use_resume_generator:    from resume_generator import is_logged_in_GPT, login_GPT, open_resume_chat, create_custom_resume


def desktop_alerts_enabled() -> bool:
    return os.environ.get("AUTO_APPLIER_SUPPRESS_ALERTS", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def browser_session_lost(error: Exception) -> bool:
    return isinstance(error, (NoSuchWindowException, InvalidSessionIdException)) or (
        "invalid session id" in str(error).lower()
    )


# < Global Variables and logics

if run_in_background:
    pause_at_failed_question = False
    pause_before_submit = False
    run_non_stop = False

first_name = first_name.strip()
middle_name = middle_name.strip()
last_name = last_name.strip()
full_name = (
    first_name + " " + middle_name + " " + last_name
    if middle_name
    else first_name + " " + last_name
)

useNewResume = True
randomly_answered_questions = set()
MANUAL_WRITEUP_JOBS_FILE = "all excels/manual_writeup_jobs.csv"
NETWORKING_JOBS_FILE = "all excels/networking_jobs.csv"
LINKEDIN_APPLICATION_ACTIONS_FILE = "all excels/linkedin_application_actions.csv"
EASY_APPLY_QUESTIONS_FILE = "all excels/easy_apply_questions_seen.csv"
MAX_DAILY_LINKEDIN_APPLICATIONS = resolve_max_daily_linkedin_applications()
MAX_RUN_LINKEDIN_APPLICATIONS = resolve_max_run_linkedin_applications()
DAILY_LIMIT_WINDOW_HOURS = resolve_daily_application_window_hours()
MAX_RESULT_PAGES_PER_SEARCH = resolve_max_result_pages_per_search(default=2)
MANUAL_WRITEUP_PLACEHOLDER_TEXT = "Will personalize before submitting."
MANUAL_WRITEUP_LABEL_KEYWORDS = (
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
    "work sample",
    "writing sample",
    "yourself",
)
QUESTION_BANK = QuestionBank.from_file()
DAILY_EASY_APPLY_LIMIT_PATTERNS = (
    "exceeded the daily application limit",
    "we limit the number of applications you can submit each day",
    "limit the number of applications you can submit each day",
    "you've exceeded the daily application limit",
    "we limit daily submissions to maintain quality and prevent bots",
    "save this job and apply tomorrow",
)

tabs_count = 1
easy_applied_count = 0
linkedin_applications_today_count = count_recent_linkedin_application_actions(
    file_name,
    LINKEDIN_APPLICATION_ACTIONS_FILE,
    window_hours=DAILY_LIMIT_WINDOW_HOURS,
)
linkedin_applications_this_run_count = 0
external_jobs_count = 0
failed_count = 0
skip_count = 0
manual_writeup_jobs_count = 0
dailyEasyApplyLimitReached = False
RESULTS_PAGE_SIZE = 25
MINIMUM_HOURLY_RATE_USD = round(minimum_annual_salary_usd / 2080, 2)


def configured_daily_linkedin_application_cap_reached() -> bool:
    return daily_linkedin_application_cap_reached(
        linkedin_applications_today_count, MAX_DAILY_LINKEDIN_APPLICATIONS
    ) or daily_linkedin_application_cap_reached(
        linkedin_applications_this_run_count, MAX_RUN_LINKEDIN_APPLICATIONS
    )


def mark_configured_daily_linkedin_application_cap_reached() -> None:
    global dailyEasyApplyLimitReached
    dailyEasyApplyLimitReached = True
    if daily_linkedin_application_cap_reached(
        linkedin_applications_this_run_count, MAX_RUN_LINKEDIN_APPLICATIONS
    ):
        print_lg(
            "\n###############  Configured per-run LinkedIn application cap reached"
            f" ({linkedin_applications_this_run_count}/{MAX_RUN_LINKEDIN_APPLICATIONS})."
            " Stopping run."
            "  ###############\n"
        )
        return
    print_lg(
        "\n###############  Configured daily LinkedIn application cap reached"
        f" ({linkedin_applications_today_count}/{MAX_DAILY_LINKEDIN_APPLICATIONS})."
        f" Window: last {DAILY_LIMIT_WINDOW_HOURS}h."
        " Stopping run."
        "  ###############\n"
    )


def record_linkedin_application_submitted(
    job_id: str, title: str, company: str, application_link: str
) -> None:
    global linkedin_applications_this_run_count, linkedin_applications_today_count
    if not looks_like_linkedin_application_marker(application_link):
        return
    record_linkedin_application_action(
        LINKEDIN_APPLICATION_ACTIONS_FILE, job_id, title, company
    )
    linkedin_applications_today_count += 1
    linkedin_applications_this_run_count += 1


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
    rf"(?:salary range|compensation range|salary|compensation)[^$\n]{{0,30}}\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?\s*{_YEAR_DASH_PATTERN}\s*\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?",
    re.IGNORECASE,
)

desired_salary_lakhs = str(round(desired_salary / 100000, 2))
desired_salary_monthly = str(round(desired_salary / 12, 2))
desired_salary = str(desired_salary)

current_ctc_lakhs = str(round(current_ctc / 100000, 2))
current_ctc_monthly = str(round(current_ctc / 12, 2))
current_ctc = str(current_ctc)

notice_period_months = str(notice_period // 30)
notice_period_weeks = str(notice_period // 7)
notice_period = str(notice_period)

aiClient = None
# >

PROFILE_CONTEXT = load_profile_context()

SIMPLE_QUESTION_USER_CONTEXT = "\n".join(
    [
        f"Full name: {full_name}",
        f"Location: {current_city}, {state}, {country}",
        f"Phone: {phone_number}",
        f"LinkedIn: {linkedIn}",
        f"Website: {website}",
        f"Headline: {linkedin_headline}",
        f"Summary: {linkedin_summary.strip()}",
        f"Years of experience: {years_of_experience}",
        f"Citizenship/work authorization: {us_citizenship}",
        f"Recent employer: {recent_employer}",
        f"Desired salary: ${desired_salary}",
        "Verified negative facts: no J.D., no law degree, not an attorney, no Ph.D.,"
        " no Palantir employment, no CSSLP certification, no active security clearance;"
        " no end-to-end ASR/TTS/speech or voice AI ownership, no commercial SAR/InSAR"
        " workflow ownership, no robotics/drone/UAV field exposure, no medical device"
        " manufacturing/SAP/regulated QMS experience, and no production payments or"
        " billing-system ownership; willing to obtain a clearance if required.",
        "Do not claim more than 5 full years of post-UVA professional experience as"
        " of April 2026.",
        PROFILE_CONTEXT,
    ]
)


# < Login Functions
def is_logged_in_LN() -> bool:
    """
    Function to check if user is logged-in in LinkedIn
    * Returns: `True` if user is logged-in or `False` if not
    """
    if driver.current_url == "https://www.linkedin.com/feed/":
        return True
    if try_linkText(driver, "Sign in"):
        return False
    if try_xp(driver, '//button[@type="submit" and contains(text(), "Sign in")]'):
        return False
    if try_linkText(driver, "Join now"):
        return False
    print_lg("Didn't find Sign in link, so assuming user is logged in!")
    return True


def login_LN() -> None:
    """
    Function to login for LinkedIn
    * Tries to login using given `username` and `password` from `secrets.py`
    * If failed, tries to login using saved LinkedIn profile button if available
    * If both failed, asks user to login manually
    """
    # Find the username and password fields and fill them with user credentials
    driver.get("https://www.linkedin.com/login")
    try:
        wait.until(EC.presence_of_element_located((By.LINK_TEXT, "Forgot password?")))
        try:
            text_input_by_ID(driver, "username", username, 1)
        except Exception:
            print_lg("Couldn't find username field.")
            # print_lg(e)
        try:
            text_input_by_ID(driver, "password", password, 1)
        except Exception:
            print_lg("Couldn't find password field.")
            # print_lg(e)
        # Find the login submit button and click it
        driver.find_element(
            By.XPATH, '//button[@type="submit" and contains(text(), "Sign in")]'
        ).click()
    except Exception:
        try:
            profile_button = find_by_class(driver, "profile__details")
            profile_button.click()
        except Exception:
            # print_lg(e1, e2)
            print_lg("Couldn't Login!")

    try:
        # Wait until successful redirect, indicating successful login
        WebDriverWait(driver, 15).until(
            EC.url_to_be("https://www.linkedin.com/feed/")
        )  # wait.until(EC.presence_of_element_located((By.XPATH, '//button[normalize-space(.)="Start a post"]')))
        return print_lg("Login successful!")
    except Exception:
        print_lg(
            "Seems like login attempt failed! Possibly due to wrong credentials or already logged"
            " in! Try logging in manually!"
        )
        # print_lg(e)
        if not manual_login_retry(is_logged_in_LN, 2):
            raise RuntimeError(
                "LinkedIn login could not be completed in the current browser session."
            )


# >


def get_logged_job_ids(csv_path: str) -> set[str]:
    job_ids = set()
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                job_id = row.get("Job ID", "").strip()
                if job_id:
                    job_ids.add(job_id)
    except FileNotFoundError:
        print_lg(f"The CSV file '{csv_path}' does not exist.")
    return job_ids


def get_applied_job_ids() -> set[str]:
    """
    Function to get a `set` of applied job's Job IDs
    * Returns a set of Job IDs from existing applied jobs history csv file
    """
    return get_logged_job_ids(file_name)


def get_manual_writeup_job_ids() -> set[str]:
    return get_logged_job_ids(MANUAL_WRITEUP_JOBS_FILE)


def get_networking_job_ids() -> set[str]:
    return get_logged_job_ids(NETWORKING_JOBS_FILE)


def needs_manual_writeup(label: str) -> bool:
    normalized_label = label.strip().lower()
    if not normalized_label or normalized_label == "unknown":
        return True
    return any(keyword in normalized_label for keyword in MANUAL_WRITEUP_LABEL_KEYWORDS)


def add_manual_writeup_reason(reasons: set[str], field_type: str, label: str) -> None:
    reasons.add(manual_writeup_reason(field_type, label))


def remove_manual_writeup_reason(
    reasons: set[str], field_type: str, label: str
) -> None:
    reasons.discard(manual_writeup_reason(field_type, label))


def add_question_record(
    questions_list: set,
    label: str,
    answer: object,
    field_type: str,
    previous_answer: object,
    answer_source: str,
) -> None:
    questions_list.add(
        (
            label.strip() if label else "Unknown",
            answer,
            field_type,
            previous_answer,
            answer_source,
        )
    )


def serialize_questions(questions_list: set | None) -> str:
    if not questions_list:
        return ""
    return "\n".join(sorted(str(question) for question in questions_list))


def build_job_question_context(title: str, company: str, description: str) -> str:
    return "\n".join(
        [
            f"Job title: {title}",
            f"Company: {company}",
            "Job description:",
            description[:5000],
        ]
    )


def save_easy_apply_questions_seen(
    questions_list: set | None,
    job_id: str,
    title: str,
    company: str,
    status: str,
) -> None:
    try:
        append_easy_apply_question_log(
            EASY_APPLY_QUESTIONS_FILE,
            questions_list,
            job_id,
            title,
            company,
            status,
        )
    except Exception as e:
        print_lg("Failed to update Easy Apply question log!", e)


def daily_easy_apply_limit_reached(scope: WebElement | WebDriver | None = None) -> bool:
    fragments = []
    if scope is not None:
        try:
            fragments.append(scope.text.lower())
        except Exception:
            pass
    try:
        fragments.append(driver.find_element(By.TAG_NAME, "body").text.lower())
    except Exception:
        pass

    haystack = "\n".join(fragment for fragment in fragments if fragment)
    return any(pattern in haystack for pattern in DAILY_EASY_APPLY_LIMIT_PATTERNS)


def set_search_location() -> None:
    """
    Function to set search location
    """
    if search_location.strip():
        try:
            print_lg(f'Setting search location as: "{search_location.strip()}"')
            search_location_ele = try_xp(
                driver,
                ".//input[@aria-label='City, state, or zip code'and not(@disabled)]",
                False,
            )  #  and not(@aria-hidden='true')]")
            text_input(actions, search_location_ele, search_location, "Search Location")
        except ElementNotInteractableException:
            try_xp(
                driver,
                ".//label[@class='jobs-search-box__input-icon jobs-search-box__keywords-label']",
            )
            actions.send_keys(Keys.TAB, Keys.TAB).perform()
            actions.key_down(Keys.CONTROL).send_keys("a").key_up(Keys.CONTROL).perform()
            actions.send_keys(search_location.strip()).perform()
            sleep(2)
            actions.send_keys(Keys.ENTER).perform()
            try_xp(driver, ".//button[@aria-label='Cancel']")
        except Exception as e:
            try_xp(driver, ".//button[@aria-label='Cancel']")
            print_lg(
                "Failed to update search location, continuing with default location!", e
            )


def apply_filters() -> None:
    """
    Function to apply job search filters
    """
    set_search_location()

    try:
        recommended_wait = 1 if click_gap < 1 else 0

        wait.until(
            EC.presence_of_element_located(
                (By.XPATH, '//button[normalize-space()="All filters"]')
            )
        ).click()
        buffer(recommended_wait)

        wait_span_click(driver, sort_by)
        wait_span_click(driver, "Filter by " + date_posted)
        buffer(recommended_wait)

        multi_sel_noWait(driver, ["Filter by " + i for i in experience_level])
        multi_sel_noWait(driver, companies, actions)
        if experience_level or companies:
            buffer(recommended_wait)

        multi_sel_noWait(driver, ["Filter by " + i for i in job_type])
        multi_sel_noWait(driver, ["Filter by " + i for i in on_site])
        if job_type or on_site:
            buffer(recommended_wait)

        if easy_apply_only:
            boolean_button_click(driver, actions, "Easy Apply")

        multi_sel_noWait(driver, location)
        multi_sel_noWait(driver, industry)
        if location or industry:
            buffer(recommended_wait)

        multi_sel_noWait(driver, job_function)
        multi_sel_noWait(driver, job_titles)
        if job_function or job_titles:
            buffer(recommended_wait)

        if under_10_applicants:
            boolean_button_click(driver, actions, "Under 10 applicants")
        if in_your_network:
            boolean_button_click(driver, actions, "In your network")
        if fair_chance_employer:
            boolean_button_click(driver, actions, "Fair Chance Employer")

        wait_span_click(driver, salary)
        buffer(recommended_wait)

        multi_sel_noWait(driver, benefits)
        multi_sel_noWait(driver, commitments)
        if benefits or commitments:
            buffer(recommended_wait)

        show_results_button: WebElement = driver.find_element(
            By.XPATH, '//button[contains(@aria-label, "Apply current filters to show")]'
        )
        show_results_button.click()
        wait.until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//li[@data-occludable-job-id]")
            )
        )

        global pause_after_filters
        if pause_after_filters and "Turn off Pause after search" == pyautogui.confirm(
            "These are your configured search results and filter. It is safe to change them while"
            " this dialog is open, any changes later could result in errors and skipping this"
            " search run.",
            "Please check your results",
            ["Turn off Pause after search", "Look's good, Continue"],
        ):
            pause_after_filters = False
        print_lg("Filters applied.")

    except Exception as e:
        print_lg("Setting the preferences failed!")
        print_lg(e)
        time.sleep(3)


def load_full_results_page(max_scroll_rounds: int = 8) -> int:
    previous_count = -1
    stable_rounds = 0
    latest_count = 0

    for _ in range(max_scroll_rounds):
        latest_count = len(get_visible_job_listing_ids())
        if latest_count == previous_count:
            stable_rounds += 1
        else:
            previous_count = latest_count
            stable_rounds = 0

        if stable_rounds >= 2:
            break

        driver.execute_script(
            """
            const panel =
                document.querySelector('.jobs-search-results-list') ||
                document.querySelector('.scaffold-layout__list-container') ||
                document.querySelector('.jobs-search-results-list__list');
            if (panel) {
                panel.scrollTop = panel.scrollHeight;
            } else {
                window.scrollTo(0, document.body.scrollHeight);
            }
            """
        )
        buffer(max(click_gap, 2))

    return latest_count


def get_visible_job_listing_ids(limit: int | None = None) -> list[str]:
    job_ids = (
        driver.execute_script(
            """
        return Array.from(document.querySelectorAll('li[data-occludable-job-id]'))
            .map((item) => item.getAttribute('data-occludable-job-id'))
            .filter((jobId) => jobId);
        """
        )
        or []
    )
    return job_ids[:limit] if limit is not None else job_ids


def find_job_listing(job_id: str, time: float = 5.0) -> WebElement | None:
    try:
        return WebDriverWait(driver, time).until(
            EC.presence_of_element_located(
                (By.XPATH, f"//li[@data-occludable-job-id='{job_id}']")
            )
        )
    except Exception:
        return None


def get_job_main_details_by_id(
    job_id: str, blacklisted_companies: set, rejected_jobs: set
) -> tuple[str, str, str, str, str, bool]:
    for attempt in range(3):
        job = find_job_listing(job_id)
        if job is None:
            raise NoSuchElementException(f"Couldn't find job card for {job_id}")
        try:
            return get_job_main_details(job, blacklisted_companies, rejected_jobs)
        except StaleElementReferenceException:
            print_lg(
                f"Job card {job_id} rerendered while loading details."
                f" Retrying ({attempt + 1}/3)."
            )
            buffer(2)
    raise StaleElementReferenceException(
        f"Job card {job_id} kept rerendering after multiple retries."
    )


def _find_pagination_root() -> WebElement | None:
    selectors = [
        "//nav[contains(@aria-label, 'Page') and .//button]",
        "//div[contains(@class, 'artdeco-pagination') and .//button]",
        "//ul[contains(@class, 'artdeco-pagination__pages')]/ancestor::nav[1]",
        "//ul[contains(@class, 'artdeco-pagination__pages')]/ancestor::div[1]",
    ]
    for selector in selectors:
        try:
            return driver.find_element(By.XPATH, selector)
        except NoSuchElementException:
            continue
    return None


def _extract_page_number(pagination_element: WebElement | None) -> int | None:
    if pagination_element is None:
        return None

    selectors = [
        ".//button[@aria-current='true' or @aria-current='page']",
        ".//li[contains(@class, 'active')]//*[self::button or self::span][1]",
        ".//li[contains(@class, 'selected')]//*[self::button or self::span][1]",
    ]
    for selector in selectors:
        try:
            current_marker = pagination_element.find_element(By.XPATH, selector)
        except NoSuchElementException:
            continue

        marker_text = (
            current_marker.text.strip()
            or current_marker.get_attribute("aria-label")
            or ""
        )
        match = re.search(r"\d+", marker_text)
        if match:
            return int(match.group())
    return None


def _find_next_page_button(
    pagination_element: WebElement | None, current_page: int | None
) -> WebElement | None:
    root = pagination_element or driver
    selectors = []

    if current_page is not None:
        next_page = current_page + 1
        selectors.extend(
            [
                f".//button[@aria-label='Page {next_page}']",
                f".//button[@aria-label='View Page {next_page}']",
                f".//button[contains(@aria-label, 'Page {next_page}')]",
                f".//button[contains(@aria-label, 'View Page {next_page}')]",
            ]
        )

    selectors.extend(
        [
            ".//li[contains(@class, 'active')]/following-sibling::li[1]//button",
            ".//button[@aria-current='true']/ancestor::li/following-sibling::li[1]//button",
            ".//button[contains(@aria-label, 'Next')]",
            ".//button[contains(@aria-label, 'next')]",
        ]
    )

    for selector in selectors:
        try:
            button = root.find_element(By.XPATH, selector)
        except (NoSuchElementException, StaleElementReferenceException):
            root = driver
            continue

        if not button.is_enabled():
            continue
        if button.get_attribute("aria-disabled") == "true":
            continue
        if current_page is not None:
            label = button.get_attribute("aria-label") or button.text
            if str(current_page) == label.strip():
                continue
        return button

    return None


def get_page_info() -> tuple[WebElement | None, int | None]:
    """
    Function to get pagination element and current page number
    """
    pagination_element = _find_pagination_root()
    if pagination_element is None:
        print_lg("Failed to find Pagination element, hence couldn't scroll till end!")
        return None, None

    scroll_to_view(driver, pagination_element)
    current_page = _extract_page_number(pagination_element)
    return pagination_element, current_page


def _extract_results_start(current_url: str) -> int:
    query = parse_qs(urlsplit(current_url).query)
    raw_start = query.get("start", ["0"])[0]
    try:
        return int(raw_start)
    except (TypeError, ValueError):
        return 0


def _build_results_page_url(current_url: str, next_start: int) -> str:
    parsed_url = urlsplit(current_url)
    query = parse_qs(parsed_url.query, keep_blank_values=True)
    query.pop("currentJobId", None)
    query["start"] = [str(next_start)]
    return urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            urlencode(query, doseq=True),
            parsed_url.fragment,
        )
    )


def go_to_next_results_page(
    pagination_element: WebElement | None, current_page: int | None
) -> bool:
    del pagination_element, current_page
    previous_job_ids = get_visible_job_listing_ids(limit=3)
    current_url = driver.current_url
    next_start = _extract_results_start(current_url) + RESULTS_PAGE_SIZE
    next_url = _build_results_page_url(current_url, next_start)
    driver.get(next_url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//li[@data-occludable-job-id]")
            )
        )
    except Exception:
        pass

    load_full_results_page()
    current_job_ids = get_visible_job_listing_ids(limit=3)
    if not current_job_ids or current_job_ids == previous_job_ids:
        print_lg(
            f"\n>-> No new results appeared after requesting offset {next_start}."
            " Probably at the end page of results!\n"
        )
        return False

    print_lg(f"\n>-> Now on results offset {next_start} \n")
    return True


def get_job_main_details(
    job: WebElement, blacklisted_companies: set, rejected_jobs: set
) -> tuple[str, str, str, str, str, bool]:
    """
    # Function to get job main details.
    Returns a tuple of (job_id, title, company, work_location, work_style, skip)
    * job_id: Job ID
    * title: Job title
    * company: Company name
    * work_location: Work location of this job
    * work_style: Work style of this job (Remote, On-site, Hybrid)
    * skip: A boolean flag to skip this job
    """
    job_details_button = job.find_element(
        By.TAG_NAME, "a"
    )  # job.find_element(By.CLASS_NAME, "job-card-list__title")  # Problem in India
    scroll_to_view(driver, job_details_button, True)
    job_id = job.get_dom_attribute("data-occludable-job-id")
    title = job_details_button.text
    title = title[: title.find("\n")]
    # company = job.find_element(By.CLASS_NAME, "job-card-container__primary-description").text
    # work_location = job.find_element(By.CLASS_NAME, "job-card-container__metadata-item").text
    other_details = job.find_element(
        By.CLASS_NAME, "artdeco-entity-lockup__subtitle"
    ).text
    index = other_details.find(" · ")
    company = other_details[:index]
    work_location = other_details[index + 3 :]
    work_style = work_location[work_location.rfind("(") + 1 : work_location.rfind(")")]
    work_location = work_location[: work_location.rfind("(")].strip()

    # Skip if previously rejected due to blacklist or already applied
    skip = False
    if company in blacklisted_companies:
        print_lg(
            f'Skipping "{title} | {company}" job (Blacklisted Company). Job ID: {job_id}!'
        )
        skip = True
    elif job_id in rejected_jobs:
        print_lg(
            f'Skipping previously rejected "{title} | {company}" job. Job ID: {job_id}!'
        )
        skip = True
    try:
        if (
            job.find_element(By.CLASS_NAME, "job-card-container__footer-job-state").text
            == "Applied"
        ):
            skip = True
            print_lg(f'Already applied to "{title} | {company}" job. Job ID: {job_id}!')
    except Exception:
        pass
    try:
        if not skip:
            job_details_button.click()
    except Exception:
        print_lg(
            f'Failed to click "{title} | {company}" job on details button. Job ID: {job_id}!'
        )
        # print_lg(e)
        discard_job()
        job_details_button.click()  # To pass the error outside
    buffer(click_gap)
    return (job_id, title, company, work_location, work_style, skip)


# Function to check for Blacklisted words in About Company
def check_blacklist(
    rejected_jobs: set, job_id: str, company: str, blacklisted_companies: set
) -> tuple[set, set, WebElement] | ValueError:
    jobs_top_card = try_find_by_classes(
        driver,
        [
            "job-details-jobs-unified-top-card__primary-description-container",
            "job-details-jobs-unified-top-card__primary-description",
            "jobs-unified-top-card__primary-description",
            "jobs-details__main-content",
        ],
    )
    about_company_org = find_by_class(driver, "jobs-company__box")
    scroll_to_view(driver, about_company_org)
    about_company_org = about_company_org.text
    about_company = about_company_org.lower()
    skip_checking = False
    if has_auto_apply_recruiter_signal(company, about_company):
        print_lg(
            f'Skipping company blacklist for recruiter/vendor posting "{company}".'
        )
        skip_checking = True
    for word in about_company_good_words:
        if word.lower() in about_company:
            print_lg(
                f'Found the word "{word}". So, skipped checking for blacklist words.'
            )
            skip_checking = True
            break
    if not skip_checking:
        for word in about_company_bad_words:
            if word.lower() in about_company:
                rejected_jobs.add(job_id)
                blacklisted_companies.add(company)
                raise ValueError(f'\n"{about_company_org}"\n\nContains "{word}".')
    buffer(click_gap)
    scroll_to_view(driver, jobs_top_card)
    return rejected_jobs, blacklisted_companies, jobs_top_card


# Function to extract years of experience required from About Job
def extract_years_of_experience(text: str) -> int:
    required_years = estimate_required_years(text)
    if required_years is None:
        print_lg(f"\n{text}\n\nCouldn't find experience requirement in About the Job!")
        return 0
    return required_years


def _parse_money_amount(raw_value: str, suffix: str | None) -> float:
    value = float(raw_value.replace(",", ""))
    if suffix and suffix.lower() == "k":
        value *= 1000
    return value


def _looks_like_hourly_range(
    low: float, high: float, low_suffix: str | None, high_suffix: str | None
) -> bool:
    return not low_suffix and not high_suffix and max(low, high) < 1000


def compensation_below_floor(title: str, description: str) -> tuple[bool, str | None]:
    text = f"{title}\n{description}".replace("–", "-").replace("—", "-")

    for pattern in (
        _HOURLY_RANGE_PATTERN,
        _ANNUAL_RANGE_PATTERN,
        _ANNUAL_CONTEXT_RANGE_PATTERN,
    ):
        match = pattern.search(text)
        if not match:
            continue
        low = _parse_money_amount(match.group(1), match.group(2))
        high = _parse_money_amount(match.group(3), match.group(4))
        upper_bound = max(low, high)
        if pattern is _ANNUAL_CONTEXT_RANGE_PATTERN and _looks_like_hourly_range(
            low, high, match.group(2), match.group(4)
        ):
            if upper_bound < MINIMUM_HOURLY_RATE_USD:
                return True, match.group(0)
            return False, match.group(0)
        if pattern is _HOURLY_RANGE_PATTERN and upper_bound < MINIMUM_HOURLY_RATE_USD:
            return True, match.group(0)
        if (
            pattern in (_ANNUAL_RANGE_PATTERN, _ANNUAL_CONTEXT_RANGE_PATTERN)
            and upper_bound < minimum_annual_salary_usd
        ):
            return True, match.group(0)
        return False, match.group(0)

    for pattern, floor in (
        (_HOURLY_SINGLE_PATTERN, MINIMUM_HOURLY_RATE_USD),
        (_ANNUAL_SINGLE_PATTERN, minimum_annual_salary_usd),
    ):
        match = pattern.search(text)
        if not match:
            continue
        amount = _parse_money_amount(match.group(1), match.group(2))
        if amount < floor:
            return True, match.group(0)
        return False, match.group(0)

    return False, None


def get_job_description() -> (
    tuple[
        str | Literal["Unknown"], int | Literal["Unknown"], bool, str | None, str | None
    ]
):
    """
    # Job Description
    Function to extract job description from About the Job.
    ### Returns:
    - `jobDescription: str | 'Unknown'`
    - `experience_required: int | 'Unknown'`
    - `skip: bool`
    - `skipReason: str | None`
    - `skipMessage: str | None`
    """
    try:
        jobDescription = "Unknown"
        experience_required = "Unknown"
        found_masters = 0
        jobDescription = find_by_class(driver, "jobs-box__html-content").text
        jobDescriptionLow = jobDescription.lower()
        skip = False
        skipReason = None
        skipMessage = None
        for word in bad_words:
            if word.lower() in jobDescriptionLow:
                skipMessage = f'\n{jobDescription}\n\nContains bad word "{word}". Skipping this job!\n'
                skipReason = "Found a Bad Word in About Job"
                skip = True
                break
        if (
            False
            # not skip
            # # I could get a clearance so okay
            # and security_clearance == False
            # and (
            #     "polygraph" in jobDescriptionLow
            #     or "clearance" in jobDescriptionLow
            #     or "secret" in jobDescriptionLow
            # )
        ):
            skipMessage = f'\n{jobDescription}\n\nFound "Clearance" or "Polygraph". Skipping this job!\n'
            skipReason = "Asking for Security clearance"
            skip = True
        if not skip:
            if did_masters and "master" in jobDescriptionLow:
                print_lg(f'Found the word "master" in \n{jobDescription}')
                found_masters = 1  # changed from 1
            experience_required = extract_years_of_experience(jobDescription)
            if (
                current_experience > -1
                and experience_required > current_experience + found_masters
            ):
                skipMessage = (
                    f"\n{jobDescription}\n\nExperience required {experience_required} > Current"
                    f" Experience {current_experience + found_masters}. Skipping this job!\n"
                )
                skipReason = "Required experience is high"
                skip = True
    except Exception:
        if jobDescription == "Unknown":
            print_lg("Unable to extract job description!")
        else:
            experience_required = "Error in extraction"
            print_lg("Unable to extract years of experience required!")
            # print_lg(e)
    finally:
        return jobDescription, experience_required, skip, skipReason, skipMessage


# Function to upload resume
def upload_resume(modal: WebElement, resume: str) -> tuple[bool, str]:
    try:
        modal.find_element(By.NAME, "file").send_keys(os.path.abspath(resume))
        return True, os.path.basename(default_resume_path)
    except Exception:
        return False, "Previous resume"


# Function to answer common questions for Easy Apply
def add_answer_source_review_reasons(
    questions_list: set, manual_writeup_reasons: set[str]
) -> None:
    for question_record in questions_list:
        reason = answer_record_manual_review_reason(question_record)
        if reason:
            manual_writeup_reasons.add(reason)


# Function to answer the questions for Easy Apply
def answer_questions(
    modal: WebElement,
    questions_list: set,
    work_location: str,
    manual_writeup_reasons: set[str],
    job_question_context: str = "",
) -> set:
    # Get all questions from the page

    all_questions = modal.find_elements(By.XPATH, ".//div[@data-test-form-element]")
    question_bank_occurrences: dict[tuple[str, str, str], int] = {}

    def question_bank_answer(
        label: str, field_type: str, options: list[str] | None = None
    ):
        key = (normalize_question_label(label), field_type, option_kind(options))
        question_bank_occurrences[key] = question_bank_occurrences.get(key, 0) + 1
        return QUESTION_BANK.answer_for_question(
            label,
            field_type,
            options,
            occurrence=question_bank_occurrences[key],
        )

    for Question in all_questions:
        select_elements = Question.find_elements(By.XPATH, ".//select")
        if select_elements:
            label_org = "Unknown"
            try:
                label = Question.find_element(By.TAG_NAME, "label")
                label_org = label.find_element(By.TAG_NAME, "span").text
            except Exception:
                pass
            label = label_org.lower()

            for select_element in select_elements:
                answer = ""
                answer_source = "manual"
                select = Select(select_element)
                selected_option = select.first_selected_option.text
                optionsText = []
                options = '"List of phone country codes"'
                if label != "phone country code":
                    optionsText = [option.text for option in select.options]
                    options = "".join([f' "{option}",' for option in optionsText])
                prev_answer = selected_option
                bank_answer = question_bank_answer(label_org, "select", optionsText)
                force_overwrite = should_override_existing_answer(
                    label_org, selected_option, "select"
                )
                if (
                    overwrite_previous_answers
                    or is_placeholder_answer(selected_option)
                    or force_overwrite
                ):
                    safe_answer = deterministic_answer_for_question(
                        label_org,
                        "select",
                        optionsText,
                        confidence_level,
                        str(
                            current_experience
                            if current_experience > -1
                            else years_of_experience
                        ),
                    )
                    if safe_answer:
                        answer, answer_source = safe_answer
                    elif manual_review_reason := manual_review_reason_for_question(
                        label_org
                    ):
                        print_lg(
                            f'Question "{label_org}" requires manual review:'
                            f" {manual_review_reason}."
                        )
                        add_manual_writeup_reason(
                            manual_writeup_reasons, "select", label_org
                        )
                        answer_source = "manual"
                    elif bank_answer:
                        answer = bank_answer.answer
                        answer_source = f"question_bank:{bank_answer.key}"
                    elif "email" in label or "phone" in label:
                        answer = prev_answer
                        answer_source = "existing"
                    elif "gender" in label or "sex" in label:
                        answer = gender
                        answer_source = "profile:eeo"
                    elif "disability" in label:
                        answer = disability_status
                        answer_source = "profile:eeo"
                    elif "proficiency" in label:
                        answer = "Professional"
                        answer_source = "profile:proficiency"
                    else:
                        add_manual_writeup_reason(
                            manual_writeup_reasons, "select", label_org
                        )
                    try:
                        select.select_by_visible_text(answer)
                    except NoSuchElementException:
                        ai_answer = answer_simple_question(
                            label_org, SIMPLE_QUESTION_USER_CONTEXT, optionsText
                        )
                        possible_answer_phrases = []
                        if ai_answer:
                            possible_answer_phrases = [ai_answer]
                            answer_source = "ai_cli_short"
                        elif answer == "Decline":
                            possible_answer_phrases = [
                                "Decline",
                                "not wish",
                                "don't wish",
                                "Prefer not",
                                "not want",
                            ]
                        elif answer:
                            possible_answer_phrases = [answer]
                        foundOption = False
                        for phrase in possible_answer_phrases:
                            for option in optionsText:
                                if phrase in option:
                                    select.select_by_visible_text(option)
                                    answer = (
                                        f"Decline ({option})"
                                        if len(possible_answer_phrases) > 1
                                        else option
                                    )
                                    foundOption = True
                                    break
                            if foundOption:
                                break
                        if not foundOption:
                            print_lg(
                                f'Failed to safely answer select question "{label_org}".'
                                " Requiring manual review."
                            )
                            add_manual_writeup_reason(
                                manual_writeup_reasons, "select", label_org
                            )
                            answer = prev_answer
                            answer_source = "manual"
                else:
                    answer = prev_answer
                    answer_source = "existing"
                if answer_source not in {"manual", "manual_placeholder"} and not (
                    is_placeholder_answer(answer)
                ):
                    remove_manual_writeup_reason(
                        manual_writeup_reasons, "select", label_org
                    )
                add_question_record(
                    questions_list,
                    f"{label_org} [ {options} ]",
                    answer,
                    "select",
                    prev_answer,
                    answer_source,
                )
            continue

        radio = try_xp(
            Question,
            './/fieldset[@data-test-form-builder-radio-button-form-component="true"]',
            False,
        )
        if radio:
            prev_answer = None
            label = try_xp(
                radio,
                ".//span[@data-test-form-builder-radio-button-form-component__title]",
                False,
            )
            try:
                label = find_by_class(label, "visually-hidden", 2.0)
            except Exception:
                pass
            label_org = label.text if label else "Unknown"
            answer = ""
            answer_source = "manual"
            label = label_org.lower()

            options = radio.find_elements(By.TAG_NAME, "input")
            option_display = []
            option_labels = []

            for option in options:
                id = option.get_attribute("id")
                option_label = try_xp(radio, f'.//label[@for="{id}"]', False)
                option_text = option_label.text if option_label else "Unknown"
                option_labels.append(option_text)
                option_display.append(
                    f'"{option_text}"<{option.get_attribute("value")}>'
                )
                if option.is_selected():
                    prev_answer = option_text

            force_overwrite = should_override_existing_answer(
                label_org, prev_answer, "radio"
            )
            if overwrite_previous_answers or prev_answer is None or force_overwrite:
                bank_answer = question_bank_answer(label_org, "radio", option_labels)
                safe_answer = deterministic_answer_for_question(
                    label_org,
                    "radio",
                    option_labels,
                    confidence_level,
                    str(
                        current_experience
                        if current_experience > -1
                        else years_of_experience
                    ),
                )
                if safe_answer:
                    answer, answer_source = safe_answer
                elif manual_review_reason := manual_review_reason_for_question(
                    label_org
                ):
                    print_lg(
                        f'Question "{label_org}" requires manual review:'
                        f" {manual_review_reason}."
                    )
                    add_manual_writeup_reason(
                        manual_writeup_reasons, "radio", label_org
                    )
                    answer_source = "manual"
                elif bank_answer:
                    answer = bank_answer.answer
                    answer_source = f"question_bank:{bank_answer.key}"
                elif "citizenship" in label or "employment eligibility" in label:
                    answer = us_citizenship
                    answer_source = "legal:work_authorization"
                elif "veteran" in label or "protected" in label:
                    answer = veteran_status
                    answer_source = "profile:eeo"
                elif "disability" in label or "handicapped" in label:
                    answer = disability_status
                    answer_source = "profile:eeo"
                else:
                    add_manual_writeup_reason(
                        manual_writeup_reasons, "radio", label_org
                    )
                foundOption = try_xp(
                    radio, f".//label[normalize-space()='{answer}']", False
                )
                if foundOption:
                    actions.move_to_element(foundOption).click().perform()
                else:
                    ai_answer = answer_simple_question(
                        label_org, SIMPLE_QUESTION_USER_CONTEXT, option_labels
                    )
                    possible_answer_phrases = []
                    if ai_answer:
                        possible_answer_phrases = [ai_answer]
                        answer_source = "ai_cli_short"
                    elif answer == "Decline":
                        possible_answer_phrases = [
                            "Decline",
                            "not wish",
                            "don't wish",
                            "Prefer not",
                            "not want",
                        ]
                    elif answer:
                        possible_answer_phrases = [answer]
                    ele = options[0]
                    answer = prev_answer
                    foundOption = False
                    for phrase in possible_answer_phrases:
                        for i, option_label in enumerate(option_labels):
                            if phrase in option_label:
                                foundOption = options[i]
                                ele = foundOption
                                answer = (
                                    f"Decline ({option_label})"
                                    if len(possible_answer_phrases) > 1
                                    else option_label
                                )
                                break
                        if foundOption:
                            break
                    if foundOption:
                        actions.move_to_element(ele).click().perform()
                    else:
                        print_lg(
                            f'Failed to safely answer radio question "{label_org}".'
                            " Requiring manual review."
                        )
                        add_manual_writeup_reason(
                            manual_writeup_reasons, "radio", label_org
                        )
                        answer_source = "manual"
            else:
                answer = prev_answer
                answer_source = "existing"
            if answer_source not in {"manual", "manual_placeholder"} and not (
                is_placeholder_answer(answer)
            ):
                remove_manual_writeup_reason(manual_writeup_reasons, "radio", label_org)
            add_question_record(
                questions_list,
                f'{label_org} [ {", ".join(option_display)} ]',
                answer,
                "radio",
                prev_answer,
                answer_source,
            )
            continue

        text = try_xp(Question, ".//input[@type='text']", False)
        if text:
            do_actions = False
            label = try_xp(Question, ".//label[@for]", False)
            try:
                label = label.find_element(By.CLASS_NAME, "visually-hidden")
            except Exception:
                pass
            label_org = label.text if label else "Unknown"
            answer = ""
            answer_source = "existing"
            label = label_org.lower()

            prev_answer = text.get_attribute("value")
            force_overwrite = should_override_existing_answer(
                label_org, prev_answer, "text"
            )
            field_requires_manual_writeup = needs_manual_writeup(label_org)
            if field_requires_manual_writeup and (
                overwrite_previous_answers or not prev_answer or force_overwrite
            ):
                add_manual_writeup_reason(manual_writeup_reasons, "text", label_org)
            if overwrite_previous_answers or not prev_answer or force_overwrite:
                bank_answer = question_bank_answer(label_org, "text")
                if bank_answer:
                    answer = bank_answer.answer
                    answer_source = f"question_bank:{bank_answer.key}"
                elif manual_review_reason := manual_review_reason_for_question(
                    label_org
                ):
                    print_lg(
                        f'Question "{label_org}" requires manual review:'
                        f" {manual_review_reason}."
                    )
                    add_manual_writeup_reason(manual_writeup_reasons, "text", label_org)
                    answer_source = "manual"
                elif safe_answer := deterministic_answer_for_question(
                    label_org,
                    "text",
                    None,
                    confidence_level,
                    str(
                        current_experience
                        if current_experience > -1
                        else years_of_experience
                    ),
                ):
                    answer, answer_source = safe_answer
                elif configured_answer := configured_free_text_answer(
                    label_org, cover_letter, linkedin_summary
                ):
                    answer, answer_source = configured_answer
                elif field_requires_manual_writeup:
                    ai_answer = answer_freeform_question(
                        label_org, SIMPLE_QUESTION_USER_CONTEXT, job_question_context
                    )
                    if ai_answer:
                        answer = ai_answer
                        answer_source = "ai_cli_freeform"
                    else:
                        answer = MANUAL_WRITEUP_PLACEHOLDER_TEXT
                        answer_source = "manual_placeholder"
                else:
                    answer_source = "manual"
                    if (
                        "experience" in label or "years" in label
                    ) and not is_specific_experience_label(label_org):
                        answer = str(
                            current_experience
                            if current_experience > -1
                            else years_of_experience
                        )
                        answer_source = "profile:generic_years"
                    elif "phone" in label or "mobile" in label:
                        answer = phone_number
                        answer_source = "profile:phone"
                    elif "street" in label:
                        answer = street
                        answer_source = "profile:address"
                    elif "city" in label or "location" in label or "address" in label:
                        answer = current_city if current_city else work_location
                        do_actions = True
                        answer_source = "profile:address"
                    elif "signature" in label:
                        answer = full_name
                        answer_source = "profile:identity"
                    elif "name" in label:
                        if "full" in label:
                            answer = full_name
                        elif "first" in label and "last" not in label:
                            answer = first_name
                        elif "middle" in label and "last" not in label:
                            answer = middle_name
                        elif "last" in label and "first" not in label:
                            answer = last_name
                        elif "employer" in label:
                            answer = recent_employer
                        else:
                            answer = full_name
                        answer_source = "profile:identity"
                    elif "notice" in label:
                        if "month" in label:
                            answer = notice_period_months
                        elif "week" in label:
                            answer = notice_period_weeks
                        else:
                            answer = notice_period
                        answer_source = "preference:notice_period"
                    elif (
                        "salary" in label
                        or "compensation" in label
                        or "ctc" in label
                        or "pay" in label
                    ):
                        if "current" in label or "present" in label:
                            if "month" in label:
                                answer = current_ctc_monthly
                            elif "lakh" in label:
                                answer = current_ctc_lakhs
                            else:
                                answer = current_ctc
                            answer_source = "preference:current_compensation"
                        elif recommended_answer := recommended_compensation_answer(
                            label_org, job_question_context
                        ):
                            answer, answer_source = recommended_answer
                        else:
                            if "month" in label:
                                answer = desired_salary_monthly
                            elif "lakh" in label:
                                answer = desired_salary_lakhs
                            else:
                                answer = desired_salary
                            answer_source = "preference:desired_compensation"
                    elif "linkedin" in label:
                        answer = linkedIn
                        answer_source = "profile:links"
                    elif (
                        "website" in label
                        or "blog" in label
                        or "portfolio" in label
                        or "link" in label
                    ):
                        answer = website
                        answer_source = "profile:links"
                    elif "scale of 1-10" in label:
                        answer = confidence_level
                        answer_source = "profile:rating_scale"
                    elif "headline" in label:
                        answer = linkedin_headline
                        answer_source = "profile:headline"
                    elif (
                        ("hear" in label or "come across" in label)
                        and "this" in label
                        and ("job" in label or "position" in label)
                    ):
                        answer = (
                            "https://github.com/GodsScion/Auto_job_applier_linkedIn"
                        )
                        answer_source = "profile:source"
                    elif "state" in label or "province" in label:
                        answer = state
                        answer_source = "profile:address"
                    elif is_postal_code_label(label_org):
                        answer = zipcode
                        answer_source = "profile:address"
                    elif "country" in label:
                        answer = country
                        answer_source = "profile:address"
                if answer == "" and answer_source != "manual":
                    ai_answer = answer_simple_question(
                        label_org, SIMPLE_QUESTION_USER_CONTEXT
                    )
                    if ai_answer:
                        answer = ai_answer
                        answer_source = "ai_cli_short"
                if answer == "":
                    add_manual_writeup_reason(manual_writeup_reasons, "text", label_org)
                    answer_source = "manual"
                else:
                    text.clear()
                    text.send_keys(answer)
                    remove_manual_writeup_reason(
                        manual_writeup_reasons, "text", label_org
                    )
                    if do_actions:
                        sleep(2)
                        actions.send_keys(Keys.ARROW_DOWN)
                        actions.send_keys(Keys.ENTER).perform()
            add_question_record(
                questions_list,
                label_org,
                text.get_attribute("value"),
                "text",
                prev_answer,
                answer_source,
            )
            continue

        text_area = try_xp(Question, ".//textarea", False)
        if text_area:
            label = try_xp(Question, ".//label[@for]", False)
            label_org = label.text if label else "Unknown"
            prev_answer = text_area.get_attribute("value")
            answer_source = "existing"
            if overwrite_previous_answers or not prev_answer:
                safe_answer = deterministic_answer_for_question(
                    label_org,
                    "textarea",
                    None,
                    confidence_level,
                    str(
                        current_experience
                        if current_experience > -1
                        else years_of_experience
                    ),
                )
                if safe_answer:
                    answer, answer_source = safe_answer
                    text_area.clear()
                    text_area.send_keys(answer)
                    remove_manual_writeup_reason(
                        manual_writeup_reasons, "textarea", label_org
                    )
                elif manual_review_reason := manual_review_reason_for_question(
                    label_org
                ):
                    print_lg(
                        f'Question "{label_org}" requires manual review:'
                        f" {manual_review_reason}."
                    )
                    add_manual_writeup_reason(
                        manual_writeup_reasons, "textarea", label_org
                    )
                    text_area.clear()
                    text_area.send_keys(MANUAL_WRITEUP_PLACEHOLDER_TEXT)
                    answer_source = "manual_placeholder"
                elif bank_answer := question_bank_answer(label_org, "textarea"):
                    text_area.clear()
                    text_area.send_keys(bank_answer.answer)
                    answer_source = f"question_bank:{bank_answer.key}"
                    remove_manual_writeup_reason(
                        manual_writeup_reasons, "textarea", label_org
                    )
                elif configured_answer := configured_free_text_answer(
                    label_org, cover_letter, linkedin_summary
                ):
                    answer, answer_source = configured_answer
                    text_area.clear()
                    text_area.send_keys(answer)
                    remove_manual_writeup_reason(
                        manual_writeup_reasons, "textarea", label_org
                    )
                elif should_try_long_form_answer(label_org):
                    ai_answer = answer_freeform_question(
                        label_org, SIMPLE_QUESTION_USER_CONTEXT, job_question_context
                    )
                    if ai_answer:
                        text_area.clear()
                        text_area.send_keys(ai_answer)
                        answer_source = "ai_cli_freeform"
                        remove_manual_writeup_reason(
                            manual_writeup_reasons, "textarea", label_org
                        )
                    else:
                        add_manual_writeup_reason(
                            manual_writeup_reasons, "textarea", label_org
                        )
                        text_area.clear()
                        text_area.send_keys(MANUAL_WRITEUP_PLACEHOLDER_TEXT)
                        answer_source = "manual_placeholder"
                else:
                    add_manual_writeup_reason(
                        manual_writeup_reasons, "textarea", label_org
                    )
                    text_area.clear()
                    text_area.send_keys(MANUAL_WRITEUP_PLACEHOLDER_TEXT)
                    answer_source = "manual_placeholder"
            else:
                remove_manual_writeup_reason(
                    manual_writeup_reasons, "textarea", label_org
                )
            add_question_record(
                questions_list,
                label_org,
                text_area.get_attribute("value"),
                "textarea",
                prev_answer,
                answer_source,
            )
            continue

        checkbox = try_xp(Question, ".//input[@type='checkbox']", False)
        if checkbox:
            label = try_xp(Question, ".//span[@class='visually-hidden']", False)
            label_org = label.text if label else "Unknown"
            answer = try_xp(Question, ".//label[@for]", False)
            answer = answer.text if answer else "Unknown"
            prev_answer = checkbox.is_selected()
            checked = prev_answer
            add_question_record(
                questions_list,
                f"{label_org} ([X] {answer})",
                checked,
                "checkbox",
                prev_answer,
                "existing",
            )
            continue

    add_answer_source_review_reasons(questions_list, manual_writeup_reasons)

    try_xp(driver, "//button[contains(@aria-label, 'This is today')]")

    return questions_list


def _is_external_application_url(url: str, job_link: str) -> bool:
    return is_real_external_apply_url(url, job_link)


def _handle_external_apply_modal_once() -> bool:
    modal = try_xp(
        driver,
        ".//div[contains(@class, 'artdeco-modal') or contains(@class, 'jobs-apply-modal')]",
        False,
    )
    if not modal:
        return False
    modal_text = modal.text.lower()
    if "share your profile" in modal_text or "share your full profile" in modal_text:
        print_lg(
            "LinkedIn external-apply profile-share modal appeared; not clicking"
            " Continue automatically. Saving this job for manual review."
        )
        return False
    if wait_span_click(modal, "Continue", 2, True, True, False):
        return True
    return bool(wait_span_click(modal, "Apply", 1, True, True, False))


def _close_non_linkedin_tabs() -> None:
    for handle in list(driver.window_handles):
        if handle == linkedIn_tab:
            continue
        try:
            driver.switch_to.window(handle)
            driver.close()
        except Exception:
            pass
    driver.switch_to.window(linkedIn_tab)


def _find_external_application_url_in_open_windows(
    before_handles: set[str],
    job_link: str,
) -> str | None:
    try:
        original_handle = driver.current_window_handle
    except NoSuchWindowException:
        original_handle = None

    handles = list(driver.window_handles)
    new_handles = [handle for handle in handles if handle not in before_handles]
    ordered_handles = new_handles + [
        handle for handle in handles if handle not in new_handles
    ]

    for handle in ordered_handles:
        try:
            driver.switch_to.window(handle)
            candidate_link = driver.current_url
        except NoSuchWindowException:
            continue
        real_candidate_link = extract_real_external_apply_url(candidate_link, job_link)
        if real_candidate_link:
            return real_candidate_link

    if original_handle in driver.window_handles:
        driver.switch_to.window(original_handle)
    return None


def _wait_for_external_application_url(
    before_handles: set[str],
    job_link: str,
    timeout_seconds: int = 15,
) -> str | None:
    end_wait = time.time() + timeout_seconds
    while time.time() < end_wait:
        candidate_link = _find_external_application_url_in_open_windows(
            before_handles, job_link
        )
        if candidate_link:
            return candidate_link
        _handle_external_apply_modal_once()
        buffer(1)
    return None


def _return_to_linkedin_job(job_link: str) -> None:
    try:
        driver.switch_to.window(linkedIn_tab)
        if _is_external_application_url(driver.current_url, job_link):
            driver.get(job_link)
            buffer(click_gap)
    except Exception:
        pass


def external_apply(
    pagination_element: WebElement | None,
    job_id: str,
    job_link: str,
    resume: str,
    date_listed,
    application_link: str,
    screenshot_name: str,
) -> tuple[bool, str, int]:
    """
    Function to open new tab and save external job application links
    """
    global tabs_count, dailyEasyApplyLimitReached
    if easy_apply_only:
        if daily_easy_apply_limit_reached():
            dailyEasyApplyLimitReached = True
            print_lg(
                "\n###############  LinkedIn Easy Apply daily limit detected."
                " Stopping run.  ###############\n"
            )
        print_lg("Easy apply failed I guess!")
        if pagination_element is not None:
            return True, application_link, tabs_count
    try:
        before_handles = set(driver.window_handles)
        apply_button = wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    (
                        ".//button[contains(@class,'jobs-apply-button') and contains(@class,"
                        " 'artdeco-button--3')]"
                    ),
                )
            )
        )
        apply_button.click()
        buffer(click_gap)

        if _handle_external_apply_modal_once():
            print_lg("Clicked LinkedIn external-apply continuation once.")

        candidate_link = _wait_for_external_application_url(before_handles, job_link)
        tabs_count = len(driver.window_handles)

        if candidate_link and _is_external_application_url(candidate_link, job_link):
            application_link = candidate_link
            print_lg(f'Got the external application link "{application_link}"')
        else:
            application_link = job_link
            print_lg(
                "External apply did not expose a usable external URL after one click;"
                f' saving LinkedIn job link "{application_link}" for manual review.'
            )

        if close_tabs:
            _close_non_linkedin_tabs()
            _return_to_linkedin_job(job_link)
        else:
            driver.switch_to.window(linkedIn_tab)
        actions.send_keys(Keys.ESCAPE).perform()
        return False, application_link, tabs_count
    except Exception as e:
        # print_lg(e)
        print_lg("Failed to apply!")
        failed_job(
            job_id,
            job_link,
            resume,
            date_listed,
            "Probably didn't find Apply button or unable to switch tabs.",
            e,
            application_link,
            screenshot_name,
        )
        global failed_count
        failed_count += 1
        return True, application_link, tabs_count


def follow_company(modal: WebDriver = driver) -> None:
    """
    Function to follow or un-follow easy applied companies based om `follow_companies`
    """
    try:
        follow_checkbox_input = try_xp(
            modal, ".//input[@id='follow-company-checkbox' and @type='checkbox']", False
        )
        if (
            follow_checkbox_input
            and follow_checkbox_input.is_selected() != follow_companies
        ):
            try_xp(modal, ".//label[@for='follow-company-checkbox']")
    except Exception as e:
        print_lg("Failed to update follow companies checkbox!", e)


# < Failed attempts logging
def failed_job(
    job_id: str,
    job_link: str,
    resume: str,
    date_listed,
    error: str,
    exception: Exception,
    application_link: str,
    screenshot_name: str,
) -> None:
    """
    Function to update failed jobs list in excel
    """
    try:
        with open(failed_file_name, "a", newline="", encoding="utf-8") as file:
            fieldnames = [
                "Job ID",
                "Job Link",
                "Resume Tried",
                "Date listed",
                "Date Tried",
                "Assumed Reason",
                "Stack Trace",
                "External Job link",
                "Screenshot Name",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "Job ID": job_id,
                    "Job Link": job_link,
                    "Resume Tried": resume,
                    "Date listed": date_listed,
                    "Date Tried": datetime.now(),
                    "Assumed Reason": error,
                    "Stack Trace": exception,
                    "External Job link": application_link,
                    "Screenshot Name": screenshot_name,
                }
            )
            file.close()
    except Exception as e:
        print_lg("Failed to update failed jobs list!", e)
        pyautogui.alert(
            "Failed to update the excel of failed jobs!\nProbably because of 1 of the following"
            " reasons:\n1. The file is currently open or in use by another program\n2. Permission"
            " denied to write to the file\n3. Failed to find the file",
            "Failed Logging",
        )


def screenshot(driver: WebDriver, job_id: str, failedAt: str) -> str:
    """
    Function to to take screenshot for debugging
    - Returns screenshot name as String
    """
    screenshot_name = "{} - {} - {}.png".format(job_id, failedAt, str(datetime.now()))
    path = logs_folder_path + "/screenshots/" + screenshot_name.replace(":", ".")
    # special_chars = {'*', '"', '\\', '<', '>', ':', '|', '?'}
    # for char in special_chars:  path = path.replace(char, '-')
    driver.save_screenshot(path.replace("//", "/"))
    return screenshot_name


# >


def submitted_jobs(
    job_id: str,
    title: str,
    company: str,
    work_location: str,
    work_style: str,
    description: str,
    experience_required: int | Literal["Unknown", "Error in extraction"],
    skills: list[str] | Literal["In Development"],
    hr_name: str | Literal["Unknown"],
    hr_link: str | Literal["Unknown"],
    resume: str,
    reposted: bool,
    date_listed: datetime | Literal["Unknown"],
    date_applied: datetime | Literal["Pending"],
    job_link: str,
    application_link: str,
    questions_list: set | None,
    connect_request: Literal["In Development"],
) -> None:
    """
    Function to create or update the Applied jobs CSV file, once the application is submitted successfully
    """
    try:
        with open(file_name, mode="a", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "Job ID",
                "Title",
                "Company",
                "Work Location",
                "Work Style",
                "About Job",
                "Experience required",
                "Skills required",
                "HR Name",
                "HR Link",
                "Resume",
                "Re-posted",
                "Date Posted",
                "Date Applied",
                "Job Link",
                "External Job link",
                "Questions Found",
                "Connect Request",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "Job ID": job_id,
                    "Title": title,
                    "Company": company,
                    "Work Location": work_location,
                    "Work Style": work_style,
                    "About Job": description,
                    "Experience required": experience_required,
                    "Skills required": skills,
                    "HR Name": hr_name,
                    "HR Link": hr_link,
                    "Resume": resume,
                    "Re-posted": reposted,
                    "Date Posted": date_listed,
                    "Date Applied": date_applied,
                    "Job Link": job_link,
                    "External Job link": application_link,
                    "Questions Found": serialize_questions(questions_list),
                    "Connect Request": connect_request,
                }
            )
        csv_file.close()
    except Exception as e:
        print_lg("Failed to update submitted jobs list!", e)
        pyautogui.alert(
            "Failed to update the excel of applied jobs!\nProbably because of 1 of the following"
            " reasons:\n1. The file is currently open or in use by another program\n2. Permission"
            " denied to write to the file\n3. Failed to find the file",
            "Failed Logging",
        )


def save_manual_writeup_job(
    queued_job_ids: set[str],
    job_id: str,
    title: str,
    company: str,
    search_term: str,
    work_location: str,
    work_style: str,
    description: str,
    experience_required: int | Literal["Unknown", "Error in extraction"],
    skills: list[str] | Literal["In Development"],
    hr_name: str | Literal["Unknown"],
    hr_link: str | Literal["Unknown"],
    resume: str,
    reposted: bool,
    date_listed: datetime | Literal["Unknown"],
    job_link: str,
    application_link: str,
    questions_list: set | None,
    backlog_type: str,
    backlog_reason: str,
) -> bool:
    if job_id in queued_job_ids:
        print_lg(
            f'Skipping duplicate manual-writeup job "{title} | {company}". Job ID: {job_id}'
        )
        return False

    try:
        with open(
            MANUAL_WRITEUP_JOBS_FILE, mode="a", newline="", encoding="utf-8"
        ) as csv_file:
            fieldnames = [
                "Job ID",
                "Title",
                "Company",
                "Search Term",
                "Backlog Type",
                "Backlog Reason",
                "Work Location",
                "Work Style",
                "About Job",
                "Experience required",
                "Skills required",
                "HR Name",
                "HR Link",
                "Resume",
                "Re-posted",
                "Date Posted",
                "Job Link",
                "Application Link",
                "Questions Found",
                "Date Saved",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "Job ID": job_id,
                    "Title": title,
                    "Company": company,
                    "Search Term": search_term,
                    "Backlog Type": backlog_type,
                    "Backlog Reason": backlog_reason,
                    "Work Location": work_location,
                    "Work Style": work_style,
                    "About Job": description,
                    "Experience required": experience_required,
                    "Skills required": skills,
                    "HR Name": hr_name,
                    "HR Link": hr_link,
                    "Resume": resume,
                    "Re-posted": reposted,
                    "Date Posted": date_listed,
                    "Job Link": job_link,
                    "Application Link": application_link,
                    "Questions Found": serialize_questions(questions_list),
                    "Date Saved": datetime.now(),
                }
            )
        queued_job_ids.add(job_id)
        print_lg(f'Queued "{title} | {company}" for manual writeup review.')
        return True
    except Exception as e:
        print_lg("Failed to update manual writeup jobs list!", e)
        pyautogui.alert(
            "Failed to update the manual writeup jobs csv!\nProbably because of 1 of the"
            " following reasons:\n1. The file is currently open or in use by another program\n2."
            " Permission denied to write to the file\n3. Failed to find the file",
            "Failed Logging",
        )
        return False


# Function to discard the job application
def _click_first_visible(xpath: str) -> bool:
    for element in driver.find_elements(By.XPATH, xpath):
        try:
            if element.is_displayed() and element.is_enabled():
                element.click()
                buffer(click_gap)
                return True
        except Exception:
            continue
    return False


def discard_job() -> None:
    for _ in range(2):
        actions.send_keys(Keys.ESCAPE).perform()
        buffer(0.5)
        if wait_span_click(driver, "Discard", 1):
            return
        if wait_span_click(driver, "Discard application", 1):
            return
        if _click_first_visible(
            ".//button[contains(@aria-label, 'Dismiss')"
            " or contains(@aria-label, 'Close')]"
        ):
            if wait_span_click(driver, "Discard", 1) or wait_span_click(
                driver, "Discard application", 1
            ):
                return
        if not driver.find_elements(
            By.XPATH,
            ".//div[contains(@class, 'jobs-easy-apply-modal')"
            " or @data-test-modal-id='easy-apply-modal']",
        ):
            return
    print_lg("Failed to close Easy Apply modal after discard attempts.")


# Function to apply to jobs
def apply_to_jobs(search_terms: list[str]) -> None:
    applied_jobs = get_applied_job_ids()
    queued_job_ids = get_manual_writeup_job_ids()
    networking_job_ids = get_networking_job_ids()
    rejected_jobs = set()
    blacklisted_companies = set()
    global current_city, failed_count, skip_count, easy_applied_count, external_jobs_count, tabs_count, pause_before_submit, pause_at_failed_question, useNewResume, manual_writeup_jobs_count, linkedin_applications_today_count, dailyEasyApplyLimitReached
    current_city = current_city.strip()

    if randomize_search_order:
        shuffle(search_terms)
    for searchTerm in search_terms:
        if configured_daily_linkedin_application_cap_reached():
            mark_configured_daily_linkedin_application_cap_reached()
            return
        seen_job_ids = set()
        driver.get(f"https://www.linkedin.com/jobs/search/?keywords={searchTerm}")
        print_lg(
            "\n________________________________________________________________________________________________________________________\n"
        )
        print_lg(f'\n>>>> Now searching for "{searchTerm}" <<<<\n\n')

        apply_filters()

        current_count = 0
        pages_processed = 0
        try:
            while current_count < switch_number:
                if configured_daily_linkedin_application_cap_reached():
                    mark_configured_daily_linkedin_application_cap_reached()
                    return
                # Wait until job listings are loaded
                wait.until(
                    EC.presence_of_all_elements_located(
                        (By.XPATH, "//li[@data-occludable-job-id]")
                    )
                )

                # Find all job listings in current page
                buffer(3)
                load_full_results_page()
                pagination_element, current_page = None, None
                job_listing_ids = get_visible_job_listing_ids()
                if not job_listing_ids:
                    print_lg("No job listings found on the current results page.")
                    break

                for listing_job_id in job_listing_ids:
                    if configured_daily_linkedin_application_cap_reached():
                        mark_configured_daily_linkedin_application_cap_reached()
                        return
                    if keep_screen_awake:
                        pyautogui.press("shiftright")
                    if current_count >= switch_number:
                        break
                    if listing_job_id in seen_job_ids:
                        print_lg(
                            f"Already inspected Job ID: {listing_job_id} in this search pass."
                        )
                        continue
                    print_lg("\n-@-\n")

                    try:
                        job_id, title, company, work_location, work_style, skip = (
                            get_job_main_details_by_id(
                                listing_job_id, blacklisted_companies, rejected_jobs
                            )
                        )
                    except NoSuchElementException:
                        print_lg(
                            f"Job card {listing_job_id} disappeared before it could be processed."
                        )
                        continue
                    except StaleElementReferenceException:
                        print_lg(
                            f"Job card {listing_job_id} kept rerendering. Skipping it for"
                            " this pass."
                        )
                        continue
                    seen_job_ids.add(job_id)

                    if skip:
                        continue
                    previously_networked = job_id in networking_job_ids
                    # Redundant fail safe check for applied jobs!
                    try:
                        if job_id in applied_jobs:
                            print_lg(
                                f'Already applied to "{title} | {company}" job. Job ID: {job_id}!'
                            )
                            continue
                        if job_id in queued_job_ids:
                            print_lg(
                                f'Already queued "{title} | {company}" for manual writeup. Job ID:'
                                f" {job_id}!"
                            )
                            continue
                        if previously_networked:
                            print_lg(
                                f'Previously saved "{title} | {company}" for networking. '
                                "Rechecking current policy before skipping. Job ID:"
                                f" {job_id}!"
                            )
                        if find_by_class(driver, "jobs-s-apply__application-link", 2):
                            print_lg(
                                f'Already applied to "{title} | {company}" job. Job ID: {job_id}!'
                            )
                            continue
                    except Exception:
                        print_lg(
                            f'Trying to Apply to "{title} | {company}" job. Job ID: {job_id}'
                        )

                    job_link = "https://www.linkedin.com/jobs/view/" + job_id
                    application_link = "Easy Applied"
                    date_applied = "Pending"
                    hr_link = "Unknown"
                    hr_name = "Unknown"
                    connect_request = "In Development"  # Still in development
                    date_listed = "Unknown"
                    skills = "Needs an AI"  # Still in development
                    resume = "Pending"
                    reposted = False
                    questions_list = None
                    screenshot_name = "Not Available"

                    try:
                        rejected_jobs, blacklisted_companies, jobs_top_card = (
                            check_blacklist(
                                rejected_jobs, job_id, company, blacklisted_companies
                            )
                        )
                    except ValueError as e:
                        print_lg(e, "Skipping this job!\n")
                        failed_job(
                            job_id,
                            job_link,
                            resume,
                            date_listed,
                            "Found Blacklisted words in About Company",
                            e,
                            "Skipped",
                            screenshot_name,
                        )
                        skip_count += 1
                        continue
                    except Exception:
                        print_lg("Failed to scroll to About Company!")
                        # print_lg(e)

                    # Hiring Manager info
                    try:
                        hr_info_card = WebDriverWait(driver, 2).until(
                            EC.presence_of_element_located(
                                (By.CLASS_NAME, "hirer-card__hirer-information")
                            )
                        )
                        hr_link = hr_info_card.find_element(
                            By.TAG_NAME, "a"
                        ).get_attribute("href")
                        hr_name = hr_info_card.find_element(By.TAG_NAME, "span").text
                        # if connect_hr:
                        #     driver.switch_to.new_window('tab')
                        #     driver.get(hr_link)
                        #     wait_span_click("More")
                        #     wait_span_click("Connect")
                        #     wait_span_click("Add a note")
                        #     message_box = driver.find_element(By.XPATH, "//textarea")
                        #     message_box.send_keys(connect_request_message)
                        #     if close_tabs: driver.close()
                        #     driver.switch_to.window(linkedIn_tab)
                        # def message_hr(hr_info_card):
                        #     if not hr_info_card: return False
                        #     hr_info_card.find_element(By.XPATH, ".//span[normalize-space()='Message']").click()
                        #     message_box = driver.find_element(By.XPATH, "//div[@aria-label='Write a message…']")
                        #     message_box.send_keys()
                        #     try_xp(driver, "//button[normalize-space()='Send']")
                    except Exception:
                        print_lg(
                            f'HR info was not given for "{title}" with Job ID: {job_id}!'
                        )
                        # print_lg(e)

                    # Calculation of date posted
                    try:
                        # try: time_posted_text = find_by_class(driver, "jobs-unified-top-card__posted-date", 2).text
                        # except:
                        time_posted_text = jobs_top_card.find_element(
                            By.XPATH, './/span[contains(normalize-space(), " ago")]'
                        ).text
                        print("Time Posted: " + time_posted_text)
                        if time_posted_text.__contains__("Reposted"):
                            reposted = True
                            time_posted_text = time_posted_text.replace("Reposted", "")
                        date_listed = calculate_date_posted(time_posted_text)
                    except Exception as e:
                        print_lg("Failed to calculate the date posted!", e)

                    description, experience_required, skip, reason, message = (
                        get_job_description()
                    )
                    if skip:
                        print_lg(message)
                        failed_job(
                            job_id,
                            job_link,
                            resume,
                            date_listed,
                            reason,
                            message,
                            "Skipped",
                            screenshot_name,
                        )
                        rejected_jobs.add(job_id)
                        skip_count += 1
                        continue
                    if description == "Unknown":
                        review_reason = (
                            "Job description unavailable; policy and profile checks cannot"
                            " safely choose auto-apply vs networking."
                        )
                        print_lg(
                            f'Queueing "{title} | {company}" for manual review.'
                            f" {review_reason}"
                        )
                        if save_manual_writeup_job(
                            queued_job_ids,
                            job_id,
                            title,
                            company,
                            searchTerm,
                            work_location,
                            work_style,
                            description,
                            experience_required,
                            skills,
                            hr_name,
                            hr_link,
                            resume,
                            reposted,
                            date_listed,
                            job_link,
                            application_link,
                            questions_list,
                            "policy_review",
                            review_reason,
                        ):
                            manual_writeup_jobs_count += 1
                        skip_count += 1
                        continue

                    below_salary_floor, compensation_snippet = compensation_below_floor(
                        title, description
                    )
                    if below_salary_floor:
                        skip_message = (
                            f'Compensation "{compensation_snippet}" is below your floor of'
                            f" ${minimum_annual_salary_usd:,}/year (~${MINIMUM_HOURLY_RATE_USD}/hr)."
                        )
                        print_lg(skip_message)
                        failed_job(
                            job_id,
                            job_link,
                            resume,
                            date_listed,
                            "Compensation below salary floor",
                            skip_message,
                            "Skipped",
                            screenshot_name,
                        )
                        rejected_jobs.add(job_id)
                        skip_count += 1
                        continue

                    profile_reject_reason = hard_reject_reason(
                        title, company, description
                    )
                    if profile_reject_reason:
                        print_lg(profile_reject_reason)
                        failed_job(
                            job_id,
                            job_link,
                            resume,
                            date_listed,
                            "Hard profile mismatch",
                            profile_reject_reason,
                            "Skipped",
                            screenshot_name,
                        )
                        rejected_jobs.add(job_id)
                        skip_count += 1
                        continue

                    # Policy classifier: networking-edge jobs get saved, not auto-applied.
                    if description != "Unknown":
                        try:
                            classification = classify_job(
                                title,
                                company,
                                description,
                                job_id=job_id,
                                job_link=job_link,
                            )
                            print_lg(
                                f"Job policy classified '{title} | {company}' as: {classification}"
                            )
                            if classification == "network":
                                save_networking_job(
                                    job_id,
                                    title,
                                    company,
                                    work_location,
                                    work_style,
                                    description,
                                    hr_name,
                                    hr_link,
                                    job_link,
                                    date_listed,
                                )
                                networking_job_ids.add(job_id)
                                skip_count += 1
                                continue
                        except Exception as e:
                            print_lg("Job policy classification failed.", e)
                            review_reason = (
                                "Job policy classification failed; manual review required"
                                " before applying."
                            )
                            if save_manual_writeup_job(
                                queued_job_ids,
                                job_id,
                                title,
                                company,
                                searchTerm,
                                work_location,
                                work_style,
                                description,
                                experience_required,
                                skills,
                                hr_name,
                                hr_link,
                                resume,
                                reposted,
                                date_listed,
                                job_link,
                                application_link,
                                questions_list,
                                "policy_review",
                                review_reason,
                            ):
                                manual_writeup_jobs_count += 1
                            skip_count += 1
                            continue

                    if use_AI and description != "Unknown":
                        skills = ai_extract_skills(aiClient, description)

                    uploaded = False
                    # Case 1: Easy Apply Button
                    if try_xp(
                        driver,
                        ".//button[contains(@class,'jobs-apply-button') and contains(@class,"
                        " 'artdeco-button--3') and contains(@aria-label, 'Easy')]",
                    ):
                        try:
                            manual_writeup_saved = False
                            manual_writeup_reason = ""
                            try:
                                errored = ""
                                modal = find_by_class(driver, "jobs-easy-apply-modal")
                                # if description != "Unknown":
                                #     resume = create_custom_resume(description)
                                resume = "Previous resume"
                                job_question_context = build_job_question_context(
                                    title, company, description
                                )
                                next_button = True
                                questions_list = set()
                                manual_writeup_reasons = set()
                                next_counter = 0
                                while next_button:
                                    next_counter += 1
                                    if next_counter >= 15:
                                        if pause_at_failed_question:
                                            screenshot(
                                                driver,
                                                job_id,
                                                "Needed manual intervention for failed question",
                                            )
                                            pyautogui.alert(
                                                "Couldn't answer one or more questions.\nPlease"
                                                ' click "Continue" once done.\nDO NOT CLICK Back,'
                                                " Next or Review button in LinkedIn.\n\n\n\n\nYou"
                                                ' can turn off "Pause at failed question" setting'
                                                " in config.py",
                                                "Help Needed",
                                                "Continue",
                                            )
                                            next_counter = 1
                                            continue
                                        if questions_list:
                                            print_lg(
                                                "Stuck for one or some of the following"
                                                " questions...",
                                                questions_list,
                                            )
                                        screenshot_name = screenshot(
                                            driver, job_id, "Failed at questions"
                                        )
                                        errored = "stuck"
                                        raise Exception(
                                            "Seems like stuck in a continuous loop of next,"
                                            " probably because of new questions."
                                        )
                                    questions_list = answer_questions(
                                        modal,
                                        questions_list,
                                        work_location,
                                        manual_writeup_reasons,
                                        job_question_context,
                                    )
                                    manual_writeup_reason = "; ".join(
                                        sorted(manual_writeup_reasons)
                                    )
                                    if useNewResume and not uploaded:
                                        uploaded, resume = upload_resume(
                                            modal, default_resume_path
                                        )
                                    try:
                                        next_button = modal.find_element(
                                            By.XPATH,
                                            './/span[normalize-space(.)="Review"]',
                                        )
                                    except NoSuchElementException:
                                        next_button = modal.find_element(
                                            By.XPATH,
                                            './/button[contains(span, "Next")]',
                                        )
                                    try:
                                        next_button.click()
                                    except ElementClickInterceptedException:
                                        break  # Happens when it tries to click Next button in About Company photos section
                                    buffer(click_gap)

                            except NoSuchElementException:
                                errored = "nose"
                            finally:
                                if questions_list and errored != "stuck":
                                    print_lg(
                                        "Answered the following questions...",
                                        questions_list,
                                    )
                                    print(
                                        "\n\n"
                                        + "\n".join(
                                            str(question) for question in questions_list
                                        )
                                        + "\n\n"
                                    )
                                if manual_writeup_reason:
                                    save_easy_apply_questions_seen(
                                        questions_list,
                                        job_id,
                                        title,
                                        company,
                                        "manual_writeup",
                                    )
                                    if save_manual_writeup_job(
                                        queued_job_ids,
                                        job_id,
                                        title,
                                        company,
                                        searchTerm,
                                        work_location,
                                        work_style,
                                        description,
                                        experience_required,
                                        skills,
                                        hr_name,
                                        hr_link,
                                        resume,
                                        reposted,
                                        date_listed,
                                        job_link,
                                        "Easy Apply",
                                        questions_list,
                                        "easy_apply_manual_writeup",
                                        manual_writeup_reason,
                                    ):
                                        manual_writeup_jobs_count += 1
                                    discard_job()
                                    manual_writeup_saved = True
                                else:
                                    wait_span_click(driver, "Review", 1, scrollTop=True)
                                    cur_pause_before_submit = pause_before_submit
                                    if errored != "stuck" and cur_pause_before_submit:
                                        decision = pyautogui.confirm(
                                            "1. Please verify your information.\n2. If you edited"
                                            " something, please return to this final screen.\n3. DO NOT"
                                            ' CLICK "Submit Application".\n\n\n\n\nYou can turn off'
                                            ' "Pause before submit" setting in config.py\nTo'
                                            ' TEMPORARILY disable pausing, click "Disable Pause"',
                                            "Confirm your information",
                                            [
                                                "Disable Pause",
                                                "Discard Application",
                                                "Submit Application",
                                            ],
                                        )
                                        if decision == "Discard Application":
                                            raise Exception(
                                                "Job application discarded by user!"
                                            )
                                        pause_before_submit = (
                                            False
                                            if "Disable Pause" == decision
                                            else True
                                        )
                                    follow_company(modal)
                                    if wait_span_click(
                                        driver, "Submit application", 2, scrollTop=True
                                    ):
                                        date_applied = datetime.now()
                                        record_linkedin_application_submitted(
                                            job_id, title, company, application_link
                                        )
                                        save_easy_apply_questions_seen(
                                            questions_list,
                                            job_id,
                                            title,
                                            company,
                                            "submitted",
                                        )
                                        if not wait_span_click(driver, "Done", 2):
                                            actions.send_keys(Keys.ESCAPE).perform()
                                    elif (
                                        errored != "stuck"
                                        and cur_pause_before_submit
                                        and "Yes"
                                        in pyautogui.confirm(
                                            "You submitted the application, didn't you 😒?",
                                            "Failed to find Submit Application!",
                                            ["Yes", "No"],
                                        )
                                    ):
                                        date_applied = datetime.now()
                                        record_linkedin_application_submitted(
                                            job_id, title, company, application_link
                                        )
                                        save_easy_apply_questions_seen(
                                            questions_list,
                                            job_id,
                                            title,
                                            company,
                                            "submitted_user_confirmed",
                                        )
                                        wait_span_click(driver, "Done", 2)
                                    else:
                                        print_lg(
                                            "Since, Submit Application failed, discarding the job"
                                            " application..."
                                        )
                                        if errored == "nose":
                                            raise Exception(
                                                "Failed to click Submit application 😑"
                                            )

                            if manual_writeup_saved:
                                skip_count += 1
                                continue

                        except Exception as e:
                            if daily_easy_apply_limit_reached():
                                dailyEasyApplyLimitReached = True
                                print_lg(
                                    "\n###############  LinkedIn Easy Apply daily limit"
                                    " detected. Stopping run.  ###############\n"
                                )
                                save_easy_apply_questions_seen(
                                    questions_list,
                                    job_id,
                                    title,
                                    company,
                                    "daily_limit",
                                )
                                failed_job(
                                    job_id,
                                    job_link,
                                    resume,
                                    date_listed,
                                    "LinkedIn Easy Apply daily limit reached",
                                    "Daily limit message detected in Easy Apply flow",
                                    application_link,
                                    screenshot_name,
                                )
                                try:
                                    discard_job()
                                except Exception:
                                    pass
                                return
                            print_lg("Failed to Easy apply!")
                            # print_lg(e)
                            critical_error_log("Somewhere in Easy Apply process", e)
                            save_easy_apply_questions_seen(
                                questions_list,
                                job_id,
                                title,
                                company,
                                "failed_easy_apply",
                            )
                            failed_job(
                                job_id,
                                job_link,
                                resume,
                                date_listed,
                                "Problem in Easy Applying",
                                e,
                                application_link,
                                screenshot_name,
                            )
                            failed_count += 1
                            discard_job()
                            continue
                    else:
                        # Case 2: Apply externally
                        skip, application_link, tabs_count = external_apply(
                            pagination_element,
                            job_id,
                            job_link,
                            resume,
                            date_listed,
                            application_link,
                            screenshot_name,
                        )
                        if dailyEasyApplyLimitReached:
                            print_lg(
                                "\n###############  Daily application limit for Easy Apply is"
                                " reached!  ###############\n"
                            )
                            return
                        if skip:
                            continue
                        if save_manual_writeup_job(
                            queued_job_ids,
                            job_id,
                            title,
                            company,
                            searchTerm,
                            work_location,
                            work_style,
                            description,
                            experience_required,
                            skills,
                            hr_name,
                            hr_link,
                            resume,
                            reposted,
                            date_listed,
                            job_link,
                            application_link,
                            questions_list,
                            "external_apply",
                            "External apply link collected for later batch application",
                        ):
                            manual_writeup_jobs_count += 1
                        print_lg(
                            f'Saved external apply "{title} | {company}" for later review.'
                            f" Job ID: {job_id}"
                        )
                        current_count += 1
                        external_jobs_count += 1
                        continue

                    submitted_jobs(
                        job_id,
                        title,
                        company,
                        work_location,
                        work_style,
                        description,
                        experience_required,
                        skills,
                        hr_name,
                        hr_link,
                        resume,
                        reposted,
                        date_listed,
                        date_applied,
                        job_link,
                        application_link,
                        questions_list,
                        connect_request,
                    )
                    if questions_list and date_applied != "Pending":
                        save_easy_apply_questions_seen(
                            questions_list,
                            job_id,
                            title,
                            company,
                            "submitted",
                        )
                    if uploaded:
                        useNewResume = False

                    print_lg(
                        f'Successfully saved "{title} | {company}" job. Job ID: {job_id} info'
                    )
                    current_count += 1
                    applied_jobs.add(job_id)
                    if looks_like_linkedin_application_marker(application_link):
                        if application_link == "Easy Applied":
                            easy_applied_count += 1
                        if configured_daily_linkedin_application_cap_reached():
                            mark_configured_daily_linkedin_application_cap_reached()
                            return
                    else:
                        external_jobs_count += 1

                # Switching to next page
                pages_processed += 1
                if (
                    MAX_RESULT_PAGES_PER_SEARCH > 0
                    and pages_processed >= MAX_RESULT_PAGES_PER_SEARCH
                ):
                    print_lg(
                        "Reached configured result-page limit for this search"
                        f" ({pages_processed}/{MAX_RESULT_PAGES_PER_SEARCH})."
                    )
                    break
                if not go_to_next_results_page(pagination_element, current_page):
                    break

        except Exception as e:
            print_lg("Failed while processing the current job listings page!")
            critical_error_log("In Applier", e)
            if browser_session_lost(e):
                print_lg(
                    "Browser session was lost; ending this run so the launcher can restart."
                )
                raise
            try:
                print_lg(f"Current URL: {driver.current_url}")
            except Exception as url_error:
                critical_error_log("Failed to read current URL", url_error)
            try:
                print_lg(f"Current page title: {driver.title}")
            except Exception as title_error:
                critical_error_log("Failed to read current page title", title_error)


def run(total_runs: int) -> int:
    if dailyEasyApplyLimitReached:
        return total_runs
    print_lg(
        "\n########################################################################################################################\n"
    )
    print_lg(f"Date and Time: {datetime.now()}")
    print_lg(f"Cycle number: {total_runs}")
    print_lg(
        f"Currently looking for jobs posted within '{date_posted}' and sorting them by '{sort_by}'"
    )
    apply_to_jobs(search_terms)
    print_lg(
        "########################################################################################################################\n"
    )
    if not dailyEasyApplyLimitReached:
        print_lg(f"Sleeping for {run_sleep_seconds} seconds before next run...")
        sleep(run_sleep_seconds)
    buffer(3)
    return total_runs + 1


chatGPT_tab = False
linkedIn_tab = False


def main() -> None:
    exit_code = 0
    try:
        global linkedIn_tab, tabs_count, useNewResume, aiClient
        alert_title = "Error Occurred. Closing Browser!"
        total_runs = 1
        validate_config()

        if not os.path.exists(default_resume_path):
            if desktop_alerts_enabled():
                pyautogui.alert(
                    text=(
                        'Your default resume "{}" is missing! Please update it\'s folder path'
                        ' "default_resume_path" in config.py\n\nOR\n\nAdd a resume with exact name'
                        " and path (check for spelling mistakes including cases).\n\n\nFor now the"
                        " bot will continue using your previous upload from LinkedIn!".format(
                            default_resume_path
                        )
                    ),
                    title="Missing Resume",
                    button="OK",
                )
            useNewResume = False

        # Login to LinkedIn
        tabs_count = len(driver.window_handles)
        driver.get("https://www.linkedin.com/login")
        if not is_logged_in_LN():
            login_LN()

        linkedIn_tab = driver.current_window_handle

        # # Login to ChatGPT in a new tab for resume customization
        # if use_resume_generator:
        #     try:
        #         driver.switch_to.new_window('tab')
        #         driver.get("https://chat.openai.com/")
        #         if not is_logged_in_GPT(): login_GPT()
        #         open_resume_chat()
        #         global chatGPT_tab
        #         chatGPT_tab = driver.current_window_handle
        #     except Exception as e:
        #         print_lg("Opening OpenAI chatGPT tab failed!")
        if use_AI:
            aiClient = ai_create_openai_client()

        # Start applying to jobs
        driver.switch_to.window(linkedIn_tab)
        total_runs = run(total_runs)
        while run_non_stop:
            if cycle_date_posted:
                date_options = ["Any time", "Past month", "Past week", "Past 24 hours"]
                global date_posted
                date_posted = (
                    date_options[
                        (
                            date_options.index(date_posted) + 1
                            if date_options.index(date_posted) + 1 > len(date_options)
                            else -1
                        )
                    ]
                    if stop_date_cycle_at_24hr
                    else date_options[
                        (
                            0
                            if date_options.index(date_posted) + 1 >= len(date_options)
                            else date_options.index(date_posted) + 1
                        )
                    ]
                )
            if alternate_sortby:
                global sort_by
                sort_by = (
                    "Most recent" if sort_by == "Most relevant" else "Most relevant"
                )
                total_runs = run(total_runs)
                sort_by = (
                    "Most recent" if sort_by == "Most relevant" else "Most relevant"
                )
            total_runs = run(total_runs)
            if dailyEasyApplyLimitReached:
                break

    except NoSuchWindowException as e:
        exit_code = 1
        critical_error_log("In Applier Main", e)
    except Exception as e:
        exit_code = 1
        critical_error_log("In Applier Main", e)
        if not run_in_background and desktop_alerts_enabled():
            pyautogui.alert(e, alert_title)
    finally:
        print_lg("\n\nTotal runs:                     {}".format(total_runs))
        print_lg("Jobs Easy Applied:              {}".format(easy_applied_count))
        print_lg("External job links collected:   {}".format(external_jobs_count))
        print_lg("Jobs queued for manual writeup: {}".format(manual_writeup_jobs_count))
        print_lg("                              ----------")
        print_lg(
            "Total applied or collected:     {}".format(
                easy_applied_count + external_jobs_count
            )
        )
        print_lg("\nFailed jobs:                    {}".format(failed_count))
        print_lg("Irrelevant jobs skipped:        {}\n".format(skip_count))
        if randomly_answered_questions:
            print_lg(
                "\n\nQuestions randomly answered:\n  {}  \n\n".format(
                    ";\n".join(
                        str(question) for question in randomly_answered_questions
                    )
                )
            )
        quote = choice(
            [
                "You're one step closer than before.",
                "All the best with your future interviews.",
                "Keep up with the progress. You got this.",
                "If you're tired, learn to take rest but never give up.",
                (
                    "Success is not final, failure is not fatal: It is the courage to continue that"
                    " counts. - Winston Churchill"
                ),
                (
                    "Believe in yourself and all that you are. Know that there is something inside"
                    " you that is greater than any obstacle. - Christian D. Larson"
                ),
                (
                    "Every job is a self-portrait of the person who does it. Autograph your work"
                    " with excellence."
                ),
                (
                    "The only way to do great work is to love what you do. If you haven't found it"
                    " yet, keep looking. Don't settle. - Steve Jobs"
                ),
                "Opportunities don't happen, you create them. - Chris Grosser",
                (
                    "The road to success and the road to failure are almost exactly the same. The"
                    " difference is perseverance."
                ),
                (
                    "Obstacles are those frightful things you see when you take your eyes off your"
                    " goal. - Henry Ford"
                ),
                (
                    "The only limit to our realization of tomorrow will be our doubts of today. -"
                    " Franklin D. Roosevelt"
                ),
            ]
        )
        msg = (
            f"\n{quote}\n\n\nBest regards,\nSai Vignesh"
            " Golla\nhttps://www.linkedin.com/in/saivigneshgolla/\n\n"
        )
        if not run_in_background and desktop_alerts_enabled():
            pyautogui.alert(msg, "Exiting..")
        print_lg(msg, "Closing the browser...")
        if tabs_count >= 10:
            msg = (
                "NOTE: IF YOU HAVE MORE THAN 10 TABS OPENED, PLEASE CLOSE OR BOOKMARK THEM!\n\nOr"
                " it's highly likely that application will just open browser and not do anything"
                " next time!"
            )
            if not run_in_background and desktop_alerts_enabled():
                pyautogui.alert(msg, "Info")
            print_lg("\n" + msg)
        ai_close_openai_client(aiClient)
        try:
            driver.quit()
        except Exception as e:
            critical_error_log("When quitting...", e)
        if exit_code:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
