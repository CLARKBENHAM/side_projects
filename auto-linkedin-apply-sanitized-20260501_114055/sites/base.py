"""Base classes and shared utilities for multi-site job applier."""

import csv
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import uniform

from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from modules.ai.cli_classifier import hard_reject_reason
from modules.experience_requirements import estimate_required_years
from sites.classifier import classify_job, save_networking_job
from sites.config import (
    ApplicantProfile,
    SearchConfig,
    CHROME_BETA_PATH,
    BAD_WORDS,
    SKIP_COMPANIES,
)

logger = logging.getLogger(__name__)


@dataclass
class JobListing:
    job_id: str
    title: str
    company: str
    location: str
    description: str = ""
    salary: str = ""
    url: str = ""
    date_posted: str = ""
    work_style: str = ""  # remote, hybrid, onsite
    source: str = ""  # indeed, dice, ziprecruiter, wellfound


class SiteApplier(ABC):
    """Base class for site-specific job appliers."""

    SITE_NAME: str = "base"
    BASE_URL: str = ""

    def __init__(
        self,
        profile: ApplicantProfile,
        search_config: SearchConfig,
        driver: WebDriver | None = None,
    ):
        self.profile = profile
        self.search_config = search_config
        self.driver = driver or self._create_driver()
        self.wait = WebDriverWait(self.driver, 10)
        self.applied_ids: set[str] = set()
        self.results = ApplyResults(site=self.SITE_NAME)
        self._load_applied_ids()

    def _create_driver(self) -> WebDriver:
        options = Options()
        options.binary_location = CHROME_BETA_PATH
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # Use a persistent profile so logins survive across runs
        profile_dir = os.path.expanduser("~/.chrome-bot-profile")
        os.makedirs(profile_dir, exist_ok=True)
        options.add_argument(f"--user-data-dir={profile_dir}")
        options.add_argument(f"--profile-directory={self.SITE_NAME}")
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        return driver

    def _load_applied_ids(self) -> None:
        csv_path = self.results.csv_path
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row:
                        self.applied_ids.add(row[0])
            logger.info(
                "Loaded %d previously applied job IDs from %s",
                len(self.applied_ids),
                csv_path,
            )

    def run(self) -> "ApplyResults":
        """Main entry point: login, search, classify, apply."""
        logger.info("Starting %s applier", self.SITE_NAME)
        self.login()
        for term in self.search_config.search_terms:
            logger.info("Searching for: %s", term)
            try:
                jobs = self.search_jobs(term)
                for job in jobs:
                    self._process_job(job)
            except Exception:
                logger.exception("Error searching for '%s'", term)
        self.results.log_summary()
        return self.results

    def _process_job(self, job: JobListing) -> None:
        if job.job_id in self.applied_ids:
            logger.debug("Already applied to %s, skipping", job.job_id)
            self.results.skipped += 1
            return

        if self._should_skip(job):
            self.results.skipped += 1
            return

        if not self._safe_to_apply(job):
            return

        try:
            success = self.apply_to_job(job)
            if success:
                self.results.applied += 1
                self.applied_ids.add(job.job_id)
                self._save_applied(job)
                logger.info("Applied to '%s | %s'", job.title, job.company)
            else:
                self.results.failed += 1
                logger.warning("Failed to apply to '%s | %s'", job.title, job.company)
        except Exception:
            self.results.failed += 1
            logger.exception("Error applying to '%s | %s'", job.title, job.company)
        finally:
            self.results.attempted += 1

    def _should_skip(self, job: JobListing) -> bool:
        company_lower = job.company.lower()
        for skip in SKIP_COMPANIES:
            if skip.lower() in company_lower:
                logger.info("Skipping blacklisted company: %s", job.company)
                return True
        # Check bad words in title only (description is too noisy for substring matching)
        title_lower = job.title.lower()
        for word in BAD_WORDS:
            if re.search(r"\b" + re.escape(word.lower()) + r"\b", title_lower):
                logger.info("Skipping job with bad word '%s': %s", word, job.title)
                return True
        return False

    def _safe_to_apply(self, job: JobListing) -> bool:
        description = job.description.strip()
        if not description:
            logger.warning(
                "Skipping '%s | %s': no description available for policy review",
                job.title,
                job.company,
            )
            self.results.skipped += 1
            return False

        description_lower = description.lower()
        for word in BAD_WORDS:
            if word.lower() in description_lower:
                logger.info(
                    "Skipping '%s | %s': description contains bad word '%s'",
                    job.title,
                    job.company,
                    word,
                )
                self.results.skipped += 1
                return False

        reject_reason = hard_reject_reason(job.title, job.company, description)
        if reject_reason:
            logger.info("Skipping '%s | %s': %s", job.title, job.company, reject_reason)
            self.results.skipped += 1
            return False

        required_years = estimate_required_years(description)
        if (
            required_years is not None
            and required_years > self.profile.years_experience
        ):
            logger.info(
                "Skipping '%s | %s': requires %s years > profile %s years",
                job.title,
                job.company,
                required_years,
                self.profile.years_experience,
            )
            self.results.skipped += 1
            return False

        try:
            classification = classify_job(
                job.title,
                job.company,
                description,
                job_id=job.job_id,
                job_link=job.url,
            )
        except Exception:
            logger.exception(
                "Policy classification failed for '%s | %s'; skipping",
                job.title,
                job.company,
            )
            self.results.skipped += 1
            return False

        logger.info("Policy: '%s | %s' -> %s", job.title, job.company, classification)
        if classification == "network":
            save_networking_job(
                job_id=job.job_id,
                title=job.title,
                company=job.company,
                work_location=job.location,
                work_style=job.work_style,
                description=description,
                hr_name="",
                hr_link="",
                job_link=job.url,
                date_listed=job.date_posted,
                source=self.SITE_NAME,
            )
            self.results.networked += 1
            return False

        return True

    def _save_applied(self, job: JobListing) -> None:
        csv_path = self.results.csv_path
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Job ID",
                    "Title",
                    "Company",
                    "Location",
                    "URL",
                    "Date Applied",
                    "Source",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "Job ID": job.job_id,
                    "Title": job.title,
                    "Company": job.company,
                    "Location": job.location,
                    "URL": job.url,
                    "Date Applied": datetime.now().isoformat(),
                    "Source": self.SITE_NAME,
                }
            )

    @abstractmethod
    def login(self) -> None:
        """Login to the job site. May prompt user for manual login."""

    @abstractmethod
    def search_jobs(self, search_term: str) -> list[JobListing]:
        """Search for jobs and return listings with descriptions populated."""

    @abstractmethod
    def apply_to_job(self, job: JobListing) -> bool:
        """Apply to a single job. Returns True on success."""

    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception:
            pass


@dataclass
class ApplyResults:
    site: str
    attempted: int = 0
    applied: int = 0
    failed: int = 0
    skipped: int = 0
    networked: int = 0

    @property
    def csv_path(self) -> str:
        Path("all excels").mkdir(exist_ok=True)
        return f"all excels/{self.site}_applied_history.csv"

    def log_summary(self) -> None:
        logger.info(
            "\n=== %s Results ===\n"
            "  Attempted: %d\n  Applied: %d\n  Failed: %d\n"
            "  Skipped: %d\n  Networking: %d",
            self.site.upper(),
            self.attempted,
            self.applied,
            self.failed,
            self.skipped,
            self.networked,
        )


# --- Selenium helpers ---


def safe_click(driver: WebDriver, element: WebElement, retries: int = 2) -> bool:
    for attempt in range(retries + 1):
        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", element
            )
            time.sleep(0.3)
            element.click()
            return True
        except (ElementClickInterceptedException, ElementNotInteractableException):
            if attempt < retries:
                time.sleep(0.5)
                # Try JS click as fallback
                try:
                    driver.execute_script("arguments[0].click();", element)
                    return True
                except Exception:
                    pass
        except StaleElementReferenceException:
            return False
    return False


def wait_and_click(driver: WebDriver, by: str, value: str, timeout: float = 10) -> bool:
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )
        return safe_click(driver, el)
    except TimeoutException:
        return False


def fill_field(driver: WebDriver, element: WebElement, value: str) -> None:
    element.clear()
    element.send_keys(value)


def find_and_fill(
    driver: WebDriver, by: str, value: str, text: str, timeout: float = 5
) -> bool:
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        fill_field(driver, el, text)
        return True
    except TimeoutException:
        return False


def random_delay(low: float = 0.5, high: float = 1.5) -> None:
    time.sleep(uniform(low, high))
