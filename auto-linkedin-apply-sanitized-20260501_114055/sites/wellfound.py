"""Wellfound (formerly AngelList Talent) job applier - startup-focused."""

import logging
import re
import time
from urllib.parse import quote_plus

from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from sites.base import (
    JobListing,
    SiteApplier,
    fill_field,
    random_delay,
    safe_click,
)

logger = logging.getLogger(__name__)


class WellfoundApplier(SiteApplier):
    SITE_NAME = "wellfound"
    BASE_URL = "https://wellfound.com"

    def login(self) -> None:
        self.driver.get(f"{self.BASE_URL}/login")
        random_delay(2, 3)

        # Check if already logged in
        try:
            self.driver.find_element(
                By.CSS_SELECTOR, '[data-test="UserMenu"], .styles_component__nav'
            )
            logger.info("Already logged in to Wellfound")
            return
        except NoSuchElementException:
            pass

        print("\n>>> Please log in to Wellfound in the browser window.")
        print(">>> Waiting 60 seconds for you to log in...")
        for _ in range(30):
            time.sleep(2)
            try:
                self.driver.find_element(
                    By.CSS_SELECTOR, '[data-test="UserMenu"], .styles_component__nav'
                )
                logger.info("Wellfound login detected")
                return
            except NoSuchElementException:
                pass
        logger.info("Wellfound login wait complete (proceeding regardless)")

    def search_jobs(self, search_term: str) -> list[JobListing]:
        jobs: list[JobListing] = []

        # Wellfound uses role-based browsing
        url = (
            f"{self.BASE_URL}/jobs"
            f"?q={quote_plus(search_term)}"
            f"&location={quote_plus(self.search_config.location)}"
        )
        self.driver.get(url)
        random_delay(3, 5)

        # Scroll to load more jobs
        for _ in range(3):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            random_delay(2, 3)

        page_jobs = self._extract_job_cards()
        for job in page_jobs:
            self._populate_description(job)
            jobs.append(job)

        logger.info("Found %d jobs for '%s' on Wellfound", len(page_jobs), search_term)
        return jobs

    def _extract_job_cards(self) -> list[JobListing]:
        jobs: list[JobListing] = []

        # Wellfound job cards
        cards = self.driver.find_elements(
            By.CSS_SELECTOR,
            '[data-test="StartupResult"], .styles_result__rECEo, div[class*="JobSearchResult"]',
        )

        if not cards:
            # Try broader selector
            cards = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='/jobs/']")

        for card in cards:
            try:
                if card.tag_name == "a":
                    title = card.text.strip().split("\n")[0]
                    href = card.get_attribute("href") or ""
                else:
                    title_el = card.find_element(
                        By.CSS_SELECTOR, "a[href*='/jobs/'], h2, h3"
                    )
                    title = title_el.text.strip()
                    href = title_el.get_attribute("href") or ""

                if not title or not href:
                    continue

                # Job ID from URL
                match = re.search(r"/jobs/(\d+)", href)
                job_id = match.group(1) if match else href

                company = ""
                try:
                    company = card.find_element(
                        By.CSS_SELECTOR,
                        "[class*='company'], [data-test='startup-name']",
                    ).text.strip()
                except NoSuchElementException:
                    # Parse from card text
                    lines = card.text.strip().split("\n")
                    if len(lines) > 1:
                        company = lines[1]

                location = ""
                try:
                    location = card.find_element(
                        By.CSS_SELECTOR, "[class*='location']"
                    ).text.strip()
                except NoSuchElementException:
                    pass

                jobs.append(
                    JobListing(
                        job_id=job_id,
                        title=title,
                        company=company,
                        location=location,
                        url=href,
                        source=self.SITE_NAME,
                    )
                )
            except (NoSuchElementException, IndexError):
                continue

        return jobs

    def _populate_description(self, job: JobListing) -> None:
        if not job.url:
            return
        original_window = self.driver.current_window_handle
        try:
            self.driver.execute_script("window.open(arguments[0], '_blank');", job.url)
            self.driver.switch_to.window(self.driver.window_handles[-1])
            random_delay(1, 2)

            desc_el = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "[class*='description'], [data-test='job-description'], .job-description",
                    )
                )
            )
            job.description = desc_el.text
        except (TimeoutException, NoSuchElementException):
            try:
                main = self.driver.find_element(By.CSS_SELECTOR, "main, [role='main']")
                job.description = main.text[:3000]
            except NoSuchElementException:
                logger.debug("Could not find description for %s", job.job_id)
        finally:
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(original_window)

    def apply_to_job(self, job: JobListing) -> bool:
        original_window = self.driver.current_window_handle
        if job.url:
            self.driver.execute_script("window.open(arguments[0], '_blank');", job.url)
            self.driver.switch_to.window(self.driver.window_handles[-1])
        random_delay(2, 3)

        try:
            apply_btn = self._find_apply_button()
            if not apply_btn:
                logger.info("No apply button for '%s'", job.title)
                return False

            safe_click(self.driver, apply_btn)
            random_delay(2, 3)

            return self._complete_application()
        finally:
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(original_window)

    def _find_apply_button(self):
        selectors = [
            'button[data-test="apply-button"]',
            '//button[contains(text(), "Apply")]',
            '//a[contains(text(), "Apply")]',
            "button.apply-button",
        ]
        for sel in selectors:
            try:
                if sel.startswith("//"):
                    btn = self.driver.find_element(By.XPATH, sel)
                else:
                    btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                if btn.is_displayed():
                    return btn
            except NoSuchElementException:
                continue
        return None

    def _complete_application(self) -> bool:
        """Handle Wellfound's apply flow - usually a modal with resume + note."""
        random_delay(1, 2)

        # Handle resume upload if needed
        self._handle_resume_upload()

        # Fill in cover note if present
        try:
            note_area = self.driver.find_element(
                By.CSS_SELECTOR,
                "textarea[name*='note'], textarea[name*='cover'], textarea[placeholder*='note']",
            )
            if not note_area.get_attribute("value"):
                fill_field(
                    self.driver,
                    note_area,
                    (
                        f"Hi, I'm {self.profile.first_name} - a software engineer with "
                        f"{self.profile.years_experience} years of software experience "
                        "and hands-on AI/ML project experience. "
                        f"I'd love to learn more about this role."
                    ),
                )
        except NoSuchElementException:
            pass

        # Submit
        for text in ["Submit Application", "Submit", "Apply", "Send Application"]:
            try:
                btns = self.driver.find_elements(
                    By.XPATH, f'//button[contains(text(), "{text}")]'
                )
                for btn in btns:
                    if btn.is_displayed() and btn.is_enabled():
                        safe_click(self.driver, btn)
                        random_delay(2, 3)
                        return self._is_complete()
            except Exception:
                pass

        return self._is_complete()

    def _handle_resume_upload(self) -> None:
        try:
            file_input = self.driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            file_input.send_keys(self.profile.resume_path)
            random_delay(1, 2)
        except NoSuchElementException:
            pass

    def _is_complete(self) -> bool:
        markers = [
            "application submitted",
            "application sent",
            "you've applied",
            "successfully",
            "thank you",
        ]
        page_text = self.driver.page_source.lower()
        return any(m in page_text for m in markers)
