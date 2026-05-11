"""ZipRecruiter job applier - has one-click apply for many jobs."""

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


class ZipRecruiterApplier(SiteApplier):
    SITE_NAME = "ziprecruiter"
    BASE_URL = "https://www.ziprecruiter.com"

    def login(self) -> None:
        self.driver.get(f"{self.BASE_URL}/login")
        random_delay(2, 3)

        # Check if already logged in
        if "login" not in self.driver.current_url:
            logger.info("Already logged in to ZipRecruiter")
            return

        print("\n>>> Please log in to ZipRecruiter in the browser window.")
        print(">>> Waiting 60 seconds for you to log in...")
        for _ in range(30):
            time.sleep(2)
            try:
                if "login" not in self.driver.current_url:
                    logger.info("ZipRecruiter login detected")
                    return
            except Exception:
                pass
        logger.info("ZipRecruiter login wait complete (proceeding regardless)")

    def search_jobs(self, search_term: str) -> list[JobListing]:
        jobs: list[JobListing] = []
        location = self.search_config.location

        for page in range(3):
            url = (
                f"{self.BASE_URL}/jobs-search"
                f"?search={quote_plus(search_term)}"
                f"&location={quote_plus(location)}"
                f"&radius={self.search_config.radius_miles}"
                f"&days={self.search_config.date_posted_days}"
                f"&page={page + 1}"
            )
            self.driver.get(url)
            random_delay(2, 4)

            page_jobs = self._extract_job_cards()
            if not page_jobs:
                logger.info("No more jobs on page %d for '%s'", page + 1, search_term)
                break

            for job in page_jobs:
                self._populate_description(job)
                jobs.append(job)

            logger.info(
                "Page %d: found %d jobs for '%s'", page + 1, len(page_jobs), search_term
            )

        return jobs

    def _extract_job_cards(self) -> list[JobListing]:
        jobs: list[JobListing] = []
        random_delay(1, 2)

        cards = self.driver.find_elements(
            By.CSS_SELECTOR,
            "article.job_result, div.job_result_two_pane, .jobList-item",
        )

        for card in cards:
            try:
                title_el = card.find_element(
                    By.CSS_SELECTOR, "a.job_link, h2 a, a[data-testid='job-title']"
                )
                title = title_el.text.strip()
                href = title_el.get_attribute("href") or ""

                # Job ID from URL or data attribute
                job_id = card.get_attribute("data-job-id") or ""
                if not job_id:
                    match = re.search(r"/jobs?/([^/?]+)", href)
                    job_id = match.group(1) if match else href

                company = ""
                try:
                    company = card.find_element(
                        By.CSS_SELECTOR,
                        "a.t_org_link, [data-testid='company-name'], .job_org",
                    ).text.strip()
                except NoSuchElementException:
                    pass

                location = ""
                try:
                    location = card.find_element(
                        By.CSS_SELECTOR, ".job_location, [data-testid='job-location']"
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
            except NoSuchElementException:
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
                        ".job_description, .jobDescriptionSection, [data-testid='job-description']",
                    )
                )
            )
            job.description = desc_el.text
        except (TimeoutException, NoSuchElementException):
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
            random_delay(1, 2)

            return self._complete_application()
        finally:
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(original_window)

    def _find_apply_button(self):
        selectors = [
            'button[data-testid="apply-button"]',
            "button.apply_button",
            "a.apply_button",
            '//button[contains(text(), "1-Click Apply") or contains(text(), "Apply Now") or contains(text(), "Easy Apply")]',
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
        """ZipRecruiter often does 1-click apply with pre-filled profile."""
        max_steps = 8
        for step in range(max_steps):
            random_delay(0.5, 1)

            if self._is_complete():
                return True

            self._fill_form_fields()
            self._handle_resume_upload()

            if not self._click_next_or_submit():
                break

        return self._is_complete()

    def _fill_form_fields(self) -> None:
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR,
            "input[type='text'], input[type='tel'], input[type='email'], textarea",
        )
        for inp in inputs:
            if not inp.is_displayed():
                continue
            if inp.get_attribute("value"):
                continue

            label = (
                inp.get_attribute("aria-label")
                or inp.get_attribute("placeholder")
                or inp.get_attribute("name")
                or ""
            ).lower()

            if "first" in label and "name" in label:
                fill_field(self.driver, inp, self.profile.first_name)
            elif "last" in label and "name" in label:
                fill_field(self.driver, inp, self.profile.last_name)
            elif "name" in label:
                fill_field(self.driver, inp, self.profile.full_name)
            elif "email" in label:
                fill_field(self.driver, inp, self.profile.email)
            elif "phone" in label:
                fill_field(self.driver, inp, self.profile.phone)
            elif "linkedin" in label:
                fill_field(self.driver, inp, self.profile.linkedin_url)

    def _handle_resume_upload(self) -> None:
        try:
            file_input = self.driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            file_input.send_keys(self.profile.resume_path)
            random_delay(1, 2)
        except NoSuchElementException:
            pass

    def _click_next_or_submit(self) -> bool:
        for text in ["Submit", "Apply", "1-Click Apply", "Next", "Continue"]:
            try:
                btns = self.driver.find_elements(
                    By.XPATH, f'//button[contains(text(), "{text}")]'
                )
                for btn in btns:
                    if btn.is_displayed() and btn.is_enabled():
                        safe_click(self.driver, btn)
                        random_delay(0.5, 1)
                        return True
            except Exception:
                pass
        return False

    def _is_complete(self) -> bool:
        markers = [
            "application submitted",
            "application has been sent",
            "you've applied",
            "successfully applied",
            "thank you for applying",
            "application complete",
        ]
        page_text = self.driver.page_source.lower()
        return any(m in page_text for m in markers)
