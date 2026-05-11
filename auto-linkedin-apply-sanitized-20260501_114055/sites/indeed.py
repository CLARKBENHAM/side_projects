"""Indeed job applier using Indeed's "Apply now" / "Easy Apply" flow."""

import logging
import re
import time
from urllib.parse import quote_plus

from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from modules.application_answer_safety import (
    deterministic_answer_for_question,
    recommended_compensation_answer,
)
from sites.base import (
    JobListing,
    SiteApplier,
    fill_field,
    random_delay,
    safe_click,
    wait_and_click,
)

logger = logging.getLogger(__name__)


class IndeedApplier(SiteApplier):
    SITE_NAME = "indeed"
    BASE_URL = "https://www.indeed.com"

    def login(self) -> None:
        self.driver.get(f"{self.BASE_URL}")
        random_delay(2, 3)
        # Indeed doesn't strictly require login for searching, but does for applying.
        # Check if already logged in by looking for account icon.
        try:
            self.driver.find_element(
                By.CSS_SELECTOR, '[data-gnav-element-name="AccountMenu"]'
            )
            logger.info("Already logged in to Indeed")
            return
        except NoSuchElementException:
            pass

        # Navigate to sign in
        try:
            wait_and_click(
                self.driver, By.CSS_SELECTOR, 'a[href*="account/login"]', timeout=5
            )
        except Exception:
            self.driver.get(f"{self.BASE_URL}/account/login")

        random_delay(1, 2)
        print("\n>>> Please log in to Indeed in the browser window.")
        print(">>> Waiting 60 seconds for you to log in...")
        # Poll for login instead of blocking on input()
        for _ in range(30):
            time.sleep(2)
            try:
                self.driver.find_element(
                    By.CSS_SELECTOR, '[data-gnav-element-name="AccountMenu"]'
                )
                logger.info("Indeed login detected")
                return
            except NoSuchElementException:
                pass
        logger.info("Indeed login wait complete (proceeding regardless)")

    def search_jobs(self, search_term: str) -> list[JobListing]:
        jobs: list[JobListing] = []
        location = self.search_config.location

        # Build search URL
        params = {
            "q": search_term,
            "l": location,
            "radius": str(self.search_config.radius_miles),
            "fromage": str(self.search_config.date_posted_days),
            "sort": "date",
        }
        salary = self.search_config.salary_min
        if salary:
            params["salary"] = str(salary)

        query_str = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        search_url = f"{self.BASE_URL}/jobs?{query_str}"

        for page in range(3):  # max 3 pages per search term
            url = f"{search_url}&start={page * 10}"
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
            By.CSS_SELECTOR, "div.job_seen_beacon, div.cardOutline"
        )
        for card in cards:
            try:
                title_el = card.find_element(
                    By.CSS_SELECTOR, "h2.jobTitle a, a.jcs-JobTitle"
                )
                title = title_el.text.strip()
                href = title_el.get_attribute("href") or ""

                # Extract job ID from data attribute or URL
                job_id = card.get_attribute("data-jk") or ""
                if not job_id:
                    match = re.search(r"jk=([a-f0-9]+)", href)
                    job_id = match.group(1) if match else href

                company = ""
                try:
                    company = card.find_element(
                        By.CSS_SELECTOR,
                        "[data-testid='company-name'], span.companyName",
                    ).text.strip()
                except NoSuchElementException:
                    pass

                location = ""
                try:
                    location = card.find_element(
                        By.CSS_SELECTOR,
                        "[data-testid='text-location'], div.companyLocation",
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
        """Load job description by opening in new tab (preserves search page)."""
        if not job.url:
            return
        original_window = self.driver.current_window_handle
        try:
            self.driver.execute_script("window.open(arguments[0], '_blank');", job.url)
            self.driver.switch_to.window(self.driver.window_handles[-1])
            random_delay(1, 2)

            try:
                desc_el = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.ID, "jobDescriptionText"))
                )
                job.description = desc_el.text
            except TimeoutException:
                logger.debug("Could not find description for %s", job.job_id)
        except Exception:
            logger.debug("Error populating description for %s", job.job_id)
        finally:
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(original_window)

    def apply_to_job(self, job: JobListing) -> bool:
        # Open job in a new tab to preserve search page and login state
        original_window = self.driver.current_window_handle
        self._current_job = job
        if job.url:
            self.driver.execute_script("window.open(arguments[0], '_blank');", job.url)
            self.driver.switch_to.window(self.driver.window_handles[-1])
        random_delay(2, 3)

        try:
            # Look for the "Apply now" button (Indeed Easy Apply)
            apply_button = self._find_apply_button()
            if not apply_button:
                logger.info("No Easy Apply for '%s' - external application", job.title)
                return False

            safe_click(self.driver, apply_button)
            random_delay(1, 2)

            # Handle the multi-step application form
            return self._complete_application()
        finally:
            # Close this tab and return to search results
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(original_window)

    def _find_apply_button(self) -> WebElement | None:
        # Try CSS selectors first
        for sel in [
            "button#indeedApplyButton",
            'button[id*="applyButton"]',
            "button.ia-IndeedApplyButton",
            'button[class*="apply"]',
        ]:
            try:
                btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                if btn.is_displayed():
                    return btn
            except NoSuchElementException:
                continue
        # Scan all buttons for "apply" text (handles nested spans)
        for btn in self.driver.find_elements(By.TAG_NAME, "button"):
            try:
                text = btn.text.strip().lower()
                if btn.is_displayed() and "apply" in text:
                    return btn
            except Exception:
                continue
        return None

    def _complete_application(self) -> bool:
        """Walk through Indeed's multi-step apply form."""
        max_steps = 10
        for step in range(max_steps):
            random_delay(0.5, 1)

            # Fill any visible form fields
            self._fill_form_fields()

            # Handle resume upload
            self._handle_resume_upload()

            # Look for Continue/Next/Submit button
            if self._click_submit():
                random_delay(1, 2)
                # Check if we've reached the confirmation
                if self._is_application_complete():
                    return True
                continue

            if self._click_continue():
                continue

            # If no button found, we might be stuck
            logger.debug("No continue/submit button found at step %d", step)
            break

        return self._is_application_complete()

    def _fill_form_fields(self) -> None:
        """Fill visible text inputs with profile data."""
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR,
            "input[type='text'], input[type='tel'], input[type='email']",
        )
        for inp in inputs:
            if not inp.is_displayed() or inp.get_attribute("value"):
                continue
            label = self._get_label_for(inp).lower()
            if not label:
                continue

            if "first" in label and "name" in label:
                fill_field(self.driver, inp, self.profile.first_name)
            elif "last" in label and "name" in label:
                fill_field(self.driver, inp, self.profile.last_name)
            elif "full" in label and "name" in label:
                fill_field(self.driver, inp, self.profile.full_name)
            elif "email" in label:
                fill_field(self.driver, inp, self.profile.email)
            elif "phone" in label or "mobile" in label:
                fill_field(self.driver, inp, self.profile.phone)
            elif "city" in label or "location" in label:
                fill_field(self.driver, inp, self.profile.city)
            elif "zip" in label or "postal" in label:
                fill_field(self.driver, inp, self.profile.zipcode)
            elif "linkedin" in label:
                fill_field(self.driver, inp, self.profile.linkedin_url)
            elif "website" in label or "portfolio" in label:
                fill_field(self.driver, inp, self.profile.website)
            elif "github" in label:
                fill_field(self.driver, inp, self.profile.github)
            elif "experience" in label or "years" in label:
                answer = deterministic_answer_for_question(
                    label,
                    "text",
                    generic_years_of_experience=str(self.profile.years_experience),
                )
                if answer:
                    fill_field(self.driver, inp, answer[0])
            elif "salary" in label:
                job = getattr(self, "_current_job", None)
                job_context = ""
                if job:
                    job_context = "\n".join(
                        [job.title, job.company, job.description[:3000]]
                    )
                answer = recommended_compensation_answer(
                    label, job_context, default_annual=self.profile.desired_salary
                )
                fill_field(
                    self.driver,
                    inp,
                    answer[0] if answer else str(self.profile.desired_salary),
                )

    def _get_label_for(self, element: WebDriverWait) -> str:
        """Find the label text associated with a form element."""
        el_id = element.get_attribute("id") or ""
        if el_id:
            try:
                label = self.driver.find_element(
                    By.CSS_SELECTOR, f'label[for="{el_id}"]'
                )
                return label.text
            except NoSuchElementException:
                pass
        # Try aria-label
        aria = element.get_attribute("aria-label") or ""
        if aria:
            return aria
        # Try placeholder
        return element.get_attribute("placeholder") or ""

    def _handle_resume_upload(self) -> None:
        try:
            file_input = self.driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            if file_input.is_displayed() or file_input.get_attribute("name"):
                file_input.send_keys(self.profile.resume_path)
                random_delay(1, 2)
        except NoSuchElementException:
            pass

    def _click_continue(self) -> bool:
        for text in ["Continue", "Next", "continue"]:
            try:
                buttons = self.driver.find_elements(
                    By.XPATH,
                    f'//button[contains(text(), "{text}") or contains(@aria-label, "{text}")]',
                )
                for btn in buttons:
                    if btn.is_displayed():
                        safe_click(self.driver, btn)
                        random_delay(0.5, 1)
                        return True
            except Exception:
                pass
        return False

    def _click_submit(self) -> bool:
        for text in ["Submit", "Apply", "Submit your application"]:
            try:
                buttons = self.driver.find_elements(
                    By.XPATH, f'//button[contains(text(), "{text}")]'
                )
                for btn in buttons:
                    if btn.is_displayed():
                        safe_click(self.driver, btn)
                        return True
            except Exception:
                pass
        return False

    def _is_application_complete(self) -> bool:
        markers = [
            "application has been submitted",
            "you've applied",
            "application sent",
            "successfully applied",
            "thank you for applying",
        ]
        page_text = self.driver.page_source.lower()
        return any(marker in page_text for marker in markers)
