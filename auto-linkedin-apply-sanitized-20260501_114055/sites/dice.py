"""Dice.com job applier - tech-focused job board with Easy Apply."""

import logging
import re
import time
from urllib.parse import quote_plus

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
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
)

logger = logging.getLogger(__name__)


class DiceApplier(SiteApplier):
    SITE_NAME = "dice"
    BASE_URL = "https://www.dice.com"

    def login(self) -> None:
        # Navigate to a job page to check real auth state.
        # Dashboard can load at /dashboard/profiles even with expired sessions.
        self.driver.get(f"{self.BASE_URL}/jobs?q=test&pageSize=1")
        random_delay(3, 4)
        self._dismiss_cookies()

        if not self._is_login_required():
            logger.info("Already logged in to Dice")
            return

        # Not logged in - go to login page
        self.driver.get(f"{self.BASE_URL}/dashboard/login")
        random_delay(2, 3)

        print("\n>>> Please log in to Dice in the browser window.")
        print(">>> Waiting up to 3 minutes for you to log in...")
        self.driver.execute_script(
            """
            var banner = document.createElement('div');
            banner.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#ff6600;color:white;padding:20px;font-size:24px;z-index:999999;text-align:center;';
            banner.textContent = 'Please log in to Dice. The bot will detect login automatically.';
            document.body.prepend(banner);
        """
        )

        for i in range(90):  # 3 minutes
            time.sleep(2)
            try:
                if "login" not in self.driver.current_url.lower():
                    # Verify real auth on a page
                    self.driver.get(f"{self.BASE_URL}/jobs?q=test&pageSize=1")
                    random_delay(2, 3)
                    if not self._is_login_required():
                        logger.info("Dice login confirmed")
                        return
            except Exception:
                pass
        logger.warning(
            "Dice login wait expired after 3 minutes - CANNOT APPLY without login"
        )

    def _is_login_required(self) -> bool:
        """Check if Login/Register button is visible - the real auth indicator on Dice."""
        for btn in self.driver.find_elements(By.TAG_NAME, "button"):
            try:
                text = btn.text.strip()
                if btn.is_displayed() and text in (
                    "Login/Register",
                    "Login",
                    "Sign In",
                ):
                    logger.info("Login required - '%s' button visible", text)
                    return True
            except StaleElementReferenceException:
                continue
        return False

    def _dismiss_cookies(self) -> None:
        """Dismiss cookie consent banner if present."""
        for sel in [
            '//button[contains(text(), "Accept all") or contains(text(), "Accept All")]',
            '//button[contains(text(), "Reject all") or contains(text(), "Reject All")]',
            "#cmpwrapper button",
        ]:
            try:
                if sel.startswith("//"):
                    btn = self.driver.find_element(By.XPATH, sel)
                else:
                    btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                if btn.is_displayed():
                    btn.click()
                    random_delay(0.5, 1)
                    return
            except (NoSuchElementException, Exception):
                continue

    def search_jobs(self, search_term: str) -> list[JobListing]:
        jobs: list[JobListing] = []
        location = self.search_config.location

        # Dice search URL
        params = {
            "q": search_term,
            "location": location,
            "radius": str(self.search_config.radius_miles),
            "postedDate": str(self.search_config.date_posted_days),
            "pageSize": "20",
        }

        query_str = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())

        for page in range(3):
            url = f"{self.BASE_URL}/jobs?{query_str}&page={page + 1}"
            self.driver.get(url)
            random_delay(2, 3)

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
        seen_ids: set[str] = set()
        random_delay(1, 2)

        # Dice now uses plain <a> tags with /job-detail/ hrefs as job title links
        title_links = self.driver.find_elements(
            By.CSS_SELECTOR, 'a[href*="/job-detail/"]'
        )

        for link in title_links:
            try:
                href = link.get_attribute("href") or ""
                title = link.text.strip()

                # Skip non-title links (Apply buttons, short text)
                if not title or title.lower() in ("apply now", "apply", "easy apply"):
                    continue
                if len(title) < 5:
                    continue

                match = re.search(r"/job-detail/([a-f0-9-]+)", href)
                if not match:
                    continue
                job_id = match.group(1)

                if job_id in seen_ids:
                    continue
                seen_ids.add(job_id)

                # Try to get company/location from nearby sibling elements
                company = ""
                location = ""
                # Walk up to the parent card container
                try:
                    parent = link.find_element(
                        By.XPATH,
                        "./ancestor::div[contains(@class, 'flex')][position()=1]/..",
                    )
                    # Company is usually an <a> to /company-profile/
                    try:
                        company_el = parent.find_element(
                            By.CSS_SELECTOR, 'a[href*="/company-profile/"]'
                        )
                        company = company_el.text.strip()
                    except NoSuchElementException:
                        pass
                    # Location from text that contains a state/city pattern
                    card_text = parent.text
                    loc_match = re.search(
                        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z][a-z]+)", card_text
                    )
                    if loc_match:
                        location = loc_match.group(1)
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
            except (NoSuchElementException, StaleElementReferenceException):
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
                        '[data-testid="jobDescriptionHtml"], .job-description, #jobDescription, [class*="description"]',
                    )
                )
            )
            job.description = desc_el.text
        except (TimeoutException, NoSuchElementException):
            # Fallback: grab main content text
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
        self._dismiss_cookies()
        # Store job info for Gemini field answers
        self._current_job = job

        try:
            apply_btn = self._find_apply_button()
            if not apply_btn:
                logger.info("No apply button for '%s'", job.title)
                return False

            # Use JS click to avoid stale element issues after page mutations
            try:
                self.driver.execute_script("arguments[0].click();", apply_btn)
            except StaleElementReferenceException:
                # Re-find the button
                apply_btn = self._find_apply_button()
                if apply_btn:
                    self.driver.execute_script("arguments[0].click();", apply_btn)
                else:
                    return False
            random_delay(2, 3)

            # Log state after clicking Apply
            logger.info(
                "Post-apply: URL=%s windows=%d",
                self.driver.current_url[:120],
                len(self.driver.window_handles),
            )

            # Check if a new window/tab opened (external application)
            if len(self.driver.window_handles) > 2:
                # Switch to the newest tab
                self.driver.switch_to.window(self.driver.window_handles[-1])
                logger.info("New tab opened: %s", self.driver.current_url[:120])

            # Check if we got redirected to login
            current_url = self.driver.current_url.lower()
            if "login" in current_url:
                logger.warning("Redirected to login - not logged in to Dice")
                return False

            # Check for "login required" modal
            try:
                modals = self.driver.find_elements(
                    By.CSS_SELECTOR, "[role='dialog'], section"
                )
                for modal in modals:
                    if (
                        modal.is_displayed()
                        and "log in" in modal.text.lower()
                        and "create an account" in modal.text.lower()
                    ):
                        logger.warning("Login required modal appeared - not logged in")
                        # Try clicking "Log in" to redirect
                        for btn in modal.find_elements(By.TAG_NAME, "button"):
                            if btn.text.strip().lower() == "log in":
                                btn.click()
                                random_delay(2, 3)
                                logger.info(
                                    "Clicked 'Log in' in modal, redirected to: %s",
                                    self.driver.current_url[:80],
                                )
                                break
                        else:
                            # Close modal
                            for btn in modal.find_elements(By.TAG_NAME, "button"):
                                if btn.text.strip().lower() in ("cancel", "close"):
                                    btn.click()
                                    break
                        return False
            except Exception:
                pass

            return self._complete_application()
        finally:
            if len(self.driver.window_handles) > 1:
                self.driver.close()
                self.driver.switch_to.window(original_window)

    def _find_apply_button(self):
        # Scan both buttons AND links for "Apply" text
        for _ in range(3):
            for tag in ["button", "a"]:
                elements = self.driver.find_elements(By.TAG_NAME, tag)
                for el in elements:
                    try:
                        text = el.text.strip().lower()
                        if el.is_displayed() and text in (
                            "apply",
                            "apply now",
                            "easy apply",
                        ):
                            logger.info(
                                "Found apply element: <%s> '%s'", tag, el.text.strip()
                            )
                            return el
                    except StaleElementReferenceException:
                        continue
            time.sleep(2)
        return None

    def _complete_application(self) -> bool:
        """Handle Dice's apply flow - may be modal, external redirect, or Easy Apply."""
        random_delay(1, 2)

        # Check if we got redirected externally
        current_url = self.driver.current_url.lower()
        if "dice.com" not in current_url:
            logger.info("Redirected to external site: %s", current_url[:80])
            # Try to fill external application form
            return self._handle_external_application()

        # Dice Easy Apply or modal flow
        max_steps = 8
        for step in range(max_steps):
            random_delay(0.5, 1)
            self._fill_form_fields()
            self._handle_resume_upload()

            if self._is_complete():
                return True

            if not self._click_next_or_submit():
                break

        return self._is_complete()

    def _handle_external_application(self) -> bool:
        """Try to fill out an external application form."""
        random_delay(2, 3)
        max_steps = 8
        for step in range(max_steps):
            self._fill_form_fields()
            self._handle_resume_upload()

            if self._is_complete():
                return True

            if not self._click_next_or_submit():
                break
            random_delay(1, 2)

        return self._is_complete()

    def _fill_form_fields(self) -> None:
        inputs = self.driver.find_elements(
            By.CSS_SELECTOR,
            "input[type='text'], input[type='tel'], input[type='email'], textarea",
        )
        for inp in inputs:
            try:
                if not inp.is_displayed():
                    continue
                current_val = inp.get_attribute("value") or ""
                if current_val.strip():
                    continue

                label = self._get_label(inp).lower()
                if not label:
                    continue

                if "first" in label and "name" in label:
                    fill_field(self.driver, inp, self.profile.first_name)
                elif "last" in label and "name" in label:
                    fill_field(self.driver, inp, self.profile.last_name)
                elif "name" in label:
                    fill_field(self.driver, inp, self.profile.full_name)
                elif "email" in label:
                    fill_field(self.driver, inp, self.profile.email)
                elif "phone" in label or "mobile" in label:
                    fill_field(self.driver, inp, self.profile.phone)
                elif "zip" in label or "postal" in label:
                    fill_field(self.driver, inp, self.profile.zipcode)
                elif "city" in label or "location" in label:
                    fill_field(self.driver, inp, self.profile.city)
                elif "linkedin" in label:
                    fill_field(self.driver, inp, self.profile.linkedin_url)
                elif "website" in label or "portfolio" in label or "github" in label:
                    fill_field(self.driver, inp, self.profile.website)
                elif "experience" in label or "years" in label:
                    answer = deterministic_answer_for_question(
                        label,
                        "text",
                        generic_years_of_experience=str(self.profile.years_experience),
                    )
                    if answer:
                        fill_field(self.driver, inp, answer[0])
                elif "salary" in label or "compensation" in label:
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
                else:
                    logger.info(
                        "Leaving unknown Dice field for manual handling: %s", label
                    )
            except StaleElementReferenceException:
                continue

    def _get_label(self, element) -> str:
        el_id = element.get_attribute("id") or ""
        if el_id:
            try:
                return self.driver.find_element(
                    By.CSS_SELECTOR, f'label[for="{el_id}"]'
                ).text
            except NoSuchElementException:
                pass
        return (
            element.get_attribute("aria-label")
            or element.get_attribute("placeholder")
            or ""
        )

    def _handle_resume_upload(self) -> None:
        try:
            file_input = self.driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            file_input.send_keys(self.profile.resume_path)
            random_delay(1, 2)
        except NoSuchElementException:
            pass

    def _click_next_or_submit(self) -> bool:
        targets = ["submit", "apply", "next", "continue"]
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            try:
                if not btn.is_displayed() or not btn.is_enabled():
                    continue
                text = btn.text.strip().lower()
                if any(t in text for t in targets):
                    safe_click(self.driver, btn)
                    random_delay(0.5, 1)
                    return True
            except (StaleElementReferenceException, Exception):
                continue
        return False

    def _is_complete(self) -> bool:
        markers = [
            "application submitted",
            "you have applied",
            "successfully applied",
            "thank you for applying",
            "application complete",
            "application has been submitted",
            "your application has been sent",
            "we received your application",
            "thanks for applying",
        ]
        try:
            page_text = self.driver.page_source.lower()
            return any(m in page_text for m in markers)
        except Exception:
            return False
