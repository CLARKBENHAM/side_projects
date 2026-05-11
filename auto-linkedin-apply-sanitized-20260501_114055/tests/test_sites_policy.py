import os
import sys
import types

os.environ["AUTO_APPLIER_DISABLE_AI_CLI"] = "true"

sys.modules.setdefault(
    "pyautogui", types.SimpleNamespace(alert=lambda *args, **kwargs: None)
)

if "selenium.common.exceptions" not in sys.modules:
    selenium_module = sys.modules.get("selenium") or types.ModuleType("selenium")
    webdriver_module = sys.modules.get("selenium.webdriver") or types.ModuleType(
        "selenium.webdriver"
    )
    common_module = sys.modules.get("selenium.common") or types.ModuleType(
        "selenium.common"
    )
    exceptions_module = types.ModuleType("selenium.common.exceptions")
    chrome_module = types.ModuleType("selenium.webdriver.chrome")
    options_module = types.ModuleType("selenium.webdriver.chrome.options")
    remote_module = types.ModuleType("selenium.webdriver.remote")
    remote_webdriver_module = types.ModuleType("selenium.webdriver.remote.webdriver")
    remote_webelement_module = types.ModuleType("selenium.webdriver.remote.webelement")
    support_module = types.ModuleType("selenium.webdriver.support")
    expected_conditions_module = types.ModuleType(
        "selenium.webdriver.support.expected_conditions"
    )
    support_ui_module = types.ModuleType("selenium.webdriver.support.ui")
    for package_module in (
        selenium_module,
        webdriver_module,
        common_module,
        chrome_module,
        remote_module,
        support_module,
    ):
        package_module.__path__ = []

    class SeleniumException(Exception):
        pass

    class Options:
        def __init__(self):
            self.binary_location = ""

        def add_argument(self, *_args, **_kwargs):
            return None

        def add_experimental_option(self, *_args, **_kwargs):
            return None

    class WebDriverWait:
        def __init__(self, driver, timeout):
            self.driver = driver
            self.timeout = timeout

        def until(self, condition):
            return condition(self.driver)

    for name in (
        "ElementClickInterceptedException",
        "ElementNotInteractableException",
        "StaleElementReferenceException",
        "TimeoutException",
    ):
        setattr(exceptions_module, name, SeleniumException)

    options_module.Options = Options
    remote_webdriver_module.WebDriver = object
    remote_webelement_module.WebElement = object
    support_ui_module.WebDriverWait = WebDriverWait
    webdriver_module.Chrome = lambda *args, **kwargs: None
    selenium_module.webdriver = webdriver_module
    selenium_module.common = common_module
    webdriver_module.chrome = chrome_module
    webdriver_module.remote = remote_module
    webdriver_module.support = support_module
    common_module.exceptions = exceptions_module
    chrome_module.options = options_module
    remote_module.webdriver = remote_webdriver_module
    remote_module.webelement = remote_webelement_module
    support_module.expected_conditions = expected_conditions_module
    support_module.ui = support_ui_module

    sys.modules["selenium"] = selenium_module
    sys.modules["selenium.webdriver"] = webdriver_module
    sys.modules["selenium.common"] = common_module
    sys.modules["selenium.common.exceptions"] = exceptions_module
    sys.modules["selenium.webdriver.chrome"] = chrome_module
    sys.modules["selenium.webdriver.chrome.options"] = options_module
    sys.modules["selenium.webdriver.remote"] = remote_module
    sys.modules["selenium.webdriver.remote.webdriver"] = remote_webdriver_module
    sys.modules["selenium.webdriver.remote.webelement"] = remote_webelement_module
    sys.modules["selenium.webdriver.support"] = support_module
    sys.modules["selenium.webdriver.support.expected_conditions"] = (
        expected_conditions_module
    )
    sys.modules["selenium.webdriver.support.ui"] = support_ui_module

from sites.base import JobListing, SiteApplier  # noqa: E402
from sites.classifier import answer_field  # noqa: E402
from sites.config import ApplicantProfile, SearchConfig  # noqa: E402


class FakeSiteApplier(SiteApplier):
    SITE_NAME = "fake"

    def __init__(self):
        self.applied_jobs = []
        super().__init__(
            profile=ApplicantProfile(),
            search_config=SearchConfig(search_terms=[]),
            driver=types.SimpleNamespace(quit=lambda: None),
        )

    def login(self) -> None:
        return None

    def search_jobs(self, search_term: str) -> list[JobListing]:
        return []

    def apply_to_job(self, job: JobListing) -> bool:
        self.applied_jobs.append(job.job_id)
        return True


def test_multisite_skips_job_without_description():
    applier = FakeSiteApplier()

    applier._process_job(
        JobListing(
            job_id="1",
            title="Robotics Software Engineer",
            company="Direct Robotics Co",
            location="Los Angeles, CA",
        )
    )

    assert applier.applied_jobs == []
    assert applier.results.skipped == 1
    assert applier.results.applied == 0


def test_multisite_skips_when_policy_classifier_fails(monkeypatch):
    applier = FakeSiteApplier()

    def fail_classifier(*args, **kwargs):
        raise RuntimeError("classifier unavailable")

    monkeypatch.setattr("sites.base.classify_job", fail_classifier)

    applier._process_job(
        JobListing(
            job_id="2",
            title="Software Engineer",
            company="Example Co",
            location="Los Angeles, CA",
            description="Build B2B SaaS workflow software.",
        )
    )

    assert applier.applied_jobs == []
    assert applier.results.skipped == 1
    assert applier.results.applied == 0


def test_multisite_answer_field_uses_safety_rules_before_ai(monkeypatch):
    def fail_get_model():
        raise AssertionError("AI fallback should not be used for deterministic answers")

    monkeypatch.setattr("sites.classifier._get_model", fail_get_model)

    assert (
        answer_field("Expected annual compensation", "Principal AI Engineer")
        == "220000"
    )
    assert answer_field("Are you willing to relocate to San Francisco?") == "No"
    assert answer_field("Do you have a professional certification?") == "No"
    assert (
        answer_field("How many years of experience do you have with Azure OpenAI?")
        == ""
    )
