import importlib
import os
import sys
import types


os.environ["AUTO_APPLIER_SKIP_BROWSER_INIT"] = "true"
sys.modules.setdefault(
    "pyautogui", types.SimpleNamespace(alert=lambda *args, **kwargs: None)
)


class _FakeOptions:
    def __init__(self):
        self.arguments = []
        self.binary_location = None
        self.experimental_options = {}

    def add_argument(self, value):
        self.arguments.append(value)

    def add_experimental_option(self, name, value):
        self.experimental_options[name] = value


selenium_module = types.ModuleType("selenium")
webdriver_module = types.ModuleType("selenium.webdriver")
webdriver_module.Chrome = lambda *args, **kwargs: None
chrome_module = types.ModuleType("selenium.webdriver.chrome")
chrome_options_module = types.ModuleType("selenium.webdriver.chrome.options")
chrome_options_module.Options = _FakeOptions
common_module = types.ModuleType("selenium.webdriver.common")
action_chains_module = types.ModuleType("selenium.webdriver.common.action_chains")
action_chains_module.ActionChains = lambda driver: ("actions", driver)
support_module = types.ModuleType("selenium.webdriver.support")
support_ui_module = types.ModuleType("selenium.webdriver.support.ui")
support_ui_module.WebDriverWait = lambda driver, timeout: ("wait", driver, timeout)

sys.modules.setdefault("selenium", selenium_module)
sys.modules.setdefault("selenium.webdriver", webdriver_module)
sys.modules.setdefault("selenium.webdriver.chrome", chrome_module)
sys.modules.setdefault("selenium.webdriver.chrome.options", chrome_options_module)
sys.modules.setdefault("selenium.webdriver.common", common_module)
sys.modules.setdefault("selenium.webdriver.common.action_chains", action_chains_module)
sys.modules.setdefault("selenium.webdriver.support", support_module)
sys.modules.setdefault("selenium.webdriver.support.ui", support_ui_module)

open_chrome = importlib.import_module("modules.open_chrome")


def test_build_macos_headless_launch_command():
    command = open_chrome._build_macos_headless_launch_command(
        ["--headless=new", "--remote-debugging-port=9333"]
    )

    assert command[:5] == ["open", "-n", "-a", "Google Chrome Beta", "--args"]
    assert "--headless=new" in command
    assert "--remote-debugging-port=9333" in command


def test_run_with_startup_timeout_can_be_disabled(monkeypatch):
    monkeypatch.setattr(open_chrome, "CHROME_DRIVER_START_TIMEOUT_SECONDS", 0)

    assert (
        open_chrome._run_with_startup_timeout("Starting Chrome", lambda: "driver")
        == "driver"
    )


def test_wrap_driver_quit_stops_browser_and_cleans_profile(monkeypatch):
    calls = []

    class FakeDriver:
        def quit(self):
            calls.append(("quit", None))

    monkeypatch.setattr(
        open_chrome,
        "_stop_browser_listening_on_port",
        lambda port: calls.append(("stop", port)),
    )
    monkeypatch.setattr(
        open_chrome,
        "_cleanup_profile_dir",
        lambda path: calls.append(("cleanup", path)),
    )

    driver = FakeDriver()
    wrapped = open_chrome._wrap_driver_quit(
        driver, debug_port=9333, cleanup_profile_dir="/tmp/fake-profile"
    )

    wrapped.quit()

    assert calls == [
        ("quit", None),
        ("stop", 9333),
        ("cleanup", "/tmp/fake-profile"),
    ]
