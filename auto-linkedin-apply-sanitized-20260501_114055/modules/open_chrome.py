# Claude Version
# try to connect to existing open chrome

"""
Author:     Sai Vignesh Golla
LinkedIn:   https://www.linkedin.com/in/saivigneshgolla/

Copyright (C) 2024 Sai Vignesh Golla

License:    GNU Affero General Public License
            https://www.gnu.org/licenses/agpl-3.0.en.html

GitHub:     https://github.com/GodsScion/Auto_job_applier_linkedIn

version:    24.12.29.12.30
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from modules.helpers import make_directories
from config.settings import (
    run_in_background,
    stealth_mode,
    disable_extensions,
    safe_mode,
    file_name,
    failed_file_name,
    logs_folder_path,
    generated_resume_path,
)
from config.questions import default_resume_path

if stealth_mode:
    import undetected_chromedriver as uc
else:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    # from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from modules.helpers import find_default_profile_directory, critical_error_log, print_lg

CHROME_BETA_APP_NAME = "Google Chrome Beta"
CHROME_BETA_BINARY = (
    "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta"
)
DEBUGGER_READY_TIMEOUT_SECONDS = 20
DEBUGGER_POLL_INTERVAL_SECONDS = 0.25
CHROME_DRIVER_START_TIMEOUT_SECONDS = int(
    os.environ.get("AUTO_APPLIER_CHROME_DRIVER_START_TIMEOUT_SECONDS", "90")
)


def _browser_init_disabled() -> bool:
    return os.environ.get("AUTO_APPLIER_SKIP_BROWSER_INIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _find_free_debug_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_debugger_endpoint(
    port: int, timeout: float = DEBUGGER_READY_TIMEOUT_SECONDS
) -> None:
    deadline = time.time() + timeout
    endpoint = f"http://127.0.0.1:{port}/json/version"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(endpoint, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(DEBUGGER_POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Chrome debugger endpoint on port {port} did not become ready within {timeout}s."
    )


def _build_macos_headless_launch_command(chrome_args: list[str]) -> list[str]:
    return ["open", "-n", "-a", CHROME_BETA_APP_NAME, "--args", *chrome_args]


def _run_with_startup_timeout(label: str, start_browser):
    timeout = CHROME_DRIVER_START_TIMEOUT_SECONDS
    if timeout < 1 or sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        return start_browser()

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"{label} did not finish within {timeout}s.")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return start_browser()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _find_browser_pids_for_port(port: int) -> list[int]:
    result = subprocess.run(
        ["lsof", "-nP", "-tiTCP:" + str(port), "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _stop_browser_listening_on_port(port: int) -> None:
    pids = _find_browser_pids_for_port(port)
    if not pids:
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + 5
    while time.time() < deadline:
        remaining = _find_browser_pids_for_port(port)
        if not remaining:
            return
        time.sleep(0.2)

    for pid in _find_browser_pids_for_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _cleanup_profile_dir(profile_dir: str | None) -> None:
    if not profile_dir:
        return
    try:
        shutil.rmtree(profile_dir, ignore_errors=True)
    except Exception as e:
        critical_error_log(
            f"Failed to remove temporary Chrome profile {profile_dir}", e
        )


def _wrap_driver_quit(
    driver,
    debug_port: int | None = None,
    cleanup_profile_dir: str | None = None,
):
    original_quit = driver.quit

    def _quit_with_cleanup():
        original_error = None
        try:
            original_quit()
        except Exception as exc:
            original_error = exc
        finally:
            if debug_port is not None:
                _stop_browser_listening_on_port(debug_port)
            _cleanup_profile_dir(cleanup_profile_dir)
        if original_error is not None:
            raise original_error

    driver.quit = _quit_with_cleanup
    return driver


def _configure_profile_options(options) -> str | None:
    temp_profile_dir = None
    if safe_mode:
        print_lg(
            "SAFE MODE: Will login with a temporary profile, browsing history will not be saved"
            " in the browser!"
        )
        if run_in_background and sys.platform == "darwin" and not stealth_mode:
            temp_profile_dir = tempfile.mkdtemp(prefix="auto-applier-chrome-")
            options.add_argument(f"--user-data-dir={temp_profile_dir}")
        return temp_profile_dir

    profile_dir = os.environ.get("AUTO_APPLIER_CHROME_USER_DATA_DIR", "").strip()
    if profile_dir:
        options.add_argument(f"--user-data-dir={profile_dir}")
        print_lg(f"Using configured Chrome profile directory: {profile_dir}")
        return temp_profile_dir

    profile_dir = find_default_profile_directory()
    if profile_dir:
        options.add_argument(f"--user-data-dir={profile_dir}")
        return temp_profile_dir

    print_lg(
        "Default profile directory not found. Logging in with a temporary profile, Web history"
        " will not be saved!"
    )
    if run_in_background and sys.platform == "darwin" and not stealth_mode:
        temp_profile_dir = tempfile.mkdtemp(prefix="auto-applier-chrome-")
        options.add_argument(f"--user-data-dir={temp_profile_dir}")
    return temp_profile_dir


def _launch_macos_background_driver(base_options, cleanup_profile_dir: str | None):
    debug_port = _find_free_debug_port()
    chrome_args = list(base_options.arguments)
    chrome_args.append(f"--remote-debugging-port={debug_port}")
    launch_cmd = _build_macos_headless_launch_command(chrome_args)
    print_lg(
        "Launching Chrome Beta headlessly via LaunchServices on debugger port "
        f"{debug_port}."
    )
    subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_for_debugger_endpoint(debug_port)
        attach_options = Options()
        attach_options.binary_location = CHROME_BETA_BINARY
        attach_options.add_experimental_option(
            "debuggerAddress", f"127.0.0.1:{debug_port}"
        )
        driver = _run_with_startup_timeout(
            "Attaching ChromeDriver to Chrome Beta",
            lambda: webdriver.Chrome(options=attach_options),
        )
    except Exception:
        _stop_browser_listening_on_port(debug_port)
        _cleanup_profile_dir(cleanup_profile_dir)
        raise
    return _wrap_driver_quit(
        driver,
        debug_port=debug_port,
        cleanup_profile_dir=cleanup_profile_dir,
    )


def _create_driver():
    options = uc.ChromeOptions() if stealth_mode else Options()
    if run_in_background:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1440,900")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    if disable_extensions:
        options.add_argument("--disable-extensions")

    print_lg(
        "IF YOU HAVE MORE THAN 10 TABS OPENED, PLEASE CLOSE OR BOOKMARK THEM! Or it's highly"
        " likely that application will just open browser and not do anything!"
    )

    cleanup_profile_dir = _configure_profile_options(options)

    if stealth_mode:
        print_lg(
            "Downloading Chrome Driver... This may take some time. Undetected mode requires"
            " download every run!"
        )
        return _run_with_startup_timeout(
            "Starting undetected ChromeDriver",
            lambda: uc.Chrome(
                options=options, browser_executable_path=CHROME_BETA_BINARY
            ),
        )

    options.binary_location = CHROME_BETA_BINARY
    if run_in_background and sys.platform == "darwin":
        return _launch_macos_background_driver(options, cleanup_profile_dir)

    driver = _run_with_startup_timeout(
        "Starting ChromeDriver", lambda: webdriver.Chrome(options=options)
    )
    return _wrap_driver_quit(driver, cleanup_profile_dir=cleanup_profile_dir)


driver = None
wait = None
actions = None

if not _browser_init_disabled():
    try:
        make_directories(
            [
                file_name,
                failed_file_name,
                logs_folder_path + "/screenshots",
                default_resume_path,
                generated_resume_path + "/temp",
            ]
        )

        driver = _create_driver()
        if not run_in_background:
            driver.maximize_window()
        wait = WebDriverWait(driver, 5)
        actions = ActionChains(driver)
    except Exception as e:
        print(e)
        msg = (
            "Seems like either... \n\n1. Chrome is already running. \nA. Close all Chrome windows"
            " and try again. \n\n2. Google Chrome or Chromedriver is out dated. \nA. Update"
            ' browser and Chromedriver (You can run "windows-setup.bat" in /setup folder for'
            " Windows PC to update Chromedriver)! \n\n3. If error occurred when using"
            ' "stealth_mode", try reinstalling undetected-chromedriver. \nA. Open a terminal and'
            ' use commands "pip uninstall undetected-chromedriver" and "pip install'
            ' undetected-chromedriver". \n\n\nIf issue persists, try Safe Mode. Set, safe_mode ='
            " True in config.py \n\nPlease check GitHub discussions/support for solutions"
            " https://github.com/GodsScion/Auto_job_applier_linkedIn \n                               "
            "    OR \nReach out in discord ( https://discord.gg/fFp7uUzWCY )"
        )
        if isinstance(e, TimeoutError):
            msg = (
                "Couldn't launch Chrome within the startup timeout. Check whether the"
                " bot Chrome profile is locked or Chrome Beta is already running, then retry."
            )
        print_lg(msg)
        critical_error_log("In Opening Chrome", e)
        if not run_in_background:
            from pyautogui import alert

            alert(msg, "Error in opening chrome")
        try:
            driver.quit()
        except AttributeError:
            pass
        except NameError:
            exit()
