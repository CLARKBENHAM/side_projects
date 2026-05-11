"""
Author:     Sai Vignesh Golla
LinkedIn:   https://www.linkedin.com/in/saivigneshgolla/

Copyright (C) 2024 Sai Vignesh Golla

License:    GNU Affero General Public License
            https://www.gnu.org/licenses/agpl-3.0.en.html

GitHub:     https://github.com/GodsScion/Auto_job_applier_linkedIn

version:    24.12.29.12.30
"""

# Imports

import arm64_patch  # noqa: F401 - must be before pyautogui
import os
import json
import re

from time import sleep
from random import randint
from datetime import datetime, timedelta
from pyautogui import alert
from pprint import pprint

from config.settings import logs_folder_path

_RELATIVE_TIME_PATTERN = re.compile(
    r"(?P<value>\d+)\s+(?P<unit>second|minute|hour|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)


#### Common functions ####


# < Directories related
def make_directories(paths: list[str]) -> None:
    """
    Function to create missing directories
    """
    for path in paths:
        path = path.replace("//", "/")
        if "/" in path and "." in path:
            path = path[: path.rfind("/")]
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print(f'Error while creating directory "{path}": ', e)


def find_default_profile_directory() -> str | None:
    """
    Function to search for Chrome Profiles within default locations
    """
    import sys

    if sys.platform == "darwin":
        default_locations = [
            os.path.expanduser("~/Library/Application Support/Google/Chrome Beta"),
            os.path.expanduser("~/Library/Application Support/Google/Chrome"),
        ]
    else:
        default_locations = [
            r"%LOCALAPPDATA%\Google\Chrome\User Data",
            r"%USERPROFILE%\AppData\Local\Google\Chrome\User Data",
            r"%USERPROFILE%\Local Settings\Application Data\Google\Chrome\User Data",
        ]
    for location in default_locations:
        profile_dir = os.path.expandvars(location)
        if os.path.exists(profile_dir):
            return profile_dir
    return None


# >


# < Logging related
def critical_error_log(possible_reason: str, stack_trace: Exception) -> None:
    """
    Function to log and print critical errors along with datetime stamp
    """
    print_lg(possible_reason, stack_trace, datetime.now(), from_critical=True)


def get_log_path():
    """
    Function to replace '//' with '/' for logs path
    """
    try:
        path = os.environ.get("AUTO_APPLIER_LOG_PATH") or logs_folder_path + "/log.txt"
        return path.replace("//", "/")
    except Exception as e:
        critical_error_log(
            "Failed getting log path! So assigning default logs path: './logs/log.txt'",
            e,
        )
        return "logs/log.txt"


__logs_file_path = get_log_path()


def print_lg(
    *msgs: str | dict,
    end: str = "\n",
    pretty: bool = False,
    flush: bool = False,
    from_critical: bool = False,
) -> None:
    """
    Function to log and print. **Note that, `end` and `flush` parameters are ignored if `pretty = True`**
    """
    try:
        for message in msgs:
            pprint(message) if pretty else print(message, end=end, flush=flush)
            with open(__logs_file_path, "a+", encoding="utf-8") as file:
                file.write(str(message) + end)
    except Exception as e:
        trail = (
            f'Skipped saving this message: "{message}" to log.txt!'
            if from_critical
            else "We'll try one more time to log..."
        )
        alert(
            f"log.txt in {logs_folder_path} is open or is occupied by another program! Please close it! {trail}",
            "Failed Logging",
        )
        if not from_critical:
            critical_error_log("Log.txt is open or is occupied by another program!", e)


# >


def buffer(speed: int = 0) -> None:
    """
    Function to wait within a period of selected random range.
    * Will not wait if input `speed <= 0`
    * Will wait within a random range of
      - `0.6 to 1.0 secs` if `1 <= speed < 2`
      - `1.0 to 1.8 secs` if `2 <= speed < 3`
      - `1.8 to speed secs` if `3 <= speed`
    """
    if speed <= 0:
        return
    elif speed <= 1 and speed < 2:
        return sleep(randint(6, 10) * 0.1)
    elif speed <= 2 and speed < 3:
        return sleep(randint(10, 18) * 0.1)
    else:
        return sleep(randint(18, round(speed) * 10) * 0.1)


def manual_login_retry(is_logged_in: callable, limit: int = 2) -> bool:
    """
    Function to ask and validate manual login
    """
    count = 0
    while not is_logged_in():
        print_lg("Seems like you're not logged in!")
        print_lg(
            "Please log in manually in the browser window, then press Enter here..."
        )
        try:
            input(">>> Press Enter after logging in to LinkedIn... ")
        except EOFError:
            print_lg("Non-interactive mode, waiting 60s for manual login...")
            sleep(60)
        count += 1
        if count > limit:
            print_lg("Login confirmation limit reached.")
            return is_logged_in()
    return True


def calculate_date_posted(time_string: str) -> datetime | None | ValueError:
    """
    Function to calculate date posted from string.
    Returns datetime object | None if unable to calculate | ValueError if time_string is invalid
    Valid time string examples:
    * 10 seconds ago
    * 15 minutes ago
    * 2 hours ago
    * 1 hour ago
    * 1 day ago
    * 10 days ago
    * 1 week ago
    * 1 month ago
    * 1 year ago
    """
    time_string = time_string.strip()
    now = datetime.now()
    lowered = time_string.lower()
    if "just now" in lowered or "today" in lowered:
        return now
    if "yesterday" in lowered:
        return now - timedelta(days=1)

    match = _RELATIVE_TIME_PATTERN.search(lowered)
    if not match:
        return None

    value = int(match.group("value"))
    unit = match.group("unit")
    if unit == "second":
        date_posted = now - timedelta(seconds=value)
    elif unit == "minute":
        date_posted = now - timedelta(minutes=value)
    elif unit == "hour":
        date_posted = now - timedelta(hours=value)
    elif unit == "day":
        date_posted = now - timedelta(days=value)
    elif unit == "week":
        date_posted = now - timedelta(weeks=value)
    elif unit == "month":
        date_posted = now - timedelta(days=value * 30)
    else:
        date_posted = now - timedelta(days=value * 365)
    return date_posted


def convert_to_lakhs(value: str) -> str:
    """
    Converts str value to lakhs, no validations are done except for length and stripping.
    Examples:
    * "100000" -> "1.00"
    * "101,000" -> "10.1," Notice ',' is not removed
    * "50" -> "0.00"
    * "5000" -> "0.05"
    """
    value = value.strip()
    value_len = len(value)
    if value_len > 0:
        if value_len > 5:
            value = value[: value_len - 5] + "." + value[value_len - 5 : value_len - 3]
        else:
            value = "0." + "0" * (5 - value_len) + value[:2]
    return value


def convert_to_json(data) -> dict:
    """
    Function to convert data to JSON, if unsuccessful, returns `{"error": "Unable to parse the response as JSON", "data": data}`
    """
    try:
        result_json = json.loads(data)
        return result_json
    except json.JSONDecodeError:
        return {"error": "Unable to parse the response as JSON", "data": data}
