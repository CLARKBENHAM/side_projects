# import tkinter

# print("Tkinter is installed and available.")

# import pymsgbox

# pymsgbox.alert("This is an alert box!", "Alert")

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
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
import subprocess
import time
import psutil


import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", 0))
port = sock.getsockname()[1]
sock.close()
print(f"Debugging port: {port}")
