import numpy as np
import pandas as pd
# import beautifulsoup4
import lxml.html
import requests
# import requests_cache
import re

from datetime import datetime
import time
import random

from collections import namedtuple
import pickle

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException, TimeoutException, StaleElementReferenceException
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import collections
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import pyautogui as pygu
import sys
import time
import re
import random
#%%
base_url = "https://openpayrolls.com/university-college/university-of-virginia/page-1"
# chromedriver_path = 'C:\Users\student.DESKTOP-UT02KBN\Desktop\chromedriver_win32\chromedriver'
chromedriver_path = 'C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\chromedriver_win32\\chromedriver'
driver =  webdriver.Chrome(executable_path = chromedriver_path)
wait = WebDriverWait(driver, 10)
driver.get(base_url)

#%%
results = []
names = []
employee_table = driver.find_element_by_xpath("//*[@id='employeesTable']/tbody")
for row in employee_table.find_elements_by_xpath("tr"):
    #%%
row = employee_table.find_elements_by_xpath("tr")[3]
results += [[i.text for i in row.find_elements_by_xpath("td")]]
names += [results[-1][2]]
#have to merge results on the name without a middle inital
#what if share a first+last name? check but hope it doesn't happen
#%%
name = names[-1]
name_search = f"https://openpayrolls.com/search/{name.replace(" ", "-")}?sort=source-desc"
driver.get(name_search)
employee_records =  driver.find_elements_by_xpath("//*tr[@itemprop='employee']")


names_loc_col = driver.find_elements_by_xpath("//*[@itemprop='worksFor']")
names_job_title = driver.find_elements_by_xpath("//*[@itemprop='jobTitle']")
names_year = driver.find_elements_by_xpath("//*tr[@itemprop='employee']/td[0]")


