# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:51:25 2019

@author: Clark Benham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import timeit
import math
import re
import requests
import pickle
import os
from datetime import datetime, date, timedelta
import time
from pandas.compat import StringIO

import subprocess
import sys
import importlib.util
def check_install(package):
    spec = importlib.util.find_spec(package)
    if spec == None:
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        print("Downloaded", package)
check_install('tika')
from tika import parser
import os
from urllib.request import urlretrieve

#download selenium
check_install('selenium')
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import collections

#%%
pdf_urls = []
pdf_cookies = []
row_text = []
missing_rows = []
url = "https://rrcsearch3.neubus.com/esd3-rrc/index.php?_module_=esd&_action_=keysearch&profile=29"
driver = webdriver.Chrome(executable_path = 'C:\Program Files\chromedriver_win32\chromedriver')
#driver.implicitly_wait(10)#will implicitly wait; a little slower?
wait = WebDriverWait(driver, 10)
driver.get(url)

#need to iter over every year
for yr in [str(yr) for yr in range(2013, datetime.today().year + 1)]:
    report_yr = Select(driver.find_element_by_xpath('//*[(@id = "report_period_yearDROPDOWN")]'))
    report_yr.select_by_visible_text(yr)
    driver.find_element_by_xpath('//*[(@id = "docSearchButton")]').click()
    
    wait.until(EC.element_to_be_clickable((By.ID,'selectSize' )))
    display_size = Select(driver.find_element_by_xpath('//*[(@id = "selectSize")]'))
    display_size.select_by_value('50')
    
    #read in table as dataframe?
#%
    num_results = wait.until(EC.element_to_be_clickable((By.XPATH, '//span[@id="searchResultsCount"]')))
    #num_results = driver.find_element_by_xpath('//span[@id="searchResultsCount"]')
    num_pages = math.ceil(int(re.findall("\d+", num_results.text)[1])/50)
    for n in range(num_pages):
        #issue is that these elements will always be availible; how can you identify that a new page has come up?
        dropdowns = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')))
        #wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]')))
#        rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]//tbody//tr[60]//td[1]//div[2]//a//img')))
        rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')))
        
        #need to make sure next page has loaded before dropdowns are assigned
#        dropdowns = driver.find_elements_by_xpath('//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')
#        rows = driver.find_elements_by_xpath('//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')
        i = 0 
        while i < 2:# len(rows):
            #wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'showActionMenu')))#always exists, after the FIRST run
            #time.sleep(0.25)#row/drop is already assigned
            
            #each time come back to the page; get a new instance
            row = rows[i]
            drop = dropdowns[i]
            
            #opens dropdown in main page
            try:
                if row.text == "":
                    missing_rows.append(n*50 + i)
                row_text.append(row.text)#shouldn't append each time
                drop.click()
            except:
                #need to let the new page load
                print(f"reloaded page {n+1}, @{i}")
                time.sleep(1)
                dropdowns = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')))
                rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')))
                #undo any extra calls
                row_text = row_text[:n*50]
                missing_rows = [k for k in missing_rows if k< n*50]
                i = 0#restart processing of page
                break
                continue#jumps back to top of while loop
                
            #goes to pdf page
#            at_pdf_page = False
#            while not at_pdf_page:
#                try:
#                    wait.until(EC.element_to_be_clickable((By.XPATH, "//img[@alt='Load Document']")))
#                    driver.find_element_by_xpath("//img[@alt='Load Document']").click()
#                except:
#                    driver.find_element_by_xpath('//*[@id="view"]').click()
#                try:#checks if at pdf page
#                    driver.find_element_by_xpath('//*[@id="pdfcontainer"]')
#                    at_pdf_page = True
#                except:
#                    time.sleep(0.5) 
            try:
                goto_pdf = wait.until(EC.element_to_be_clickable((By.XPATH, "//img[@alt='Load Document']")))
            except:
                drop.click()
                goto_pdf = wait.until(EC.element_to_be_clickable((By.XPATH, "//img[@alt='Load Document']")))
            goto_pdf.click()
            
            #dropdown to open pdf
            pdf_dropdown = wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@id='downloadMenuButton']")))
                
            #commented so don't actually download
            pdf_dropdown.click()#option to download pdf
#            driver.find_element_by_xpath("//a[@id='downloadAllButton']").click()#download pdf
#            driver.switch_to.window(driver.window_handles[1])
#            #in newly opened window
#            time.sleep(0.2)#no attributes on the pdf page
#            pdf_cookies.append(driver.get_cookies())
#            pdf_urls.append(driver.current_url)
#            driver.close()
            
            driver.switch_to.window(driver.window_handles[0])#gets back to original window
            driver.find_element_by_xpath('//a[(@id="closeDoc")]').click()#goes back to search results page
            i += 1
        if n != num_pages - 1:#don't go to next page on last day
            #clicks to next search results page
            try:      
                next_results_page = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@class='page-link next']")))#to next page
                next_results_page.click()
            except:
                time.sleep(0.5)
                driver.find_element_by_xpath('//*[contains(concat( " ", @class, " " ), concat( " ", "next", " " ))]').click()
            #hold till get to next search results page
            while True:
                time.sleep(0.5)
                page_number = driver.find_element_by_xpath('//*[@id="searchPagination"]//ul//li[@class="active"]//span[@class="current"]')
                if int(page_number.text) == n+2:#got to next page
                    break
        print(f"finished page {n+1}")
        time.sleep(0.75)
#%%
row_text = []
for n in range(10):
    dropdowns = driver.find_elements_by_xpath('//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')
    dropdowns2 = driver.find_elements_by_xpath('//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document"]')
    rows = driver.find_elements_by_xpath('//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')
    print(len(dropdowns), len(dropdowns2))
    for i, (row, drop, drop2) in enumerate(zip(rows, dropdowns, dropdowns2)):
    #    print(drop.click())
        if row.text == "":
            print(i)
        row_text.append(row.text)
        try:#opens dropdown in main page
            drop.click()
        except:
            drop2.click()
        try:#closes dropdown; 
            driver.find_element_by_xpath('//a[@class="contextMenuClose close"]').click()
        except:
            try:
                driver.find_element_by_xpath('//*[@id="searchResults"]/tbody/tr//td//div[2]/a/img').click()
            except:
                driver.find_element_by_xpath('//*[@id="imageMenu"]/a[5]').click()
                
   
    print(f"finished page {n}") 
    time.sleep(0.5)

#%%
#driver.find_element_by_xpath('//*[@id="searchPagination"]//ul//li[@class="page-link next"//a')
print(pdf_urls)
#%%
#os.system("mkdir Desktop\side_projects\scrap_pdfs")
import urllib3
pdf_url = 'https://rrcsearch3.neubus.com//rrcdisplay/5aa5461eb53151c12fbc59bf182d06d2_1563257038.pdf'
def get_pdf(pdf_url):
    "Downloads pdfs; by using urllib3 can multi thread"
    http = urllib3.PoolManager()
    r = http.request('GET', pdf_url, preload_content=False)
    path = "Desktop\side_projects\scrap_pdfs\\" + pdf_url.split("/")[-1]
    with open(path, 'wb') as out:
        while True:
            data = r.read()
            if not data:
                break
            out.write(data)
    r.release_conn()
 
get_pdf(pdf_url)
#%%
driver = webdriver.Chrome(executable_path = 'C:\Program Files\chromedriver_win3264\chromedriver')
#driver.implicitly_wait(10)#will implicitly wait; a little slower?
wait = WebDriverWait(driver, 10)
driver.get(url)
#%%
def goto_page(driver, wait, n):
    locs = {'min':0, 'current':0, 'max':0}
    base_x = '//*[@id="searchPagination"]/ul/'
    for key, x in zip(locs.keys(), ['li[2]/a', "li/span[@class='Current']", 'li[position() == (last() -1)]/a']):
        locs[key] = wait.until(EC.element_to_be_clickable((By.XPATH, base_x + x)))
        locs[key] = int(locs[key].text)
    assert(n <= locs['max'])
    assert(n >= locs['min'])
    steps = n - locs['current']
    while steps > 0:
        click_next = wait.until(EC.element_to_be_clickable(('//*[@id="searchPagination"]/ul/li[last()]/a')))
        click_next.click()
        steps -= 1
    while steps < 0:#go 
        click_prev = wait.until(EC.element_to_be_clickable(('//*[@id="searchPagination"]/ul/li[last()]/a')))
        click_prev.click()
        steps += 1
goto_page(driver, wait, 10)


#%%
driver = webdriver.Chrome(executable_path = 'C:\Program Files\chromedriver_win3264\chromedriver')
#driver.implicitly_wait(10)#will implicitly wait; a little slower?
wait = WebDriverWait(driver, 10)
driver.get(url)

pdf_urls =[]
pdf_cookies = []
for yr in [str(yr) for yr in range(2013, datetime.today().year + 1)]:
    report_yr = Select(driver.find_element_by_xpath('//*[(@id = "report_period_yearDROPDOWN")]'))
    report_yr.select_by_visible_text(yr)
    driver.find_element_by_xpath('//*[(@id = "docSearchButton")]').click()
    
    wait.until(EC.element_to_be_clickable((By.ID,'selectSize' )))
    display_size = Select(driver.find_element_by_xpath('//*[(@id = "selectSize")]'))
    display_size.select_by_value('50')
    time.sleep(1)

    #read in table as dataframe?
    num_results = [0,0]
    while num_results[0] != 50:
        time.sleep(1)
        num_results = wait.until(EC.element_to_be_clickable((By.XPATH, '//span[@id="searchResultsCount"]')))
        num_results = [int(i) for i in re.findall("\d+", num_results.text)]
    #num_results = driver.find_element_by_xpath('//span[@id="searchResultsCount"]')
    num_pages = math.ceil(num_results[1]/50)

    for n in range(num_pages):
        #issue is that these elements will always be availible; how can you identify that a new page has come up?
#        dropdowns = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')))
#        rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')))
        
        #need to make sure next page has loaded before dropdowns are assigned
#        dropdowns = driver.find_elements_by_xpath('//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')
#        rows = driver.find_elements_by_xpath('//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')
        dropdowns = []
        z = iter(range(100))
        while len(dropdowns) != 50 and len(dropdowns) != num_results[1] %50:
            dropdowns = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')))
            rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')))     
            time.sleep(1)
            print(next(z))
        i = 0
        old_rows = []
        while i < 50:# iter thru page
            print(f"pg {n} lp{i}")
            #each time come back to the page; get a new instance
            row = rows[i]
            drop = dropdowns[i]
            time.sleep(0.5)
            #opens dropdown in main page
            if row.text == "":
                missing_rows.append(n*50 + i)
            row_text.append(row.text)#shouldn't append each time
            drop.click()
            
            #goto seperate pdf page
            goto_pdf = wait.until(EC.element_to_be_clickable((By.XPATH, "//img[@alt='Load Document']")))
            goto_pdf.click()
            
            #dropdown to open pdf
            pdf_dropdown = wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@id='downloadMenuButton']")))
                
            #commented so don't actually download
            pdf_dropdown.click()#option to download pdf
            driver.find_element_by_xpath("//a[@id='downloadAllButton']").click()#download pdf
            driver.switch_to.window(driver.window_handles[1])
            #in newly opened window
            time.sleep(0.5)#no attributes on the pdf page
            pdf_cookies.append(driver.get_cookies())
            pdf_urls.append(driver.current_url)
            get_pdf(driver.current_url)
            time.sleep(4)
            driver.close()
            
            driver.switch_to.window(driver.window_handles[0])#gets back to original window
            try:
                driver.find_element_by_xpath('//a[(@id="closeDoc")]').click()#goes back to search results page
            except:
                driver.find_element_by_xpath('//a[(@id="closeDoc2")]').click()
            
            i += 1
        if n != num_pages - 1:#don't go to next page on last day
            old_rows = rows
            try:      #clicks to next search results page
                next_results_page = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@class='page-link next']")))#to next page
                next_results_page.click()
            except:
                time.sleep(0.5)
                next_results_page = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[contains(concat( " ", @class, " " ), concat( " ", "next", " " ))]')))
                next_results_page.click()
            #hold till get to next search results page
            tm = datetime.now()
            while collections.Counter(old_rows) == collections.Counter(rows):
                time.sleep(0.5)
                dropdowns = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]//tbody//tr//td[@headers="thAction"]//div//a[@class="showActionMenu"]//img[@alt="Document Actions"]')))
                rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="searchResults"]/tbody/tr[not(contains(@style, "display:none;"))]')))
                if (datetime.now() - tm).seconds > 30:
                    next_results_page.click()#sometimes seems to timeout
                
                #wait till next page loads
#                page_number = driver.find_element_by_xpath('//*[@id="searchPagination"]//ul//li[@class="active"]//span[@class="current"]')
#                if int(page_number.text) == n+2:#got to next page
#                    break
        print(f"finished page {n+1}")
        time.sleep(0.75)
#%%
f = open('Desktop\side_projects\pdf_urls', 'a')
f.write("#######################")
f.write("\n".join(row_text[-1450:]))#row text != pdf_urls?
f.close()
past_driver = driver
with open('Desktop\side_projects\pdf_cookies', 'wb') as f:
    pickle.dump(pdf_cookies, f)
#%%
