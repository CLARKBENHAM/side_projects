import numpy as np
import urllib3
import json
import requests
import matplotlib.dates as mdates

soymeal_url = "https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/soybean-meal_quotes_globex.html"
soy_json_u = "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?quoteCodes=null&_=1591390802348"
#%%
#http = urllib3.PoolManager()
#r = http.request('GET', soymeal_url)
#json.loads(r.data.decode('utf-8'))

#%%
#CME endpoints
r = requests.get(soy_json_u)
date_prx = [(datetime.strptime(item['expirationDate'], '%Y%m%d'), 
              float(item['last']))
                 for item in r.json()['quotes'] 
                    if int(item['volume'].replace(",", "")) > 0]
months, prices = list(zip(*date_prx)) 
#plt.plot()
#%%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

yr_loc = mdates.YearLocator()   # every year
mo_loc = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load a numpy structured array from yahoo csv data with fields date, open,
# close, volume, adj_close from the mpl-data/example directory.  This array
# stores the date as an np.datetime64 with a day unit ('D') in the 'date'
# column.

#with cbook.get_sample_data('goog.npz') as datafile:
#    data = np.load(datafile)['price_data']
#%%
data = {'date':months, 'adj_close':prices}
fig, ax = plt.subplots()
ax.scatter('date', 'adj_close', data=data)

# format the ticks
ax.xaxis.set_major_locator(yr_loc)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(mo_loc)

# round to nearest years.
datemin = np.datetime64(data['date'][0], 'Y')
datemax = np.datetime64(data['date'][-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
plt.show()



#%%
b = requests.get('https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/4707/G?quoteCodes=null&_=1560171518204').json()
print([(item['expirationDate'], item['last']) for item in b['quotes']])



#%%
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 9)
u = 'https://www.bloomberg.com/quote/CC1:COM'
driver.get(u)
val = wait.until(
        EC.presence_of_element_located(
                (By.XPATH, '//span[contains(@class, "priceText")]')
                ))
print(val.text)
prx = driver.find_elements_by_xpath('//span[contains(@class, "priceText")]')
print(prx.text)
#%%
#driver.get(soymeal_url)
#table = driver.find_element_by_xpath('//*[@id="quotesFuturesProductTable1"]')
#tb = driver.find_elements_by_xpath('//td[contains(@id, "quotesFuturesProductTable1_")]')

#%%
driver.close()


