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
#%%
base_url = "https://finance.yahoo.com/quote/GME/options"
r = requests.get(base_url)
root = lxml.html.fromstring(r.content)
root.make_links_absolute(base_url)
url_l = root.xpath('//*[@id="Col1-1-OptionContracts-Proxy"]/section/div/div[1]/select')
url_expiry_dates = url_l[0].value_options
urls = [url +f"?date={i}" for i in url_expiry_dates]

header = root.xpath("(//thead)[1]/tr/th/span")
header = [i.text.replace(" ", "_").replace("%", "Per") for i in header]
assert header == ['Contract_Name',
         'Last_Trade_Date',
         'Strike',
         'Last_Price',
         'Bid',
         'Ask',
         'Change',
         'Per_Change',
         'Volume',
         'Open_Interest',
         'Implied_Volatility'], \
        "Columns have changed"
row_tup = namedtuple("rows", header)

d = {u:[] for u in urls}
df = pd.DataFrame()
#%%
def proc_row(row):
        """header = ('Contract_Name',
             'Last_Trade_Date',
             'Strike',
             'Last_Price',
             'Bid',
             'Ask',
             'Change',
             'Per_Change',
             'Volume',
             'Open_Interest',
             'Implied_Volatility')"""
        out = [i.text for i in row.xpath("td")]
        # Contract_name, Strike
        out[0], out[2] = [i.text for i in row.xpath("td/a")]
        for i in (2,3,4,5, 8,9,10):
            if out[i] == "-":
                out[i] = 0
            else:
                out[i] = float(re.sub("\%|\,", "", out[i]))
        #Change, %change
        out[6], out[7] = [float(re.sub("\%|\,", "",i.text))
                          if i.text != '-' else 0 
                          for i in row.xpath("td/span")]
        #Last Trade 
        #but hours aren't 0 padded!?
        out[1] = datetime.strptime(out[1], "%Y-%m-%d %I:%M%p EST")
        return row_tup._make(out)

for url in urls[5:]:
    r2 = requests.get(url)
    d[url] += [r2]
    option_page = lxml.html.fromstring(r2.content)
    calls, puts = option_page.xpath("//tbody")    
    call_objs = [proc_row(row) for row in calls.xpath("tr")]
    put_objs = [proc_row(row) for row in puts.xpath("tr")]
    d[url] += [call_objs, put_objs]
    df = df.append(put_objs).append(call_objs)
    time.sleep(random.random()*3)

print(df)
#%%
df.to_pickle("current_option_prices")
df =  pd.read_pickle("current_option_prices")

#%% Scrap
def dfs(e):
    if isinstance(e, list):
        return [dfs(e) for i in e]
    q = [e]
    out = []
    while len(q) > 0:
        e = e.pop(-1)
        try:
            kids = header[0].xpath("/*")
            q += kids
        except:
            out += [e]
    return out
