import numpy as np
import urllib3
import json
import requests
import matplotlib.dates as mdates
import defs#mine
import pdb
import random, time, datetime

#%% Bloomberg scraping
def get_bbl_com_prx(ticker, driver):
    "Get's the *Front* month contract prices from bloomberg for a given Comdty ticker"
    if 'www.bloomberg.com' in ticker:
        url = ticker
    elif ticker in defs.abv_bbl_url:
        if defs.abv_bbl_url[ticker]:
            url = defs.abv_bbl_url[ticker]
        else:
            raise Exception("Ticker without URL")
    else:
        if len(ticker) == 1:
            url = f'https://www.bloomberg.com/quote/{ticker}\%201:COM'
        elif len(ticker) == 2:
            url = f'https://www.bloomberg.com/quote/{ticker}1:COM'
        else:
            raise Exception("Invalid ticker")
    try:
        diver.get(url)
        prx = driver.find_elements_by_xpath('//span[contains(@class, "priceText")]')
        return float(prx.replace(",", ""))
    except:
        driver.sleep(2)
        prx = driver.find_elements_by_xpath('//span[contains(@class, "priceText")]')
        return float(prx.replace(",", ""))

#%%
#CME endpoints
def convert_cme_quote(item):
    "converts string quote returned from CME endpoint to float"
    if item["productName"] in ('Corn Futures', 
                               "Soybean Futures", 
                               'Chicago SRW Wheat Futures', 
                               'KC HRW Wheat Futures'):
        qt = item['last']  if item['last'] != '-' else item['priorSettle']
        return float(qt[:-2]) + float(qt[-1])/8
    elif item['productName'] == 'Soybean Oil Futures':
        qt = item['last']  if item['last'] != '-' else item['priorSettle']
        return float(qt)/100
    else:
        if item['last'] != '-':
            return float(item['last'])
        elif item['priorSettle'] != '-':                                        
            return float(item['priorSettle']) 
        else:
            return np.nan

def get_cme_com_prx(ticker):
    """Get's ALL contract prices from CME for a given Comdty ticker
    If dynamic content, must have JS endpoint defined in defs.defs.abv_cme_jsEndpts
    else will be treated as static by getting that ticker's URL or treating ticker as URL
    eg. 'CL' return *tuples* of datetime expiries, most reccent price.
    """
    #have dynamically generated prices, use endpoints to process
    if ticker in defs.abv_cme_jsEndpts and defs.abv_cme_jsEndpts[ticker]:
        urls = defs.abv_cme_jsEndpts[ticker]
        for u in urls:
            try:
                r = requests.get(u).json()
                date_prx = [(datetime.strptime(item['expirationDate'], 
                                              '%Y%m%d'
                                              ), 
                            convert_cme_quote(item))
                             for item in r['quotes']] 
                                #if int(item['volume'].replace(",", "")) > 0]
                return list(zip(*date_prx)) 
            except Exception as e:
                print(e)
                last_exception = e
        else:
            print(f"Valid ticker: {ticker}, but stale/out of date endpoints")
            raise last_exception
    else:
        raise f"Invalid ticker: {ticker} doesn't have CME JS endpoint, so has 0-Volume"
#    elif ticker in defs.abv_cme_url:
#        url = defs.abv_cme_url[ticker]
#    elif 'www.cmegroup.com' in ticker:
#        url = ticker
#    #get static content
                
def get_all_cme_prx(cme_data = {}):
    """Updated cme_data dict with current CME prices, 
    5 sec average delay per request"""
    for tckr in defs.abv_cme_jsEndpts.keys():
        if tckr not in cme_data.keys():
            try:
                months, prices = get_cme_com_prx(tckr)
                cme_data[tckr] = [months, prices]
                time.sleep(8*random.random()+1)
            except Exception as e:
                print(tckr, e)
    return cme_data

def make_cme_df(cme_data):
    cme_d = [pd.Series(data = cme_data[k][1], 
                       index =cme_data[k][0],
                       name = k) 
                for k in cme_data.keys()]
    long_ix = longest_index(cme_d)
    df_idx = cme_d.pop(long_ix)
    cme_df = df_idx.to_frame().join(cme_d, how='outer')
    cme_d += [df_idx]
    return cme_df

#cme_df = make_cme_df(get_all_cme_prx(cme_data))
#%%
def get_barchart_historicals(ticker):
    pass

def make_barchart_historicals(tickers):
    pass
r = requests.get('https://www.barchart.com/futures/quotes/FLN20/overview')
r.content




#%%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from cycler import cycler
clz = plt.close('all')

def base_formatter(title):
    "returns fig, ax with all appropriate Initalized formating settings"
    fig, ax = plt.subplots(figsize=(16,12))
    ax.set_title(title, fontsize=20)
    ax.set_prop_cycle(cycler('color', ['c', 'c', 
                                       'y', 'y', 
                                       'm', 'm', 
                                       'k', 'k', 
                                       'r', 'r']))
    
    yr_loc = mdates.YearLocator()   # every year
    mo_loc = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')    
    # format the ticks
    ax.xaxis.set_major_locator(yr_loc)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(mo_loc)
    # format the coords message box
    yformatter = mticker.FormatStrFormatter('$%.0f')
    ax.yaxis.set_major_formatter(yformatter)
    
    ax.grid(True)
    plt.tick_params(labelsize=14)    
    return fig, ax
    
def end_formatter(dates, fig, ax):
    "modifies axis ax to contain the year bounds for dates"
    # round to nearest years.
    datemin = np.datetime64(dates[0], 'Y')
    datemax = np.datetime64(dates[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)
    
    plt.legend(prop={'size': 13})
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
#    removes white space
    plt.tight_layout()
    
def plot_prices(data_l, old_data_l, tckrs, save_path= ''):
    """takes 2 lists of dicts of 'date', and 'adj_close' 
    for forward forcasts and previous prices"""
    if not isinstance(data_l, list):
        data_l = [data_l]
        old_data_l = [old_data_l]
        tckrs = [tckrs]   
        title = f"Price of {tckrs[0]}: {defs.abv_name[tckrs[0]]}"    
    elif all([t in defs.abv_name for t in tckrs]):
        title = f"Prices of {', '.join([defs.abv_name[tckr] + f'({tckr})' for tckr in tckrs])}"
    else:
        title = f"Prices of {', '.join(tckrs)}"
    fig, ax = base_formatter(title)
        
    for data, old_data, tckr in zip(data_l, old_data_l, tckrs):
        ax.scatter('date', 'adj_close', data=data, 
                   s = 999, 
                   marker="+", 
                   label= tckr + ' Futures Prices')
        ax.scatter('date', 'adj_close', data=old_data,
                   s= 15, 
                   alpha = 0.5,
                   label = tckr +  ' Historical Prices' )   
    dates =sorted(old_data_l[0]['date']) + sorted(data_l[0]['date'])
    end_formatter(dates, fig, ax)

    #add uncertanty, grib

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
#multi_plots(tckrs_l, target_dir = "")#temp")

#%% 
os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio")
#cme_data = get_all_cme_prx()

def single_plots(target_dir = 'sing_prx_graphs'):
    "makes plots of individual commodities, and zips them"
    for (tckr, (months, prices)) in cme_data.items():
        data = {'date':months,
                'adj_close':prices}
        column = tckr + "1"
        old_data = {'date': curve_prices_df.index, 
                    'adj_close':curve_prices_df[column]}
        plot_prices(data, old_data, tckr, save_path = f"{target_dir}\\" + tckr)
#    os.system(f'powershell -command "Compress-Archive {target_dir} {target_dir}.zip"')
    
single_plots(target_dir = 'sing_prx_graphs')
#%% Makes plots of multiple values together
def multi_plots(tckrs_l, target_dir = 'mult_prx_graphs'):
    "Makes plots of multiple prices together"
    for tckrs in tckrs_l:
        tckrs = [tck for tck in tckrs 
                    if tck in cme_data]
        data_l = [{'date':cme_data[tck][0], 
                   'adj_close': cme_data[tck][1]}
                    for tck in tckrs]
        old_data_l = [{'date': curve_prices_df.index, 
                        'adj_close':curve_prices_df[tck + '1']}
                        for tck in tckrs]
        plot_prices(data_l, old_data_l, tckrs,
                    save_path = f"{target_dir}\\" + ", ".join(tckrs))
#    os.system(f'powershell -command "Compress-Archive {target_dir} {target_dir}.zip"')
        
tckrs_l = [('W', 'KW', 'MW', 'C'), ('S', 'SM', 'RS', 'BO'), ('CL', 'HO')]
multi_plots(tckrs_l)

#%% Plot Rin prices
import xlrd 
import os

def eia_renewable_table(table = 17):
    os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio\\Data")
    renew_file = "eia renewable fuels, All Tables in One.xls"
    xl_bk = xlrd.open_workbook(renew_file)
    if table == 17:
        b = xl_bk.sheet_by_name("Table017")
        dates = [i.replace("-", " ").replace(".", "").strip() for i in b.col_values(0)[8:-5]]
        dates = [datetime.strptime(i, "%b %y") for i in dates]
        biodiesel_prx = [float(i) for i in b.col_values(1)[8:] if i]
        diesel_prx = [float(i) for i in b.col_values(2)[8:] if i]
        return dates, biodiesel_prx, diesel_prx
    
def eia_renewable_table_plots(table=17):
    if table == 17:
        dates, biodiesel_prx, diesel_prx = eia_renewable_table(table=table)
        old_data_l = [{'date':dates, 
                           'adj_close': biodiesel_prx},
                            {'date':dates, 
                           'adj_close': diesel_prx}
                            ]
        data_l = [{'date': dates, 
                   'adj_close': [None]*len(dates)}]*2
        tckrs =("Bio", "Diesel")
        plot_prices(data_l, old_data_l, tckrs,
                            save_path = f"{target_dir}\\" + ", ".join(tckrs))
    
#eia_renewable_table_plots(table=17)
#%%







#%%
#Alternative Scrapping methods
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
driver_path = "C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\chromedriver_win32\\chromedriver.exe"

def get_blb(u = 'https://www.bloomberg.com/quote/CC1:COM', existing_driver = None):
    "Get current prices from Bloomberg"
    driver = existing_driver or webdriver.Chrome(executable_path = driver_path)
    wait = WebDriverWait(driver, 9)
    driver.get(u)
    val = wait.until(
            EC.presence_of_element_located(
                    (By.XPATH, '//span[contains(@class, "priceText")]')
                    ))
    print(val.text)
    prx = driver.find_elements_by_xpath('//span[contains(@class, "priceText")]')
    print([i.text for i in prx])
    if not existing_driver:
        driver.close()
        
def check_barchart(existing_driver = None):
    driver = existing_driver or webdriver.Chrome(executable_path = driver_path)
    wait = WebDriverWait(driver, 9)
    for ticker in defs.abv_name.keys():
        u = f"https://www.barchart.com/futures/quotes/{ticker}N20/overview"
        driver.get(u)
        try:
            name = wait.until(
                    EC.presence_of_element_located(
                            (By.XPATH, '//span[@class="symbol"]')#//*[@id="main-content-column"]/div/div[1]/div[1]/h1/span[1]s
                            ))
            print(f"{ticker}: {name.text}")
        except:
            print(f"{ticker}:    ,#FAILED")
        time.sleep(0.5)
    if not existing_driver:
        driver.close()

#driver = webdriver.Chrome(executable_path = driver_path)
#get_blb(existing_driver = driver)
#check_barchart(existing_driver = driver)
#%%

#%%TODO: Try and hve 2 y axis

def plot_prices(data_l, old_data_l, tckrs, save_path= ''):
    """takes 2 lists of dicts of 'date', and 'adj_close' 
    for forward forcasts and previous prices"""
    fig, ax = plt.subplots(figsize=(16,12))
    if not isinstance(data_l, list):
        data_l = [data_l]
        old_data_l = [old_data_l]
        tckrs = [tckrs]   
        plt.suptitle(f"Price of {tckrs[0]}: {defs.abv_name[tckrs[0]]}", 
                        fontsize=30)
    else:
        plt.suptitle(f"Prices of {', '.join([defs.abv_name[tckr] + f'({tckr})' for tckr in tckrs])}", 
                        fontsize=30)
    
    yr_loc = mdates.YearLocator()   # every year
    mo_loc = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    
    # format the ticks
    ax.xaxis.set_major_locator(yr_loc)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(mo_loc)
    
    # round to nearest years.
    dates =sorted(old_data_l[0]['date']) + sorted(data_l[0]['date'])
    datemin = np.datetime64(dates[0], 'Y')
    datemax = np.datetime64(dates[-1], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')    
    
    # format the coords message box
    formatter = mticker.FormatStrFormatter('$%.0f')
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True)
    plt.tick_params(labelsize=14)
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    
    ax.set_prop_cycle(cycler('color', ['c', 'c', 
                                       'y', 'y', 
                                       'm', 'm', 
                                       #'k', 'k', 
                                       'r', 'r']))
    for data, old_data, tckr in zip(data_l, old_data_l, tckrs):
        if len(tckrs) > 1 and tckr == tckrs[-1]:#double axis if >=2 products
            ax2 = ax.twinx()
            ax2.scatter('date', 'adj_close', data=data, 
                       color = 'k', 
                       s = 999, marker="+", label= tckr + ' Futures Prices')
            ax2.scatter('date', 'adj_close', data=old_data,
                       color = 'k',
                       alpha = 0.5,
                       s= 15, label = tckr +  ' Historical Prices' )
            ax2.legend(prop={'size': 13})

        else:
            ax.scatter('date', 'adj_close', data=data, 
                       #color = 'blue', 
                       s = 999, marker="+", label= tckr + ' Futures Prices')
            ax.scatter('date', 'adj_close', data=old_data,
                       #color = 'grey',
                       alpha = 0.5,
                       s= 15, label = tckr +  ' Historical Prices' )

    plt.legend(prop={'size': 13})

    try:
        ax2.yaxis.set_major_formatter(formatter)
    except:
        pass
    
    #add uncertanty, grib

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

