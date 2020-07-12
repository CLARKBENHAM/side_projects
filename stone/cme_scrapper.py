import numpy as np
import pandas as pd
import json
import requests
import matplotlib.dates as mdates
import pdb
import random, time
from datetime import datetime


import defs#mine
#%%
#CME endpoints
def convert_cme_quote(item):
    "converts string quote returned from CME endpoint to float"
    if item['last'] != '-':
        qt = item['last']
    elif item['priorSettle'] != '-':
        qt = item['priorSettle']
    else:
        return np.nan

    if item["productName"] in ('Corn Futures',
                               "Soybean Futures",
                               'Chicago SRW Wheat Futures',
                               'KC HRW Wheat Futures'):
        (dollars, eigth_cents) = qt.split("'")
        return float(dollars) + float(eigth_cents)/8
    elif item['productName'] == 'Soybean Oil Futures':
        return float(qt)/100
    else:
        return float(qt)

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
                pdb.set_trace()
                print(e)
                last_exception = e
        else:
            raise f"Valid ticker: {ticker}, but stale/out of date endpoints"
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
                print(f"Getting {tckr} failed: ", e)
    return cme_data

def make_cme_df(cme_data=None):
    if cme_data is None:
        cme_data = get_all_cme_prx()
    cme_d = [pd.Series(data = cme_data[k][1],
                       index =cme_data[k][0],
                       name = k)
                for k in cme_data.keys()]
    cme_df = pd.concat(cme_d,
                              axis=1,
                              join='outer',
                              sort=True).iloc[::-1]
    return cme_df


#%%
#TODO
def get_barchart_historicals(ticker):
    pass

def make_barchart_historicals(tickers):
    pass
# r = requests.get('https://www.barchart.com/futures/quotes/FLN20/overview')
# r.content

#%%
def corp_bond_fred():
    """Gets FRED's bond data; from following links:
        (IG) https://fred.stlouisfed.org/release/tables?rid=402&eid=219299
        (BB) https://fred.stlouisfed.org/series/BAMLH0A1HYBB
        (CCC) https://fred.stlouisfed.org/series/BAMLH0A3HYC
        """
    def get_fred(series_id, make_series = False, extra_specs = ""):
        key = "8013774bf23efdce1c2a57dc5b021689"
        url = f"https://api.stlouisfed.org/fred/series/observations?" \
                + f"series_id={series_id}&api_key={key}" \
                + "&observation_start=2000-01-01&file_type=json" \
                + extra_specs
        r = requests.get(url).json()
        if make_series:
            print(series_id)
            return pd.Series([float(i['value']) if i['value'] != "." else np.nan for i in r['observations']],
                             index =[datetime.strptime(i['date'], "%Y-%m-%d")
                                     for i in r['observations']],
                             name = series_id)[::-1]
        else:
            return r

    corp_bond_series = "HQMCB"#high quality
    bonds = [(i//2, i%2) for i in list(range(1,15)) + [20, 30, 40, 60]]#don't need past 30yrs
    bond_ids = [corp_bond_series + "6MT" if y == 0
                else corp_bond_series + f"{y}YR" if m == 0
                else corp_bond_series + f"{y}Y{6}M"
                for y,m in bonds]
    reqs = [None]*len(bond_ids)
    for i,b in enumerate(bond_ids):
        reqs[i] = _get_fred(b)
    #what if dates don't match?
    corp_yc = pd.DataFrame([[float(i['value']) for i in j['observations']]
                             for j in reqs],
                        columns = [datetime.strptime(i['date'], "%Y-%m-%d")
                                   for i in reqs[0]['observations']],
                        index = [bond_ids]).T[::-1]
    #ICE, option adjusted spreads
    ccc_bonds = _get_fred("BAMLH0A3HYC", make_series=True)
    bb_bonds = _get_fred("BAMLH0A1HYBB", make_series=True)
    b_bonds = _get_fred("BAMLH0A2HYB", make_series=True)
    #Moody
    baa_bonds = _get_fred("DBAA", make_series=True)
    aaa_bonds = _get_fred("DAAA", make_series=True)

    corp_credit = pd.concat((ccc_bonds, bb_bonds, b_bonds, baa_bonds, aaa_bonds),
                            axis=1)
    # corp_credit[corp_credit == "."] = np.nan
    # corp_credit = corp_credit.astype(np.float64, errors='raise')
    # corp_credit.index = pd.to_datetime(corp_credit.index)
    return corp_yc, corp_credit


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

#%%
def single_plots(cme_data, target_dir = 'sing_prx_graphs', save_plots = False):
    "makes plots of individual commodities, and zips them"
    for (tckr, (months, prices)) in cme_data.items():
        data = {'date':months,
                'adj_close':prices}
        column = tckr + "1"
        old_data = {'date': curve_prices_df.index,
                    'adj_close':curve_prices_df[column]}
        plot_prices(data, old_data, tckr, save_path = f"{target_dir}\\" + tckr)
    if save_plots and target_dir:
        os.system(f'powershell -command "Compress-Archive {target_dir} {target_dir}.zip"')

# Makes plots of multiple values together
def multi_plots(tckrs_l, target_dir = 'mult_prx_graphs', save_plots = False):
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
    if save_plots and target_dir:
        os.system(f'powershell -command "Compress-Archive {target_dir} {target_dir}.zip"')

# single_plots(target_dir = 'sing_prx_graphs')
# tckrs_l = [('W', 'KW', 'MW', 'C'), ('S', 'SM', 'RS', 'BO'), ('CL', 'HO')]
# multi_plots(tckrs_l)

#%% Plot Rin prices
import xlrd
import os

def eia_renewable_table(table = 17):
    os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio\\Data")
    renewables_file = 'eia renewable fuels, All Tables in One.xlsx'
    xl_bk = xlrd.open_workbook(renewables_file)
    if table == 17:
        b = xl_bk.sheet_by_name("Table017")
        dates = [i.replace("-", " ").replace(".", "").strip() for i in b.col_values(0)[8:-5]]
        dates = [datetime.strptime(i, "%b %y") for i in dates]
        biodiesel_prx = [float(i) for i in b.col_values(1)[8:] if i]
        diesel_prx = [float(i) for i in b.col_values(2)[8:] if i]
        bio_df = pd.DataFrame({'Retail Biodiesel': biodiesel_prx,
                               'Retail Diesel': diesel_prx},
                           index = dates)
        return  bio_df.reindex(sorted(dates, reverse = True))

def eia_renewable_table_plots(table=17, target_dir = ""):
    if table == 17:
        bio_df = eia_renewable_table(table=table)
        old_data_l = [{'date':bio_df.index,
                           'adj_close': bio_df['Retail Biodiesel']},
                            {'date':bio_df.index,
                           'adj_close': bio_df['Retail Diesel']}
                            ]
        data_l = [{'date': bio_df.index,
                   'adj_close': [None]*len(bio_df.index)}]*2
        tckrs =("Bio", "Diesel")
        if target_dir:
            plot_prices(data_l, old_data_l, tckrs,
                                save_path = f"{target_dir}\\" + ", ".join(tckrs))
        else:
            plot_prices(data_l, old_data_l, tckrs)


# eia_bio_df = eia_renewable_table(table=17)
# eia_renewable_table_plots(table=17)
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
        driver.get(url)
        prx = driver.find_elements_by_xpath('//span[contains(@class, "priceText")]')
        return float(prx.replace(",", ""))
    except:
        driver.sleep(2)
        prx = driver.find_elements_by_xpath('//span[contains(@class, "priceText")]')
        return float(prx.replace(",", ""))



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
    "Want to know if the CME ticker's I'm using correspond to the correct Barchart Tickers"
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

def plot_prices_2ax(data_l, old_data_l, tckrs, save_path= ''):
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

