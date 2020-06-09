# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:25:00 2020

@author: Clark Benham
"""
#copy ..\Stone_Presidio\*.py .\stone\*


name_abv = {'Corn':'C',
         'Soybean':'S',
         'Chicago Wheat':'W',
         'Kansas Wheat':'KW',
         'Minneapolis Wheat':'MW',
         'NY Cocoa':'CC',
         'London Cocoa':'QC',
         'Coffee':'KC',
         'Crude Oil':'CL',
         'Heating Oil':'HO',
         'Platinum':'PL',
         'Palladium':'PA',
         'NFDM':'LE',#Non-Fat Dry Milk
         'BeanOil':'BO',
         'Canola':'RS',
         'Soymeal':'SM'}
abv_name = {v:k for k,v in name_abv.items()}

#for i in abv_name.keys():
#    print(f"'{i}': ',")
#%%
abv_bbl_url = {'C': 'https://www.bloomberg.com/quote/C%201:COM',
               'S': 'https://www.bloomberg.com/quote/S%201:COM',
               'W': 'https://www.bloomberg.com/quote/W%201:COM',
               'KW': '',
               'MW': '',
               'CC': 'https://www.bloomberg.com/quote/CC1:COM',
               'QC': '',
               'KC': 'https://www.bloomberg.com/quote/KC1:COM',
               'CL': 'https://www.bloomberg.com/quote/CL1:COM',
               'HO': 'https://www.bloomberg.com/quote/HO1:COM',
               'BO': 'https://www.bloomberg.com/quote/BO1:COM',
               'RS': 'https://www.bloomberg.com/quote/RS1:COM',
               'SM': 'https://www.bloomberg.com/quote/SM1:COM'} 
        
#%%
abv_cme_url = {'C': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/corn.html',
                'S': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/soybean.html',
                'W': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/wheat.html',
                'KW': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/kc-wheat.html',
                'MW': '',
                'CC': '',#no CME volume; ICE abreviation; https://www.cmegroup.com/trading/agricultural/softs/coffee.html
                'QC': '',
                'KC': '',
                'CL': 'https://www.cmegroup.com/trading/energy/crude-oil/light-sweet-crude.html',
                'HO': 'https://www.cmegroup.com/trading/energy/refined-products/heating-oil.html',
                'PL': None,
                'PA': None,
                'LE': None,
                'BO': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/soybean-oil.html',
                'RS': 'https://www.theice.com/products/251/Canola-Futures',#ICE
                'SM': "https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/soybean-meal_quotes_globex.html"}

abv_cme_jsEndpts = {'C': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/300/G?quoteCodes=&_=1591408462672',
                          'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/300/G?pageSize=50&_=1591408462674',
                          'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/300/G?pageSize=50&_=1591408462675'],
                'S': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/320/G?quoteCodes=&_=1591408664446',
                      'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/320/G?pageSize=50&_=1591408664448',
                      'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/320/G?pageSize=50&_=1591408664449'],
                'W': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/323/G?quoteCodes=&_=1591408794487',
                      'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/323/G?quoteCodes=null&_=1591408794491',
                      'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/323/G?pageSize=50&_=1591408794490'],
                'KW': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/348/G?quoteCodes=&_=1591408953599',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/348/G?pageSize=50&_=1591408953601',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/348/G?pageSize=50&_=1591408953602'],
                'MW': None, #['https://www.barchart.com/futures/quotes/MW*0/futures-prices'],
                'CC': None,
                'QC': None,
                'KC': None,
                'CL': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?pageSize=50&_=1591409833680',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?pageSize=50&_=1591409833681',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?quoteCodes=null&_=1591409833682'],
                'HO': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/426/G?quoteCodes=&_=1591409968011',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/426/G?pageSize=50&_=1591409968013',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/426/G?pageSize=50&_=1591409968014'],
                'PL': None,
                'PA': None,
                'LE': None,
                'BO': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/312/G?quoteCodes=&_=1591407855935',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/312/G?pageSize=50&_=1591407855937',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/312/G?pageSize=50&_=1591407855938'],
                'RS': None, #['https://futures.tradingcharts.com/marketquotes/RS.html'],
                'SM': ["https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?quoteCodes=null&_=1591390802348",
                          "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?quoteCodes=null&_=1591390802347",
                          "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?quoteCodes=null&_=1591390802349",
                          "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?pageSize=50&_=1591390802344"]}

#%%
#https://www.barchart.com/futures/quotes/KCN20/historical-prices?page=all
                
#%% 

#%% temp
#collection_date = max([i for i in curve_prices_df.index 
#                       if type(i) == pd._libs.tslibs.timestamps.Timestamp])
#
#futures = [i.replace("COMB", "").replace("Comdty", "").replace(" ", "") 
#            for i in curve_prices_df]#eg CL 1
#futures_ab = set([re.sub("\d+", "",i) 
#                    for i in futures])#eg CL
#
#has_expired = {i:False  #if the contract trading on dec 18th 2019 is for January settle
#                   for i in futures_ab}
#contract_moAhead = [re.findall("([a-zA-Z]+)(\d+)",i)[0] 
#                       for i in futures]#CL 8, not CL X20
#for contract, mo in contract_moAhead: 
#    if mo == '0':
#        has_expired[contract] = True
#
#expiry_month = {con + mo: collection_date 
#                        + relativedelta(months=int(mo)+has_expired[con])
#                            for con, mo in contract_moAhead}



#contract_expiry_dates = {'BO':14,
#                         'C ':14,
#                         'CC',
#                         'CL':20,
#                         'HO',
#                         'KC',
#                         'KW',
#                         'LE',
#                         'MW',
#                         'PA',
#                         'PL',
#                         'QC',
#                         'RS',
#                         'S ':14,
#                         'SM':14,
#                         'W ':14}