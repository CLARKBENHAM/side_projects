# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:25:00 2020
@author: Clark Benham

cd ..\side_projects
copy ..\Stone_Presidio\*.py .\stone\*
git add .
git commit -m
git push
cd ..\Stone_Presidio
"""
import time, requests, random

#bloomberg
name_abv = {'Corn':'C',
         'Soybean':'S',
         'Chicago Wheat':'W', #Wheat Pit
         'Kansas Wheat':'KW', #hard Red Wheat
         'Minneapolis Wheat':'MW', #spring wehat
         'NY Cocoa':'CC',
         'London Cocoa':'QC',
         'Coffee':'KC',
         'Crude Oil':'CL',
         'Heating Oil':'HO',
         'Platinum':'PL',
         'Palladium':'PA',
         'NFDM':'LE',#Non-Fat Dry Milk
         'Soybean Oil':'BO',
         'Canola':'RS',
         'Soymeal':'SM',
         'CBOT Ethanol': 'EH',#deliverable
         'Chicago Ethanol': 'CU'#financial
         }
abv_name = {v:k for k,v in name_abv.items()}

cme_to_blb = {'CJ':'CC',
              'ZW': 'W'#CME Globex, but clearPort and Clearing use 'W'
              }
#for i in abv_name.keys():
#    print(f"'{i}': ',")
#%%
abv_cme_units = {'C': 'U.S. cents per bushel',
               'S': 'U.S. cents per bushel',
               'W': 'U.S. cents per bushel',
               'KW': 'U.S. cents per bushel',
               'MW': '',
               'CC': '',
               'CJ': 'U.S. dollars and cents per Metric Ton',
               'QC': '',
               'KC': 'U.S. dollars and cents per pound',
               'CL': 'U.S. dollars and cents per barrel',
               'HO': 'U.S. dollars and cents per gallon',
               'PL': 'U.S. dollars and cents per troy ounce',
               'PA': 'U.S. dollars and cents per troy ounce',
               'BO': 'U.S. cents per pound',
               'RS': '',
               'SM': 'U.S. dollars and cents per short ton',
               'EH': 'U.S. dollars and cents per gallon',
               'CU': 'U.S. dollars and cents per gallon'}

def unit_formatter(unit_des):
    if 'U.S. dollars and cents per ' in unit_des:
        return chr(36) + unit_des.replace('U.S. dollars and cents per ', '/')
    elif 'U.S. cents per ' in unit_des:
        return chr(162) + unit_des.replace('U.S. cents per ', '/')
    else:
        return unit_des

abv_formatted_units = {k:unit_formatter(v) for k,v in abv_cme_units.items()}
    
abv_cme_url = {'C': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/corn.html',
                'S': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/soybean.html',
                'W': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/wheat.html',
                'KW': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/kc-wheat.html',
                'MW': '',
                'CC': '', #is an ICE Unit
                'CJ': 'https://www.cmegroup.com/trading/agricultural/softs/cocoa.html',
                'QC': '',
                'KC': 'https://www.cmegroup.com/trading/agricultural/softs/coffee.html',#no CME volume; ICE abreviation; ,
                'CL': 'https://www.cmegroup.com/trading/energy/crude-oil/light-sweet-crude.html',
                'HO': 'https://www.cmegroup.com/trading/energy/refined-products/heating-oil.html',
                'PL': 'https://www.cmegroup.com/trading/metals/precious/palladium.html',
                'PA': 'https://www.cmegroup.com/trading/metals/precious/platinum.html',
                'LE': None,
                'BO': 'https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/soybean-oil.html',
                'RS': '',#'https://www.theice.com/products/251/Canola-Futures',#ICE
                'SM': "https://www.cmegroup.com/trading/agricultural/grain-and-oilseed/soybean-meal_quotes_globex.html",
                'EH': "https://www.cmegroup.com/trading/energy/ethanol/cbot-ethanol_quotes_globex.html", #no Volume
                'CU': 'https://www.cmegroup.com/trading/energy/ethanol/chicago-ethanol-platts-swap_quotes_globex.html'
                }

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
                'CJ': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/423/G?quoteCodes=&_'],
                'QC': None,
                'KC': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/440/G?quoteCodes=&_'],
                'CL': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?pageSize=50&_=1591409833680',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?pageSize=50&_=1591409833681',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?quoteCodes=null&_=1591409833682'],
                'HO': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/426/G?quoteCodes=&_=1591409968011',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/426/G?pageSize=50&_=1591409968013',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/426/G?pageSize=50&_=1591409968014'],
                'PL': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/446/G?quoteCodes=&_'],
                'PA': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/445/G?quoteCodes=&_'],
                'LE': None,
                'BO': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/312/G?quoteCodes=&_=1591407855935',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/312/G?pageSize=50&_=1591407855937',
                       'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/312/G?pageSize=50&_=1591407855938'],
                'RS': None, #['https://futures.tradingcharts.com/marketquotes/RS.html'],
                'SM': ["https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?quoteCodes=null&_=1591390802348",
                          "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?quoteCodes=null&_=1591390802347",
                          "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?quoteCodes=null&_=1591390802349",
                          "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/310/G?pageSize=50&_=1591390802344"],
                'EH': ['https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/338/G?quoteCodes=&_'],
                'CU': None}

def get_CME_endpoints(start = 344, end=500):
    with open("cme_notation", 'a') as f:
        while start < end:
            try:
                for i in range(start, end):
                    u = f'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/{i}/G?quoteCodes=&_'
                    r = requests.get(u)
                    if r.content != b'' and  not r.json()['empty']:
                        r = r.json()
                        print(f"{i}: {r['quotes'][0]['productName']}", file=f)
                        print(f"{i}: {u},","\n", file=f)
                    time.sleep(1+8 
                               *random.random())
            except:
                print("failed on", i)
                time.sleep(10)
            start = i + 1      
#get_CME_endpoints(start=500, end=1000)
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
#https://www.barchart.com/futures/quotes/KCN20/historical-prices?page=all
#barchart.com specific tikers
barcharts_tickers= {'QC': 'CA',
                    'LE': 'DF',
                    'EH': 'FH',#Barchart is NY, CME is CME GLobex
                    'CU': 'FL'#Barchart is CHI, CME is CHI; quotes differ
                        }
barcharts_tickers.update({k:k for k in abv_name.keys() 
                            if k not in barcharts_tickers})
#%%        
con_month_abv = {'January': 'F',
                'February': 'G',
                'March': 'H',
                'April': 'J',
                'May': 'K',
                'June': 'M',
                'July': 'N',
                'August': 'Q',
                'September': 'U',
                'October': 'V',
                'November': 'X',
                'December': 'Z'}

month_abv_con = {k[:3]: v for k,v in con_month_abv.items()}
int_month_abv = {i+1:v for i, v in enumerate(con_month_abv.values())}

#%%

def write_data():
    "writes all data"

    out_col_names = [abv_name[re.sub("\d+", "",i)] + "_" + re.findall("\d+",i)[0]
                        for i in futures]
    curve_prices_df.columns = out_col_names
    curve_prices_df.to_csv("Historical prxs")
    
    #CME DATA
#    if 'cme_data' not in globals(): 
#        cme_data = get_all_cme_prx()
#    cme_df = cme_scrapper.make_cme_df(globals()['cme_data'])
    cme_df.to_csv("Spot Futures")

    #Notes
    def ticker_notes(k):
        "Data for CME ticker k"
        return [abv_name[k], 
                abv_cme_units[k], 
                abv_cme_url[k], 
                abv_name[k] + f" ({abv_formatted_units[k]})"]
    notes_body = {k: ticker_notes(k) if k not in cme_to_blb \
                      else ticker_notes(cme_to_blb[k])
                        for k in cme_data.keys()
                            if k and k!=''
                        }
    pd.DataFrame(notes_body,
                  index = ['Name', 'Price Def', 'Title', 'Link' ]
                ).to_csv("notes")
    #RINS
 
    return None

# write_data()
#%%
groups = {'grains': ('C', 'S', 'W', 'KW', 'MW', 'SM'),
          'coffe_cocoa': ('CC', 'QC', 'KC'),
          'ag_oil': ('RS', 'SM'),
          'other_ags': ('LE'),
          'hydrocarbons': ('CL', 'HO'),
          'bio-energy': ('EH', 'CU'),
          'metals': ('PL', 'PA')
          }


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