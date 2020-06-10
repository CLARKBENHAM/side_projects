import numpy as np
import pandas as pd
import scipy.stats as ss
import xlrd
import openpyxl
import os
import sklearn
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import re

os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio\\Data")
prices_file = "16.16 Historical Commodity Price Data.xlsx"

#xl_bk = xlrd.open_workbook(prices_file) 
commodities = xl_bk.sheet_names()
security_d = []
curve_prices_d = []
ix = 15
b = xl_bk.sheet_by_index(ix)
sz = len(b.row_values(6))
date_lens = [len(b.col_values(i)) 
                for i in range(0,sz,4)]
long_col = 4*date_lens.index(max(date_lens))
dates = [datetime(*xlrd.xldate_as_tuple(i,0)) if type(i) == float
         else None
            for i in b.col_values(long_col)[7:]]
com_maturities = b.row_values(0)[1::4]
com_ab = commodities[ix]
    
short_col = 4*date_lens.index(min(date_lens))
short_col_len = min(date_lens)
short_col_date = b.col_values(short_col)[-1]
is_date_aligned = [b.col_values(i)[short_col_len-1] == short_col_date
                       for i in range(0,sz,4)]
bad_date_cols = [4*ix for ix, i in enumerate(is_date_aligned) 
                                if not i]

bad_date_cols
#%%import Data from Com pricing spreadsheet

#com_dict = {name:pd.read_excel("16.16 Historical Commodity Price Data.xlsx", name )
#            for name in xl_bk.sheet_names()}
os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio\\Data")
prices_file = "16.16 Historical Commodity Price Data.xlsx"

def int_col(n,base=26):
    "convert 0 index column number to excel column name"
    if n < 0:
        return ""
    elif n < base: 
        return chr(65 + n)         
    return int_col(n//base-1,base) + int_col(n%base,base)

#file_name = prices_file
#sheet_ix = 15
#target_indx = dates
def make_blb_row_aligned(file_name, sheet_ix, bad_date_cols, target_indx, bk = None):
    """for each sheet in this file of bloomberg formatted data, all columns will
        be row aligned on each date. Must be xlsx workbook.
        file_name: name of book this operation is to be performed on
        sheet_ix:
        bad_date_cols: list of columns of dates(as ints) which needs to be extended+next 2 cols
        target_indx: the dates that data_ix will now hold (as dt objects)
        returns: the book that was opened, *MUST* be {{saved + closed}} by caller"""
#    target_indx = [i.strftime("%Y-%m-%d %H:%M:%S") for i in target_indx]
    bk = bk or openpyxl.load_workbook(file_name, data_only = True)                 
    sheet = bk.worksheets[sheet_ix]
    for c in bad_date_cols:
        d_col = [i.value for i in sheet[int_col(c)]][7:]
        for i,v in enumerate(reversed(d_col)):#openpyxl appends a bunch of empty cells
            if v is not None:   
                valid_till = len(d_col) - i
                break
        d_col = d_col[:valid_till]
        t_col =[i.value for i in sheet[int_col(c+1)]][7:valid_till]
        v_col = [i.value for i in sheet[int_col(c+2)]][7:valid_till]
        for ix, v in enumerate(target_indx):
            if d_col[ix] !=v and d_col[ix] != "#N/A N/A": 
                print(f"inserted {v} @ {ix}, col {c}, was {d_col[ix]}")
                d_col.insert(ix, v)
                t_col.insert(ix, "#N/A N/A")
                v_col.insert(ix, "#N/A N/A")
#                time.sleep(0.1)
        t_col += [ "#N/A N/A"]*(len(d_col) - len(t_col))#??, not sure why not same sz
        v_col += [ "#N/A N/A"]*(len(d_col) - len(v_col))
    
        #writeback uses excel locations, 1-indexed not 0
        for row in range(8, 8 + len(d_col)):
            sheet.cell(column=c+1, row=row, value="{0}".format(d_col[row-8]))
            sheet.cell(column=c+2, row=row, value="{0}".format(t_col[row-8]))
            sheet.cell(column=c+3, row=row, value="{0}".format(v_col[row-8]))
    return bk
    
#make_blb_row_aligned(prices_file, 
#                     15, 
#                     bad_date_cols,
#                     dates)
#%%
    
def get_blb_excel(prices_file, individual_securities = True):
    """Reads in data from a workbook with many sheets of Bloomberg formated
    historical future's prices, with Top Row Dates aligned and 
    the **Bottom** date of each contract aligned. 
        [will fail on Jan 1, Jan 3, Jan 4 vs. Jan 1, Jan 2, Jan 4]
    """
    wrote_bk = None
    xl_bk = xlrd.open_workbook(prices_file)
    commodities = xl_bk.sheet_names()
    security_d = []
    curve_prices_d = []
    for ix, name in enumerate(commodities):
        b = xl_bk.sheet_by_index(ix)
        sz = len(b.row_values(6))
        date_lens = [len(b.col_values(i)) 
                        for i in range(0,sz,4)]
        long_col = 4*date_lens.index(max(date_lens))
        dates = [datetime(*xlrd.xldate_as_tuple(i,0)) if type(i) == float
                 else None
                    for i in b.col_values(long_col)[7:]]
        com_maturities = b.row_values(0)[1::4]
        com_ab = commodities[ix]
            
        short_col = 4*date_lens.index(min(date_lens))
        short_col_len = min(date_lens)
        short_col_date = b.col_values(short_col)[-1]
        is_date_aligned = [b.col_values(i)[short_col_len-1] == short_col_date
                               for i in range(0,sz,4)]
        if not all(is_date_aligned):
            print(f"WARNING: Row's not date aligned, some days missing, for {name}")
            bad_date_cols = [i for i in enumerate(is_date_aligned) 
                                if not i]
            #keeps book open in case have to write to another sheet; opening takes forever
            wrote_bk = make_blb_row_aligned(prices_file, 
                                 ix, 
                                 bad_date_cols,
                                 b.col_values(long_col)[7:],
                                 bk = wrote_bk)
            
        curve_prices = pd.DataFrame(np.array([pd.to_numeric(b.col_values(i)[7:],
                                                            errors = 'coerce'
                                                            )
                                                for i in range(2,sz,4)
                                                ]).T,
                                     columns = com_maturities,
                                     index = dates,
                                     )            
        curve_prices_d += [curve_prices]#.dropna()

        if individual_securities:#slow
            sec_df = pd.DataFrame(list(zip(*[b.col_values(i)[7:]
                                                    for i in range(1,sz,4)]
                                                    )) )
            securities = np.unique(sec_df.values)
            securities = securities[~np.isin(securities, ('', '#N/A N/A'))]
                                                          
            def get_security(s):
                "returns prices for 1 single future"
                rows, _ = np.where(sec_df == s)
                #get all of row, column not just those indexes.
                df = pd.Series(data = curve_prices.values[sec_df == s], 
                          index = curve_prices.index[rows],
                          name = s
                          ).dropna()
                return df[~pd.isna(df.index)]
            sec_list = [get_security(s) for s in securities]
            security_d += sec_list
    if wrote_bk:
        wrote_bk.save(prices_file)
        wrote_bk.close()
    xl_bk.close()
    return curve_prices_d, security_d
        
def longest_index(df_list):
    "returns the index of the longest dataframe from a list of df's"
    return max([(i, df.shape[0]) 
                for i, df in enumerate(df_list)], 
                   key = lambda i: i[1]
                   )[0]
curve_prices_d, security_d = get_blb_excel(prices_file)

long_ix = longest_index(curve_prices_d)
df_idx = curve_prices_d.pop(long_ix)
curve_prices_df = df_idx.join(curve_prices_d)
#curve_prices_df.dropna(inplace = True)
#curve_prices_df.reindex(curve_prices_df.index[::-1]) #NA axis
curve_prices_d += [df_idx]

futures = [i.replace("COMB", "").replace("Comdty", "").replace(" ", "") 
            for i in curve_prices_df]#eg CL 1
futures_ab = set([re.sub("\d+", "",i) 
                    for i in futures])#eg CL
curve_prices_df.columns = futures


#%%
os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio")
import defs
#import cme_scrapper

def write_data():
    "writes all data"
    out_col_names = [defs.abv_name[re.sub("\d+", "",i)] + "_" + re.findall("\d+",i)[0]
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
        return [abv_name[k], 
                     abv_cme_units[k], 
                     abv_cme_url[k], 
                     abv_name[k] + f" ({abv_formatted_units[k]})"]
    notes_body = {k: ticker_notes(k) if k not in defs.cme_to_blb \
                      else ticker_notes(defs.cme_to_blb[k])
                        for k in cme_data.keys()
                            if k and k!=''
                        }
    pd.DataFrame(notes_body,
                  index = ['Name', 'Price Def', 'Title', 'Link' ]
                ).to_csv("notes")
    #RINS
 
    return None

write_data()

#%% figuring out how to combine securities
t = time.time()
long_ix = longest_index(security_d)
df_idx = security_d.pop(long_ix)
securities_df = df_idx.to_frame().join(security_d)
security_d += [df_idx]
print("1: ", time.time() - t)
#%%
t = time.time()
securities_df  = curve_prices_df
securities_df.drop(labels=securities_df.columns,
                   axis=1,
                   inplace=True)
securities_df = securities_df.join(security_d, how='outer')
print("2: ", time.time() - t)

#securities_df = pd.concat(security_d, axis=1, join='outer', sort=True)
#%%
future_contracts = [i.name for i in security_d]
security_df = pd.DataFrame(security_d, 
                           columns = future_contracts, 
                           index = curve_prices_df)

#%%
securities_df = pd.concat(security_d, axis=1, join_axes = [curve_prices_df.index])
#%%
#have duplicates na's
securities_df = pd.concat(security_d, axis=1, join='outer', sort=True)

#%%
l= [i.to_frame() for i in security_d ]
pd.concat(l, axis=1, join='outer', sort=True)
#%%
cs = []
for i in security_d:
    if list(i.index) != list(set(i.index)):
        c = i
        s = set(c.index)
        for j in c.index:
            try:
                s.remove(j)
            except:
                print(c.name, j)
#%%
for c in cs:
    
#%%

#%%
a = pd.Series([1,2,3,4], index=[np.nan,2,8,9], name='1')
b = pd.Series([5,6,7,8], index=[np.nan,2,8, 9], name='1')
c = pd.Series([5,6,7,8], index=[np.nan, 6,7,8], name='2')
l = [a,a,c]
pd.concat(l, axis=1, join='outer', sort=True)

#%%
a = pd.Series([1,2,3,4]*100, index=list(range(1,401)), name='1')
b = pd.Series([5,6,7,8]*100, index=list(range(400)), name='1')
c = pd.Series([5,6,7,8]*1000, index=list(range(2000))*2, name='2')
l = [a,a,b, c]
pd.concat(l, axis=1, join='outer', sort=True)


#%% Make Price Graphs from future's

#%%
from scipy import integrate
import defs

hist_val = curve_prices_df.reindex(curve_prices_df.index[::-1])
hist_pct = hist_val.pct_change(periods = 252)#6mo out
hist_vol = hist_pct.std(axis=0)

for ab in defs.abv_cme_url:
    if ab:
        tick = ab + "1"
        ix = futures.index(tick)
        base_ix = hist_val.iloc[:,ix].first_valid_index()
        base_prx = hist_val.loc[base_ix, tick]
        dist = ss.lognorm(loc = 0, scale = 1, s = hist_vol.values[ix])
        
        x = np.linspace(hist_pct.iloc[:,ix].min(),
                        hist_pct.iloc[:,ix].max(),
                        100)*2
        
        fig, ax =plt.subplots()
        x_axis = [i for i in  base_prx * (1 + x) if i > 0]
        
        probs = [integrate.quad(lambda x: dist.pdf(x/base_prx)/base_prx, i,j)[0]
                         for i,j in zip(x_axis[:-1], x_axis[1:])]
        probs = [100*i/sum(probs) for i in probs]
        
        plt.bar(x_axis[:-1], probs, label="Probabilities")
        plt.legend()
        plt.title(f"Probability of ending prices of {hist_vol.index[ix]} in 12mo")
        xformatter = mticker.FormatStrFormatter('$%.0f')
        ax.xaxis.set_major_formatter(xformatter)
        yformatter = mticker.FormatStrFormatter('%.1f%%')
        ax.yaxis.set_major_formatter(yformatter)
        plt.show()


#plt.plot(x_axis, dist.pdf(x), label='dist')
#plt.yscale('log')
#%%






#%% Predicting comodity pries
#impute missing
curve_prices_df = curve_prices_df.dropna(axis=0)
a=(curve_prices_df =='#N/A N/A')
curve_prices_df = curve_prices_df[~a.any(axis=1)]

#%%
front_cols = [i for i in curve_prices_df.columns 
             if '1 ' in i and '11' not in i and '21' not in i]
rest_cols = [i for i in curve_prices_df.columns if i not in front_cols]

X = curve_prices_df.loc[:,rest_cols]
Y = curve_prices_df.loc[:,front_cols[0]]
#%%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
pipe = Pipeline([('impute', SimpleImputer()),
                     ('scaler', StandardScaler()),
                     ('linear', LinearRegression())])

pipe.fit(X_train, y_train)

#%%
#curve_prices_df = pd.concat(curve_prices_d, axis=1)

#%%
#Predicting revenue for each of the companies, by reading in revenue from model

#import pyxlsb
program_Cos = ['Express Grain (KCO)']
fatoil_Cos = ['Western Dubuque', 'Hero', 'Kolmar', 'Sinclair', 'Mendota', 'Verbio']
energy_Cos = ['SGR Energy', 'Hiper Gas', 'Petromax']
cocoa_Cos = ['Hershey']
Cos =  [program_Cos, fatoil_Cos, energy_Cos, cocoa_Cos]

sheet_names = ['Rev Build_ Programs', 'Rev Build_ Fats&Oils', 
               'Rev Build_ Energy', 'Rev Build_ Cocoa Trading']
hist_ixs = [slice(12, 21), slice(12, 21), slice(12, 21), slice(12, 21)]

model_file_dir = 'C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio'
os.chdir(model_file_dir)
model_file_name = '20200520 02 FCStone Presidio Model WIP.xlsx'
model_bk = xlrd.open_workbook(model_file + "\\" + model_file_name)
#print(model_bk.sheet_names())    
#model_pybk = pyxlsb.open_workbook(model_file + "\\" + model_file_name)
#print(model_pybk.sheets)
for sheet_name, h_ix, sector_Cos in zip(sheet_names, hist_ixs, Cos):
    b = model_bk.sheet_by_name(sheet_name)
    co = sector_Cos[0]
    co_ix = b.col_values(0).index(co)    
    historical_rev = b.row_values(co_ix)[h_ix]
    
    try:
        gross_ix =  b.col_values(0, 
                                start_rowx = 0, 
                                end_rowx = 200).index('Gross Revenue')
        cos_ix =  b.col_values(0, 
                                start_rowx = 0, 
                                end_rowx = 200).index('Cost of Sales')
        margin = [1-j/i for i,j in zip(b.row_values(gross_ix)[h_ix],
                                     b.row_values(cos_ix)[h_ix])]#COS is line below gross revenue
    except:
        print(b.col_values(0, start_rowx = 0, end_rowx = 200), "\n\n\n\n")
        margin = None
    print(margin)
#for co in fatoil_Cos: 
#%%
    
    
    
    
    
    
    
    
    
    
    
#a = pd.read_excel("16.16 Historical Commodity Price Data.xlsx", 'Corn')
#get size before read in?
with open(prices_file) as f:
    print(f.readline())
#%%
a = pd.read_excel(prices_file, 'Corn', 
                  skiprows = 6, 
                  index_col = 0,
                  use_cols = list(range(2,sz,3)))
#indx = pd.to_datetime(a.iloc[6:,0])
#%%
curve_prices = a.iloc[6:,2::4].astype(float)
curve_prices.columns = a.columns[1::4]
curve_prices.index = indx
#%%
sec_names = a.columns[1::4].unqiue()
#a.drop(labels = [1:5], axis=1)

a.drop(columns=[2::3])
#space cols
#unified index
#%%
a.iloc[:,1]


