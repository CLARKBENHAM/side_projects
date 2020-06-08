import numpy as np
import pandas as pd
import scipy.stats as ss
import xlrd
import os
import sklearn
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

#%%import Data from Com pricing spreadsheet

#com_dict = {name:pd.read_excel("16.16 Historical Commodity Price Data.xlsx", name )
#            for name in xl_bk.sheet_names()}
os.chdir("C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\Stone_Presidio\\Data")
prices_file = "16.16 Historical Commodity Price Data.xlsx"
xl_bk = xlrd.open_workbook(prices_file)
commodities = xl_bk.sheet_names()

models = {}
security_d = {}
curve_prices_d = []
for ix, name in enumerate(commodities):
    b = xl_bk.sheet_by_index(ix)
    sz = len(b.row_values(6))
    dates = [datetime(*xlrd.xldate_as_tuple(i,0)) if type(i) == float
             else None
                for i in b.col_values(0)[7:]]
    com_maturities = b.row_values(0)[1::4]
    com_ab = commodities[ix]
    curve_prices = pd.DataFrame(list(zip(*[b.col_values(i)[7:]
                                            for i in range(2,sz,4)]
                                            )),
                                 columns = com_maturities,
                                 index = dates,
                                 dtype = float)
    curve_prices = curve_prices.dropna()
    curve_prices_d += [curve_prices]
    
#    sec_df = pd.DataFrame(list(zip(*[b.col_values(i)[7:]
#                                            for i in range(1,sz,4)]
#                                            )) )
#    securities = np.unique(sec_df.to_numpy().flatten())
#    securities = securities[~np.isin(securities, ('', '#N/A N/A'))]
#                                                  
#    def get_security(s):
#        "returns prices for 1 single future"
##        vals = curve_prices.to_numpy()[sec_df == s]
##        tf = ~pd.isnull(vals)
##        return vals[~pd.isnull(vals)].values
#        return pd.Series(data = curve_prices.to_numpy()[sec_df == s], 
#                  index = curve_prices.index[(sec_df == s).any(axis=1)],
#                  name = s
#                  )
#    sec_df = [get_security(s) for s in securities]
##    sec_df = {s+ " " + com_ab: get_security(s)
##                for s in securities}
#    security_d.update(sec_df)
#    break
long_ix = max([(i, df.shape[0]) for i, df in enumerate(curve_prices_d)], 
               key = lambda i: i[1])[0]
df_idx = curve_prices_d.pop(long_ix)
curve_prices_df = df_idx.join(curve_prices_d)
curve_prices_d += [df_idx]
#securities_df = pd.concat(security_d)
#%% Make Price Graphs from future's

#%%
import matplotlib.pyplot as plt

#wheat plots, W, KW, MW
#Corn, soybean, soymeal: C, S, SM
#Canola, BeanOil: RS, BO
#NY Lon Cocoa, coffee: CC, QC, KC
#Crude, heating: CL, HO



#Ethanol/RINS/Biodiesel

#No: Platinum, Palladium, Milk: PA, PL, NFDM

#earliest expiration date in month, an approximation
#%%
#[re.findall("([a-zA-Z]+).*(\d+)",i)[0] for i in futures]

#ix = 8
#name = 'Crude Oil'



#%%

#%%
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
#Predicting revenue for each of the companies

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
                  use_cols = list(range(2,,3)))
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


