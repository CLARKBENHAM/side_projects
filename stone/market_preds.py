import numpy as np
import pandas as pd
import scipy.stats as ss
import xlrd
import os
import sklearn
#os.system("pip install xlrd")
os.chdir("Desktop\Stone_presidio\Data\FFIEC CDR Call Bulk All Schedules 03312020")
#%%
names =[]
def find_ffiec(col_name):
    for f_name in os.listdir():
        f = open(f_name, 'r')
        if col_name in f.readline():
            data = pd.read_csv(f_name, 
                               delimiter='\t',
                               skiprows=0,
                               na_values=' '
                               )
            data.fillna(0, inplace=True)
            names.append(data[col_name][0])
            return( data[col_name].drop(labels=0).astype(int))
            
data_series = ("RCON1590",
                "RCON3386",
                "RCON5577",
                "RCON5584",
                "RCON5585",
                "RCON5586",
                "RCON5587",
                "RCON5588",
                "RCON5589",
                )
#data = zip(*[find_ffiec(i) for i in data_series])
data_dict = {i:find_ffiec(i) for i in data_series}
data_names = {i:j for i,j in zip(data_series, names)}
    #%%
#all domestic
#5584 subset of 5578?
tot_num = data_dict['RCON5577'].sum()#Wrong?
num_less100 = data_dict['RCON5584'].sum()
num_less250 = data_dict['RCON5586'].sum()
num_less500 = data_dict['RCON5588'].sum()
#num_more500 = tot_num - (num_less100 + num_less250 + num_less500)
#%%
tot_ag = data_dict['RCON1590'].sum()

amnt_less100 = data_dict['RCON5585'].sum()
amnt_less250 = data_dict['RCON5587'].sum()
amnt_less500 = data_dict['RCON5589'].sum()
amnt_more500 = tot_ag - amnt_less100 - amnt_less250 - amnt_less500
print(amnt_more500 )
[i/tot_ag for i in (amnt_less100 , amnt_less250, amnt_less500)]
#all 6%? did/didn't include residential improvements
#%%









