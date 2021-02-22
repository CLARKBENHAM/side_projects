#give research lab summary statistics on all test results
import os
import mailbox
from zipfile import ZipFile
from datetime import datetime
from collections import namedtuple
import re
from pathlib import Path
import pdb

import xlrd
import os, sys
import win32com.client
import pandas as pd
from datetime import timedelta
#%%
def Remove_password_xlsx(filename, pw_str, new_name = ""):
    xcl = win32com.client.Dispatch("Excel.Application")
    wb = xcl.Workbooks.Open(filename, False, False, None, pw_str)
    xcl.DisplayAlerts = False
    wb.SaveAs(new_name, None, '', '')
    xcl.Quit()
   
def import_ro_weekly_sheet(dir_path):
    """imports who has tested from the Research office's weekly excel sheet"""
    file_name = max([i for i in os.listdir(dir_path) if re.match("\A\d{4}-\d{2}-\d{2}",i)],
                    key = lambda i: datetime.strptime(i[:10], "%Y-%m-%d") )
    file_path = dir_path + "\\" +file_name
    pword = input("password string for excel sheet: ")
    Remove_password_xlsx(file_path, "2021", "del.xlsx")
    file_path = dir_path + "\\" + "del.xlsx"
    df = pd.read_excel(file_path, 
                       sheet_name = "Weekly Tests", 
                       header = 1, 
                       usecols = list(range(10)))
    df = df.drop(df.columns[6:9], axis=1)
    df = df.rename(columns = {df.columns[-1]: "computing_id"})
    df['date'] = df['barcode'].apply(lambda i:
                                datetime.strptime(i.split("-")[2],
                                                  "%Y%m%d%H%M"))
    df['time'] = df['date'].apply(lambda i: i.time())
    df['date'] = df['date'].apply(lambda i: i.date())
    return df 
dir_path = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab\\hide"
df = import_ro_weekly_sheet(dir_path)
#%%
def get_result_df(n = len(df)):
    """TEMP. 
    Returns a DF of all tests
    """
    result_df = df.head(n)[['barcode', 'date']]
    result_df['result'] = pd.Series(["Positive"]*(n//10) 
                                    + ['Invalid']*(n//10) 
                                    + ['Negative']*(n*7//10) 
                                    + ['Inconclusive']*(n - n*9//10))
    result_df.index = result_df['barcode']
    del result_df['barcode']
    return result_df

result_df = get_result_df()
out_df = df.merge(result_df, on="barcode", how='left').dropna(subset=["result"], axis=0)
# out_df.loc[out_df['result'].isna(), 'result'] = 'Notfound'
if len(out_df) < len(df):
    print(f"!!There are {len(df) - len(out_df)} missing tests!!")
#will use date recieved result not date took test. date_y is recieved results, date_x is took test
table_vals = out_df.loc[:,['result', 'date_y']].groupby(['result', 'date_y']).size()
table_header = ['Positive', 'Negative', 'Invalid', 'Inconclusive']
table_rows = list(sorted(out_df.date_y.unique()))
# table_rows[-1] += timedelta(1) #not gonna seperate this out since will start doing sat/sun tests

table_df = pd.DataFrame(index = table_rows, columns = table_header)
for c,r in table_vals.index:
    table_df.loc[r,c] = table_vals[c,r] 
table_df[table_df.isna()] = 0

#%%
os.remove(file_path)



