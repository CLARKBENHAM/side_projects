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
def remove_password_xlsx(file_path, pw_str, new_name):
    xcl = win32com.client.Dispatch("Excel.Application")
    wb = xcl.Workbooks.Open(file_path, False, False, None, pw_str)
    xcl.DisplayAlerts = False
    wb.SaveAs(new_name, None, '', '')
    xcl.Quit()
    
def add_password_xlsx(file_path, pw_str, new_name):
    xcl = win32com.client.Dispatch("Excel.Application")
    wb = xcl.Workbooks.Open(file_path, False, False, None, pw_str)
    xcl.DisplayAlerts = True
    wb.SaveAs(new_name, None, '', '')
    xcl.Quit()
    
def import_ro_weekly_sheet(file_path, ret_filtered = True):
    """imports who has tested from the Research office's weekly excel sheet
    ret_filtered: if True only returns those rows which correspond to people in summary table
            approved studies less Cheryl Wagner, anybody supervised by Cheryl Wagner, people in not found status, and people in pending status, and anybody who reports to Dave Hudson.
    """
    # pword = input("password string for excel sheet: ")
    # remove_password_xlsx(file_path, pword, "del.xlsx")
    # file_path = dir_path + "\\" + "del.xlsx"
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
    if ret_filtered:
        pass
    return df 

dir_path = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab\\hide"
file_name = max([i for i in os.listdir(dir_path) if re.match("\A\d{4}-\d{2}-\d{2}",i)],
                    key = lambda i: datetime.strptime(i[:10], "%Y-%m-%d") )
file_path = dir_path + "\\" +file_name
df = import_ro_weekly_sheet(file_path)

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
#date_y is recieved results, date_x is took test
if len(out_df) < len(df):
    # out_df.loc[out_df['result'].isna(), 'result'] = 'Notfound'
    print(f"!!There are {len(df) - len(out_df)} missing tests!!")
            
def update_ro_weekly_sheet_summary(out_df, file_path):
    """"given a df with results for a set of researcher tests; writes to summary table
        will use date recieved result not date took test. 
        saves as a pword protected file
    """
    #%%
table_vals = out_df.loc[:,['result', 'date_y']].groupby(['result', 'date_y']).size()
table_header = ['Positive', 'Negative', 'Invalid', 'Inconclusive']
table_rows = list(sorted(out_df.date_y.unique()))

table_df = pd.DataFrame(index = table_rows, columns = table_header)
for c,r in table_vals.index:
    table_df.loc[r,c] = table_vals[c,r] 
table_df[table_df.isna()] = 0
table_df.sort_index(ascending=False, inplace=True)

#wrap tests (plates collected on sunday are included on the following monday)
#how to get break down? 
#what is 'Total' column?
table_rows[-1] += timedelta(1)

#append for last week's wrap
table_df.index.name = "Date"
table_df.reset_index(inplace=True)
prev_wk_wrap = [5,6,7,8]
table_df.append([table_rows[0], *prev_wk_wrap] )

#%%
wb = xlrd.open_workbook(file_path)
sh = wb.sheet_by_name("Summary")

#%%
if __name__ == '__main__':
    dir_path = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab\\hide"
    file_name = max([i for i in os.listdir(dir_path) if re.match("\A\d{4}-\d{2}-\d{2}",i)],
                        key = lambda i: datetime.strptime(i[:10], "%Y-%m-%d") )
    file_path = dir_path + "\\" +file_name
    df = import_ro_weekly_sheet(file_path)
    
    result_df = get_result_df()    
    out_df = df.merge(result_df, on="barcode", how='left').dropna(subset=["result"], axis=0)
    if len(out_df) < len(df):
        # out_df.loc[out_df['result'].isna(), 'result'] = 'Notfound'
        print(f"!!There are {len(df) - len(out_df)} missing tests!!")
        
    update_ro_weekly_sheet_summary(out_df, file_path)
    # add_password_xlsx(file_path, pw_str, new_name)
    os.remove(file_path)



