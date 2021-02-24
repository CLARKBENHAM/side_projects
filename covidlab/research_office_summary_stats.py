#give research lab summary statistics on all test results
import os
import mailbox
from zipfile import ZipFile
from datetime import datetime
from collections import namedtuple
import re
from pathlib import Path
import pdb

import os, sys
import win32com.client
import pandas as pd
from datetime import timedelta

import openpyxl

from win32com.client.gencache import EnsureDispatch
#%%
def remove_password_xlsx(file_path, pw_str):
    xcl = win32com.client.Dispatch("Excel.Application")
    wb = xcl.Workbooks.Open(file_path, False, False, None, pw_str)
    xcl.DisplayAlerts = False
    wb.SaveAs(file_path, None, '', '')
    xcl.Quit()
    
def add_password_xlsx(file_path, pw_str):   
    xlApp = EnsureDispatch("Excel.Application")
    xlwb = xlApp.Workbooks.Open(file_path)
    xlApp.DisplayAlerts = False
    xlwb.Visible = False
    xlwb.SaveAs(file_path, Password = pw_str)
    xlwb.Close()
    xlApp.Quit()
    
def import_ro_weekly_sheet(file_path, ret_filtered = True):
    """imports who has tested from the Research office's weekly excel sheet
    ret_filtered: if True only returns those rows which correspond to people in summary table
            approved studies less Cheryl Wagner, anybody supervised by Cheryl Wagner, people in not found status, and people in pending status, and anybody who reports to Dave Hudson.
    """
    r_test_df = pd.read_excel(file_path, 
                       sheet_name = "Weekly Tests", 
                       header = 1, 
                       usecols = list(range(10))
                       )#this is *all* weekly tests
    r_test_df = r_test_df.drop(r_test_df.columns[6:9], axis=1)
    r_test_df = r_test_df.rename(columns = {r_test_df.columns[-1]: "computing_id"})
    r_test_df['date'] = r_test_df['barcode'].apply(lambda i:
                                datetime.strptime(i.split("-")[2],
                                                  "%Y%m%d%H%M"))
    r_test_df['time'] = r_test_df['date'].apply(lambda i: i.time())
    r_test_df['date'] = r_test_df['date'].apply(lambda i: i.date())
    
    test_check_df = pd.read_excel(file_path, 
                                  sheet_name = "Weekly Test Check", 
                                  header = 0, 
                                  usecols = list(range(4))
                                  ).drop(0)
    not_found_status = set(test_check_df[test_check_df['Status'] == "not found"]['Computing ID'])
    pending_status = set(test_check_df[test_check_df['Status'] == "PENDING"]['Computing ID'])
    no_test_list =  set(pd.read_excel(file_path, 
                                  sheet_name = "Weekly No Test List", 
                                  header = 0, 
                                  usecols = list(range(1))
                                  ))
    def _include(row, not_found = not_found_status, pending = pending_status):
        """Ret True if should be included in summary stats:
            approved studies less:
                Cheryl Wagner, 
                anybody supervised by Cheryl Wagner,
                people in not found status,
                people in pending status,
                and anybody who reports to Dave Hudson.
        """
        if row['status'] != "APPROVED":
            return False
        if row['name'].lower() == 'Cheryl Wagner' or row['email'] in ("cheryl-vpr@virginia.edu", "cdr9c@virginia.edu"):
            return False
        if row['Supervisor Name or Computing ID'].lower() in ("cdr9c", 'cheryl d wagner', 'cheryl wagner'):
            return False
        if row['uid'] in not_found:
            return False
        if row['uid'] in pending:
            return False
        if row['Supervisor Name or Computing ID'].lower() in ("david hudson", "djh2t", 'david j hudson'):
            return False
        assert row['uid'] not in no_test_list, f"{row['uid']} on No Test List but passed all checks"
        return True
        
    approved_pending_df = pd.read_excel(file_path, 
                              sheet_name = "Approved & Pending", 
                              header = 1, 
                              usecols = list(range(13)))
    valid_ids = approved_pending_df['uid'][approved_pending_df.apply(lambda r: _include(r), axis=1)]
    valid_ids = valid_ids.apply(lambda i: i.upper())
    r_test_df = r_test_df[r_test_df['computing_id'].isin(valid_ids)]
    return r_test_df
    
def get_result_df(n = 1600):
    """TEMP. 
    Returns a DF of all tests
    """
    result_df = r_test_df.head(n)[['barcode', 'date']].reset_index()
    
    result_df['result'] = pd.Series(["Positive"]*(n//10) 
                                    + ['Invalid']*(n//10) 
                                    + ['Negative']*(n*7//10) 
                                    + ['Inconclusive']*(n//10))
    result_df.loc[result_df['result'].isna(), 'result'] = 'Inconclusive'
    result_df.index = result_df['barcode']
    del result_df['barcode']
    return result_df

def lask_week_sun_wrap():
    "the plate that was collected a week ago, but tested this weeks monday. Will be subtracted from this weeks total"           
    return [5,600,7,8]
    
def next_week_sun_wrap():
    "the plate that was collected this, but tested next this weeks monday. Will be add to this weeks total"           
    return [5,600,7,8]

def update_ro_weekly_sheet_summary(r_test_df, file_path):
    """"writes to summary table
        r_test_df: results for a set of researcher tests 
        uses date *took* test 
        
       ** #what is 'Total' column?** 
    """
    result_df = get_result_df()    
    out_df = r_test_df.merge(result_df, on="barcode", how='left').dropna(subset=["result"], axis=0)
    #date_y is recieved results, date_x is took test
    if len(out_df) < len(r_test_df):
        # out_df.loc[out_df['result'].isna(), 'result'] = 'Notfound'
        print(f"!!There are {len(r_test_df) - len(out_df)} missing tests!!")
     
    table_vals = out_df.loc[:,['result', 'date_y']].groupby(['result', 'date_y']).size()
    table_header = ['Positive', 'Negative', 'Invalid', 'Inconclusive']
    table_rows = list(sorted(out_df.date_y.unique()))
    
    table_df = pd.DataFrame(index = table_rows, columns = table_header)
    for c,r in table_vals.index:
        table_df.loc[r,c] = table_vals[c,r] 
    table_df[table_df.isna()] = 0
    table_df.sort_index(ascending=False, inplace=True)
    
    #append for last week's wrap
    table_df.index.name = "Date"
    table_df.reset_index(inplace=True)
    table_df.loc[0,'Date'] += timedelta(1) #since tests collected on sunday are marked as monday
    # next_wk_wrap = next_week_sun_wrap() #not needed
    # table_df = table_df.append(pd.Series([table_df['Date'][0], *next_wk_wrap],
    #                                       index= table_df.columns), 
    #                             ignore_index=True)
    # table_df.sort_values("Date", inplace=True, ascending=False, axis=0)
    prev_wk_wrap = lask_week_sun_wrap()
    table_df = table_df.append(pd.Series([table_df.iloc[-1,0], *prev_wk_wrap], 
                                         index= table_df.columns), 
                               ignore_index=True)
    if ".xlsm" in file_path:
        wb = openpyxl.load_workbook(filename=file_path, keep_vba=True, read_only=False)
        #might not work
    else:
        wb = openpyxl.load_workbook(filename=file_path, keep_vba=False)
    sh = wb['Summary']
    for r in range(8,15):
        for c in range(5):
            if type(table_df.iloc[r-8,c]).__module__ == 'numpy':
                sh[f"{chr(65+c)}{r}"] = int(table_df.iloc[r-8,c])
            else:
                sh[f"{chr(65+c)}{r}"] = table_df.iloc[r-8,c]
    r = 15
    for c in range(1,5):
        sh[f"{chr(65+c)}{r}"] = f"={chr(65+c)}{13}-{chr(65+c)}{14}"
    r=16
    for c in range(1,5):
        sh[f"{chr(65+c)}{r}"] = f"=SUM({chr(65+c)}8:{chr(65+c)}13)"
    #update total column?
    wb.save(file_path)

#%%
if __name__ == '__main__':
    #Manage pword stuff here
    dir_path = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab\\hide"
    file_name = max([i for i in os.listdir(dir_path) if re.match("\A\d{4}-\d{2}-\d{2}",i)],
                        key = lambda i: datetime.strptime(i[:10], "%Y-%m-%d"))
    file_path = dir_path + "\\" +file_name
    add_pword = False
    try:
        r_test_df = import_ro_weekly_sheet(file_path)
    except:
        pword = input("password string for excel sheet: ")
        remove_password_xlsx(file_path, pword)
        add_pword = True
        r_test_df = import_ro_weekly_sheet(file_path)
    update_ro_weekly_sheet_summary(r_test_df, file_path)
    if add_pword:  
        add_password_xlsx(file_path, pword)
#%%


