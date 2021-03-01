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


def remove_password_xlsx(file_path, pw_str):
    "wb.SaveAs doesn't work with / only \ as the seperator"
    file_path = file_path.replace("/", "\\")
    xcl = win32com.client.Dispatch("Excel.Application")
    wb = xcl.Workbooks.Open(file_path, False, False, None, pw_str)
    xcl.DisplayAlerts = False
    wb.SaveAs(file_path, None, '', '')
    xcl.Quit()

def remove_password_xlsx2(file_path, pw_str):
    file_path = file_path.replace("/", "\\")
    xcl = win32com.client.Dispatch('Excel.Application')
    wb = xcl.Workbooks.Open(file_path)
    xcl.DisplayAlerts = False
    wb.Unprotect(pw_str)
    wb.UnprotectSharing(pw_str)
    wb.Save()
    xcl.Quit()
    
def add_password_xlsx(file_path, pw_str):   
    file_path = file_path.replace("/", "\\")
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

def get_result_aggregates_df(date = None, last_week = True):
    """"returns most reccent 'result agg' file from 
    'Z:\test results aggregate counts' on pi11
    date: a datetime object, the result on a particular date
    last_week: only return rows for previous full monday-sunday time 
        (eg. Today Fri -> gets Monday 12 days ago to sun 5 days go);
            today sun -> prev mon to today
        else returns all historical aggregates
    NOTE: since file is on Linux, have to change paths to use `/` not `\`; os. ignores
    """
    i = "test results aggregate counts"
    file_re = re.compile("(Copy of ){0,1}test_results_aggregate_counts_\d{8}.xlsx")
    if date is None:
        file = max([f for f in os.listdir(f"Z:\{i}")#doesn't work wo f-string ($#@!!!!)
                    if re.match(file_re,f)],
                    key = lambda f: datetime.strptime(f[-13:-5], "%Y%m%d"))
        assert datetime.strptime(file[-13:-5], "%Y%m%d") \
                > datetime.today() - timedelta(days = 7), "Regex might have changed; file is more than 7 days old"
    else:
        date_str = datetime.strftime(date, "%Y%m%d") 
        if f"Copy of test_results_aggregate_counts_{date_str}.xlsx" in os.listdir(f"Z:\{i}"):
            file = f"Copy of test_results_aggregate_counts_{date_str}.xlsx"
        elif f"test_results_aggregate_counts_{date_str}.xlsx" in  os.listdir(f"Z:\{i}"):
            file = f"test_results_aggregate_counts_{date_str}.xlsx"
        else:
            closest = min([i for i in os.listdir(f"Z:\{i}") 
                           if re.match(file_re,i) and i > date_str],
                          key = lambda i: abs(date- datetime.strptime(i[-13:-5], 
                                                                      "%Y%m%d")))
            closest = datetime.strptime(closest[-13:-5], "%Y%m%d")
            assert False, f"no aggreate file for {date.date()}, try: {closest.date()}?"
    file_path = f"Z:\{i}\{file}"
    if not os.path.exists(f"Z:\{i}\{file}"):
        file_path = f"Z:/{i}/{file}"
    df = pd.read_excel(file_path,
                        sheet_name = "ALL_COUNTS", 
                        header = 0, 
                        usecols = list(range(6)),
                        converters = {'Date': lambda i: pd.to_datetime(i).date()},
                        dtypes = {'Positive': 'int', 'Negative': 'int', 
                                  'Inconclusive': 'int', 'Invalid': 'int',
                                  'NotTested':'int'},
                        names = ['Date', 'Positive', 'Negative', 'Inconclusive', 'Invalid', 'NotTested']
                        )
    if last_week:
        if date is None:
            wk_end = datetime.today()
        else:
            wk_end = date
        if wk_end.weekday() == 6:#sunday, inclusive
            wk_start = wk_end - timedelta(days=6)
        else:
            wk_end = wk_end - timedelta(days = 1 + wk_end.weekday())
            wk_start = wk_end - timedelta(days = 6 )
        wk_start, wk_end = wk_start.date(), wk_end.date()
        df =  df[df['Date'].between(wk_start, wk_end, inclusive=True)]
    return df

def make_table_df(r_test_df, result_df):#confirm nessisary? #GRIB
    """merge to produce table with barcode of test and result df
    r_test_df: data about researcher imported from same sheet as will modify
    results_df: df[barcode] to uid research and df[result] to place in table
        results in ['Positive', 'Negative', 'Inconclusive', 'Invalid', 'NotTested']
    ret: table_df as should be written dates by results [cols]
        includes the offset/dealing with wrapping around sunday collection
    """
    out_df = r_test_df.merge(result_df, on="barcode", how='left').dropna(subset=["result"], axis=0)
    #date_y is recieved results, date_x is took test
    if len(out_df) < len(r_test_df):
        # out_df.loc[out_df['result'].isna(), 'result'] = 'Notfound'
        print(f"!!There are {len(r_test_df) - len(out_df)} missing tests!!")
     
    table_vals = out_df.loc[:,['result', 'date_y']].groupby(['result', 'date_y']).size()
    table_vals.index = table_vals.index.set_names(['result', 'date'])
    table_header = ['Positive', 'Negative', 'Invalid', 'Inconclusive', 'NotTested']
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
    # prev_wk_wrap = lask_week_sun_wrap()
    # table_df = table_df.append(pd.Series([table_df.iloc[-1,0], *prev_wk_wrap], 
    #                                      index= table_df.columns), 
    #                            ignore_index=True)
    return table_df


def update_ro_weekly_sheet_summary(table_df, file_path, extra_writes = [], r_offset = 0):
    """"writes to summary table, uses date *took* test 
        table_df: date (7) vs results [6 columns]
            last row has a repeated date and subtracts out last weeks values
        extra_writes: [([sheet_name],CR#,value)] a list of tuples of 
                        sheet name, where to write ('Z98'), what to write
                        if sheet_name not included then uses 'Summary' tab,
        r_offset: if the table should be shifted
       'Total' column is name for 'NotTested' results#** #what is 'Total' column?** 
    """
    if ".xlsm" in file_path:
        wb = openpyxl.load_workbook(filename=file_path, keep_vba=True, read_only=False)
        #might not work
    else:
        wb = openpyxl.load_workbook(filename=file_path, keep_vba=False)
    sh = wb['Summary']
    for r in range(8,15):
        r += r_offset
        r_ix = r-8 - r_offset
        for c in range(6):
            if type(table_df.iloc[r_ix,c]).__module__ == 'numpy':
                sh[f"{chr(65+c)}{r}"] = int(table_df.iloc[r_ix,c])#can't write np datatypes
            else:
                sh[f"{chr(65+c)}{r}"] = table_df.iloc[r_ix,c]
    r = 15 + r_offset
    for c in range(1,6):
        sh[f"{chr(65+c)}{r}"] = f"={chr(65+c)}{r-2}-{chr(65+c)}{r-1}"
    r=16 + r_offset
    for c in range(1,6):
        sh[f"{chr(65+c)}{r}"] = f"=SUM({chr(65+c)}{r-8}:{chr(65+c)}{r-3})"
    #update total column?
    if len(extra_writes)>0:
        if len(extra_writes[0]) == 3:
            for sh_n, pos, val in extra_writes:
                if sh_n != 'Summary':
                    sh =wb[sh_n]
                sh[pos] = val
        elif len(extra_writes[0]) == 2:
            for pos, val in extra_writes:
                sh[pos] = val        
        else:
            print("\n\nDid not include Extra writes, wrong format \n\n")
    wb.save(file_path)

#%%
if __name__ == '__main__':
    #pull in from individuals results; which currently don't have access to
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
    result_df = get_result_df()    
    table_df = make_table_df(r_test_df, result_df)    
    update_ro_weekly_sheet_summary(table_df, file_path)
    if add_pword:  
        add_password_xlsx(file_path, pword)
#%% 
def update_sheet_wagg():
    "updates the table w/o the sun-mon wrap; + ans week specific query"
    def _numeric_query(update_sheet, info_sheet):
        """Wanted to know number of researchers in ReviewFeb15.21.xlsx
        update_sheet: the sheet to be modified
        info_sheet: sheet to take UIDs from 
        """
        i = "Research Testing"
        file_path = f"Z:\{i}\{info_sheet}"
        try:
            df = pd.read_excel(file_path,
                            header = None, 
                            usecols = list(range(3)),
                            names = ['Barcode', 'Date', 'Plate']
                            )
        except:
            pword = input("password string for query sheet: ")
            remove_password_xlsx(file_path, pword)
            df = pd.read_excel(file_path,
                            header = None, 
                            usecols = list(range(3)),
                            names = ['Barcode', 'Date', 'Plate']
                            )        
            add_password_xlsx(file_path, pword)
        uids = set(df['Barcode'].apply(lambda i: i.split("-")[1].lower()).unique())
        compare_uids = pd.read_excel(update_sheet, 
                                           sheet_name = "Approved & Pending", 
                                           header = 1, 
                                           usecols = [4])#5th
        compare_uids = set(compare_uids['uid'].unique())
        # n = len([i for i in uids if i in compare_uids])
        pdb.set_trace()
        return [("A10",len(uids.intersection(compare_uids)) )]
        
        
    #update sheet as desires for 02/22
    table_df = get_result_aggregates_df(date = datetime(year = 2021, month =2 , day = 24),last_week = True)
    #taken from previous table
    table_df.loc[1] = [datetime(year = 2021, month =2 , day = 21).date(), 0,	0,	0,	0,	0]
    table_df.sort_values("Date", inplace=True, ascending=False, axis=0)
    extra_w = [("H14", "unknown; not seperated in aggreagtes file")]
    table_df.loc[7] = [datetime(year = 2021, month =2 , day = 15).date(), 3,	305,	1,	1,	10]
    
    file_path = "Z:/Research Testing/2021-02-22 Researcher COVID TestingCB - Copy.xlsx"
    pword = input("password string for excel sheet: ")
    remove_password_xlsx(file_path, pword)
    extra_w += _numeric_query(update_sheet = file_path, info_sheet = "ReviewFeb15.21.xlsx")
    
    update_ro_weekly_sheet_summary(table_df, file_path, extra_writes = extra_w, r_offset = 6)
    #doesn't work!!?!
    try:
        add_password_xlsx(file_path, pword)
    except Exception as e:
        print(e)
        print("add password manually")


