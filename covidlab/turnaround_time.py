#creating a plot average turnaround time by date
import os
import mailbox
from zipfile import ZipFile
from datetime import datetime, timedelta
from collections import namedtuple
import re
from pathlib import Path
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import date2num
import matplotlib.gridspec as gridspec
#%%
#response datetime; taking others from 'RAW FILE POST-PROCESS COUNTS'
plate_results = namedtuple("emailed_plate_results", ['Datetime',
                                                    'File',#day result name
                                                    'Positive',
                                                     'Negative',
                                                     'Invalid',
                                                     'Inconclusive',
                                                     'Notfound',
                                                     'Duplicate',
                                                     'Preliminary'])

def get_email_body(message, decode=False):
    """attachments can contain sub attachments;
    decode: if true returns bytes, else str
    """
    if message.is_multipart():
        content = ''.join(get_email_body(part) for part in message.get_payload())
    else:
        content = message.get_payload(decode=decode)
    return content

def proc_gmail_export(email_dir):
    "given directory to gmail exported archive of emails, "
    #Data is in the email body itself, not the attachments
    #max date takeout is taken
    if email_dir[-4:] == ".zip":    
        zf = ZipFile(f'{base_dir}/{email_dir}', 'r')
        zf.extractall(f"{base_dir}/{email_dir[:-4]}")
        zf.close()
        os.remove(f'{base_dir}/{email_dir}')
        email_dir = email_dir[:-4]
    mbox = mailbox.mbox(f"{base_dir}\{email_dir}\Takeout\Mail\covid_response_results.mbox")
    
    file_re = re.compile("([a-zA-Z0-9\_]+\.xlsx) COVID test results counts")
    result_re = re.compile('Positive = (\d+)\n'\
                            'Negative = (\d+)\n'\
                            'Invalid = (\d+)\n'\
                            'Inconclusive = (\d+)\n'\
                            'Notfound = (\d+)\n'\
                            'Duplicate = (\d+)\n'\
                            'Preliminary = (\d+)\n')
    #that result_re has fewer matches than file_re is correct, see: [c for c in content if len(re.findall(result_re, c)) ^ len(re.findall(file_re, c))]
    emails = []
    for message in mbox:
        content = get_email_body(message)
        try:
            dt = datetime.strptime(" ".join(message['date'].split(" ")[:5]),
                                   "%a, %d %b %Y %X")
            file = re.findall(file_re, content)[0]
            values = re.findall(result_re, content)[0]
            email = plate_results._make([dt, file, *map(int, values)])
            emails += [email]
        except Exception as e:
            #non plate result emails 
            pass
    return emails

base_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab"
email_dir = max([i for i in os.listdir(base_dir) if 'takeout' in i],
                    key = lambda p: int(p[8:16]))
response_emails = proc_gmail_export(email_dir)
email_responses = {i.File: i.Datetime for i in response_emails}
#%%
def get_plates(email_responses):
    """
    gets tests corresponding to 
    email_responses {file: Datetime recieved}
    ret pd.Df
    """
    # earliest = min([i.Datetime for i in response_emails]).date()
    # #date completed: average time for test to finish
    # tested_dates = {earliest + timedelta(i):[]
    #                 for i in range((datetime.today().date() - earliest).days + 99)}
    # result_files = []
    folder_path = "Z:\ResearchData\BeSafeSaliva\Reported_File"
    month_folders = os.listdir(folder_path)
    every_plate = []
    
    for month in month_folders:
        response_files = os.listdir(f"{folder_path}\{month}")
        # result_files += [f"{folder_path}\{f}" for f in response_files]
        for file in response_files:
            if file in email_responses:
                finished_dt = email_responses[file]
                plate = pd.read_excel(f"{folder_path}\{month}\{file}",
                                       sheet_name = "Weekly Tests", 
                                       header = 1, 
                                       usecols = list(range(10))
                                       )
                plate['start_dt'] = plate['barcode'].apply(lambda i:
                                              datetime.strptime(i.split("-")[2],
                                                                "%Y%m%d%H%M"))
                #not unique so can't make index
                plate['time'] = plate['start_dt'].apply(lambda i: i.time())
                plate['date'] = plate['start_dt'].apply(lambda i: i.date())
                plate['duration'] = plate['start_dt'].apply(lambda i: finished_dt - i)
                plate = pd.concat({file: plate}, names=["file"])
                # tested_dates[finished_dt] += [plate['duration'].mean()]
                # plate.drop('start_dt', inplace=True)
                every_plate += [plate]
                
    plates_df = pd.concat(every_plate)
    return plates_df

plates_df = get_plates(email_responses)

#**is** what you want: what if Nov 3rd all plates are slow and spread out over the course of following week?
#next couple days get increased 
#%%
def _add_labels(ax, r_lst):
    "to barplots"
    # def _autolabel(rects):
    for rects in r_lst:
        # """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height >sum(ax.get_ylim())/2:
                ax.annotate(f"{height:.0f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -height/8),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom', 
                            color='w')
            else:
                ax.annotate(f"{height:.0f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')   
                    
def make_plots(grp):
    """Takes a groupby object on the date as index with timedelta values"""
    fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize = (20,12), constrained_layout=True)
    # gs1 = gridspec.GridSpec(3, 1)
    # gs1.update(hspace=0, wspace=0)
    
    ax.set_title("Daily Average Time to complete test")
    daily_avg = grp.apply(lambda g: g.astype(np.int64).mean())
    daily_std = grp.apply(lambda g: g.astype(np.int64).mean())
    ax.scatter(daily_avg.index, daily_avg.values)
    ax.plot(daily_avg.index, daily_avg.values,  linewidth=3, label = "mean")
    ax.plot(daily_avg.index, daily_avg.values -2*daily_std, 'r--', label = "lower 2SD")
    ax.plot(daily_avg.index, daily_avg.values + 2*daily_std, 'r--', label = "upper 2SD")
    ax.legend()
    ax.set_ylabel("time till completion")
    ax.set_xlabel("Test Collection Date")
    
    ax2.set_title("Percent of samples delayed by")
    daily_12hr =  grp.apply(lambda g: sum(g > timedelta(hours=12))/len(g))*100
    daily_24hr = grp.apply(lambda g: sum(g > timedelta(hours=24))/len(g))*100
    daily_36hr = grp.apply(lambda g: sum(g > timedelta(hours=36))/len(g))*100
    daily_48hr = grp.apply(lambda g: sum(g > timedelta(hours=48))/len(g))*100
    daily_72hr = grp.apply(lambda g: sum(g > timedelta(hours=72))/len(g))*100
    #groupby casts to object
    ix = date2num([n for n,_ in grp])
    width = (ix[1] - ix[0])/6
    rects1 = ax2.bar(ix - width*2, daily_12hr.values, width, label='% >12 hours')
    rects2 = ax2.bar(ix - width, daily_24hr.values, width, label='% >24 hours')
    rects3 = ax2.bar(ix, daily_36hr.values, width, label='% >36 hours')
    rects4 = ax2.bar(ix + width, daily_48hr.values, width, label='% >48 hours')
    rects5 = ax2.bar(ix + width*2, daily_72hr.values, width, label='% >72 hours')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_ylabel('Percent')
    ax2.set_xticks(ix)
    ax2.set_xticklabels([n for n,_ in grp])
    ax2.legend()
                 
    _add_labels(ax2, [rects1, rects2, rects3, rects4, rects5])
    #percentiles
    ax3.set_title("Percentile time to completetion")
    per_30 = grp.apply(lambda g: np.quantile(g, 0.30))
    per_80 =grp.apply(lambda g: np.quantile(g, 0.80))
    per_95 =grp.apply(lambda g: np.quantile(g, 0.95))
    per_99 =grp.apply(lambda g: np.quantile(g, 0.99))
    ax3.plot(per_30, label = "30th percentile")
    ax3.plot(per_80, label = "80th percentile")
    ax3.plot(per_95, label = "95th percentile")
    ax3.plot(per_99, label = "99th percentile")
    ax3.legend()
    ax3.set_ylabel("Days")
    # fig.tight_layout()
    return fig
    # plt.show()

def weekly_plot(df, wk_end = None, mon_sun= False):
    """
    plot values for 7 days preceding wk_end
        df: df['date'] is datetime.date object
        mon_sun: if monday-sunday is true will cast to the monday-sunday interval 
            that is inclusive of wk_end
    """
    if wk_end is None:
        wk_end = datetime.today().date()
    if mon_sun and wk_end.weekday() != 0:
        wk_end = wk_end + timedelta(days = 7 - wk_end.weekday())
        wk_start = wk_end - timedelta(days =wk_end.weekday())
    else:
        wk_start = wk_end - timedelta(days=7)            
    wk_start, wk_end = wk_start.date(), wk_end.date()
    grp =  df[df['date'].between(wk_start, wk_end, inclusive=True)].groupby("date")['duration']
    fig = make_plots(grp)
    fig.suptitle("Plots for week of f{wk_start}-{wk_end}")
    fig.show()

def daily_plot(df):
    "group by portion of day when submitted test"
    pass
# r_test_df['datetime'] = r_test_df.apply(lambda r: datetime.combine(r['date'], r['time']), axis=1)
r_test_df['duration'] = r_test_df['datetime'] - min(r_test_df['datetime'])
weekly_plot(r_test_df, wk_end = datetime(year=2021, month = 2, day=15))
# fig.show()
#%%bad order
for f,res_dt in email_responses:
    #filter out previously seen
    month_folder = [i for i in month_folders if str(res_dt.month).upper() in i]
    response_files = os.listdir("{folder_path}\{month_folder}")            
            
            
            
 #%%
class gmail_archive_extra_helpers: 
    def unmatched_plates(email_responses):
        """set of files that aren't in the responses"""
        folder_path = "Z:\ResearchData\BeSafeSaliva\Reported_File"
        month_folders = os.listdir(folder_path)
        every_result = []
        for month in month_folders:
            response_files = os.listdir(f"{folder_path}\{month}")
            every_result += response_files
        return set(every_result) - set(email_responses.keys())
    
    #currently unused
    def plate_response(mbox):
        """"returns {day_result_string: Datetime,...}
        a tuple of plate ID found from excel sheet name and the response's datetime.
        Note: many emails from this address aren't about a particular plate
            eg: general cordination emails, aggregate results for the whole day, error notifications
        returns (-1,-1) on invalid
        **ALWAYS RETURNS (-1,-1) as ALL Date is in body iteself; the attached email files are general summaries of the day
        """
        plate_response_dt = {}
        for message in mbox:
            if message.get_content_maintype() == 'multipart':
                for part in message.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition') is None:
                        continue
                    filename = part.get_filename()
                    print(filename)
                    if filename  and  ".xlsx" in filename: 
                        #need to fix below
                        if "test_results_aggregate_counts" not in filename:
                            dt = datetime.strptime(" ".join(message['date'].split(" ")[:5]),
                                   "%a, %d %b %Y %X")
                            plate_id = re.findall(file_re, filename)[0]
                            plate_response_dt[plate_id] = dt
                        else:
                            break
        return plate_response_dt
    #bunch of bad emails don't get picked up in file regex in *body*. only None file names?
    
    def _no_attachments(message):
        "detects if email contains any attachments; *some* of the emails not containing results"
        if message.get_content_maintype() == 'multipart':
            for part in message.walk():
                if part.get_content_maintype() == 'multipart':
                    continue
                if part.get('Content-Disposition') is None:
                    continue
                filename = part.get_filename()
                if filename:
                    return False
        return True
    
    def _extract_attachements(mbox, target_path):
        """get data in excel file attachments that wasn't included in .txt
        no .txt files exist without matching .xlsx file;
            .txt files weren't included until Dec 24th 2020
        """
        #target_path = f'{base_dir}\{email_dir}'
        os.chdir(target_path)
        if "temp" not in target_path and "temp" not in os.listdir():
            os.mkdir("temp")
            os.chdir("temp")
        for message in mbox:
            if message.get_content_maintype() == 'multipart':
                for part in message.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition') is None:
                        continue
                    filename = part.get_filename()
                    if filename and ".png" not in filename:
                        fb = open(filename,'wb')
                        fb.write(part.get_payload(decode=True))
                        fb.close()
                        
    def compare_xlsx2txt():
        "in the dir  _extract_attachements exports to"
        txt =[]
        xlsx = []
        for f in os.listdir():
            if ".xlsx" in f:
                xlsx += [f[-13:-5]]
            elif ".txt" in f:
                txt += [f[-18:-10]]
        len(txt), len(xlsx)
        set(txt) - set(xlsx)

                        
# r_test_df['dateime'] = r_test_df.apply(lambda r: datetime.combine(r['date'], r['time']), axis=1)              
                        
                        