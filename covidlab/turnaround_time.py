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
    gets tests corresponding to email_responses from the health directory
        ??how to get health dir? (only on desktop? But got perm. to have on others?)??
    email_responses {file: Datetime recieved}
    ret pd.Df [date, time] as when results Delievered
        finished_dt = duration + start_dt
    """
    # earliest = min([i.Datetime for i in response_emails]).date()
    # #date completed: average time for test to finish
    # tested_dates = {earliest + timedelta(i):[]
    #                 for i in range((datetime.today().date() - earliest).days + 99)}
    # result_files = []
    folder_path = "Z:\ResearchData\BeSafeSaliva\Reported_File"
    try:
        month_folders = os.listdir(folder_path)
    except:
        print("WARING: SIMULATED DatA IS PROVIDED Based on r_test_df")# from import_ro_weekly_sheet(file_path)
        r_test_df['start_dt'] = r_test_df.apply(lambda r: 
                                                datetime.combine(r['date'], r['time']),
                                                axis=1)
        # r_test_df['time'] = r_test_df['start_dt'].apply(lambda i: i.time())
        # r_test_df['date'] = r_test_df['start_dt'].apply(lambda i: i.date())
        r_test_df['duration'] = max(r_test_df['start_dt']) + timedelta(hours=4) - r_test_df['start_dt']
        plate1_sz = len(r_test_df)//2
        r_test_df.iloc[:plate1_sz, df.columns.get_loc('duration')] += timedelta(hours=7)
        r_test_df['plate'] = ["TEMP1"]*plate1_sz + ["TEMP2"]*len(r_test_df[plate1_sz:])
        return r_test_df
    
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
                # plate = pd.concat({file: plate}, names=["file"])#makes multiindex (don't want to deal)
                plate['plate'] = file
                # tested_dates[finished_dt] += [plate['duration'].mean()]
                every_plate += [plate]
                
    plates_df = pd.concat(every_plate)
    return plates_df

plates_df = get_plates(email_responses)
#**is** what you want: what if Nov 3rd all plates are slow and spread out over the course of following week?
#next couple days get increased 
#%%
def _add_labels(ax, r_lst):
    """to barplots 
    ax: axis
    r_lst: [ax.bar() object]
    """
    # def _autolabel(rects):
    for rects in r_lst:
        # """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height >sum(ax.get_ylim())/2:
                t = ax.annotate(f"{height:.0f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -height/8),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom', 
                            size = 10,
                            color='w')
            else:
                t = ax.annotate(f"{height:.0f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom',
                            size = 10)
            #to not write if bars are too skinny
            txt_w = t.get_window_extent(plt.gcf().canvas.get_renderer()).width
            bar_w = rect.get_window_extent().width
            if txt_w > bar_w:
                if bar_w > txt_w/2:
                    t.set_size(t.get_size() * bar_w/txt_w)
                else:
                    t.remove()
                    
def make_plots(grp):
    """
    Takes a groupby object with timedelta values and plots those against indicies
    #format tick labels? eg. for datetime to ignore year for plates 
    """
    fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize = (20,12), constrained_layout=True)
    # gs1 = gridspec.GridSpec(3, 1)
    # gs1.update(hspace=0, wspace=0)
    
    ax.set_title("Daily Average Time to complete test")
    daily_avg = grp.apply(lambda g: g.astype(np.int64).mean())
    daily_std = grp.apply(lambda g: g.astype(np.int64).mean())
    ax.scatter(daily_avg.index, daily_avg.values)
    ax.plot(daily_avg.index, daily_avg.values,  linewidth=3, label = "mean")
    ax.plot(daily_avg.index, 
            [max(i,0) for i in daily_avg.values -2*daily_std],
            'r--',
            label = "lower 2SD")
    ax.plot(daily_avg.index, daily_avg.values + 2*daily_std, 'r--', label = "upper 2SD")
    ax.legend()
    ax.set_ylabel("time till completion")
    ax.set_xlabel("Test Collection Date")
    ax.set_ylim(max(0, ax.get_ylim()[0]))

    ax2.set_title("Percent of samples delayed by")
    daily_12hr =  grp.apply(lambda g: sum(g > timedelta(hours=12))/len(g))*100
    daily_24hr = grp.apply(lambda g: sum(g > timedelta(hours=24))/len(g))*100
    daily_36hr = grp.apply(lambda g: sum(g > timedelta(hours=36))/len(g))*100
    daily_48hr = grp.apply(lambda g: sum(g > timedelta(hours=48))/len(g))*100
    daily_72hr = grp.apply(lambda g: sum(g > timedelta(hours=72))/len(g))*100
    #groupby casts to object
    try:
        ix = date2num([n for n,_ in grp])
    except:
        ix = np.arange(len(grp))#groupedby on non-dates
    try:#might only have 1 group
        width = (ix[1] - ix[0])/8
    except:
        width = ix/8
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
    plt.scatter(per_30.index.astype('str'), per_30 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "30th percentile")
    plt.scatter(per_80.index.astype('str'), per_80 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "80th percentile")
    plt.scatter(per_95.index.astype('str'), per_95 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "95th percentile")
    plt.scatter(per_99.index.astype('str'), per_99 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "99th percentile")
    ax3.legend()
    ax3.set_ylabel("Days")
    ax3.set_ylim(max(0, ax3.get_ylim()[0]))
    # fig.tight_layout()
    return fig

def _not_groupby(a):
    assert not isinstance(a, pd.core.groupby.generic.SeriesGroupBy), \
        "Given Series Groupby instead of df"
    assert not isinstance(a, pd.core.groupby.generic.DataFrameGroupBy),\
        "Given df.Groupby instead of df"
    
def weekly_plot(df, wk_end = None, plot_result_dates = False):
    """
    plot values for  monday-sunday interval that is inclusive of wk_end
        df: df['date'] is datetime object
        if wk_end is none will plot for values thus far this 
        plot_result_dates: plots based on date results found instead of collection date 
    """
    _not_groupby(df)
    if wk_end is None:
        wk_end = datetime.today()
    if not isinstance(wk_end, datetime):#cast date to datetime
        wk_end = datetime.combine(wk_end, datetime.min.time())
    wk_end = wk_end + timedelta(days = 6 - wk_end.weekday())
    wk_start = wk_end - timedelta(days =wk_end.weekday())
    if plot_result_dates:
        # pdb.set_trace()
        finished = df[df['start_dt'].between(wk_start, wk_end, inclusive=True)]
        grp =  finished.groupby(finished.apply(lambda r: r['start_dt'] + r['duration'],
                                               axis=1)
                                )['duration']        
    else:#collection date
        wk_start, wk_end = wk_start.date(), wk_end.date()
        grp =  df[df['date'].between(wk_start, wk_end, inclusive=True)
                  ].groupby("date")['duration']        
    fig = make_plots(grp)
    if plot_result_dates:
        fig.suptitle(f"Plots for week of {wk_start}-{wk_end}", size="xx-large")
        fig.get_axes()[0].set_xlabel("Test Results Date")
    else:
        fig.suptitle(f"Plots for week of {wk_start}-{wk_end})", size="xx-large")
    fig.show()

def trailing_plot(df, end_day = datetime(year=2021, month = 2, day=14).date(), n_trailing = 0):
    """"group by day when submitted test
        n_trailing: the number of days (inclusive) that occured before 
            'end_day' to be included
    """
    _not_groupby(df)
    if isinstance(end_day, datetime):
        end_day = end_day.date()
    start_day = end_day - timedelta(days = n_trailing)
    grp =  df[df['date'].between(start_day, end_day, inclusive=True)]
    grp = grp.groupby("date")['duration']
    fig = make_plots(grp)
    fig.suptitle(f"Plots for {n_trailing} Days preceding {end_day}", size="x-large")
    fig.show()
    
def time_of_day(df, day =  None, n_trailing = 0):
    """"segments by ~delivery batch
    # email_responses: {file_name: recievd_dt} for all files in df['plate']
    makes plot with '11/19/20 18:13' type format
   """
    _not_groupby(df)
    if day is None:
        day = datetime.today().date()
    elif isinstance(day, datetime):#cast datetime to date
        day = day.date()
    df = df[df['date'].between(day, day, inclusive=True)]
    #group by batch; but cast to the datetime recieved
    grp = df.groupby(df.apply(lambda r: r['start_dt'] + r['duration'],
                              axis=1)
                     )['duration']
    fig = make_plots(grp)
    fig.suptitle(f"Plots by Plate Completion Date", size="xx-large")#not centered, but don't know how to help
    
    fig.get_axes()[0].set_xlabel("Test Completion Date")
    fig.show()    

plates_df = get_plates(email_responses)
# f = grp.groupby(grp["computing_id"].apply(lambda i: i[0]))['duration']
f = plates_df.groupby("date")['duration']
# make_plots(f)
# weekly_plot(plates_df, 
#             wk_end = datetime(year = 2021, month= 2, day = 14),
#             plot_result_dates = False)
# trailing_plot(df, end_day = datetime(year=2021, month = 2, day=14).date(), n_trailing = 2)
# time_of_day(df, day = datetime(year=2021, month = 2, day=14))
#%%
class plate_factory:
    
    base_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab"
    def __init__(self, base_dir = base_dir):
        if "results_df" not in os.listdir(base_dir):
            #first time setup
            response_emails = []
            for email_dir in os.listdir(base_dir):
                if 'takeout' in email_dir:
                    response_emails += [proc_gmail_export(email_dir)]
            email_responses = {i.File: i.Datetime for i in response_emails}
            plates_df = get_plates(email_responses)
            plates_df.to_pickle(f"{base_dir}\results_df")
        else:
            plates_df = pd.read_pickle(f"{base_dir}\results_df")
        self.plates_df = plates_df

with open("c:\\Users\\student.DESKTOP-UT02KBN\\Downloads\\credentials.json") as f:
    print(f.readlines())

#%%
def pred_test_response(admin_dt):
    """given a datatime for when the test was administered predict when the result will be provided
    """
    
    
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
                        
                        