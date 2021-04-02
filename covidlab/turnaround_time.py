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
import pickle5 as pickle
from scipy.stats import invgauss
import random
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

#%%
github_dir="c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab"
# github_dir="c:\\Users\\cb5ye\\Desktop\\side_projects\\covidlab"

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

def proc_gmail_export(email_dir_name):
    "given directory to gmail exported archive of emails, "
    #Data is in the email body itself, not the attachments
    #max date takeout is taken
    if email_dir_name[-4:] == ".zip":    
        zf = ZipFile(f'{github_dir}/{email_dir_name}', 'r')
        zf.extractall(f"{github_dir}/{email_dir_name[:-4]}")
        zf.close()
        os.remove(f'{github_dir}/{email_dir_name}')
        email_dir_name = email_dir_name[:-4]
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
    base_path = f"{github_dir}\{email_dir_name}\Takeout\Mail"
    for mbox_name in os.listdir(base_path):#exports might be under multiple names
        mbox = mailbox.mbox(f"{base_path}\{mbox_name}")
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

def get_email_responses(github_dir=github_dir, newest = False, only_robot = True):
    """"
    newest: only return files from the most reccent takeout file
    only_robot: When Got a robot on Jan 27, 21 the format for barcode files changed
        only return the email responses for these files
    """
    if newest:
        email_dir = max([i for i in os.listdir(github_dir) if 'takeout' in i],
                        key = lambda p: int(p[8:16]))
        response_emails = proc_gmail_export(email_dir)
        email_responses = {i.File.replace(".xlsx", ""): i.Datetime for i in response_emails}
    else:    
        response_emails = []
        for email_dir in [i for i in os.listdir(github_dir) if 'takeout' in i]:
            response_emails += proc_gmail_export(email_dir)
            email_responses = {i.File.replace(".xlsx", ""): i.Datetime for i in response_emails}
    
    if only_robot:
        return {k:v for k,v in email_responses.items()
                if v >= datetime(year=2021,month=1, day= 27)}
    else:
        print("WARNING: These files have a format that wont be adequalty mapped if use email_response2file_name")
        return email_responses

#%
folder_date_re = re.compile('\d{2}\.\d{2}\.\d{4}\-\d{2}\.\d{2}\.\d{4}')
def _get_plate_files(folder_path = "Z:\ResearchData\BeSafeSaliva\BarcodeScan", prev_seen=set(), ending="."):
    """returns full path to all files of interest
    prev_seen: optional set of paths to be excluded
    returns list of paths
    ending: string to check that the path contains [doesn't acctualy check this is the ending vs in path]
    """
    all_files = []
    for f_name,*files in os.walk(folder_path):
        if re.match(folder_date_re, f_name.split("\\")[-1]):
            for f_lst in files:
                if len(f_lst) > 0:
                    # print(f_lst[0])
                    all_files += [f"{f_name}\{f}" for f in f_lst if f not in prev_seen and ending in f]
    return all_files

path2file = lambda i: i.split("\\")[-1]#keep ending

def email_response_file_mod(full_path = True):
    """explores directory Z:\ResearchData\BeSafeSaliva\BarcodeScan and 
        assigns file result time to when that plate was last modified.
    Fake Data: but close enough for explory.
    ?What is dif in .xlsm vs .csv? Seems to be none; will only use .csv
    """    
    print("FAKE DATA: email_response_file_mod")
    if full_path:
        return {p: datetime.fromtimestamp(os.stat(p).st_mtime) 
                        for p in _get_plate_files(ending=".csv")}
    else:
        return {path2file(p): datetime.fromtimestamp(os.stat(p).st_mtime) 
                for p in _get_plate_files(ending=".csv")}
    
def email_responses2file_names(email_names, ret_path = True, base_dir = "Z:\ResearchData\BeSafeSaliva\BarcodeScan"):
    """ dif formats for what is confirmed in email and what is saved in drive
    map by the date of 
    ret_path: if true returns full path, else just file name
    Bad files: ['15_113344_Repeats_LB_M2', 'TR_2020_12_14_1112920_CRTDID_MH_CW', '2021_01_25_100905_REP_DL_M1', '02_112028_BABB_JH_M1_P3', '2021_02_02_141528_BABBBCBD_JH_M1_P5', 'TR_2021_0201_172136_BABBBCBD_JH_M1_P5', 'TR_2021_0201_192111_BABBBC_JH_M1_P7']
    ret: {email_file_name: barcode_file_name}
    """
    out = {}
    for email_name in email_names:
        path_dir = base_dir
        try:
            if email_name[:2] == "TR":
                date = datetime.strptime("".join(email_name.split("_")[1:5]),
                                         "%Y%m%d%H%M%S")
                # return "".join(email_name.split("_")[5:])
            else:
                date = datetime.strptime("".join(email_name.split("_")[:4]),
                                         "%Y%m%d%H%M%S")
            plate_num = email_name[-1]#plate_num could be 'p' as part of a 'rep.csv'
            if plate_num in ('W', 'H'):#every the ending for a valid barcode file
                print("wont process since not an equivalent barcode file", email_name)
                continue# return ""          
                
            if f"{date.year} all files and folders" in os.listdir(path_dir):
                path_dir += f"\{date.year} all files and folders"
            for wk_name in os.listdir(path_dir):
                if re.match(folder_date_re, wk_name):
                    start,end = [datetime.strptime(i, "%m.%d.%Y") 
                                 for i in wk_name.split("-")]
                    if start <= date and date <= end:
                        path_dir += f"\{wk_name}"
                        break
            else:
                print("No week folder for file: ", email_name)
                continue    
            for file in os.listdir(path_dir):
                file_date = re.match("\A\d{4}_\d{2}_\d{2}", file)
                # print(file_date, date)
                if file_date and date.date() == datetime.strptime(file_date.group(),
                                                           "%Y_%m_%d").date():
                    if file.split(".")[0][-1] == plate_num:
                        if ret_path:
                            out[email_name] = f"{path_dir}\{file}"
                        else:
                            out[email_name] = file
                        continue
        except Exception as e:    
            print("bad", email_name, e)
            continue# return ""
    return out

# email_responses = get_email_responses()
# d_email_responses2file_names = email_responses2file_names(email_responses)
# barcode_email_responses = {barcode: email_responses[email]
#                             for email, barcode in d_email_responses2file_names.items()}

# valid_files = set(["".join(i.split("\\")[-1].split("_")[1:]).split(".")[0]  
#                     for i in _get_plate_files()])
# sum(email_response2file_name(i) in valid_files for i in email_responses.keys())
# assert all(email_response2file_name(i) in valid_files for i in email_responses.keys()), "have files don't know how to map"

barcode_re = re.compile("\d{9}-([A-Z0-9]+)-(\d{12})-\d{4}")
def load_plate_barcodes(github_dir=github_dir, fake=False):
    """Warnig, some older commits include a plate with negative duration, 
            but this was filtered going forward
    """
    if fake:
        try:
            return pd.read_pickle(f"{github_dir}\\faked_plates_by_modify_dt")
        except:
            with open(f"{github_dir}\\faked_plates_by_modify_dt", "rb") as fh:
              return  pickle.load(fh)
    try:
        return pd.read_pickle(f"{github_dir}\hide\plates_by_modify_dt")
    except:
        return pd.DataFrame(columns = ['barcode', 'start_dt', 'time', 'date', 'duration', 'plate'])
        
def save_plate_barcodes(df, github_dir=github_dir, fake=False):
    if not fake:
        df.to_pickle(f"{github_dir}\hide\plates_by_modify_dt")
    else:
        def _new_barcode(b):
            end = "-".join(b.split("-")[2:])
            new_num = str(random.randint(0,10**9))
            new_num += "0"*(9-len(new_num))
            new_id = "".join([chr(random.randint(65,90)) for _ in range(random.randint(4,7))])
            new_id = new_id[:2] + str(random.randint(0,9)) + new_id[2:]
            return new_num + "-" + new_id + "-" + end
        df['barcode'] = df['barcode'].apply(_new_barcode)
        df.to_pickle(f"{github_dir}\\faked_plates_by_modify_dt")
        
#all barcodes are duplicated 4 times, one for each gene type in file
def get_plate_barcodes(email_responses):
    """
    #email_repsonses: from email_response_file_mod

    gets tests corresponding to email_responses from the health directory
        can get health dir only on desktop
    email_responses {full_file_path: Datetime recieved}
    ret pd.Df  ['barcode', 'start_dt', 'time', 'date', 'duration', 'plate'] 
        time,date when started    
        finished_dt = duration + start_dt
    """    
    every_plate = []
    prev_df = load_plate_barcodes()
    first_fail = True
    for path in _get_plate_files(ending=".csv", prev_seen = set(prev_df['plate'].unique())):
        try:
            finished_dt = email_responses[path]
        except:
            if first_fail:
                print("WARNING: email_responses is not up to date, missing:")
                first_fail = False
            print(path)
            continue
        try:
            plate = pd.read_csv(path, 
                                header = 1, 
                                ).dropna(how='all', axis=0
                                ).rename({"Sample Name": "barcode"},
                                          axis=1
                                )[['barcode']]
        except:#    NOTE: some files only have 1 column: the Barcode
            try:
                plate = pd.read_csv(path,  
                                header=0,
                                ).rename({ "BarCode Scan": "barcode", 
                                          "Barcode": 'barcode'}, 
                                          axis=1
                                ).dropna(how='all', axis=0
                                )[['barcode']]
            except:
                #theres lots of little exceptions: 10292020_430PM_JWEH_BABBBCBD.csv, 10052020_BA_JD_1205PM_HIS1_2.csv
                plate = pd.read_csv(path, header=0)
                ix = plate.iloc[3].apply(lambda i: 
                                          re.match(barcode_re, str(i)) is not None)
                plate = plate[plate.columns[ix]]
                plate.columns = ['barcode']

        plate = plate[plate['barcode'].apply(lambda i: 
                                                re.match(barcode_re, str(i)) is not None
                                                )][::4]
        plate['start_dt'] = plate['barcode'].apply(lambda i:
                                              datetime.strptime(i.split("-")[2],
                                                                "%Y%m%d%H%M"))

        plate['time'] = plate['start_dt'].apply(lambda i: i.time())
        plate['date'] = plate['start_dt'].apply(lambda i: i.date())
        plate['duration'] = plate['start_dt'].apply(lambda i: finished_dt - i)
        plate['plate'] = path2file(path)
        plate['finish_dt'] = finished_dt
        every_plate += [plate]                
    plates_df = prev_df.append(pd.concat(every_plate))
    bad_ix = plates_df['duration'] < timedelta(hours=0)
    bad_plates = plates_df.loc[bad_ix,'plate'].unique()
    if sum(bad_ix) > 0:
        print(f"""THere are {sum(bad_ix)} samples which are marked as finished 
              before they were collected. THis includes the following plates:
              {bad_plates}. All of these plates will be excluded.""")
        if len(set(bad_plates).difference(set(['2021_02_22_BABBBCBD_LB_P2.csv']))) >= 1:
            print("\n\n***WARNING***: This problem has repeated; as of March 9th only '2021_02_22_BABBBCBD_LB_P2.csv' had this issue")
    plates_df = plates_df[~bad_ix]
    return plates_df

# plates_df = get_plate_barcodes(barcode_email_responses)
plates_df =  load_plate_barcodes(fake=True)
# save_plate_barcodes(plates_df, fake=True)
def sep_wkends(plates_df):
    """
    seprate plates that are collected on or before saturday 
    but aren't processed till at least monday.
    ret(exlc, )
    """
    if 'finish_dt' not in plates_df.columns:#side effecting
        plates_df['finish_dt'] = plates_df['start_dt'] + plates_df['duration']
    make_mon = lambda i: i.date() - timedelta(days=i.date().weekday())
    is_wkend = plates_df.apply(lambda r: make_mon(r['finish_dt']) > make_mon(r['start_dt'])\
                                                 and r['start_dt'].weekday() not in (6,),
                               axis=1)
    weekdays = plates_df[~is_wkend ]
    weekends = plates_df[is_wkend ]
    return weekdays, weekends
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
def _make_tick_labels(grp):
    """
    makes barplot and percentile plot's tick labels so they don't overlap
    """                
    tick_labels = [n for n,_ in grp]
    # if len(tick_labels) > 15: #seems to start overlapping around that, a guess though
    #     prev = None
    return tick_labels

def make_plots(grp):
    """
    Takes a groupby object with timedelta values and plots those against indicies
    #format tick labels? eg. for datetime to ignore year for plates 
    """
    fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize = (20,12), constrained_layout=True, sharex=True)    
    ax.set_title("Average Time to complete test")

    daily_avg = grp.apply(lambda g: (g / np.timedelta64(1, 'D')).mean())
    daily_std = grp.apply(lambda g: (g / np.timedelta64(1, 'D')).std())
    daily_std[daily_std.isna()] = daily_std.max()
    ax.scatter(daily_avg.index, daily_avg.values)
    ax.plot(daily_avg.index, daily_avg.values,  linewidth=3, label = "mean")
    ax.plot(daily_avg.index, 
            [max(i,0) for i in daily_avg.values -2*daily_std],
            'r--',
            label = "mean - 2SD of day")
    ax.plot(daily_avg.index, daily_avg.values + 2*daily_std, 'r--', label = "mean + 2SD of day")
    ax.legend()
    ax.set_ylabel("days till completion")
    ax.set_xlabel("Test Collection Date")
    ax.set_ylim(bottom = 0)#max(0, ax.get_ylim()[0]))
    ax.xaxis.set_tick_params(which='both', labelbottom=True)

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
        print("BAD IX")
        ix = np.arange(len(grp))#groupedby on non-dates
    try:#might only have 1 group
        width = min([i-j for i,j in zip(ix[1:], ix[:-1])])/8
    except:
        width = ix[0]/8

    rects1 = ax2.bar(ix - width*2, daily_12hr.values, width, label='% >12 hours')
    rects2 = ax2.bar(ix - width, daily_24hr.values, width, label='% >24 hours')
    rects3 = ax2.bar(ix, daily_36hr.values, width, label='% >36 hours')
    rects4 = ax2.bar(ix + width, daily_48hr.values, width, label='% >48 hours')
    rects5 = ax2.bar(ix + width*2, daily_72hr.values, width, label='% >72 hours')
    _add_labels(ax2, [rects1, rects2, rects3, rects4, rects5])

    ax2.set_ylabel('Percent')
    ax2.xaxis.set_tick_params(which='both', labelbottom=True)
    ax2.legend()

    #percentiles
    ax3.set_title("Percentile time to completetion")
    per_30 = grp.apply(lambda g: np.quantile(g, 0.30))
    per_80 =grp.apply(lambda g: np.quantile(g, 0.80))
    per_95 =grp.apply(lambda g: np.quantile(g, 0.95))
    per_99 =grp.apply(lambda g: np.quantile(g, 0.99))
    ax3.scatter(_make_tick_labels(grp), per_30 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "30th percentile")
    ax3.scatter(_make_tick_labels(grp), per_80 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "80th percentile")
    ax3.scatter(_make_tick_labels(grp), per_95 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "95th percentile")
    ax3.scatter(per_99.index.astype('str'), per_99 / np.timedelta64(1, 'D'), 
              marker = "_", 
              s= 999,
              label = "99th percentile")
    ax3.legend()
    # ax3.set_xticks(ax.get_xticks())
    ax3.set_ylabel("Days")
    ax3.set_ylim(max(0, ax3.get_ylim()[0]))

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
        fig.suptitle(f"Plots for week of {wk_start} to {wk_end}", size="xx-large")
        fig.get_axes()[0].set_xlabel("Test Results Date")
    else:
        fig.suptitle(f"Plots for week of {wk_start} to {wk_end}", size="xx-large")
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
    fig.suptitle(f"Plots by Plate Completion Date {day}", size="xx-large")#not centered, but don't know how to help
    
    fig.get_axes()[0].set_xlabel("Test Completion Date")
    fig.show()   
    
def _check_weekday_vs_end(weekdays,weekends):
    "visually inspect seperation"
    print("Weekend samples w duration < 10 hours: ", 
          sum(weekends['duration'] < timedelta(hours=10)))
    fig, ax = plt.subplots(1,1)
    ax.scatter(weekends['start_dt'],
                weekends['duration']/np.timedelta64(1,"D"),
                label="weekends: samples held unprocessed")#s=0.3)
    ax.scatter(weekdays['start_dt'],
                weekdays['duration'] /np.timedelta64(1,"D"),
                label="weekdays")# s=0.3)
    end_dt = [datetime(year=2021, month=1,day=2) + timedelta(days=7*i+2*j) 
              for i in range(3,5+len(plates_df['date'].unique())//7) 
              for j in range(2)]
    for d in end_dt:#weekend boundaries
        ax.axvline(x=date2num(d))
    ax.set_xlabel("collection date")
    ax.set_ylabel("days duration")
    fig.suptitle("Collection Date vs. Duration", size='xx-large')
    ax.set_title("Collected on or before saturday but aren't processed till at least monday", 
                 size="large")
    ax.legend()
    fig.show()
    
def plot_weekday_vs_end(weekdays, weekends):
    fig_end = make_plots(weekends.groupby("date")['duration'])
    fig_day = make_plots(weekdays.groupby("date")['duration'])
    fig_end.suptitle(f"Plots for Samples held over weekend", size="xx-large")
    fig_day.suptitle(f"Plots for Samples ex-held over weekend", size="xx-large")
    fig_end.show()
    fig_day.show()

    #take average of values held overnight as else averaging different sets
    #biased to lower friday's collection as that is only day where samples
    #that take more than a day aren't included
    weekday_overnight = weekdays[weekdays.apply(lambda r: (r['start_dt'] + 
                                                           r['duration']).date()
                                                         > r['date'], 
                                                axis=1)]
    avg_diff = weekends['duration'].mean() - weekday_overnight['duration'].mean()
    dur = weekends['duration'].copy()
    # weekends['duration'] -= avg_diff
    replacement_for_neg = weekdays['duration'].quantile(0.1)#smallest 10th percent
    weekends['duration'] = weekends['duration'].apply(lambda i: 
                                                      max(i - avg_diff,
                                                          replacement_for_neg))
    normalized = weekdays.append(weekends)
    fig_normalized = make_plots(normalized.groupby("date")['duration'])
    fig_normalized.suptitle("Duration with weekend samples normalized ", 
                            size="xx-large")
    weekends['duration'] = dur
    fig_normalized.show()

def plot_specific_date(df, date):
    """
    Parameters
    ----------
    df : pd.df
    date :datetime.date

    Returns
    -------
    None.

    """
    if isinstance(date, datetime):
        date = date.date()
    day = df[df['date'] == date]
    # mpl_data = mdates.epoch2num(day['duration']/np.timedelta64(1, 'D'))

    fig, ax = plt.subplots(1,1)
    # ax.hist(mpl_data, bins=50, density = True, color='lightblue')
    # ax.xaxis.set_major_locator(mdates.DayLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d, %b'))
    # fig.show()
    # return day
    data = day['duration']/np.timedelta64(1, 'D')
    ax.hist(data,
             weights=np.ones(len(data)) / len(data),
             bins=40)
    ax.set_title(f"Test Duration for samples collected on {str(date.strftime('%c'))[:-14]}")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Time for completion in days")
    fig.show()
    
def plot_percent_same_day(df):
    same_day = df[df.apply(lambda r: (r['start_dt'] + 
                                      r['duration']).date()
                                    > r['date'], 
                                    axis=1)]
    same_day = same_day.groupby("date")
    all_days = df[df['date'].isin(same_day.indices.keys())
                  ].groupby("date")
    
    fig,ax = plt.subplots()
    ax.plot(same_day.count()/all_days.count())
    # ax.plot([n.strftime("%b %d") for n,_ in same_day],
    #         same_day.size()/all_days.size())
    # ax.set_xticks(rotation=45)
    fig.autofmt_xdate(rotation=45)
    ax.set_title("% of samples collected & finished same day")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("date")
    # ax.xaxis.set_tick_params(which='both', labelbottom=True)

    # end_dt = [datetime(year=2021, month=1,day=2) + timedelta(days=7*i+2*j) 
    #               for i in range(3,5+len(plates_df['date'].unique())//7) 
    #               for j in range(2)]
    sd, ed = min(same_day.indices.keys()), max(same_day.indices.keys())
    end_dt = [sd + timedelta(days=i)
              for i in range((ed-sd).days + 3)
              if (sd + timedelta(days=i)).weekday() in (5,0)]
    for d in end_dt:#weekend boundaries
        ax.axvline(x=date2num(d))
    fig.show()
# plot_specific_date(plates_df, datetime(year=2021, month=2, day=28))
# weekdays, weekends = sep_wkends(plates_df)
# _check_weekday_vs_end(weekdays, weekends)
# _check_weekday_vs_end(*sep_wkends(old_plates_df[old_plates_df['duration'] > np.timedelta64(0, 'D')]))
# plot_weekday_vs_end(weekdays, weekends)
plot_percent_same_day(plates_df)
#%%
fig,ax = plt.subplots()
ax.plot(same_day.count()/all_days.count())
# ax.plot([n.strftime("%b %d") for n,_ in same_day],
#         same_day.size()/all_days.size())
# ax.set_xticks(rotation=45)
fig.autofmt_xdate(rotation=45)
ax.set_title("% of samples collected, finished today")
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("date")
# ax.xaxis.set_tick_params(which='both', labelbottom=True)

# end_dt = [datetime(year=2021, month=1,day=2) + timedelta(days=7*i+2*j) 
#               for i in range(3,5+len(plates_df['date'].unique())//7) 
#               for j in range(2)]
sd, ed = min(same_day.indices.keys()), max(same_day.indices.keys())
end_dt = [sd + timedelta(days=i)
          for i in range((ed-sd).days + 3)
          if (sd + timedelta(days=i)).weekday in (5,0)]
print(end_dt)
for d in end_dt:#weekend boundaries
    ax.axvline(x=date2num(d))
fig.show()
#%%
if __name__ == "__main__":
    email_responses = get_email_responses()

    d_email_responses2file_names = email_responses2file_names(email_responses)
    barcode_email_responses = {barcode: email_responses[email]
                               for email, barcode in d_email_responses2file_names.items()}
    
    plates_df = get_plate_barcodes(barcode_email_responses)
    # save_plate_barcodes(plates_df)
    # plates_df =  load_plate_barcodes()
    
    # #1 plate was 0; don't run this cause want to cach if happens again
    #plates_df = plates_df[plates_df['duration'] > np.timedelta64(0, 'D')] 
    # #the first date Jan 25, has only 1 sample, misleading
    plates_df = plates_df[plates_df['date'] > 
                          datetime(year=2021, month=1, day=25).date()]
    
    #modify axes to be more informative, and to deal with grouping by hour. grib
    f = plates_df.groupby("date")['duration']
    make_plots(f)
    # weekly_plot(plates_df, 
    #             wk_end = datetime(year = 2021, month= 2, day = 28),
    #             plot_result_dates = False)
    #these are funky bc issue above
    # trailing_plot(plates_df, end_day = datetime(year=2021, month = 2, day=14).date(), n_trailing = 2)
    # time_of_day(plates_df, day = datetime(year=2021, month = 2, day=14))
    weekdays, weekends = sep_wkends(plates_df)
    _check_weekday_vs_end(weekdays, weekends)
    plot_weekday_vs_end(weekdays, weekends)
#%%
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, TweedieRegressor

def filter_outliers(df):
    """.2% >5 days duration; all of those in 2020
    1 plate had a time of < 0
    """
    biggest_2021 = max(df['duration'][df['start_dt'] >= datetime(year=2021, month=1, day=1)])
    if min(df['date']).year == 2020:
        n_2020 = sum(df['duration'] >= biggest_2021) -1
        print(f"Biggest in 2021: {biggest_2021}, of which there were {n_2020} in 2020")
    else:
        print(f"Biggest in 2021: {biggest_2021}")
    ix = df['duration'] <= np.timedelta64(5, 'D') 
    print(f"removing {sum(~ix)} which were more than  5 days")
    
    early_ix = plates_df['duration']/np.timedelta64(1, "D") < 0
    bad_plates = set(plates_df[early_ix]['plate'])
    bad_dates = [v for k,v in barcode_email_responses.items() for j in bad_plates if j in k]
    if len(bad_dates) != 0:
        print(f"the following plates were marked complete before the tests were administered: {bad_plates} across {bad_dates}")
        if len(bad_dates) > 1:
            print(f"\n\n\nWARNING!!!!! As of march 03, 2021 only 1 plate had this issue. it has now repeated. {sum(early_ix)} tests total\n\n\n")
    return df[ix & ~early_ix]
    
df = filter_outliers(plates_df)
y,X = df['duration'], df.drop(['duration', 'plate'], axis=1)
y = y/ np.timedelta64(1, 'D')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def encoder_wrapper(func):
    #don't use as decorator since weirds syntax
    """"have to define encoder outside the function so obj persists, 
    will refit in the same manner; 
        (eg. if fewer categories 2nd time still have to output og # columns)
    """
    return Pipeline([
             ('func_endings', FunctionTransformer(func)),
             ('func_onehot', OneHotEncoder()),
             ])

def day_of_week(dates):
    return dates.apply(lambda r: r[0].weekday(), 
                                axis=1
                                ).values.reshape(-1,1)
    # return enc.fit_transform(
    #                     dates.apply(
    #                             lambda r: r[0].weekday(), 
    #                             axis=1
    #                             ).values.reshape(-1,1)
    #                         )

#apply works on row. Have to return a df
def is_weekend(dates):
    "1 if weekend"
    # print(dates.apply(lambda r: int(r[0].weekday() >=5), axis=1).shape, "n\n\na\na\na\na")
    return dates.apply(lambda r: int(r[0].weekday() >=5), axis=1).values.reshape(-1,1)
           
def got_machine(dates):
    "Started using the machine on Feb 1st(?)"
    if isinstance(dates.iloc[0], datetime):
        when_machine = datetime(year=2021, month=2, day=1)
    else:
        when_machine = datetime(year=2021, month=2, day=1).date()
    return dates.apply(lambda r: int(r[0] >= when_machine),
                       axis=1
                       ).values.reshape(-1,1)

date_feats = FeatureUnion([
                 ('day_of_week', encoder_wrapper(day_of_week)),
                 ('is_weekend', FunctionTransformer(is_weekend)),
                 ('got_machine', FunctionTransformer(got_machine)),
                 ])
    
def time2int(time):#int(r[0].strftime("%Y%m%d%H%M%S")
    "cyclical in day"
    # print(type(start_dt.iloc[0][0]))
    return time.apply(lambda r: int(r[0].strftime("%H%M")), 
                          axis=1
                          ).values.reshape(-1,1)
def is_2ndshift(time):
    "1 if 2nd shift"
    change_over = datetime(year=999, month=9, day=9, hour=13, minute = 30).time()
    return time.apply(lambda r: int(r[0] >= change_over), 
                       axis=1
                       ).values.reshape(-1,1)

time_feats = FeatureUnion([
                 ('time2int', FunctionTransformer(time2int)),
                 ('is_2ndshift', FunctionTransformer(is_2ndshift)),
                 ])

#want to append with which plate a batch got submitted to, drawing data from when the barcodes were scanned
#but is that informative? Yes, could learn that plates get scanned @ 930, 11, 2, etc. and map forward
#want mapping of time-> what # plate likely to be today
#wait till have true result times and make modified times an extra column
# def cutoff_time(times):
#     "want to group by when people submit tests over the course of a week"

def timestamp2int(start_dt):#int(r[0].strftime("%Y%m%d%H%M%S")
    # print(type(start_dt.iloc[0][0]))
    return start_dt.apply(lambda r: r[0].value, 
                          axis=1
                          ).values.reshape(-1,1)

start_dt_feats = FeatureUnion([
                        ('timestamp2int', FunctionTransformer(timestamp2int)),
                        ])

def barcode_endings(barcode):
    #['0104\', '0219e', '0101000', '0106q', '0218\', '9925', '0511ATCC(+)','7070', '0404] are few enough to be questionable
    #not sure what endings mean, maybe collection point? 46 unique total 
    endings = barcode.apply(lambda r: r[0].split("-")[-1], axis=1)
    bad_endings = endings.value_counts()[endings.value_counts() < len(barcode)//1000]
    endings = endings.apply(lambda i: 'mistake?' if i in bad_endings else i)
    return endings.values.reshape(-1,1)

barcode_feats = FeatureUnion([
                        # ('barcode_feats', barcode_feats3),
                        ('barcode_feats', encoder_wrapper(barcode_endings)),
                        ])

# class submit_wave:
#     m_knn = None
#     nm_knn = None
    
#     def fit(start_dt):
#         "want to group by when people submit tests over the course of a week"
#         ix = got_machine(start_dt['date'])
#         day_fraction = start_dt['time'].apply(lambda r: 
#                                               r.hour*3600 + r.minute*60 + r.second
#                                               ) / timedelta(days=1).total_seconds()
#         wk_tm = day_of_week(start_dt[['date']]) + day_fraction #BAD? since decorated
#         #started making everyone get tested 1/wk when got machine
#         wm_tm = start_dt[ix]
#         wom_tm = start_dt[~ix]
        
#         return start_dt#.values.reshape(-1,1)

#     def transform(start_dt):
#         pass
        
cols_sep = ColumnTransformer(
                [('all_date_feats', date_feats, ['date']),
                     ('all_time_feats', time_feats, ['time']),
                    ('all_start_dt_feats', start_dt_feats, ['start_dt']),
                     # ('all_barcode_feats', barcode_feats, ['barcode']),#3.6e-05 w/ vs. 0.03 wo
                   # ('submit_wave_feat', submit_wave, ['date', 'time']),
                 ],
                remainder='drop'#'passthrough'
                )

pipe = Pipeline([
            ("pre_procs", cols_sep),
            ("center", StandardScaler()),
            ('lin_reg', TweedieRegressor(power=3, max_iter=1000)),#using inverse noraml dist for glm
            ])

#Multiple comparisions alert
#Tried svm grid search <0, lin w/ & wo/ barcodes ~0, tweedie ~0, and tweedie with standard scaler 0.25. Removing outliers (>5 days) is 0.27

# param_grid = dict(svm__degree=[1,2], svm__C=[0.01, 0.05, 0.1, 1, 10], svm__kernel=['rbf'])
# grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=1)
# grid_search.fit(X_train, y_train)
# grid_search.score(X_test, y_test)
# pipe.fit(X_train.head(200), y_train.head(200))
# pipe.score(X_train.head(100), y_train.head(100))

# pipe.fit(X_train, y_train)#0.037827532107559625
# pipe.score(X_test, y_test)

# out = pipe.fit_transform(X.head(1000))
# out[:5]
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
with open(f"{github_dir}\\robot_submit_model.p", 'wb') as f:
    pickle.dump(pipe, f)

#%%
try:
    with open(f"{github_dir}\model.p", 'rb') as f:
        pipe = pickle.load(f)
except:    
    pipe.fit(X_train, y_train)
    print(pipe.score(X_test, y_test))
    with open(f"{github_dir}\model.p", 'wb') as f:
        pickle.dump(pipe, f)
#%% plots
# res = y_test.tail(100) - pipe.predict(X_test.tail(100))
# plt.scatter(y_test.tail(100) * np.timedelta64(1, 'D'), res * np.timedelta64(1, 'D'))
# plt.xlabel("True")
# plt.ylabel("residual")
# plt.show()

# plt.scatter(y_test.tail(100) * np.timedelta64(1, 'D'), pipe.predict(X_test.tail(100)) * np.timedelta64(1, 'D'))
# plt.xlabel("True")
# plt.ylabel("predicted")
# plt.show()

# plt.hist(pipe.predict(X_test), density=True)# * np.timedelta64(1, 'D'))
# plt.title("Predicted Dist")
# plt.ylabel("Prob")
# plt.xlabel("Pred Value")
# plt.show()

res = y_test - pipe.predict(X_test)
# plt.scatter(range(len(res)), sorted(res), s=0.1)
# plt.show()
 
def plot_pred_dist(dt=None, res = res[::10]):
    "Monte carlos residuals and plots around mean of prediction"
    if dt is None:
        dt = datetime.now()
    cur_x = pd.DataFrame({"barcode": ["nan"],
                      "start_dt":[dt], 
                      "time":[dt.time()],
                      "date":[dt.date()], 
                      })
    loc = pipe.predict(cur_x)
    #guessing here
    mu = pipe['lin_reg'].get_params()['alpha']
    plt.plot(loc, invgauss.pdf(loc, mu), 'r*')
    # # x_axis = np.linspace(invgauss.ppf(0.01, mu),
    # #                       invgauss.ppf(0.99, mu),
    # #                       100)
    # plt.plot(x_axis, invgauss.pdf(x_axis, mu),
    #           'r-', lw=5, alpha=0.6, label='result pdf')
    plt.hist(res + loc, density=True, bins=20)
    plt.gca().set_xlim(0,5)
    plt.title(f"Dist of Duration if tested at: {str(dt.strftime('%c'))[:-8]}")
    plt.xlabel("Days")
    plt.ylabel("Prob")
    plt.show()
    return 
plot_pred_dist()

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
                        
#%% old
def get_plates2(email_responses):
    """
    Written w/o access, makes some assumptions that aren't true at the current moment; but may be with full data
    #email_repsonses: from proc_gmail_export
    gets tests corresponding to email_responses from the health directory
        can get health dir only on desktop
    email_responses {file: Datetime recieved}
    ret pd.Df [date, time] as when results Delievered
        finished_dt = duration + start_dt
    """
    # earliest = min([i.Datetime for i in response_emails]).date()
    # #date completed: average time for test to finish
    # tested_dates = {earliest + timedelta(i):[]
    #                 for i in range((datetime.today().date() - earliest).days + 99)}
    # result_files = []
    folder_path = "Z:\ResearchData\BeSafeSaliva\BarcodeScan"
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

# plates_df = get_plates(email_responses)
#**is** what you want: what if Nov 3rd all plates are slow and spread out over the course of following week?
#next couple days get increased 
#%%
class plate_factory:
    
    def __init__(self):
        if "results_df" not in os.listdir(github_dir):
            #first time setup 
            response_emails = []
            for email_dir in os.listdir(github_dir):
                if 'takeout' in email_dir:
                    response_emails += [proc_gmail_export(email_dir)]
            email_responses = {i.File: i.Datetime for i in response_emails}
            plates_df = get_plates(email_responses)
            plates_df.to_pickle(f"{github_dir}\results_df")
        else:
            plates_df = pd.read_pickle(f"{github_dir}\results_df")
        self.plates_df = plates_df

with open("c:\\Users\\student.DESKTOP-UT02KBN\\Downloads\\credentials.json") as f:
    print(f.readlines())

def pred_test_response(admin_dt):
    """given a datatime for when the test was administered predict when the result will be provided
    """
    pass

def plates_per_day(df):
    pass

#%% also bad
def sep_wkend(plates_df):
    """
    There's downtime over the weekend that we'd want to adjust for
    fixed_gap: 3pm collection on Fri to 8 am Mon is 63 hours
                5pm sat is 39 hours
    ret (weekday samples, samples that are collected but then sit over the weekend)
    #the collection time is Friday but was finished on (?monday, but sometimes on Saturday)
    """
    plate_finish = lambda r: r['start_dt'] + r['duration']
    plates_df['finish_dt'] = plates_df.apply(plate_finish,axis=1)
    
    mon_by_wk = plates_df[plates_df.apply(lambda r: r['start_dt'].weekday() == 0,
                                          axis=1)
                          ].groupby("date")
    first_wk_collec = mon_by_wk['start_dt'].apply(min)#first collection dt on mon of new week
    def _wk_st_partition(i):
        """
        partitions by num partition dates is >= to. (so later date has smaller number)
        by samples by the earliest sample collected on monday is start of new week
        """
        return sum(first_wk_collec.ge(i))
    
    #seg by collection time
    plates_df_bywk = plates_df.groupby(
                                    plates_df['start_dt'].apply(_wk_st_partition)
                                       )
    
    last_finished_plate = plates_df.groupby(
                                    plates_df['finish_dt'].apply(_wk_st_partition)
                                       )['finish_dt'].apply(max)
    
    # first_wk_collec.index = first_wk_collec.apply(lambda i: sum(first_wk_collec.lt(i)))
    
    # #OR
    # def _finish_lt_1wk(g):
    #     if g.name > max(last_finished_plate.index):
    #         return pd.Series(data = [True]*len(g), index=g.index) #haven't had the last weekend yet
    #     else:
    #         return g['finish_dt'] - last_finished_plate[g.name] < timedelta(days=6.5)#slight var in start times
        
    # finished_same_wk = plates_df_bywk.apply(lambda g: _finish_lt_1wk(g))
    # weekdays = plates_df_bywk[finished_same_wk.values]
    # weekends = plates_df_bywk[~finished_same_wk]
    # # _check(weekends, weekdays)
    
    
    # #OR
    # last_finished_plate = plates_df_bywk.apply(lambda g: g.iloc[g['finish_dt'].values.argmin()]['plate'])
    # sort_plates = plates_df.sort_values("finish_dt")['plate'].unique()
    # sort_plates = pd.Series(range(len(sort_plates)), index=sort_plates)
    # def later_plate(grp, is_weekend=False):
    #     cutoff = last_finished_plate[grp.name]
    #     if is_weekend:
    #         ix = grp['plate'].apply(lambda r: sort_plates[r] > cutoff)
    #     else:
    #         ix = grp['plate'].apply(lambda r: sort_plates[r] <= cutoff)
    #     return grp[ix]
    
    # last_finished = plates_df_bywk.apply(lambda g: max(g['finished_dt']))
    # weekends = plates_df_bywk.apply(lambda grp: later_plate(grp, is_weekend=True))
    # weekdays = plates_df_bywk.apply(lambda grp: later_plate(grp, is_weekend=False))# save_plate_barcodes(plates_df)
    # _check(weekends, weekdays)
    
    # new_week = plates_df_bywk.apply(lambda grp:
    #                                 grp.iloc[grp['start_dt'].values.argmin()])#syntax forced by non-unqiue index
    # min_offset = timedelta(hours=10) #gap between first plate with all monday samples and end of last week
    # old_week_plate_max_finish = new_week.apply(lambda r: r['finish_dt'] - min_offset,
    #                                            axis=1)
    min_offset = timedelta(hours=10) #gap between first plate with all monday samples and end of last week
    
    old_week_plate_max_finish = plates_df_bywk.apply(lambda r: min(r['finish_dt'] 
                                                                   - min_offset))
    
    old_week_plate_max_finish.drop(len(old_week_plate_max_finish)-1, inplace=True)#smallest at largest ix
    
    #errors if the cutoff for this and last week should differ, else didn't work if have to find cutoff for this friday (there's no monday to make an end yet)
    old_week_plate_max_finish[-1] = max(old_week_plate_max_finish) + timedelta(days=7)
    #0 is oldest
    old_week_plate_max_finish = old_week_plate_max_finish.sort_values(ascending=False
                                          ).reset_index(drop=True, inplace=False)
    
    def _max_lt(grp):
        "maximum plate finish of week less than old_week_plate_max_finish"
        max_cutoff = old_week_plate_max_finish[grp.name]
        finish = grp['finish_dt']#.apply(plate_finish,axis=1)
        ix = finish <= max_cutoff
        if sum(ix) == 0:
            print("WARNING: for group ", grp.name, " are using default")
        return max(finish[ix], default=max_cutoff)#grp.columns.get_loc("start_dt")])
    
    old_week_plate_finish = plates_df_bywk.apply(_max_lt)
    # old_week_plate_finish = plates_df_bywk.apply(lambda grp: 
    #                                               max(grp['start_dt']\
    #                                                   [grp.apply(plate_finish) 
    #                                                    <= old_week_plate_max_finish[grp.name]
    #                                                   ],
    #                                                   default = grp['start_dt']))
    
    def _wkend_st_partition(grp, is_weekend=True):
        """returns 0 if during the week, 1 if after the weekend cutoff
        any sample not making the last plate run at least 10 hours before 
        the first plate of new week is part of the cutoff. 
        (have to deal with possible sat's; monday plates including both Fri and Mon collections)
        """
        ix = grp.name
        cutoff = old_week_plate_finish[ix]
        finish = grp.apply(plate_finish,axis=1)
        weekend = grp[finish >= cutoff]
        weekday =  grp[finish < cutoff]
        if is_weekend:
            return weekend
        else:
            return weekday
    
    weekends = plates_df_bywk.apply(lambda grp: _wkend_st_partition(grp,
                                                                    is_weekend=True))
    weekdays = plates_df_bywk.apply(lambda grp: _wkend_st_partition(grp,
                                                                    is_weekend=False))# save_plate_barcodes(plates_df)
    return weekdays, weekends
#%% bad
def sep_wkend(plates_df, email_responses, fixed_gap = True):
    """
    There's downtime over the weekend that we'd want to adjust for
    fixed_gap: 3pm collection on Fri to 8 am Mon is 63 hours
                5pm sat is 39 hours
    ret (weekday plates, plates that are collected on Friday but then sit over the weekend)
    #the collection time is Friday but was finished on (?monday, but sometimes on Saturday)
    """    


    # did_sit = plates_df.apply(lambda r: r['start_dt'] ,axis=1)
    week_monday = plates_df.apply(lambda r: r['date'] - r['date'].weekday() ,axis=1)
    fridays = plates_df[plates_df['date'].apply(datetime.weekday) == 5]#plates submitted Friday
    # finished_dt = plates_df['duration'] + plates_df['start_dt']
    last_friday_plates = fridays.groupby(week_monday
                                         ).apply(lambda g: max(g,  
                                                        key = lambda i: i['duration'] + i['start_dt']
                                                            )['plate'])#1 sample could be finished much latter
#but Feb 27th ran on saturday!?!, add check  
#also missing 2 weekends
email_responses = barcode_email_responses
sorted_plates, _ = zip(*sorted(email_responses.items(), 
                                key=lambda item: item[1]))
fridays = pd.Series(dict(sorted([(k,v) for k,v in email_responses.items() 
                               if v.weekday() == 5],
                          key=lambda item: item[1])))
last_friday_dates = fridays.groupby(
                    fridays.apply(lambda i: i.date() - timedelta(days=i.weekday()))#group by week
                            ).apply(max)
last_fridays = last_friday_dates.apply(lambda i: 
                                       fridays.index[fridays.apply(lambda j: j==i)][0])#[k for k,v in fridays.items() if v in last_friday_dates]
after_friday = [sorted_plates[ix+1] 
                for ix, plate in enumerate(sorted_plates)
                if plate in last_fridays.values]

plates2num = lambda k,k2: (email_responses[k] - email_responses[k2]).total_seconds()/(24*60*60)
wkend_gaps = [plates2num(k1,k2)
                  for k1,k2 in zip(last_fridays, after_friday)]
all_gap = [plates2num(k2,k1)
               for k1,k2 in zip(sorted_plates[:-1], sorted_plates[1:])
               if email_responses[k1].date() < email_responses[k2].date()]
avg_gap = (sum(all_gap) - sum(wkend_gaps))/(len(all_gap) - len(wkend_gaps))
wkend_norm_gaps = [i - sum(wkend_gaps) + avg_gap for i in wkend_gaps]


#most reccent sample collected for given plate
latest_element = plates_df[['start_dt', 'plate']].groupby("plate").apply(max)
latest_element_mo = latest_element[latest_element.apply(lambda r: r[0].weekday() == 0, axis=1)]
#largest start_dt of earliest monday run plate with >=1 monday collection time
new_wk_dates = latest_element_mo.groupby(
                            latest_element_mo.apply(lambda i: i[0].date(),axis=1)
                            ).apply(lambda g: min(g['start_dt']))
#monday plates before this are wholly from other collections. 
#Issue of combining samples from multiple days on 1 plate? do it by sample

# new_wk_collection = latest_element[latest_element['start_dt'].isin(new_wk_dates.values)]['plate'].values
plates2norm = [k for k,v in email_responses.items() 
                   for ix in range(min(len(new_wk_dates), len(after_friday)))
                       if v <= new_wk_dates[ix] and v >= after_friday[ix]]#wrong

#Just use as dividers?
