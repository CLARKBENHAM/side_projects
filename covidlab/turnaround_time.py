#creating a plot average turnaround time by date
import os
import mailbox
from zipfile import ZipFile
from datetime import datetime
from collections import namedtuple
import re
from pathlib import Path
import pdb
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

def proc_gmail_export(base_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab"):
    "given directory to gmail exported archive of emails, "
    #Data is in the email body itself, not the attachments
    #max date takeout is taken
    email_dir = max([i for i in os.listdir(base_dir) if 'takeout' in i],
                    key = lambda p: int(p[8:16]))
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

response_emails = proc_gmail_export()
email_responses = {i.File: i.Datetime for i in response_emails}
len(email_responses)
#%%
#  = {file:response_dt}

#Nov 2 to present (?) future
tested_dates = {d:[] for d in datetimerange()}#date: time for test to finish

result_files = []
folder_path = "Z:\ResearchData\BeSafeSaliva\Reported_File"
month_folders = os.listdir(folder_path)

for f,res_dt in email_responses:
    #filter out previously seen
    month_folder = [i for i in month_folders if str(res_dt.month).upper() in i]
    response_files = os.listdir("{folder_path}\{month_folder}")
    


#%% bad order; over files instead of over responses
for month in month_folders:
    response_files = os.listdir("{folder_path}\{month}")
    result_files += response_files
    for file in day_folders:
        if file in email_responses:
            finished_dt = email_responses[file]
            #open excel and get start dates for each test
            for row in file:
                tested_date = re.find(row)
                tested_dt = re.find(row)
                tested_dates[tested_date] += [finished_date - tested_time]
                
            
            
            
            
            
 #%%
class gmail_archive_extra_helpers: 
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

                        
                        
                        
                        