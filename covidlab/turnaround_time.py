#creating a plot average turnaround time by date
import os
import mailbox
from zipfile import ZipFile
from datetime import datetime
from collections import namedtuple
import re
#%%
#response datetime; taking others from 'RAW FILE POST-PROCESS COUNTS'
plate_results = namedtuple("emailed_plate_results", ['Datetime',
                                                    'File',#plate name
                                                    'Positive',
                                                     'Negative',
                                                     'Invalid',
                                                     'Inconclusive',
                                                     'Notfound',
                                                     'Duplicate',
                                                     'Preliminary'])

def _proc_email_file(email_path = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab"):
    "given directory to gmail archive of emails, "
    #os.listdir returns sorted by modified date, earliest first
    #FIX!!
    file = next(iter([i for i in os.listdir(email_path)[::-1] if 'takeout' in i]))
    if file[-4:] == ".zip":    
        zf = ZipFile(f'{email_path}/{file}', 'r')
        zf.extractall(f"{email_path}/{file[:-4]}")
        zf.close()
        os.remove(f'{email_path}/{file}')
        file = file[:-4]
    mbox = mailbox.mbox(f"{email_path}\{file}\Takeout\Mail\covid_response_results.mbox")#f"{email_path}/{file[:-4]}")
    
# [mailbox_body(m) for m in mbox]
#%% bad regex del
regex = re.compile("""([a-zA-Z0-9\_]+).xlsx COVID test results counts. Please review the attachments
-------------------------------------------
RAW FILE PRE-PROCESS COUNTS: 
-------------------------------------------
Negative = \d+
Inconclusive = \d+
-------------------------------------------
                                           
-------------------------------------------
RAW FILE POST-PROCESS COUNTS: 
-------------------------------------------
Positive = (\d+)
Negative = (\d+)
Invalid = (\d+)
Inconclusive = (\d+)
Notfound = (\d+)
Duplicate = (\d+)
Preliminary = (\d+)
-------------------------------------------""")
#'TR_2020_11_24_102140_REP_LB_M2.xlsx COVID test results counts. Please review the attachments\n-------------------------------------------\nRAW FILE PRE-PROCESS COUNTS: \n-------------------------------------------\nNegative = 22\nInconclusive = 2\n-------------------------------------------\n                                           \n-------------------------------------------\nRAW FILE POST-PROCESS COUNTS: \n-------------------------------------------\nPositive = 0\nNegative = 22\nInvalid = 0\nInconclusive = 2\nNotfound = 0\nDuplicate = 0\nPreliminary = 0
re.findall(regex, content)
#%%
file_re = re.compile("([a-zA-Z0-9\_]+.xlsx) COVID test results counts.")
result_re = re.compile("""Positive = (\d+)
Negative = (\d+)
Invalid = (\d+)
Inconclusive = (\d+)
Notfound = (\d+)
Duplicate = (\d+)
Preliminary = (\d+)""")
emails = []
excel_only = []
bad = ""

def get_email_body(message, decode=False):
    """attachments can contain sub attachments;
    decode: if true returns bytes, else str
    """
    if message.is_multipart():
        	content = ''.join(get_email_body(part) for part in message.get_payload())
    else:
        	content = message.get_payload(decode=decode)
    return content

def no_attachments(message):
    "detects if email contains any attachments"
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

for message in mbox:
    if not no_attachments(message):
        content = get_email_body(message)
        try:
            dt = datetime.strptime(" ".join(message['date'].split(" ")[:5]),
                           "%a, %d %b %Y %X")
            file = re.findall(file_re, content)[0]
            values = re.findall(regex, content)[0]
            email = plate_results._make([dt, file, *map(int, values)])
            emails += [email]
        except Exception as e:
            # break
            print(e)
            excel_only += [message]
            bad = content
#bunch of bad emails have only None file names
#%%
os.chdir(f'{email_path}\{file}')
c = None
def extractattachements(message):
    global c
    i = 0
    if message.get_content_maintype() == 'multipart':
        for part in message.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            filename = part.get_filename()
            if filename:
                i = 1
                print(message['date'], filename )
                # fb = open(filename,'wb')
                # fb.write(part.get_payload(decode=True))
                # fb.close()
    if i == 0:
        c = message
        print(1/0)
[extractattachements(b) for b in excel_only]
#%%

excel_only = [b for b in excel_only if not isblank(b)]
len(excel_only)
# [get_email_body(i) for i in blank_msg]

#list(message.walk())[3].get_filename()
#%%
# email_responses = {file:response_dt}

#Nov 2 to present (?) future
tested_dates = {d:[] for d in datetimerange()}#date: time for test to finish

result_files = []
folder_path = "Z:\ResearchData\BeSafeSaliva\Reported_File"
month_folders = os.listdir(folder_path)

for f,res_dt in email_responses:
    #filter out previously seen
    month_folder = [i for i in month_folders if str(res_dt.month ).upper() in i]
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
                
            
            
            
            
            
            