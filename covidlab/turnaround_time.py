import os
#how to access .mbox?
email_responses = {file:response_dt}

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
                
            
            
            
            
            
            