import os
import shutil

github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects"
os.chdir(github_dir)
from helpful_scripts import *

download_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Downloads"
cal_zip = max([i for i in os.listdir(download_dir) if 'takeout' in i],
              key = lambda j: j.split("-")[1][:8])
os.rename(f"{download_dir}\{cal_zip}", f"{github_dir}\\Self_Tracking\\{cal_zip}")
#%%
# def encrypt(cal_zip):
os.chdir(f"{github_dir}\\Self_Tracking")
with open(f"{github_dir}\\Self_Tracking\\{cal_zip}",'rb') as f:
    contents = f.read()

pword = input("Pword to encrypt Calendar: ")
enfolder = f"encrypt_calendar-{cal_zip.split('-')[1]}"
data_path = f"{github_dir}\\Self_Tracking\\{enfolder}"
send(data_path, contents, pword, COMMIT_MESSAGE = 'calendar update')

# encrypt(cal_zip)
#%%
# def decrypt(cal_zip):
# pword = input("Pword to decrypt Calendar: ")
enfolder = f"encrypt_calendar-{cal_zip.split('-')[1]}"
defolder = f"calendar-{cal_zip.split('-')[1]}"
data_path = f"{github_dir}\\Self_Tracking\\{enfolder}"
with open(f"{github_dir}\\Self_Tracking\\{defolder}.zip",'wb') as f:
    zip_str = recieve(data_path, pword)
    f.write(zip_str)
    
#%% process
os.chdir(f"{github_dir}\\Self_Tracking")
shutil.unpack_archive(f"{defolder}.zip", "unzipped")
os.rename("unzipped\\Takeout\\Calendar", defolder)
for f in os.listdir(defolder):
    print(f)
#other proc
shutil.rmtree("unzipped")
shutil.rmtree(defolder)
#%%



