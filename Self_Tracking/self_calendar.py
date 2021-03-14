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
# as_str = contents.decode('latin1') #grib, need to invert this step
# as_str = contents.decode('utf-8') #grib, need to invert this step
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
    #zip_str.decode('utf-8') == str(contents)
    f.write(zip_str) #invert back to as_str
    
#%% process
os.chdir(f"{github_dir}\\Self_Tracking")
# shutil.unpack_archive(f"{defolder}.zip", "unzipped")
# os.rename("unzipped\\Takeout\\Calendar", defolder)
for f in os.listdir(defolder):
    print(f)
shutil.rmtree("unzipped")
shutil.rmtree(defolder)
#%%





del send
github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects"
os.chdir(github_dir)
from helpful_scripts import *

# pword = input("Pword to decrypt Calendar: ")
# data_path = f"{github_dir}\\Self_Tracking\\{folder}"
# with open(f"{github_dir}\\Self_Tracking\\{defolder}",'wb') as f:
#     f.write(recieve(data_path, pword))
    
    
