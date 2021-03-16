import os
import shutil
import json
from collections import Counter
import pyautogui as pygu
import time

github_dir = "C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\Self_Tracking"
os.chdir(github_dir)
def most_common_twtr_handles_from_hist():
    take_dir = max([i for i in os.listdir(f"{github_dir}\\hide") if 'takeout' in i],
                  key = lambda j: j.split("-")[1][:8])
    shutil.unpack_archive(f"hide\{take_dir}", "hide\history")
    os.remove(f"hide\{take_dir}")
    
    with open(f"{github_dir}\\hide\\history\\Takeout\\Chrome\\BrowserHistory.json", 'r', encoding="cp866") as f:
        json_hist = json.load(f)
    
    g_handle = lambda u: u.split("/status")[0].split("/")[-1] \
                        if 'status' in u \
                        else u.split("/")[-1] 
    handle_cnts = Counter([g_handle(i['url']) 
                             for i in json_hist['Browser History'] 
                             if 'https://twitter.com' in i['url'] 
                             or 'https://mobile.twitter.com' in i['url']
                             and '?' not in i['url']])
    return handle_cnts

def follow_handles(names):
    search_bar_loc = (833,194)
    top_follow_pos = (1189, 336)
    for n in names:
        pygu.doubleClick(*search_bar_loc)
        pygu.write(n, interval=0.02)
        pygu.press('enter')
        time.sleep(1)
        pygu.click(*top_follow_pos)
        time.sleep(0.1)
        pygu.press("esc") #in case alread followed
        time.sleep(0.2)

# handle_cnts = most_common_twtr_handles_from_hist()
# names = [n for n,_ in handle_cnts.most_common(100)]
# follow_handles(names)