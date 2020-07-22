import requests
# import lxml
# import xml.etree.ElementTree as ET
import re
import os
import time
archive_page_url = ""#""
r = requests.get(archive_page_url)
lnk_str = r.content.decode('cp437')

results = re.findall('<li class="archive-list-item">(\d{4}\.\d{2}\.\d{2}): <a\s*href="([^"]+)">([^<]*)<',
               lnk_str)
dates, links, titles = zip(*results)
titles = [" ".join(re.findall("[a-zA-Z]+", t)) for t in titles]

#%%
target_dir = "C:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\Archive"
os.chdir(target_dir)
os.mkdir("pirate_archives")
os.chdir(target_dir + "\\pirate_archives")

failed_links = []
for lnk, title in zip(links, titles):
    try:
         with open(f"{title}.html", 'wb') as f:
             f.write(requests.get(lnk).content)
    except:
        os.remove(title)
        failed_links += [(lnk, title)]
    time.sleep(0.3)
print(failed_links)
