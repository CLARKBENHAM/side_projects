import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import timeit
import math
import re
import requests
import pickle
import os
from datetime import datetime, date, timedelta
#%%
#Weather token: XUxckTkzjdLZvkPvtIpjVwRSawSPGETi
baseurl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
today_dt = datetime.strftime(datetime.now(), "%Y-%m-%d")#gets date from yesturday to today, yyyy-mm-dd
yester_dt = datetime.strftime(datetime.now() - timedelta(2), "%Y-%m-%d")
todaystr = 'enddate=' + today_dt
yesterstr = 'startdate=' + yester_dt
#stationid=COOP:310090
         
def write_info(completed_req, filename):
    "takes a request or a df and writes the info to a file"
    if isinstance(completed_req, pd.DataFrame):
        js_df = completed_req
    else:
        js_df = pd.DataFrame.from_dict(completed_req.json()['results'])
    f = open(filename, 'a')#append mode
    f.write("############################# \n===" + info + '===\n')
    f.write(js_df.to_csv(sep=',', index = False))
    f.write("##################################\n")
    f.close()

def iter_thru_req(requrl, maxresults = 10000):#a decorator
    "gets all count values in requrl, returns a dataframe with those values"
    if requrl[:5] != 'https':
        requrl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/' + requrl
    assert(requrl[:41] == 'https://www.ncdc.noaa.gov/cdo-web/api/v2/')
    assert('offset' not in requrl)
    
    frst_req = requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
    assert(frst_req.status_code < 400)
    print(requrl[41:], frst_req)
    frst_js = frst_req.json()
    offset = frst_js['metadata']['resultset']['offset']
    size = min(frst_js['metadata']['resultset']['count'], maxresults)
    try:
        incremented_by = frst_js['metadata']['resultset']['limit']
    except:
        incremented_by = 25
    
    js_df = pd.DataFrame.from_dict(frst_js['results'])
    while offset + incremented_by < size:
        print(requrl,"\n\n")
        frst_req = requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
        frst_js = frst_req.json()
        offset = frst_js['metadata']['resultset']['offset']
        size = frst_js['metadata']['resultset']['count']
        try:
            incremented_by = frst_js['metadata']['resultset']['limit']
        except:
            incremented_by = 25#default limit size

        new_df = pd.DataFrame.from_dict(frst_js['results'])
        js_df = pd.concat([js_df, new_df])#, ignore_indexs = True)
        new_offset = offset + incremented_by
        new_limit = min(1000, size - new_offset)#maximium value 
        
        lmt_locs = [i for m in re.finditer('[\?]?limit=\d+', requrl) for i in m.span()]   
        if len(lmt_locs) == 0:
            requrl += '?limit=' + str(new_limit)
        else:
            requrl = requrl[:lmt_locs[0]] + '?limit=' + str(new_limit) +  requrl[lmt_locs[1]:]
    
        offset_locs = [i for m in re.finditer('[\?]?offset=\d+', requrl) for i in m.span()]
        if len(offset_locs) == 0:
            requrl += '&offset=' + str(new_offset)
        else:
            requrl = requrl[:offset_locs[0]] + '&offset=' + str(new_offset) +  requrl[offset_locs[1]:]
#    print(js_df.shape)
    return js_df, frst_req
#%%
#for info in ['datasets', 'datacategories', 'locationcategories', 'datatypes']:
#    df, req = iter_thru_req(baseurl + info)#,# + '&' + todaystr + '&' + yesterstr,#baseurl + 'datasets', 
##                     headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
#    assert(req.status_code < 400)
#    print(req.json()['metadata'], info, "\n")
#    write_info(df, 'Desktop\side_projects\weather_scrape.txt')
#    
    
#%%
requrl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/stations?limit=1000' + '&' + yesterstr + '&' + todaystr
station_df, req = iter_thru_req(requrl, maxresults = 3000)

filename = 'Desktop\side_projects\stations.p'
with open(filename, 'wb') as filehandler:
    pickle.dump(station_df, filehandler)

station_df.to_csv(r'Desktop\side_projects\stations.txt', index = None)


#frst_req = requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
#frst_req = requests.get('https://www.ncdc.noaat.gov/cdo-web/api/v2/locationcategories?limit=5', headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
#iter_thru_req('https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets')
        #%%
#need to map COOP to NOAA?
requrl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?limit=1000' + '&' + yesterstr + '&' + todaystr
frst_req = requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
print(frst_req.json())


#data_df, req = iter_thru_req(requrl, maxresults = 1000)


#%%


req = requests.get(baseurl + 'stations?limt=5',# + '&' + todaystr + '&' + yesterstr,#baseurl + 'datasets', 
                 headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
assert(req.status_code < 400)
print(req.json()['metadata'], info, "\n")











#%%Agriculture
#import nass as ns#have downloaded pluggin; isn't working tho
agapi = ns.NassApi('5DBE4EA9-4929-390F-A755-9532F4176392')#agriculture API
q = agapi.query()
q.filter('commodity_desc', 'ALMONDS').filter('ZIP CODE', 22901)
q.count()