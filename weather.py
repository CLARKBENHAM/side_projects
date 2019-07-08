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
import time

#%%
#Weather token: XUxckTkzjdLZvkPvtIpjVwRSawSPGETi
baseurl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
today_dt = datetime.strftime(datetime.now(), "%Y-%m-%d")#gets date from yesturday to today, yyyy-mm-dd
yester_dt = datetime.strftime(datetime.now() - timedelta(2), "%Y-%m-%d")
todaystr = 'enddate=' + today_dt
yesterstr = 'startdate=' + yester_dt
headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'}
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


def write_pickle_file(filename, data, index = None):
    "both writes and pickles data, filename is in current directory"
    filename1 = r"Desktop\side_projects\\" + filename + '.p'
    with open(filename1, 'wb') as filehandler:
        pickle.dump(data, filehandler)
    
    if isinstance(data, pd.DataFrame):
        print(f"WARNING: the index in the csv is: {index}")
        data.to_csv(r'Desktop\side_projects\\' + filename + '.txt', index = index)
    else:
        f = open(r'Desktop\side_projects\\' + filename + '.txt', 'w')
        try:
            f.write(data)
        except:
            f.write(str(data))
        finally:
            f.close()


def make_request(requrl):
    "Makes request w/ requrl added to end of url"
    if requrl[:5] != 'https':
        requrl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/' + requrl
    assert(requrl[:41] == 'https://www.ncdc.noaa.gov/cdo-web/api/v2/')
    return requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})


def get_date_stat_val(req):
    "given a request returns list of date, station, and value"
    adate = [datetime.strptime(i['date'], "%Y-%m-%dT00:00:00") for i in req.json()['results']]
    astation= [i['station'] for i in req.json()['results']]
    aval = pd.to_numeric([i['value'] for i in req.json()['results']])
    return adate, astation, aval


def iter_thru_req(requrl, maxresults = 365*835, index = None, columns = None ):
    "gets all count values in requrl, returns a dataframe with those values"
    if requrl[:5] != 'https':
        requrl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/' + requrl
    assert(requrl[:41] == 'https://www.ncdc.noaa.gov/cdo-web/api/v2/')
    assert('offset' not in requrl)
    
    frst_req = requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
    assert(frst_req.status_code < 400)
   
    out_df = pd.DataFrame(index = index, columns = columns, dtype = np.float32)#might be an int; but not sure
    i = 0
    try:
        offset = 1
        incremented_by = 0
        size = 100
        while offset + incremented_by < size:
    #        print(requrl,"\n\n")
            frst_req = requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
            frst_js = frst_req.json()
            offset = frst_js['metadata']['resultset']['offset']
            size = min(frst_js['metadata']['resultset']['count'], maxresults)
            try:
                incremented_by = frst_js['metadata']['resultset']['limit']
            except:
                incremented_by = 25#default limit size
            offset += incremented_by
    #            req_lst[i] = pd.DataFrame.from_dict(frst_js['results'])
            adate, astat, aval = get_date_stat_val(frst_req)
            out_df.loc[adate, astat] = aval
    
            new_limit = min(1000, size - offset)#maximium value 
            
            lmt_locs = [i for m in re.finditer('limit=\d+', requrl) for i in m.span()]   
            if len(lmt_locs) == 0:
                requrl += '&limit=' + str(new_limit)
            else:
                requrl = requrl[:lmt_locs[0]] + 'limit=' + str(new_limit) +  requrl[lmt_locs[1]:]
        
            offset_locs = [i for m in re.finditer('offset=\d+', requrl) for i in m.span()]
            if len(offset_locs) == 0:
                requrl += '&offset=' + str(offset)
            else:
                requrl = requrl[:offset_locs[0]] + 'offset=' + str(offset) +  requrl[offset_locs[1]:]
    #        print(requrl, size, offset, incremented_by, "\n", sep = " | ")
            print(i)
            i += 1
            time.sleep(1)

    except:
        print("\n\n\n\got an error\n\n\n\n\n###################################")
        return out_df,frst_req 
    return out_df, frst_req

#%%
requrl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/stations?limit=1000' + '&' + yesterstr + '&' + todaystr
station_df, req = iter_thru_req(requrl, maxresults = 3000)

#filename = 'Desktop\side_projects\stations.p'
#with open(filename, 'wb') as filehandler:
#    pickle.dump(station_df, filehandler)
#
#station_df.to_csv(r'Desktop\side_projects\stations.txt', index = None)
#%%
station_latlon =station_df.loc[:,['latitude','longitude']]
station_df.to_csv(r'Desktop\side_projects\station_latlon.txt', index = None)

#frst_req = requests.get(requrl, headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
#frst_req = requests.get('https://www.ncdc.noaat.gov/cdo-web/api/v2/locationcategories?limit=5', headers = {'token': 'XUxckTkzjdLZvkPvtIpjVwRSawSPGETi'})
#iter_thru_req('https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets')
        #%%
#need to map COOP to NOAA?
        #using the fcc API
url = "https://geo.fcc.gov/api/census/block/find?latitude=49.259&longitude=-122.8591&showall=false&format=json"
def fcc_fips_api(lat,lon):
    url = "https://geo.fcc.gov/api/census/block/find?latitude=" +\
    str(lat)+ "&longitude=" + str(lon)+ "&showall=false&format=json"#can't usefstrings for py2
    js =requests.get(url).json()
    return js['Block']['FIPS']

#below didn't work
def fcc_api(lat,lon):
    "takes FCC API calls to pd.Series"
    if isinstance(lat, float):
        url = "https://geo.fcc.gov/api/census/block/find?latitude=" +\
        str(lat)+ "&longitude=" + str(lon)+ "&showall=false&format=json"#can't usefstrings for py2
        js = requests.get(url).json()
        return pd.io.json.json_normalize(js).iloc[0]
    elif isinstance(lat, pd.Series):
        latlon_df = pd.DataFrame(index = range(lat.shape[0]), columns = ['datacoverage', 'elevation', 'elevationUnit', 'id', 'latitude',
       'longitude', 'maxdate', 'mindate', 'name'])
        for i in range(lat.shape[0]):
            latlon_df.iloc[i,:] = fcc_api(lat.iloc[i], lon.iloc[i])
        return latlon_df
    else:
        print("an Error")
fcc_df = fcc_api(station_df['latitude'], station_df['longitude'])#allnan?
write_pickle_file("fcc_df", fcc_df)
#[(j,k) for i in zip(tzt_df.loc[:,['latitude', 'longitude']].values) for j,k in i]
#fips = [fcc_fips_api(lat,lon) for lat,lon in zip(station_df['latitude'], station_df['longitude'])]

#%%
#convert station_df to actual types#grib
wa_stations = station_df.loc[:,['id', 'fips']]
wa_stations = wa_stations.dropna()
wa_stations = wa_stations.loc[wa_stations['fips'].apply(lambda rw: rw[:2] == '53'), :]
stat_id = wa_stations.iloc[0,0]
stat_id2 = wa_stations.iloc[1,0]
stat_id3 = wa_stations.iloc[2,0]

#do all stations have the same amount of info?
#wa_station_dict = {i:[0] for i in wa_stations}

#wa_weather_df = pd.DataFrame(index =ex_indx, columns = wa_stations)
#%%
#datatypes you want
f = open('Desktop\side_projects\search_catagories.txt','r')
search_catagories_txt = f.read()
indx = search_catagories_txt.rfind('===datatypes===') + len('===datatypes===\n')
from pandas.compat import StringIO
search_cat_df = pd.read_csv(StringIO(search_catagories_txt[indx:]), sep=',').dropna()[:-1]

datatype_names = search_cat_df.loc[search_cat_df['name'].apply(\
                  lambda rw: not re.search(r'wind', rw, re.IGNORECASE) is None), 'id'].values.astype('str')

datatype_reqs = [0]*len(datatype_names)
datatypes_named_wind = pd.Series(index = datatype_names, dtype = 'str')
for i, name in enumerate(datatype_names):
    req = make_request(f'datasets?datatypeid={name}')
    datatype_reqs[i] = req
    try:
        datatypes_named_wind[name] = req.json()['results'][0]['id']
    except:
        datatypes_named_wind[name] = None
print(datatypes_named_wind)
datatypes_named_wind = datatypes_named_wind.dropna()
#These are the datatypes that have 'wind' in their description
#%%
#These are the data_types that are in the wind category; all of them are in datatypes named wind
#this can be ignored
req2 = make_request('datatypes?datacategoryid=WIND&limit=1000')
wind_datatypes = pd.DataFrame(data = {'name': [i['name'] for i in req2.json()['results']], 
                                               'id': [i['id'] for i in req2.json()['results']]})
#diff = [i for i in datatypes_named_wind.index if i not in wind_datatypes['id'].values]
#print([i for i in search_cat_df.loc[search_cat_df.loc[:,'id'].apply(lambda nm: nm in diff), 'name']])
#only other interesting one is WT11, 'High or Damaging winds' not in WIND data categorys
wind_datatypes = wind_datatypes.append({'name':'High or damaging winds', 'id':'WT11'}, ignore_index = True)

#all of the values only in wind_datatypes are none
wind_dataset = pd.Series(index = [i for i in wind_datatypes['id'] if i not in datatypes_named_wind])
wind_reqs = [0]*wind_dataset.size
for i, name in enumerate(wind_dataset.values):
    req = make_request(f'datasets?datatypeid={name}')
    wind_reqs[i] = req
    try:
        wind_dataset[name] = req.json()['results'][0]['id']
    except:
        wind_dataset[name] = None
#%%
def get_datatype_description(given_ix):
    "gets the datatype descriptions of given a set of datatype IDs as an index\
    It returns a dataframe"
    b = [list(re.findall("[^,]*,([^,]+),[^,]+,[^,]+,(.*)", j)[0])\
            for j in search_catagories_txt[indx+37:-35].split("\n")]
    ids,des = zip(*b)
    d = pd.DataFrame(pd.Series(des, index = ids), columns = ['Description'])
    both_idx = given_ix.intersection(d.index)
    return d.loc[both_idx]

datatypes_named_wind['Description'] = get_datatype_description(datatypes_named_wind.index)
good_datatypes = pd.DataFrame(
        {"Dataset": datatypes_named_wind, 
         "Description": get_datatype_description(datatypes_named_wind.index).Description},
         index = datatypes_named_wind.index)
good_datatypes.drop('Description', inplace = True)
write_pickle_file("wind_datasets", good_datatypes, index = True)

#%%
#getting what I think are the most relavent datatypes
my_datatypes = list(good_datatypes.drop(good_datatypes.index[2:24]).dropna().index)
my_datatypes = my_datatypes[:3] + my_datatypes[4:]
for i in my_datatypes:
    print(f"{i}: {good_datatypes.loc[i, 'Description']}")

#%%
#gets all the current weather stations in WA
req_wa_stat = make_request('stations?locationid=FIPS:53&startdate=2016-01-01&limit=1000')
e = [[i['id'], i['name']] for i in req_wa_stat.json()['results']]
all_stations_in_wa = pd.Series(list(zip(*e))[1], index = list(zip(*e))[0], dtype='str')

#%%trying to actually pull the data

end_datetime = datetime.strptime("2019-06-14", "%Y-%m-%d")
end_date = end_datetime.strftime("%Y-%m-%d")
start_date = (end_datetime - timedelta(days=364)).strftime("%Y-%m-%d")
data_type = my_datatypes[0]
date_ix = pd.DatetimeIndex([end_datetime - timedelta(x) for x in range(364,-1,-1)])

all_req_dict = {i:[0] for i in good_datatypes.index}
for data_type in good_datatypes.index:
    print(data_type)
    dataset_id = good_datatypes.loc[data_type,'Dataset']#Am I sure I got the dataset right?
    data_url = str(f"https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid={dataset_id}&datatypeid={data_type}&locationid=FIPS:53&startdate={start_date}&enddate={end_date}&limit=1000")
    print(data_url)
    all_req_dict[data_type], frst_req = iter_thru_req(data_url, 
                                                maxresults=365*835, 
                                                index =date_ix, 
                                                columns = all_stations_in_wa.index)
#    print(data_url, "\n\n", all_req_dict[data_type])
    try:#am getting weird errors with empty data being returned?
        print(frst_req.json()['metadata'], "\n\n\n\n\n\n\n\n\n##############")
    except:
        pass
    if all_req_dict[data_type].isna().sum().sum() != 365*835:
        write_pickle_file(str(f"api_data\{str(data_type)}"), all_req_dict[data_type], index = True)
#write_pickle_file("pulled_data", [i for i in all_req_dict.values()])
print(all_req_dict)





#%%
#Don't Know what velocity datatype is
req3 = make_request('datatypes?datacategoryid=VELOCITY&limit=1000')
#req3.json()
vel_datatypes = pd.DataFrame(data = {'name': [i['name'] for i in req3.json()['results']], 
                                               'id': [i['id'] for i in req3.json()['results']]})


#%%

    
#Getting Nameplate capacity
    
#From BPA Website
import subprocess
import sys
import importlib.util
def check_install(package):
    spec = importlib.util.find_spec(package)
    if spec == None:
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        print("Downloaded", package)
check_install('tika')
from tika import parser
import os
from urllib.request import urlretrieve



def get_bpa_nameplate_capacity(cumulative_how, path = ""):
    "Parameters: Path to current Directory(doesn't matter, file will be deleted)\
    returns 2 DataFrames, a df of cummulative BPA capacity, including sales of assets\
    and a df of plant name, 50% completion date, and capacity"
    assert(cumulative_how in ["upto_peak", "bpa_owned", "only_increasing"])
    i = 1
    fname = f"temp_download_{i}.pdf"#so won't get duplicated file names
    while os.path.exists(path + fname):
        i += 1
        fname = f"temp_download_{i}.pdf"
    nameplate_url_bpa = "https://transmission.bpa.gov/business/operations/Wind/WIND_InstalledCapacity_LIST.pdf"
    print("Name Used: ", path + fname)
    urlretrieve(nameplate_url_bpa, path + fname)

    raw = parser.from_file(path + fname)
    os.remove(path + fname)
    regex = '(.*?) (-?\d+) (\d+\/\d+\/\d+) (\d+)'
    capacity_changes = [list(j) for i in raw['content'].split('\n') \
                        for j in re.findall(regex, i) \
                        if len(j) != 1 and j[1] != '0']#refindall returns list of single tuple which equals j
    capacity_changes = pd.DataFrame.from_records(capacity_changes, 
                                                 columns = ['Plant Name', 'Plant Capacity', 'Date', 'Cum Capacity'])
    capacity_changes.iloc[:,1] = pd.to_numeric(capacity_changes.iloc[:,1])    
    capacity_changes.iloc[:,3] = pd.to_numeric(capacity_changes.iloc[:,3])
    capacity_changes.Date = pd.to_datetime(capacity_changes.Date)
    
    plant_idx = capacity_changes.loc[:, "Plant Capacity"] > 0
    plants = capacity_changes.iloc[:, [0,1,2]][plant_idx]
    
    if cumulative_how == "upto_peak":
        peak_idx = capacity_changes.loc[:,'Cum Capacity'].idxmax()
        end_idx =  capacity_changes.index[-1]
        cum_capacity = capacity_changes.drop(columns = ["Plant Name", "Plant Capacity"], index = range(peak_idx + 1,  end_idx + 1))
    elif cumulative_how == "historical":
        cum_capacity = capacity_changes.drop(columns = ["Plant Name", "Plant Capacity"])
    elif cumulative_how == "only_increasing":
        dec_index = capacity_changes.index[~plant_idx]
        cum_capacity = capacity_changes.drop(columns = ["Plant Name", "Plant Capacity"], index = dec_index)
        cum_capacity.loc[:, 'Cum Capacity'] = np.cumsum(plants.loc[:, 'Plant Capacity'])
    return cum_capacity, plants

bpa_cum_cap, plants = get_bpa_nameplate_capacity("only_increasing", path = "Desktop\side_projects\\")

#%%
#Get's plant capacity from a diff website


import requests
def get_renerableorg_info(plant_status = ""):
    aweb_url = "https://renewablenw.org/project_map?field_project_state_value%5B%5D=ID&field_project_state_value%5B%5D=MT&field_project_state_value%5B%5D=OR&field_project_state_value%5B%5D=WA&tid%5B%5D=7&field_project_opstatus_value%5B%5D=Approved&field_project_opstatus_value%5B%5D=Proposed&field_project_opstatus_value%5B%5D=Operating&field_project_opstatus_value%5B%5D=In+Permitting+Process&field_project_opstatus_value%5B%5D=Under+Construction"
    got = requests.get(aweb_url)
    got_str = got.content.decode('utf-8')
    table_start_ix = got_str.find("<tbody>")
    table_end_ix = got_str.find("</tbody>")
    table_str = got_str[table_start_ix:table_end_ix]
    regex = '(.*)<\/td\>'#null spaces should be included
    table_list = [list(re.findall(regex, i )) for i in table_str.split("</tr>")]
    cols = ["Name",	"State",	"County", "Capacity", "Unit", "Developers", "Partners", "Operating Status", "Year"]
    all_plants = pd.DataFrame.from_records(table_list, columns = cols)
    if plant_status!= "":
        status_ix = all_plants.loc[:, "Operating Status"] == plant_status
        return all_plants[status_ix]
    return all_plants
#%%
head_start_ix = 0#got_str.find("<head>")
head_end_ix = got_str.find("</head>")
head_str = got_str[head_start_ix:head_end_ix]

script_start_ix = head_str.find("<script>")
script_end_ix = head_str.find("</script>")
script_str = got_str[script_start_ix:script_end_ix]
script_str



#%%    
r = requests.get(url).json()
print(r['Block']['bbox'],
r['Block']['FIPS'],
r['County']['FIPS'],
r['County']['name'],
r['State']['FIPS'],
r['State']['code'],
r['State']['name'],
r['status'],
r['executionTime'])



#%%

