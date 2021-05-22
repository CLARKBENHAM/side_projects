"""
import .ipynb jupter notebooks from other directories: vk1
encrypted and decrypt files, send and recieve with git: vk2
memoize cache results, excluding unfilterable results: vk3
total time in the copied results from google calendar: vk4
Class to add get_feature_names() to FunctionTransformer: vk5
add labels to each rect in a barplot along 1 axis: vk6 

"""
#vk1
from IPython import get_ipython
import os

def import_ipynb(is_jupyter = True, is_desktop = True):
    if is_jupyter:
        ipython = get_ipython()
        #script_dir = %pwd #uncomment to run, but considered invalid in imports
        script_dir = os.getcwd()
        file_path = os.path.abspath(os.path.join(script_dir, '..\option_classes.ipynb'))
        old_locs = locals()
        ipython.magic(f"run -n {file_path}")
        new_locs = locals()
    elif is_desktop:
        #below only works if in same dir
        old_dir = os.getcwd()
        new_dir = 'C:\\Users\\student.DESKTOP-UT02KBN\\MSTG'
        os.chdir(new_dir)
        ipynb_name = 'option_classes'
        old_locs = locals()
        exec(f"from ipynb.fs.full.{ipynb_name} import *")
        new_locs = locals()
        os.chdir(old_dir)
    else:    #deal with google drive
        pass
    #since made a function, need to update with only 
    new_vars = {k: new_locs[k] for k in set(new_locs) - set(old_locs)
                if k != 'old_locs'}
    globals().update(new_vars)    

#%% vk2
from git import Repo
from cryptography.fernet import Fernet
import base64
import pickle

PATH_OF_GIT_REPO = r'C:\Users\student.DESKTOP-UT02KBN\Desktop\side_projects\.git'  # make sure .git folder is properly configured
def git_push(COMMIT_MESSAGE = 'comment from python script'):
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add(update=True)
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occured while pushing the code') 

def git_pull():
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        # repo.git.pull(update=True)
        # repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote(name='origin')
        origin.pull()
    except:
        print('Some error occured while pushing the code')    
        
def send(data_path, data, pword, COMMIT_MESSAGE = 'comment from python script'):
    """
    Have to use a hack since have access on health data and researchers on diff computers
    dir_path: directory to save within
    data: can only encrypt bytes; else will cast with str.encode('utf-8')
    """        
    if not isinstance(data, bytes):
        data = data.encode("utf-8")
    if pword == "":
        with open(data_path, 'wb') as file:
            pickle.dump(data, file)
    else:
        pword = "0"*(32-len(pword)) + pword
        pword = base64.b64encode(pword.encode("utf-8"))
        f = Fernet(pword)
        token = f.encrypt(data)
        with open(data_path, 'wb') as file:
            file.write(token)
    # git_push(COMMIT_MESSAGE)
    
def recieve(data_path, pword):
    """
    return byte object
    """
    git_pull()
    if pword == "":
        with open(data_path, 'rb') as file:
                out = pickle.load(file)
        return out
    else:
        pword = "0"*(32-len(pword)) + pword
        pword = base64.b64encode(pword.encode("utf-8"))
        f = Fernet(pword)
        with open(data_path, 'rb') as file:
            token = file.read()
        return f.decrypt(token)#.decode('utf-8')
#%% vk3

from functools import lru_cache#doesn't work for nonhashable fns
import collections

def memoize(func):
    """incase potentially have unhashable inputs and need to filter out
    """
    mx_size = 32
    cache = dict()
    lru_l = []
    def memoized_func(*args, **kwargs):
        vrs_tup = tuple(list(args) + list(kwargs.keys()) + list(kwargs.values()))
        if not all(isinstance(i, collections.Hashable) for i in vrs_tup):
            return func(*args, **kwargs)
        
        if vrs_tup in cache:
            return cache[vrs_tup]
        result = func(*args, **kwargs)
        cache[vrs_tup] = result
        nonlocal lru_l, mx_size
        lru_l += [vrs_tup]
        if len(lru_l) > mx_size:
            first = lru_l.pop(0)
            del cache[first]
        return result
    return memoized_func 

#%% vk4
import re
def total_time(s):
    """Given the copied results of a google calendar in list view 
    returns sum of time.
    eg. 
    'Algo
    Clark Benham, Accepted
    Friday, January 22, 2021
    11:45pm – 2:15am
    Finished Book: Algo
    Clark Benham, Accepted
    23
    JAN 2021, SAT
    11:45pm – 2:15am
    Finished Book: Algo' -> 3.0
    """
    l = re.findall("\d+:{0,1}\d*(?:am|pm)* – \d+:{0,1}\d*(?:am|pm)*", s)
    def t2int(t):
        s,e = t.split(" – ")
        def _mk_int(s):
            st = re.findall("\d+", s)
            if len(st) > 1:
                st = int(st[0]) + int(st[1]) / 60
            else:
                st = int(st[0])
            if st >= 12:
                st -= 12
            return st
        
        st,et = _mk_int(s), _mk_int(e)
        if t.count('m') == 1:
            return et-st
        elif 'pm' in s:
            return et + (12-st)
        else:#pm in e
            return et + 12 - st
    # [i for i in l if t2int(i)  > 5]
    return   sum([t2int(i) for i in l])
        
#%% vk5
import numpy as np
from sklearn.preprocessing import FunctionTransformer

class FuncTrans_Named(FunctionTransformer):
    def __init__(self, func, feat_name):
        super().__init__(func)
        self.feat_name = feat_name
    
    def get_feature_names(self):
        return np.array([self.feat_name])
    
#eg. 
# df2float = FuncTrans_Named(_df2float, "Linear_Date")
# origin_enc = ColumnTransformer([
#                             ("onehot", OneHotEncoder(drop='first'), ['came_from']),
#                             ("dt", df2float, ['date']),
#                                 ],
#                                 remainder='passthrough')
# origin_enc.get_feature_names()
#%% vk6
import matplotlib.pyplot as plt

def _add_barplot_labels(ax, r_lst, fmt = lambda h: f"{h:.0f}%"):
    """label data within barplots 
    ax: axis
    r_lst: [ax.bar() object]
    fmt: lambda to format height
    """
    bbox = ax.get_window_extent().transformed(
                                    plt.gcf().dpi_scale_trans.inverted())
    _, ax_h = bbox.width, bbox.height
    ax_h *= plt.gcf().dpi
    y_offset = ax_h/8
    for rects in r_lst:
        # """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height > sum(ax.get_ylim())/2:
                t = ax.annotate(fmt(height), 
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, -y_offset),
                            textcoords="offset points",
                            ha='center',
                            va='bottom', 
                            size = 15,
                            color='w')
            else:
                #text on x-axis since plot small
                t = ax.annotate(fmt(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 0),  
                            textcoords="offset points",
                            ha='center',
                            va='bottom',
                            size = 15)
            #to not write if bars are too skinny
            txt_w = t.get_window_extent(plt.gcf().canvas.get_renderer()).width
            bar_w = rect.get_window_extent().width
            if txt_w > bar_w:
                if bar_w > txt_w/2:
                    t.set_size(t.get_size() * bar_w/txt_w)
                else:
                    t.remove()

#%% vk7
