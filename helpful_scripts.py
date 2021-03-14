"""
import .ipynb jupter notebooks from other directories: vk1
encrypted and decrypt files, send and recieve with git: vk2

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
#%%


