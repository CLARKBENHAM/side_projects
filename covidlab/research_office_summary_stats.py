#give research lab summary statistics on all test results
import os
import mailbox
from zipfile import ZipFile
from datetime import datetime
from collections import namedtuple
import re
from pathlib import Path
import pdb

import xlrd
#%%
dir_path = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\\covidlab\\hide"

file_path = max([i for i in os.listdir(dir_path) if re.match("\A\d+-\d+-\d+",i)],
                key = lambda i: datetime.strptime(i[:10], "%Y-%m-%d") )
wb = xlrd.open_workbook(dir_path + "\\"+file_path)




