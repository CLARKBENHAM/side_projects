# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 02:28:31 2019

@author: Clark Benham
"""

import requests

url = "https://collab.its.virginia.edu/portal/site/c180a0be-4cb3-4b27-897f-53601fb4ceb5/tool/ee60ec01-a66e-4ad4-99af-46c545d49c16?panel=Main"
data = {"email address": cb5ye@virginia.edu,
        "password": "A8DCZartNb5Kn82VzGnk$"}
response = requests.get(url)
response.content

#%%

