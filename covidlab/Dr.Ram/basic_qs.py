import os
import mailbox
from zipfile import ZipFile
from datetime import datetime
from collections import namedtuple
import re
from pathlib import Path
import pdb

import os, sys
import win32com.client
import pandas as pd
from datetime import timedelta

import openpyxl
from win32com.client.gencache import EnsureDispatch

from cryptography.fernet import Fernet
import base64
import pickle

from itertools import combinations

github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects"
file_path = f"{github_dir}\\covidlab\\hide\\UVA Publications from Dimensions 2017 to 2019.xlsx"

df = pd.read_excel(file_path,
                   header=1)

#entries containing lists as strings
col_of_lists = df.columns[df.head(1).apply(lambda r: r.iloc[0].count(";") 
                                                   > r.iloc[0].count(".")  
                                           if isinstance(r.iloc[0], str)
                                           else False)]#isue of using ';' in acknowledgements
for col in col_of_lists:
    df[col] = df[col].apply(lambda s: s.split("; ")
                                     if isinstance(s, str)
                                     else [str(s)]
                            )

#%%
def _basic_table(df):
    """ Construct a table with the number of publications by year, 
        along with the mean number of citations for each year."""
    
    mk_series = lambda r: pd.Series([r['Publication ID'].count(),
                                     r['Times cited'].mean(),
                                     ],
                                    index = ["N Publications", 
                                             "Mean Citations",
                                    ])
    out = df.groupby("PubYear"
                     ).apply(mk_series
                     ).round({"N Publications":0, 
                              "Mean Citations":1,
                              }
                     ).astype({"N Publications":int,
                               }
                     )
    return out

def search_col_of_lists(df, names, column):
    """
    returns series of counts of by name in a column 
    where the objects in df['column'] are lists
    """
    name_idc = pd.Series(zip(*df[column].apply(
                                           lambda r: [any([n in l for l in r.values])
                                                      for n in names],
                                                  axis=1)),
                          index = names
                          )
    out = name_idc.apply(lambda t: sum(t))
    out.name = "Co-authored with"
    return out
#%% grib
#check to see if some w or w/o middle initial, inital instead of name
def _possible_names(names):
    """returns all possible formats from s:
        first (middle) last -> last, first middle & last, first mid_initial.
    """
    reformat_name = lambda i: f"{i.split(' ')[-1]}, {' '.join(i.split(' ')[:-1])}"
    mid_in = lambda i: i[0] if len(i) == 1 else f"{i[0]} {i[1][0]}."
    reformat_name2 = lambda i: f"{i.split(' ')[-1]}, {mid_in(i)}"
    mid_in_wo_p = lambda i: i[0] if len(i) == 1 else f"{i[0]} {i[1][0]}"
    reformat_name3 = lambda i: f"{i.split(' ')[-1]}, {mid_in(i)}"
    names = [k for i in names.split("; ")
             for k in (reformat_name(i), reformat_name2(i))]
    return names
# names = possible_names(names)

uva_authors = [j.split(" (")[0] for l in df['Authors (Raw Affiliation)']
               for j in l 
               if 'University of Virginia' in j]
sorted(set(uva_authors))
#14321 in set vs 35782 total
#lots of variance, w or w/0 the space: In *space* period
#'Zhang, Y.', 'Zhang, Yajun', 'Zhang, Yan'
#%%
edges = [e for l in df['Authors (Raw Affiliation)']
         for e in combinations(
                 [j.split(" (")[0]
                  for j in l 
                      if 'University of Virginia' in j],
                                 2)]
disjoint_sets = []
parents = {}
parent2kids = {}
for ix,(p,q) in enumerate(edges):
    if ix%100 ==0:
        print(100*ix/len(edges))
    if p in parents and q in parents:
        p1,q1 = p,q
        np = 1
        while parents[p] is not None:
            p = parents[p]
            np += 1
        nq = 1
        while parents[q] is not None:
            q = parents[q]
            nq += 1
        if p != q:
            if np < nq:#p assumed2b larger
                p,q = q,p
            parent2kids[p] += parent2kids[q]
            del parent2kids[q]
            parents[q] = p
        else:#already share a parent, this is speed up
            # print(p1,q1, p,q, sep=";  ")
            if p != p1:
                parents[p1] = p
            if q != q1:
                parents[q1] = p#identical
    elif p in parents or q in parents:
        if q in parents:
            p,q = q,p
        while parents[p] is not None:
            p = parents[p]
        parents[q] = p
        parent2kids[p] += [q]
    else:
        parents[q] = p
        parents[p] = None
        parent2kids[p] = [p, q]
        
        
len(parents), len(parent2kids)
set(uva_authors) - set([j for v in parent2kids.values() for j in v])
#%%
if __name__ == '__main__':
    print('1.       Construct a table with the number of publications by year, along with the mean number of citations for each year.') 
    print( _basic_table(df), "\n")
    
    print("Among the 14,806 publications, how many are co-authored with the following institutions: Johns Hopkins University; Northwestern University; University of Florida?")
    names = ["Johns Hopkins University", 
             "Northwestern University", 
             "University of Florida"]
    column = ['Research Organizations - standardized']
    print(search_col_of_lists(df, names, column), "\n")

    print("3.       Among the publications, how many are authored by the following UVA faculty: Stephen S. Rich; Zongli Lin; James Hunter Mehaffey?")
    names = "Stephen S. Rich; Zongli Lin; James Hunter Mehaffey"
    reformat_name = lambda i: f"{i.split(' ')[-1]}, {' '.join(i.split(' ')[:-1])}"
    names = [reformat_name(i) for i in names.split("; ")]
    column = ['Authors']
    print(search_col_of_lists(df, names, column), "\n")

    print("4.       Create a network graph of these publications with UVA faculty as the nodes, and the edges showing the collaborations. If possible, weight the nodes and edges to show the number of publications.")
    all_pairs =  [e for l in df['Authors (Raw Affiliation)']
                     for e in combinations(
                                            [j.split(" (")[0]
                                             for j in l 
                                                 if 'University of Virginia' in j],
                                             2)]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    