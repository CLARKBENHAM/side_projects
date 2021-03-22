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
col_of_lists = df.columns[df.head(1).apply(lambda r: r.iloc[0].count(";") > 0
                                            if isinstance(r.iloc[0], str)
                                            else False)]#isue of using ';' in acknowledgements doesn't happen for first row

for col in col_of_lists:
    df[col] = df[col].apply(lambda s: s.split("; ")
                                     if isinstance(s, str)
                                     else [str(s)]
                            )
col_of_lists
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

def search_col_of_lists(df, names, column, out_name = "Co-authored with"):
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
    out.name = out_name
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

uva_authors = [j.split(" (")[0] 
               for l in df['Authors (Raw Affiliation)']
               for j in l 
               if 'University of Virginia' in j]
sorted(set(uva_authors))
#14321 in set vs 35782 total
#lots of variance, w or w/0 the space: In *space* period
#'Zhang, Y.', 'Zhang, Yajun', 'Zhang, Yan'
author2inst 
#%%
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import math
#WARNING: 'Research Organizations - standardized' entries are a set,
#    not the same length as 'Authors' values if 2 people from same inst

#No non-UVA collaborators
_df = df.head(1000)

edges = [e for l in _df[['Authors', 'Research Organizations - standardized']
                        ].apply(lambda r: combinations([n
                                                        for n,inst in zip(*r) 
                                                            if 'University of Virginia' in inst],
                                                          2), 
                                axis=1)
         for e in l]

#includes not at uva
# _df = df.head(10)
edges2 = [e for l in _df['Authors']
          for e in combinations(l,2) #combinations returns [] if len < 2, filters single entries
          ]

# uva_ix = _df['Research Organizations - standardized'
#               ].apply(lambda l: [ix for ix,inst in enumerate(l) 
#                                 if 'University of Virginia' in inst])                               
# org_col_ix = df.columns.get_loc('Research Organizations - standardized')
# uva_nodes = [(_df.iloc[row_ix, org_col_ix][ix],
#               {"color": "orange"})
#              for row_ix, l_ixs in enumerate(uva_ix)
#                  for ix in l_ixs
#                       if len(l_ixs) >=2]

uva_ix = _df['Research Organizations - standardized'
              ].apply(lambda l: ['University of Virginia' in inst
                                for inst in enumerate(l)])
# get_where_t = lambda i: 
#%%
uva_nodes = [(_df['Authors'][row_ix][ix],
              {"color": "orange"})
             for row_ix, l_ixs in enumerate(uva_ix)
                      if len(l_ixs) >=2
                      for ix,v  in enumerate(l_ixs) 
                      if v]
nonuva_nodes = [(_df['Authors'][row_ix][ix],
                  {"color": "blue"})
                 for row_ix, l_ixs in enumerate(uva_ix)
                      if len(l_ixs) >=2
                      for ix,v  in enumerate(l_ixs) 
                      if not v]
nodes = uva_nodes + nonuva_nodes

def _plot_edges(edges):
    nodes = sorted(set(i for e in edges for i in e))

    cnter = Counter(edges)
    w_edges = [(k[0], k[1], {"weight": n})
               for k,n in cnter.items()]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(w_edges)
    
    disjoint_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    f = plt.figure(figsize=(12,12))
    f.add_subplot(121)
    nx.draw(disjoint_graphs[0], with_labels=True, font_weight='bold')
    f.add_subplot(122)
    pos = nx.spring_layout(G, scale=3, k = 2/math.sqrt(len(G)))
    nx.draw(G, pos, with_labels=True, font_weight='bold')
_plot_edges(edges)

edges_by_dept =1
    
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
    
    
    
#%%
edges = [e for l in df['Authors (Raw Affiliation)']
          for e in combinations(
                              [j.split(" (")[0]
                              for j in l 
                                  if 'University of Virginia' in j],
                                2)]
# edges = [(i,j) for i,j in zip(range(9), range(1,10))] \
#         + [(j,i) for i,j in zip(range(10,19), range(11,20))]\
#             + [(i,j) for i,j in zip(range(15,25,2), range(16,25,2))] \

disjoint_sets = []
parents = {}
parent2kids = {}
for ix,(p,q) in enumerate(edges):
    if ix%(len(edges)//10) ==0 and ix >0:
        print(f"{100*ix/len(edges):.0f}% done")
        
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
                parents[q1] = q#identical
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
        
        
# len(parents)
# set(uva_authors) - set([j for v in parent2kids.values() for j in v])
parent2kids
#%%
par2kids = {}
for k,v in parents.items():
    if v is None and k not in par2kids:
        par2kids[k] = [k]
    elif v is None:
        par2kids[k] += [k]        
    else:
        k1 = k
        while parents[k] is not None:
            k = parents[k]
        if k in par2kids:
            par2kids[k] += [k1]
        else:
            par2kids[k] = [k1]
par2kids  
for k1, k2 in zip(sorted(par2kids.keys()), sorted(parent2kids.keys())):
    assert k1==k2, f"{k1}, {k2}"
    l1,l2 = par2kids[k1], parent2kids[k2]
    assert sorted(l1) == sorted(l2), f"{l1}, \n\n{l2}"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    