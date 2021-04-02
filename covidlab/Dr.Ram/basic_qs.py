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
df2 = df.copy()

df = df2.copy()
#entries containing lists as strings
col_of_lists = df.columns[df.head(1).apply(lambda r: r.iloc[0].count(";") > 0
                                            if isinstance(r.iloc[0], str)
                                            else False)]#isue of using ';' in acknowledgements doesn't happen for first row
col_of_parens = df.columns[df.head(1).apply(lambda r: r.iloc[0].count(");") > 0
                                            if isinstance(r.iloc[0], str)
                                            else False)]

no_comma = set()
known_no_comma = ("U.Bhawandeep", "Ashok Kumar")
known_no_comma_rep = ("Bhawandeep, U.", "Kumar, Ashok")
def filter_nocomma(l):
    """Some names don't have a comma, remove those.
        Side effecting update to no_comma
    """
    if l == 'ret':
        return no_comma - set(known_no_comma)
    out = []
    for i in l:
        if len(i) <= 1:#sometimes cruft on end
            print("cruft on end ", l, i)
            continue
        if i.count(",") > 0:
            out += [i]
        elif i in  known_no_comma:
            out += [known_no_comma_rep[known_no_comma.index(i)]]
        else:
            no_comma.update([i])
    return out
        
#spliting on ; may not work
# filter_len1 = lambda l: [i for i in l if len(i) > 1]
# only_comma_gen = filter_nocomma()
def col2lambda(col):
    if col in col_of_parens:
        seperator = "); "
    else:
        seperator = "; "
    has_comma_cols = ['Authors', 'Authors (Raw Affiliation)']
    if col in has_comma_cols:
        s_lambda =  lambda s: filter_nocomma(
                                s.strip(seperator).strip(" ").split(seperator))
    else:
        s_lambda =  lambda s: s.strip(seperator).split(seperator)
    return lambda s: s_lambda(s) \
                     if isinstance(s, str) \
                     else [str(s)]
            
df[col_of_lists] = df[col_of_lists].apply(lambda col:
                                          col.apply(col2lambda(col.name)))

#WARNING! this is sufficent, but may not be nessisary to be bad
df['author_align_bad'] = df['Authors'].apply(len) \
                        != df['Authors (Raw Affiliation)'].apply(len)

print(no_comma - set(known_no_comma))#combines has_comma_cols
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
#%%
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

uva_authors = set([j.split(" (")[0] 
               for l in df['Authors (Raw Affiliation)']
               for j in l 
               if 'University of Virginia' in j])
sorted(uva_authors)
print("Num uva authors: ", len(uva_authors))
#14321 in set vs 35782 total
#lots of variance, w or w/0 the space: In *space* period
#'Zhang, Y.', 'Zhang, Yajun', 'Zhang, Yan'
#%% combine names
from collections import Counter
last_names_cnt = Counter([j.split(",")[0]
                  for l in df['Authors'] for j in l])
last2first = {}
for ix, l in enumerate(df['Authors']):
    # try:
    for ix2, j in enumerate(l):
        #1 person had comma before 'jr'; 'Gilliard, Jr., Robert J.'
        j = j.title().replace("  ", " ")
        comma_ix = j.rfind(",")
        last = j[:comma_ix].replace(",", "").replace(" ", "")
        rest = j[comma_ix:].replace(",", ""
                                    ).replace("-", "."
                                    ).strip(" ")
        #some entries have group included in author list, skip entirely
        other_exclusion = any([i in rest for i in ('Of ',
                                                   "For ",
                                                   'Lab ',
                                                   'The ',
                                                   'And ',      
                                                   ' In ',
                                                   'Alzheimer',
                                                   "Mary'S", 
                                                   "Medicine",
                                                   'Medical',
                                                   'Health ',
                                                   'Writing',
                                                   )])
        if last in ('Group', 'Consortium', 'Council', 'Centers') or other_exclusion:
            continue

        r_pieces = []
        #some names are only last, will append as ''
        if len(rest) > 0 and rest!="N/A":
            for p in rest.split(" "): 
                if "." in p:
                    for ps in p.strip(".").split("."):
                        r_pieces += [f"{ps.upper()}."]
                elif len(p) == 1:
                    r_pieces += [f"{p.upper()}."]
                elif ord(p[-1]) in range(65, 91):#is uppercase
                    r_pieces += [p[:-1]]
                    r_pieces += f"{p[-1].upper()}."
                else:
                    r_pieces += [p]
        rest = " ".join(r_pieces)
        if last in last2first:
            last2first[last] += [(rest, ix)]
        else:
            last2first[last] = [(rest, ix)]
    # except Exception as e:
    #     try:
    #         print(e)
    #         print(ix, len(l),
    #               'University of Virginia' in df['Authors (Raw Affiliation)'][ix][ix2],
    #               "\n")
    #     except:
    #         print("FAILED: ", ix, ix2, j)

cnt =0
for k, names in last2first.items():
    for s in names:
        if s.count(" ") > 0:
            f_ix = s.index(" ")
            first, rest = s[:f_ix], s[f_ix+1:]
            if len(rest) != 2 and rest[0][-1] == "." and rest.count(" ") ==0:
                print(k, i[0])
                cnt += 1
cnt
#%% use for advanced
from collections import namedtuple
from itertools import groupby
tupe_named = namedtuple("nonlast_name", 
                        ("first", "f_initial", "rest", "r_initial", 'all_text'))
def _make_name(s):
    """"s is the name, excluding the last name
    so first and middle initials, in order
    """
    #first name only
    if s.count(" ") == 0:
        return tupe_named._make([s, None, None, None, s])
        
    f_ix = s.index(" ")
    first, rest = s[:f_ix], s[f_ix+1:]
    if first[-1] == ".":
        f_initial = first[0]
        first = None
    else:
        f_initial = None
    if rest.count(" ") > 0:#middle names
        m_ix = rest.rfind(" ")
        middle, back = rest[:m_ix], rest[m_ix:]
        if back[-1] == ".":
            # print(s, rest)
            r_initial = back[0]
        else:
            r_initial = None
    else:
        if rest[-1] == ".":
            r_initial = rest[0]
            rest = None
        else:
            r_initial = None
    return tupe_named._make([first, f_initial, rest, r_initial, s])

# _make_name(i[0])
#%%
def _group_name(s):
    pass

names2ix = {}
for last, firsts in list(last2first.items()):    
    do_basic = True
    if do_basic:
        for n,g in groupby(sorted(firsts, key = lambda i: i[0])):
            ixs = [i[1] for i in g]
            person = f"{last}, {n[0]}"
            if person in names2ix:
                names2ix[person] += ixs
            else:
                names2ix[person] = ixs   
        # name2ix = {}
        # for n,ix in firsts:
        #     if n in name2ix:
        #         name2ix[n] += [ix]
        #     else:
        #         name2ix[n] = [ix]
        # for n, ixs in name2ix.items():
        #     person = f"{last}, {n}"
        #     if person in names2ix:
        #         names2ix[person] += ixs
        #     else:
                # names2ix[person] = ixs
    else:#grib
        _names, ixs = list(zip(*firsts))
        _names = [_make_name(s) for s in _names]
    
        n_counts = Counter(_names)
        u_names = set(n_counts.keys())
        #combine if, unique last name for a last_initial 
        #first name and initial, last name and initial, and unique initials
        
        fullname2names = {k.text: [k] for k,v in n_counts.items() if v > 1}
        nonunique_n = fullname2names
        # unique_n = [k for k,v in n_counts.items() if v == 1]
        
        # n_firsts = Counter([i.first for i in u_names])
        # n_rest =  Counter([i.rest for i in u_names])
    
        by_first_i = groupby(
                        sorted([(i.f_initial if i.f_initial 
                                   else i.first[0] if i.first
                                   else None, 
                                   i)
                               for i in _names],
                             key = lambda i: i[0]),
                        key = lambda i: i[0])
        for key, group in by_first_i :
            #combine if for a given first_initial, all share first name(unique first name) 
            if all(group[0].first == g.first or g.first is None for g in group):
                u_first_name = max(group,
                                   key = lambda g: len(g.text) 
                                   ).text
                if u_first_name not in fullname2names:
                    fullname2names[u_first_name] = [u_first_name]
                for g in set(group):
                    if g.text == u_first_name:
                        continue
                    if g.text in fullname2names and g.text != u_first_name:
                        fullname2names[u_first_name] += fullname2names[g.text]
                    if g.text not in fullname2names:
                        fullname2names[u_first_name] += [g.text]    
# len(names2ix.keys())

ix2names = {}
for n,ixs in names2ix.items():
    for ix in ixs:
        if ix in ix2names:
            ix2names[ix] += [n]
        else:
            ix2names[ix] = [n]
df['Clark_names'] = pd.Series(ix2names.values(), index= ix2names.keys())

#len([i for n in ix2names.values() for i in n]) # 980k unique authors
#%%
def make_affiliations_df(df):
    """but some people not given departments at all
    """
    # aff_re = re.compile("([a-zA-Z-]+\, [a-zA-Z -]+) \(([^,]+), ([^,]+), ([^\.]+)")
    # #some lack address; eg. '(Curry School of Education, University of Virginia'
    # #some lack department eg  'Shezan, Faysal Hossain (University of Virginia'
    aff_re = re.compile("([^(]+) \(([^,]+), ([^,]+), ([^\.]+)")
    def _split_raw_aff(s):
        """string of 'Authors (Raw Affiliation)' to list
            ret: (name, department, univsersity, address)
                name is 'last, first' but may include:
                    middle names
                    names w/ '-'
            if doesn't match this format don't know how to process, 
            the ordering of department vs uni vs address isn't consistent if
            not all 3
        """
        # name, rest = s.split(" (")
        aff = re.match(aff_re, s).groups()
        #but dept and uni sometimes combined
        if 'University of Virginia' in aff[1]:
            aff = (aff[0],
                   aff[1].replace('University of Virginia', '').strip(' '),
                   'University of Virginia',
                   f"{aff[2]}, {aff[3]}")
        return list(aff)
    #uses print date since 3k less nans than online date.  df[['Publication Date (print)', 'Publication Date (online)']].isna().sum()
    df_cols = df[['Authors (Raw Affiliation)', 'Publication Date (print)']].values
    out = pd.DataFrame([_split_raw_aff(raw_aff) + [pub_date]
                        for auth_list, pub_date in df_cols
                         for raw_aff in auth_list
                          if re.match(aff_re, raw_aff)],
                      columns =  ["Name", "Department", 
                                  "University", "Address",
                                  "Pub_date"])
    out.drop_duplicates(subset = ["Name", "Department", "University", "Address"],
                            inplace=True)
    invalid = [raw_aff
               for auth_list, pub_date in df[['Authors (Raw Affiliation)',
                                             'Publication Date (print)']].values
                for raw_aff in auth_list
                if not re.match(aff_re, raw_aff)]
    return out, invalid

author_inst, invalid = make_affiliations_df(df)
#grib, lossing a lot of names, 200k
clark_names = set([i for n in ix2names.values() for i in n])
author_inst = author_inst[author_inst['Name'].isin(clark_names)]
author_inst
print("Change from old way of calculating:", len(uva_authors)
                                           - len(set(author_inst['Name'])))
#%%
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import math
#WARNING: 'Research Organizations - standardized' entries are a set,
#    not the same length as 'Authors' values if 2 people from same inst

#has more institutionss than researchers
# df[['Authors', 'Research Organizations - standardized']][77]

def og_df_proc(df, uva_authors = uva_authors):
    assert isinstance(uva_authors, set)
    
    _df = df[df['Authors'].apply(len) < 8]
    
    uva_auth = [a for l in _df['Authors'] for a in l if a in uva_authors]
    nonuva_auth = [a for l in _df['Authors'] for a in l if a not in uva_authors]
    uva_nodes = [(a,
                  {"color": "orange"})
                 for a in uva_auth]
    print(f"{100*len(uva_nodes)/len([a for l in _df['Authors'] for a in l]):.1f}% of authors kept as UVA")
    nonuva_nodes = [(a,
                      {"color": "blue"})
                    for a in nonuva_auth]
    # nodes = uva_nodes + nonuva_nodes
    edges = [e for l in _df['Authors']
              for e in combinations(l,2)
              if any([n in uva_authors for n in l])]
    return edges, uva_nodes, nonuva_nodes
    
    # #No non-UVA collaborators
    #includes not at uva
    # _df = df.head(10)
    # edges2 = [e for l in _df['Authors']
    #           for e in combinations(l,2) #combinations returns [] if len < 2, filters single entries
    #           ]
    
    # uva_ix = _df['Research Organizations - standardized'
    #               ].apply(lambda l: ['University of Virginia' in inst
    #                                 for inst in l])

def _segment_by_dept(athor_inst):
    "unfinished"
    _df = author_inst.head(1000)
    edges = [e for l in _df[['Name', 'University']
                            ].apply(lambda r: 
                                    combinations([n
                                                for n,inst in zip(*r) 
                                                    if 'University of Virginia' in inst],
                                                  2), 
                                    axis=1)
            for e in l]
    
    # uva_names = set(author_inst['Name'][author_inst['University'].apply(lambda inst: 
    #                                  'University of Virginia' in inst)])
    #eliminates ~everyone; GRIB
    uva_ix = author_inst['Address'].apply(lambda i: 'Charlottesville' in i)
    uva_names = set(author_inst.loc[uva_ix, 'Name'])
    print(len(uva_names))
    uva_nodes = [(_df['Name'][row_ix][ix],
                  {"color": "orange"})
                 for row_ix, l_ixs in enumerate(uva_ix)
                          if len(l_ixs) >=2
                          for ix,v  in enumerate(l_ixs) 
                          if v]
    nonuva_nodes = [(_df['Name'][row_ix][ix],
                      {"color": "blue"})
                     for row_ix, l_ixs in enumerate(uva_ix)
                          if len(l_ixs) >=2
                          for ix,v  in enumerate(l_ixs) 
                          if not v]
    nodes = uva_nodes + nonuva_nodes


def _plot_edges(edges, nodes= None):
    if nodes is None:
       nodes = sorted(set(i for e in edges for i in e))
    cnter = Counter(edges)
    w_edges = [(k[0], k[1], {"weight": n})
               for k,n in cnter.items()]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(w_edges)
    #in sorted order with largest first
    disjoint_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    disjoint_graphs = list(sorted(disjoint_graphs, 
                                  key = lambda i: len(i),
                                  reverse= True))
    f = plt.figure(figsize=(12,12))
    f.add_subplot(121)
    nx.draw(disjoint_graphs[len(disjoint_graphs)//10], 
            with_labels=True,
            font_weight='bold')
    f.add_subplot(122)
    pos = nx.spring_layout(G, scale=3, k = 2/math.sqrt(len(G)))
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    
edges, uva_nodes, nonuva_nodes = og_df_proc(df)
nodes = uva_nodes + nonuva_nodes
# _plot_edges(edges, nodes)

#%% graph 
color_map ={ **{n[0]:'orange' for n in uva_nodes}, 
            **{n[0]:'blue' for n in nonuva_nodes}}

cnter = Counter(edges)
w_edges = [(k[0], k[1], {"weight": n})
            for k,n in cnter.items()]
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(w_edges)

disjoint_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
disjoint_graphs = list(sorted(disjoint_graphs, 
                                  key = lambda i: len(i),
                                  reverse= True))

nr, nc = 4,4
f, axes = plt.subplots(nrows=nr, ncols = nc, figsize=(25,12))
ax = axes.flatten()
for i in range(nr*nc):
    # f.add_subplot(int(f"55{i}"))
    g = disjoint_graphs[100+i]
    colors = [color_map[n] for n in g.nodes]
    nx.draw(g,
            with_labels=True,
            font_weight='bold',
            node_color = colors,
            ax = ax[i])
    ax[i].set_axis_off()
f.tight_layout()
f.show()
#%%
if __name__ == '__main__':
    print('1.       Construct a table with the number of publications by year, along with the mean number of citations for each year.') 
    print( _basic_table(df), "\n")
    
    print("2. Among the 14,806 publications, how many are co-authored with the following institutions: Johns Hopkins University; Northwestern University; University of Florida?")
    names = ["Johns Hopkins University", 
             "Northwestern University", 
             "University of Florida"]
    column = ['Research Organizations - standardized']
    print(search_col_of_lists(df, names, column).to_string(), "\n")

    print("3.       Among the publications, how many are authored by the following UVA faculty: Stephen S. Rich; Zongli Lin; James Hunter Mehaffey?")
    names = "Stephen S. Rich; Zongli Lin; James Hunter Mehaffey"
    reformat_name = lambda i: f"{i.split(' ')[-1]}, {' '.join(i.split(' ')[:-1])}"
    names = [reformat_name(i) for i in names.split("; ")]
    column = ['Authors']#['Clark_names']
    print(search_col_of_lists(df, names, column).to_string(), "\n")

    print("4.       Create a network graph of these publications with UVA faculty as the nodes, and the edges showing the collaborations. If possible, weight the nodes and edges to show the number of publications.")
    all_pairs =  [e for l in df['Authors (Raw Affiliation)']
                     for e in combinations(
                                            [j.split(" (")[0]
                                             for j in l 
                                                 if 'University of Virginia' in j],
                                             2)]
    
    
    
#%% scrape
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    