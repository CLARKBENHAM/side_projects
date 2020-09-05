# side_projects
Repo for all my extracurricular code

Scrap:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import timeit
import math
import sklearn as sk
import scipy.stats as stats
import pickle
import datetime


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
data1 = pd.read_json(r'cbenham6\Desktop\RCI_MODEL_TRAINING_DATA.json', orient='split')

x = data1.drop('RCI', axis = 1)
y = data1['RCI']
date_chng = "2018-07-01"
#0.6, 0.2, 0.2 train, test, cv sizes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)
x_test, x_cv, y_test, y_cv = train_test_split(x_test, y_test, test_size = 0.5)
xb_train, xb_test, yb_train, yb_test = train_test_split(x.loc[:date_chng], y.loc[:date_chng], test_size = 0.25)
xg_train, xg_test, yg_train, yg_test = train_test_split(x.loc[date_chng:], y.loc[date_chng:], test_size = 0.25)

date_chng1 = "2018-03-31"
date_chng2 = "2019-03-31"#till final prediction; cross validation period
tx_train = x.loc[:date_chng1,:]
tx_test =  x.loc[date_chng1:date_chng2,:]
tx_cv = x.loc[date_chng2:, :]
ty_train = y.loc[:date_chng1]
ty_test = y.loc[date_chng1:date_chng2]
ty_cv = y.loc[date_chng2:]

def train_tree(**kwargs):
    tree = DecisionTreeRegressor(**kwargs)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    score = np.mean([(i-j)**2 for i,j in zip(y_pred, y_test)])
    return score, tree

def train_forest(**kwargs):
    "training for the forest"
    forest = RandomForestRegressor(**kwargs)
    forest.fit(tx_train, ty_train)
    ty_pred = forest.predict(tx_test)
    score = np.mean([(i-j)**2 for i,j in zip(ty_pred, ty_test)])
#    bias = np.mean(y_pred - y_test)
    sm = np.mean(ty_pred)    
    var = np.var(ty_pred - sm)
    return score, sm - np.mean(ty_test), var, forest

print("OMW\n")
n = datetime.datetime.now()

forest_param_name = ['n_estimators', 'min_samples_leaf', 'max_features', 'max_depth', 'bootstrap']
#least regularized location should be at 'origion' most variance element 0 in array

for_n_est = np.exp2(np.arange(4,13,1)).astype(int)[::-1]
for_min_leaf = np.exp2(np.arange(0,8,1)).astype(int)
for_max_features = np.arange(3, data1.shape[1]+5, 5).astype(int)[::-1]
for_max_depth = np.append(np.arange(2, math.ceil(math.log(data1.shape[1],2)) + 3,1), [None])[::-1]
for_bootstrap = [False, True]

#for_n_est = np.exp2(np.arange(4,5,1)).astype(int)[::-1]
#for_min_leaf = np.exp2(np.arange(0,1,1)).astype(int)
#for_max_features = np.arange(3, data1.shape[1]+2, 10).astype(int)[::-1]
#for_max_depth = np.append(np.arange(2, math.ceil(math.log(data1.shape[1],10)) + 3,1), [None])[::-1]
#for_bootstrap = [False, True]

for_params = [for_n_est, for_min_leaf, for_max_features, for_max_depth, for_bootstrap]

cv_forest_score = np.zeros(tuple([len(i) for i in for_params]))
cv_forest_bias = np.zeros(tuple([len(i) for i in for_params]))
cv_forest_var = np.zeros(tuple([len(i) for i in for_params]))

for i_est, est in enumerate(for_n_est):
    for i_leaf, leaf in enumerate(for_min_leaf):
        for i_ft, ft in enumerate(for_max_features):
            for i_dep, dep in enumerate(for_max_depth):
                for i_boot, boot in enumerate(for_bootstrap):
                    cv_forest_score[i_est, i_leaf, i_ft, i_dep, i_boot], \
                    cv_forest_bias[i_est, i_leaf, i_ft, i_dep, i_boot], \
                    cv_forest_var[i_est, i_leaf, i_ft, i_dep, i_boot], _ = \
                    train_forest(n_estimators = est, min_samples_leaf = leaf, \
                                 max_features = ft, max_depth = dep,\
                                 bootstrap = boot)
        print(f"ran leaf {i_leaf}, {datetime.datetime.now() - n}")
    print(f"ran estimate {i_est}, took {datetime.datetime.now() - n}")
    
print(f"Finished running at {datetime.datetime.now()}, it took {datetime.datetime.now() - n}")

def write_pickle_file(filename, data):
    filename = r'C:\Users\cbenham6\\' + filename + '.p'
    filename = filename.replace("\\\\", "\\")
#    with open(filename, 'wb') as filehandler:
#        pickle.dump(data, filehandler)
#    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filename[:-2] + '.csv', index = None)
    elif isinstance(data, np.ndarray):
        data.tofile(filename[:-2] + '.csv', sep = ",")
    else:
        f = open(filename[:-2] + '.txt', 'w')
        f.write(data)
        f.close()
        
write_pickle_file('cv_forest_score',cv_forest_score)
write_pickle_file('cv_forest_bias',cv_forest_bias)
write_pickle_file('cv_forest_var',cv_forest_var)

cp = cv_forest_score.copy()
j_est, j_leaf, j_ft, j_dep, j_boot = np.unravel_index(cp.argmin(), cp.shape)
best_indx = [j_est, j_leaf, j_ft, j_dep, j_boot]
#should update those parameters for best when running on different timeframe
best_params = {forest_param_name[i]:for_params[i][best_indx[i]] for i in range(4)}
best_forest = RandomForestRegressor(**best_params)#n_estimators = j_est, min_samples_leaf = j_leaf, \
#                                 max_features = j_ft,max_depth = j_dep, \
#                                 bootstrap = j_boot)



best_forest.fit(x.loc[:date_chng2,:], y.loc[:date_chng2])
t_pred = best_forest.predict(tx_cv)
residuals = t_pred - ty_cv
print(f"RMSE: {np.mean([i**2 for i in residuals])**0.5}")
print(f"Residual Bias: {np.mean(residuals)}")
print(f"Residual Var: {np.var(residuals)}")

print(f"\nIn sample Score: {cv_forest_score[best_indx]}, \
Bias: {cv_forest_bias[best_indx]}, & Variance {cv_forest_var[best_indx]}")

#%%
min_split_sz = 2*(math.log(data1.shape[1],2) + 2)/data1.shape[0]
cv_tree_results = pd.DataFrame(index = range(1, math.ceil(math.log(data1.shape[1],2)) + 4),\
                          columns = np.arange(min_split_sz,0.5, min_split_sz))
#change df columns to be on log basis, not linear increment
#most regularized at [0,-1]NE, least at [-1,0] SW
for dp in cv_tree_results.index:
    for sz in cv_tree_results.columns:
        cv_tree_results.loc[dp, sz], _ = train_tree(max_depth = dp, min_samples_split = sz)
#%%
cv_tree_results.loc[cv_tree_results.idxmax(axis = 0)]
potential_min_vals = zip(cv_tree_results.idxmax(axis = 1).index, cv_tree_results.idxmax(axis = 1).values)
#min_val = min(cv_tree_results.loc[cv_tree_results.idxmax(axis = 1).index, cv_tree_results.idxmax(axis = 1).values])
min_by_row = [(cv_tree_results.loc[row+1,col], row+1, col) for row, col in enumerate(cv_tree_results.idxmin(axis = 1).values)]
min_at = min(min_by_row, key = lambda m: m[0])
best_params = min_at[1:]

_, best_tree = train_tree(max_depth = best_params[0], min_samples_split = best_params[1])
y_cv_pred = best_tree.predict(x_cv)
#%%
best_tree.fit(tx_train, ty_train)
ty_pred = best_tree.predict(tx_test)
print(np.mean([(i-j)**2 for i,j in zip(ty_pred, ty_test)]))
print(np.mean(ty_pred) - np.mean(ty_test))
#%%
residuals = sorted(y_cv_pred - y_cv)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.hist(residuals, bins = 30, normed = True)
ax1.axvline(0, color = 'r')
norm_plt = stats.norm.pdf(residuals, np.mean(residuals), np.std(residuals))
plt.plot(residuals, norm_plt, '-o')
plt.title(f"MSE for CV set of best regression tree is\
          {np.sum([(i-j) for i,j in zip(y_cv, y_cv_pred)]):.1f},\n \
          KS-test give p-val {stats.kstest(residuals, 'norm')[1]:.3f}")
#np.mean(residuals) #-9; can increase variance

#%%
#heat map for 2d results
import seaborn as sns
sns.heatmap(cv_tree_results, annot=True)
#%%
#3d histogram
_x, _y = np.meshgrid(cv_tree_results.index, cv_tree_results.columns)
xloc, yloc = _x.ravel(), _y.ravel()

top= cv_tree_results.values.ravel()
bottom = np.zeros_like(top)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')
ax1.bar3d(xloc, yloc, bottom, 0.5,0.5,top )
plt.show()

#%%
plt.imshow(cv_tree_results.values)


#train_tree(max_depth = 1)
depth_str = [f"max_depth={i}" for i in range(1, math.ceil(math.log(data1.shape[1],2)) + 2)]#can be slightly unbalanced
depth_str += ["max_depth= None"]
min_samples_str = [f"min_samples_split={i}" for i in np.arange(0,1, min_split_sz)]
