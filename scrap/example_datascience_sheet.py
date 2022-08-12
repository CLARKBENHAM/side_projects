#Written by Clark Benham 
import numpy as np
import pandas as pd
from scipy import stats
import os 
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import subprocess    

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion

github_dir = "c:\\Users\\student.DESKTOP-UT02KBN\\Desktop\\side_projects\interview_practice_takehome"
os.chdir(f"{github_dir}\\hide\\data")

# [print(f"{i[:-4]} = pd.read_csv('{i}')") for i in os.listdir()]

conversion_rates = pd.read_csv('conversion_rates.csv')
insurance = pd.read_csv('insurance.csv')
lead_sale_stats = pd.read_csv('lead_sale_stats.csv')
names_id_age = pd.read_csv('names_id_age.csv')

def print_full(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,'expand_frame_repr', False):  # more options can be specified also
        print(df)
os.chdir(f"{github_dir}\\hide")

#%% SECTION I
assert set(insurance['sex']) == {'female', 'male'}, "insurance.csv sex column format changed"
insurance['is_male'] = insurance['sex'] == 'male'
assert  set(insurance['smoker']) == {'no', 'yes'}, "insurance.csv smoker column format changed"
insurance['is_smoker'] = insurance['smoker'] == 'yes'
insurance.drop(['sex', 'smoker'],axis=1, inplace=True)
        
og_names = {'age': 'age',
              'is_male': 'sex',
              'bmi': 'bmi',
              'children': 'children',
              'is_smoker': 'smoker',
              'charges': 'charges'}

prez_names = {'age': 'age',
              'is_male': 'male %',
              'bmi': 'bmi',
              'children': 'children',
              'is_smoker': 'smoker %',
              'charges': 'charges'}
freq_cols = ['male %', 'smoker %']

#1
grp = insurance[['age', 'is_male', 'bmi', 'children', 'is_smoker', 'charges']
                ].rename(columns = prez_names
                ).groupby(insurance["region"])
ag = grp.agg(['mean', 'std']).round(2)
ag.drop([(i,  'std') for i in freq_cols], 
        axis=1,
        inplace = True)
freq_mean = [(i, 'mean') for i in freq_cols]
ag[freq_mean] = ag[freq_mean].apply(lambda c: (c*100).map("{:,.0f}%".format))#already rounded
print_full(ag)

#              age        male %    bmi       children       smoker %   charges          
#             mean    std   mean   mean   std     mean   std     mean      mean       std
# region                                                                                 
# northeast  38.83  13.85    49%  29.41  6.09     1.05  1.18      20%  13387.63  11126.07
# northwest  39.39  13.86    49%  29.24  5.19     1.19  1.19      19%  12609.90  11329.23
# southeast  38.94  14.15    53%  33.40  6.69     1.01  1.13      27%  14952.59  13933.80
# southwest  40.00  14.04    51%  30.69  5.70     1.18  1.27      18%  12530.71  11592.10

#%%
#2
insurance = insurance.astype({'is_male':int, 'is_smoker':int})
fig, axes = plt.subplots(nrows = 3, ncols = 3)
axes = iter([j for i in axes for j in i])
fig.suptitle("Column Histograms")
for c in insurance.columns:
    col = insurance[c]
    ax = next(axes)
    ax.set_title(c, size=14)
    ax.hist(col.values, weights=np.ones(len(col)) / len(col))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
for ax in axes:
    ax.axis('off')
fig.show()

corr = insurance.corr()
html_corr = corr.style.background_gradient(cmap='Blues', axis=None
                                           ).set_precision(2
                                           ).render()
with open("insurance_corr.html", 'w') as f:
    f.write(html_corr)
os.startfile("insurance_corr.html")

print(f"{insurance['is_smoker'].mean()*100:.0f}% Smokers")
print(f"BMI Avg: {insurance['bmi'].mean():.0f}, Median: {insurance['bmi'].median():.0f}")
sp_mn = insurance.loc[:,['age', 'children']].corr(method='spearman').iloc[0,1]
print(f"Spearman's correlation between Age and # Kids: {sp_mn:.2f} ")
#21% Smokers
#BMI Avg: 31, Median: 30
# Spearman's correlation between Age and # Kids: 0.05 

#%%
#3
insurance.groupby('is_male')['age'].agg(['mean', 'std']).round(2)
#there is no significant difference between the ages of men and women
#           mean    std
# is_male              
# 0        40.00  13.92
# 1        38.61  14.00

#%%
#4
insurance['is_smoker'].groupby(insurance['children'] > 0
                               ).agg(['mean', 'std']).round(2)
#there is no significant difference between the fraction which smokes between 
# parents and non-parents
#           mean   std
# has_children            
# False     0.22  0.41
# True      0.21  0.41

#%%
#5

region_enc = ColumnTransformer([
                            ("onehot_regions", OneHotEncoder(), ['region']),
                                ],
                                remainder='passthrough')
X = region_enc.fit_transform(insurance)
arr_col_names = region_enc.get_feature_names()
vif_cols = pd.Series([variance_inflation_factor(X, i) 
                          for i in range(X.shape[1])],
                     index = arr_col_names)
print(vif_cols)

#looking at heat map plot, there is a high degrees of correlation between
#being a smoker and charges, and a moderate correlation between age and charges.
#However, looking at the Variance inflation factor, only the linearly dependant
#region encodings are highly colinear.
# onehot_regions__x0_northeast     8.767287
# onehot_regions__x0_northwest     9.668099
# onehot_regions__x0_southeast    12.691814
# onehot_regions__x0_southwest    10.642860
# age                              1.397391
# bmi                              1.219409
# children                         1.018285
# charges                          4.065111
# is_male                          1.015066
# is_smoker                        3.613976

for d_ix in [ix for ix,n in enumerate(arr_col_names) if 'onehot_regions' in n]:
    X2 = np.delete(X, d_ix, axis=1)
    print(pd.Series([variance_inflation_factor(X2, i) 
                         for i in range(X2.shape[1])],
                    index = np.delete(arr_col_names,d_ix)))
    
#Removing a single encoded region leaves Age and BMI highly colinear 
#with the rest of the in dataset. 
#eg. 
# onehot_regions__x0_northeast     1.733812
# onehot_regions__x0_northwest     1.816909
# onehot_regions__x0_southeast     2.142458
# age                              9.715703
# bmi                             11.067182
# children                         1.843932
# charges                          8.014977
# is_male                          2.022580
# is_smoker                        4.084595

#%%
#6
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm

#Using a linear model as with high VIF for charges, will get high accuracy.
#The colinearity of some factors suggests regularization, that some features 
#seem correlated while no coefficents are expected to be 0 advocates ridge 
#regression. The regularization factor will be set to both the default and 
#as suggested by a Bayesian approach.

y_ix = arr_col_names.index("charges")
y = X[:, y_ix]
reg_X = np.delete(X, [0,y_ix], axis=1)

lin_reg = LinearRegression().fit(reg_X,y)
ridge_reg = Ridge().fit(reg_X,y)
residual_var_prior = 1000**2 #not true prior as looked at graph and guessed SD=1k
coef_var_prior = 10#doesn't make sense as var factor dependant
bayes_reg = Ridge(alpha = residual_var_prior/coef_var_prior).fit(reg_X,y)
print(f"Linear: {lin_reg.score(reg_X, y):.2f}"\
      f" vs Ridge: {ridge_reg.score(reg_X, y):.2f}"\
      f" vs Bayes: {bayes_reg.score(reg_X, y):.2f}R^2")
coef_summary = pd.DataFrame(zip(np.append(lin_reg.coef_, lin_reg.intercept_),
                                np.append(ridge_reg.coef_, ridge_reg.intercept_),
                                np.append(bayes_reg.coef_, bayes_reg.intercept_),), 
                   columns = ['Linear', 'Ridge', 'Bayes'], 
                   index = arr_col_names[1:y_ix] + arr_col_names[y_ix+1:] + ['intercept']
                   ).round(2)
print(coef_summary)
# Linear: 0.75 vs Ridge: 0.75 vs Bayes: 0.10R^2
#                                 Linear     Ridge    Bayes
# onehot_regions__x0_northwest   -589.03   -575.84    -1.66
# onehot_regions__x0_southeast  -1269.82  -1241.88     3.76
# onehot_regions__x0_southwest  -1085.35  -1071.08    -2.51
# age                             263.68    263.59   169.77
# bmi                             329.17    328.71    91.49
# children                        533.55    532.51     6.49
# is_male                           5.15     14.69     4.22
# is_smoker                     23738.38  23595.18    39.45
# intercept                    -11947.28 -11917.47  3901.70

#With such poor performance to the Bayesaian approach, it will be excluded. 
#The realized coefficents for OLS and L2 Norm appear similar so will use OLS 
#going forward, to aid interpretability. 

reg_X_int = np.append(reg_X, np.ones((len(reg_X),1)), axis=1)
ols = sm.OLS(y, reg_X_int)
ols_result = ols.fit()
print(ols_result.summary(xname = list(coef_summary.index)))
# ================================================================================================
#                                    coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------
# onehot_regions__x0_northwest  -589.0285    557.419     -1.057      0.291   -1682.880     504.823
# onehot_regions__x0_southeast -1269.8151    559.391     -2.270      0.023   -2367.536    -172.094
# onehot_regions__x0_southwest -1085.3520    555.302     -1.955      0.051   -2175.050       4.346
# age                            263.6798     13.828     19.068      0.000     236.544     290.816
# bmi                            329.1746     32.418     10.154      0.000     265.558     392.791
# children                       533.5534    160.623      3.322      0.001     218.354     848.753
# is_male                          5.1513    384.660      0.013      0.989    -749.686     759.989
# is_smoker                     2.374e+04    470.627     50.440      0.000    2.28e+04    2.47e+04
# intercept                    -1.195e+04   1132.255    -10.552      0.000   -1.42e+04   -9725.399
# ==============================================================================
# Omnibus:                      207.125   Durbin-Watson:                   2.051
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              440.214
# Skew:                           1.154   Prob(JB):                     2.56e-96
# Kurtosis:                       5.279   Cond. No.                         312.
# ==============================================================================
#Limited Skew and Kurtosis implies treating as linear in factors is valid;
#But the Jarque-Bera test rejects that the residuals are normal.
pred_y = ols_result.predict(reg_X_int)
fig,(ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
ax.scatter(pred_y,
            y - pred_y,
            alpha = 0.8)
ax.set_ylabel("Residual Cost")
ax2.scatter(pred_y,
            y,
            alpha = 0.8)
ax2.set_ylabel("Actual Cost")
ax2.axhline(y=50000, c='black')
ax2.set_xlabel("Predicted Cost")
fig.tight_layout()
fig.show()
#Looking at plots there’s 2 distinct regimes: for predicted cost < 20k true costs
#are mostly linear with predicted costs, with right skewed residuals and with 
#a hinge at 0 where negative costs are predicted. While for Predicted cost > 20k
#there’s 2 separate ~linear groups.

hc_y = y[y>=20000]
hc_X = reg_X_int[y>=20000]
hc_ols = sm.OLS(hc_y, hc_X)
hc_ols_result = hc_ols.fit()
print(hc_ols_result.summary(xname = list(coef_summary.index)))
# ================================================================================================
#                                    coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------
# onehot_regions__x0_northwest   568.0968   1238.729      0.459      0.647   -1874.774    3010.968
# onehot_regions__x0_southeast  -706.6318   1177.568     -0.600      0.549   -3028.889    1615.625
# onehot_regions__x0_southwest  1236.6523   1272.502      0.972      0.332   -1272.823    3746.128
# age                            127.9051     30.055      4.256      0.000      68.634     187.176
# bmi                            991.1586     79.896     12.406      0.000     833.598    1148.720
# children                       267.4571    393.019      0.681      0.497    -507.608    1042.522
# is_male                        198.4184    876.956      0.226      0.821   -1531.009    1927.846
# is_smoker                     9143.5495   1025.407      8.917      0.000    7121.365    1.12e+04
# intercept                     -1.09e+04   3007.059     -3.626      0.000   -1.68e+04   -4974.135
# ==============================================================================
# Omnibus:                        1.060   Durbin-Watson:                   1.995
# Prob(Omnibus):                  0.589   Jarque-Bera (JB):                0.728
# Skew:                          -0.076   Prob(JB):                        0.695
# Kurtosis:                       3.248   Cond. No.                         398.
# ==============================================================================
#Sex is not a driver of which group in high cost segment an individual is placed in.

#6.1 The T-test fails to find an impact on charges from being Male vs. Female.
#6.2 Being a Smoker has a highly significant increase in cost, $24k.
# 6.3 Being 1 year older has a statistically significant increase in cost, $260/year

#%%
#7 
cutoff_pred_y = min(pred_y[y>=50000])
print(f"{sum(y>50000)} Points >=50k")
print(f"{sum(pred_y >= cutoff_pred_y)} Values predicted >= 50k with OLS")
print(f"Avg cost ${np.mean(y[pred_y >= cutoff_pred_y]):,.2f} for excluded values vs. ${np.mean(y):,.2f} avg cost")
#using High Segment Costs
hc_pred_y = hc_ols_result.predict(hc_X)
hc_cutoff_pred_y = min(hc_pred_y[hc_y>=50000])
print(f"{sum(hc_pred_y >= hc_cutoff_pred_y)} Values predicted >= 50k with OLS for High Cost group only")
print(f"Avg cost ${np.mean(hc_y[hc_pred_y >= hc_cutoff_pred_y]):,.2f} for excluded values vs. ${np.mean(hc_y):,.2f} for High Cost Group")
# 4 Points >=50k
# 111 Values predicted >= 50k with OLS
# Avg cost $37,940.03 for excluded values vs. $13,408.08 avg cost
# 59 Values predicted >= 50k with OLS for High Cost group only
# Avg cost $44,886.17 for excluded values vs. $34,182.07 for High Cost Group as a whole

#The naive logit glm function is either missing at least 25% of points >=25 
#or would exclude >10% of the original data set. 
#Need some non-linear prediction mechanism, as a seperate regression on high cost
#produces more accurate answers: either SVM-Regression or spline methods.

#To evaluate the effectiveness you would minimize the loss of excluding 
#non-high risk patients in reduce premiums vs. the additional payouts to patients >50k.
#There's some confusion in this sinario, of using the emperical max-cost as part
#of the loss function; modeling the probability of costs >50k instead of modeling 
#expected cost directly implies a concern that the maximum possible cost 
#is not well defined/ given enough samples will be much larger than the emperical.

#Looking solely at the regression for the high cost group (final model should be at least this accurate)
#Assuming profits on premiums of 5% ($34.1k*0.05 = ~1.7k profit),  and that expected costs for those >=50k 
#are 50% higher than the empirical average causes a cost of $51k (np.mean((y[y>=50000]))*1.5 -  34.1k*1.05)
#gives a probability of being in high cost group of (1-p)*1.7 = p*51 implies
#p=3% will make the costs from Type I and type II errors balance


n_hc = sum(hc_y >= 50000)
t1l, t2l, t1c, t2c  = [], [], [], []
for i in range(n_hc+1):
    try:
        hc_cutoff_pred_y = sorted(hc_pred_y[hc_y>=50000])[i]
    except:
        hc_cutoff_pred_y = sum(hc_pred_y)
    nt2 = i#ignore a truth
    pt2 = nt2/n_hc
    found_costly = n_hc - i
    say_costly = sum(hc_pred_y >= hc_cutoff_pred_y)
    nt1 = say_costly - found_costly #false positive
    if say_costly == 0:
        pt1 = 0
    else:
        pt1 = nt1/say_costly
    prob_say_costly = say_costly/len(hc_pred_y)#not actual decision prob
    # print(f"{pt2*100:.0f}% High Cost missed (type II) and {pt1*100:.0f}% of Positives False(TYpe I)")
    # print(f"${int(nt2*39700):,} Type II & ${int(nt1*2250):,} Type I costs = ${nt2*39700 + nt1*2250:,} total")
    t1l += [f"{pt1*100:.0f}%"]
    t2l += [f"{pt2*100:.0f}%"]
    t1c += [float(nt2*39700)]
    t2c += [float(nt1*2250)]
    
dis_df = pd.DataFrame(zip(t1l, t2l, t1c, t2c ), 
                   columns = ['Type I%', 'Type II%', 'Type I Costs', 'Type II Costs'],
                   )
dis_df['Total Costs'] = dis_df['Type I Costs'] + dis_df['Type II Costs']
with pd.option_context('display.float_format', "${:,.0f}".format):
    print(dis_df.to_string(index=False))

#Possible points in empirical trade-off
# Type I% Type II%  Type I Costs  Type II Costs  Total Costs
#     93%       0%            $0       $123,750     $123,750
#     93%      25%       $39,700        $87,750     $127,450
#     94%      50%       $79,400        $67,500     $146,900
#     50%      75%      $119,100         $2,250     $121,350
#      0%     100%      $158,800             $0     $158,800
#%%SECTION II
from datetime import timedelta
from datetime import date
import matplotlib.dates as mdates

new_feat_d = date(month=9, day=5, year=2018)

conversion = pd.read_csv("data\conversion_rates.csv",
                         parse_dates = ['date'])

fig,ax = plt.subplots()
ax.hist(conversion['date'])
weekends = [d for d in conversion['date'] if d.weekday() in (5,6)]
weekdays = [d for d in conversion['date'] if d.weekday() not in (5,6)]
ax.axvline(x=new_feat_d, c='black')
ax.set_title("Histogram of click dates vs. Introduction of new Feature")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
fig.show()

X = conversion.drop(['reached_end'], axis=1)
y = conversion['reached_end']

class FuncTrans_Named(FunctionTransformer):
    def __init__(self, func, feat_name = ''):
        super().__init__(func)
        self.feat_name = feat_name
    
    def get_feature_names(self):
        return np.array([self.feat_name])
    
def _dt2float(df):
    "df: Series of datetime objects"
    return (df['date'] - min(df['date'])
            ).apply(lambda i: i/timedelta(days=1)).values.reshape(-1,1)

def _isweekend(df):
    "df: Series of datetime objects"
    return df.apply(lambda d: d.iloc[0].weekday() in (5,6), axis=1).values.reshape(-1,1)

def _is_newfeat(df, new_feat_d  = new_feat_d):
    "df: Series of datetime objects"
    return df.apply(lambda d: d.iloc[0] >= new_feat_d, axis=1).values.reshape(-1,1)
    
date_feats = FeatureUnion([("Linear_Date", FuncTrans_Named(_dt2float)),
                            ("is_weekend", FuncTrans_Named(_isweekend)),                            
                            ("has_new_feature", FuncTrans_Named(_is_newfeat,
                                                                feat_name="    <--")),                            
                            ])

origin_enc = ColumnTransformer([
                            ("onehot", OneHotEncoder(drop='first'), ['came_from']),
                            ("dt", date_feats, ['date']),
                                ],
                                remainder='passthrough')

reg_X = origin_enc.fit_transform(X)
reg_X = np.append(reg_X, np.ones((len(reg_X),1)), axis=1)
logit = sm.Logit(y.values.reshape(-1, 1), reg_X)
logit_result = logit.fit()
col_names = origin_enc.get_feature_names() + ['intercept']
print(logit_result.summary(xname = col_names))
#Based on high z-score for having a new feature, we reject the null that feature
#has no impact at p=0.01 level; the positive coefficient means the feature change 
#was helpful in improving conversion rates.
# ================================================================================================
#                                    coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------------------------
# onehot__x0_Insurance Site A      0.4721      0.395      1.194      0.232      -0.303       1.247
# onehot__x0_Insurance Site B      0.6884      0.354      1.946      0.052      -0.005       1.382
# onehot__x0_Insurance Site C      0.4225      0.333      1.268      0.205      -0.231       1.076
# dt__Linear_Date__               -0.1232      0.128     -0.966      0.334      -0.373       0.127
# dt__is_weekend__                -0.1898      0.274     -0.693      0.488      -0.727       0.347
# dt__has_new_feature__    <--     1.4589      0.524      2.784      0.005       0.432       2.486
# male                             0.1025      0.253      0.406      0.685      -0.393       0.598
# age                             -0.0465      0.054     -0.864      0.387      -0.152       0.059
# has_insurance                   -0.1870      0.255     -0.732      0.464      -0.688       0.314
# intercept                        0.7140      1.762      0.405      0.685      -2.739       4.167
# ================================================================================================

vif_cols2 = pd.Series([variance_inflation_factor(reg_X, i) 
                          for i in range(reg_X.shape[1])],
                      index = col_names)
print(vif_cols2)
# onehot__x0_Insurance Site A       1.518568
# onehot__x0_Insurance Site B       1.394263
# onehot__x0_Insurance Site C       1.432960
# dt__Linear_Date__                 4.382285
# dt__is_weekend__                  1.043243
# dt__has_new_feature__    <--      4.289369
# male                              1.020866
# age                               1.216348
# has_insurance                     1.036128
# intercept                       200.334371
#The high VIF of intercept has inflated the R^2, but since intercept is 
#uncorrelated with the conditional variable about when the new feature was 
#introduced this has no effect on validity of coefficient t-test.


#%%SECTION III
names = pd.read_csv("data\\names_id_age.csv")
lead = pd.read_csv("data\\lead_sale_stats.csv")
nl_drop = sum(lead['lead_id'].isna())
lead = lead[~lead['lead_id'].isna()]
lead['lead_type'] = lead['lead_id'].apply(lambda i: re.search("[a-c]", i
                                                              ).group().upper())
lead['lead_id'] = lead['lead_id'].apply(lambda i: int(re.search("\d+", i).group()))

nn_drop = len(set(pd.unique(lead['lead_id'])) - set(pd.unique(names['lead_id'])))
print(f"Dropped {nl_drop} rows from lead_sale_stats and {nn_drop} from names_id_age")
# Dropped 4 rows from lead_sale_stats and 0 from names_id_age

df = lead.merge(names, on='lead_id', how='left')
for i in df.columns:
    if i[-2:] == '_y':
        dup = i[:-2] + "_x"
        assert all(df[i] == df[dup]), f"{i}, {dup}"
        df.drop(dup, inplace=True, axis=1)
        df.rename({i:i[:-2]}, inplace=True, axis=1)

def _add_barplot_labels(ax, r_lst, fmt = lambda h: f"{h:.0f}%"):
    """label data within barplots 
    ax: axis
    r_lst: [ax.bar() object]
    fmt: lambda to format height
    """
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    _, ax_h = bbox.width, bbox.height
    ax_h *= fig.dpi
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
                    
per_closed = df.groupby('lead_type')['bought_policy'].mean() * 100
avg_sale = df.groupby('lead_type')['policy_amount'].mean()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
fig.suptitle("Increase sales by focusing on type 'C' Leads", size=20)

colors =['gold', 'darkorange', 'orangered'] 
rect1 = ax1.bar(per_closed.index, per_closed.values, color=colors)
_add_barplot_labels(ax1, [rect1], fmt = lambda h: f"{h:.0f}%")

ax1.set_ylabel("% BUY")
ax1.set_xlabel("LEAD TYPE")
ax1.set_title("Conversion % by Lead Type")

rect2 = ax2.bar(avg_sale.index, avg_sale.values, color=colors)
_add_barplot_labels(ax2, [rect2], fmt = lambda h: f"${h:,.0f}")
ax2.set_ylabel("EXPECTED SALE $")
ax2.set_xlabel("LEAD TYPE")
ax2.set_title("Average Sales Amount by Lead Type")

fig.show()
#It turns out that type ‘C’ leads are the most profitable leads.


