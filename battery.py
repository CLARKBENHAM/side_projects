import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import timeit
import math
#import itertools

#hourly_rates = np.random.randint(0, high = 10, size = 110)

sz = 10
#print(timeit.timeit("hourly_rates.append([False])", setup = "import random; import numpy as np; hourly_rates =np.random.randint(0, high = 10, size = 10)", number = 100000), \
#      timeit.timeit("list(np.random.randint(0, high = 10, size = 10))", setup = "import random; import numpy as np", number = 100000))
#list(np.rand faster 1.3781812000088394 vs 0.3674951998982578
#swap to xrange in python 2, range in python 3
hourly_rates = np.random.randint(0, high = 100, size = 744)
blocked_rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])#xrange for py2
#%%


#%% 
#Scenario: Battery must start, charge for 4 hours, completely discharge
#0.928313700016588
def plot_bs(blocked_rates):
    #blocked_rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])#xrange for py2
    increasing = np.array(blocked_rates[:-1] < blocked_rates[1:], dtype = bool)#does NOT work with lists
    inflect_up = increasing[:-1] < increasing[1:]#not increasing then is increasing; apparent have to wrap in list
    inflect_up = np.concatenate(([increasing[0]], inflect_up, [False]))#buy on first hour if increases after that, don't buy on last
    
    inflect_down = increasing[:-1] > increasing[1:]
    inflect_down = np.concatenate(([False], inflect_down, [increasing[-1]]))#can't sell while empty; sell on last if increased up to that point
    possible_profits = blocked_rates[inflect_down] - blocked_rates[inflect_up]
    profits = np.zeros(len(hourly_rates)//24 + 1)
    #which_trade = np.zeros(len(hourly_rates)//24 + 1, dtype = int)
    day = 0
    
    for i in range(len(hourly_rates)//24 + 1):
        num_possible = np.sum(inflect_up[i*24:i*24+24])
        profits[i] = np.max(possible_profits[day:day + num_possible], initial= 0)#can only charge once per day(could discharge twice)
        
    #    which_trade[i] = list(possible_profits[day:day + num_possible]).index(profits[i])#plotting
    #    which_trade[i] = i*24 + np.flatnonzero(np.array(inflect_up[i*24:i*24+24]) == True)[which_trade[i]]
        day += num_possible
    
    #plotting
    a = ["black" if i == j else "green" if i else "red" for i,j in zip(inflect_up, inflect_down)]
    plot_lmt = 40
    dayz = plot_lmt//24 #math.ceil(plot_lmt/24)
    plt.bar(range(len(blocked_rates[:plot_lmt])), blocked_rates[:plot_lmt], color = a)
    for i in range(len(blocked_rates[:plot_lmt])//24 + 1):#deliniates days
        plt.axvline(x = i*24)#change type? 
    #plt.scatter(which_trade[:dayz], profits[:dayz] + blocked_rates[which_trade[:dayz]])
plot_bs(blocked_rates[-24:])
#%%#%%
#qwer
def block(hourly_rates, n, ret_array = True):
    if ret_array:
        return np.array([np.mean(hourly_rates[i:i+n]) for i in range(len(hourly_rates) - n)])
    else:
        return [sum(hourly_rates[i:i+n])/n for i in range(len(hourly_rates) - n)]
       
def bs_2_seriesBad(mins, maxs, cycles = 1, sell_twice = False, i = None, prices = None):
    "pass in a sequence of buying prices PRIOR to time of selling prices; \
    gets best buying, selling locs; profits"
    #issue of different lengths
    #below occurs when prices only increase or decrease all day; will only be selling once regardless if have "selltwice" twice
    if cycles != 1:
        raise Exception('bs_2_series currently assumes can only buy/sell at 1 point; hr_cap must be 0 not ' + str(hr_capacity))

    if len(mins) ==0:
#        print("min len = 0")
        mins = [min(prices[i*24:i*24+24])]#since looking at the actual prices per mwh have to give time to sell 
        sell_twice = False
    if len(maxs) == 0:    
#        print("max len = 0")
        maxs = [max(prices[i*24:i*24+24])]#changed
        sell_twice = False
        
    min_b = min(mins)
    max_s = max(maxs)
    if isinstance(mins, list):    
        bi = mins.index(min_b)
        si = maxs.index(max_s)
    else:
        bi = min(np.where(mins == min_b)[0])#where returns a tuple
        si = max(np.where(maxs == max_s)[-1])#in case of multipl equal indicies, sell at later dates
    #print(maxs, si, mins, bi, "\n")
    best_buy= max(maxs[bi:], default = -2*32-1) - min_b
    best_sell= max_s - min(mins[:si+1], default = 2*32-1)#have [:si+1] since sells occur after buys
    
    if best_buy > best_sell:
        todays_profit = best_buy
        if isinstance(maxs, list):
            ix = [bi, maxs.index(max(maxs[bi:]))]
        else:
            ix= [bi, np.where(maxs == max(maxs[bi:]))[0][-1]]
    else:
        todays_profit = best_sell
        if isinstance(mins, list):
            ix = [mins.index(min(mins[:si+1])), si]
        else:
            ix = [np.where(mins== min(mins[:si+1]))[0][0], si]
#     = max(best_buy, best_sell)#can sell at or later than time of buy
#    if sell_twice:
#        profit2, min2, s2i = bs_2_series(mins[:bi] + mins[bi+1:], maxs[:si] + maxs[si+1:])
#        later_sale = max(si, s2i)
#        return todays_profit + profit2, min(mins[later_sale+1:], default = 2**32-1), later_sale#default if sold at end of period
    return todays_profit, ix#since can check up to index of selling value 
#returns lowest buying price after sold, highest selling price
      
def bs_2_series2(minprx, maxprx):
    "given 2 list of buying, selling prices(buying occuring before selling at same index)\
    return max profit, indxs. "
    prft = 0
    indxs = [0,0]
    if len(minprx) == 0:#if only increases will be included
        return prft, indxs
    for j in range(len(maxprx)):
        for i in range(j+1):
            if maxprx[j] - minprx[i] > prft:
                prft = maxprx[j] - minprx[i]
                indxs = [i,j]
    return prft, indxs
#unittest.main()
    
def bs_2_series(minprx, maxprx):
    "given 2 list of buying, selling prices(buying occuring before selling at same index)\
    return max profit, indxs."
    if len(minprx) == 0:#if only increases will be included
        return 0, [0,0]
    mxmax = maxprx[-1]
    indx = [0] * len(maxprx)
    for i,val in enumerate(maxprx[-2::-1]):
        if val > mxmax:
            mxmax = val
            indx[len(maxprx) - i - 2] = len(maxprx) - i - 2
        else:
            indx[len(maxprx) - i - 2] = indx[len(maxprx) - i - 1]#carry previous best forward
        maxprx[len(maxprx) - i - 2] = mxmax
    out_ix = [0,0]
    prft = 0
    for i, val in enumerate(minprx):
        if (maxprx[i] - val) > prft:
            prft = maxprx[i] - val
            out_ix = [i, indx[i]]
    return prft, out_ix

t =  [int(i) for i in "79 71 63 62 52 63 36 27  9 97 78 69 37 67  2 48 13 23 99 65 35 14 52 49".split(" ") if i != ""]
f = list(range(24))
#%%
def n_ahead_ave(series, n, ret_array = False):
    "take list(array) and returns list(array) w/ len - n average of current AND **N-1** elements\
    in front of each index; drops last n elements. is more efficent starting around n>20\
    has side effects of changing underlying. Potential issue w/ storing floats"
    if n < 22:
        if not ret_array:
            return [sum(series[i:i+n])/n for i in range(len(series) - n)]
        else: 
            return np.array([np.mean(series[i:i+n]) for i in range(len(series) - n)]) 
    saved = series[:n]
    series[0] = sum(series[:n])/n
    for i in range(1, len(series)-n):
        saved[i%n] = series[i]
        series[i] = series[i-1] + (series[i+n-1] - saved[(i-1)%n ])/n#  - series[i-2] - series[i+n-1]/n
#        print(series[i], saved[i%n])
#        print(saved,series[i:i+n+1], "\n")
    return series[:-n]
##n_ahead_ave(hourly_rates, 4) == blocked_rates
#n_ahead_ave(list(range(10)), 3)   
##times = [0]*50
#times2 = [0]*50
#for i in list(range(1,50)):
#    times2[i] = timeit.timeit(f"[sum(tezt[j:j+{i}])/{i} for j in range(len(tezt) - {i})]", 
#    setup = "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; \
#    import random; import math; tezt = np.random.randint(0, high = 10, size = 744); \
#    from __main__ import best_of_block2;from __main__ import n_ahead_ave;",
#    number = 100)

#hourly_rates = [float(i) for i in np.random.randint(0, high = 10, size = 300)]
#sz = 30
#blocked_rates = block(hourly_rates, sz, ret_array = True)
#for i,j in zip(np.array(n_ahead_ave(hourly_rates, sz, ret_array = True)), blocked_rates):
#    print(i,j)
#n_ahead_ave(hourly_rates, 24, ret_array = True) blocked_rates
        
#grib
def goto_best_sell(sell_arr, buy_arr, i):#pass in refernce to array would be faster than copy a iterater?
    "will go through iterator of boolean array till condition is true; but have NOT returned a buy before it index"
    j = i
    try:
        while not sell_arr[i] and not buy_arr[i]:
            i += 1            
    except:
        return -1#if not a selling point in last day; sell at end; what if always declining?
    if sell_arr[i]:#price has been increasing since last night, 
        return i
    return j - 1#got to a buy opportunity before a sell oportunity, will make prices 0

def get_maxmin_n(rates = hourly_rates, n = 4, get_max = True):
    "get into 24 hr blocks; get max n values"
    num_days = math.ceil(len(rates)/24)
    rates = rates[:-len(rates)%24]#must be |24
    maxmin_n = []*num_days
    for i in range(num_days):
        val_loc = sorted(zip(rates[i*24:i*24+24], list(range(i*24,i*24+24))), key = lambda pair: pair[0])
        if get_max:
            maxmin_n[i] = [val for val, _ in val_loc[-n:]]
        else:
            maxmin_n[i] = [val for val, _ in val_loc[:n]]
    return maxmin_n
#        rates[i*24:i*24+24].sort()#modifies rates

def test_equal(*argv):
    lns = [len(arg) for arg in argv]
    for i in range(1,len(argv)):
        if len(argv[i]) != len(argv[i-1]):
            print(f"have mismatching lengths {[len(arg) for arg in argv ]}")
            return 
    print(f"The Following errors")
    errors = []
    for i, vals in enumerate(zip(*argv)):
        if any([val != vals[0] for val in vals ]):
            print(f"at index {i}, got {vals}")
            errors.append(i)
    print("FIN.\n\n")
    return errors

def plot_tf(rates, buy, sell, plot_lmt =24):
    "takes 2 TF bool arrs of when to buy, sell and plots them"
    clrs = ["green" if buy[i] else "red" if sell[i] else "black" for i in range(len(rates[:plot_lmt]))]
#    fig, ax = plt.plot()
    plt.bar(range(len(rates[:plot_lmt])), rates[:plot_lmt], color = clrs)
    for i in range(len(rates[:plot_lmt])//24 + 1):#deliniates days
        plt.axvline(x = i*24)#change type? 
    plt.plot()
    
def plot_bs_loc(rates, buy_loc, sell_loc, plot_lmt=24):
    "take b/s locations and plot them on graph"
    clrs = ["green" if i in buy_loc else "red" if i in sell_loc else "black" for i in range(len(rates[:plot_lmt]))]
#    fig, ax = plt.plot()
    plt.bar(range(len(rates[:plot_lmt])), rates[:plot_lmt], color = clrs)
    for i in range(len(rates[:plot_lmt])//24 + 1):#deliniates days
        plt.axvline(x = i*24)#change type? 
    plt.plot()
#Trying to get a faster implementation of above
#pre-define inflect_up?Grib
#9.513454200001433
#hourly_rates = np.random.randint(0, high = 100, size = 744)#evaluate months at a time
#hourly_rates = np.random.choice(1000, 48, replace = False )

def get_loc_minmax1(rates):
    "taking series returns 2 T/F bool arrays for when gets loc min, max"
    inflect_up, inflect_down = np.zeros(len(rates), dtype = bool), np.zeros(len(rates), dtype = bool)
    increasing = np.array(rates[:-1] <= rates[1:])#if trades at same level "increasing" as will only buy if declined then started increasing; or was increasing and then decreasing.
    inflect_up[1:-1] = np.array(increasing[:-1] < increasing[1:])#not increasing then is increasing; apparent have to wrap in list
    inflect_down[1:-1] = np.array(increasing[:-1] > increasing[1:])
    inflect_up[np.array([0,-1])] = [increasing[0], False]#buy on first hour if increases after that, don't buy on last 
    inflect_down[np.array([0,-1])] = [False, increasing[-1]]#can't sell while empty; sell on last if increased up to that point
    return inflect_up, inflect_down
                                                                                                                           
def get_loc_minmax2(rates, locs = True, sep_days = False):#sells at end of platea
    if sep_days:
        num_days = math.ceil(len(rates)/24)
        minmax_loc= dict()# {i:[] for i in range(num_days)}#which is faster?
        for i in range(num_days):
            minmax_loc[i] = get_loc_minmax2(rates[i*24:min(i*24+24, len(rates))], sep_days = False, locs = locs)
        return minmax_loc
    else:    
        if locs:
            buy_loc = []
            sell_loc = []
            increasing = False
            decreasing = True
            for i in range(1, len(rates)):
                if rates[i-1] <= rates[i]:  
                    increasing = True
                    if decreasing:
                        buy_loc.append(i-1)#buy loc
                    decreasing = False#rates[i-1] > rates[i]
                elif rates[i-1] > rates[i]:
                    decreasing = True
                    if increasing:
                        sell_loc.append(i-1)#sell loc
                    increasing = False
            if increasing:#already bought, sell on last day
                sell_loc.append(len(rates)-1)
            return buy_loc, sell_loc
        else:
            buy = [False]*len(rates)
            sell = [False]*len(rates)
            increasing = False
            decreasing = True
            for i in range(1, len(rates)):
                if rates[i-1] <= rates[i]:  
                    increasing = True
                    if decreasing:
                        buy[i-1] = True
                    decreasing = False
                elif rates[i-1] > rates[i]:
                    decreasing = True
                    if increasing:
                        sell[i-1] = True
                    increasing = False
            if increasing:#already bought, sell on last day
                sell[-1] = True
            return buy, sell


def loc_abs_minmax(rates, into_days = False):
    "Buy at either the absolute min and highest price after that; or the lowest price before absolute max" 
    if into_days:
        num_days = math.ceil(len(rates)/24)
        by_day = {i:get_loc_minmax3(rates[i*24:min(i*24+24, len(rates))]) for i in range(num_days)}
        return by_day
    else:
        best_low_hi = [2**32-1, -2**32]
        low_best_hi = [2**32-1, -2**32] 
        best_low_hi_ix, low_best_hi_ix = [0,0], [0,0]
        start = 0
        while rates[start] >= rates[start+1]:#only start looking at values once you've had a minimum.
            start += 1
        for i in range(start, len(rates)):
    #        print(rates[i], best_low_hi[0])
            if rates[i] < best_low_hi[0]:#found new best low
                best_low_hi[0] = rates[i]
                best_low_hi[1] = rates[i]#nets to 0 if min val is last in series
                best_low_hi_ix = [i, i]
    
            if rates[i] > low_best_hi[1]:#found new best high
                low_best_hi[1] = rates[i]
                low_best_hi[0] = best_low_hi[0]#become best low seen up to this point; is fixed till best high changed
                low_best_hi_ix = [best_low_hi_ix[0], i]
                
            if rates[i] > best_low_hi[1]:#better high after best low
                best_low_hi[1] = rates[i]    
                best_low_hi_ix[1] = i
        options = [i[1] - i[0] for i in [low_best_hi, best_low_hi]]
        day_best = max(options)
        if options[0] == day_best:
            return low_best_hi_ix
        else:
            return best_low_hi_ix
#%%
            #qwer
def best_of_block1T(rates = hourly_rates, into_blocks = True, hr_capacity = 4):
    #Version which will be tested
    #should rename hr_capacity
    if into_blocks:
        rates = block(rates, hr_capacity)    
    num_days = int(math.ceil(len(rates)/24))   
    profits = np.zeros(num_days)
    indxs = [[0,0]]*num_days
    inflect_up, inflect_down = np.zeros(24, dtype = bool), np.zeros(24, dtype = bool)
    for i in range(num_days):#need to adjust for when len(rates)%24!=0
        day_ix = slice(i*24, min(i*24+24, len(rates)))#  
        inflect_up, inflect_down = get_loc_minmax1(rates[day_ix])
        up_indxs = np.where(inflect_up)[0]#faster like this vs. usign T/F arr as indxs; 3.7 vs. 4.8
        down_indxs = np.where(inflect_down)[0]
        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs])#slices dont include end; add extra buying location, but not selling
#        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs], cycles = cycles_per_day)#slices dont include end; add extra buying location, but not selling
        indxs[i] = [i*24 + up_indxs[day_minmax_ix[0]], i*24 + down_indxs[day_minmax_ix[1]]]
#        print(profits[i], indxs[i])
#    i = num_days - 1
#    profits[i], indxs[i] = bs_2_series(rates[i*24:][inflect_up[i*24:]], rates[i*24:][inflect_down[i*24:]])#
    return profits, indxs
best_of_block1T(rates = hourly_rates)#need to get 70


def best_of_block2T(rates = hourly_rates, into_blocks = True, hr_capacity = 4): 
    #DOES NOT go across 24hr blocks
    #version which will be tested
    "Python2 version when no longer supported; needs to take out numpy"
    #creats1 dict outside loop and uses loc
    if into_blocks:
        rates = block(rates, hr_capacity)#[sum(hourly_rates[i:i+4])/4 for i in range(len(hourly_rates) - 4)]#write better that increments average   
    num_days = int(math.ceil(len(rates)/24))
    profits = [0]*num_days#{i:0 for i in range(num_days + 1)}
    indxs = [[0,0]]*num_days
    for i in range(num_days):
        day_ix = slice(i*24, min(i*24+24, len(rates)))
        min_loc, max_loc = get_loc_minmax2(rates[day_ix], sep_days = False, locs = True)
#        print(min_loc, max_loc, rates[day_ix][min_loc], rates[day_ix][max_loc])
        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][min_loc], rates[day_ix][max_loc])#slices dont include end; add extra buying location, but not selling
        indxs[i] = [i*24 + min_loc[day_minmax_ix[0]], i*24 + max_loc[day_minmax_ix[1]]]
    return profits, indxs
best_of_block2T(rates = hourly_rates)#should be 70
#unittest.main()#fdsa


#qwer
def best_of_day_r(rates = hourly_rates, cycles_per_day = 1, hr_capacity = 4, into_blocks = True):
    "gets best buy/sell of ALL rates, recurs w/o those indexs to recalculate. \
    Can alternate Buying, Selling. cycles_per_day is number of alternations; hr_cap is \
    how many hours can store for"
    #O(24*hr_capacity); has to copy over each array for each recursion.
    if into_blocks:
        profit, indxs =  best_of_day_r(block(rates, hr_capacity), cycles_per_day = cycles_per_day, hr_capacity = 1, into_blocks = False)
        b_locs = [j for i in indxs[::2] for j in range(i,i+hr_capacity)]
        s_locs = [j for i in indxs[1::2] for j in range(i,i+hr_capacity)]
        return profit, [var for pair in zip(b_locs,s_locs) for var in pair]#averages are "combined" into the present hour from future hours.
    #above WILL go wrong
    
    #doesn't change w/ hr_capacity changing
    assert(hr_capacity == 1)
    assert(cycles_per_day == 1)#cycling per day is wrong as could buy twice then sell twice
    best_prft, period_prft, cumsum_prft = 0,0,0
    best_indx, period_indx = [0,0], [0,0]
    passed_loc_min = False
    i = 0
    while i < len(rates)-1:
        if not passed_loc_min:
            try:
                while rates[i] >= rates[i+1]:#only start looking at values once you've had a minimum.
                    i += 1
                period_indx[0] = i
                passed_loc_min= True
            except:#case where rates only decreased from i; reached end
                break
        cumsum_prft += rates[i+1] - rates[i]
        if cumsum_prft > period_prft:#better loc
            period_prft = cumsum_prft
            period_indx[1] = i+1#are selling at later loc
        if cumsum_prft <= 0:#end of period
            if period_prft > best_prft:
                best_prft = period_prft
                best_indx = period_indx
            period_prft = 0
            period_indx = [i,0]#don't need this?
            cumsum_prft = 0
            passed_loc_min = False
        i += 1
    if period_prft > best_prft:#update for selling at end
        best_prft = period_prft
        best_indx = period_indx    
        
    if cycles_per_day == 1:
        return best_prft, best_indx
    else:          
        if isinstance(rates, list):
            for ix in best_indx:#should never have none; have to adjust when add carrying multiple charges per day
                del rates[ix]
        else:#is np array
            rates = np.delete(rates, list(best_indx))#doesn't occur in place  
        day_best_r, ix_used_r = best_of_day_r(rates = rates, cycles_per_day = cycles_per_day-1, into_blocks = into_blocks, hr_capacity = hr_capacity)
        return best_prft + day_best_r, best_indx+ ix_used_r

def best_of_block3(rates = hourly_rates, cycles_per_day= 1, into_blocks = True, hr_capacity = 4):
    if into_blocks:#blocking all at once
        rates = block(rates, hr_capacity)#Can't block rates by day, need to do all at once
        hr_capacity = 1
    num_days = math.ceil(len(rates)/24)
    profit = [0]*num_days
    indxs = [0]*num_days
    for i in range(num_days):
        profit[i], indxs[i] = best_of_day_r(rates = rates[i*24:min(i*24+24,len(rates))], cycles_per_day= cycles_per_day,  into_blocks = False, hr_capacity = hr_capacity)        
        indxs[i] = [i*24 + j for j in indxs[i]]#increments indxs to match absolute value of indexs

#        next_day_jmp = rates[i*24+1] - rates[i*24]
#        if next_day_jmp > last_discharge_profit:
#            profit[i] -= last_discharge_profit
#            extra_charge = True
            
    #for last iteration
    #profit[i] = best_of_day_r(rates = rates[len(rates) - (num_days-1)*24:], hr_capacity = hr_capacity, extra_charge = extra_charge, into_blocks =False)        
    return profit, indxs#[i for j in indxs for i in j]
best_of_block3(rates = hourly_rates, into_blocks = False, hr_capacity = 1, cycles_per_day = 1)
#unittest.main()
    
#%%
#doesn't work yet
def best_of_block4(rates = hourly_rates, hr_capacity = 4, into_blocks = True, sep_days = True):
    "a bad/lazy way to get best in period,   \
    note best profit; profit till rates go below start of period local min; \
        and from local min to max. "
    if into_blocks:
        rates = block(rates, hr_capacity)
    if sep_days:
        num_days = int(math.ceil(len(rates)/24))
        prfts = {i:0 for i in range(num_days)}
        indxs = {i:[] for i in range(num_days)}
        for i in range(num_days):
            prfts[i], indxs[i] = best_of_block4(rates = hourly_rates[i*24:min(i*24+24, len(rates))], cycles_per_day = cycles_per_day, hr_capacity = 4, into_blocks = False, sep_days = False)
    else:
        try:
            while rates[start] >= rates[start+1]:#only start looking at values once you've had a minimum.
                start += 1
        except:
            indx = [0, 0]#values only decrease
            return 0, indx
        bst_prft, prd_bst_prft, prd_cum_prft = 0, 0, 0
        i = 0
        try:
            while rates[i] >= rates[i+1]:#only start looking at values once you've had a minimum.
                i += 1
        except:#prices only decreased over period
            return 0, [0,0]
        bst_ix, ix  = [i,0], [i,0]
        while i < len(rates)-1:
            prft_cng = rates[i+1] - rates[i]
            prd_cum_prft += prft_cng
            if prd_cum_prft <= 0:
                ix = [i,0]
                if prd_bst_prft >= bst_prft:
                    
            if prft_cng > 0:
            elif prft <= 0:#book best profit        
                
        #%%
best_low_hi = [2**32-1, -2**32]
low_best_hi = [2**32-1, -2**32] 
start = 0
best_low_hi_ix, low_best_hi_ix = [0,0], [0,0]
for i in range(start, len(rates)):
#        print(rates[i], best_low_hi[0])
    if rates[i] < best_low_hi[0]:#found new best low
        best_low_hi[0] = rates[i]
        best_low_hi[1] = rates[i]#nets to 0 if min val is last in series
        best_low_hi_ix = [i, i]
    
    if rates[i] > low_best_hi[1]:#found new best high
        low_best_hi[1] = rates[i]
        low_best_hi[0] = best_low_hi[0]#become best low seen up to this point; is fixed till best high changed
        low_best_hi_ix = [best_low_hi_ix[0], i]
        
    if rates[i] > best_low_hi[1]:#better high after best low
        best_low_hi[1] = rates[i]    
        best_low_hi_ix[1] = i
        
options = [i[1] - i[0] for i in [low_best_hi, best_low_hi]]
day_best = max(options)
if options[0] == day_best:
    ix_used = low_best_hi_ix#
else:
    ix_used = best_low_hi_ix
if cycles_per_day == 1:
    return day_best, ix_used 
else:          
#        indxs = set((*low_best_hi_ix, *best_low_hi_ix))#has <= 4 elements
#        indxs.discard(None)#removes None if it exists
    if isinstance(rates, list):
        for ix in ix_used:#should never have none; have to adjust when add carrying multiple charges per day
            del rates[ix]
    else:#is np array
#            print("\n\n\n",rates)
#            print(list(ix_used))
        rates = np.delete(rates, list(ix_used))#doesn't occur in place  
#            print("\n",rates)
    day_best_r, ix_used_r = best_of_day_r(rates = rates, cycles_per_day = cycles_per_day-1, into_blocks = into_blocks, hr_capacity = hr_capacity)
    return day_best + day_best_r, ix_used + ix_used_r

        
#%% testing
unittest.main()

#%%


#%%######################################################################
#scenario: battery can partially charge/discharge at will, up to 4 times per day
#can't just use current as that relies on changes at lowest. 

#%%delete all this?
testarr1 = [34, 89, 96, 67, 42, 56, 16, 84, 93, 37, 36, 63, 77, 48, 40, 92, 98,
       24, 24, 48, 15, 72, 30, 17]
expected_b = [0,4,5,6]
expected_s = [7,8,15,16]
expected_profit = sum([testarr1[i] for i in expected_s])- sum([testarr1[j] for j in expected_b])
print(expected_profit)

def inline_test1(argv):
    profits, buy_locs, sell_locs = argv
#    for i, (lstb,lsts) in enumerate(zip(list(buy_locs.values())[1:-1], list(sell_locs.values())[1:-1])):
    for i, (lstb,lsts) in enumerate(zip(list(buy_locs.values()), list(sell_locs.values()))):
        if min(lsts) < max(lstb):    
            print(f"Day {i} has overlapping indexs ",i, lstb, lsts)
        if len(lsts) != hr_capacity:
            print(f"Day {i} ; s len wrong ", lsts)
        elif len(lstb) != hr_capacity:
            print(f"Day {i}; b len wrong", i, lstb)

k = [410,
 182,
 194,
 538,
 8,
 612,
 604,
 297,
 583,
 638,
 323,
 242,
 11,
 324,
 338,
 131,
 65,
 837,
 531,
 892,
 105,
 637,
 612,
 101]

#%%
#scenario: battery must completely charge then discharge, but can wait between increasing/decreasing charges doing so.
#qwer
def insert_sort(rates, current_indxs, potential_ix, hr_capacity, buying = True):
    "returns NEW list w/ inserted/deletes values (still sorted); \
    returns list of all indexs hold equivalent values (or none). \
    Want to swap 0th[0 is largest value for buying, lowest value for selling. \
    Can't expand more than 1 index per iteration"
    #all indexs based on location in rates
    ix_pntr = 0
    sz_change = hr_capacity - len(current_indxs)    
    if sz_change <= 0:
        current_indxs = current_indxs[-hr_capacity:]
        profit_change = -1*sum(rates[:-sz_change])#value of rates that are no longer included
        #will drive buy/sell profit closer to 0. Make less negative/positive. add/subtract
    else: 
            current_indxs = sorted([potential_ix] + current_indxs, 
                                   key = lambda ix: rates[ix], 
                                   reverse = buying)#when buying largest values should be in 0th position
            profit_change = rates[potential_ix]
            try:
                alt = [rates[i] for i in current_indxs].index(rates[potential_ix])
                return current_indxs, profit_change, (alt, potential_ix)
            except:
                return current_indxs, profit_change, None
            
    #current_indxs should be sorted based on value they index, in reverse order rates[indx[0]] > rates[indx[-1]]
    if buying:#val shuold be smaller than largest element in list(which is at indx 0)
#        print(rates[current_indxs[ix_pntr]], rates[potential_ix], "testing")
        while rates[current_indxs[ix_pntr]] > rates[potential_ix]:
            ix_pntr += 1
            if ix_pntr == hr_capacity:#pntr interated thru list, smaller than all
                #profit change will be negative; subtracting large from smaller  
                return current_indxs[1:] + [potential_ix], rates[potential_ix] - rates[current_indxs[0]] - profit_change , None           
        if ix_pntr != 0:
            if rates[current_indxs[ix_pntr]] == rates[potential_ix]:#a duplicate of some value in list
                return  current_indxs[1:ix_pntr] + [potential_ix] + current_indxs[ix_pntr:], -profit_change , (current_indxs[ix_pntr], potential_ix)
            else:#was smaller than atleast 1 val
                return current_indxs[1:ix_pntr] + [potential_ix] + current_indxs[ix_pntr:], rates[potential_ix] - rates[current_indxs[0]] - profit_change , None
        else:#no changes made; was larger
            return current_indxs, -profit_change , None
        
    else:#selling, larger than smallest val in list
        #normal order. rates[indx[0]] < rates[indx[-1]]. Want to swap 0th
        while rates[current_indxs[ix_pntr]] < rates[potential_ix]:
            ix_pntr += 1
            if ix_pntr == hr_capacity:#pntr interated thru list; larger than all
                return current_indxs[1:] + [potential_ix], profit_change + rates[potential_ix] - rates[current_indxs[0]], None
            #Above is increase size: rates[current_indxs[0]] was set equal to rates[potential_indx], cancels leaving profit change. If decrease size; the value you replace is updated
        if ix_pntr != 0:
            if rates[current_indxs[ix_pntr]] == rates[potential_ix]:#a duplicate; currently selling earlier(iterating backwards)
                return  current_indxs[1:ix_pntr] + [potential_ix] + current_indxs[ix_pntr:], profit_change , (current_indxs[ix_pntr], potential_ix)
            else:#was smaller than atleast 1 val
                return current_indxs[1:ix_pntr] + [potential_ix] + current_indxs[ix_pntr:], profit_change + rates[potential_ix] - rates[current_indxs[0]], None
        else:#no changes made; 
            return current_indxs, profit_change , None
        
#tzt = list(range(10)) + list(range(10,0,-1))
#current_indx = [0]
#for i in range(1,20):
#    current_indx, prft_change, _ = insert_sort(tzt, current_indx, i, 5, buying = False)
#print( "index: " + str(current_indx) +  \
#                           "rates: " + str([tzt[i] for i in current_indx]) +\
#                           "prftchange: " +  str(prft_change) + " " + str(i))
##%%   
def max_indx(series, buying):
    if buying:#buy is negative
        best_buy = max(series)#where spent the least
        bi = series.index(best_buy)#first buyable location
        return best_buy, bi
    else:
        best_sell = max(series)
        si = [i for i,val in enumerate(series) if val == best_sell][-1]#want last sellable location
        return best_sell, si

#def sum_buys_less_than_sells(b,s):
#    "takes in a list of buying, selling prices and returns the sum of values where\
#    buy is less than selling"
    

def intermitant(rates, hr_capacity = 4, cycles_per_day = 1, into_blocks = False, use_add = True, for_testing = False):#what does use add do?
    "returns dicts of profit, buy locs, sell locs[locs are absolute] with buying at best hr_capacity buys,\
    then selling at best hr_cap sells. All sells occur after all buys. Assumes always completely, charge, discharge each day"
    assert(hr_capacity*2 <= 24)
    if not isinstance(rates, np.ndarray):
        rates = np.array(rates)
    if into_blocks:#calculates assuming can only charge in continious hours
        return intermitant(rates=block(rates, hr_capacity), hr_capacity = 1, cycles_per_day = 1, into_blocks = False, for_testing = for_testing )
    
    
    if len(rates)%24 <= 2*hr_capacity and len(rates)%24 != 0:#need to be able to completely buy, sell on last day
        print("Droping the last day's hours to make times equal")
        rates = rates[:-len(rates)%24]
        
    num_days = int(math.ceil(len(rates)/24))
    buy_locs = {i:[] for i in range(num_days)}
    sell_locs = {i:[] for i in range(num_days)}
    profits = {i:[] for i in range(num_days)}
    
    buy_profit = [0]*(23)#will be negative for buying; positve for selling
    sell_profit = [0]*(23)
    buy_ix = [[0] for _ in range(hr_capacity)]*(23)#list of lists; each sublist is of best places to buy up to that point(indexs for rates)
    sell_ix = [[0] for _ in range(hr_capacity)]*(23)#sublist is hour withen TOTAL PERIOD

    #should treat first as special too

    for i in range(num_days):#2, num_days-1):
        #lengths are 24 - hr_capacity + 1 as start at 0 is w/ hr_capacity joined together, have 24-hr_cap other options "in front"
        #list buy_ix[i] is list of indicies for the smallest values in list thus seen, sorted with largest value in 0th position
        loc_in_buy_ix = 0
        buy_ix[loc_in_buy_ix] = [i*24]#[ix for _, ix in sorted(zip(rates[i*24:i*24+hr_capacity], range(i*24,i*24+hr_capacity)), reverse = True)]
        buy_profit[loc_in_buy_ix] = -rates[i*24]
        buy_alternates = [0]*(23)#have to stop selling with room to sell at least once 
        
        #don't reverse the range as need to pair with the locations of vals in rates   
        loc_in_sell_ix = 22#starts at 'end'
        sell_ix[loc_in_sell_ix] = [i*24+23]#[ix for _, ix in sorted(zip(rates[(i+1)*24-hr_capacity:(i+1)*24], range((i+1)*24-hr_capacity,(i+1)*24)), reverse = False)]
        sell_profit[loc_in_sell_ix] = rates[i*24 + 23]#sum(rates[(i+1)*24-hr_capacity:(i+1)*24])
        sell_alternates = [0]*(23)#25-2*hr_capacity)
        
#        for j in range(24*i+23-hr_capacity, i*24+hr_capacity-1, -1):#calc indexes of selling; end of night to first hr_cap where just buying
#            sell_ix[loc_in_sell_ix-1], profit_change, alt = insert_sort(rates, sell_ix[loc_in_sell_ix], j, hr_capacity, buying = False)
#            sell_profit[loc_in_sell_ix-1] = sell_profit[loc_in_sell_ix] + profit_change#profit_change pos; sell for more
#            sell_alternates[loc_in_sell_ix] = alt
#            loc_in_sell_ix -= 1
        
        #calc indexes of selling; end of night to first hr_cap where just buying
        for j in range(24*i+22, i*24, -1):#range has to leave room to buy at first loc; initialized w/ selling at 24*i+23
            hrs_used = min(hr_capacity, j%24, abs(24-j))
            sell_ix[loc_in_sell_ix-1], profit_change, alt = insert_sort(rates, sell_ix[loc_in_sell_ix], j, hrs_used, buying = False)
            sell_profit[loc_in_sell_ix-1] = sell_profit[loc_in_sell_ix] + profit_change#profit_change pos; sell for more
            sell_alternates[loc_in_sell_ix] = alt
            loc_in_sell_ix -= 1
#            print(j)
#            print("selling stuff", j, hrs_used, sell_profit[loc_in_sell_ix-1], "\n")
        #take into account situation where only discharge a little on the last few hours of each day
        
        #adjust for situation when abs(buy[0]) > sell[0]
        day_prft = 0
        day_indxs = [[0], [0]]
        
        for j in range(i*24, 24*i+23):#calc indexes of buying from possible option to last before would just be selling hour of night
            hrs_used = min(hr_capacity, (j+1)%24, abs(23-j))
            if j%24 != 0:#need to be able to compare buy @0, sell 1@;(both have loc = 0)
                buy_ix[loc_in_buy_ix+1], profit_change, alt = insert_sort(rates, buy_ix[loc_in_buy_ix], j, hrs_used, buying = True)
                buy_profit[loc_in_buy_ix+1] = buy_profit[loc_in_buy_ix] + profit_change#profit_change negative; less outlay for buying power
                buy_alternates[loc_in_buy_ix] = alt
                loc_in_buy_ix += 1

            #if make a copy have to redo deletes each time; don't want this?
            #after delete highest buy, next value could have been even higher and be profitable.
#            print(buy_ix[loc_in_buy_ix], sell_ix[loc_in_buy_ix])
            max_cycle = min(len(buy_ix[loc_in_buy_ix]), len(sell_ix[loc_in_buy_ix]))#have to buy/sell in equal quantities
            day_buy = np.array(buy_ix[loc_in_buy_ix][-max_cycle:], dtype = np.int_)
            day_sell = np.array(sell_ix[loc_in_buy_ix][-max_cycle:], dtype = np.int_)
            calc_prft = sum(rates[day_sell]) - sum(rates[day_buy])#sell_profit[loc_in_buy_ix] + buy_profit[loc_in_buy_ix + 1]
#            reps = max(0, (2**32)*(len(day_buy)==0 or len(day_sell)==0))
            reps = 0
#            print("\n\n###########\n", hrs_used, day_buy, day_sell, i)
            try:
                while reps < hrs_used and rates[day_buy[0]] >= rates[day_sell[0]]:
    #                print(day_buy, day_sell, day_buy[0], day_sell[0], reps)
#                    print(calc_prft, rates[day_buy[0]], rates[day_sell[0]])
                    calc_prft += rates[day_buy[0]]
                    calc_prft -= rates[day_sell[0]]
                    day_buy = np.delete(day_buy, 0)
                    day_sell = np.delete(day_sell, 0)
#                    print(day_buy, day_sell, calc_prft,"\n")
                    if len(day_buy) == 0 or len(day_sell) == 0:
                        break
                    reps += 1
#                print(buy_ix[loc_in_buy_ix+1], "\n", sell_ix[loc_in_buy_ix+1], "\n\n\n")
#            print(buy_profit, sell_profit, day_buy, day_sell, calc_prft, hrs_used, j, "\n")
            except Exception as e:
                print(reps, hrs_used, day_buy, day_sell, rates)
                return e
            if calc_prft > day_prft:#this buys as early, sells as late as possible
                day_prft = calc_prft#sell_profit[loc_in_buy_ix] + buy_profit[loc_in_buy_ix]
                day_indxs = [day_buy, day_sell]#sell_ix[loc_in_buy_ix] 
        
        profits[i] = day_prft
        [buy_locs[i], sell_locs[i]] = day_indxs 
        #get best buy/sell locations
        #SELL PROFITS are REVERSED! 
        #give them 'room' to buy; sell. needs hr_cap left in day to unwind; hr_cap in start to buy
        
        
#        
#        #buying price always decreases/constant, selling increases/ as go towards indx 0
#        assert(use_add)
#        if use_add:
##            print(buy_ix, sell_ix, sell_alternates, sep = "\n")
#            prft = [i+j for i,j in zip(buy_profit, sell_profit)]
#            profits[i] = max(prft)
#            indx = prft.index(profits[i])
#            #doesn't HAVE to use entire capacity in a given day
##            partial_charge = buy_ix[indx] <= sell_ix[indxs]#'worst' prices are both in 0th location
#            buy_locs[i] = buy_ix[indx]#[partial_charge]
#            sell_locs[i] = sell_ix[indx]#[partial_charge]
#
#        else:#doesn't work; use method above  
#            best_buy, bi = max_indx(buy_profit, buying = True)#get where spent the least(neg val)
#            best_sell, si = max_indx(sell_profit, buying = False)
#            #hr = buy_ix + hr_cap - 1#hours start at 0
#            #hr = hrs_in_day(24) - hr_cap  - sell_ix
#            #buy_ix + hr_cap - 1 = hr_in_day - hr_cap - sell_ix
#            #buy_ix = hr_in_day - 2*hr_cap + 1 - sell_ix
#            #sell_ix = hr_in_day - 2*hr_cap + 1 - buy_ix
##            print(f"{i}: \n {bi} {buy_profit} \n {si} {sell_profit}")
#            #25-2*hr_capacity-si-1]
#            loc_min, bi2 = max_indx(buy_profit[:len(buy_profit)-si], buying = True)#could sell where bought; 
##            print(loc_min, bi2, best_sell, si)
#            #25-2*hr_capacity-bi-1
#            loc_max, si2 = max_indx(sell_profit[:len(sell_profit)-bi], buying = False)
##            print(best_buy, bi, loc_max, si,"\n\n\n")
#    
#            if best_sell + loc_min > loc_max + best_buy:#buys are negative
#    #            print(i,"indxs sell max:", si, best_sell, bi2, loc_min)
#                profits[i] = best_sell + loc_min
#                buy_locs[i] = buy_ix[bi2]
#                sell_locs[i] = sell_ix[si]
#            else:
#    #            print(i,"indxs buy min: ", si2, loc_max, bi, best_buy)
#                profits[i] = loc_max + best_buy
#                buy_locs[i] = buy_ix[bi]
#                sell_locs[i] = sell_ix[si2]    
#      


        #this is where are making the mistake about when to buy/sell twice in a day; 
        #by solely prioritizing present profit vs. comparing against the benifit to alternative days
        #would get around this by also calculating selling starting at i*24-hr_cap+1:i*24+1 and comparing the benifits of starting at each location
        #with curves; buy_profit[i] + sell_profit[i] vs. value of extending it on the ends
        
        #for buy want latest buy index to be as early as possible so have more possible sales
        #but have earliest buy index as late as possible, so more sales could be given to yesterday
        #for sales is flipped; latest is earlier; earlies is later            
            
    #need to treat last day as special case of not looking forward
#    print("\n\n\n\n\n")
    if for_testing:#format it in the same way as the others
        return profits, [[i, j] for i,j in zip(buy_locs.values(), sell_locs.values())]
    else:
        return profits, buy_locs, sell_locs
    
import itertools
def locs_to_01(buy_locs, sell_locs, hr_capacity):
    if isinstance(buy_locs, dict):
        out = [0]*24*len(buy_locs.keys())
        b_ix = [i for j in buy_locs.values() for i in j]
        s_ix = [i for j in sell_locs.values() for i in j]
        rm_val =[i for i in b_ix if i in s_ix]
        for dup in rm_val:
            del b_ix[b_ix.index(dup)]
            del s_ix[s_ix.index(dup)]
        list(map(out.__setitem__, b_ix + s_ix,
                           iter([1 for _ in range(len(b_ix))] + [-1 for _ in range(len(s_ix))])))     
    else:
        print("Got a single list, Assuming it's just for a single day")
        out = [0]*24
        list(map(out.__setitem__, [j%24 for j in buy_locs] + [j%24 for j in sell_locs],\
                           iter([1 for _ in range(len(buy_locs))] \
                                 + [-1 for _ in range(len(sell_locs))])))#list is just so map evals     
    return out
#hourly_rates[:24] = np.arange(0,24-)
#tzt = np.array([2,93, 5, 1, 44,82,63,80,74,14,70,12,87,63, 8,25,39,22,81,23,18,80,0, 99])
#tzt = np.array([2,93, 5, 1, 44,82,63,80,74,14,70,12,87,63, 8,25,39,22,81,23,-1000,-1000,99, 99])
#tzt = [7, 253, 758, 931, 569, 315, 602, 730, 858, 790, 613, 775, 247, 805, 239, 305, 942, 135, 162, 70, 434, 776, 614, 120]
#tzt = [93,2, 5, 0,44,82,63,80,74,14,70,12,87,63, 8,25,39,22,81,23,18,80,53, 82]
#tzt = [284, 178, 912, 228, 121, 572, 324, 15, 776, 363, 9, 399, 596, 747, 816, 830, 219, 637, 220, 925, 463, 168, 235, 731]
#tzt = [39, 947, 881, 357, 263, 708, 686, 271, 979, 551, 259, 815, 623, 470, 452, 454, 390, 792, 21, 127, 266, 251, 843, 293]
#for i in range(31):
#    tzt = instr[i*24:i*24+24]
#    profits, buy_locs, sell_locs = intermitant(tzt, hr_capacity = 4)
#    cal_prft = sum([tzt[i] for i in sell_locs[0]]) - sum([tzt[i] for i in buy_locs[0]])
#    print(profits, buy_locs, sell_locs, cal_prft, cal_prft == profits[0], "\n")
#
tzt = [int(i) for i in re.findall('\d+', '[42 18 65 25 49 11 87  8 29 36  4 52 79  3 74 82 68 27 95 33 60 85 48 87]')]
#tzt = [100, -1500, -1500, 2000, 2000, 44, 82, 63, 80, 74, 14, 70, 12, 87, 63, 8, 25, 39, 22, 81, 23, 18, 80, 0]

profits, buy_locs, sell_locs = intermitant(tzt, hr_capacity = 4)
i = 0   
plot_bs_loc(tzt[i*24:i*24+24], [j%24 for j in buy_locs[i]], [j%24 for j in sell_locs[i]])
print(locs_to_01(buy_locs, sell_locs, 4))
#binary = locs_to_01(buy_locs, sell_locs, 1)
#for i in range(len(tzt)//24):
#    print(sum(binary[i*24:i*24+24]), "\n")
#unittest.main()   
#%%
#print(intermitant(hourly_rates), "out\n\n")
hr_capacity = 4

profits, buy_locs, sell_locs = intermitant(tztrt, hr_capacity = hr_capacity)
print(profits, buy_locs, sell_locs)
inline_test1([profits, buy_locs, sell_locs])

w=0
bl = buy_locs[w]#plots first day    
sl = sell_locs[w]     
print(profits[w], [i%24 for i in bl], [i%24 for i in sl])
plot_bs_loc(tztrt[24*w:24*w+24], [i%24 for i in bl], [i%24 for i in sl])


unittest.main()

#%%
import unittest

class TestBattery(unittest.TestCase):
    
#    def __init__(self, num_days = 31, hourly_rates = None, hr_capacity = 4, blocked_rates = None):
    num_days = 31
    hr_capacity = 4
    hourly_rates = np.random.randint(0, high = 1000, size = int(24*num_days))
#    blocked_rates = np.array([np.mean(hourly_rates[i:i+hr_capacity]) for i in range(num_days - hr_capacity)]) 
    profits, buy_locs, sell_locs = intermitant(hourly_rates, hr_capacity = hr_capacity)     
    
    f = open("hourly_rates.txt", "w")#need to wipe old data, "a" = append instead of overwrite
    f.write(np.array2string(hourly_rates))
    f.close()
    longMessage = True
        
    def day_decorator(test):
        "decorater that runs test for all possible days"
        def wrapper(self):
            for i in range(self.num_days):
                try:
                    test(self, rates = hourly_rates[i*24:i*24+24])
                except Exception as e:
                    e.args += ("error with day ", str(i) ," for test " ,str(test) , " at loc " , str(hourly_rates[i*24:i*24+24]))
                    raise e
        return wrapper
    
    def hr_block_decorator(test):
        "decorator that runs test for various size of hour blocks 1,11"
        def wrapper2(self):
            for i in range(1,10):
                try:
                    test(self, hr_capacity = i)
                except Exception as e:
                    e.args += ("error with block size ", str(i) ," for test " ,str(test))
                    raise e
        return wrapper2
    #!!!!!Change assertCountEqual to assertItemsEqual when go back to python 2.7(evn that won't work in 2.6)  
       
    
    def negative_rates_decorator(test):
        "decorator that runs test for various size of hour blocks 1,11"
        rts = np.random.randint(-500, high = 500, size = 24*num_days)
        def wrapper3(self):
                test(self, rates = rts)
        return wrapper3
    
#    @day_decorator#wrong
#    def test_bs_2_series(self, rates):#mins, maxs, prices, into_blocks = False, hr_capacity = 4, cycles = 1, sell_twice = False, i = None):
#        self.assertFalse(isinstance(rates, list), msg = "currently assume rates is a numpy array")
#        prft, ix = bs_2_series(rates,rates, cycles = 1)
#        self.assertEqual(prft, rates[ix[1]] - rates[ix[0]], msg = f"{rates}, {prft}, {ix}")
#        if rates[ix[0]] == min(rates):#bough low
#            #will accept duplicates currently
#            self.assertEqual(rates[ix[1]], rates[np.where(rates == max(rates[ix[0]:]))[0][-1]])
##            self.assertEqual(ix[1], np.where(rates == max(rates[ix[0]:]))[0][-1])
#        else:#sold high
#            self.assertEqual(rates[ix[0]], rates[np.where(rates == min(rates[:ix[1]]))[0][0]])
##            self.assertEqual(ix[0], np.where(rates == max(rates[:ix[1]]))[0][0])
    
#    @day_decorator
#    def test_indxs_match(self, rates = None, hr_capacity = 4):#get decorator to work
#        "test profits equal indxs"
#        for fn in [best_of_block1T, best_of_block2T, best_of_block3]:
#                tezt_blocked = block(rates, hr_capacity)
#                prfts, indxs = fn(rates, hr_capacity = hr_capacity, into_blocks = True)
#                for i,j in zip(prfts, indxs):
#                    self.assertEqual(i, sum(tezt_blocked[k] for k in j[1::2]) - sum(tezt_blocked[k] for k in j[::2]), msg = str(fn))
                    
#    @hr_block_decorator
#    def test_indxs_consistent(self, hr_capacity = 4):#change to self, buy_locs, sell_locs
#        "takes dict of buy/selling indexs and checks that no buy occurs after sell"
#        for fn in [best_of_block1T, best_of_block2T]:
#            _, indxs = fn(self.hourly_rates, hr_capacity = hr_capacity, into_blocks = True)
##            buy_locs, sell_locs = [i[0] for i in indxs], [i[1] for i in indxs]
#            for i, (lst_b,lst_s) in enumerate(indxs):#zip(self.buy_locs.values(), self.sell_locs.values())):
#                self.assertLessEqual(lst_b, lst_s, msg = "indexs overlap")#INDEXS ?should never be equal?; (lstb) < min(lsts)
#                self.assertLess(lst_b, lst_s, msg = "indexs overlap or are equal")#INDEXS ?should never be equal?; max(lstb) < min(lsts)
##                self.assertEqual(len(lst_b), self.hr_capacity, msg = "sell indxs wrong len")
##                self.assertEqual(len(lst_s), self.hr_capacity, msg = "buy indxs wrong len")

    @day_decorator#add hr block
    def test_indxs_best_itermitant(self, rates = None):
        "tests that there aren't better indexs; where prices lower before first sell\
        higher after last buy."
        day_rates = [i for i in rates]
        profit, buy_loc, sell_loc = intermitant(day_rates, hr_capacity = self.hr_capacity, cycles_per_day = 1, into_blocks = False)
        if len(buy_loc[0]) !=0 and len(sell_loc[0]) !=0:
            worst_buy = max([day_rates[i] for i in buy_loc[0]])
            worst_sell = min([day_rates[i] for i in sell_loc[0]])
            last_buy_loc =max(buy_loc[0])
            first_sell_loc = min(sell_loc[0])
    #        for i in sorted(set(sell_loc[0] + buy_loc[0]), reverse = True):#delete elements from list back to front so don't change indexs of elements
    #            del day_rates[i] #if buy/sell at same indx will only delete one 
    #        try:
            self.assertLessEqual(worst_buy, min(day_rates[last_buy_loc+1:first_sell_loc], default = 2**32), \
                                 msg = "bought more expensive on day " + str(i) + "; rates: " + str(day_rates) +\
                                 str(buy_loc[0])+str(sell_loc[0]))
            self.assertGreaterEqual(worst_sell, max(day_rates[last_buy_loc+1:first_sell_loc], default = -1*2**32), \
                                    msg = "sold cheaply on day " + str(i) + "; rates: " + str(day_rates) +\
                                    str(buy_loc[0])+str(sell_loc[0]))
#        except Exception as e:
#            plot_bs_loc(rates[24*i:24*i+24], [j%24 for j in buy_loc], [j%24 for j in sell_loc])
#            print(day_rates)
#            return e
      
#    @day_decorator
#    def test_indx_best_day_block(self, rates = None):
#        "test best of block are getting best indxs"
#        for fn in [best_of_block1T, best_of_block2T, best_of_block3]:
              
#    def test_block1T2T3_val1(self):
#        for rates  in [[79, 71, 63, 62, 52, 63, 36, 27, 9, 97, 78, 69, 37, 67, 2, 48, 13, 23, 99, 65, 35, 14, 52, 49]]:#, \
#                       #[31, 49, 96, 56, 77, 56, 70, 88, 95, 86, 32, 39, 30, 73, 96, 21, 45, 88, 13, 32, 83, 38, 7, 37]]:
#            profits1, indxs1 = best_of_block1T(rates = rates, into_blocks = True, hr_capacity = 1)
#            profits2, indxs2 = best_of_block2T(rates = rates,   into_blocks = True, hr_capacity = 1)
#            profits3, indxs3 = best_of_block3(rates = rates, cycles_per_day = 1, into_blocks = True, hr_capacity = 1)
#    #        print(profits1, indxs1, profits2, indxs2, profits3, indxs3)
#            self.assertCountEqual([97,97,97], [profits1[0], profits2[0], profits3[0]])#day decorator only gives 1 day
#            self.assertTrue([[14,18],[14,18],[14,18]], [indxs1[0], indxs2[0], indxs3[0]])#day decorator only gives 1 day
#            
#    def test_block1_equals_block3T(self):
#        profits1, indxs1 = best_of_block1T(rates = self.hourly_rates, into_blocks = True, hr_capacity = 4)
#        profits3, indxs3 = best_of_block3(rates = self.hourly_rates, cycles_per_day = 1, into_blocks = True, hr_capacity = 4)
##        for ix, (p1, p3) in enumerate(zip(profits1, profits3)):
##            if p1 != p3:
##                print("Error at: ", indxs1[ix], indxs3[ix])
#        self.assertCountEqual(profits1, profits3, msg = f"indx1: {indxs1}\n indx3: {indxs3}")

            
#    @day_decorator
#    def test_block1T2T3_equal(self, rates = None):#gribb; going to have to debug this
#        profits1, indxs1 = best_of_block1T(rates = rates, into_blocks = True, hr_capacity = 4)
#        profits2, indxs2 = best_of_block2T(rates = rates, into_blocks = True, hr_capacity = 4)
#        profits3, indxs3 = best_of_block3(rates = rates, cycles_per_day = 1, into_blocks = True, hr_capacity = 4)
##        for ix, p1p2p3 in enumerate(zip(profits1, profits2, profits3)):
##            if 3 != p1p2p3.count(p1p2p3[0]):
##                print("Error at: ", indxs1[ix], indxs2[ix], indxs3[ix])
##                print([self.hourly_rates[i[0]-4:i[1]+4] for i in (indxs1[ix], indxs2[ix], indxs3[ix])])
##                print("\n\n")
#        self.assertTrue(all((profits1[0] == profits2[0], 
#                            profits1[0] == profits3[0],
#                            profits2[0] == profits3[0])), 
#                            msg = str("profits: \n" + str(profits1[0]) + "\n" +
#                            str(profits2[0]) + "\n"+ str(profits3[0])))#day decorator only gives 1 day
#        self.assertTrue(all((indxs1[0] == indxs2[0], 
#                            indxs1[0] == indxs3[0],
#                            indxs2[0] == indxs3[0])), 
#                          msg = str(str(indxs1[0]) + "\n" + str(indxs2[0]) + "\n" + str(indxs3[0])))
            
    def test_loc_minmax1_equals_loc_minmax2(self):
        b,s = get_loc_minmax2(self.hourly_rates, locs = False)
        buy_locs, sell_locs = get_loc_minmax1(self.hourly_rates)
        self.assertCountEqual(b, buy_locs)#, msg = {f"test_equal(buy_locs, b)"})
        self.assertCountEqual(s, sell_locs)#, msg = {f"test_equal(sell_locs, s)"})
        b,s = get_loc_minmax2(self.hourly_rates, locs = True)
        self.assertCountEqual(b, list(np.where(buy_locs)[0]))#, msg = {f"test_equal(buy_locs, b)"})
        self.assertCountEqual(s, list(np.where(sell_locs)[0]))#, msg = {f"test_equal(sell_locs, s)"})
        
    ###############################################################
# insert_sort(rates, current_indxs, potential_ix, hr_capacity, buying = True)      
    def iter_thru_insert_sort(self, rates, sz, hr_capacity = None):
        "test insert sort does get minimum/maximum n elements, going forward and backwards.\
        Buy Forward, sell forward, buy backwards, sell backwards"
        hr_capacity = hr_capacity or self.hr_capacity
#        print("\n\n\n##################", hr_capacity, "##################\n\n\n\n\n")
        out = [0]*5
        out[4] = hr_capacity
        potential_ix = [[0]*hr_capacity for _ in range(sz-hr_capacity+1)]
        potential_ix[0] = [ix for ix, val in sorted(zip(range(hr_capacity), rates[:hr_capacity]), key = lambda tup: tup[1], reverse = True)]#indexs
        for i in range(hr_capacity,sz):
            potential_ix[i-hr_capacity+1],_ ,_ = insert_sort(rates[:sz], potential_ix[i-hr_capacity], i, hr_capacity = hr_capacity, buying = True)
        out[0] =  potential_ix[-1]
        
        potential_ix = [[0]*hr_capacity for _ in range(sz-hr_capacity+1)]      
        potential_ix[0] = [ix for ix, val in sorted(zip(range(hr_capacity), rates[:hr_capacity]), key = lambda tup: tup[1], reverse = False)]#indexs
        for i in range(hr_capacity,sz):
            potential_ix[i-hr_capacity+1], _, _ = insert_sort(rates[:sz], potential_ix[i-hr_capacity], i, hr_capacity = hr_capacity, buying = False)
        out[1] =  potential_ix[-1]
        
        #iterating backwards
        potential_ix = [[0]*hr_capacity for _ in range(sz-hr_capacity+1)]
        potential_ix[-1] = [ix for ix, val in sorted(zip(range(sz-hr_capacity,sz), rates[sz-hr_capacity:sz]), key = lambda tup: tup[1], reverse = True)]#indexs
        for i in range(sz-hr_capacity-1,-1,-1):
            potential_ix[i], _, _ = insert_sort(rates[:sz], potential_ix[i+1], i, hr_capacity = hr_capacity, buying = True)
        out[2] =  potential_ix[0]
        
        potential_ix = [[0]*hr_capacity for _ in range(sz-hr_capacity+1)]
        potential_ix[-1] = [ix for ix, val in sorted(zip(range(sz-hr_capacity,sz), rates[sz-hr_capacity:sz]), key = lambda tup: tup[1], reverse = False)]#indexs
        for i in range(sz-hr_capacity-1,-1,-1):
            potential_ix[i], _, _ = insert_sort(rates[:sz], potential_ix[i+1], i, hr_capacity = hr_capacity, buying = False)
#            print([rates[i] for i in potential_ix[i]])
#            print(potential_ix[i])
        out[3] =  potential_ix[0]
        return out
    
    @day_decorator
    def test_insert_sort_getting_minmax(self, rates):
        "test insert sort does get minimum/maximum n elements"#
#        print("test insert sort now", self.hr_capacity)
        ix = self.iter_thru_insert_sort(rates, 24, hr_capacity = self.hr_capacity)
#        print("test insert sort now has ended")
        rates_sorted = sorted(rates[:24])
        self.assertCountEqual([rates[i] for i in ix[0]], sorted(rates_sorted[:self.hr_capacity], reverse=True),
                              msg = f"{ix[0]}")  
        self.assertCountEqual([rates[i] for i in ix[1]], sorted(rates_sorted[-self.hr_capacity:], reverse=False),
                              msg = f"{ix[1]}")
        #iterating backwards
        self.assertCountEqual([rates[i] for i in ix[2]], sorted(rates_sorted[:self.hr_capacity], reverse=True),
                              msg = f"{ix[2]}")  
        self.assertCountEqual([rates[i] for i in ix[3]], sorted(rates_sorted[-self.hr_capacity:], reverse=False),
                              msg = f"{ix[3]}, {ix[4]}, {[rates[i] for i in ix[3]]}, { sorted(rates_sorted[-self.hr_capacity:], reverse=False)}")
           
    @day_decorator
    def test_insert_sort_duplicates(self, rates):
        q = [i for i in rates]
        q[12:12+self.hr_capacity] = [-100]*self.hr_capacity
        ix = TestBattery.iter_thru_insert_sort(self, q, 24)
        self.assertEqual(ix[0], sorted(list(range(12,12+4)), reverse = True), msg = "buying")
        self.assertEqual(ix[2], sorted(list(range(12,12+4)), reverse = False), msg = "buying back")

        q[12:12+self.hr_capacity] = [11000]*self.hr_capacity
        ix = TestBattery.iter_thru_insert_sort(self, q, 24)
        self.assertEqual(ix[1], sorted(list(range(12,12+4)), reverse = True), msg = "selling")
        self.assertEqual(ix[3], sorted(list(range(12,12+4)), reverse = False), msg = "selling, iter backwards")    
        
        
    def test_insert_sort_peaked_rates(self):#replace w/ iter_thru
        tzt = list(range(10,0,-1))+list(range(10))
        ix = TestBattery.iter_thru_insert_sort(self, tzt, 20, hr_capacity = 5)
        self.assertEqual([tzt[i] for i in ix[0]], [2,2,1,1,0])
        self.assertEqual([tzt[i] for i in ix[2]], [2,2,1,1,0])

        tzt = list(range(10)) + list(range(10,0,-1))
        ix = TestBattery.iter_thru_insert_sort(self, tzt, 20, hr_capacity = 5)
        self.assertEqual([tzt[i] for i in ix[1]], [8,8,9,9,10])
        self.assertEqual([tzt[i] for i in ix[3]], [8,8,9,9,10])


    def driving_tests(self, testlist, expected_prfit, expected_indx, kwargs):
        "**kwarg is a named vector of the arguments for functions. \
        Name is function object for value of argument"
        for fn in [intermitant]:# [best_of_block1T, best_of_block2T, best_of_block3, intermitant]:
#            print(kwargs)
            profits, indxs = fn(testlist, **kwargs[fn])
            profit_from_indx = sum([testlist[si] for day in indxs for si in day[1]])- sum([testlist[bi] for day in indxs for bi in day[0]])
#            print(indxs, [day for day in indxs], [day for cycle in indxs for day in cycle[0]])
#            self.assertTrue(1==0, msg = f"{[testlist[cycle[1]] for day in indxs for cycle in day]}; buys: {[testlist[cycle[0]] for day in indxs for cycle in day]}")
            self.assertTrue(all([expected_prfit == profit_from_indx, expected_prfit == profits[0]]), \
                            msg = f"sell Locs: {[si for day in indxs for si in day[1]]}; \n \
                            Buy Locs: {[bi for day in indxs for bi in day[0]]}; \n \
                            expected_prfit: {expected_prfit}; \n \
                            profit_from_indx: {profit_from_indx}; \n \
                            outputed profits: {profits[0]} \n \
                            testarr: {testlist}")
            for got, ex in zip(indxs, expected_indx):
                for g1, ex1 in zip(got, ex):
                    self.assertCountEqual(sorted(g1), sorted(ex1),msg = f"{got}, {ex}")
            print(kwargs[fn]['hr_capacity'])
            self.assertEqual(0,
                             sum(locs_to_01([bi for day in indxs for bi in day[0]], 
                                        [si for day in indxs for si in day[1]], 
                                        kwargs[fn]['hr_capacity'])))
                    
        
    def test_intermitant_vals(self):
        "tests buy/sell indx, profit for an example day for intermitant"
        for k, testarr1 in  enumerate([\
                [34, 89, 96, 67, 42, 56, 16, 84, 93, 37, 36, 63, 77, 48, 40, 92, 98,24, 24, 48, 15, 72, 30, 17],
                [2,93, 5, 0,44,82,63,80,74,14,70,12,87,63, 8,25,39,22,81,23,18,80,53, 82],#test can buy 0, sell 1
                [2,83, 5, 1, 44,82,63,80,74,14,70,12,87,63, 8,25,39,22,81,23,18,80,0, 99],#test buy 22, sell 23
                [2,93, 5, 1, 44,82,63,80,74,14,70,12,87,63, 8,25,39,22,81,23,-1000,-1000,99, 99],#test don't use entiterity of the hour capacity
                [234, 199, 669, 616, 596, 93, 63, 698, 10, 348, 136, 847, 795, 980, 121, 160, 188, 171, 309, 530, 717, 408, 327, 626],#shouldn't buy/sell at 0,1
                [234, 199, 669, 616, 596, 93, 63, 698, 10, 348, 136, 847, 795, 980, 121, 160, 188, 171, 309, 530, 717, 408, 1000, 0],#shouldn't buy/sell at 22,23
                [-1500, -1500, 2000, 2000, 44,82,63,80,74,14,70,12,87,63, 8,25,39,22,81,23,18,80,0, 99],#only use partial charging on first hours
                [100, -1500, -1500, 2000, 2000, 44,82,63,80,74,14,70,12,87,63,8,25,39,22,81,23,18,80,0],#ignore buys of partial charging on first hours
                [0, 3000, 100, -1500, -1500, 2000, 2000, 44,82,63,80,74,14, 99 ,12,87,63, 8,25,39,22,81,23,80],#ignore buy/sell cycle with partial charging later
                ]):
    #, [79, 71, 63, 62, 52, 63, 36, 27, 9, 97, 78, 69, 37, 67, 2, 48, 13, 23, 99, 65, 35, 14, 52, 49]
            expected_b = [[0,4,5,6], [0], [22], [20,21], [8], [8], [0,1], [1,2], [0,3,4]][k]
            expected_s = [[7,8,15,16], [1], [23], [22,23], [13], [22], [2,3], [3,4], [5,6,13]][k]
            expected_profit = sum([testarr1[i] for i in expected_s])- sum([testarr1[j] for j in expected_b])
            kwargs = [{intermitant: {'hr_capacity': 4, "for_testing": True}}, \
                      {intermitant: {'hr_capacity': 1, "for_testing": True}},\
                      {intermitant: {'hr_capacity': 1, "for_testing": True}},\
                      {intermitant: {'hr_capacity': 5, "for_testing": True}},\
                      {intermitant: {'hr_capacity': 1, "for_testing": True}},\
                      {intermitant: {'hr_capacity': 1, "for_testing": True}},\
                      {intermitant: {'hr_capacity': 6, "for_testing": True}},
                      {intermitant: {'hr_capacity': 6, "for_testing": True}},
                      {intermitant: {'hr_capacity': 6, "for_testing": True}},
                      ][k]
            self.driving_tests(testarr1, expected_profit, [[expected_b, expected_s]], kwargs)

    def test_vals_dec(self):#need to deal with case where there are no infleciton points.
        "values only decrease"
        testlist = list(range(48,24,-1))
#        for fn in [best_of_block1T, best_of_block2T, best_of_block3]:
#            profits, indxs = fn(testlist)
#            expected_profit = sum([testlist[i[1]] for i in indxs])- sum([testlist[j[1]] for j in indxs])
#            self.assertTrue(all([0==expected_profit, 0==profits[0]]), msg = f"{indxs}")
##            self.assertEqual(indxs, [0,0])
        profits, buy_locs, sell_locs = intermitant(testlist)
        expected_profit = sum([testlist[i] for i in sell_locs])- sum([testlist[j] for j in buy_locs])
        self.assertCountEqual([True, True], [0==expected_profit, 0==profits[0]], msg = f"{buy_locs}, {sell_locs}")
        self.assertEqual(0, sum(locs_to_01(buy_locs, sell_locs, 4)), msg="testing locs to 01")
#        self.assertEqual(buy_locs[0], [0])
#        self.assertEqual(sell_locs[0], [0])
        
        
    def test_vals_dec_hold(self):
        "values either decrease or stay constant"
        testlist = list(range(48,24,-1))
        testlist[4:8] = [44]*4
        testlist[16:21] = [30]*5
#        for fn in [best_of_block1T, best_of_block2T, best_of_block3]:
#            profits, indxs = fn(testlist)
#            expected_profit = sum([testlist[i[1]] for i in indxs])- sum([testlist[j[0]] for j in indxs])
#            self.assertTrue(all([0==expected_profit, 0==profits[0]]), msg = f"{indxs}")
##            self.assertEqual(indxs, [0,0])
        profits, buy_locs, sell_locs = intermitant(testlist)
        expected_profit = sum([testlist[i] for i in sell_locs])- sum([testlist[j] for j in buy_locs])
        self.assertCountEqual([True, True], [0==expected_profit, 0==profits[0]], msg = f"{buy_locs}, {sell_locs}")
        self.assertEqual(0, sum(locs_to_01(buy_locs, sell_locs, 4)), msg="testing locs to 01")
#        self.assertEqual(buy_locs[0], [0])
#        self.assertEqual(sell_locs[0], [0])
            
    @hr_block_decorator
    def test_vals_inc(self, hr_capacity = 4):
        "values only increase"
        testlist = list(range(24,48))
        true_prft = 23*hr_capacity - hr_capacity*(hr_capacity - 1)
#        for fn in [best_of_block1T, best_of_block2T, best_of_block3]:
#            profits, indxs = fn(testlist)
#            expected_profit = sum([testlist[i[1]] for i in indxs])- sum([testlist[j[1]] for j in indxs])
#            self.assertCountEqual([True, True], [true_prft==expected_profit, true_prft==profits[0]], msg = f"{indxs}")
#            self.assertEqual(indxs, [0,23])
        profits, buy_locs, sell_locs = intermitant(testlist, hr_capacity = hr_capacity)
        expected_profit = sum([testlist[i] for i in sell_locs[0]])- sum([testlist[j] for j in buy_locs[0]])
        self.assertCountEqual([True, True], [true_prft==expected_profit, true_prft==profits[0]], 
                              msg = f"{buy_locs}, {sell_locs}, {true_prft}, {expected_profit}, {profits}")
        self.assertCountEqual(buy_locs[0], list(range(hr_capacity)))
        self.assertCountEqual(sell_locs[0], list(range(23,23-hr_capacity,-1)))
        self.assertEqual(0, sum(locs_to_01(buy_locs, sell_locs, 4)))

    @hr_block_decorator
    def test_vals_inc_hold(self, hr_capacity = 1):
        "values either increase or hold constant"
        testlist = list(range(24,48))
        testlist[4:8] = [28]*4
        testlist[16:21] = [42]*5
        true_prft = sum(testlist[-hr_capacity:])- sum(testlist[:hr_capacity])
#        for fn in [best_of_block1T, best_of_block2T, best_of_block3]:
#            profits, indxs = fn(testlist)
#            expected_profit = sum([testlist[i[1]] for i in indxs])- sum([testlist[j[1]] for j in indxs])
#            self.assertCountEqual([True, True], [true_prft==expected_profit, true_prft==profits[0]], msg = f"{indxs}")
#            self.assertEqual(indxs, [list(range(hr_capacity)),list(range(23,23-hr_capacity,-1))] )
     
        profits, buy_locs, sell_locs = intermitant(testlist, hr_capacity = hr_capacity)
        expected_profit = sum([testlist[i] for i in sell_locs[0]])- sum([testlist[j] for j in buy_locs[0]])
        self.assertCountEqual([True, True], [true_prft==expected_profit, true_prft==profits[0]],\
                          msg = f"{buy_locs}, {sell_locs}, {true_prft},{expected_profit}, {profits}")
        self.assertCountEqual(buy_locs[0], list(range(hr_capacity)))
        self.assertCountEqual(sell_locs[0], list(range(23,23-hr_capacity,-1)))
        self.assertEqual(0, sum(locs_to_01(buy_locs, sell_locs, 4)))

    @day_decorator
    def test_best_of_day_r_vs_intermitant(self, rates):#handle duplicate values differently
        "are getting best high/low prices in a period"
        profit_r, ix_r =  best_of_day_r(rates = rates, cycles_per_day = 1, hr_capacity = 1, into_blocks = False)
        profits, buy_locs, sell_locs = intermitant(rates, cycles_per_day = 1, hr_capacity = 1, into_blocks = False)
        self.assertEqual(profit_r, profits[0])#, msg =f"profits failed at day {i}")
#        self.assertEqual([rates[i] for i in ix_r], [rates[buy_locs[0]], rates[sell_locs[0]]])#, msg =f"indx failed at day {i}")
            #allowing diff indxs as long as rates val constant


#    @day_decorator#intermitant can't yet cycle multiple times; is expected to fail. best_of_r is wrong for cylcing.
#    def test_best_of_day_r_vs_intermitant_multiple(self, rates):
#        "are getting best high/low prices in a period; multiple cycles"
#        hr_capacity = self.hr_capacity#doesn't actually change hr_cap currently
#        for num_cycles in range(2, (24-hr_capacity*2)//2 - 1):#-1 for offset
##            num_cycles = 1#haven't yet updated this
#            profit_r, ix_r =  best_of_day_r(rates = rates, cycles_per_day = num_cycles, hr_capacity = 1, into_blocks = False)
#            profits, buy_locs, sell_locs = intermitant(rates, cycles_per_day = num_cycles, hr_capacity = 1, into_blocks = False)
#            self.assertEqual(profit_r, profits[0], msg = str(num_cycles) + " Cycles were run")
#            self.assertEqual(ix_r, [buy_locs[0], sell_locs[0]], msg = str(num_cycles) + " Cycles were run")


tzt = [int(i) for i in re.findall("\d+", '[79 71 63 62 52 63 36 27  9 97 78 69 37 67  2 48 13 23 99 65 35 14 52 49]')]
#tztbat = TestBattery()
#tztbat.test_intermitant_vals()
#tztbat.test_indxs_best_itermitant()
#print(tztbat.hourly_rates[:24])
#tztbat.test_insert_sort_duplicates()

#print("\nasdf")
#rt = tztbat.self.hourly_rates
unittest.main()
#%%
 f = open("hourly_rates.txt", "r")#need to wipe old data, "a" = append instead of overwrite
instr = f.read()
f.close()
print([instr[i] for i in []])
instr = [int(i) for i in re.findall('\d+', instr)]
print([instr[i] for i in [411,414,426]])
#print(best_of_block1T(rates = instr[528:552]), best_of_block3(rates = instr[528:552]))
best_of_block1T(instr[408:432])
best_of_block3(instr[408:432])

#%%
tztclass = TestBattery()
tztclass.foo()
print("\n\n\n\n\n\n\n")
tztclass.main()
#%%






#a = [3,4,5,2,1,3,9,8]
#a.sort(reverse = True)
#b = [3,4,5,2,1,3,9,8]
#b.sort()
#print(a,b)
#print(
#      insert_sort(a,100),a,
#      insert_sort(a,0),a,"\n",
#      insert_sort(b,100, buying = False),b,
#      insert_sort(b,0, buying =False),b)


#%%
#    by_day_up = {i:(np.zeros(), np.zeros) for i in range(num_days - 1)}# for j in range(i*24, i*24+24) if inflect_up[j]}
    

#    by_day_down = {i:(j,rates[j]) for i in range(num_days - 1) for j in range(i*24, i*24+24) if inflect_down[j]}
#    by_day_up[num_days] = {i:(j,rates[j]) for j in range((num_days-1)*24, len(rates)) if inflect_down[j]}
#    by_day_down[num_days] = {i:(j,rates[j]) for j in range((num_days-1)*24, len(rates)) if inflect_down[j]}
#    by_day_up_ix = {i:np.where(by_day_up[i] == True) for i in range(num_days)}
#    by_day_down_ix = {i:np.where(by_day_down[i] == True) for i in range(num_days)}


def bs_in_period(buy_sell):
    #need to update for end of period being maximium price
    "Takes a list of alternating buy sell prices and gives you best time; assumes numbers positive < 2**32"
    #best buy,sell locations will be at the highest selling price after the absolue lowest of the series or lowest buying price in region before absolute highest selling price
    #would be best factoring in c++ code, way slower in python. Look at C-python? cython?
    bi, si = 0, 0
    b_low, s_high = 2**32, -1028
    for i in range(0, len(buy_sell), 2):
        if buy_sell[i] < b_low:#only increment if higher buy later 
            bi = i
            b_low = buy_sell[i]
    for i in range(1,len(buy_sell), 2):
        if buy_sell[i] >= s_high:#if a later value at least as large then update
            si = i
            s_high = buy_sell[i]
    local_b, local_s = 2**32, -1028
    for i in range(bi+1,len(buy_sell),2):#from absolute min buy to end
        local_s = max(local_s, buy_sell[i])
    for i in range(0,si,2):#from start to absolute max sell, best place
        local_b = max(local_b, buy_sell[i])
    return max(local_s - b_low, s_high - local_b)








