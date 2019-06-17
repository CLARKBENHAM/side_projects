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
#%%
def bs_2_series(mins, maxs, prices=blocked_rates, sell_twice = False, i = None):
    "pass in a sequence of buys PRIOR to time of sells; \
    gets highest sell after min buying price vs. lowest buying price before max selling price\
    MUST have predefined a blocked_rates variable"
    #issue of different lengths
    #below occurs when prices only increase or decrease all day; will only be selling once regardless if have "selltwice" twice
    if len(mins) ==0:
        mins = [min(prices[i*24:i*24+23])]#since looking at the actual prices per mwh have to give time to sell 
        sell_twice = False
    if len(maxs) == 0:    
        maxs = [max(prices[i*24+1:i*24+24])]
        sell_twice = False
    min_b = min(mins)
    max_s = max(maxs)
    if isinstance(mins, list):    
        bi = mins.index(min_b)
        si = maxs.index(max_s)
    else:
        bi = int(np.where(mins == min_b)[0][0])#where returns a tuple
        si = int(np.where(maxs == max_s)[0][-1])
    #print(maxs, si, mins, bi, "\n")
    todays_profit = max(max(maxs[bi:], default = -2*32-1) - min_b, max_s - min(mins[:si+1], default = 2*32-1))#can sell at or later than time of buy
    if sell_twice:
        profit2, min2, s2i = bs_2_series(mins[:bi] + mins[bi+1:], maxs[:si] + maxs[si+1:])
        later_sale = max(si, s2i)
        return todays_profit + profit2, min(mins[later_sale+1:], default = 2**32-1), later_sale#default if sold at end of period
    return todays_profit, min(mins[si+1:], default = 2**32-1), si #since can check up to index of selling value 
#returns lowest buying price after sold, highest selling price
    
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
#    times2[i] = timeit.timeit(f"[sum(tezt[j:j+{i}])/{i} for j in range(len(tezt) - {i})]", setup = "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; import random; import math; tezt = np.random.randint(0, high = 10, size = 744); from __main__ import best_of_block2;from __main__ import n_ahead_ave;", number = 100)
def block(hourly_rates, n, ret_array = True):
    if ret_array:
        return np.array([np.mean(hourly_rates[i:i+n]) for i in range(len(hourly_rates) - n)])
    else:
        return [sum(hourly_rates[i:i+n])/n for i in range(len(hourly_rates) - n)]
       
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
 
#%%Trying to get a faster implementation of above
#pre-define inflect_up?Grib
#9.513454200001433
#hourly_rates = np.random.randint(0, high = 100, size = 744)#evaluate months at a time
#hourly_rates = np.random.choice(1000, 48, replace = False )

def best_of_block1(hourly_rates = hourly_rates):
    blocked_rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])#xrange for py2
    increasing = np.array(blocked_rates[:-1] < blocked_rates[1:])#does NOT work with lists
    inflect_up, inflect_down = np.zeros(len(blocked_rates), dtype = bool), np.zeros(len(blocked_rates), dtype = bool)
    
    inflect_up[1:-1] = np.array(increasing[:-1] < increasing[1:])#not increasing then is increasing; apparent have to wrap in list
    inflect_down[1:-1] = np.array(increasing[:-1] > increasing[1:])
    inflect_up[np.array([0,-1])] = [increasing[0], False]#buy on first hour if increases after that, don't buy on last 
    inflect_down[np.array([0,-1])] = [False, increasing[-1]]#can't sell while empty; sell on last if increased up to that point
    
    num_days = int(math.ceil(len(blocked_rates)/24))   
    profits = np.zeros(num_days, dtype = np.float32)   
    prev_night_low = 2**32-1
    sell_twice = False
    
    return blocked_rates[-24:][inflect_up[-24:]], blocked_rates[-24:][inflect_down[-24:]]#np.where(inflect_up == True)[0]#possible_profits#(profits)
        #grib write fn to do that
    
    for i in range(1,num_days):
#        print(blocked_rates[i*24:i*24+24][inflect_up[i*24:i*24+24]], blocked_rates[i*24:i*24+24][inflect_down[i*24:i*24+24]])
        if profits[i-1] < blocked_rates[goto_best_sell(inflect_up, inflect_down, i*24)] - prev_night_low:#price you can buy at increased so much over 24hr mark from last night to today
            profits[i-1] = 0#only charge yesterday to sell twice today
#            possible_buys[i+1].insert(0, prev_night_low)
            inflect_up[i*24-1] = True#add this as potential buy
            sell_twice = True
#            profits[i] = bs_2_series(possible_buys[i], possible_sells[i])
#        profits[i], prev_night_low, _ = bs_2_series(possible_buys[i], possible_sells[i], i = i, prices = blocked_rates, sell_twice = sell_twice)
        #can alway 
        if sum(inflect_up[i*24:i+24]) == 0:#buy at start of period
            inflect_up[i*24] = True
            #print("buy at start")
        if sum(inflect_down[i*24:i*24+24]) == 0:
            inflect_down[i*24+23] = True
            #print("sell at end")
        #try:
        profits[i], _, _ = bs_2_series(blocked_rates[i*24-sell_twice:i*24+24][inflect_up[i*24-sell_twice:i*24+24]], blocked_rates[i*24:i*24+24][inflect_down[i*24:i*24+24]])#slices dont include end; add extra buying location, but not selling
        #finally:
        #    print(i, sell_twice, blocked_rates[i*24-sell_twice:i*24+24])
        sell_twice = False        

        
        #doesn't work in case where there's no max withen period
        
        #possible_buys[i] = blocked_rates[i*24:i*24+24][inflect_up[i*24:i*24+24]]
        #possible_sells[i] = blocked_rates[i*24:i*24+24][inflect_down[i*24:i*24+24]]
        
    #possible_profits = blocked_rates[inflect_down] - blocked_rates[inflect_up]
    
        #below for max of trends
#        num_possible = np.sum(inflect_up[i*24:i*24+24])
#        profits[i] = np.max(possible_profits[day:day + num_possible])#can only charge once per day(could discharge twice)
#        day += num_possible

best_of_block1()
#%%
#Gribb, DOES NOT WORK
def best_of_block2(hourly_rates = hourly_rates): 
    "Python2 version when no longer supported; needs to take out numpy"
    blocked_rates = [sum(hourly_rates[i:i+4])/4 for i in range(len(hourly_rates) - 4)]#write better that increments average
    mn = 2^63 - 1
    mx = -1024
    was_increasing, was_decreasing = False, True#can buy first
    num_days = int(math.ceil(len(blocked_rates)/24))
#    possible_buys = {i:[blocked_rates[i*24]] for i in range(num_days)}#you can always buy at the start of the day
#    possible_sells = {i:[blocked_rates[min(i*24+23, len(blocked_rates)-1)]] for i in range(num_days)}#or sell at end of period
    possible_buys = {i:[] for i in range(num_days)}#you can always buy at the start of the day
    possible_sells = {i:[] for i in range(num_days)}#or sell at end of period
 
    #change checking i%24 ==0 in loop?
    
    #possible_profits = {i:[] for i in range(num_days)}
    profits = [0]*num_days#{i:0 for i in range(num_days + 1)}
    for i in range(len(blocked_rates)-1):
        #adding possible sells twice.
        if blocked_rates[i] < blocked_rates[i+1]:#increasing
            if was_decreasing:
                mn = blocked_rates[i]#swapped from increasing to decreasing; must be local min
            else:
                #if was same price would buy 'later' in day, if decreasing from 1->2 don't buy now
                if i %24 ==0:
                    possible_buys[i//24].append(blocked_rates[i])
                    mn = 2**32-1#reset min
            was_increasing = True
            was_decreasing = False
            
        elif blocked_rates[i] > blocked_rates[i+1]:#decreasing
            if was_increasing:
                mx = blocked_rates[i]
                if mn < 2^31:
                    possible_buys[i//24].append(mn)#this could be from the previous day.
                    
                    possible_sells[i//24].append(mx)#this evaluates at the time of selling                    
                    #possible_profits[i//24].extend((mn,mx))
                    mn = 2**32 - 1
                #make dict and compare key each time; must dis and re charge in same day
            else:
                if i%24 == 23:#last hour in day; could buy
                    possible_sells[i//24].append(blocked_rates[i])
            was_increasing = False
            was_decreasing = True    
        #if is level will trade at the later instance

        
    #for last entity
    if was_increasing:#then would have bought, will sell on last day
        possible_buys[num_days - 1].append(mn)
        possible_sells[num_days - 1].append(blocked_rates[-1])  
        #possible_profits[num_days].extend((mn,mx))
        
#    for i in range(num_days):#optimisation looks at ending points for possible sales 
#        print(possible_sells[i], "changed to")
#        try:
#            possible_sells[i] = np.concatenate((possible_sells[i][1:], [possible_sells[i][0]]))
#            print(possible_sells[i])
#        except:
#            pass#if there's only 1 possible sell date in a day then don't need to change it
#         
        #possible_sells[i][0], possible_sells[i][-1] = possible_sells[i][-1], possible_sells[i][0] 
    return possible_buys

    return list(possible_buys[num_days-1]), list(possible_sells[num_days-1])

    #get from possible to best       
    prev_night_low = 2**63-1
#    prev_night_high = -2**63-1
    sell_twice = False
    profits[0], _, _ = bs_2_series(possible_buys[0], possible_sells[0], i = 0, prices = blocked_rates)
    for i in range(1, num_days):#2.912781700026244; 8.423050199984573
        if profits[i-1] < blocked_rates[i*24] - prev_night_low:#price you can buy at increased so much over 24hr mark from last night to today
            profits[i-1] = 0#only charge yesterday to sell twice today
            possible_buys[i].insert(0, prev_night_low)#got yesterday's low for today
            sell_twice = True
#            profits[i] = bs_2_series(possible_buys[i], possible_sells[i])
        profits[i], prev_night_low, _ = bs_2_series(possible_buys[i], possible_sells[i], i = i, prices = blocked_rates, sell_twice = sell_twice)
        sell_twice = False
#        elif profits[i-1] < prev_night_high - possible_sells[i][0]:
#            profits[i-1] = prev_night_high - possible_sells[i][0] #waited, while charged, to sell this morning
#            possible_sells[i][0] = -1024         
#        prev_night_low = night_low
#        prev_night_high = night_high
    #if not way1: 
    #    for i in range(num_days):#29.437535100034438, 3.14; 9.014132300042547
    #        profits[i] = bs_in_period(possible_profits[i])
    #return profits#, possible_sells#possible_profits.values()
best_of_block2()
#for i in zip(best_of_block1(), best_of_block2()):
#    
#    for j,k in zip(i[0], i[1]):
#        print(j,k)
#    print("\n")
#for b1, b2 in zip(*best_of_block1(), *best_of_block2()):
#    print(b1, b2)
#%%             
#for i,j in zip(times, times2):
#    print(i < j)
#timeit.timeit("best_of_block2()", setup = "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; import random; import math; hourly_rates = np.random.randint(0, high = 10, size = 744); from __main__ import best_of_block2;from __main__ import best_of_block1;", number = 3000)
#print(np.array(best_of_block2()) - np.array(best_of_block1()))
#for i,j in zip(best_of_block1(), best_of_block2()):
#    print(i,j)
#possible_buys, possible_sells = best_of_block2()
#possible_profits = best_of_block2(way1 = False)
#best_of_block2(way1 = True) == best_of_block2(way1 = False) #== [5.0, 7.75, 6.75, 6.75, 8.25, 8.25, 6.25, 7.75, 6.25, 6.25, 5.75, 5.25, 5.75, 7.25, 6.25, 5.0, 7.5, 7.25, 6.75, 6.75, 7.25, 6.5, 5.5, 6.5, 6.0, 7.75, 5.25, 6.75, 5.75, 5.0, 5.0, 7.25, 8.5, 4.75, 6.5, 8.0, 7.75, 4.5, 6.5, 7.0, 5.5, 6.25, 6.25, 5.5, 4.75]
#np.where(best_of_block1() != best_of_block2())
best_of_block2()

#%%
#5.4, 6.2; where is way faster than converting to a list
timeit.timeit("np.where(a == 49)[0]", setup = "import numpy as np; a = np.random.choice(50,50,replace = False)", number = 1000000)
#%%
a,b,c,d,e,f = 40,3,4,50,0,25
sell_buy = [40,3,4,50,0,25]
buy_sell = {}
buy_sell = (a,b,c,d,e,f)
a = 1
sell_buy[0] = 1
print(buy_sell[0],a)
#for i,j in zip(sell_buy, buy_sell):
#    print(i==j)
#assumes same number of local min/max in same "day"

    
#%%
def best_of_block3(rates = hourly_rates, hr_capacity = 4, into_blocks = True):
    if into_blocks:#blocking all at once
        rates = block(rates, 4)#Can't block rates by day, need to do all at once
    num_days = math.ceil(len(rates)/24)
    profit = [0]*num_days
    extra_charges = 0
    indxs = [0]*num_days
    for i in range(num_days):
        profit[i], indxs[i] = best_of_day_r(rates = rates[i*24:min(i*24+24,len(rates))], hr_capacity = hr_capacity, extra_charges = extra_charges, into_blocks = False)        
#        next_day_jmp = rates[i*24+1] - rates[i*24]
#        if next_day_jmp > last_discharge_profit:
#            profit[i] -= last_discharge_profit
#            extra_charge = True
            
    #for last iteration
    #profit[i] = best_of_day_r(rates = rates[len(rates) - (num_days-1)*24:], hr_capacity = hr_capacity, extra_charge = extra_charge, into_blocks =False)        
    return profit, [i for j in indxs for i in j]
best_of_block3(hourly_rates[:72])
#        
#        if rates[i] < best_low_hi[0]:#found new best low
#            best_low_hi[0] = rates[i]
#            best_low_hi[1] = -2**32
#        if rates[i] < low_best_hi[0]:#found better low after best high
#            low_best_hi[0] = rates
#            
#        if rates[i] > low_best_hi[1]:#found new best low
#            low_best_hi[1] = rates[i]
#            low_best_hi[0] = 2**32-1
#        if rates[i] > best_low_hi[1]:#better low after best high
#            best_low_hi[1] = rates[i]  
#        if i%24 == 23:#looked at last prices at end of day
#            profit[i//24] += max([i[1] - i[0] for i in (low_best_hi, best_low_hi)])#always has to be >0? only error if constant till last which is greater or first was least then constant
#        if profit[i//24] < rates[i+1] - best_low_hi[0]:#it's more profitable to only charge today to sell twice tommorow
#            profit[i//24] = 0
#            profit[i//24 + 1] += rates[i+1] - best_low_hi[0]#and can call twice
#        else:#how to deal with always decreasing/increasing?
#            best_low_hi = (rates[i], -2*32)#in case it always increases from here
#            low_best_hi[1] = (2**32-1, rates[min(i+23, len(rates)-1)])# in case it always decreases from here

#%%
def best_of_day_r(rates = hourly_rates, hr_capacity = 4, extra_charges = 0, into_blocks = True):
    #O(24*hr_capacity); has to copy over each array for each recursion.
    if into_blocks:
        profit, indxs =  best_of_day_r(block(rates, hr_capacity), hr_capacity = 1, into_blocks = False)
        b_locs = [j for i in indxs[::2] for j in range(i,i+hr_capacity)]
        s_locs = [j for i in indxs[1::2] for j in range(i,i+hr_capacity)]
        return profit, [var for pair in zip(b_locs,s_locs) for var in pair]#averages are "combined" into the present hour from future hours.
    #This WILL go wrong
    
#    best_low_hi = [[2**32-1,0, -2**32, 0] for i in range(hr_capacity)]#max low, indx, high after low,indx
#    low_best_hi = [[2**32-1,0, -2**32,0] for i in range(hr_capacity)]#low before max high,indx, mx hi, indx
#    #to add more hours add more of the above and sequentially fill
    if extra_charges == 0:
        best_low_hi = [2**32-1, -2**32]
        low_best_hi = [2**32-1, -2**32] 
        best_low_hi_ix, low_best_hi_ix = [0,0], [0,0]
        for i in range(len(rates)):
    #        print(rates[i], best_low_hi[0])
            if rates[i] < best_low_hi[0]:#found new best low
                best_low_hi[0] = rates[i]
                best_low_hi[1] = -2**32
                best_low_hi_ix = [i, None]
    
            if rates[i] > low_best_hi[1]:#found new best high
                low_best_hi[1] = rates[i]
                low_best_hi[0] = best_low_hi[0]#become best low seen up to this point; is fixed till best high changed
                low_best_hi_ix = [best_low_hi_ix[0], i]
                
            if rates[i] > best_low_hi[1]:#better high after best low
                best_low_hi[1] = rates[i]    
                best_low_hi_ix[1] = i
    #        print(best_low_hi, low_best_hi)
    
    else:#THIS IS NOT RIGHT as it assumes can store an unlimited amount of energy: first thing w/ free charge is fill the maximium daily price orders, then recur?      
        best_low_hi = [0, 0]#no effect
        low_best_hi = [0, max(rates)] 
        best_low_hi_ix, low_best_hi_ix = [None,None], [None, rates.index(max(rates))]
    #how to calculate losing a charge?    
    
    options = [i[1] - i[0] for i in [low_best_hi, best_low_hi]]
    day_best = max(options)
    if options[0] == day_best:
        ix_used = list(low_best_hi_ix)#
    else:
        ix_used = list(best_low_hi_ix)
    print(day_best, ix_used)
    
    if hr_capacity == 1:
            return day_best, ix_used 
    else:          
#        indxs = set((*low_best_hi_ix, *best_low_hi_ix))#has <= 4 elements
#        indxs.discard(None)#removes None if it exists
        if isinstance(rates, list):
            for ix in ix_used:#should never have none; have to adjust when add carrying multiple charges per day
                del rates[ix]
        else:#is np array
            rates = np.delete(rates, list(ix_used))#doesn't occur in place    
        day_best_r, ix_used_r = best_of_day_r(rates = rates, hr_capacity = hr_capacity-1, into_blocks = into_blocks)
        return day_best + day_best_r, ix_used + ix_used_r
tezt = blocked_rates[:24]
best_of_day_r(rates = tezt, hr_capacity = 4, into_blocks = True)
#best_of_day(rates = c, into_blocks = False)
##tezt = blocked_rates[:24]
##for i in tezt:
##    print(i)
##best_of_day(rates = tezt, hr_capacity = 1, as_one = False)
#%%
       


#%%
def best_of_day_i(rates = hourly_rates, hr_capacity = 4, into_blocks = True):
    #to add more hours add more of the above and sequentially fill?
    if into_blocks:
        return best_of_day(block(rates, hr_capacity), hr_capacity = 1, into_blocks = False)
    
    best_low_hi = [2**32-1,-2**32]#max low,  high after low
    low_best_hi = [2**32-1,-2**32]#low before max high
    #to add more hours add more of the above and sequentially fill
    changes = 0
    for i in range(len(rates)):

#        print(rates[i], best_low_hi[0])
        if rates[i] < best_low_hi[0]:#found new best low
            best_low_hi[0] = rates[i]
            best_low_hi[1] = -2**32
#        if rates[i] < low_best_hi[0] and low_best_hi[1] == -2**32:#found better low after best high
#            low_best_hi[0] = rates[i]
        
        if rates[i] > low_best_hi[1]:#found new best high
            low_best_hi[1] = rates[i]
            low_best_hi[0] = best_low_hi[0]#become best low seen up to this point; is fixed till best high changed
        if rates[i] > best_low_hi[1]:#better high after best low
            best_low_hi[1] = rates[i]     
        print(best_low_hi, low_best_hi)
    return max([i[1] - i[0] for i in [low_best_hi, best_low_hi]])
best_of_day_i(rates = tezt, hr_capacity = 6, into_blocks = False)

    #%%
a = list(range(10))
b = [-1,-2,-3]
b[0] = a[1] 
#%%
def bs_in_period(buy_sell = buy_sell):
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

#%%######################################################################
#scenario: battery can partially charge/discharge at will, up to 4 times per day
#can't just use current as that relies on changes at lowest. 

#check near changes near bottom, at top.
#have all local minimia, maxima; need to "hunt around" those points.

increasing = np.array(hourly_rates[:-1] < hourly_rates[1:])#does NOT work with lists

#pre-define inflect_up?Grib

inflect_up = list(increasing[:-1] < increasing[1:])#not increasing then is increasing; apparent have to wrap in list
inflect_up = [increasing[0]] + inflect_up + [False]#buy on first hour if increases after that, don't buy on last
inflect_down = list(increasing[:-1] > increasing[1:])
inflect_down = [False] + inflect_down + [increasing[-1]]#can't sell while empty; sell on last if increased up to that point
#possible_trade = ["hold" if i == j else "charge" if i else "discharge" for i,j in zip(inflect_up, inflect_down)]
possible_profits = blocked_rates[inflect_down] - blocked_rates[inflect_up]
profits = np.zeros(len(hourly_rates)//24 + 1)
which_trade = np.zeros(len(hourly_rates)//24 + 1, dtype = int)
day = 0

for i in range(len(hourly_rates)//24 + 1):
    num_possible = np.sum(inflect_up[i*24:i*24+24])
    profits[i] = np.max(possible_profits[day:day + num_possible])
    
    which_trade[i] = list(possible_profits[day:day + num_possible]).index(profits[i])
    which_trade[i] = i*24 + np.flatnonzero(np.array(inflect_up[i*24:i*24+24]) == True)[which_trade[i]]

    day += num_possible

#plotting
a = ["black" if i == j else "green" if i else "red" for i,j in zip(inflect_up, inflect_down)]
plot_lmt = 40
dayz = plot_lmt//24 #math.ceil(plot_lmt/24)
plt.bar(range(len(blocked_rates[:plot_lmt])), blocked_rates[:plot_lmt], color = a)
for i in range(len(blocked_rates[:plot_lmt])//24 + 1):#deliniates days
    plt.axvline(x = i*24)#change type? 
plt.scatter(which_trade[:dayz], profits[:dayz] + blocked_rates[which_trade[:dayz]])



#%%
#scenario: battery must completely charge then discharge, but can wait between doing so.(Reduction to knapsack problem?)
#%%    
num_days = math.ceil(len(hourly_rates)/24)
possible_sells = {i:[blocked_rates[min(i*24+23, len(blocked_rates)-1)]] for i in range(num_days)}

def test1():
    "append first then move"
    possible_sells = {i:[hourly_rates[min(i*24+23, len(hourly_rates)-1)]] for i in range(num_days+1)}
    for i, val in enumerate(hourly_rates): 
        possible_sells[i//24].append(val); 
    for i in range(num_days):
        possible_sells[i] = possible_sells[i][1:] + possible_sells[i][0]
    return possible_sells

def test2():
    "only append"
    possible_sells = {i:[] for i in range(num_days)}
    for i, val in enumerate(hourly_rates): 
        possible_sells[i//24].append(val); 
    for i in range(num_days):
        possible_sells[i] = possible_sells[i][1:] + possible_sells[i]
    for i in range(num_days):
        possible_sells[i].append(hourly_rates[min(i*24+23, len(hourly_rates)-1)])
    return possible_sells
    
def test3():
    "insert 2nd from back"
    possible_sells = {i:[hourly_rates[min(i*24+23, len(hourly_rates)-1)]] for i in range(num_days)}
    for i, val in enumerate(hourly_rates): 
        possible_sells[i//24].insert(-1, val); 
    return possible_sells

def test4():#Best
    "prefill, append and swap"
    possible_sells = {i:[hourly_rates[min(i*24+23, len(hourly_rates)-1)]] for i in range(num_days)}#actually buying range, but ignore for timeing
    for i, val in enumerate(hourly_rates): 
        possible_sells[i//24].append(val); 
    for i in range(num_days):
        possible_sells[i] = possible_sells[i][1:] + possible_sells[i]
        possible_sells[i][0], possible_sells[i][-1] = possible_sells[i][-1], possible_sells[i][0] 
    return possible_sells

print(
    timeit.timeit("test1()", \
              setup = "import numpy as np; \
              hourly_rates = np.random.randint(0, high = 10, size = 744);\
              num_days = 31;\
              from __main__ import test1",\
              number = 3000),
    timeit.timeit("test2()", \
              setup = "import numpy as np; \
              hourly_rates = np.random.randint(0, high = 10, size = 744);\
              num_days = 31;\
              from __main__ import test2",\
              number = 3000),
    timeit.timeit("test3()", \
              setup = "import numpy as np; \
              hourly_rates = np.random.randint(0, high = 10, size = 744);\
              num_days = 31;\
              from __main__ import test3",\
              number = 3000),
    timeit.timeit("test4()", \
              setup = "import numpy as np; \
              hourly_rates = np.random.randint(0, high = 10, size = 744);\
              num_days = 31;\
              from __main__ import test4",\
              number = 3000)), 
#%%
timeit.timeit("increasing = np.array(blocked_rates[:-1] < blocked_rates[1:], dtype = bool);\
    inflect_up = list(increasing[:-1] < increasing[1:]);\
    inflect_up = [increasing[0]] + inflect_up + [False];", 
    setup = "import numpy as np;\
    hourly_rates = np.random.randint(0, high = 100, size = 744);\
    blocked_rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])",
    number = 300000)#5.537616600002366

timeit.timeit("increasing = np.array(blocked_rates[:-1] < blocked_rates[1:], dtype = bool);\
    inflect_up = increasing[:-1] < increasing[1:];\
    inflect_up = np.concatenate(([increasing[0]],inflect_up, [False]));", 
    setup = "import numpy as np;\
    hourly_rates = np.random.randint(0, high = 100, size = 744);\
    blocked_rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])",
    number = 300000)#1.8055378999997629
#%%
#scrape
def tezt1(string,n):
    series = [[i] for i in string]#[""]*(len(string) - n)
    saved = string[:n]
    series[0] = saved
    for i in range(1,len(string) - n):
        saved[i%n] = series[i]
        series[i] = series[i-1] + series[i+n-1]
        print(series[i], saved[i])
    return series[:-n]
tezt1("asdfasdfasdf",4)

















