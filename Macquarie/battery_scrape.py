def best_of_block1T(rates = hourly_rates, into_blocks = True, hr_capacity = 4, cycles_per_day = 1):
    #Version which will be tested
    #should rename hr_capacity
    if into_blocks:
        rates = block(rates, hr_capacity)    
    #rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])#xrange for py2
   
    #below is how you would adjust for possibility of multiple charges across days
#    increasing = np.array(rates[:-1] < rates[1:])#does NOT work with lists
#    inflect_up, inflect_down = np.zeros(len(rates), dtype = bool), np.zeros(len(rates), dtype = bool)
#    
#    inflect_up[1:-1] = np.array(increasing[:-1] < increasing[1:])#not increasing then is increasing; apparent have to wrap in list
#    inflect_down[1:-1] = np.array(increasing[:-1] > increasing[1:])
#    inflect_up[np.array([0,-1])] = [increasing[0], False]#buy on first hour if increases after that, don't buy on last 
#    inflect_down[np.array([0,-1])] = [False, increasing[-1]]#can't sell while empty; sell on last if increased up to that point
    

    num_days = int(math.ceil(len(rates)/24))   
    profits = np.zeros(num_days)
    indxs = [[0,0]]*num_days#change this

    #also need to do indexs
    
#    sell_twice = False
    inflect_up, inflect_down = np.zeros(24, dtype = bool), np.zeros(24, dtype = bool)
    for i in range(num_days):#need to adjust for when len(rates)%24!=0
#        print(i, rates[i*24:i*24+24][inflect_up[i*24:i*24+24]], rates[i*24:i*24+24][inflect_down[i*24:i*24+24]])
#        #2nd derive = 0; but can only profit if prices are continously INCREASING; lets ignore this for now
#        if sum(inflect_up[i*24:i+24]) == 0:
#            inflect_up[i*24] = rates[i*24] <= rates[i*24+23]
#            inflect_up[i*24+23] = rates[i*24] > rates[i*24+23]
#        if sum(inflect_down[i*24:i*24+24]) == 0:
#            inflect_down[i*24] = rates[i*24] > rates[i*24+23]
#            inflect_down[i*24+23] = rates[i*24] <= rates[i*24+23] 

        day_ix = slice(i*24, min(i*24+24, len(rates)))#  
        inflect_up, inflect_down = get_loc_minmax1(rates[day_ix])
        up_indxs = np.where(inflect_up)[0]
        down_indxs = np.where(inflect_down)[0]
        
        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs], cycles = cycles_per_day)#slices dont include end; add extra buying location, but not selling
#        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs], rates[day_ix], into_blocks = False, cycles = cycles_per_day, hr_capacity = hr_capacity)#slices dont include end; add extra buying location, but not selling
        indxs[i] = [i*24 + up_indxs[day_minmax_ix[0]], i*24 + down_indxs[day_minmax_ix[1]]]
#        print(profits[i], indxs[i])
#    i = num_days - 1
#    profits[i], indxs[i] = bs_2_series(rates[i*24:][inflect_up[i*24:]], rates[i*24:][inflect_down[i*24:]])#
    return profits, indxs
    
#    return rates[-24:][inflect_up[-24:]], rates[-24:][inflect_down[-24:]]#np.where(inflect_up == True)[0]#possible_profits#(profits)
        #grib write fn to do that
best_of_block1T()



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
print(
    timeit.timeit("best_of_block1()", \
              setup = "import numpy as np; \
              hourly_rates = np.random.randint(0, high = 100, size = 744);\
              num_days = 31;\
              from __main__ import best_of_block1",\
              number = 300),#2.102505900002143
    timeit.timeit("best_of_block3()", \
              setup = "import numpy as np; \
              hourly_rates = np.random.randint(0, high = 10, size = 744);\
              num_days = 31;\
              from __main__ import best_of_block3",\
              number = 300))#3.1197811000020


#%%
#def best_of_block21(rates = hourly_rates, to_blocks = True, hr_capacity = 4, bs_2_series=bs_2_series): 
#    #DOES NOT go across 24hr blocks
#    "Python2 version when no longer supported; needs to take out numpy"
#    #using TF bool arr
#    if to_blocks:
#        rates = block(rates, hr_capacity)#[sum(hourly_rates[i:i+4])/4 for i in range(len(hourly_rates) - 4)]#write better that increments average
#    num_days = int(math.ceil(len(rates)/24))
##    possible_buys = {i:[rates[i*24]] for i in range(num_days)}#you can always buy at the start of the day
##    possible_sells = {i:[rates[min(i*24+23, len(rates)-1)]] for i in range(num_days)}#or sell at end of period
#    
##    possible_buys = {i:[] for i in range(num_days)}#you can always buy at the start of the day
##    buys_ix = {i:[] for i in range(num_days)}
##    possible_sells = {i:[] for i in range(num_days)}#or sell at end of period
##    sells_ix = {i:[] for i in range(num_days)}
#    profits = [0]*num_days#{i:0 for i in range(num_days + 1)}
#    indxs = [[0,0]]*num_days
#    for i in range(num_days):#need to adjust for when len(rates)%24!=0
#        day_ix = slice(i*24, min(i*24+24, len(rates)))#  
#        inflect_up, inflect_down = get_loc_minmax2(rates[day_ix], sep_days = False, locs = False)
#        up_indxs = np.where(inflect_up)[0]
#        down_indxs = np.where(inflect_down)[0]
#        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs])#slices dont include end; add extra buying location, but not selling
#        indxs[i] = [i*24 + up_indxs[day_minmax_ix[0]], i*24 + down_indxs[day_minmax_ix[1]]]
#    return profits, indxs
#best_of_block21()
#
#
#def best_of_block22(rates = hourly_rates, to_blocks = True, hr_capacity = 4, bs_2_series=bs_2_series): 
#    #DOES NOT go across 24hr blocks
#    "Python2 version when no longer supported; needs to take out numpy"
#    #using TF bool arr
#    if to_blocks:
#        rates = block(rates, hr_capacity)#[sum(hourly_rates[i:i+4])/4 for i in range(len(hourly_rates) - 4)]#write better that increments average
#    num_days = int(math.ceil(len(rates)/24))
##    possible_buys = {i:[rates[i*24]] for i in range(num_days)}#you can always buy at the start of the day
##    possible_sells = {i:[rates[min(i*24+23, len(rates)-1)]] for i in range(num_days)}#or sell at end of period
#    
##    possible_buys = {i:[] for i in range(num_days)}#you can always buy at the start of the day
##    buys_ix = {i:[] for i in range(num_days)}
##    possible_sells = {i:[] for i in range(num_days)}#or sell at end of period
##    sells_ix = {i:[] for i in range(num_days)}
#    profits = [0]*num_days#{i:0 for i in range(num_days + 1)}
#    indxs = [[0,0]]*num_days
#    loc_dict = get_loc_minmax2(rates, sep_days = True, locs = False)
#    for i in range(num_days):#need to adjust for when len(rates)%24!=0
#        day_ix = slice(i*24, min(i*24+24, len(rates)))#  
#        inflect_up, inflect_down = loc_dict[i]
#        up_indxs = np.where(inflect_up)[0]
#        down_indxs = np.where(inflect_down)[0]
#        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs])#slices dont include end; add extra buying location, but not selling
#        indxs[i] = [i*24 + up_indxs[day_minmax_ix[0]], i*24 + down_indxs[day_minmax_ix[1]]]
#    return profits, indxs
#
#def best_of_block23(rates = hourly_rates, to_blocks = True, hr_capacity = 4, bs_2_series=bs_2_series): 
#    #DOES NOT go across 24hr blocks
#    "Python2 version when no longer supported; needs to take out numpy"
#    #creats1 dict outside loop and uses loc
#    if to_blocks:
#        rates = block(rates, hr_capacity)#[sum(hourly_rates[i:i+4])/4 for i in range(len(hourly_rates) - 4)]#write better that increments average
#        
#    num_days = int(math.ceil(len(rates)/24))
#    profits = [0]*num_days#{i:0 for i in range(num_days + 1)}
#    indxs = [[0,0]]*num_days
#    loc_dict = get_loc_minmax2(rates, sep_days = True, locs = True)
#    for i in range(num_days):
#        min_loc, max_loc = loc_dict[i]
#        day_ix = slice(i*24, min(i*24+24, len(rates)))
#        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][min_loc], rates[day_ix][max_loc])#slices dont include end; add extra buying location, but not selling
#        indxs[i] = [i*24 + min_loc[day_minmax_ix[0]], i*24 + max_loc[day_minmax_ix[1]]]
#    return profits, indxs  
def best_of_block2T(rates = hourly_rates, into_blocks = True, hr_capacity = 4, cycles_per_day = 1): 
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
        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][min_loc], rates[day_ix][max_loc], cycles = cycles_per_day)#slices dont include end; add extra buying location, but not selling
        indxs[i] = [i*24 + min_loc[day_minmax_ix[0]], i*24 + max_loc[day_minmax_ix[1]]]
    return profits, indxs

#    for i in range(num_days):#need to adjust for when len(rates)%24!=0
#        day_ix = slice(i*24, min(i*24+24, len(rates)))#  
#      
#        profits[i], day_minmax_ix = bs_2_series(rates[possible_loc[i]], rates[possible_loc[i]])#slices dont include end; add extra buying location, but not selling
#        indxs[i] = [i*24 + up_indxs[day_minmax_ix[0]], i*24 + down_indxs[day_minmax_ix[1]]]

#for j in [False, True]:
#    for i in range(1,5):
#        how = ["bs_2_series", "best_of_day_r2"][j]
#        print(f"best_of_block2{i}(get_best= {how})",
#        timeit.timeit(f"best_of_block2{i}(bs_2_series= {how})", \
#                  setup = f"import numpy as np; \
#                  hourly_rates = np.random.randint(0, high = 100, size = 744);\
#                  num_days = 31;\
#                  from __main__ import best_of_block2{i};\
#                  from __main__ import bs_2_series;\
#                  from __main__ import best_of_day_r2",\
#                  number = 3))
#best_of_block21(get_best= bs_2_series) 226.99859100000322
#best_of_block22(get_best= bs_2_series) 218.99242300000333
#best_of_block23(get_best= bs_2_series) 218.3737761000084
#best_of_block24(get_best= bs_2_series) 219.95427420000487
#%%


def best_of_block1(rates = hourly_rates):#, into_blocks = True, block_sz = 4):
    
    "changed bs_2_series to required (preblocked) rates, hr_cap = 1"
    
    if True:
        rates = block(rates, 4)    
    #rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])#xrange for py2
   
    #below is how you would adjust for possibility of multiple charges across days
    increasing = np.array(rates[:-1] < rates[1:])#does NOT work with lists
    inflect_up, inflect_down = np.zeros(len(rates), dtype = bool), np.zeros(len(rates), dtype = bool)
    
    inflect_up[1:-1] = np.array(increasing[:-1] < increasing[1:])#not increasing then is increasing; apparent have to wrap in list
    inflect_down[1:-1] = np.array(increasing[:-1] > increasing[1:])
    inflect_up[np.array([0,-1])] = [increasing[0], False]#buy on first hour if increases after that, don't buy on last 
    inflect_down[np.array([0,-1])] = [False, increasing[-1]]#can't sell while empty; sell on last if increased up to that point
    

    num_days = int(math.ceil(len(rates)/24))   
    profits = np.zeros(num_days)
    indxs = [[0,0]]*num_days#change this

    #also need to do indexs
    
#    sell_twice = False
    inflect_up, inflect_down = np.zeros(24, dtype = bool), np.zeros(24, dtype = bool)
    for i in range(num_days):#need to adjust for when len(rates)%24!=0
#        print(i, rates[i*24:i*24+24][inflect_up[i*24:i*24+24]], rates[i*24:i*24+24][inflect_down[i*24:i*24+24]])
#        #2nd derive = 0; but can only profit if prices are continously INCREASING; lets ignore this for now
#        if sum(inflect_up[i*24:i+24]) == 0:
#            inflect_up[i*24] = rates[i*24] <= rates[i*24+23]
#            inflect_up[i*24+23] = rates[i*24] > rates[i*24+23]
#        if sum(inflect_down[i*24:i*24+24]) == 0:
#            inflect_down[i*24] = rates[i*24] > rates[i*24+23]
#            inflect_down[i*24+23] = rates[i*24] <= rates[i*24+23] 
        
        day_ix = slice(i*24, min(i*24+24, len(rates)))#        
        up_indxs = np.where(inflect_up[day_ix])[0]
        down_indxs = np.where(inflect_down[day_ix])[0]
        
        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs])#slices dont include end; add extra buying location, but not selling
        indxs[i] = [i*24 + up_indxs[day_minmax_ix[0]], i*24 + down_indxs[day_minmax_ix[1]]]
        print(indxs[i], indxs[i][0] < indxs[i][1], "\n", )
#        print(profits[i], indxs[i])
#    i = num_days - 1
#    profits[i], indxs[i] = bs_2_series(rates[i*24:][inflect_up[i*24:]], rates[i*24:][inflect_down[i*24:]])#
    return profits, indxs
best_of_block1()


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
#%%
#for i,j in zip(sell_buy, buy_sell):
#    print(i==j)
#assumes same number of local min/max in same "day"
g = [16, 20, 0, 12, 6, 10, 14, 16]
b_locs = [tezt[i] for i in g[::2]]
s_locs = [tezt[i] for i in g[1::2]]
diff = 0
for i in range(4):
    diff += b_locs[i] - s_locs[i]
    print(b_locs[i], s_locs[i])
print(diff)
#%%
cycles_per_day




















#%% #######################Trying to account for varying when can charge
#doesn't work yet
    break
def best_of_block1(rates = hourly_rates):#, into_blocks = True, block_sz = 4):
    if True:
        rates = block(rates, 4)    
    #rates = np.array([np.mean(hourly_rates[i:i+4]) for i in range(len(hourly_rates) - 4)])#xrange for py2
   
    #below is how you would adjust for possibility of multiple charges across days
#    increasing = np.array(rates[:-1] < rates[1:])#does NOT work with lists
#    inflect_up, inflect_down = np.zeros(len(rates), dtype = bool), np.zeros(len(rates), dtype = bool)
#    
#    inflect_up[1:-1] = np.array(increasing[:-1] < increasing[1:])#not increasing then is increasing; apparent have to wrap in list
#    inflect_down[1:-1] = np.array(increasing[:-1] > increasing[1:])
#    inflect_up[np.array([0,-1])] = [increasing[0], False]#buy on first hour if increases after that, don't buy on last 
#    inflect_down[np.array([0,-1])] = [False, increasing[-1]]#can't sell while empty; sell on last if increased up to that point
    

    num_days = int(math.ceil(len(rates)/24))   
    profits = np.zeros(num_days)
    indxs = [[0,0]]*num_days#change this

    #also need to do indexs
    
#    sell_twice = False
    inflect_up, inflect_down = np.zeros(24, dtype = bool), np.zeros(24, dtype = bool)
    for i in range(num_days):#need to adjust for when len(rates)%24!=0
#        print(i, rates[i*24:i*24+24][inflect_up[i*24:i*24+24]], rates[i*24:i*24+24][inflect_down[i*24:i*24+24]])
#        #2nd derive = 0; but can only profit if prices are continously INCREASING; lets ignore this for now
#        if sum(inflect_up[i*24:i+24]) == 0:
#            inflect_up[i*24] = rates[i*24] <= rates[i*24+23]
#            inflect_up[i*24+23] = rates[i*24] > rates[i*24+23]
#        if sum(inflect_down[i*24:i*24+24]) == 0:
#            inflect_down[i*24] = rates[i*24] > rates[i*24+23]
#            inflect_down[i*24+23] = rates[i*24] <= rates[i*24+23] 

        day_ix = slice(i*24, min(i*24+24, len(rates)))#  
        inflect_up, inflect_down = get_loc_minmax1(rates[day_ix])
        up_indxs = np.where(inflect_up)[0]
        down_indxs = np.where(inflect_down)[0]
        
        profits[i], day_minmax_ix = bs_2_series(rates[day_ix][up_indxs], rates[day_ix][down_indxs])#slices dont include end; add extra buying location, but not selling
        indxs[i] = [i*24 + up_indxs[day_minmax_ix[0]], i*24 + down_indxs[day_minmax_ix[1]]]
#        print(profits[i], indxs[i])
#    i = num_days - 1
#    profits[i], indxs[i] = bs_2_series(rates[i*24:][inflect_up[i*24:]], rates[i*24:][inflect_down[i*24:]])#
    return profits, indxs
    
#    return rates[-24:][inflect_up[-24:]], rates[-24:][inflect_down[-24:]]#np.where(inflect_up == True)[0]#possible_profits#(profits)
        #grib write fn to do that

    #%%
    #for dealing with extra charges across days; continuation of above
    prev_night_low = 2**32-1
    sell_twice = False
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


#%%
#Gribb, DOES NOT WORK
        break
def best_of_block2(rates = hourly_rates, to_blocks = True, hr_capacity = 4): 
    #DOES NOT go across 24hr blocks
    "Python2 version when no longer supported; needs to take out numpy"
    if to_blocks:
        rates = block(rates, hr_capacity)#[sum(hourly_rates[i:i+4])/4 for i in range(len(hourly_rates) - 4)]#write better that increments average
    mn = 2**32 - 1
    mx = -2**32
    bi = 0
    was_increasing, was_decreasing = False, True#can buy first
    num_days = int(math.ceil(len(rates)/24))
#    possible_buys = {i:[rates[i*24]] for i in range(num_days)}#you can always buy at the start of the day
#    possible_sells = {i:[rates[min(i*24+23, len(rates)-1)]] for i in range(num_days)}#or sell at end of period
    possible_buys = {i:[] for i in range(num_days)}#you can always buy at the start of the day
    buys_ix = {i:[] for i in range(num_days)}
    possible_sells = {i:[] for i in range(num_days)}#or sell at end of period
    sells_ix = {i:[] for i in range(num_days)}
 
    #change checking i%24 ==0 in loop?
    
    #possible_profits = {i:[] for i in range(num_days)}
    profits = [0]*num_days#{i:0 for i in range(num_days + 1)}
    indxs = [[0,0]]*num_days
    for i in range(len(rates)-1):
        #adding possible sells twice.
        if rates[i] < rates[i+1]:#increasing
            if was_decreasing:
                mn = rates[i]#swapped from increasing to decreasing; must be local min
                bi = i#can't buy here in case price goes lower
            #if was same price would buy 'later' in day, if decreasing from 1->2 don't buy now
            elif len(possible_buys[i//24]) == 0:#first possible place to buy in new day
                    possible_buys[i//24].append(rates[i])
                    buys_ix[i//24].append(i)
                    mn = 2**32-1#reset min
            was_increasing = True
            was_decreasing = False
            
        elif rates[i] > rates[i+1]:#decreasing
            if was_increasing:
                mx = rates[i]
                if mn < 2^31:#need to be sure was charged at this point
                    possible_buys[i//24].append(mn)#this could be from the previous day.
                    buys_ix[i//24].append(bi)#have already bought it
                    possible_sells[i//24].append(mx)#this evaluates at the time of selling
                    sells_ix[i//24].append(i)#have already bought it
                    #possible_profits[i//24].extend((mn,mx))
                    mn = 2**32 - 1
                #make dict and compare key each time; must dis and re charge in same day
#            else:
#                if i%24 == 23:#last hour in day; could sell
#                    possible_sells[i//24].append(rates[i])
#                    sells_ix[i//24].append(i)
            was_increasing = False
            was_decreasing = True    
        #"resets" the day; calculating for new period
        
        if i%24 == 23:
            if rates[i] > rates[i-1]:#increased up to last hour in day
                was_increasing = False
                was_decreasing = True
                mn = 2**32-1
        #Notes: if prices level then will trade at the later instance

    #for last entity
    if was_increasing:#then would have bought, will sell on last day
        possible_buys[num_days - 1].append(mn)
        possible_sells[num_days - 1].append(rates[-1])  
        buys_ix[num_days - 1].append(bi)
        sells_ix[num_days - 1].append(len(rates) - 1)  

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
#    return possible_buys
#
#    return list(possible_buys[num_days-1]), list(possible_sells[num_days-1])

    #get from possible to best       
    prev_night_low = 2**63-1
#    prev_night_high = -2**63-1
    sell_twice = False
#    profits[0], _ = bs_2_series(possible_buys[0], possible_sells[0], i = 0, prices = rates)
    for i in range(num_days):#2.912781700026244; 8.423050199984573
#        if profits[i-1] < rates[i*24] - prev_night_low:#price you can buy at increased so much over 24hr mark from last night to today
#            profits[i-1] = 0#only charge yesterday to sell twice today
#            possible_buys[i].insert(0, prev_night_low)#got yesterday's low for today
#            sell_twice = True
##            profits[i] = bs_2_series(possible_buys[i], possible_sells[i])
            
        profits[i], minmax_ix = bs_2_series(possible_buys[i], possible_sells[i], i = i, prices = rates, sell_twice = sell_twice)
        print(i, indxs[i], buys_ix[i], sells_ix[i])
        indxs[i] = [buys_ix[i][minmax_ix[0]] + 24*i, sells_ix[i][minmax_ix[1]] + 24*i] 
        sell_twice = False
#        elif profits[i-1] < prev_night_high - possible_sells[i][0]:
#            profits[i-1] = prev_night_high - possible_sells[i][0] #waited, while charged, to sell this morning
#            possible_sells[i][0] = -1024         
#        prev_night_low = night_low
#        prev_night_high = night_high
    #if not way1: 
    #    for i in range(num_days):#29.437535100034438, 3.14; 9.014132300042547
    #        profits[i] = bs_in_period(possible_profits[i])
    return profits, indxs#, possible_sells#possible_profits.values()
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
#timeit.timeit("best_of_block2()", 
    #setup = "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; \
    #import random; import math; hourly_rates = np.random.randint(0, high = 10, size = 744); \
    #from __main__ import best_of_block2;from __main__ import best_of_block1;",
    # number = 3000)
    #timeit.timeit("best_of_block1()", 
    #setup = "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; \
    #import random; import math; hourly_rates = np.random.randint(0, high = 10, size = 744); \
    #from __main__ import best_of_block1;from __main__ import best_of_block1;",
    # number = 3000)
#print(np.array(best_of_block2()) - np.array(best_of_block1()))
#for i,j in zip(best_of_block1(), best_of_block2()):
#    print(i,j)
#possible_buys, possible_sells = best_of_block2()
#possible_profits = best_of_block2(way1 = False)
#best_of_block2(way1 = True) == best_of_block2(way1 = False) 
#np.where(best_of_block1() != best_of_block2())
best_of_block2()
#%%
    #delete below
def best_of_day_r2(buy,sell, cycles_per_day = 1, hr_capacity = 4, extra_charges = 0, into_blocks = False):
    "gets best buy/sell of ALL rates, recurs w/o those indexs to recalculate. Can alternate Buying, Selling"
    #O(24*hr_capacity); has to copy over each array for each recursion.
    rates = [k for i in zip(buy,sell) for k in i]
    if extra_charges == 0:
        best_low_hi = [2**32-1, -2**32]
        low_best_hi = [2**32-1, -2**32] 
        best_low_hi_ix, low_best_hi_ix = [0,0], [0,0]
        start = 0
        try:
            while rates[start] >= rates[start+1]:#only start looking at values once you've had a minimum.
                start += 1
        except:
            indx = [0, len(rates)-1]
            return rates[indx[1]] - rates[indx[0]], indx
        
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
    
    else:#THIS IS NOT RIGHT as it assumes can store an unlimited amount of energy: first thing w/ free charge is fill the maximium daily price orders, then recur?      
        best_low_hi = [0, 0]#no effect
        low_best_hi = [0, max(rates)] 
        best_low_hi_ix, low_best_hi_ix = [None,None], [None, rates.index(max(rates))]
    #how to calculate losing a charge?  
    
    options = [i[1] - i[0] for i in [low_best_hi, best_low_hi]]
    day_best = max(options)
    if options[0] == day_best:
        ix_used = low_best_hi_ix#
    else:
        ix_used = best_low_hi_ix
#    print(day_best, ix_used, rates[ix_used[1]] - rates[ix_used[0]])
    return day_best, ix_used
#%%
    
p = [i for i in k]
p[12:12+4] = [-10]*4

sz = 24
hr_capacity = 4

potential_ix = [[0]*hr_capacity]*(sz-hr_capacity+1)
potential_ix[0] = [ix for ix, val in sorted(zip(range(hr_capacity), p[:hr_capacity]), key = lambda tup: tup[1], reverse = True)]#indexs
print(potential_ix[0])
for i in range(hr_capacity,sz):
    potential_ix[i-hr_capacity+1],_ ,_ = insert_sort(p[:sz], potential_ix[i-hr_capacity], i, hr_capacity = hr_capacity, buying = True)
    print(potential_ix[i-hr_capacity+1], [p[i] for i in potential_ix[i-hr_capacity+1]])
print(potential_ix[-1], sorted(list(range(12,12+4)), reverse = True))
#ix = TestBattery.iter_thru_insert_sort(self, q, 24)
#self.assertEqual(list(range(12,12+hr_capacity)), sorted(ix[0]), msg = "buying")

#tztbat = TestBattery()
#print(tztbat.hourly_rates[:24])
#tztbat.test_insert_sort_duplicates()
#%%
#unittest.main()

def intermitant(rates, hr_capacity = 4, cycles_per_day = 1, into_blocks = False, use_add = True, for_testing = False):#what does use add do?
    "returns dicts of profit, buy locs, sell locs[locs are absolute] with buying at best hr_capacity buys,\
    then selling at best hr_cap sells. All sells occur after all buys. Assumes always completely, charge, discharge each day"
    assert(hr_capacity*2 <= 24)
    if into_blocks:#calculates assuming can only charge in continious hours
        return intermitant(rates=block(rates, hr_capacity), hr_capacity = 1, cycles_per_day = 1, into_blocks = False, for_testing = for_testing )
    
    
    if len(rates)%24 <= 2*hr_capacity and len(rates)%24 != 0:#need to be able to completely buy, sell on last day
        print("Droping the last day's hours to make times equal")
        rates = rates[:-len(rates)%24]
        
    num_days = int(math.ceil(len(rates)/24))
    buy_locs = {i:[] for i in range(num_days)}
    sell_locs = {i:[] for i in range(num_days)}
    profits = {i:[] for i in range(num_days)}
    
    buy_profit = [0]*(25-2*hr_capacity)#will be negative for buying; positve for selling
    sell_profit = [0]*(25-2*hr_capacity)
    buy_ix = [[0] for _ in range(hr_capacity)]*(25-2*hr_capacity)#list of lists; each sublist is of best places to buy up to that point(indexs for rates)
    sell_ix = [[0] for _ in range(hr_capacity)]*(25-2*hr_capacity)#sublist is hour withen TOTAL PERIOD

    #should treat first as special too

    for i in range(num_days):#2, num_days-1):
        #lengths are 24 - hr_capacity + 1 as start at 0 is w/ hr_capacity joined together, have 24-hr_cap other options "in front"
        #list buy_ix[i] is list of indicies for the smallest values in list thus seen, sorted with largest value in 0th position
        loc_in_buy_ix = 0
        buy_ix[loc_in_buy_ix] = [ix for _, ix in sorted(zip(rates[i*24:i*24+hr_capacity], range(i*24,i*24+hr_capacity)), reverse = True)]
        buy_profit[loc_in_buy_ix] = -1*sum(rates[i*24:i*24+hr_capacity])
        buy_alternates = [0]*(25-2*hr_capacity)#have to stop selling with room for 
        
        #swap firs value in list; smallest value in 0th position
        #don't reverse the range as need to pair with the locations of vals in rates   
        loc_in_sell_ix = 24-2*hr_capacity
        sell_ix[loc_in_sell_ix] = [ix for _, ix in sorted(zip(rates[(i+1)*24-hr_capacity:(i+1)*24], range((i+1)*24-hr_capacity,(i+1)*24)), reverse = False)]
        sell_profit[loc_in_sell_ix] = sum(rates[(i+1)*24-hr_capacity:(i+1)*24])
        sell_alternates = [0]*(25-2*hr_capacity)
        
        for j in range(24*i+23-hr_capacity, i*24+hr_capacity-1, -1):#calc indexes of selling; end of night to first hr_cap where just buying
            sell_ix[loc_in_sell_ix-1], profit_change, alt = insert_sort(rates, sell_ix[loc_in_sell_ix], j, hr_capacity, buying = False)
            sell_profit[loc_in_sell_ix-1] = sell_profit[loc_in_sell_ix] + profit_change#profit_change pos; sell for more
            sell_alternates[loc_in_sell_ix] = alt
            loc_in_sell_ix -= 1
        
        #take into account situation where only discharge a little on the last few hours of each day
        
        
        #adjust for situation when abs(buy[0]) > sell[0]
        day_prft = 0
        day_indxs = [[0], [0]]
        for j in range(i*24+hr_capacity, 24*i+24-hr_capacity):#calc indexes of buying from possible option to last before would just be selling hour of night
            buy_ix[loc_in_buy_ix+1], profit_change, alt = insert_sort(rates, buy_ix[loc_in_buy_ix], j, hr_capacity, buying = True)
            buy_profit[loc_in_buy_ix+1] = buy_profit[loc_in_buy_ix] - profit_change#profit_change negative; less outlay for buying power
            buy_alternates[loc_in_buy_ix] = alt
            
            reps = 0
            #if make a copy have to redo deletes each time; don't want this?
            #after delete highest buy, next value could have been even higher and be profitable.
            day_buy = np.copy(buy_ix[loc_in_buy_ix+1])
            day_sell = np.copy(sell_ix[loc_in_buy_ix+1])
            calc_prft = sell_profit[loc_in_buy_ix] + buy_profit[loc_in_buy_ix + 1]
            while rates[day_buy[0]] >= rates[day_sell[0]] and reps < hr_capacity-1:
#                print(day_buy, day_sell, day_buy[0], day_sell[0], reps)
                calc_prft += day_buy[0]
                calc_prft -= day_sell[0]
                day_buy = np.delete(day_buy, 0)
                day_sell = np.delete(day_sell, 0)
                reps += 1
#                print(buy_ix[loc_in_buy_ix+1], "\n", sell_ix[loc_in_buy_ix+1], "\n\n\n")
            
                
            if sell_profit[loc_in_buy_ix] + buy_profit[loc_in_buy_ix] > day_prft:#this buys as early, sells as late as possible
                day_prft = sell_profit[loc_in_buy_ix] + buy_profit[loc_in_buy_ix]
                day_indxs = [buy_ix[loc_in_buy_ix], sell_ix[loc_in_buy_ix] ]
            loc_in_buy_ix += 1
        
        profits[i] = day_prft
        [buy_locs[i], sell_locs[i]] = day_indxs 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import timeit
import math
import itertools

#Way Beth Wants it; actually use bottom
def intermitant2(rates, hr_capacity = 4, cycles_per_day = 1, into_blocks = False, use_add = True, for_testing = False):#what does use add do?
    "returns dicts of profit, buy locs, sell locs[locs are absolute] with buying at best hr_capacity buys,\
    then selling at best hr_cap sells. All sells occur after all buys. Assumes always completely, charge, discharge each day"
    assert(hr_capacity*2 <= 24)
    if not isinstance(rates, np.ndarray):
        rates = np.array(rates)
    if into_blocks:#calculates assuming can only charge in continious hours
        return intermitant2(rates=block(rates, hr_capacity), hr_capacity = 1, cycles_per_day = 1, into_blocks = False, for_testing = for_testing )
    
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
        
        for j in range(24*i+23-hr_capacity, i*24+hr_capacity-1, -1):#calc indexes of selling; end of night to first hr_cap where just buying
            sell_ix[loc_in_sell_ix-1], profit_change, alt = insert_sort(rates, sell_ix[loc_in_sell_ix], j, hr_capacity, buying = False)
            sell_profit[loc_in_sell_ix-1] = sell_profit[loc_in_sell_ix] + profit_change#profit_change pos; sell for more
            sell_alternates[loc_in_sell_ix] = alt
            loc_in_sell_ix -= 1
                

        
        for j in range(i*24, 24*i+23):#calc indexes of buying from possible option to last before would just be selling hour of night
            buy_ix[loc_in_buy_ix+1], profit_change, alt = insert_sort(rates, buy_ix[loc_in_buy_ix], j, hrs_used, buying = True)
            buy_profit[loc_in_buy_ix+1] = buy_profit[loc_in_buy_ix] + profit_change#profit_change negative; less outlay for buying power
            buy_alternates[loc_in_buy_ix] = alt
            loc_in_buy_ix += 1

        #get best buy/sell locations
        #SELL PROFITS are REVERSED! 
        #give them 'room' to buy; sell. needs hr_cap left in day to unwind; hr_cap in start to buy 
        
        
#        #buying price always decreases/constant, selling increases/ as go towards indx 0
        prft = [i+j for i,j in zip(buy_profit, sell_profit)]
        profits[i] = max(prft)
        indx = prft.index(profits[i])
        buy_locs[i] = buy_ix[indx]#[partial_charge]
        sell_locs[i] = sell_ix[indx]#[partial_charge]
        
    return profits, buy_locs, sell_locs

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
#%%############################################################################
    



#version going in pixel
def insert_sort(rates, current_indxs, potential_ix, hr_capacity, buying = True):
    "returns NEW list w/ inserted/deletes values (still sorted) and profit change of changing that\
    returns list of all indexs hold equivalent values (or none). \
    Want to swap 0th[0 is largest value for buying, lowest value for selling. \
    Can't expand more than 1 index per iteration"
    #all indexs based on location in rates
    ix_pntr = 0           
    #current_indxs should be sorted based on value they index, in reverse order rates[indx[0]] > rates[indx[-1]]
    if buying:#val shuold be smaller than largest element in list(which is at indx 0)
#        print(rates[current_indxs[ix_pntr]], rates[potential_ix], "testing")
        while rates[current_indxs[ix_pntr]] > rates[potential_ix]:
            ix_pntr += 1
            if ix_pntr == hr_capacity:#pntr interated thru list, smaller than all
                #profit change will be negative; subtracting large from smaller  
                return current_indxs[1:] + [potential_ix], rates[potential_ix] - rates[current_indxs[0]]       
        if ix_pntr != 0:
            return current_indxs[1:ix_pntr] + [potential_ix] + current_indxs[ix_pntr:], rates[potential_ix] - rates[current_indxs[0]]
        else:#no changes made; was larger
            return current_indxs, 0
        
    else:#selling, larger than smallest val in list
        #normal order. rates[indx[0]] < rates[indx[-1]]. Want to swap 0th
        while rates[current_indxs[ix_pntr]] < rates[potential_ix]:
            ix_pntr += 1
            if ix_pntr == hr_capacity:#pntr interated thru list; larger than all
                return current_indxs[1:] + [potential_ix], rates[potential_ix] - rates[current_indxs[0]]
            #Above is increase size: rates[current_indxs[0]] was set equal to rates[potential_indx], cancels leaving profit change. If decrease size; the value you replace is updated
        if ix_pntr != 0:#sell iter's thru backward so if later see an identical value; don't want to replace it
                return current_indxs[1:ix_pntr] + [potential_ix] + current_indxs[ix_pntr:], rates[potential_ix] - rates[current_indxs[0]]
        else:#no changes made
            return current_indxs, 0




def intermitant2(rates, hr_capacity = 4, cycles_per_day = 1, into_blocks = False, use_add = True, for_testing = False):#what does use add do?
    "returns dicts of profit, buy locs, sell locs[locs are absolute] with buying at best hr_capacity buys,\
    then selling at best hr_cap sells. All sells occur after all buys. Assumes always completely, charge, discharge each day"
    assert(hr_capacity*2 <= 24)
    if not isinstance(rates, np.ndarray):
        rates = np.array(rates)
    if into_blocks:#calculates assuming can only charge in continious hours
        return intermitant2(rates=block(rates, hr_capacity), hr_capacity = 1, cycles_per_day = 1, into_blocks = False, for_testing = for_testing )
    
    if len(rates)%24 <= 2*hr_capacity and len(rates)%24 != 0:#need to be able to completely buy, sell on last day
        print("Droping the last day's hours to make times equal")
        rates = rates[:-len(rates)%24]

    num_days = int(math.ceil(len(rates)/24))
    buy_locs = {i:[] for i in range(num_days)}
    sell_locs = {i:[] for i in range(num_days)}
    profits = {i:[] for i in range(num_days)}
    
    buy_profit = [0]*(25-2*hr_capacity)#will be negative for buying; positve for selling
    sell_profit = [0]*(25-2*hr_capacity)
    buy_ix = [[0]*hr_capacity]*(25-2*hr_capacity)#list of lists; each sublist is of best places to buy up to that point(indexs for rates)
    sell_ix = [[0]*hr_capacity]*(25-2*hr_capacity)#sublist is hour withen TOTAL PERIOD

    #should treat first as special too

    for i in range(num_days):#2, num_days-1):
        #lengths are 24 - hr_capacity + 1 as start at 0 is w/ hr_capacity joined together, have 24-hr_cap other options "in front"
        #list buy_ix[i] is list of indicies for the smallest values in list thus seen, sorted with largest value in 0th position
        loc_in_buy_ix = 0
        buy_ix[loc_in_buy_ix] = [ix for _, ix in sorted(zip(rates[i*24:i*24+hr_capacity], range(i*24,i*24+hr_capacity)), reverse = True)]
        buy_profit[loc_in_buy_ix] = -1*sum(rates[i*24:i*24+hr_capacity])
        buy_alternates = [0]*(25-2*hr_capacity)#have to stop selling with room for 
        
        #swap firs value in list; smallest value in 0th position
        #don't reverse the range as need to pair with the locations of vals in rates   
        loc_in_sell_ix = 24-2*hr_capacity
        sell_ix[loc_in_sell_ix] = [ix for _, ix in sorted(zip(rates[(i+1)*24-hr_capacity:(i+1)*24], range((i+1)*24-hr_capacity,(i+1)*24)), reverse = False)]
        sell_profit[loc_in_sell_ix] = sum(rates[(i+1)*24-hr_capacity:(i+1)*24])
        sell_alternates = [0]*(25-2*hr_capacity)
        
        for j in range(i*24+hr_capacity, 24*i+24-hr_capacity):#calc indexes of buying from possible option to last before would just be selling hour of night
#            print("Buying rates: ", rates[buy_ix[loc_in_buy_ix]], ". ix: ", buy_ix[loc_in_buy_ix], "cost", buy_profit[loc_in_buy_ix])
            buy_ix[loc_in_buy_ix+1], profit_change, alt = insert_sort(rates, buy_ix[loc_in_buy_ix], j, hr_capacity, buying = True)
            buy_profit[loc_in_buy_ix+1] = buy_profit[loc_in_buy_ix] - profit_change#profit_change negative; less outlay for buying power
            buy_alternates[loc_in_buy_ix] = alt
            loc_in_buy_ix += 1

        for j in range(24*i+23-hr_capacity, i*24+hr_capacity-1, -1):#calc indexes of selling; end of night to first hr_cap where just buying
#            print("Selling rates: ", rates[sell_ix[loc_in_sell_ix]], ". ix: ", sell_ix[loc_in_sell_ix], "sold", sell_profit[loc_in_sell_ix])
            sell_ix[loc_in_sell_ix-1], profit_change, alt = insert_sort(rates, sell_ix[loc_in_sell_ix], j, hr_capacity, buying = False)
            sell_profit[loc_in_sell_ix-1] = sell_profit[loc_in_sell_ix] + profit_change#profit_change pos; sell for more
            sell_alternates[loc_in_sell_ix] = alt
            loc_in_sell_ix -= 1

        #issue: buying locations are indexed from 0; so if last rate is at 6th loc then that's index 2

        #get best buy/sell locations
        #SELL PROFITS are REVERSED! 
        #give them 'room' to buy; sell. needs hr_cap left in day to unwind; hr_cap in start to buy
        
        #buying price always decreases/constant, selling increases/ as go towards indx 0
        prft = [i+j for i,j in zip(buy_profit, sell_profit)]
        profits[i] = max(prft)
        indx = prft.index(profits[i])
        buy_locs[i] = buy_ix[indx]
        sell_locs[i] = sell_ix[indx]
        
    return buy_locs, sell_locs, profits

            
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