import pandas as pd
import sys
from pathlib import Path
import numpy as np
import subprocess
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF
from util.formatting.convert_order_stream import convert_stream_to_format
from util.formatting.convert_order_book import process_orderbook, is_wide_book
import itertools

from multiprocessing import get_context

mkt_open=pd.to_datetime("20200603")+ pd.to_timedelta("09:30:00")
mkt_close= pd.to_datetime("20200603")+ pd.to_timedelta("10:30:00")
historical_date = int(mkt_open.date().strftime('%Y%m%d'))
startsimTime = mkt_open.time().strftime('%H:%M:%S')
endsimTime = mkt_close.time().strftime('%H:%M:%S')
num_noise = 50
num_momentum = 25
num_value = 10


# time this
import time
"""start_time = time.time()

for i in range(3): #self.sim_num):
    print(f"Simulation {i+1} of {3}")
    # +self.sim_time} "
    subprocess.run([f"python3 -u abides.py -c bap -t ABM -d {historical_date} {startsimTime} {endsimTime} -l test -n {num_noise} -m {num_momentum} -a {num_value}"], shell=True)

    processed_orderbook =  make_orderbook_for_analysis("log/test/EXCHANGE_AGENT.bz2", "log/test/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                    hide_liquidity_collapse=False) # estimates parameters
    cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                            (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
    #remove nan value in first row
    #cleaned_orderbook = cleaned_orderbook.drop(cleaned_orderbook.index[0])
    
    #get rid of columns that are not needed, only keep MID_PRICE
    ##### try with price instead of mid price
    cleaned_orderbook = cleaned_orderbook.drop(['ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'ask_price_1', 'ask_size_1','bid_price_1',
                                                'bid_size_1','SPREAD','ORDER_VOLUME_IMBALANCE','VWAP'], axis=1)

    if i == 0:
        sim_orderbook = cleaned_orderbook.copy()
    else:
        sim_orderbook = sim_orderbook.append(cleaned_orderbook)

end_time = time.time()


print("time taken:", end_time - start_time)"""

#do the same with multiprocessing

import multiprocessing as mp

if __name__ == '__main__':


    print(mp.cpu_count())

    def run_simulation(i):
        cleaned_orderbook = np.array([])
        while cleaned_orderbook.shape[0] == 0:
            try:
                # +self.sim_time} "
                subprocess.run([f"python3 -u abides.py -c bap -t ABM -d {historical_date} {startsimTime} {endsimTime} -l test_{i} -n {num_noise} -m {num_momentum} -a {num_value}"], shell=True)
            except UnboundLocalError:
                continue
            else:
                processed_orderbook =  make_orderbook_for_analysis(f"log/test_{i}/EXCHANGE_AGENT.bz2", f"log/test_{i}/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                                hide_liquidity_collapse=False) # estimates parameters
                cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                                        (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
                
                #get rid of columns that are not needed, only keep MID_PRICE
                cleaned_orderbook = cleaned_orderbook.drop(['ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'ask_price_1', 'ask_size_1','bid_price_1',
                                                            'bid_size_1','SPREAD','ORDER_VOLUME_IMBALANCE','VWAP'], axis=1)
                print("cleaned orderbook:", cleaned_orderbook.head(5))
                print("cleaned orderbook shape:", cleaned_orderbook.shape)
        
        return cleaned_orderbook

    start_time = time.time()

    pool_obj = mp.Pool(mp.cpu_count())
    results = pool_obj.map(run_simulation, range(3))

    end_time = time.time()

    print("time taken:", end_time - start_time)

    print("results:", results[0].shape)
    print("results full:", results)







"""num_levels = 1
hide_liquidity=False
ignore_cancellations=True

stream_df = pd.read_pickle("log/bap_test/EXCHANGE_AGENT.bz2")
orderbook_df = pd.read_pickle("log/bap_test/ORDERBOOK_ABM_FULL.bz2")
summary_df = pd.read_pickle("log/bap_test/summary_log.bz2")
fundamental_df = pd.read_pickle("log/bap_test/fundamental_ABM.bz2")
pov_ex_agent_df = pd.read_pickle("log/bap_test/POV_EXECUTION_AGENT.bz2")

print("stream_df:", stream_df)
print(stream_df["EventType"].unique())
print(stream_df[stream_df.EventType=="LIMIT_ORDER"])
print(stream_df[stream_df.EventType=="LIMIT_ORDER"].Event[0])
print(stream_df[stream_df.EventType=="ORDER_CANCELLED"])
print(stream_df[stream_df.EventType=="ORDER_CANCELLED"].Event[0])
print(stream_df[stream_df.EventType=="ORDER_EXECUTED"])
print(stream_df[stream_df.EventType=="ORDER_EXECUTED"].Event[0])
print(stream_df[stream_df.EventType=="ORDER_EXECUTED"].Event[1])


stream_processed = convert_stream_to_format(stream_df.reset_index(), fmt='plot-scripts')
stream_processed = stream_processed.set_index('TIMESTAMP')
print(stream_processed[stream_processed.TYPE=="ORDER_EXECUTED"])
print(stream_processed.TYPE.unique())"""



"""ob_processed = process_orderbook(orderbook_df, num_levels)

if not is_wide_book(orderbook_df):  # orderbook in skinny format
    ob_processed.index = orderbook_df.index.levels[0]
else:  # orderbook in wide format
    ob_processed.index = orderbook_df.index

columns = list(itertools.chain(
    *[[f'ask_price_{level}', f'ask_size_{level}', f'bid_price_{level}', f'bid_size_{level}'] for level in
        range(1, num_levels + 1)]))
merged = pd.merge(stream_processed, ob_processed, left_index=True, right_index=True, how='left')
merge_cols = ['ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE'] + columns
merged = merged[merge_cols]
merged['PRICE'] = merged['PRICE'] / 100

# clean
# merged = merged.dropna()
merged = merged.ffill()

# Ignore cancellations
if ignore_cancellations:
    merged = merged[merged.SIZE != 0]

merged['MID_PRICE'] = (merged['ask_price_1'] + merged['bid_price_1']) / (2 * 100)
merged['SPREAD'] = (merged['ask_price_1'] - merged['bid_price_1']) / 100
merged['ORDER_VOLUME_IMBALANCE'] = merged['ask_size_1'] / (merged['bid_size_1'] + merged['ask_size_1'])

hide_liquidity_collapse=True
if hide_liquidity_collapse:
    merged = mid_price_cutoff(merged)

# add VWAP
from realism.realism_utils import augment_with_VWAP
merged = augment_with_VWAP(merged)
print("Orderbook construction complete!")


processed_orderbook =  make_orderbook_for_analysis("log/bap_test/EXCHANGE_AGENT.bz2", "log/bap_test/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                               hide_liquidity_collapse=False)# estimates parameters
cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                                    (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
#remove headers
cleaned_orderbook = cleaned_orderbook.drop(cleaned_orderbook.index[0])

#change true to 1 and false to 0 in buy_sell_flag
cleaned_orderbook['BUY_SELL_FLAG'] = cleaned_orderbook['BUY_SELL_FLAG'].astype(int)

# change "LIMIT_ORDER" to 0, "ORDER_EXECUTED" to 1, "ORDER_CANCELLED" to 2
cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('LIMIT_ORDER', 0)
cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_EXECUTED', 1)
cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_CANCELLED', 2)


cleaned_orderbook = cleaned_orderbook.iloc[:,1:]
cleaned_orderbook = cleaned_orderbook.to_numpy().flatten().tolist()


cleaned_orderbook = np.array(cleaned_orderbook)
cleaned_orderbook = cleaned_orderbook.reshape(cleaned_orderbook.shape[0]//12, 12)

print("step1:", cleaned_orderbook[:2, :])
print("step1 shape:", cleaned_orderbook.shape)


cleaned_orderbook = np.delete(cleaned_orderbook, [0, 4], axis=1)
#cleaned_orderbook.to_csv("bap/ORDERBOOK_ABM_CLEANED.csv")
#cleaned_orderbook = pd.DataFrame(cleaned_orderbook).drop(columns=['ORDER_ID']) #, 'TYPE'])



#print("cleand orderbook:", cleaned_orderbook.to_numpy())
print("cleand orderbook shape:", cleaned_orderbook.shape)
print("step2:", cleaned_orderbook[:2, :])

cleaned_orderbook = pd.DataFrame(cleaned_orderbook)

prop_time_skip = 0.3 # proportion of time to skip at the beginning of the orderbook
data = cleaned_orderbook
# get the first and last time of the orderbook
first_time = data.index[0]
last_time = data.index[-1]
start_time = first_time + prop_time_skip*(last_time - first_time)
#retain only the part of the orderbook after the start_time
orderbook = data.loc[data.index >= start_time]

orderbook = np.mean(orderbook, axis=0)
print(orderbook.shape) #add fundamental
print(orderbook)
pd.DataFrame(orderbook).to_csv("bap/SUM_STATS_ABM.csv")
print("done")

# read in sumstats csv
sumstats = pd.read_csv("bap/SUM_STATS_ABM.csv")
print(sumstats.to_numpy().T[1])"""


### convert to numpy array so that distance recognises as list
# make sure headers are correctly removed in sumstats (should be correct since no errors)
## understand why it doesn't make sums stats csv