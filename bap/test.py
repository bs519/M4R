import pandas as pd
import sys
from pathlib import Path
import numpy as np
import subprocess
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF


"""subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 '10:00:00' '11:30:00' "
                f"-l bap_test -n 10 "
                f"-z 100000"], shell=True)"""


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
print(sumstats.to_numpy().T[1])


### convert to numpy array so that distance recognises as list
# make sure headers are correctly removed in sumstats (should be correct since no errors)
## understand why it doesn't make sums stats csv