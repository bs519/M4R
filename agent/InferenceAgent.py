import pandas as pd
import numpy as np
import subprocess
import abcpy
from multiprocessing import get_context
import multiprocessing as mp
from datetime import datetime
from abcpy.output import Journal
import os
import time

import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

from agent.TradingAgent import TradingAgent
from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF
from bap.inference_functions import Model, SummaryStatistics

os.makedirs("Results", exist_ok = True)


class InferenceAgent(TradingAgent):
    """
    Inference agent that uses the SMCABC algorithm to infer the parameters of the ABIDES simulator.

    Attributes
    ----------
    id : str
        Agent identifier
    name : str
        Agent name
    type : str
        Agent type (e.g. 'MomentumAgent')
    symbol : str
        Symbol to trade
    starting_cash : float
        Starting cash balance
    min_size : int
        Minimum order size
    max_size : int
        Maximum order size
    size : int
        Optional fixed order size
    wake_up_freq : str
        Wake up frequency (e.g. '10s' for 10 seconds)
    subscribe : bool
        Flag to determine whether to subscribe to data or use polling mechanism
    L : int
        Length of order book history to use (number of transactions)
    log_orders : bool
        Flag to determine whether to log order book data
    random_state : numpy.random.RandomState
        Random number generator
    init_wakeup_time : str
        Time to start the inference agent
    mkt_open : datetime
        Market open time
    mkt_close : datetime
        Market close time
    k : int
        Number of simulations to run
    m : int
        Number of standard deviations to use for the Bollinger bands
    num_agents : int
        Number of agents to use for the inference
    inf_log : str
        Name of the folder to save the inference results
    r_bar : float
        Parameter of the simulator
    sigma_n : float
        Parameter of the simulator
    kappa : float
        Parameter of the simulator
    lambda_a : float
        Parameter of the simulator
    sim_check : bool
        Flag to determine whether the simulations have been run
    sim_OB : bool
        Flag to determine whether the simulations have been run
    sim_num : int
        Number of simulations to run
    sim_time : str
        Length of time to run the simulations
    historical_date : int
        Date of the historical data to use
    startsimTime : str
        Time to start the simulations
    endsimTime : str
        Time to end the simulations
    mid_list : list
        List of mid prices
    subscription_requested : bool
        Flag to determine whether the subscription has been requested
    state : str
        State of the agent
    sim_OB : pandas.DataFrame
        Simulated order book
    sim_check : bool
        Flag to determine whether the simulations have been run

    Methods
    -------
    kernelStarting(startTime)
        Method called when the kernel is starting
    wakeup(currentTime)
        Method called when the agent wakes up
    receiveMessage(currentTime, msg)    
        Method called when the agent receives a message
    placeOrders(bid, ask, currentTime)
        Simulate price and place a limit order at the best bid or ask depending on current price trend
    getWakeFrequency()
        Get the wake up frequency
    format_history(history, num_levels=1)
        Convert orderbook history to the format required by the inference method
    infer(history, n=None, n2=None, log_orders = None)
        Perform initial inference
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 min_size, max_size, size = None, wake_up_freq='60s',
                 subscribe=False, L=1000000, log_orders=False, random_state=None, init_wakeup_time="01:30:00",
                 mkt_open=pd.to_datetime("20200603")+ pd.to_timedelta("09:30:00"),
                 mkt_close= pd.to_datetime("20200603")+ pd.to_timedelta("11:30:00"), k=5, m=2, num_agents=1, inf_log = 'inference_agent'):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        if size:
            self.size = size
        else:
            self.size = self.random_state.randint(self.min_size, self.max_size)
        self.wake_up_freq = wake_up_freq
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.mid_list, self.avg_20_list, self.avg_50_list = [], [], []
        self.L = L  # length of order book history to use (number of transactions)
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        self.init_wakeup_time = pd.to_timedelta(init_wakeup_time)
        self.sim_time =  pd.to_timedelta(wake_up_freq) #pd.to_timedelta(sim_time)?
        self.historical_date = int(mkt_open.date().strftime('%Y%m%d'))
        self.startsimTime = str(mkt_open.time().strftime('%H:%M:%S') + pd.to_timedelta(init_wakeup_time))[slice(7,15)]
        self.endsimTime = mkt_close.time().strftime('%H:%M:%S')  
        self.k = k
        self.sim_num = k
        self.m = m
        self.num_agents = num_agents
        self.inf_log = inf_log
        self.r_bar = 1e5
        self.sigma_n = 1e5/10
        self.kappa = 1.67e-15
        self.lambda_a = 7e-11
        self.sim_check = False
        self.sim_OB = False

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        self.getOrderStream(self.symbol, length=self.L)
        if self.subscribe and not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=1, freq=10e9)
            self.subscription_requested = True
            self.state = 'AWAITING_MARKET_DATA'
        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        """ Momentum agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        
        if currentTime >= self.mkt_open + self.init_wakeup_time and not self.sim_check:
            print("initiating set up task")
            # print self.init_wakeup_time formatted to remove the days
            filesave = str(self.init_wakeup_time)[slice(7,15)]
            try:
                #journal = Journal.fromFile(f"Results/inference_agent_{self.init_wakeup_time}_{self.k}_{self.m}.jrnl")
                # Instead of reading the large journal file, we read the parameter file
                num_noise, num_momentum, num_value = np.genfromtxt(f"Results/{self.inf_log}/params_{filesave}_k{self.k}_m{self.m}_numb{self.num_agents}.txt")
                print("Parameters loaded")
            except FileNotFoundError:
                print("Run with inference SMCABC")
                history = self.stream_history[self.symbol]
                # convert orderbook history ot the format required by the inference method
                history = self.format_history(history)
                history = [np.array([history.to_numpy()])]
                n = history[0][0][-1, 0]
                n2 = history[0][0][1, 0]
                journal = InferenceAgent.infer(history, n, n2, self.log_orders)

                # save the final journal file, depending on the experiment
                #journal.save(f"Results/inference_agent_{self.init_wakeup_time}_{self.k}_{self.m}.jrnl")
                # Since the file is too large, instead we save only the estimates
                posterior_samples = np.array(journal.get_accepted_parameters()).squeeze()
                parameters = np.mean(posterior_samples, axis=0)
                num_noise, num_momentum, num_value = parameters
                # create a folder for the results
                os.makedirs(f"Results/{self.inf_log}", exist_ok=True)
                # save these parameters to a txt file
                np.savetxt(f"Results/{self.inf_log}/params_{filesave}_k{self.k}_m{self.m}_numb{self.num_agents}.txt", (num_noise, num_momentum, num_value))
                journal = 0

            def run_simulation(i, rng = np.random.RandomState()):
                cleaned_orderbook = np.array([])
                j =0
                while j < 5:
                    try:
                        subprocess.check_output([f"python3 -u abides.py -c bap -t ABM -d {self.historical_date} --start-time {self.startsimTime} --end-time {self.endsimTime} -l inference_agent_sim_{self.log_orders} -n {num_noise} -m {num_momentum} -a {num_value} -z {self.starting_cash} -r {self.r_bar} -g {self.sigma_n} -k {self.kappa} -b {self.lambda_a} -s {(i+1)*rng.randint(self.sim_num)+16+j}"], shell=True)
                    except subprocess.CalledProcessError as e:
                        print("error:", e.output)
                        j += 1
                        print(f"We try again for the {j}th time")
                        continue
                    else:
                        processed_orderbook =  make_orderbook_for_analysis(f"log/inference_agent_sim_{self.log_orders}/EXCHANGE_AGENT.bz2", f"log/inference_agent_sim_{self.log_orders}/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                                            hide_liquidity_collapse=False) # estimates parameters
                        cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                                                (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
                        
                        #get rid of columns that are not needed, only keep MID_PRICE
                        cleaned_orderbook = cleaned_orderbook.drop(['ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'ask_price_1', 'ask_size_1','bid_price_1',
                                                                    'bid_size_1','SPREAD','ORDER_VOLUME_IMBALANCE','VWAP'], axis=1)
                        
                        if cleaned_orderbook.shape[0] != 0:
                            break
                        else:
                            continue
                
                return cleaned_orderbook

            print("simulating")
            for i in range(self.sim_num):
                print(f"Simulation {i+1} of {self.sim_num}")
                cleaned_orderbook = run_simulation(i)

                if i == 0:
                    sim_orderbook = cleaned_orderbook.copy()
                else:
                    sim_orderbook = sim_orderbook.append(cleaned_orderbook)
            
            sim_orderbook.sort_index(inplace=True)
            #sim_orderbook.to_csv("Results/sim_orderbook.csv") #store the simulated orderbook depending on RAM availability

            self.sim_check = True
            self.sim_OB = sim_orderbook.copy()

            
            
                
        
        if not self.subscribe and self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            
            if currentTime >= self.mkt_open + self.init_wakeup_time:
                self.placeOrders(bid, ask, currentTime)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'
        elif self.subscribe and self.state == 'AWAITING_MARKET_DATA' and msg.body['msg'] == 'MARKET_DATA':
            bids, asks = self.known_bids[self.symbol], self.known_asks[self.symbol]
            if bids and asks and currentTime >= self.mkt_open + self.init_wakeup_time: self.placeOrders(bids[0][0], asks[0][0], currentTime)
            self.state = 'AWAITING_MARKET_DATA'

    def placeOrders(self, bid, ask, currentTime):
        """ Simulate price and place a limit order at the best bid or ask depending on current price trend"""
        
        """#possibility to subsitute wake up time with order book history length
        if len(self.stream_history[self.symbol]) < self.L:
            # Not enough history for inference.
            log_print("Insufficient history for inference: length {}, L {}", len(self.stream_history[self.symbol]), self.L)"""
        
        if bid and ask:
            m= self.m
            
            # An improvement to save some memory would be to compute this mid price mean and standard deviation before in the setup
            # part for every wake up time and store this much shorter time series instead of the whole simulations time series.
            #get the simulated orderbook
            sim_orderbook = self.sim_OB.copy()

            #obtain the average price around the current time
            sim_orderbook = sim_orderbook[(sim_orderbook.index > currentTime) & (sim_orderbook.index < currentTime+self.sim_time)]
            price = sim_orderbook['MID_PRICE'].mean() #mid price or price?
            # get standard deviation of price
            std = sim_orderbook['MID_PRICE'].std()

            #compare Upper and Lower Bollingers bands with bid and ask respectively and place order
            if price-m * std > ask:
                self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=True, limit_price=ask) # buy order at ask
            elif price + m * std < bid:
                self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=False, limit_price=bid) # sell order at bid


    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    @staticmethod
    def format_history(history, num_levels=1):
        """
        Function to convert orderbook history to the format required by the inference method.
        """
    
        # change "LIMIT_ORDER" to 0, "ORDER_EXECUTED" to 1, "ORDER_CANCELLED" to 2
        ob_processed = pd.DataFrame([(order['entry_time'], item_id, order['limit_price'], order['quantity'], order['is_buy_order'], 0)
                   for orders in history
                   for item_id, order in orders.items()],
                  columns=['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE'])
        transactions = pd.DataFrame([(i[0], item_id, order['limit_price'], i[1], j, 1)
                     for orders in history
                     for item_id, order in orders.items()
                     for j in [0,1]
                        for i in order['transactions'][1:]
                            if order['transactions']],
                     columns=['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE'])
        cancellations = pd.DataFrame([(i[0], item_id, order['limit_price'], i[1], order['is_buy_order'], 2)
                        for orders in history
                            for item_id, order in orders.items()
                                for i in order['cancellations']
                                    if order['cancellations']],
                        columns=['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE'])
        ob_processed = ob_processed.append(transactions)
        ob_processed = ob_processed.append(cancellations)
        #order by timestamp
        ob_processed = ob_processed.sort_values(by=['TIMESTAMP'])
      
        # change true to 1 and false to 0 in buy_sell_flag
        ob_processed['BUY_SELL_FLAG'] = ob_processed['BUY_SELL_FLAG'].astype(int)

        #make headers the first row
        ob_processed.iloc[0] = ob_processed.columns
        ob_processed = ob_processed
        return ob_processed


    def infer(history, n=None, n2=None, log_orders = None):
        """
        Function to perform initial inference.
        """
        from abcpy.backends import BackendDummy as Backend
        # Define backend
        backend = Backend()

        from abcpy.output import Journal
        from abcpy.continuousmodels import Uniform

        noise = Uniform([[0], [10000]], name = "noise")
        momentum = Uniform([[0], [100]], name = "momentum")
        value = Uniform([[0], [200]], name = "value")
        model = Model([noise, momentum, value], n, n2, log_orders, name = "model")

        ## define the summary statistic
        statistics_calculator = SummaryStatistics(degree = 1, cross = False)


        # Define distance
        from bap.inference_functions import KS_statistic
        distance_calculator = KS_statistic(statistics_calculator)
        # Define perturbation kernel
        from abcpy.perturbationkernel import DefaultKernel
        kernel = DefaultKernel([noise, momentum, value])
        
        from abcpy.inferences import SMCABC
        
        sampler = SMCABC([model], [distance_calculator], backend, kernel)
        # Define sampling parameters
        #full output = 0 for no intermediary values
        steps, n_samples, n_samples_per_param, full_output = 1, 2, 1, 0
        # Sample
        journal = sampler.sample([history], steps, n_samples,
                                n_samples_per_param, full_output = full_output)
        
        return journal
