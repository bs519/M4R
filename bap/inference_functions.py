import os
import abcpy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import subprocess
from math import trunc
import scipy as sp

from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.statistics import Statistics

### Summary Statistics
import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF
from util.formatting.convert_order_stream import convert_stream_to_format
#from market_simulations import rmsc03_4

os.makedirs("Results", exist_ok = True)


from abcpy.distances import Distance




class KS_statistic(Distance):
    """
    This class implements the Kolmogorov-Smirnov statistic between two vectors.

    The maximum value of the distance is np.inf.

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.

    """

    def __init__(self, statistics_calc):
        super(KS_statistic, self).__init__(statistics_calc)
    
    def distance(self, d1, d2):

        s1, s2 = self._calculate_summary_stat(d1, d2)

        dist_matrix = np.zeros(shape=(s1.shape[0], s2.shape[0]))

        for ind1 in range(0, s1.shape[0]):
            s1 = s1[ind1]
            for ind2 in range(0, s2.shape[0]):
                s2 = s2[ind2]
                for s in [s1,s2]:
                    buy_sell_flag = s[:, -2]  # Extract the buy/sell flag column
                    types = s[:, -1]  # Extract the type column

                    # Create separate arrays for each combination of buy/sell flag and type
                    if s is s1:
                        arrays1 = {}
                    if s is s2:
                        arrays2 = {}

                    for flag in [0, 1]:
                        for t in [0, 1, 2]:
                            mask = np.logical_and(buy_sell_flag == flag, types == t)
                            if s is s1:
                                arrays1[(flag, t)] = s[mask]
                            if s is s2:
                                arrays2[(flag, t)] = s[mask]


                dist = np.zeros(shape=(2, 3, s1.shape[0], s2.shape[0]))
                for flag in [0, 1]:
                    for t in [0, 1, 2]:
                        s1_small = arrays1[(flag, t)]
                        s2_small = arrays2[(flag, t)]
                        if s1_small.shape[0] != 0 and s2_small.shape[0] != 0:
                            # compute distance between the statistics
                            dist[flag, t, ind1, ind2] = (sp.stats.ks_2samp(s1_small[:, 1], s2_small[:, 1]).statistic + sp.stats.ks_2samp(s1_small[:, 2], s2_small[:, 2]).statistic)/2

                dist_matrix[ind1, ind2] = np.mean(dist)

        return dist_matrix.mean()
    
    def dist_max(self):
        return np.inf



class Model(ProbabilisticModel, Continuous):
    """
    A model for the inference of the parameters of ABIDES
    """

    def __init__(self, parameters, n=None, symbol = "ABM", starting_cash = 10000000, r_bar = 1e5, sigma_n = 1e5/10, kappa = 1.67e-15, lambda_a = 7e-11, name='Model'):
        """
        Parameters
        ----------
        parameters: list of abcpy.discretevariables.DiscreteVariable objects
            Defines the variables the model is taking as input.
        name: string, optional
            Name of the model.
        """
        self.parameters = parameters
        self.n = n
        #self.output_variables = [Continuous(np.array([0]), name='output', limits=np.array([[0, 1]]))]
        self.symbol = symbol
        self.starting_cash = starting_cash
        self.r_bar = r_bar
        self.sigma_n = sigma_n
        self.kappa = kappa
        self.lambda_a = lambda_a

        # We expect input of type parameters = [num_noise, num_momentum_agents, num_value]
        if not isinstance(parameters, list):
            raise TypeError("Input of model is of type list")

        if len(parameters) != 3:
            raise RuntimeError("Input list must be of length 3, containing [num_noise, num_momentum_agents, num_value].")

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 3:
            raise ValueError("Number of parameters of model must be 3.")

        # Check whether input is from correct domain
        num_noise = input_values[0]
        num_momentum_agents = input_values[1]
        num_value = input_values[2]

        if num_noise < 0 or num_momentum_agents < 0 or num_value < 0:# or isinstance(num_noise, int) == False or isinstance(num_momentum_agents, int) == False or isinstance(num_value, int) == False:
            return False

        return True

    def _check_output(self, values):
        if not isinstance(values, np.array):
            raise ValueError('This returns an array')
        
        return True

    def get_output_dimension(self):
        return 2

    def forward_simulate(self, parameters, k, rng = np.random.RandomState()):
        # Extract the input parameters
        num_noise = parameters[0]
        num_momentum_agents = parameters[1]
        num_value = parameters[2]
        #time simulated
        #n_timestep = parameters[3]

        # Do the actual forward simulation
        vector_of_k_samples = self.Market_sim(num_noise, num_momentum_agents, num_value, k)
        # Format the output to obey API
        result = [np.array([x]) for x in vector_of_k_samples]
        return result

    """def forward_simulate_true_model(self, k, rng = np.random.RandomState()):
        # Do the actual forward simulation
        vector_of_k_samples = self.Market_sim_true(k, rng = rng) # is x full orderbook and y is summary statistics???
        # Format the output to obey API
        result = [np.array([x]) for x in vector_of_k_samples]
        return result"""
    
    def Market_sim(self, num_noise, num_momentum_agents, num_value, k):
        """
        k market simulations for n_timstep time using abides package with a configuration /
        of num_noise noise agents, num_momentum momentum agents, num_value value agents.
        Parameters
        ----------
        num_noise: number of noise agents
        num_momentum: number of momentum agents
        num_value:number of value agents

        Return:
        List of length k each containing the order book of the simulation
        """

        result = []
        n = self.n
        end_sim = n + timedelta(minutes=5)
        for i in range(k):
            cleaned_orderbook = np.array([])
            j = 0
            while True:
                try:
                    #### make python file execute powershell command with parameters as config
                    subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 --start-time '9:30:00' --end-time {end_sim.strftime('%H:%M:%S')} -l inference -n {num_noise} -m {num_momentum_agents} -a {num_value} -z {self.starting_cash} -r {self.r_bar} -g {self.sigma_n} -k {self.kappa} -b {self.lambda_a}"], shell=True)
                except Exception as e:
                    print("An error occurred:", str(e))
                    j += 1
                    print(f"We try again for the {j}th time")
                    continue
                else:
                    stream_df = pd.read_pickle("log/inference/EXCHANGE_AGENT.bz2")
                    stream_processed = convert_stream_to_format(stream_df.reset_index(), fmt='plot-scripts')
                    stream_processed = stream_processed.set_index('TIMESTAMP')
                    cleaned_orderbook = stream_processed
                    #obtain latest time of the orderbook
                    last_time = cleaned_orderbook.index[-1]
                    start_time = cleaned_orderbook.index[0]

                    if cleaned_orderbook.shape[0] != 0 and last_time > start_time + pd.to_timedelta("3min"):
                        #change true to 1 and false to 0 in buy_sell_flag
                        cleaned_orderbook['BUY_SELL_FLAG'] = cleaned_orderbook['BUY_SELL_FLAG'].astype(int)

                        # change "LIMIT_ORDER" to 0, "ORDER_EXECUTED" to 1, "ORDER_CANCELLED" to 2
                        cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('LIMIT_ORDER', 0)
                        cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_EXECUTED', 1)
                        cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_CANCELLED', 2)
                        
                        #set index as TIMESTAMP column
                        cleaned_orderbook = cleaned_orderbook.rename_axis('TIMESTAMP').reset_index()
                        
                        # keep rows with index smaller than Datetime n
                        cleaned_orderbook = cleaned_orderbook[cleaned_orderbook['TIMESTAMP'] < n]

                        #make headers the first row
                        cleaned_orderbook.loc[0] = cleaned_orderbook.columns

                        result.append(cleaned_orderbook.to_numpy())
                        break
                    else:
                        j += 1
                        print(f"We try again for the {j}th time")
                        continue

        return result
    

"""processed_orderbook =  make_orderbook_for_analysis("log/bap_timestep/EXCHANGE_AGENT.bz2", "log/bap_timestep/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                               hide_liquidity_collapse=False)# estimates parameters
            print(processed_orderbook.head(5))
            print(type(processed_orderbook))
            cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                                    (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
            
            #remove nan value in first row
            cleaned_orderbook = cleaned_orderbook.drop(cleaned_orderbook.index[0])"""


class SummaryStatistics(Statistics):
    """
    This class implements the statistics function from the Statistics protocol. This
    extracts the statistics following Hakkarainen et. al. [1] from the multivariate timesereis
    generated by solving Lorenz 95 odes.

    [1] J. Hakkarainen, A. Ilin, A. Solonen, M. Laine, H. Haario, J. Tamminen, E. Oja, and
    H. Järvinen. On closure parameter estimation in chaotic systems. Nonlinear Processes
    in Geophysics, 19(1):127–143, Feb. 2012.
    """

    def __init__(self, degree = 2, cross = True):
        self.degree = degree
        self.cross = cross

    def statistics(self, data, prop_time_skip = 0.3):
        """ Creates summary statistics for the order book. We skip the first prop_time_skip of the orderbook. """
        
        if isinstance(data, list):
            """if np.array(data).shape == (len(data),):
                if len(data) == 1:
                    data = np.array(data).reshape(1, 1)
                data = np.array(data).reshape(len(data), 1)
            else:
                data = np.concatenate(data).reshape(len(data), -1)"""
        else:
            raise TypeError("Input data should be of type pd.list, but found type {}".format(type(list)))
        
        num_element = len(data)
        result = [None]*num_element

        # Compute statistics
        for ind_element in range(0, num_element):
            data_ind_element = pd.DataFrame(data[ind_element][0])
            #make the first row headers
            data_ind_element.columns = data_ind_element.iloc[0]
            #drop the first row
            data_ind_element = data_ind_element[1:]
            #set Timestamp as index
            data_ind_element = data_ind_element.set_index('TIMESTAMP')
            # remove the order id column
            data_ind_element = data_ind_element.drop(columns=['ORDER_ID'])

            # compute the mean of each statistic every minute for each type
            limit_order = data_ind_element[data_ind_element["TYPE"] == 0]
            order_executed = data_ind_element[data_ind_element["TYPE"] == 1]
            order_cancelled = data_ind_element[data_ind_element["TYPE"] == 2]

            # compute the mean of each statistic every minute for each type
            first_time = data_ind_element.index[0]
            last_time = data_ind_element.index[-1]
            #compute the number of minutes in the orderbook, minus the last incomplete minute
            n_minutes = trunc((last_time - first_time).total_seconds()/60)

            #compute the mean of each statistic every minute for each type
            mean_limit = np.zeros(shape = (n_minutes*6, 5))

            for i in range(0, n_minutes):
                start_time = first_time + i*pd.Timedelta(minutes=1)
                end_time = start_time + pd.Timedelta(minutes=1)
                #retain only the part of the orderbook after the start_time
                limit_order_minute = limit_order.loc[(limit_order.index >= start_time) & (limit_order.index < end_time)]
                order_executed_minute = order_executed.loc[(order_executed.index >= start_time) & (order_executed.index < end_time)]
                order_cancelled_minute = order_cancelled.loc[(order_cancelled.index >= start_time) & (order_cancelled.index < end_time)]
                
                #set index as MINUTE column
                limit_order_minute = limit_order_minute.reset_index().rename_axis('MINUTE').reset_index()
                order_executed_minute = order_executed_minute.reset_index().rename_axis('MINUTE').reset_index()
                order_cancelled_minute = order_cancelled_minute.reset_index().rename_axis('MINUTE').reset_index()

                # set the entries of the MINUTE column to be i
                limit_order_minute['MINUTE'] = i
                order_executed_minute['MINUTE'] = i
                order_cancelled_minute['MINUTE'] = i
                
                #drop TIMESTAMP column
                limit_order_minute = limit_order_minute.drop(columns=['TIMESTAMP'])
                order_executed_minute = order_executed_minute.drop(columns=['TIMESTAMP'])
                order_cancelled_minute = order_cancelled_minute.drop(columns=['TIMESTAMP'])
                
                # separate buy and sell orders
                limit_order_minute_buy = limit_order_minute[limit_order_minute['BUY_SELL_FLAG'] == 1]
                limit_order_minute_sell = limit_order_minute[limit_order_minute['BUY_SELL_FLAG'] == 0]
                order_executed_minute_buy = order_executed_minute[order_executed_minute['BUY_SELL_FLAG'] == 1]
                order_executed_minute_sell = order_executed_minute[order_executed_minute['BUY_SELL_FLAG'] == 0]
                order_cancelled_minute_buy = order_cancelled_minute[order_cancelled_minute['BUY_SELL_FLAG'] == 1]
                order_cancelled_minute_sell = order_cancelled_minute[order_cancelled_minute['BUY_SELL_FLAG'] == 0]

                #compute the mean of each statistic for each type and buy/sell
                mean_limit[6*i, :] = np.mean(limit_order_minute_buy, axis=0)
                mean_limit[6*i + 1, :] = np.mean(limit_order_minute_sell, axis=0)
                mean_limit[6*i + 2, :] = np.mean(order_executed_minute_buy, axis=0)
                mean_limit[6*i + 3, :] = np.mean(order_executed_minute_sell, axis=0)
                mean_limit[6*i + 4, :] = np.mean(order_cancelled_minute_buy, axis=0)
                mean_limit[6*i + 5, :] = np.mean(order_cancelled_minute_sell, axis=0)
            
            result[ind_element] = mean_limit

            return np.array(result)
                
                
            

            
                
            
            
            """print(data[ind_element].shape[1])
            data_ind_element = data[ind_element].reshape(data[ind_element].shape[1]//12, 12)
            print(data_ind_element)
            data_ind_element = pd.DataFrame(data_ind_element)
            # get the first and last time of the orderbook
            first_time = data_ind_element.index[0]
            last_time = data_ind_element.index[-1]
            start_time = first_time + prop_time_skip*(last_time - first_time)
            #retain only the part of the orderbook after the start_time
            orderbook = data_ind_element.loc[data_ind_element.index >= start_time]
            # remove the second and sixth columns corresponding to the order id and the type of order
            orderbook = orderbook.to_numpy()
            orderbook = np.delete(orderbook, [0, 4], axis=1)"""
            #orderbook = orderbook.drop(columns=['ORDER_ID', 'TYPE'])
            #drop headers
            #orderbook = orderbook.drop(orderbook.index[0])
            
            #return the mean of the orderbook
            #result[ind_element, :] = np.mean(orderbook, axis=0).tolist() #add fundamental

            """# Expand the data with polynomial expansion
        result = self._polynomial_expansion(result)"""
        