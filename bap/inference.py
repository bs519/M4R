import os
import abcpy
import numpy as np
import pandas as pd
import subprocess

# from abcpy.backends import BackendMPI as Backend
# the above is in case you want to use MPI, with `mpirun -n <number tasks> python code.py`
from abcpy.backends import BackendDummy as Backend
from abcpy.output import Journal
from abcpy.perturbationkernel import DefaultKernel
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector, Discrete
from abcpy.statistics import Identity, Statistics
from abcpy.statisticslearning import SemiautomaticNN

### Summary Statistics
import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF
#from market_simulations import rmsc03_4

os.makedirs("Results", exist_ok = True)


# Test: we perform inference on the number of noise, momentum and value agents

#from plotting.sumstats import make_sumstats as make_sumstats

class Model(ProbabilisticModel, Continuous):
    """
    A model for the inference of the parameters of ABIDES
    """

    def __init__(self, parameters, symbol = "ABM", starting_cash = 10000000, r_bar = 1e5, sigma_n = 1e5/10, kappa = 1.67e-15, lambda_a = 7e-11, name='Model'):
        """
        Parameters
        ----------
        parameters: list of abcpy.discretevariables.DiscreteVariable objects
            Defines the variables the model is taking as input.
        name: string, optional
            Name of the model.
        """
        self.parameters = parameters

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
    
        # if theta1 <= 0 or theta2 <= 0:
        # why? this does not make much sense, the parameters of the deterministic part could be smaller than 0
        #    return False

        if num_noise < 0 or num_momentum_agents < 0 or num_value < 0:# or isinstance(num_noise, int) == False or isinstance(num_momentum_agents, int) == False or isinstance(num_value, int) == False:
            return False

        return True

    def _check_output(self, values):
        if not isinstance(values, np.array):
            raise ValueError('This returns an array')
        
        """if values.shape[0] != 5:
            raise RuntimeError('The size of the output has to be 5.')"""
        
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
        num_value:number of valuem agents

        Return:
        List of length k each containing the order book of the simulation
        """
        #### make python file execute powershell command with parameters as config

        result = []

        for i in range(k):
            subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 '10:00:00' '11:30:00' "
            f"-l bap_timestep -n {num_noise} -m {num_momentum_agents} -a {num_value} "
            f"-z {self.starting_cash} -r {self.r_bar} -g {self.sigma_n} -k {self.kappa} -b {self.lambda_a}"], shell=True) #check it correctly runs the 3 lines
            

            processed_orderbook =  make_orderbook_for_analysis("log/bap_timestep/EXCHANGE_AGENT.bz2", "log/bap_timestep/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                               hide_liquidity_collapse=False)# estimates parameters
            cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                                    (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
            
            #remove nan value in first row
            cleaned_orderbook = cleaned_orderbook.drop(cleaned_orderbook.index[0])
            #change true to 1 and false to 0 in buy_sell_flag
            cleaned_orderbook['BUY_SELL_FLaAG'] = cleaned_orderbook['BUY_SELL_FLAG'].astype(int)

            # change "LIMIT_ORDER" to 0, "ORDER_EXECUTED" to 1, "ORDER_CANCELLED" to 2
            cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('LIMIT_ORDER', 0)
            cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_EXECUTED', 1)
            cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_CANCELLED', 2)
            
            cleaned_orderbook = cleaned_orderbook.iloc[:,1:]
            result.append(cleaned_orderbook.to_numpy().flatten().tolist())

        return result
        
    """def Market_sim_true(self, k):
        """"""
        k market simulations for n_timstep time using abides package with a configuration /
        of the true parameters.
        Parameters
        ----------
        num_noise: number of noise agents
        num_momentum: number of momentum agents
        num_value:number of valuem agents

        Return:
        List of length k each containing the order book of the simulation
        """"""
        #### make python file execute powershell command with bap.py as config

        result = []

        for i in range(k):
            subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 '10:00:00' '11:30:00' "
            f"-z {self.starting_cash} -r {self.r_bar} -g {self.sigma_n} -k {self.kappa} "
            f"-b {self.lambda_a}"], shell=True)
            print("complete2")
            

            processed_orderbook =  make_orderbook_for_analysis("log/bap_timestep_true/EXCHANGE_AGENT.bz2", "log/bap_timestep_true/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                               hide_liquidity_collapse=False)# estimates parameters
            #remove nan value in first row
            cleaned_orderbook = cleaned_orderbook.drop(cleaned_orderbook.index[0])
            #change true to 1 and false to 0 in buy_sell_flag
            cleaned_orderbook['BUY_SELL_FLAG'] = cleaned_orderbook['BUY_SELL_FLAG'].astype(int)

            # change "LIMIT_ORDER" to 0, "ORDER_EXECUTED" to 1, "ORDER_CANCELLED" to 2
            cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('LIMIT_ORDER', 0)
            cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_EXECUTED', 1)
            cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_CANCELLED', 2)

            #change the date type of the time column, the first column, to float
            #cleaned_orderbook.iloc[:,0] = cleaned_orderbook.iloc[:,0].astype(float)
            # remove temporarily the first column, might be removed already
            #cleaned_orderbook = cleaned_orderbook.iloc[:,1:]

            result.append(processed_orderbook.to_numpy().flatten()[-5:].tolist())

        return result"""


### Summary Statistics

"""from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF

PLOT_PARAMS_DICT = None
# binwidth = 120
LIQUIDITY_DROPOUT_BUFFER = 360  # Time in seconds used to "buffer" as indicating start and end of trading


def create_orderbooks(exchange_path, ob_path):
    """ "Creates orderbook DataFrames from ABIDES exchange output file and orderbook output file." """

    print("Constructing orderbook...")
    processed_orderbook = make_orderbook_for_analysis(exchange_path, ob_path, num_levels=1,
                                                      hide_liquidity_collapse=False)
    cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                            (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
    transacted_orders = cleaned_orderbook.loc[cleaned_orderbook.TYPE == "ORDER_EXECUTED"]
    transacted_orders['SIZE'] = transacted_orders['SIZE'] / 2

    return processed_orderbook, transacted_orders, cleaned_orderbook


def bin_and_sum(s, binwidth):
    """ "Sums the values of a pandas Series indexed by Datetime according to specific binwidth."

"        :param s: series of values to process"
"        :type s: pd.Series with pd.DatetimeIndex index"
"        :param binwidth: width of time bins in seconds"
"        :type binwidth: float"
"""
    bins = pd.interval_range(start=s.index[0].floor('min'), end=s.index[-1].ceil('min'),
                             freq=pd.DateOffset(seconds=binwidth))
    binned = pd.cut(s.index, bins=bins)
    counted = s.groupby(binned).sum()
    return counted



def make_liquidity_dropout_events(processed_orderbook):
    """ """Return index series corresponding to liquidity dropout point events for bids and asks.""" """
    no_bid_side = processed_orderbook.loc[processed_orderbook['MID_PRICE'] < - MID_PRICE_CUTOFF]
    no_ask_side = processed_orderbook.loc[processed_orderbook['MID_PRICE'] > MID_PRICE_CUTOFF]
    no_bid_idx = no_bid_side.index[~no_bid_side.index.duplicated(keep='last')]
    no_ask_idx = no_ask_side.index[~no_ask_side.index.duplicated(keep='last')]

    return no_bid_idx, no_ask_idx



def load_fundamental(ob_path):
    """ "Retrieves fundamental path from orderbook path." """

    # get ticker name from ob path ORDERBOOK_TICKER_FULL.bz2
    basename = os.path.basename(ob_path)
    ticker = basename.split('_')[1]

    # fundamental path from ticker fundamental_TICKER.bz2
    fundamental_path = f'{os.path.dirname(ob_path)}/fundamental_{ticker}.bz2'

    # load fundamental as pandas series
    if os.path.exists(fundamental_path):
        fundamental_df = pd.read_pickle(fundamental_path)
        fundamental_ts = fundamental_df['FundamentalValue'].sort_index() / 100  # convert to USD from cents
        fundamental_ts = fundamental_ts.loc[~fundamental_ts.index.duplicated(keep='last')]

        return fundamental_ts
    else:
        return None"""


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
            raise TypeError("Input data should be of type ndarray, but found type {}".format(type(data)))
        
        num_element = len(data)
        print(num_element)
        result = np.zeros(shape = (num_element, 11))
        # Compute statistics
        for ind_element in range(0, num_element):
            print(data[ind_element].shape[1])
            data_ind_element = data[ind_element].reshape(data[ind_element].shape[1]//13, 13)
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
            orderbook = np.delete(orderbook, [0, 4], axis=1)
            #orderbook = orderbook.drop(columns=['ORDER_ID', 'TYPE'])
            #drop headers
            #orderbook = orderbook.drop(orderbook.index[0])
            
            #return the mean of the orderbook
            result[ind_element, :] = np.mean(orderbook, axis=0).tolist() #add fundamental

            """# Expand the data with polynomial expansion
        result = self._polynomial_expansion(result)"""
        return np.array(result)




    # get the first and last time of the fundamental
    #if fundamental_ts is not None:
    #    first_time_fundamental = fundamental_ts.index[0]
    #    last_time_fundamental = fundamental_ts.index[-1]
    #else:
    #    first_time_fundamental = None
    #    last_time_fundamental = None
    
    
from abcpy.discretemodels import DiscreteUniform
from abcpy.continuousmodels import Uniform

"""noise = DiscreteUniform([[0], [100]], name = "noise") #10000
momentum = DiscreteUniform([[0], [50]], name = "momentum") #50
value = DiscreteUniform([[0], [20]], name = "value") #200"""

noise = Uniform([[0], [10000]], name = "noise") #100
momentum = Uniform([[0], [50]], name = "momentum") #50
value = Uniform([[0], [200]], name = "value") #20
model = Model([noise, momentum, value], name = "model")



## define the summary statistic
#statistics_calculator = Identity()#degree = 1, cross = False)
statistics_calculator = SummaryStatistics(degree = 1, cross = False)

# Define distance
from abcpy.distances import Euclidean
distance_calculator = Euclidean(statistics_calculator)


# Define perturbation kernel
from abcpy.perturbationkernel import DefaultKernel
kernel = DefaultKernel([noise, momentum, value])

## Define backend
backend = Backend()


from abcpy.inferences import SMCABC

## Generate observations
true_parameter_values = [5000, 25, 100]
k=2
observation = model.forward_simulate(true_parameter_values, k, rng = np.random.RandomState(1))
#print sum of values in observation
print(observation)
print(observation[0])
stat_obs = statistics_calculator.statistics(observation)


try:
    journal = Journal.fromFile("Results/test_sumstats1_smcabc_big.jrnl")
except FileNotFoundError:
    print("Run with inference SMCABC")
    sampler = SMCABC([model], [distance_calculator], backend, kernel, seed = 1)
    # Define sampling parameters
    #full output = 0 for no intermediary values
    #steps, n_samples, n_samples_per_param, full_output = 20, 10000, 1, 0
    #steps, n_samples, n_samples_per_param, full_output = 2, 100, 1, 0 # quicker
    steps, n_samples, n_samples_per_param, full_output = 2, k, k, 0
    # Sample
    journal = sampler.sample([observation], steps, n_samples,
                             n_samples_per_param, full_output = full_output)
    # save the final journal file
    journal.save("Results/test_sumstats1_smcabc_big.jrnl")

posterior_samples = np.array(journal.get_accepted_parameters()).squeeze()

print(posterior_samples.shape)
print(np.mean(posterior_samples, axis=0))

# plot the posterior
"""journal.plot_posterior_distr(double_marginals_only = True, show_samples = False,
                             true_parameter_values = true_parameter_values,
                             path_to_save = "../Figures/test_sumstats1_smcabc.pdf")"""




"""if isinstance(data, list):
        if np.array(data).shape == (len(data),):
            if len(data) == 1:
                data = np.array(data).reshape(1, 1)
            data = np.array(data).reshape(len(data), 1)
        else:
            data = np.concatenate(data).reshape(len(data), -1)
    else:
        raise TypeError("Input data should be of type ndarray, but found type {}".format(type(data)))
    
    print(np.array(data).shape)
    num_element = len(data)
    print(num_element)
    result = np.zeros(shape = (num_element, 12))
    # Compute statistics
    for ind_element in range(0, num_element):
        data_ind_element = data[ind_element].reshape(data[ind_element].shape[0]//12, 12)
        data_ind_element = pd.DataFrame(data_ind_element)
        # get the first and last time of the orderbook
        first_time = data_ind_element.index[0]
        last_time = data_ind_element.index[-1]
        start_time = first_time + prop_time_skip*(last_time - first_time)
        #retain only the part of the orderbook after the start_time
        orderbook = data_ind_element.loc[data_ind_element.index >= start_time]
        # remove the second and sixth columns corresponding to the order id and the type of order
        orderbook = orderbook.to_numpy()
        orderbook = np.delete(orderbook, [0, 4], axis=1)
        #orderbook = orderbook.drop(columns=['ORDER_ID', 'TYPE'])
        #drop headers
        #orderbook = orderbook.drop(orderbook.index[0])
        
        #return the mean of the orderbook
        result[ind_element, :] = np.mean(orderbook, axis=1).tolist()"""



"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI utility for inspecting liquidity issues and transacted volumes '
                                                 'for a day of trading.')

    parser.add_argument('stream', type=str, help='ABIDES order stream in bz2 format. '
                                                 'Typical example is `ExchangeAgent.bz2`')
    parser.add_argument('book', type=str, help='ABIDES order book output in bz2 format. Typical example is '
                                               'ORDERBOOK_TICKER_FULL.bz2')
    parser.add_argument('-o', '--out_file',
                        help='Path to png output file. Must have .png file extension',
                        type=check_str_png,
                        default='liquidity_telemetry.png')
    parser.add_argument('-t', '--plot-title',
                        help="Title for plot",
                        type=str,
                        default=None
                        )
    parser.add_argument('-v', '--verbose',
                        help="Print some summary statistics to stderr.",
                        action='store_true')
    parser.add_argument('-c', '--plot-config',
                        help='Name of config file to execute. '
                             'See configs/telemetry_config.example.json for an example.',
                        default='configs/telemetry_config.example.json',
                        type=str)

    args, remaining_args = parser.parse_known_args()

    out_filepath = args.out_file
    stream = args.stream
    book = args.book
    title = args.plot_title
    verbose = args.verbose
    with open(args.plot_config, 'r') as f:
        PLOT_PARAMS_DICT = json.load(f)

    main(stream, book, title=title, outfile=out_filepath, verbose=verbose)"""


