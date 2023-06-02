from math import cos, sin, pi

import abcpy
import matplotlib.mlab as mlab
import numpy as np
import scipy
from matplotlib import gridspec, pyplot as plt
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.continuousmodels import Uniform
from abcpy.statistics import Identity
from abcpy.distances import Euclidean
from abcpy.backends import BackendDummy as Backend

"""import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)
from realism.realism_utils import make_orderbook_for_analysis, MID_PRICE_CUTOFF


processed_orderbook =  make_orderbook_for_analysis("log/bap_timestep/EXCHANGE_AGENT.bz2", "log/bap_timestep/ORDERBOOK_ABM_FULL.bz2", num_levels=1,
                                                    hide_liquidity_collapse=False)# estimates parameters
cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                        (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]

#remove nan value in first row
cleaned_orderbook = cleaned_orderbook.drop(cleaned_orderbook.index[0])
#change true to 1 and false to 0 in buy_sell_flag
cleaned_orderbook['BUY_SELL_FLAG'] = cleaned_orderbook['BUY_SELL_FLAG'].astype(int)

# change "LIMIT_ORDER" to 0, "ORDER_EXECUTED" to 1, "ORDER_CANCELLED" to 2
cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('LIMIT_ORDER', 0)
cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_EXECUTED', 1)
cleaned_orderbook['TYPE'] = cleaned_orderbook['TYPE'].replace('ORDER_CANCELLED', 2)

cleaned_orderbook = cleaned_orderbook.iloc[:,1:]
result = cleaned_orderbook.to_numpy().flatten()[-5:].tolist()
print(result)
"""

class BivariateNormal(ProbabilisticModel, Continuous):

    def __init__(self, parameters, name='BivariateNormal'):
        # We expect input of type parameters = [m1, m2, s1, s2, alpha]
        if not isinstance(parameters, list):
            raise TypeError('Input of Normal model is of type list')

        if len(parameters) != 5:
            raise RuntimeError('Input list must be of length 5, containing [m1, m2, s1, s2, alpha].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 5:
            raise ValueError('Number of parameters of BivariateNormal model must be 5.')

        # Check whether input is from correct domain
        m1 = input_values[0]
        m2 = input_values[1]
        s1 = input_values[2]
        s2 = input_values[3]
        alpha = input_values[4]
        if s1 < 0 or s2 < 0:
            return False

        return True

    def _check_output(self, values):
        if not isinstance(values, np.array):
            raise ValueError('This returns a bivariate array')
        
        if values.shape[0] != 2: 
            raise RuntimeError('The size of the output has to be 2.')
        
        return True

    def get_output_dimension(self):
        return 2

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Extract the input parameters
        m1 = input_values[0]
        m2 = input_values[1]
        s1 = input_values[2]
        s2 = input_values[3]
        alpha = input_values[4]
        
        mean = np.array([m1, m2])
        cov = self.get_cov(s1, s2, alpha)
        
        obs_pd = multivariate_normal(mean=mean, cov=cov)
        vector_of_k_samples = obs_pd.rvs(k)

        # Format the output to obey API
        result = [np.array([x]) for x in vector_of_k_samples]
        return result

    def get_cov(self, s1, s2, alpha):
        """Function to generate a covariance bivariate covariance matrix; it starts from considering a
        diagonal covariance matrix with standard deviations s1, s2 and then applies the rotation matrix with 
        angle alpha. """
        r = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]]) # Rotation matrix
        e = np.array([[s1, 0], [0, s2]]) # Eigenvalue matrix
        rde = np.dot(r, e)
        rt = np.transpose(r)
        cov = np.dot(rde, rt)
        return cov
    

def plot_dspace(ax, sl, marker, color):
    """Plot the data in 'sl' on 'ax';"""
    ax.set_xlim(100,220)
    ax.set_ylim(30,150)
    ax.set_xlabel('Height in cm')
    ax.set_ylabel('Weigth in kg')
    for samples in sl:
        ax.plot(samples[:,0], samples[:,1], marker, c=color)


def plot_pspace(ax_means, ax_vars, ax_angle, m1, m2, s1, s2, alpha, color):
    """Plot parameter space. m1 and m2 are the means of the height and weight respectively, while s1, s2 are 
    two standard deviations for the eigenvalue normal components. Finally, alpha is the angle that determines the 
    amount of rotation applied to the two independent components to get the covariance matrix."""
    ax_means.set_xlabel('Mean of height')
    ax_means.set_ylabel('Mean of weight')
    ax_means.set_xlim(120,200)
    ax_means.set_ylim(50,150)
    ax_means.plot(m1, m2, 'o', c=color)

    ax_vars.set_xlabel('Standard deviation 1')
    ax_vars.set_ylabel('Standard deviation 2')
    ax_vars.set_xlim(0,100)
    ax_vars.set_ylim(0,100)
    ax_vars.plot(s1, s2, 'o', c=color)
    
    ax_angle.set_xlabel('Rotation angle')
    ax_angle.set_xlim(0, pi/2)
    ax_angle.set_yticks([])
    ax_angle.plot(np.linspace(0, pi, 10), [0]*10, c='black', linewidth=0.2)
    ax_angle.plot(alpha, 0, 'o', c=color)


def plot_all(axs, m1, m2, s1, s2, alpha, color, marker, model, k):
    """Function plotting pameters, generating data from them and plotting data too. It uses the model 
    to generate k samples from the provided set of parameters. 
    
    m1 and m2 are the means of the height and weight respectively, while s1, s2 are 
    two standard deviations for the eigenvalue normal components. Finally, alpha is the angle that determines the 
    amount of rotation applied to the two independent components to get the covariance matrix.
    """
    ax_pspace_means, ax_pspace_vars, ax_pspace_angle, ax_dspace = axs
    plot_pspace(ax_pspace_means, ax_pspace_vars, ax_pspace_angle, m1, m2, s1, s2, alpha, color)
    samples = model.forward_simulate([m1, m2, s1, s2, alpha], k)
    plot_dspace(ax_dspace, samples, marker, color)


m1 = Uniform([[120], [200]], name="Mean_height")
m2 = Uniform([[50], [150]], name="Mean_weigth")
s1 = Uniform([[0], [100]], name="sd_1")
s2 = Uniform([[0], [100]], name="sd_2")
alpha = Uniform([[0], [pi/2]], name="alpha")

bivariate_normal = BivariateNormal([m1, m2, s1, s2, alpha])

obs_par = np.array([175, 75, 90, 35, pi/4.])
obs = bivariate_normal.forward_simulate(obs_par, 100)
print(obs[0])


statistics_calculator = Identity()
distance_calculator = Euclidean(statistics_calculator)
from abcpy.perturbationkernel import DefaultKernel
kernel = DefaultKernel([m1, m2, s1, s2, alpha])
backend = Backend()



from abcpy.inferences import SMCABC

sampler = SMCABC([bivariate_normal], [distance_calculator], backend, kernel, seed=1)
n_samples = 100  # number of posterior samples we aim for
n_samples_per_param = 100  # number of simulations for each set of parameter values
#journal = sampler.sample([obs], n_samples, n_samples_per_param, epsilon=15)
steps, n_samples, n_samples_per_param, full_output = 2, 50, 100, 0 # quicker

print(bivariate_normal.forward_simulate(obs_par, 2)[0].shape) #np.array transposes?
# Sample
journal = sampler.sample([obs], steps, n_samples,
                            n_samples_per_param, full_output = full_output)
print(journal.number_of_simulations)

posterior_samples = np.array(journal.get_accepted_parameters()).squeeze()

print(posterior_samples.shape)
print(np.mean(posterior_samples, axis=0))