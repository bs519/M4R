"""
## Simulator
@author: bapti
"""
import logging
import abides
import abcpy
#abcpy.settings.set_figure_params('abcpy')
# for debugging
for logger in ["ABC.Distance", "ABC.Predictor", "ABC.Sumstat"]:
    logging.getLogger(logger).setLevel(logging.DEBUG)

import numpy as np
import os
import tempfile
import datetime as dt
import scipy as sp

from config import rmsc03_4 as market
from abcpy.continuousmodels import Uniform



# Define distance
from abcpy.distances import Euclidean
distance_calculator = Euclidean(statistics_calculator)

# Define perturbation kernel
from abcpy.perturbationkernel import DefaultKernel
kernel = DefaultKernel([theta1, theta2])

## Define backend
backend = Backend()

from abcpy.inferences import PMCABC


## Generate a fake observation
true_parameter_values = [2, .1]
observation = lorenz.forward_simulate([2, .1, sigma_e, phi, T], 1, rng = np.random.RandomState(42))

try:
    journal = Journal.fromFile("Results/lorenz_hakkarainen_pmcabc.jrnl")
except FileNotFoundError:
    print("Run inference with PMCABC")
    sampler = PMCABC([lorenz], [distance_calculator], backend, kernel, seed = 1)
    # Define sampling parameters
    steps, n_samples, n_samples_per_param, full_output = 3, 10000, 1, 0
    eps_arr = np.array([500]); eps_percentile = 10
    


    # Sample
    journal = sampler.sample([observation], steps, eps_arr, n_samples,
                             n_samples_per_param, eps_percentile, full_output = full_output)
    # save the final journal file
    journal.save("Results/lorenz_hakkarainen_pmcabc.jrnl")

# print posterior mean and variance
print(journal.posterior_mean())
print(journal.posterior_cov())

# plot the posterior
journal.plot_posterior_distr(double_marginals_only = True, show_samples = False,
                             true_parameter_values = true_parameter_values,
                             path_to_save = "../Figures/lorenz_hakkarainen_pmcabc.pdf")