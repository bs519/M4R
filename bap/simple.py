import abcpy
import numpy as np
import pandas as pd

# define priors
from abcpy.continuousmodels import Uniform, Normal as Gaussian
mu = Uniform([[150], [200]], name="mu")
sigma = Uniform([[5], [25]], name="sigma")
# define the model
height = Gaussian([mu, sigma], name='height')

# generate 3 observations from the model with mean 185 and standard deviation 20
x_sim = height.forward_simulate([185, 20], k=3)
print(np.array(x_sim).shape)

from abcpy.statistics import Identity
statistics_calculator = Identity(degree=2, cross=False)

from abcpy.distances import Euclidean
distance_calculator = Euclidean(statistics_calculator)

# generate two observation:
x_1 = height.forward_simulate([185, 20], k=1)
x_2 = height.forward_simulate([170, 20], k=1)

print(distance_calculator.distance(x_1, x_2))

from abcpy.backends import BackendDummy as Backend
backend = Backend()

from abcpy.inferences import RejectionABC
sampler = RejectionABC([height], [distance_calculator], backend, seed=1)

height_obs = height.forward_simulate([170, 15], k=50)
print(np.array(height_obs[0]).shape)
# this may take a while according to the setup
n_sample, n_samples_per_param = 250, 10
epsilon = 5000
journal = sampler.sample([height_obs], n_sample, n_samples_per_param, epsilon)

params = journal.get_parameters()  # this returns a dict whose keys are parameter names
print("Number of posterior samples: {}".format(len(params['mu'])))
print("10 posterior samples for mu:")
print(params['mu'][0:10])
print("len of params['mu']:", len(params['mu']))

print("Posterior mean", journal.posterior_mean())
print("Covariance matrix:")
print(journal.posterior_cov())

print(journal.configuration)

journal.plot_posterior_distr(true_parameter_values=[170,15])

"""from abcpy.output import Journal
journal.save("experiments.jnl")
new_journal = Journal.fromFile('experiments.jnl')"""