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

journal = Journal.fromFile("Results/test_sumstats1_smcabc.jrnl")

# print posterior mean and variance
print(journal.posterior_mean())
print(journal.posterior_cov())

true_parameter_values = [50, 25, 10]

"""# plot the posterior
journal.plot_posterior_distr(double_marginals_only = True, show_samples = False,
                             true_parameter_values = true_parameter_values,
                             path_to_save = "../Figures/test_sumstats1_smcabc.pdf")"""

