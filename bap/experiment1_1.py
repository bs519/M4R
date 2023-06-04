import numpy as np
from abcpy.output import Journal
import matplotlib.pyplot as plt
import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

#EXPERIMENT 1.1 analysis
journal = Journal.fromFile(f"Results/experiment1.1/experiment1.1.1/journal.jrnl")
print(np.array(journal.get_accepted_parameters()).shape)

posterior_samples = np.array(journal.get_accepted_parameters()).squeeze()
print("posterior samples:", np.mean(posterior_samples, axis=0))
print("posterior mean:", journal.posterior_mean())
print("weights", journal.get_weights())
#print(np.mean(posterior_samples, axis=0))

true_parameter_values = [50, 25, 10]

fig, ax = journal.plot_ESS()
fig.savefig("Figures/experiment1/ESS_exp1.1.pdf")
plt.close(fig)

fig, ax, wass_dist_lists = journal.Wass_convergence_plot()
fig.savefig("Figures/experiment1/Wass_exp1.1.pdf")




"""journal.plot_posterior_distr(double_marginals_only = True, show_samples = False, iteration =None,
                             true_parameter_values = true_parameter_values,
                             path_to_save = "Figures/experiment1/posterior_distribution_exp1.1.pdf")"""