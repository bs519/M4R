import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import os
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)


#### experiment 1.2

### experiment 1.2.1

sim_num = [1, 2, 3, 5, 10]
startsimtime = ['00:15:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '03:00:00']
ratio = np.zeros((5, 6))
for i_index in range(5):
    i = sim_num[i_index]
    for j_index in range(6):
        try:
            j = startsimtime[j_index]
            means = pd.read_csv(f"Results/experiment1.2/experiment1.2.1/{i}/{j}/means.csv", index_col=False)
            mean_inf_agent = means[means.index== "InferenceAgent"][0]
            mean_ZeroIntelligence_agent = means[means.index== "ZeroIntelligenceAgent"][0]
            if mean_inf_agent>0 and mean_ZeroIntelligence_agent>0:
                ratio[i_index, j_index] = mean_inf_agent/mean_ZeroIntelligence_agent #(mean_inf_agent - mean_ZeroIntelligence_agent)/mean_ZeroIntelligence_agent
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j_index] = (mean_inf_agent/mean_ZeroIntelligence_agent)^(-1)
            elif mean_inf_agent>0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j_index] = -(mean_inf_agent/mean_ZeroIntelligence_agent)
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent >0:
                ratio[i_index, j_index] = -(mean_inf_agent/mean_ZeroIntelligence_agent)^(-1)  
            else:
                print("mean_inf or mean_ZeroIntelligence is 0")
        except FileNotFoundError:
            continue

ratio_optimal_index = np.argmax(ratio)
print("optimal ratio index", ratio_optimal_index)
ratio_optimal = np.max(ratio)
print("optimal ratio:", ratio_optimal)
print("corresponding simulation number and start inference time:", [sim_num[ratio_optimal_index[0]], startsimtime[ratio_optimal_index[1]]])

# use 3D plotting or plot just ratio in terms of sim_mun at best startsimtime and in terms of statsimtime at best sim_num
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(sim_num, startsimtime, ratio)
ax.set_xlabel('sim_num')
ax.set_ylabel('startsimtime')
ax.set_zlabel('ratio')
plt.show()


# or 

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colors = ['r', 'g', 'b', 'y', 'c']
yticks = [1, 2, 3, 4, 5]
y_vals = [1, 2, 3, 5, 10]
for c, k in zip(colors, yticks):
    # Generate the random data for the y=k 'layer'.
    xs = [15, 30, 60, 90, 120, 180] #['00:15:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '03:00:00']
    ys = ratio[k, :]
    t = y_vals[k]
    cs = [c] * len(xs)
    
    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax.bar(xs, ys, zs=t, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('sim_num')
ax.set_ylabel('startsimtime')
ax.set_zlabel('ratio')

# On the y-axis let's only label the discrete values that we have data for.
ax.set_yticks(yticks)
ax.set_yticklabels(y_vals)

fig.suptitle('Ratio in terms of simulation start time and number of simulations')
plt.show()



### experiment 1.2.2

m = [0.1, 0.5, 1.0, 2.0, 5.0, 25.0]
ratio = np.zeros((6, 5))
for i_index in range(6):
    i = m[i_index]
    for j_index in range(5):
        try:
            means = pd.read_csv(f"Results/experiment1.2/experiment1.2.2/{i}/{j_index}/means.csv", index_col=False)
        except FileNotFoundError:
            continue
        else:
            mean_inf_agent = means[means['Unnamed: 0']== "InferenceAgent"]["mean"].iloc[0]
            mean_ZeroIntelligence_agent = means[means['Unnamed: 0']== "ZeroIntelligenceAgent"]["mean"].iloc[0]
            print(mean_inf_agent, mean_ZeroIntelligence_agent)
            if mean_inf_agent>0 and mean_ZeroIntelligence_agent>0:
                ratio[i_index, j_index] = mean_inf_agent/mean_ZeroIntelligence_agent #(mean_inf_agent - mean_ZeroIntelligence_agent)/mean_ZeroIntelligence_agent
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j_index] = mean_ZeroIntelligence_agent/mean_inf_agent
            elif mean_inf_agent>0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j_index] = -(mean_inf_agent/mean_ZeroIntelligence_agent)
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent >0:
                ratio[i_index, j_index] = -mean_ZeroIntelligence_agent/mean_inf_agent
            else:
                print("mean_inf or mean_ZeroIntelligence is 0")
            print(f"ratio for {i_index} and {j_index}", ratio[i_index, j_index])

print(ratio)
ratio[ratio == 0] = np.nan
ratio = np.nanmean(ratio, axis=1)
ratio = np.nan_to_num(ratio)
print(ratio)

ratio_optimal_index = np.argmax(ratio)
print("optimal ratio index", ratio_optimal_index)
ratio_optimal = np.max(ratio)
print("optimal ratio:", ratio_optimal)
print("corresponding simulation number and start inference time:", m[ratio_optimal_index])

# plot ratio in terms of m

plt.plot(m, ratio)  # plot ratio in terms of m
# add line y = 1
plt.axhline(y=1, color='r', linestyle='-')
# label line y = 1
plt.text(15, 4, 'y = 1', color='red', fontsize=14)
plt.xlabel('m')
plt.ylabel('ratio')
plt.title('Ratio in terms of m')
plt.savefig('Figures/experiment1.2/experiment1.2.2/ratio_m.png')
plt.show()





### experiment 2.1

num_agent = [1, 2, 5, 10, 25]
ratio = np.zeros((5, 5))
for i_index in range(5):
    i = m[i_index]
    for j in range(5):
        try:
            means = pd.read_csv(f"Results/experiment2/experiment2.1/{i}/{j}/means.csv", index_col=False)
        except FileNotFoundError:
            continue
        else:
            mean_inf_agent = means[means['Unnamed: 0']== "InferenceAgent"]["mean"].iloc[0]
            mean_ZeroIntelligence_agent = means[means['Unnamed: 0']== "ZeroIntelligenceAgent"]["mean"].iloc[0]
            if mean_inf_agent>0 and mean_ZeroIntelligence_agent>0:
                ratio[i_index, j] = mean_inf_agent/mean_ZeroIntelligence_agent #(mean_inf_agent - mean_ZeroIntelligence_agent)/mean_ZeroIntelligence_agent
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j] = mean_ZeroIntelligence_agent/mean_inf_agent
            elif mean_inf_agent>0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j] = -(mean_inf_agent/mean_ZeroIntelligence_agent)
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent >0:
                ratio[i_index, j] = -mean_ZeroIntelligence_agent/mean_inf_agent  
            else:
                print("mean_inf or mean_ZeroIntelligence is 0")

ratio[ratio == 0] = np.nan
ratio = np.nanmean(ratio, axis=1)
ratio = np.nan_to_num(ratio)

ratio_optimal_index = np.argmax(ratio)
print("optimal ratio index", ratio_optimal_index)
ratio_optimal = np.max(ratio)
print("optimal ratio:", ratio_optimal)
print("corresponding simulatinon number and start inference time:", num_agent[ratio_optimal_index])

# plot ratio in terms of num_agent
plt.plot(num_agent, ratio)  # plot ratio in terms of num_agent
plt.xlabel('num_agent')
plt.ylabel('ratio')
plt.title('Ratio in terms of num_agent')
plt.show()





### experiment 2.2

num_agent = [1, 100, 1000] # add 1
ratio = np.zeros((3, 5))
for i_index in range(3):
    i = num_agent[i_index]
    for j in range(5):
        try:
            means = pd.read_csv(f"Results/experiment2/experiment2.2/{i}/{j}/means.csv", index_col=False) # add 1 agent case from prev exp
        except FileNotFoundError:
            continue
        else:
            mean_inf_agent = means[means['Unnamed: 0']== "InferenceAgent"]["mean"].iloc[0]
            mean_ZeroIntelligence_agent = means[means['Unnamed: 0']== "ZeroIntelligenceAgent"]["mean"].iloc[0]
            if mean_inf_agent>0 and mean_ZeroIntelligence_agent>0:
                ratio[i_index, j] = mean_inf_agent/mean_ZeroIntelligence_agent #(mean_inf_agent - mean_ZeroIntelligence_agent)/mean_ZeroIntelligence_agent
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j] = mean_ZeroIntelligence_agent/mean_inf_agent
            elif mean_inf_agent>0 and mean_ZeroIntelligence_agent <0:
                ratio[i_index, j] = -(mean_inf_agent/mean_ZeroIntelligence_agent)
            elif mean_inf_agent<0 and mean_ZeroIntelligence_agent >0:
                ratio[i_index, j] = -mean_ZeroIntelligence_agent/mean_inf_agent 
            else:
                print("mean_inf or mean_ZeroIntelligence is 0")

ratio[ratio == 0] = np.nan
ratio = np.nanmean(ratio, axis=1)
ratio = np.nan_to_num(ratio)

ratio_optimal_index = np.argmax(ratio)
print("optimal ratio index", ratio_optimal_index)
ratio_optimal = np.max(ratio)
print("optimal ratio:", ratio_optimal)
print("corresponding simulatinon number and start inference time:", ratio[ratio_optimal_index])

# plot plot of ratio in terms of volume traded
plt.plot(num_agent, ratio)  # plot ratio in terms of volume traded
plt.xlabel('volume traded')
plt.ylabel('ratio')
plt.title('Ratio in terms of volume traded')
plt.show()




### Experiment 2.3

try:
    means = pd.read_csv(f"Results/experiment2/experiment2.3/means.csv", index_col=False) # add 1 agent case from prev exp
except FileNotFoundError:
    print("File not found")
else:
    mean_inf_agent = means[means.index== "InferenceAgent"].iloc[0]
    mean_ZeroIntelligence_agent = means[means.index== "NoiseAgent"].iloc[0]
    if mean_inf_agent>0 and mean_ZeroIntelligence_agent>0:
        ratio = mean_inf_agent/mean_ZeroIntelligence_agent #(mean_inf_agent - mean_ZeroIntelligence_agent)/mean_ZeroIntelligence_agent
    elif mean_inf_agent<0 and mean_ZeroIntelligence_agent <0:
        ratio = (mean_inf_agent/mean_ZeroIntelligence_agent)^(-1)
    elif mean_inf_agent>0 and mean_ZeroIntelligence_agent <0:
        ratio[i_index, j_index] = -(mean_inf_agent/mean_ZeroIntelligence_agent)
    elif mean_inf_agent<0 and mean_ZeroIntelligence_agent >0:
        ratio[i_index, j_index] = -(mean_inf_agent/mean_ZeroIntelligence_agent)^(-1)  
    else:
        print("mean_inf or mean_ZeroIntelligence is 0")

print(ratio)







