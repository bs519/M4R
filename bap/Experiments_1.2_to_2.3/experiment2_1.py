import subprocess
from tqdm import tqdm

import sys
import os
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

#### EXPERIMENT 2.1: Now that we have optimal m, start simulation time and simulation number, study dynamics of agent as number of agents increases


def experiment6(sim_number, start_sim_time, m):
    for i in tqdm([2, 5, 10, 25]):
        print(f"Simulation with number of agents = {i}")
        # create repository for results and add folder name to bap config file
        os.makedirs(f"Results/experiment2/experiment2.1/{i}", exist_ok=True)
        l = 0
        while l<10:
            try:
                subprocess.check_output([f"python3 -u abides.py -c bap -t ABM -d 20200603 --start-time '9:30:00' --end-time '13:00:00' -l experiment_6_{i} -o 1 -i {i} -x experiment2/experiment2.1/{i} -y {sim_number} -w {start_sim_time} -f {m}"], shell=True)
            except subprocess.CalledProcessError as e:
                print("error:", e.output)
                l += 1
                print(f"We try again for the {l}th time")
                continue
            else:
                break

if __name__=='__main__':
    # change optimal parameters to those found in experiment 1.2
    optimal_sim_number = 5
    optimal_start_sim_time = "01:30:00"
    optimal_m = 2
    experiment6(optimal_sim_number, optimal_start_sim_time, optimal_m)