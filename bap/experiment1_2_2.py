import subprocess
from tqdm import tqdm

import sys
import os
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

#### EXPERIMENT 1.2.2: Find optimal m now that we have optimal start simulation time and simulation number

# even if optimal step and particle number are different, use these parameters to not waste time
################ the parameters don't impact the optimal m and simulation start time????

def experiment5(sim_number, start_sim_time):
    for i in tqdm([0.1, 0.5, 1, 2, 5, 10, 25]):
        print(f"Simulation with m = {i}")
        # create repository for results and add folder name to bap config file
        os.makedirs(f"Results/experiment1.2/experiment1.2.2/{i}", exist_ok=True)
        l = 0
        while l<3:
            try:
                subprocess.check_output([f"python3 -u abides.py -c bap -t ABM -d 20200603 --start-time '9:30:00' --end-time '13:00:00' -l experiment_5_{i} -o 1 -i 1 -x experiment1.2/experiment1.2.2/{i} -y {sim_number} -w {start_sim_time} -f {i}"], shell=True)
            except subprocess.CalledProcessError as e:
                print("error:", e.output)
                l += 1
                print(f"We try again for the {l}th time")
                continue
            else:
                break

if __name__=='__main__':
    optimal_sim_number = 5
    optimal_start_sim_time = "01:30:00"
    experiment5(optimal_sim_number, optimal_start_sim_time)