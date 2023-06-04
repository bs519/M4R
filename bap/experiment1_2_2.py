import subprocess
from tqdm import tqdm

import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

#### EXPERIMENT 1.2.2: Find optimal m now that we have optimal start simulation time and simulation number

# even if optimal step and particle number are different, use these parameters to not waste time
################ the parameters don't impact the optimal m and simulation start time????

def experiment4(sim_number, start_sim_time):
    for i in tqdm([0.1, 0.5, 1, 2, 5, 10, 25]):
        print(f"Simulation with m = {i}")
        subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 --start-time '9:30:00' --end-time '11:00:00' -l experiment_4 -o 1 -i 1 -x experiment1.2/experiment1.2.2 -y {sim_number} -w {start_sim_time}"], shell=True)


if __name__=='__main__':
    optimal_sim_number = 5
    optimal_start_sim_time = "00:05:00"
    experiment4(optimal_sim_number, optimal_start_sim_time)