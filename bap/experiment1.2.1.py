import subprocess
from tqdm import tqdm

import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

#### EXPERIMENT 1.2.1: Find optimal number of simulations and inference start time
# even if optimal step and particle number are different, use these parameters to not waste time
# the parameters don't impact the optimal m and simulation start time

def experiment3():
    for i in tqdm([1, 2, 3, 5, 10]):
        for j in ["00:05:00", "00:15:00", "00:30:00", "01:00:00", "2:00:00"]:
            print(f"Simulation {i} with wakeup time {j}")
            subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 '9:30:00' '10:00:00' -l experiment_3 -o 1 -j 1 -x experiment1.2/experiment1.2.1 -y {i} -w {j}"], shell=True)


if __name__=='__main__':
    
    experiment3()