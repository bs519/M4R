import subprocess
from tqdm import tqdm
import os

import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

#### EXPERIMENT 1.2.1: Find optimal number of simulations and inference start time
# even if optimal step and particle number are different, use these parameters to not waste time
# the parameters don't impact the optimal m and simulation start time

def experiment4():
    for i in tqdm([1, 2, 3, 5, 10]):  ### remove tqdm when running on cluster
        for j in ['00:15:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '03:00:00']:
            print(f"Simulation {i} with wakeup time {j}")
            os.makedirs(f"Results/experiment1.2/experiment1.2.1/{i}sim/{j}", exist_ok=True)
            l = 0
            while l<3:
                try:
                    subprocess.check_output([f"python3 -u abides.py -c bap -t ABM -d 20200603 --start-time '09:30:00' --end-time '13:00:00' -l experiment_4_{i}sim_{j} -o 1 -i 1 -x experiment1.2/experiment1.2.1/{i}sim/{j} -y {i} -w {j}"], shell=True) #-j experiment1.2.1 -q {i}sim --print-means4 {j} -y {i} -w {j}"], shell=True)
                except subprocess.CalledProcessError as e:
                    print("error:", e.output)
                    l += 1
                    print(f"We try again for the {l}th time")
                    continue
                else:
                    break

if __name__=='__main__':
    
    experiment4()