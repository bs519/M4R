import subprocess

import argparse

import os

parser = argparse.ArgumentParser(description='Parameter m used in Bollinger band strategy to run for experiment 2.1.')

parser.add_argument('-y',
                    '--experiment-number',
                    required=True,
                    type=int,
                    help='experiment number')

args, remaining_args = parser.parse_known_args()
import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)

#### EXPERIMENT 2.3: Now that we have optimal m, start simulation time and simulation number, observe performance on real market data.


def experiment9(sim_number, start_sim_time, m, i=1):
        # create repository for results and add folder name to bap config file
        os.makedirs(f"Results/experiment2/experiment2.3", exist_ok=True)
        l = 0
        while l<10:
            try:
                subprocess.check_output([f"python3 -u abides.py -c bap -t AAPL -d 20200603 --start-time '9:30:00' --end-time '13:00:00' -l experiment_9 -o 1 -i 1 -x experiment2/experiment2.3 -y {sim_number} -w {start_sim_time} -f {m}"], shell=True)
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
    experiment9(optimal_sim_number, optimal_start_sim_time, optimal_m, args.experiment_number)