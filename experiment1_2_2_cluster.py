import subprocess
import argparse

import os

parser = argparse.ArgumentParser(description='Parameter m used in Bollinger band strategy to run for experiment 1.2.2.')

parser.add_argument('-y',
                    '--m',
                    required=True,
                    type=int,
                    help='Parameter m used in Bollinger band strategy')

args, remaining_args = parser.parse_known_args()

#### EXPERIMENT 1.2.2: Find optimal m now that we have optimal start simulation time and simulation number

# even if optimal step and particle number are different, use these parameters to not waste time
################ the parameters don't impact the optimal m and simulation start time????

def experiment4(sim_number, start_sim_time, i):
        print(f"Simulation with m = {i}")
        # create repository for results and add folder name to bap config file
        os.makedirs(f"Results/experiment1.2/experiment1.2.2/{i}", exist_ok=True)
        l = 0
        while l<10:
            try:
                subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 --start-time '9:30:00' --end-time '11:00:00' -l experiment_4 -o 1 -i 1 -x experiment1.2/experiment1.2.2/ -y {sim_number} -w {start_sim_time}"], shell=True)
            except subprocess.CalledProcessError as e:
                print("error:", e.output)
                j += 1
                print(f"We try again for the {j}th time")
                continue
            else:
                break

if __name__=='__main__':
    optimal_sim_number = 5
    optimal_start_sim_time = "00:05:00"
    #i = [0.1, 0.5, 1, 2, 5, 10, 25]
    experiment4(optimal_sim_number, optimal_start_sim_time, args.m)