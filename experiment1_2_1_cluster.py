import argparse
import subprocess

import os

parser = argparse.ArgumentParser(description='Number of simulations in agent prediciton step to run for experiment 1.2.1.')

parser.add_argument('-y',
                    '--sim-num',
                    required=True,
                    type=int,
                    help='Number simulations the agent will execute')

args, remaining_args = parser.parse_known_args()



#### EXPERIMENT 1.2.1: Find optimal number of simulations and inference start time
# even if optimal step and particle number are different, use these parameters to not waste time
# the parameters don't impact the optimal m and simulation start time

def experiment4(i):
    for j in ["00:15:00", "00:30:00", "01:00:00", "01:30:00", "02:00:00", "03:00:00"]:
        print(f"Simulations {i} with wakeup time {j}")
        os.makedirs(f"Results/experiment1.2/experiment1.2.1/{i}sim/{j}", exist_ok=True)
        l = 0
        while l<10:
            try:
                subprocess.run([f"python3 -u abides.py -c bap -t ABM -d 20200603 --start-time '09:30:00' --end-time '13:00:00' -l experiment_3 -o 1 -i 1 -x experiment1.2/experiment1.2.1/{i}sim/{j} -y {i} -w {j}"], shell=True)
            except subprocess.CalledProcessError as e:
                print("error:", e.output)
                j += 1
                print(f"We try again for the {j}th time")
                continue
            else:
                break

if __name__=='__main__':
    experiment4(args.sim_num)