#!/bin/bash
cd ~/research/research-master/causal_transfer_learning
for (( i=10; i <=70; i = i + 5 ))
do
	PYTHONPATH=$PWD python codes/data/generate_data.py --random_obstacles 1 --height $i --width $i --game_type all_random
done

# Analyze the generated data and store it in a single tabular format
PYTHONPATH=$PWD python codes/data/analyze_data.py --game_type all_random --start 10 --stop 75

# Analyze the structure learning results (linear)
PYTHONPATH=$PWD python codes/data/test_notears.py --game_type all_random --l 0.1 --rho 1 --alpha 0.0 --mode both --disp False
