#!/bin/bash
# cd ~/research/research-master/causal_transfer_learning/
for (( i=5; i <=75; i = i + 5 ))
do
	echo $i;
	PYTHONPATH=$PWD python codes/data/generate_data_ranom_exploration.py --random_obstacles 1 --height $i --width $i --num_episodes 50 --max_episode_length 100 --render 0 --game_type trigger_non_markov_random
	PYTHONPATH=$PWD python codes/data/generate_data_ranom_exploration.py --random_obstacles 0 --height $i --width $i --num_episodes 50 --max_episode_length 100 --render 0 --game_type trigger_non_markov

done

# # Analyze the generated data and store it in a single tabular format
# PYTHONPATH=$PWD python codes/data/analyze_data.py --game_type all_random_invert --start 10 --stop 90
#
# # Analyze the structure learning results (linear)
# PYTHONPATH=$PWD python codes/data/test_notears_linear.py --game_type all_random_invert --l 0.1 --rho 1 --alpha 0.0 --mode both --disp False
#
# # Analyze the structure learning results (non-linear)
# PYTHONPATH=$PWD python codes/data/test_notears_nonlinear.py --game_type all_random_invert --l1 0.01 --l2 0.01 --rho 1.0 --mode both --disp False
