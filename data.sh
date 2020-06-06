#!/bin/bash
# cd ~/research/research-master/causal_transfer_learning/
for (( i=5; i <=75; i = i + 5 ))
do
	echo $i;
	PYTHONPATH=$PWD python codes/data/generate_data_ranom_exploration.py --random_obstacles 1 --height $i --width $i --num_episodes 50 --max_episode_length 100 --render 0 --game_type trigger_non_markov_flip
	PYTHONPATH=$PWD python codes/data/generate_data_ranom_exploration.py --random_obstacles 0 --height $i --width $i --num_episodes 50 --max_episode_length 100 --render 0 --game_type trigger_non_markov
done
