#!/bin/bash
cd ~/research/research-master/causal_transfer_learning
for (( i=5; i <=100; i = i + 5 ))
do
	PYTHONPATH=$PWD python codes/data/generate_data.py --random_obstacles 1 --height $i --width $i
done