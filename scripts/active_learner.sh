#!/bin/bash
cd ~/research/research-master/causal_transfer_learning
for (( i=10; i <=100; i = i + 10 ))
do
	PYTHONPATH=$PWD python codes/data/test_notears_nonlinear.py --game_type all_random --l1 0.01 --l2 0.01 --rho 1 --mode both --disp False --train_frac $i --save_results 1
done
