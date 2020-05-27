#!/bin/bash
#SBATCH --partition 1080ti-long
#SBATCH --mem-per-cpu=51200
#SBATCH --output=codes/logs/res_%j.txt 
cd /home/ppruthi/causal_transfer_learning/
. venv/bin/activate
PYTHONPATH=$PWD python ./codes/models/rl_approaches/DQN_main.py --height 10 --width 10 --render 0 --game_type trigger_non_markov --mode both --total-steps 10000 --EPS-DECAY 10000 --TARGET-UPDATE 2000 

