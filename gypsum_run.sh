#!/bin/bash
#SBATCH --job-name=dqn_training
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition 1080ti-long # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --time=10:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=20240    # Memory in MB per cpu allocated
#SBATCH --gres gpu:4

#cd /home/ppruthi/causal_transfer_learning/
PYTHONPATH=$PWD python ./codes/models/rl_approaches/DQN_main.py --height 10 --width 10 --render 0 --game_type trigger_non_markov --mode both
--total-steps 5000 --EPS-DECAY 2000 --TARGET-UPDATE 1000 --burning 1000 --num-trials 1 --num-episodes 10 --max-episode-length 100 --no_switches
