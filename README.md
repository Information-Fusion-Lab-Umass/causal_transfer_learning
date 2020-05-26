
# causal_transfer_learning
Repo containing code for running experiments related to causal transfer learning project.

# Reinstall mazelab package

pip install -e .

# To generate data from different running environments, run following bash script

Add game_type in below shell script
./data.sh

# To generate data for single environment (height=10 and width = 10), run the following command
PYTHONPATH=$PWD python codes/data/generate_data.py --random_obstacles 1 --height 10 --width 10

# To analyze and save combined training data, run the following command:
PYTHONPATH=$PWD python codes/data/analyze_data.py --game_type bw --start 5 --stop 105

# To run the neural-network based model, use following command.
PYTHONPATH=$PWD python codes/models/relation_learning/abstract_relation_reasoning.py --model linear --sparse 0 --group_lasso 1 --mode both --action up --penalty 1


# To run the structure learning linear model, use following command.
PYTHONPATH=$PWD python codes/data/test_notears_linear.py --game_type all_random --l 0.1 --rho 1 --alpha 0.0 --mode eval

# To run the structure learning non-linear model, use following command.
PYTHONPATH=$PWD python codes/data/test_notears_nonlinear.py --game_type all_random --l1 0.01 --l2 0.01 rho 1.0 --mode eval

# To run q-learning algorithm, use the following command.
PYTHONPATH=$PWD python codes/models/rl_approaches/q_learning.py --height 10 --width 10 --render 1

# To run DQN algorithm, use the following command.
PYTHONPATH=$PWD python codes/models/rl_approaches/DQN_main.py --height 10 --width 10 --render 0 --game_type trigger_non_markov --mode both
