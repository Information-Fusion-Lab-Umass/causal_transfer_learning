
# causal_transfer_learning
Repo containing code for running experiments related to causal transfer learning project.

# Reinstall mazelab package

pip install -e .

# To generate data from different running environments, run following bash script

./data.sh

# To generate data for single environment (height=10 and width = 10), run the following command
PYTHONPATH=$PWD python codes/data/generate_data.py --random_obstacles 1 --height 10 --width 10

# To analyze and save combined training data, run the following command:
PYTHONPATH=$PWD python codes/data/analyze_data.py

# To run the model, use following command.
PYTHONPATH=$PWD python codes/models/relation_learning/abstract_relation_reasoning.py --model linear --sparse 0 --group_lasso 1 --mode both --action up --penalty 1
