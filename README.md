# causal_transfer_learning
Repo containing code for running experiments related to causal transfer learning project.

# To generate data from different running environments, run following bash script

./deploy.sh

# To analyze and save combined training data, run the following command:
PYTHONPATH=$PWD python codes/data/analyze_data.py

# To run the model, use following command.
PYTHONPATH=$PWD python codes/models/relation_learning/abstract_relation_reasoning.py --model linear --sparse 1 --group_lasso 0 --mode both --action all --penalty 1
