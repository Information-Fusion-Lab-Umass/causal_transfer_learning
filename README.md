Repo containing code for running experiments related to causal transfer learning project.

- Install mazelab package

`pip install -e .`

- To generate data from different running environments, run following bash script
Add game_type in below shell script
`./data.sh`

- To generate data for single environment (height=10 and width = 10), run the following command
`PYTHONPATH=$PWD python codes/data/generate_data.py --random_obstacles 1 --height 10 --width 10'

- To analyze and save combined training data for structure learning, run the following command:
`PYTHONPATH=$PWD python codes/data/analyze_data.py --game_type trigger_non_markov_flip --start 5 --stop 75`

- To run the structure learning non-linear model, use following command.
`PYTHONPATH=$PWD python codes/data/test_notears_nonlinear.py --game_type trigger_non_markov_flip --l1 0.01 --l2 0.01 --rho 1.0 --mode eval`

- To train/eval DQN algorithm, use the following command.
`PYTHONPATH=$PWD python codes/models/rl_approaches/DQN_main.py --height 10 --width 10 --render 0 --game_type trigger_non_markov --mode train --gamma 0.99`

- To train/eval Causal + DQN algorithm, use the following command.
`PYTHONPATH=$PWD python ./codes/models/rl_approaches/DQN_main.py --height 10 --width 10 --render 0 --game_type trigger_non_markov --mode train --num-trials 10 --gamma 0.99 --use_causal_model  --causal_update 3000 --stop_causal_update 8000 --H 100 --max-episode-length 1000 --K 5 --mbmf`

- To generate plots for performance of DQN and Causal Algorithm use the following command.
`PYTHONPATH=$PWD python ./codes/models/rl_approaches/DQN_main.py --height 10 --width 10 --render 0 --game_type trigger_non_markov --mode eval --num-trials 10 --gamma 0.99 --use_causal_model  --causal_update 3000 --stop_causal_update 8000 --H 100 --max-episode-length 1000 --K 5 --mbmf`
