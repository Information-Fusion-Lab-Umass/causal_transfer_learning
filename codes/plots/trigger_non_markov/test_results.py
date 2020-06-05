import numpy as np
train_rewards = np.load("./codes/plots/trigger_non_markov/dqn_train_rewards.npz")["r"]

print(train_rewards.shape)
