from memory import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(rewards, plot_dir):
    n_trials, n_episodes = rewards.shape

    mean = np.mean(rewards, axis = 0)
    std = np.std(rewards, axis = 0)
    N = np.arange(n_episodes)
    plt.plot(N, mean)
    plt.fill_between(N, mean - std, mean + std, color='gray', alpha=0.2)
    # plt.plot(N, mean+std)
    # plt.plot(np.arange(n_episodes), np.mean(rewards, axis = 0))
    plt.title("Q-learning rewards")
    plt.xlabel("Number of episodes")
    plt.ylabel("Cumulative reward")
    plt.savefig(plot_dir + "reward_plot.png")

def optimize_model(optimizer, policy_net, target_net, memory, BATCH_SIZE, device, height, width, input_channels, GAMMA = 1.0):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).reshape(-1, input_channels, height, width)
    state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, input_channels, height, width)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    q_a = policy_net(state_batch)
    state_action_values = q_a.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_q_a  = target_net(non_final_next_states)
    next_state_values[non_final_mask] = next_q_a.max(1)[0].detach()

    next_state_values = next_state_values.reshape(BATCH_SIZE, 1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
