from memory import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from codes.data.structural_generation import *
from codes.data.notears_nonlinear import *
import argparse
import igraph as ig
import matplotlib.pyplot as plt
from codes.utils import *
import os
import torch
import pandas as pd
from copy import copy

actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}
colors_dict = {'white': 3, 'black': 0, 'green': 1, 'red': 2, 'yellow': 4}
s_vars = ['ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
         'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1',
         'a_t1', 'r_t1', 'ns_t1', 'ax_t2', 'ay_t2']
text_vars = [r'\textit{$agent.x^{t}$}', r'\textit{$agent.y^{t}$}', r'\textit{$agent.c^{t}$}', r'\textit{$up.x^{t}$}', r'\textit{$up.y^{t}$}', r'\textit{$up.c^{t}$}', r'\textit{$down.x^{t}$}',
         r'\textit{$down.y^{t}$}', r'\textit{$down.c^{t}$}', r'\textit{$left.x^{t}$}', r'\textit{$left.y^{t}$}', r'\textit{$left.c^{t}$}', r'\textit{$right.x^{t}$}', r'\textit{$right.y^{t}$}',
         r'\textit{$right.c^{t}$}', r'\textit{$reward^{t+1}$}', r'\textit{$num\_keys^{t}$}', r'\textit{$agent.x^{t+1}$}', r'\textit{$agent.y^{t+1}$}']


p = [0,1,2,5,8,11,14,16]
q = []

for j in range(len(text_vars)):
    if j not in p:
        q.append(j)

def get_input_data(X):
    Z = np.zeros(X.shape)
    Z[:,p] = X[:,p]
    X_orig = copy(X)
    X[:,15] = 0 # reward = 0
    X[:,17:19] = 0 # next_pos = 0
    q_l = [3,4,6,7,9,10,12,13]
    X[:,q_l] = 0
    # print("X {}".format(X[0]))
    # print("X_orig {}".format(X_orig[0]))
    # print("Z {}".format(Z[0]))
    return X, X_orig, Z

def plot_rewards(rewards, plot_name, std_error = False):
    n_trials, n_episodes = rewards.shape
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mean = np.mean(rewards, axis = 0)
    std = np.std(rewards, axis = 0)
    N = np.arange(n_episodes)
    plt.plot(N, mean, label = "DQN")
    if std_error == True:
    	plt.fill_between(N, mean - std, mean + std, color='gray', alpha=0.2)
    # plt.plot(N, mean+std)
    # plt.plot(np.arange(n_episodes), np.mean(rewards, axis = 0))
    # plt.title("Q-learning rewards")
    plt.xlabel("Number of episodes")
    gamma = r'$\gamma$'
    plt.ylabel("Cumulative reward ({}= 1.0)".format(gamma))
    plt.legend()
    plt.savefig(plot_name)

def optimize_model(optimizer, policy_net, target_net, memory, BATCH_SIZE, device, GAMMA = 1.0):
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
                                                if s is not None])
    state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, -1)
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
    return loss


def causal_model(inp, l1, l2, rho, model_dir, n_actions, train_frac = 90):
    models = {}
    inp = np.array(inp).squeeze(1)
    print(inp[:5])
    for i in range(n_actions):

        X_all = inp[inp[:, 15] == i]
        X_all = np.delete(X_all, 15, axis = 1)
        X_all[X_all[:,16] > 0, 16] = 1

        idx = np.arange(X_all.shape[0])
        np.random.shuffle(idx)
        train_size = int((train_frac/100) * X_all.shape[0])
        train_idx = np.random.choice(idx, size = train_size, replace = False)
        test_idx = []
        for j in range(X_all.shape[0]):
            if j not in train_idx:
                test_idx.append(j)
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]

        X_tr, X_tr_orig, Z_tr = get_input_data(X_train)
        X_te, X_te_orig, Z_te = get_input_data(X_test)

        model = NotearsMLP(dims=[X_tr.shape[1], 10, 1], bias=True)
        model_name = model_dir + "{}_l1_{:.2f}_l2_{:.2f}_rho_{:.2f}".format(actions[i], l1, l2, rho)
        if os.path.exists(model_name):
            print("Loading causal model")
            model.load_state_dict(torch.load(model_name))
        W_est = notears_nonlinear(model, X_tr, Z_tr, X_tr_orig, model_name = model_name, rho = rho, lambda1=l1, lambda2=l2)

        with torch.no_grad():
            X_tr_torch = torch.from_numpy(X_tr).type(torch.FloatTensor)
            Z_tr_torch = torch.from_numpy(Z_tr).type(torch.FloatTensor)
            X_tr_orig_torch = torch.from_numpy(X_tr_orig).type(torch.FloatTensor)
            train_pred = model(X_tr_torch, Z_tr_torch)
            train_loss = squared_loss(train_pred, X_tr_orig_torch)

            X_te_torch = torch.from_numpy(X_te).type(torch.FloatTensor)
            Z_te_torch = torch.from_numpy(Z_te).type(torch.FloatTensor)
            X_te_orig_torch = torch.from_numpy(X_te_orig).type(torch.FloatTensor)
            test_pred = model(X_te_torch, Z_te_torch)
            test_loss = squared_loss(test_pred, X_te_orig_torch)

            X_eng = analyze(X_tr_orig_torch[0].reshape(1,-1))
            print(len(text_vars), X_eng.shape, train_pred.shape)
            print("Train loss {}".format(train_loss.item()))
            print("Test loss {}".format(test_loss.item()))
            print("============== Action {} ==================".format(actions[i]))
            for j in range(X_tr_torch.shape[1]):
                print(s_vars[j], X_eng[0, j], train_pred[0, j].item())

        W = model.fc1_to_adj()
        est_plot_name = model_dir + "w_est_{}.png".format(actions[i])

        x_indices = q
        y_indices = p
        y_label = [text_vars[k] for k in p]
        x_label = [text_vars[k] for k in q]
        plot_weight_sem(W, est_plot_name, x_indices, y_indices, x_label, y_label, actions[i])

        models[actions[i]] = model

    return models
