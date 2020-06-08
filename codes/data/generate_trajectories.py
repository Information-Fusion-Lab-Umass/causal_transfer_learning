import argparse
import gym
from mazelab.generators import basic_maze
from mazelab.envs import SourceEnv
import time
import numpy as np
from oo_representation import *
import os
from tqdm import tqdm
from copy import copy
import random
import torch
from structural_generation import *
from notears_nonlinear import *
from codes.utils import *
actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}
parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")

parser.add_argument('--env', default="source", help='type of environment')
parser.add_argument('--random_obstacles', default= 0, type = int, help='flag to generate random obstacles')
parser.add_argument('--width', default= 10, type = int, help='width of the grid')
parser.add_argument('--height', default= 10, type = int, help='height of the grid')
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random", "trigger_non_markov", "trigger_non_markov_random", "trigger_non_markov_flip"], help = "Type of game", required = True)
parser.add_argument('--render', default = 0, choices = [1, 0], type = int, help = "Type of game")
parser.add_argument('--mode', default = "eval", choices = ['train', 'eval', 'both'], help ='Train or Evaluate')
parser.add_argument('--n_switches', default = 2, help ='Number of switches')
parser.add_argument('--n_prizes', default = 2, help ='Number of prizes')
parser.add_argument('--num_episodes', type=int, default=100, metavar='N', help='Number of episodes to run for collecting data')
parser.add_argument('--num_trials', type=int, default=100, help='Number of trials for training data')
parser.add_argument('--max_episode_length', type=int, default=int(1000), help='Maximum episode length for each trial')
parser.add_argument('--l1', default = 0.01, type = float, help = 'lambda 1: penalty for regularizer')
parser.add_argument('--l2', default = 0.01, type = float, help = 'lambda2: penalty for regularizer')
parser.add_argument('--rho', default = 0.0, type = float, help = 'rho: penalty for regularizer for acyclicity')
parser.add_argument('--train_frac', default = 100, type = float, help = 'fraction of data to be trained on')

args = parser.parse_args()

actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}

vars = ['ax_t1', 'ay_t1', 'ac_t1', 'ux_t1', 'uy_t1', 'uc_t1', 'dx_t1',
         'dy_t1', 'dc_t1', 'lx_t1', 'ly_t1', 'lc_t1', 'rx_t1', 'ry_t1', 'rc_t1','r_t1', 'ns_t1']

p = [0,1,2,5,8,11,14,16]
torch.set_printoptions(precision=3, sci_mode = False)
np.set_printoptions(precision =3)

plot_dir = "./codes/plots/{}/train_{}/lambda1_{}_lambda2_{}_rho_{}/".format(args.game_type, args.train_frac, args.l1, args.l2, args.rho)
data_dir = "./codes/data/mat/{}/matrices/".format(args.game_type)
model_dir =  "./codes/data/models/{}/train_{}/".format(args.game_type, args.train_frac)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def get_models(game_type, l1, l2, rho, n_actions):
    models = {}
    for action in range(n_actions):
        # filename = data_dir + "oo_action_{}_{}.npz".format(, args.game_type)
        # f = np.load(filename, mmap_mode='r', allow_pickle=True)
        # X_all = f["mat"]
        model = NotearsMLP(dims=[19, 10, 1], bias=True)
        model_name = model_dir + "{}_l1_{:.2f}_l2_{:.2f}_rho_{:.2f}".format(actions[action], l1, l2, rho)
        model.load_state_dict(torch.load(model_name))
        models[action] = model
    return models

def get_prediction(X, models, next_pos, reward):
    X_orig = np.zeros((1,19))
    X_orig[:,:15] = X[:,:15]
    X_orig[:,16] = int(X[:,17] > 0)
    X_orig[:,17:19] = next_pos
    X_orig[:,15] = reward

    X_train = copy(X_orig)
    print(X_train)
    X_train[:, 17:19] = 0
    X_train[:,15] = 0
    action = int(X[0,15])
    Z = np.zeros_like(X_orig)
    Z[:, p] = X_train[:, p]
    X_torch = torch.from_numpy(X_train).type(torch.FloatTensor)
    Z_torch = torch.from_numpy(Z).type(torch.FloatTensor)
    X_orig_torch = torch.from_numpy(X_orig).type(torch.FloatTensor)
    train_pred = models[action](X_torch, Z_torch)

    print("Input {}".format(X_train))
    print("Orig {}".format(X_orig))
    next_pos = train_pred[0,17:19].detach().numpy()
    reward_pred = train_pred[0,15].detach().numpy()
    train_loss = squared_loss(train_pred, X_orig_torch)
    return next_pos, reward_pred, train_loss

def main():
    x, start_idx = basic_maze(args.width, args.height, args.n_switches, args.n_prizes, args.random_obstacles)
    print(x, start_idx)
    env_id = 'SourceMaze-v0'
    gym.envs.register(id = env_id, entry_point = SourceEnv, max_episode_steps = 1000)

    if args.env == "source":
        invert = False

    if args.env == "target":
        invert = True

    env = gym.make(env_id, x = x, start_idx = start_idx, invert = invert, return_image = False)
    n_actions = env.action_space.n
    epsilon = 1.0
    empty_positions = env.maze.objects.free.positions
    switch_positions = env.maze.objects.switch.positions
    prize_positions = env.maze.objects.prize.positions
    initial_positions = {"free": empty_positions, "switch": switch_positions, "prize": prize_positions}
    count = 0
    inp = []
    reward = 0
    n_pos = len(empty_positions)
    models = get_models(args.game_type, args.l1, args.l2, args.rho, n_actions)
    for i_episode in tqdm(range(args.num_episodes)):
        env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = False)
        curr_obs = env.reset()
        curr_objects = env.maze.objects
        for step in range(args.max_episode_length):
            print("################ Step {} ################".format(step))

            action = random.randrange(n_actions)
            X = get_oo_repr(count, curr_objects, action, 0, n_actions)
            # print("Orig X {}".format(X))

            # next state
            next_obs, reward, done, info = env.step(action)
            next_objects = env.maze.objects
            # rewards
            if args.render == 1:
                env.render('human')
                time.sleep(0.1)
            X = get_oo_repr(count, next_objects, action, reward, n_actions)
            orig_pos = X[0,1:3]
            next_pos, reward_pred, loss = get_prediction(X[:,1:], models, orig_pos, reward)

            print("Prediction: x_pos {} y-pos {} reward {} loss {}".format(next_pos[0], next_pos[1], reward_pred, loss.item()))
            print("Original: x_pos {} y-pos {} reward {}".format(orig_pos[0], orig_pos[1], reward))
            count = count + 1
            curr_objects = next_objects
            curr_obs = next_obs
            if done:
                break
    env.close()

if __name__ == "__main__":
    main()
