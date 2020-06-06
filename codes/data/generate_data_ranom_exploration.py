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
args = parser.parse_args()

data_dir = "./codes/data/mat/{}/matrices/".format(args.game_type)
actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
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
    colors = []
    reward = 0
    n_pos = len(empty_positions)
    for i_episode in tqdm(range(args.num_episodes)):
        env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = False)
        curr_obs = env.reset()
        curr_objects = env.maze.objects
        for step in range(args.max_episode_length):
            if len(colors) == 0:
                for o in env.maze.objects:
                    if len(o.positions) != 0:
                        colors.append(o.colorname)
                colors = np.unique(np.array(colors))
                n_colors = len(colors)
            action = random.randrange(n_actions)
            X = get_oo_repr(count, curr_objects, action, 0, n_colors, n_actions)
            inp.append(X)
            # next state
            next_obs, reward, done, info = env.step(action)
            next_objects = env.maze.objects
            # rewards
            if args.render == 1:
                env.render('human')
                time.sleep(0.1)
            X = get_oo_repr(count, next_objects, action, reward, n_colors, n_actions)

            inp.append(X)
            count = count + 1
            curr_objects = next_objects
            curr_obs = next_obs
            if done:
                break
    env.close()
    inp = np.array(inp).squeeze(axis = 1)
    #['time_stamp', 'a_x', 'a_y', 'a_c', 'u', 'd', 'l', 'r', 'a', 'reward', 'num_switches']
    np.savez(data_dir + "oo_transition_matrix_{}.npz".format(args.height), mat = inp)

if __name__ == "__main__":
    main()
