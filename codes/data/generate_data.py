import argparse
import gym
from mazelab.generators import basic_maze
from mazelab.envs import SourceEnv
import time
import numpy as np
from oo_representation import *
import os
parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")

parser.add_argument('--env', default="source", help='type of environment')
parser.add_argument('--random_obstacles', default= 0, type = int, help='flag to generate random obstacles')
parser.add_argument('--width', default= 10, type = int, help='width of the grid')
parser.add_argument('--height', default= 10, type = int, help='height of the grid')
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random", "trigger_non_markov"], help = "Type of game", required = True)
parser.add_argument('--render', default = 0, choices = [1, 0], type = int, help = "Type of game")
parser.add_argument('--mode', default = "eval", choices = ['train', 'eval', 'both'], help ='Train or Evaluate')
parser.add_argument('--n_switches', default = 2, help ='Number of switches')
parser.add_argument('--n_prizes', default = 2, help ='Number of prizes')
args = parser.parse_args()

data_dir = "./codes/data/mat/{}/matrices/".format(args.game_type)

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
    inp = None
    colors = []

    n_pos = len(empty_positions)
    for p in empty_positions:
        start_idx = [p]
        env = gym.make(env_id, x = x, start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = False)
        for j in range(n_actions):
            curr_obs = env.reset()
            curr_objects = env.maze.objects
            if len(colors) == 0:
                for o in env.maze.objects:
                    if len(o.positions) != 0:
                        colors.append(o.colorname)
                colors = np.unique(np.array(colors))
                n_colors = len(colors)
            action = j
            X, colors_dict = get_oo_repr(count, curr_objects, action, n_colors, n_actions)
            if inp is None:
                N, M = X.shape
                inp = np.zeros((8*n_pos, N, M))
            inp[count, :, :] = X
            count = count + 1
            # next state
            next_obs, reward, done, info = env.step(action)
            next_objects = env.maze.objects
            # rewards

            if args.render == 1:
                env.render('human')
                time.sleep(0.1)
            X, colors_dict = get_oo_repr(count, next_objects, action, n_colors, n_actions)
            inp[count, :, :] = X
            count = count + 1
            if done:
                break
        env.close()
        print(inp.shape)
        print(inp)
    np.savez(data_dir + "oo_transition_matrix_{}.npz".format(args.height), mat = inp, c_dict = [colors_dict])

if __name__ == "__main__":
    main()
