import argparse
import gym
from mazelab.generators import basic_maze
from mazelab.envs import SourceEnv
import time
import numpy as np
from oo_representation import *

parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")

parser.add_argument('--env', default="source", help='type of environment')
parser.add_argument('--n_episodes', default=300, type = int,  help='number of episodes')
parser.add_argument('--n_len', default=500, type = int, help='number of steps in each episode')
parser.add_argument('--random_obstacles', default=0, type = int, help='flag to generate random obstacles')

args = parser.parse_args()

def main():
    # switch_positions = [[2,2], [4,6]]
    # prize_positions = [[7,6],[5,5]]
    switch_positions = []
    prize_positions = []
    x = basic_maze(width=10, height = 10, switch_positions = switch_positions, prize_positions = prize_positions, random_obstacles = args.random_obstacles)
    start_idx = [[7, 2]]
    env_id = 'SourceMaze-v0'

    gym.envs.register(id = env_id, entry_point = SourceEnv, max_episode_steps = 1000)

    if args.env == "source":
        invert = False

    if args.env == "target":
        invert = True

    env = gym.make(env_id, x = x, start_idx = start_idx, invert = invert, return_image = True)
    n_actions = env.action_space.n
    epsilon = 1.0
    empty_positions = env.maze.objects.free.positions
    count = 0
    inp = None
    n_colors = len(env.maze.objects)
    for p in empty_positions:
        start_idx = [p]
        env = gym.make(env_id, x = x, start_idx = start_idx, free_positions = empty_positions, invert = invert)
        for j in range(n_actions):
            curr_obs = env.reset()
            curr_objects = env.maze.objects
            env.render()
            # time.sleep(0.1)
            action = j
            X, colors_dict = get_oo_repr(count, curr_objects, action, n_colors, n_actions)
            if inp is None:
                N, M = X.shape
                inp = np.zeros((480 + 32, N, M))
            inp[count, :, :] = X
            count = count + 1
            # next state
            next_obs, reward, done, info = env.step(action)
            next_objects = env.maze.objects
            # rewards
            env.render()
            X, colors_dict = get_oo_repr(count, next_objects, action, n_colors, n_actions)
            inp[count, :, :] = X
            count = count + 1
            if done:
                break
        env.close()
    np.savez("./mat/oo_transition_matrix.npz", mat = inp, c_dict = [colors_dict])

if __name__ == "__main__":
    main()
