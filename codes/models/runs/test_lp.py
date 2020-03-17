from mazelab.generators import basic_maze
from mazelab.envs import SourceEnv
import matplotlib.pyplot as plt
import gym
import time
import argparse
from oo_dynamics import *
# from q_learning import *

parser = argparse.ArgumentParser("Arguments for environment generation for causal concept understanding")

parser.add_argument('--env', default="source", help='type of environment')
parser.add_argument('--n_episodes', default=300, type = int, metavar = "N", help='number of episodes')
parser.add_argument('--n_len', default=500, type = int, metavar = "N", help='number of steps in each episode')

args = parser.parse_args()


def main():
    switch_positions = [[2,2], [4,6]]
    prize_positions = [[7,6],[5,5]]
    # print(prize_positions)
    x = basic_maze(width=10, height = 10, switch_positions = switch_positions, prize_positions = prize_positions)
    start_idx = [[7, 2]]
    env_id = 'SourceMaze-v0'

    gym.envs.register(id = env_id, entry_point = SourceEnv, max_episode_steps = 1000)

    if args.env == "source":
        invert = False

    if args.env == "target":
        invert = True

    # q_learning(height = 10, width = 10, n_episodes = args.n_episodes,
    # n_len = args.n_len, invert = invert )

    oo_dynamics(height = 10, width = 10, n_episodes = args.n_episodes, n_len = args.n_len, invert = invert)

if __name__ == "__main__":
    main()
