import numpy as np
import time
import gym
from mazelab.envs import SourceEnv
from mazelab.generators import basic_maze
import matplotlib.pyplot as plt
import argparse
from copy import copy
import os
from tqdm import tqdm
from model_utils import *
import logging

parser = argparse.ArgumentParser("Arguments for q-learning algorithm for basic maze game")

parser.add_argument('--env', default="source", help='type of environment')
parser.add_argument('--random_obstacles', default= 0, type = int, help='flag to generate random obstacles')
parser.add_argument('--width', default= 10, type = int, help='width of the grid')
parser.add_argument('--height', default= 10, type = int, help='height of the grid')
parser.add_argument('--render', default = 0, choices = [1, 0], type = int, help = "Type of game")
parser.add_argument('--n_trials', default = 100, type = int, help = "Number of trials")
parser.add_argument('--n_episodes', default = 100, type = int, help = "Number of episodes")
parser.add_argument('--n_len', default = 100, type = int, help = "Maximum length of each episode")
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random", "trigger_markov"], help = "Type of game", required = True)
parser.add_argument('--mode', default = "eval", choices = ['train', 'eval', 'both'], help ='Train or Evaluate')

args = parser.parse_args()
np.set_printoptions(precision=3)
plot_dir = "./codes/plots/{}/".format(args.game_type)
model_dir = "./codes/stored_models/rl_approaches/{}/models/".format(args.game_type)
plot_dir = "./codes/plots/{}/".format(args.game_type)
log_dir = "./codes/logs/{}/".format(args.game_type)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def e_greedy(arr, epsilon = 0.1):
    delta = np.random.uniform(0, 1)
    if delta <= (1 - epsilon):
        return np.argmax(arr)
    else:
        return np.random.choice(np.arange(len(arr)))

format = "%(asctime)s.%(msecs)03d: - %(levelname)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
datefmt="%H:%M:%S", handlers=[logging.FileHandler("{}/tabular_q_learning.log".format(log_dir), "w+")])

logger = logging.getLogger(__name__)

def q_learning(x, start_idx, env_id, height, width, n_trials, n_episodes, n_len,
               alpha = 0.1, gamma = 0.99, invert = False, render = False ):

    env = gym.make(env_id, x = copy(x), start_idx = start_idx, invert = invert, return_image = False)
    empty_positions = env.maze.objects.free.positions
    switch_positions = env.maze.objects.switch.positions
    prize_positions = env.maze.objects.prize.positions
    initial_positions = {"free": empty_positions, "switch": switch_positions, "prize": prize_positions}
    n_actions = env.action_space.n
    total_len = 100 * n_episodes
    burning = 0.01 * total_len
    rewards = np.zeros((n_trials, n_episodes))
    tables = np.zeros((n_trials, height, width, n_actions))
    for k in tqdm(range(n_trials)):
        age = 0
        table = np.zeros([height, width, n_actions], dtype = np.float32)
        epsilon = 1.0
        for i in tqdm(range(n_episodes)):
            logger.info("Epsilon for episode {} age {} burning {} total_len {}".format(epsilon, age, burning, total_len))
            env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = False)
            current_obs = env.reset()
            pos = np.stack(np.where(current_obs == env.maze.objects.agent.value), axis = 1)[0]
            for j in range(n_len):
                age = age + 1
                epsilon = np.min([1.0, np.max([0.05, 1 - (age - burning)/(total_len - burning)])])
                action = e_greedy(table[pos[0],pos[1],:], epsilon = epsilon)
                next_obs, reward, done, info = env.step(action)
                rewards[k,i] = rewards[k,i] + reward
                next_pos = np.stack(np.where(next_obs == env.maze.objects.agent.value), axis = 1)[0]
                best_next_action = np.argmax(table[next_pos[0],next_pos[1]])
                td_target = reward + gamma * (table[next_pos[0], next_pos[1], best_next_action])
                td_delta = td_target - table[pos[0], pos[1], action]
                table[pos[0],pos[1],action] = table[pos[0],pos[1],action] + alpha * (td_delta)

                pos = next_pos
                if render == 1:
                    env.render()
                    time.sleep(0.05)
                if done:
                    logger.info("Won the game in {} steps. Terminating episode!".format(j))
                    break
            logger.info("Trial {} Episode {} rewards {} age {}".format(k, i, rewards[k,i], age))
            tables[k] = table
    np.savez(plot_dir + "q_rewards_markov.npz", r = rewards, q =  tables)
    env.close()

if __name__ == '__main__':
    if args.env == "source":
        invert = False

    if args.env == "target":
        invert = True

    switch_positions = []
    prize_positions = [[8,6], [5,5]]
    x = basic_maze(width = args.width, height = args.height, switch_positions = switch_positions, prize_positions = prize_positions, random_obstacles = args.random_obstacles)
    start_idx = [[8, 1]]
    env_id = 'TriggerGame-v0'
    gym.envs.register(id = env_id, entry_point = SourceEnv, max_episode_steps = 1000)

    if args.mode in ["train", "both"]:
        q_learning(x, start_idx, env_id, args.height, args.width, args.n_trials, args.n_episodes, args.n_len,
                       alpha = 0.1, gamma = 0.99, invert = invert, render = args.render)
    if args.mode in ["eval", "both"]:
        q_rewards = np.load(plot_dir + "q_rewards_markov.npz")
        q_values = q_rewards["q"]
        rewards = q_rewards["r"]
        plot_rewards(rewards, plot_dir + "q_learning_rewards_{}G.png".format(len(prize_positions)), std_error = True)
