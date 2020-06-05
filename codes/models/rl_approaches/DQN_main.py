import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from collections import deque
import argparse
from copy import copy
import os
from mazelab.envs import SourceEnv
from mazelab.generators import basic_maze
from codes.models.rl_approaches.DQN import *
import gym
# from utils import *
from codes.models.rl_approaches.memory import *
import time
import cv2
import math
from model_utils import *
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Arguments
parser = argparse.ArgumentParser(description="causality")
parser.add_argument('--env', default="source", help='type of environment')
parser.add_argument('--random_obstacles', default= 0, type = int, help='flag to generate random obstacles')
parser.add_argument('--width', default= 10, type = int, help='width of the grid')
parser.add_argument('--height', default= 10, type = int, help='height of the grid')
parser.add_argument('--render', default = 0, choices = [1, 0], type = int, help = "Type of game")
parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random", "trigger_markov", "trigger_non_markov"], help = "Type of game", required = True)
parser.add_argument('--mode', default = "eval", choices = ['train', 'eval', 'both'], help ='Train or Evaluate', required = True)
parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--max-episode-length', type=int, default=int(1000), help='Maximum episode length for each trial')
parser.add_argument('--history-length', type=int, default= 1, help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=64, help='size of hidden layer')
parser.add_argument('--EPS-START', type=float, default=1.0, help='Exploration constant for start')
parser.add_argument('--EPS-END', type=float, default=0.05, help='Exploration constant for end')
parser.add_argument('--EPS-DECAY', type=int, default=250000, help='Number of steps for epsilon decay')
parser.add_argument('--TARGET-UPDATE', type=int, default=2000, help='Number of steps for updating the target net')
parser.add_argument('--memory-size', type=int, default=int(10e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--batch-size', type=int, default=256, metavar='SIZE', help='Batch size')
parser.add_argument('--num-episodes', type=int, default=100, metavar='N', help='Number of episodes to run for collecting data')
parser.add_argument('--num-trials', type=int, default=100, help='Number of trials for training data')
parser.add_argument('--action-space', type=int, default=4, help='Number of default actions for the game: UP, DOWN, RIGHT, LEFT')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--gamma', type=float, default=1.0, help='Discount Factor')
parser.add_argument('--total-steps', type=int, default=250000, help='Total number of steps for training')
parser.add_argument('--burning', type=int, default=3000, help='Burning number of steps for which random policy follows')
parser.add_argument('--start-learn-thresh', type=int, default=1000, help='Learning steps after which model learns')
parser.add_argument('--no_switches', action = 'store_true', help='Disable switches')

args = parser.parse_args()

data_dir = "./codes/data/rl_approaches/{}/memory/".format(args.game_type)
model_dir = "./codes/stored_models/rl_approaches/{}/models/".format(args.game_type)
plot_dir = "./codes/plots/{}/".format(args.game_type)
log_dir = "./codes/logs/{}/".format(args.game_type)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

format = "%(asctime)s.%(msecs)03d: - %(levelname)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
datefmt="%H:%M:%S", handlers=[logging.FileHandler("{}/dqn_training.log".format(log_dir), "w+")])

logger = logging.getLogger(__name__)


if args.env == "source":
    invert = False

if args.env == "target":
    invert = True


def preprocess_image(image, device):
    image = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR).reshape(40,40,3).transpose(2,0,1)
    return torch.from_numpy(image).type(torch.FloatTensor).to(device)

def select_action(policy_net, state, args, eval_mode = False):
    # linear decay
    global steps_done
    sample = random.random()
    if eval_mode == False:
        eps_threshold = min(1.0, max(0.05, 1 - (steps_done - args.burning)/(args.EPS_DECAY - args.burning)))
        # eps_threshold = max(args.EPS_END,  min(1.0, args.EPS_START + ((steps_done - args.burning)/args.EPS_DECAY) * (args.EPS_END - args.EPS_START )))
        #logger.info("Steps done {}, Epsilon {}".format(steps_done, eps_threshold))
        steps_done += 1
    else:
        eps_threshold = 0.001
    if sample > eps_threshold:
        with torch.no_grad():
            result  = policy_net(state.reshape(1, args.input_channels, args.h, args.w))
            action = result.max(1)[1].view(1, 1).cpu().detach().numpy()[0][0]
    else:
        action = random.randrange(args.action_space)
    return action, eps_threshold

args = parser.parse_args()
logger.info(' ' * 26 + 'Options')
for k, v in vars(args).items():
  logger.info(' ' * 26 + k + ': ' + str(v))

logger.info("cuda available {} device count {}".format(torch.cuda.is_available(), torch.cuda.device_count()))
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
else:
     args.device = torch.device('cpu')


logger.info("Running on {}".format(args.device))

# Env
if args.no_switches:
    switch_positions = []
else:
    switch_positions = [[7,1],[3,2]]
prize_positions = [[2,2],[8,6]]
logger.info("Switch Positions {}, Prize positions {}".format(switch_positions, prize_positions))

x = basic_maze(width = args.width, height = args.height, switch_positions = switch_positions, prize_positions = prize_positions, random_obstacles = args.random_obstacles)
start_idx = [[8, 1]]
env_id = 'TriggerGame-v0'
gym.envs.register(id = env_id, entry_point = SourceEnv, max_episode_steps = 1000)


# max_length = 1
# history_length = 4
env = gym.make(env_id, x = copy(x), start_idx = start_idx, invert = invert, return_image = True, logger = logger)
empty_positions = env.maze.objects.free.positions
switch_positions = env.maze.objects.switch.positions
prize_positions = env.maze.objects.prize.positions
initial_positions = {"free": empty_positions, "switch": switch_positions, "prize": prize_positions}

env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = True, logger = logger)
state = preprocess_image(env.reset(), args.device)

# plt.imshow(state.detach().numpy().transpose(1,2,0))
# plt.show()
# plt.close()

screen_height = state.shape[1]
screen_width = state.shape[2]
input_channels = args.history_length * state.shape[0]

args.w = screen_width
args.h = screen_height
args.input_channels = input_channels

############################ Training #########################################################
all_rewards = []
if args.mode in ["train", "both"]:
    for trial in tqdm(range(args.num_trials)):
        logger.info("trial {}".format(trial))
        policy_net = DQN(args).to(device=args.device)
        target_net = DQN(args).to(device = args.device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.RMSprop(policy_net.parameters(), lr = args.lr)
        memory = ReplayMemory(args.memory_size)
        loss_vector = np.zeros(args.total_steps - args.burning + 1)
        reward_vector = []

        # Collect random data for initial burning period of 5000
        steps_done = 0
        total_rewards = 0
        state = preprocess_image(env.reset(), args.device)
        TARGET_UPDATE = args.TARGET_UPDATE
        count = 0
        discount_factor = 1
        for i_episode in tqdm(range(args.total_steps)):
            count = count + 1
            action, eps_threshold = select_action(policy_net, state, args)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess_image(next_state, args.device)
            total_rewards = total_rewards + discount_factor*reward
            discount_factor = discount_factor * args.gamma
            reward = torch.tensor(reward, device = args.device, dtype = torch.float32).reshape(1,1)
            action = torch.tensor(action, device = args.device, dtype = torch.long).reshape(1,1)
            # Store the transition in memory

            if steps_done >= args.start_learn_thresh:
                loss = optimize_model(optimizer, policy_net, target_net, memory, args.batch_size, args.device, args.h, args.w, args.input_channels, args.gamma)
                loss_vector[steps_done - args.burning] = loss.item()
                #logger.info("Step {} average Q-learning loss {:.4f}".format(steps_done, loss.item()))
            if done:
                #logger.info("Winning Reward {}".format(reward))
                logger.info("Won the game: count {} steps_done {} rewards {:.2f} eps_threshold {:.2f}".format(count, steps_done, total_rewards, eps_threshold))
                memory.push(state, action, reward, None)
                env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = True, logger = logger)
                state = preprocess_image(env.reset(), args.device)
                reward_vector.append(total_rewards)
                count = 0
                total_rewards = 0
                discount_factor = 1
            else:
                memory.push(state, action, reward, next_state)
                if count >= args.max_episode_length:
                    logger.info("Terminating episode: count {} steps_done {} rewards {:.2f} eps_threshold {:.2f}".format(count, steps_done, total_rewards, eps_threshold))
                    env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = True, logger = logger)
                    state = preprocess_image(env.reset(), args.device)
                    reward_vector.append(total_rewards)
                    count = 0
                    total_rewards = 0
                    discount_factor = 1
                else:
                    state = next_state
            if i_episode % TARGET_UPDATE == 0 and steps_done >= TARGET_UPDATE:
                target_net.load_state_dict(policy_net.state_dict())

        # torch.save(memory, data_dir + "replay_buffer")
        torch.save(policy_net.state_dict(), model_dir + "policy_net_DQN")
        torch.save(target_net.state_dict(), model_dir + "target_net_DQN")
        all_rewards.append(reward_vector)
    rewards_array = np.array(all_rewards)
    np.savez(plot_dir + "dqn_train_rewards.npz", r = rewards_array)



if args.mode in ["eval", "both"]:
    # memory = torch.load(data_dir + "replay_buffer")
    policy_net = DQN(args).to(device=args.device)
    policy_net.load_state_dict(torch.load(model_dir + "policy_net_DQN"))
    train_reward_vec = np.load(plot_dir + "dqn_loss_rewards.npz")['r'].reshape(1,-1)
    loss_vec = np.load(plot_dir + "dqn_loss_rewards.npz")['l'].reshape(1,-1)
    rewards = np.zeros((args.num_trials, args.num_episodes))
    for trial in tqdm(range(args.num_trials)):
        for i_episode in tqdm(range(args.num_episodes)):
            env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = True, logger = logger)
            state = preprocess_image(env.reset(), args.device)
            total_rewards = 0
            discount_factor = 1
            state = preprocess_image(env.reset(), args.device)
            for step in range(args.max_episode_length):
                action, eps_threshold = select_action(policy_net, state, args, eval_mode = True)
                next_state, reward, done, info = env.step(action)
                total_rewards = total_rewards + discount_factor* reward
                discount_factor = discount_factor * args.gamma
                if done:
                    logger.info("Won the game in {} steps. Resetting the game!".format(step))
                    break
            rewards[trial, i_episode] = total_rewards
            logger.info("Trial {} Episode {} rewards {}".format(trial, i_episode, rewards[trial, i_episode]))
    np.savez(plot_dir + "dqn_rewards", r = rewards)
    plot_rewards(rewards, plot_dir + "test_rewards.png", std_error = True)
    # plot_rewards(train_reward_vec, plot_dir + "train_rewards.png", std_error = False)
    # plot_rewards(loss_vec, plot_dir + "train_loss.png", std_error = False)