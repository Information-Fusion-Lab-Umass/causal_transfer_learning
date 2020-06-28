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
from codes.data.oo_representation import *

actions = {
0: "up",
1: "down",
2: "left",
3: "right",
}

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
parser.add_argument('--l1', default = 0.01, type = float, help = 'lambda 1: penalty for regularizer')
parser.add_argument('--l2', default = 0.01, type = float, help = 'lambda2: penalty for regularizer')
parser.add_argument('--rho', default = 1.0, type = float, help = 'rho: penalty for regularizer for acyclicity')
parser.add_argument('--use_causal_model', action = 'store_true', help='Enable causal model')
parser.add_argument('--causal_update', type=int, default=3000, help='Number of steps for updating the causal model')
parser.add_argument('--save', action = 'store_false', help='save models')
parser.add_argument('--H', type=int, default=1, help='Horizon for planning')
parser.add_argument('--K', type=int, default=1, help='Total number of candidates for random shooting')
parser.add_argument('--mbmf', action = 'store_true', help='Enable model-based and model free learning')
parser.add_argument('--stop_causal_update', type=int, default=5000, help='Number of steps till updating the causal model')

args = parser.parse_args()

data_dir = "./codes/data/rl_approaches/{}/memory/".format(args.game_type)
model_dir = "./codes/stored_models/rl_approaches/{}/models/".format(args.game_type)
plot_dir = "./codes/plots/{}/".format(args.game_type)
log_dir = "./codes/logs/{}/".format(args.game_type)
img_dir = "./codes/plots/{}/img/".format(args.game_type)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(img_dir):
    os.makedirs(img_dir)
env_id = 'TriggerGame-v0'

format = "%(asctime)s.%(msecs)03d: - %(levelname)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
datefmt="%H:%M:%S", handlers=[logging.FileHandler("{}/dqn_training.log".format(log_dir), "w+")])

logger = logging.getLogger(__name__)

indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18]


if args.env == "source":
    invert = False

if args.env == "target":
    invert = True

heights = np.arange(5,80, 5)
gym.envs.register(id = env_id, entry_point = SourceEnv, max_episode_steps = 1000)


def preprocess_image(image, device):
    image = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR).reshape(40,40,3).transpose(2,0,1)
    return torch.from_numpy(image).type(torch.FloatTensor).to(device)

def select_action(policy_net, state, args, eval_mode = False):
    # linear decay
    global steps_done
    sample = random.random()
    if eval_mode == False:
        eps_threshold = min(1.0, max(0.05, args.EPS_START - (steps_done - args.burning)/(args.EPS_DECAY - args.burning)))
        steps_done += 1
    else:
        eps_threshold = 0.05
    if sample > eps_threshold:
        with torch.no_grad():
            result  = policy_net(state)
            action = result.max(1)[1].view(1, 1).cpu().detach().numpy()[0][0]
    else:
        action = random.randrange(args.action_space)
    return action, eps_threshold

def make_env(args):
    # height = np.random.choice(heights)
    height = 10
    if args.no_switches:
        total_switches = 0
    else:
        total_switches = 2

    total_prizes = 2
    x, start_idx = basic_maze(width = height, height = height, total_switches = total_switches, total_prizes = total_prizes, random_obstacles = args.random_obstacles)
    env = gym.make(env_id, x = copy(x), start_idx = start_idx, invert = invert, return_image = True, logger = logger)
    n_actions = env.action_space.n
    empty_positions = env.maze.objects.free.positions
    switch_positions = env.maze.objects.switch.positions
    prize_positions = env.maze.objects.prize.positions
    initial_positions = {"free": empty_positions, "switch": switch_positions, "prize": prize_positions}
    logger.info("Prize positions {} Switch positions {}".format(prize_positions, switch_positions))

    env = gym.make(env_id, x = copy(x), start_idx = start_idx, initial_positions = initial_positions, invert = invert, return_image = False, logger = logger)
    curr_obs = env.reset()
    curr_objects = env.maze.objects
    curr_X = get_oo_repr(0, curr_objects, 0, 0, n_actions)
    curr_state = curr_X[:, indices]
    curr_state = torch.tensor(curr_state, device = args.device, dtype = torch.float32).reshape(1,-1)

    return env, curr_state, curr_X

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

env, curr_state, curr_X = make_env(args)
n_actions = env.action_space.n

def rollout_causal_models(models, start_state, env, n_actions, K, H, gamma):
    seq_actions = np.zeros((K,H), dtype = "int")
    rewards = np.zeros(K)
    counts = np.zeros(K, dtype = int)
    M, N = env.maze.x.shape
  
    maze = np.copy(env.maze.x)
    store_state = load_state(env, maze) 
    for k in range(K):
        # print("========== candidate {}".format(k))
        env.reset_state(store_state)
        seq_actions[k] = np.random.choice(np.arange(n_actions), size = H)
        rewards[k], count = execute_actions(models, start_state, store_state, seq_actions[k], gamma, env)
        counts[k] = count
    env.reset_state(store_state)
    idx = np.argmax(rewards)
    return seq_actions[idx][0:1]

# # plt.imshow(state.detach().numpy().transpose(1,2,0))
# # plt.show()
# # plt.close()
#
# screen_height = state.shape[1]
# screen_width = state.shape[2]
# input_channels = args.history_length * state.shape[0]
#
# args.w = screen_width
# args.h = screen_height
# args.input_channels = input_channels

############################ Training #########################################################

args.input_size = len(indices)
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
        inp = []
        if args.total_steps >= args.burning:
            loss_vector = np.zeros(args.total_steps - args.burning + 1)
        reward_vector = []
        # Collect random data for initial burning period of 5000
        steps_done = 0
        total_rewards = 0
        models = None
        # state = preprocess_image(env.reset(), args.device)
        curr_objects = env.maze.objects
        TARGET_UPDATE = args.TARGET_UPDATE
        count = 0
        discount_factor = 1
        total_episodes = 0
        pbar = tqdm(total = args.total_steps+1)
        while steps_done < args.total_steps:
            if not args.mbmf or steps_done <= args.causal_update:
                action, eps_threshold = select_action(policy_net, curr_state, args)
                action_sequence = [action]
                pbar.update(1)
                count = count + 1

            if args.mbmf and steps_done > args.causal_update and steps_done < args.stop_causal_update:
                if models is None:
                    models = causal_model(inp, args.l1, args.l2, args.rho, model_dir, n_actions, train_frac = 90)
                state = {"curr_oo" : curr_X, "next_oo" : curr_X}
                action_sequence = rollout_causal_models(models, state, env, n_actions, args.K, args.H, args.gamma)
                steps_done = steps_done + len(action_sequence)
                count = count + len(action_sequence)
                pbar.update(len(action_sequence))

                if steps_done == args.stop_causal_update:
                    args.mbmf = False
                    args.EPS_START = 0.10
                # print(env.maze.x, env.maze.objects.agent.positions)
            N = len(action_sequence)
            
            
            for seq in range(N):
                action = action_sequence[seq]
                next_obs, reward, done, info = env.step(action)
                # if args.mbmf and steps_done > args.causal_update:
                    # print("============ {}/{}========== {} {}".format(seq, N, steps_done, done))
                    
                next_objects = env.maze.objects
                next_X = get_oo_repr(count, next_objects, action, reward,n_actions)
                next_state = next_X[:,indices]
                model_X = model_based_X(curr_X[:,1:], next_X[:,1:])
                inp.append(model_X)
                total_rewards = total_rewards + discount_factor*reward
                discount_factor = discount_factor * args.gamma
                next_state = torch.tensor(next_state, device = args.device, dtype = torch.float32).reshape(1,-1)
                reward = torch.tensor(reward, device = args.device, dtype = torch.float32).reshape(1,1)
                action = torch.tensor(action, device = args.device, dtype = torch.long).reshape(1,1)
                # Store the transition in memory

                if steps_done >= args.start_learn_thresh:
                    loss = optimize_model(optimizer, policy_net, target_net, memory, args.batch_size, args.device, args.gamma)
                    # loss_vector[steps_done - args.burning] = loss.item()

                if args.use_causal_model and steps_done >= args.start_learn_thresh and steps_done % args.causal_update == 0:
                    models = causal_model(inp, args.l1, args.l2, args.rho, model_dir, n_actions, train_frac = 90)
                    
                    #logger.info("Step {} average Q-learning loss {:.4f}".format(steps_done, loss.item()))
                if done:
                    #logger.info("Winning Reward {}".format(reward))
                    logger.info("Won the game: count {} steps_done {} episodes {} rewards {:.2f} eps_threshold {:.2f}".format(count, steps_done, total_episodes, total_rewards, eps_threshold))
                    memory.push(curr_state, action, reward, next_state)
                    env, curr_state, curr_X = make_env(args)
                    curr_objects = env.maze.objects
                    reward_vector.append(total_rewards)
                    count = 0
                    total_rewards = 0
                    discount_factor = 1
                    total_episodes = total_episodes + 1
                else:
                    memory.push(curr_state, action, reward, next_state)
                    if count >= args.max_episode_length:
                        logger.info("Terminating episode: count {} steps_done {} episodes {} rewards {:.2f} eps_threshold {:.2f}".format(count, steps_done, total_episodes, total_rewards, eps_threshold))
                        env, curr_state, curr_X = make_env(args)
                        curr_objects = env.maze.objects
                        reward_vector.append(total_rewards)
                        count = 0
                        total_rewards = 0
                        discount_factor = 1
                        total_episodes = total_episodes + 1
                    else:
                        curr_objects = next_objects
                        curr_state = next_state
                        curr_X = next_X
                if steps_done % TARGET_UPDATE == 0 and steps_done >= TARGET_UPDATE:
                    target_net.load_state_dict(policy_net.state_dict())
                
                if total_episodes >= 6000:
                    break
        pbar.close()

        if args.save:
            torch.save(policy_net.state_dict(), model_dir + "policy_net_DQN_{}_{}".format(args.gamma, args.use_causal_model))
            torch.save(target_net.state_dict(), model_dir + "target_net_DQN_{}_{}".format(args.gamma, args.use_causal_model))
        all_rewards.append(reward_vector)
    rewards_array = np.array(all_rewards)
    if args.save:
        np.savez(plot_dir + "dqn_train_rewards_{}.npz".format(args.gamma), r = rewards_array)



if args.mode in ["eval", "both"]:
    vec = np.load(plot_dir + "dqn_train_rewards_{}_{}.npz".format(args.gamma, args.use_causal_model), allow_pickle = True)['r']
    min_len = 100000
    for i in range(vec.shape[0]):
        print(len(vec[i]))
        if len(vec[i]) < min_len:
            min_len = len(vec[i])
        

    # min_len = 2000
    result = np.zeros((vec.shape[0], min_len))
    for i in range(vec.shape[0]):
        for j in range(min_len):
            result[i,j] = vec[i][j]

    plot_rewards(result, plot_dir + "dqn_train_rewards_{}_{}.pdf".format(args.gamma, args.use_causal_model), args.gamma, std_error = True)

    policy_net = DQN(args).to(device=args.device)
    policy_net.load_state_dict(torch.load(model_dir + "target_net_DQN_{}_{}".format(args.gamma, args.use_causal_model), map_location = torch.device(args.device)))
    rewards = np.zeros((args.num_trials, args.num_episodes))
    for trial in tqdm(range(args.num_trials)):
        for i_episode in tqdm(range(args.num_episodes)):
            count = 0
            env, curr_state, curr_X = make_env(args)
            curr_objects = env.maze.objects
            total_rewards = 0
            discount_factor = 1
            for step in range(args.max_episode_length):
                action, eps_threshold = select_action(policy_net, curr_state, args, eval_mode = True)
                next_obs, reward, done, info = env.step(action)
                next_objects = env.maze.objects
                next_X = get_oo_repr(count, next_objects, action, reward,n_actions)
                next_state = next_X[:,indices]
                next_state = torch.tensor(next_state, device = args.device, dtype = torch.float32).reshape(1,-1)
                count = count + 1
                total_rewards = total_rewards + discount_factor* reward
                discount_factor = discount_factor * args.gamma
                curr_state = next_state
                if args.render:
                    env.render()
                    time.sleep(0.1)
                if done:
                    # print("Won the game in {} steps {}. Resetting the game!".format(step, total_rewards))
                    break
            rewards[trial, i_episode] = total_rewards
            
            logger.info("Trial {} Episode {} rewards {}".format(trial, i_episode, rewards[trial, i_episode]))