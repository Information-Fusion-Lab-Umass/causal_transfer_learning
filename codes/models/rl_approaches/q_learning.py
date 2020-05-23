import numpy as np
import time
import gym
from mazelab.envs import SourceEnv
from mazelab.generators import basic_maze
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("Arguments for q-learning algorithm for basic maze game")

parser.add_argument('--env', default="source", help='type of environment')
parser.add_argument('--random_obstacles', default= 0, type = int, help='flag to generate random obstacles')
parser.add_argument('--width', default= 10, type = int, help='width of the grid')
parser.add_argument('--height', default= 10, type = int, help='height of the grid')
parser.add_argument('--render', default = 0, choices = [1, 0], type = int, help = "Type of game")
parser.add_argument('--n_episodes', default = 10, type = int, help = "Number of episodes")
parser.add_argument('--n_len', default = 100, type = int, help = "Length of each episode")

# parser.add_argument('--game_type', default = "bw", choices = ["bw", "all_random_invert", "all_random"], help = "Type of game", required = True)

args = parser.parse_args()

def e_greedy(arr, epsilon = 0.1):
    delta = np.random.uniform(0, 1)
    if delta <= (1 - epsilon):
        return np.argmax(arr)
    else:
        return np.random.choice(np.arange(len(arr)))

def q_learning(x, start_idx, env_id, height, width, n_episodes, n_len,
               alpha = 0.1, gamma = 0.9, invert = False, render = False ):

    env = gym.make(env_id, x = x, start_idx = start_idx, invert = invert)
    empty_positions = env.maze.objects.free.positions
    switch_positions = env.maze.objects.switch.positions
    prize_positions = env.maze.objects.prize.positions
    initial_positions = {"free": empty_positions, "switch": switch_positions, "prize": prize_positions}


    n_actions = env.action_space.n
    table = np.zeros([height, width, n_actions], dtype = np.float32)
    rewards = np.zeros(n_episodes)
    epsilon = 1.0
    for i in range(n_episodes):
        # if i % 5 == 0:
        #     epsilon = epsilon * 0.75
        #     print("Epsilon for episode {}".format(epsilon))
        x, start_idx, env_id = initialize_x()
        env = gym.make(env_id, x = x, start_idx = start_idx, initial_positions = initial_positions, invert = invert)
        current_obs = env.reset()
        pos = np.stack(np.where(current_obs == env.maze.objects.agent.value), axis = 1)[0]
        for j in range(n_len):
            action = e_greedy(table[pos[0],pos[1],:], epsilon = epsilon)

            next_obs, reward, done, info = env.step(action)
            rewards[i] = rewards[i] + reward
            next_pos = np.stack(np.where(next_obs == env.maze.objects.agent.value), axis = 1)[0]

            best_next_action = np.argmax(table[next_pos[0],next_pos[1]])
            td_target = reward + gamma * (table[next_pos[0], next_pos[1], best_next_action])
            td_delta = td_target - table[pos[0], pos[1], action]
            table[pos[0],pos[1],action] = table[pos[0],pos[1],action] + alpha * (td_delta)

            pos = next_pos
            print(render)
            if render == True:
                env.render()
                time.sleep(0.05)
            # obs = env.render('rgb_array')
            # plt.imshow(obs)
            # plt.axis('off')
            # if invert:
            #     plt.savefig("../../object_tracking/images/target_env/image_{}.png".format(j))
            # else:
            #     plt.savefig("../../object_tracking/images/source_env/image_{}.png".format(j))

            time.sleep(0.05)
            if done:
                print("Won the game in {} steps.Terminating episode!".format(i))
                break
        print("Total rewards at the end of the episode {}".format(rewards[i]))
        env.close()

if __name__ == "main":
    if args.env == "source":
        invert = False

    if args.env == "target":
        invert = True

    switch_positions = [[2,2], [4,6]]
    prize_positions = [[8,6],[5,5]]
    x = basic_maze(width = width, height = height, switch_positions = switch_positions, prize_positions = prize_positions, random_obstacles = args.random_obstacles)
    start_idx = [[8, 1]]
    env_id = 'SourceMaze-v0'
    q_learning(x, start_idx, env_id, args.height, args.width, args.n_episodes, args.n_len,
                   alpha = 0.1, gamma = 0.9, invert = invert, render = args.render)
