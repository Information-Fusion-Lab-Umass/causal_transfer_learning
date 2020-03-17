import numpy as np
import time
import gym
from mazelab.envs import SourceEnv
from mazelab.generators import basic_maze
import matplotlib.pyplot as plt


def initialize_x():
    switch_positions = [[2,2], [4,6]]
    prize_positions = [[8,6],[5,5]]
    x = basic_maze(width=10, height = 10, switch_positions = switch_positions, prize_positions = prize_positions)
    start_idx = [[8, 1]]
    env_id = 'SourceMaze-v0'
    return x, start_idx, env_id

def e_greedy(arr, epsilon = 0.1):
    delta = np.random.uniform(0, 1)
    if delta <= (1 - epsilon):
        return np.argmax(arr)
    else:
        return np.random.choice(np.arange(len(arr)))

def q_learning(height, width, n_episodes, n_len,
               alpha = 0.1, gamma = 0.9, invert = False ):
    x, start_idx, env_id = initialize_x()
    env = gym.make(env_id, x = x, start_idx = start_idx, invert = invert)
    n_actions = env.action_space.n
    table = np.zeros([height, width, n_actions], dtype = np.float32)
    rewards = np.zeros(n_episodes)
    epsilon = 1.0
    for i in range(n_episodes):
        # if i % 5 == 0:
        #     epsilon = epsilon * 0.75
        #     print("Epsilon for episode {}".format(epsilon))
        x, start_idx, env_id = initialize_x()
        env = gym.make(env_id, x = x, start_idx = start_idx, invert = invert)

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
            obs = env.render('rgb_array')
            plt.imshow(obs)
            plt.axis('off')
            if invert:
                plt.savefig("../../object_tracking/images/target_env/image_{}.png".format(j))
            else:
                plt.savefig("../../object_tracking/images/source_env/image_{}.png".format(j))
            time.sleep(0.05)
            if done:
                print("Won the game in {} steps.Terminating episode!".format(i))
                break
        print("Total rewards at the end of the episode {}".format(rewards[i]))
        env.close()
    np.savez("rewards.npz", r = rewards)
