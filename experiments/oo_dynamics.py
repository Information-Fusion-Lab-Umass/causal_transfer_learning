import numpy as np
import time
import gym
from mazelab.envs import SourceEnv
from mazelab.generators import basic_maze
import matplotlib.pyplot as plt
# from object_tracker.object_tracker_model import ObjectTracker
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict
import webcolors
import time

def initialize_x():
    switch_positions = [[2,2], [4,6]]
    prize_positions = [[7,6],[5,5]]
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


def split_binary(array):
    return np.asarray(list(map(lambda x: list(x), array)), dtype = 'int')

# Get neighboring objects for only changing objects (yellow, green, and red ) because neighbors are fixed for those objects
def get_neighboring_objects(X):

    Y = []
    Z = []
    idx = 5
    for i in range(X.shape[0]):
        color = X[i,1].decode("utf-8")
        if color not in ["black", "white"]:
            features = []
            f = []
            features.append(X[i,idx:])
            f.append(X[i,:idx])
            for j in range(X.shape[0]):
                if i != j:
                    x_diff = abs(int(X[i,2]) - int(X[j,2]))
                    y_diff = abs(int(X[i,3]) - int(X[j,3]))
                    if ((x_diff == 1 and  y_diff == 0) or  (x_diff == 0 and y_diff == 1)):
                        features.append(X[j,idx+4:])
                        f.append(X[j,:idx-1])
            flat_list = []
            ff = []
            for sublist in features:
                for item in sublist:
                    flat_list.append(item)
            for sublist in f:
                for item in sublist:
                    ff.append(item)
            Y.append(flat_list)
            Z.append(ff)
    Y = np.asarray(Y).astype("float32")

    return Y, np.asarray(Z)

def binarize_object_data(objects, t, a):
    A = []
    for o in objects:
        for p1, p2 in o.positions:
                A.append([o.colorname, p1, p2])

    A = np.asarray(A)
    colors = A[:,0]
    uniq_colors, colors_int = np.unique(colors, return_inverse=True)
    n_colors = 5
    n_actions = 4
    X = np.zeros((A.shape[0], 13 + n_colors + n_actions), dtype=np.dtype('a16'))
    bin_v = np.vectorize(binarize)
    x_pos = split_binary(bin_v(A[:,1].astype('int')))
    y_pos = split_binary(bin_v(A[:,2].astype('int')))

    X[:, 0] = np.ones(A.shape[0])* t
    X[:, 1] = colors
    X[:, 2] = A[:, 1]
    X[:, 3] = A[:, 2]

    switcher = {
    0: "up",
    1: "down",
    2: "left",
    3: "right"
    }

    X[:, 4] = switcher[a]
    X[:, 5:n_actions+5] = one_hot(np.ones(A.shape[0], dtype = "int")* int(a), n_actions)
    X[:, n_actions + 5: n_actions + n_colors+5] = one_hot(colors_int, n_colors)
    X[:, n_actions + n_colors+ 5: n_actions + n_colors + 9] = x_pos
    X[:, n_actions + n_colors + 9: n_actions + n_colors + 13] = y_pos
    nbrs = get_neighboring_objects(X)
    return nbrs

def binarize_actions(action):
    a = [action]
    num_classes = 4
    return one_hot(a, num_classes)

def binarize_rewards(reward):
    r = [reward]
    num_rewards = 3
    return one_hot(r, num_rewards)

def binarize(x):
    st = "{0:04b}".format(x)
    return st

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def oo_dynamics(height, width, n_episodes, n_len,
               alpha = 0.1, gamma = 0.9, invert = False ):
    x, start_idx, env_id = initialize_x()
    env = gym.make(env_id, x = x, start_idx = start_idx, invert = invert, return_image = True)
    n_actions = env.action_space.n
    table = np.zeros([height, width, n_actions], dtype = np.float32)
    rewards = np.zeros(n_episodes)
    epsilon = 1.0
    for i in range(n_episodes):
        # ot_model = ObjectTracker(maxDisappeared = 1)
        x, start_idx, env_id = initialize_x()
        env = gym.make(env_id, x = x, start_idx = start_idx, invert = invert)
        current_obs = env.reset()
        # current state
        pos = np.stack(np.where(current_obs == env.maze.objects.agent.value), axis = 1)[0]
        curr_objects = env.maze.objects
        # curr_objects = ot_model.track(env.render('rgb_array'))
        # print(neighbors(curr_objects, 37))
        inp = None
        for j in range(n_len):
            env.render()
            # action
            # print(env.motions)
            action = env.action_space.sample()
            X, Y = binarize_object_data(curr_objects, j, action)
            if inp is None:
                N, M = X.shape
                A, B =Y.shape
                inp = np.zeros((n_len, N, M))
                inp_e = np.zeros((n_len, A, B), dtype=np.dtype('a16'))
            inp[j, :, :] = X
            inp_e[j, :, :] = Y
            # next state
            next_obs, reward, done, info = env.step(action)
            next_pos = np.stack(np.where(next_obs == env.maze.objects.agent.value), axis = 1)[0]
            # next_objects = ot_model.track(env.render('rgb_array'))
            next_objects = env.maze.objects


            # rewards
            rewards[i] = rewards[i] + reward
            # transition to next state
            curr_obs = next_obs
            pos = next_pos
            curr_objects = next_objects
            # print(curr_objects)


            if done:
                print("Won the game in {} steps.Terminating episode!".format(i))
                break
        print("Total rewards at the end of the episode {}".format(rewards[i]))
        # print(inp, inp.shape)
        env.close()
    np.savez("transition_matrix.npz", mat = inp)
    np.savez("transition_matrix_eng.npz", mat = inp_e)
