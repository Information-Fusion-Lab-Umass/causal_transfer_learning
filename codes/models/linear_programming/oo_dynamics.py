import numpy as np
import time
import gym
from mazelab.envs import SourceEnv
from mazelab.generators import basic_maze
import matplotlib.pyplot as plt
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
    M = []
    idx = 5
    for i in range(X.shape[0]):
        color = X[i,1].decode("utf-8")
        if color not in ["black", "white", "red", "green"]:
            features = []
            f = []
            mixed = []
            features.append(X[i,idx:])
            f.append(X[i,:idx])
            indices = np.r_[2:4, idx:idx+9]
            mixed.append(X[i, indices])
            for j in range(X.shape[0]):
                if i != j:
                    x_diff = int(X[i,2]) - int(X[j,2])
                    y_diff = int(X[i,3]) - int(X[j,3])
                    if y_diff == -1 and x_diff == 0:
                        r = X[j]

                    if y_diff == 1 and x_diff == 0:
                        l = X[j]

                    if x_diff == -1 and y_diff == 0:
                        d = X[j]

                    if x_diff == 1 and y_diff == 0:
                        u = X[j]

            features.append(l[idx+4:])
            features.append(r[idx+4:])
            features.append(d[idx+4:])
            features.append(u[idx+4:])
            indices = np.r_[2:4, idx+4:idx+9]

            mixed.append(l[indices])
            mixed.append(r[indices])
            mixed.append(d[indices])
            mixed.append(u[indices])
            f.append(l[:idx-1])
            f.append(r[:idx-1])
            f.append(d[:idx-1])
            f.append(u[:idx-1])

            flat_list = []
            ff = []
            mm = []
            for sublist in features:
                for item in sublist:
                    flat_list.append(item)
            for sublist in f:
                for item in sublist:
                    ff.append(item)

            for sublist in mixed:
                for item in sublist:
                    mm.append(item)

            Y.append(flat_list)
            Z.append(ff)
            M.append(mm)
    Y = np.asarray(Y).astype("float32")
    M = np.asarray(M).astype("float32")
    return Y, np.asarray(Z), M

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
    empty_positions = env.maze.objects.free.positions
    count = 0
    inp = None
    for p in empty_positions:
        start_idx = [p]
        x, sidx, env_id = initialize_x()
        env = gym.make(env_id, x = x, start_idx = start_idx, free_positions = empty_positions, invert = invert)
        for j in range(n_actions):
            curr_obs = env.reset()
            pos = np.stack(np.where(curr_obs == env.maze.objects.agent.value), axis = 1)[0]
            curr_objects = env.maze.objects
            env.render()
            # time.sleep(0.1)
            action = j
            X, Y, MM = binarize_object_data(curr_objects, count, action)
            print(MM)
            if inp is None:
                N, M = X.shape
                A, B = Y.shape
                K, L = MM.shape
                inp = np.zeros((480, N, M))
                inp_e = np.zeros((480, A, B), dtype=np.dtype('a16'))
                inp_m = np.zeros((480,K,L))
            inp[count, :, :] = X
            inp_e[count, :, :] = Y
            inp_m[count, :, :] = MM
            count = count + 1
            # next state
            next_obs, reward, done, info = env.step(action)
            next_pos = np.stack(np.where(next_obs == env.maze.objects.agent.value), axis = 1)[0]
            next_objects = env.maze.objects
            # rewards
            env.render()
            X, Y, MM = binarize_object_data(curr_objects, count, action)
            inp[count, :, :] = X
            inp_e[count, :, :] = Y
            inp_m[count,:, :] = MM
            count = count + 1

            if done:
                break
        env.close()
    np.savez("transition_matrix.npz", mat = inp)
    np.savez("transition_matrix_eng.npz", mat = inp_e)
    np.savez("transition_matrix_mixed.npz", mat = inp_m)
