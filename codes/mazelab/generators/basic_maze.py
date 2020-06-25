import numpy as np
import random
from skimage.draw import rectangle

def get_all_pos(width, height):
    all_pos = []
    for w in range(width):
        for h in range(height):
            all_pos.append((w,h))
    return all_pos

def basic_maze(width, height, total_switches, total_prizes, random_obstacles, n_colors = 4):
    r""" Maze for source environment containing 2 red colored switches (int = 2) and 2 pink prizes (int = 3).
    All switches need to be turned on before opening a door. """

    x = np.zeros([width, height], dtype=np.uint8)

    # wall
    x[0,:] = 1
    x[-1,:] = 1
    x[:,0] = 1
    x[:,-1] = 1

    all_pos = get_all_pos(width, height)
    N = len(all_pos)


    coin_flip = random.random()
    if random_obstacles:
        frac = 1/n_colors
        obstacle_size = int(frac * N)
        for c in [1,2]:
            if coin_flip >= 0.5 and c == 1:
                continue
            f = np.argwhere(x == 0)
            r = np.random.choice(len(f), size = obstacle_size)
            req_pos = [f[i] for i in r]
            x[tuple(np.array(req_pos).T)] = c + 1

    else:
        # f = np.argwhere(x == 0)
        # r = np.random.choice(len(f), size = total_switches)
        # switch_positions = [f[i] for i in r]
        # x[tuple(np.array(switch_positions).T)] = 2
        #
        #
        # f = np.argwhere(x == 0)
        # r = np.random.choice(len(f), size = total_prizes)
        # prize_positions = [f[i] for i in r]
        # x[tuple(np.array(prize_positions).T)] = 3

        # prize empty_positions
        x[2,2] = 3
        x[4,4] = 3

        # keys position
        x[4,5] = 2
        x[6,7] = 2

    f = np.argwhere(x == 0)
    start = np.random.choice(len(f), size = 1)
    start_idx = [f[i] for i in start]
    return x, start_idx
