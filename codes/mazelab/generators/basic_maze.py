import numpy as np

from skimage.draw import rectangle

def get_all_pos(width, height):
    all_pos = []
    for w in range(width):
        for h in range(height):
            all_pos.append((w,h))
    return all_pos

def basic_maze(width, height, switch_positions, prize_positions, random_obstacles, n_colors = 4):
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

    if random_obstacles:
        frac = 1/n_colors
        obstacle_size = int(frac * N)
        for c in range(3):
            f = np.argwhere(x == 0)
            r = np.random.choice(len(f), size = obstacle_size)
            req_pos = [f[i] for i in r]
            x[tuple(np.array(req_pos).T)] = c + 1

    # f = np.argwhere(x == 0)
    # if len(switch_positions) == 0:
    #     s = np.random.choice(len(f), size = 2)
    #     switch_positions = [f[i] for i in s]
    #
    # for pos in switch_positions:
    #         x[pos[0],pos[1]] = 2
    #
    # f = np.argwhere(x == 0)
    # if len(prize_positions) == 0:
    #     p = np.random.choice(len(f), size = 2)
    #     prize_positions = [f[i] for i in p]
    #
    # for pos in prize_positions:
    #     x[pos[0],pos[1]] = 3
    return x
