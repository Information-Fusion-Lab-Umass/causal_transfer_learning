import numpy as np

from skimage.draw import rectangle

def get_all_pos(width, height):
    all_pos = []
    for w in range(width):
        for h in range(height):
            all_pos.append((w,h))
    return all_pos

def basic_maze(width, height, switch_positions, prize_positions, random_obstacles):
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

    obstacle_size = int(0.5 * N)
    o = np.random.choice(np.arange(N), size = obstacle_size)
    req_pos = [all_pos[i] for i in o]

    if random_obstacles:
        x[tuple(np.array(req_pos).T)] = 1

    for pos in switch_positions:
        x[pos[0],pos[1]] = 2

    for pos in prize_positions:
        x[pos[0],pos[1]] = 3
    return x
