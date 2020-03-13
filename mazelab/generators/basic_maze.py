import numpy as np

from skimage.draw import rectangle

def basic_maze(width, height, switch_positions, prize_positions):
    r""" Maze for source environment containing 2 red colored switches (int = 2) and 2 pink prizes (int = 3).
    All switches need to be turned on before opening a door. """

    x = np.zeros([width, height], dtype=np.uint8)

    # wall
    x[0,:] = 1
    x[-1,:] = 1
    x[:,0] = 1
    x[:,-1] = 1

    for pos in switch_positions:
        x[pos[0],pos[1]] = 2

    for pos in prize_positions:
        x[pos[0],pos[1]] = 3
    return x
