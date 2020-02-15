import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import Color as color


class SourceMaze(BaseMaze):
    def __init__(self, **kwargs):
        super(SourceMaze, self).__init__(**kwargs)

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, "white", False, np.stack(np.where(self.x == 0), axis = 1))
        obstacle = Object('obstacle', 1, color.obstacle, "black", True, np.stack(np.where(self.x == 1), axis = 1))
        switch = Object('switch', 2, color.switch, "green", True, np.stack(np.where(self.x == 2), axis = 1))
        prize = Object('prize', 3, color.prize, "red", True, np.stack(np.where(self.x == 3), axis = 1))
        agent = Object('agent', 4, color.agent, "yellow", False, [])
        return free, obstacle, switch, prize, agent
