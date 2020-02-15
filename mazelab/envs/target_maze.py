import numpy as np
from mazelab import BaseMaze
from mazelab import Object
from mazelab import Color as color


class TargetMaze(BaseMaze):
    def __init__(self, **kwargs):
        super(TargetMaze, self).__init__(**kwargs)

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis = 1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis = 1))

        #inverted positions and colors of switch and prizes
        switch = Object('switch', 3, color.prize, True, np.stack(np.where(self.x == 3), axis = 1))
        prize = Object('prize', 2, color.switch, True, np.stack(np.where(self.x == 2), axis = 1))
        agent = Object('agent', 4, color.agent, False, [])
        return free, obstacle, switch, prize, agent
